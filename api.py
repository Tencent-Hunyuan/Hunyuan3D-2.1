"""FastAPI application for Image to 3D GLB Conversion."""

import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

import gc
import glob
import logging
import os
import tempfile
import threading
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import numpy as np
import torch
import trimesh
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("hunyuan3d-api")

# Type aliases for the pipelines (actual types are complex)
ShapePipeline = Any
TexturePipeline = Any
BackgroundRemoverType = Any

class PreemptedError(Exception):
    """Raised when a request is cancelled."""
    pass


class PreemptionManager:
    """Manages cancellation of in-progress generation requests.

    Supports two modes:
    - Explicit cancellation via the ``/cancel`` endpoint.
    - Request preemption via ``cancel_previous=true`` on the conversion endpoint,
      which cancels any in-progress request so the new one can start immediately.

    Between pipeline stages, the processing code calls ``check()`` with its
    cancel event.  If the event has been set, ``check()`` raises
    ``PreemptedError`` so the request can exit quickly.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processing = threading.Lock()
        self._cancel = threading.Event()

    def begin(self, cancel_previous: bool = True) -> threading.Event:
        """Register a new request and acquire the processing slot.

        Args:
            cancel_previous: If True, signal cancellation to any in-progress
                request so it exits at the next checkpoint (preempt mode).
                If False, just wait in line for the slot (queue mode).

        Returns the cancel event for this request (to pass into ``check``).
        """
        with self._lock:
            if cancel_previous:
                self._cancel.set()  # tell current request to stop
            self._cancel = threading.Event()
            cancel = self._cancel
        self._processing.acquire()
        return cancel

    def end(self) -> None:
        """Release the processing slot."""
        self._processing.release()

    def cancel(self) -> bool:
        """Cancel the current in-progress request, if any.

        Returns True if a cancellation signal was sent, False if nothing was running.
        """
        with self._lock:
            self._cancel.set()
            return self._processing.locked()

    @staticmethod
    def check(cancel: threading.Event) -> None:
        """Raise ``PreemptedError`` if this request has been superseded."""
        if cancel.is_set():
            raise PreemptedError("Request preempted by a newer request")


class PipelineManager:
    """Manages ML pipelines with lazy loading and automatic unloading after inactivity."""

    INACTIVITY_TIMEOUT = 3600  # 1 hour in seconds
    CHECK_INTERVAL = 300  # 5 minutes in seconds

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._shape_pipeline: ShapePipeline | None = None
        self._texture_pipeline: TexturePipeline | None = None
        self._rembg: BackgroundRemoverType | None = None
        self._last_usage: float | None = None
        self._checker_thread: threading.Thread | None = None
        self._stop_checker = threading.Event()

    def _load_pipelines(self) -> None:
        """Load the ML pipelines. Must be called with lock held."""
        from hy3dshape.rembg import BackgroundRemover
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

        load_start = time.time()
        logger.info("Loading ML pipelines...")

        # Apply torchvision fix if available
        try:
            from torchvision_fix import apply_fix
            apply_fix()
        except ImportError:
            pass
        except Exception:
            pass

        # Load shape generation pipeline
        t = time.time()
        logger.info("Loading shape generation pipeline...")
        model_path = 'tencent/Hunyuan3D-2.1'
        self._shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
        logger.info("Shape pipeline loaded in %.2fs", time.time() - t)

        t = time.time()
        self._shape_pipeline.enable_flashvdm(replace_vae=False, mc_algo='mc')
        logger.info("FlashVDM enabled in %.2fs", time.time() - t)

        self._rembg = BackgroundRemover()

        # Load texture generation pipeline
        t = time.time()
        logger.info("Loading texture generation pipeline...")
        max_num_view = 4
        resolution = 512
        conf = Hunyuan3DPaintConfig(max_num_view, resolution, texture_steps=8)
        conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
        conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
        self._texture_pipeline = Hunyuan3DPaintPipeline(conf)
        logger.info("Texture pipeline loaded in %.2fs", time.time() - t)

        logger.info("All ML pipelines loaded in %.2fs", time.time() - load_start)

    def get_pipelines(self) -> tuple[ShapePipeline, TexturePipeline, BackgroundRemoverType]:
        """Get the ML pipelines, loading them if needed.

        Returns:
            Tuple of (shape_pipeline, texture_pipeline, background_remover)
        """
        with self._lock:
            if self._shape_pipeline is None:
                self._load_pipelines()
            self._last_usage = time.time()
            return self._shape_pipeline, self._texture_pipeline, self._rembg

    def unload(self, reason: str = "external request") -> None:
        """Unload pipelines and free GPU memory. ``reason`` is logged so
        operators can tell idle-timeout unloads from orchestrator-driven
        peer-sidecar rotations (UNLOAD_SIDECARS=true) and cancel fanouts."""
        with self._lock:
            if self._shape_pipeline is not None:
                logger.info("Unloading ML pipelines (%s)...", reason)
                del self._shape_pipeline
                self._shape_pipeline = None
            if self._texture_pipeline is not None:
                del self._texture_pipeline
                self._texture_pipeline = None
            if self._rembg is not None:
                del self._rembg
                self._rembg = None
            self._last_usage = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("ML pipelines unloaded, GPU memory freed")

    def _checker_loop(self) -> None:
        """Background thread that checks for inactivity and unloads pipelines."""
        while not self._stop_checker.wait(self.CHECK_INTERVAL):
            with self._lock:
                if (
                    self._last_usage is not None
                    and self._shape_pipeline is not None
                    and time.time() - self._last_usage > self.INACTIVITY_TIMEOUT
                ):
                    # Unload without holding lock (release and re-acquire)
                    pass
            # Check again outside the lock to avoid holding it during unload
            should_unload = False
            with self._lock:
                if (
                    self._last_usage is not None
                    and self._shape_pipeline is not None
                    and time.time() - self._last_usage > self.INACTIVITY_TIMEOUT
                ):
                    should_unload = True
            if should_unload:
                self.unload(reason="inactivity")

    def start_checker(self) -> None:
        """Start the background inactivity checker thread."""
        if self._checker_thread is None or not self._checker_thread.is_alive():
            self._stop_checker.clear()
            self._checker_thread = threading.Thread(target=self._checker_loop, daemon=True)
            self._checker_thread.start()

    def stop_checker(self) -> None:
        """Stop the background inactivity checker thread."""
        self._stop_checker.set()
        if self._checker_thread is not None:
            self._checker_thread.join(timeout=1.0)
            self._checker_thread = None


# Global pipeline manager instance
pipeline_manager = PipelineManager()

# Global preemption manager for cancellation support
preemption = PreemptionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """FastAPI lifespan context manager for startup/shutdown."""
    # Startup: start the inactivity checker
    pipeline_manager.start_checker()
    yield
    # Shutdown: stop checker and unload pipelines
    pipeline_manager.stop_checker()
    pipeline_manager.unload(reason="shutdown")


app = FastAPI(
    title="Hunyuan3D-2.1 API",
    description="Convert images to textured 3D GLB models",
    version="1.0.0",
    lifespan=lifespan,
)


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/cancel")
def cancel_generation() -> dict[str, str]:
    """Cancel the current in-progress generation, if any.

    Uses the preemption mechanism to signal cancellation at the next checkpoint.
    Returns 200 whether or not a generation was running.
    """
    cancelled = preemption.cancel()
    if cancelled:
        logger.info("Cancel requested for in-progress generation")
        return {"status": "cancelled"}
    return {"status": "idle", "message": "No generation in progress"}


@app.post("/unload")
def unload_pipelines() -> dict[str, str]:
    """Unload ML pipelines and free GPU memory."""
    with pipeline_manager._lock:
        was_loaded = pipeline_manager._shape_pipeline is not None
    pipeline_manager.unload(reason="external /unload request")
    if was_loaded:
        return {"status": "unloaded"}
    return {"status": "already_unloaded"}


def _get_file_extension(filename: str) -> str:
    """Get lowercase file extension without the dot."""
    return os.path.splitext(filename)[1].lower().lstrip(".")


def smooth_mesh_normals(glb_path: str) -> None:
    """Add smooth vertex normals to a GLB file.

    Computes area-weighted average normals across vertices sharing the same
    position (including UV-seam splits) and writes them as an explicit NORMAL
    attribute in the glTF primitive.  This turns flat/faceted shading into
    smooth shading without changing geometry or texture coordinates.
    """
    import numpy as np
    import pygltflib

    glb = pygltflib.GLTF2().load(glb_path)
    blob = glb._glb_data
    prim = glb.meshes[0].primitives[0]

    # Read vertex positions
    pos_acc = glb.accessors[prim.attributes.POSITION]
    pos_bv = glb.bufferViews[pos_acc.bufferView]
    positions = np.frombuffer(
        blob[pos_bv.byteOffset:pos_bv.byteOffset + pos_bv.byteLength],
        dtype=np.float32,
    ).reshape(-1, 3)

    # Read face indices
    idx_acc = glb.accessors[prim.indices]
    idx_bv = glb.bufferViews[idx_acc.bufferView]
    dtype = np.uint32 if idx_acc.componentType == pygltflib.UNSIGNED_INT else np.uint16
    faces = np.frombuffer(
        blob[idx_bv.byteOffset:idx_bv.byteOffset + idx_bv.byteLength],
        dtype=dtype,
    ).reshape(-1, 3)

    # Group vertices by position (merging UV-seam splits for normal averaging)
    _, inverse = np.unique(np.round(positions, decimals=6), axis=0, return_inverse=True)

    # Face normals (area-weighted by cross product magnitude)
    v0, v1, v2 = positions[faces[:, 0]], positions[faces[:, 1]], positions[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn_len = np.linalg.norm(fn, axis=1, keepdims=True)
    fn_len[fn_len == 0] = 1
    fn /= fn_len

    # Accumulate face normals per unique position
    smooth = np.zeros((inverse.max() + 1, 3), dtype=np.float64)
    np.add.at(smooth, inverse[faces[:, 0]], fn)
    np.add.at(smooth, inverse[faces[:, 1]], fn)
    np.add.at(smooth, inverse[faces[:, 2]], fn)
    norms = np.linalg.norm(smooth, axis=1, keepdims=True)
    norms[norms == 0] = 1
    smooth /= norms

    new_normals = smooth[inverse].astype(np.float32)

    # Append normal data to binary buffer
    normal_bytes = new_normals.tobytes()
    normal_offset = len(blob)

    glb.bufferViews.append(pygltflib.BufferView(
        buffer=0,
        byteOffset=normal_offset,
        byteLength=len(normal_bytes),
        target=pygltflib.ARRAY_BUFFER,
    ))
    glb.accessors.append(pygltflib.Accessor(
        bufferView=len(glb.bufferViews) - 1,
        byteOffset=0,
        componentType=pygltflib.FLOAT,
        count=len(new_normals),
        type="VEC3",
        max=new_normals.max(axis=0).tolist(),
        min=new_normals.min(axis=0).tolist(),
    ))
    prim.attributes.NORMAL = len(glb.accessors) - 1

    glb.buffers[0].byteLength = normal_offset + len(normal_bytes)
    glb._glb_data = blob + normal_bytes
    glb.save(glb_path)


def _process_image_to_glb(
    image_path: str,
    shape_pipeline: ShapePipeline,
    texture_pipeline: TexturePipeline,
    rembg: BackgroundRemoverType,
    cancel: threading.Event | None = None,
    num_inference_steps: int = 25,
    octree_resolution: int = 384,
    mc_algo: str | None = None,
) -> str:
    """Process an image through the 3D generation pipeline.

    Args:
        image_path: Path to the input image file
        shape_pipeline: Shape generation pipeline
        texture_pipeline: Texture generation pipeline
        rembg: Background remover
        cancel: Optional event; if set, this request has been preempted
        num_inference_steps: Shape DiT denoising steps
        octree_resolution: Shape extraction grid resolution (pipeline default 384;
            raise for sharper geometry at the cost of VRAM, which scales ~grid^3)
        mc_algo: Surface extractor override ("mc" / "dmc" / None). None leaves
            whatever enable_flashvdm installed (FlashVDM uses "mc"). "dmc"
            requires the `diso` package to be installed.

    Returns:
        Path to the generated textured GLB file

    Raises:
        PreemptedError: If the request is cancelled between stages
    """
    def _check() -> None:
        if cancel is not None and cancel.is_set():
            raise PreemptedError("Request preempted by a newer request")

    def _cancel_callback(_step: int, _t: object, _outputs: object) -> None:
        _check()

    # Generate timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join("outputs", f"{input_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Output paths
    output_glb = os.path.join(output_dir, f"{input_name}.glb")
    output_textured_obj = os.path.join(output_dir, f"{input_name}_textured.obj")
    output_textured_glb = os.path.join(output_dir, f"{input_name}_textured.glb")

    # Image preprocessing - always apply background removal
    loaded_image = Image.open(image_path)
    if loaded_image.mode != "RGB":
        loaded_image = loaded_image.convert("RGB")
    t = time.time()
    logger.info("Applying background removal...")
    image: Image.Image = rembg(loaded_image)
    logger.info("Background removal: %.2fs", time.time() - t)

    _check()  # checkpoint: after rembg, before shape generation

    # Shape generation. mc_algo is only forwarded when set so the default path
    # leaves FlashVDM's surface extractor untouched (passing mc_algo=anything
    # triggers a deprecation warning in the pipeline).
    shape_kwargs: dict[str, object] = {
        "image": image,
        "num_inference_steps": num_inference_steps,
        "callback": _cancel_callback,
        "callback_steps": 1,
        "check_cancel": _check,
        "octree_resolution": octree_resolution,
    }
    if mc_algo is not None:
        shape_kwargs["mc_algo"] = mc_algo
    t = time.time()
    mesh = shape_pipeline(**shape_kwargs)[0]
    logger.info("Shape generation: %.2fs", time.time() - t)

    t = time.time()
    mesh.export(output_glb)
    logger.info("Shape export: %.2fs", time.time() - t)

    _check()  # checkpoint: after shape generation, before texture generation

    # Texture generation
    t = time.time()
    output_mesh_path = texture_pipeline(
        mesh_path=output_glb,
        image_path=image,
        output_mesh_path=output_textured_obj,
        save_glb=False,
        check_cancel=_check,
    )
    logger.info("Texture generation: %.2fs", time.time() - t)

    _check()  # checkpoint: after texture generation, before export

    # Convert to GLB with trimesh, embedding PBR textures
    t = time.time()
    mesh_textured = trimesh.load(output_mesh_path, force="mesh")

    metallic_path = output_mesh_path.replace(".obj", "_metallic.jpg")
    roughness_path = output_mesh_path.replace(".obj", "_roughness.jpg")
    normal_path = output_mesh_path.replace(".obj", "_normal.jpg")
    if os.path.isfile(metallic_path) and os.path.isfile(roughness_path):
        metallic_img = Image.open(metallic_path).convert("L")
        roughness_img = Image.open(roughness_path).convert("L")
        if metallic_img.size != roughness_img.size:
            roughness_img = roughness_img.resize(metallic_img.size)
        # glTF spec: metallicRoughnessTexture G=roughness, B=metallic
        w, h = metallic_img.size
        mr_array = np.zeros((h, w, 3), dtype=np.uint8)
        mr_array[:, :, 0] = 255  # R: occlusion (unused, white = no effect)
        mr_array[:, :, 1] = np.array(roughness_img)
        mr_array[:, :, 2] = np.array(metallic_img)
        normal_img = Image.open(normal_path) if os.path.isfile(normal_path) else None
        # OBJ loader produces SimpleMaterial; replace with PBRMaterial for GLB export
        base_color = getattr(mesh_textured.visual.material, 'image', None)
        mesh_textured.visual.material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=base_color,
            metallicRoughnessTexture=Image.fromarray(mr_array),
            normalTexture=normal_img,
            metallicFactor=1.0,
            roughnessFactor=1.0,
        )
        logger.info("Embedded PBR textures in GLB (normal=%s)", normal_img is not None)

    mesh_textured.export(output_textured_glb)
    logger.info("GLB conversion: %.2fs", time.time() - t)

    # Clean up intermediate files
    cleanup_patterns = [
        output_glb,
        output_textured_obj,
        output_mesh_path.replace(".obj", ".mtl"),
        output_mesh_path.replace(".obj", ".jpg"),
        metallic_path,
        roughness_path,
        normal_path,
        os.path.join(output_dir, "white_mesh_remesh.obj"),
    ]
    for pattern in cleanup_patterns:
        for file in glob.glob(pattern):
            if os.path.exists(file):
                os.remove(file)

    return output_textured_glb


@app.post("/convert-image-to-3d", response_model=None)
def convert_image_to_3d(
    file: UploadFile = File(...),
    cancel_previous: bool = False,
    smooth_normals: bool = False,
    steps: int = 25,
    texture_steps: int = 8,  # Hunyuan3D v2.1 default is 15 steps for texture generation, default to 8 for faster results
    texture_views: int = 4,  # Hunyuan3D v2.1 default is 6 views for texture generation, default to 4 for faster results
    fast_remesh: bool = False,
    octree_resolution: int = 384,  # Shape extraction grid resolution. Pipeline default is 384; higher = sharper geometry, VRAM scales ~grid^3 (512 ≈ 2.4×, 768 ≈ 8×).
    texture_resolution: int = 512,  # Multiview diffusion render resolution. Pipeline init uses 512; higher = sharper PBR textures.
    mc_algo: str | None = None,  # Surface extractor override ("mc" or "dmc"). None = leave whatever enable_flashvdm installed (FlashVDM uses "mc"). "dmc" needs `diso`.
) -> FileResponse | JSONResponse:
    """Convert an uploaded image to a textured 3D GLB model.

    Args:
        file: The image file to convert (png, jpg, jpeg, webp)
        cancel_previous: If true, cancel any in-progress generation before starting

    Returns:
        The generated GLB file

    Raises:
        HTTPException: 400 if file type is invalid
    """
    # Validate file type
    if file.filename is None:
        raise HTTPException(status_code=400, detail="Filename is required")

    extension = _get_file_extension(file.filename)
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {extension}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    input_filename = file.filename
    logger.info("Received conversion request for file: %s", input_filename)
    start_time = time.time()

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(
        suffix=f".{extension}", delete=False
    ) as temp_file:
        temp_path = temp_file.name
        content = file.file.read()
        temp_file.write(content)

    cancel: threading.Event | None = None
    try:
        # Always serialize through the processing lock to prevent concurrent
        # GPU access (the scheduler has mutable state that is not thread-safe).
        cancel = preemption.begin(cancel_previous=cancel_previous)
        preemption.check(cancel)

        # Get pipelines and process
        shape_pipeline, texture_pipeline, rembg = pipeline_manager.get_pipelines()
        # Apply per-request texture config overrides. `resolution` is read by
        # the texture pipeline at each call (forwarded as `custom_view_size`),
        # so mutating it here changes the multiview diffusion render size
        # without a pipeline reload.
        texture_pipeline.config.texture_steps = texture_steps
        texture_pipeline.config.max_selected_view_num = texture_views
        texture_pipeline.config.fast_remesh = fast_remesh
        texture_pipeline.config.resolution = texture_resolution
        if hasattr(texture_pipeline.models.get("multiview_model", None) or object(), "num_inference_steps"):
            texture_pipeline.models["multiview_model"].num_inference_steps = texture_steps

        output_path = _process_image_to_glb(
            temp_path, shape_pipeline, texture_pipeline, rembg, cancel,
            num_inference_steps=steps,
            octree_resolution=octree_resolution,
            mc_algo=mc_algo,
        )
        if smooth_normals:
            logger.info("Applying smooth normals to %s", output_path)
            smooth_mesh_normals(output_path)
        processing_time = time.time() - start_time
        logger.info(
            "Conversion completed for file: %s in %.2f seconds",
            input_filename,
            processing_time,
        )
    except PreemptedError:
        processing_time = time.time() - start_time
        logger.info(
            "Conversion cancelled for file: %s after %.2f seconds",
            input_filename,
            processing_time,
        )
        return JSONResponse(
            status_code=409,
            content={"error": "Request cancelled"},
        )
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            "Conversion failed for file: %s after %.2f seconds",
            input_filename,
            processing_time,
        )
        logger.error("Error details:\n%s", traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": f"Conversion failed: {str(e)}"},
        )
    finally:
        if cancel is not None:
            preemption.end()
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Return the GLB file
    filename = os.path.basename(output_path)
    return FileResponse(
        path=output_path,
        media_type="application/octet-stream",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/outputs")
def list_outputs() -> dict[str, list[str]]:
    """List all stored GLB files in the outputs directory.

    Returns:
        Dictionary with 'files' key containing list of filenames
    """
    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        return {"files": []}

    files: list[str] = []
    for entry in os.listdir(outputs_dir):
        entry_path = os.path.join(outputs_dir, entry)
        if os.path.isdir(entry_path):
            # Look for GLB files in subdirectories
            for file in os.listdir(entry_path):
                if file.endswith(".glb"):
                    # Return relative path from outputs/
                    files.append(os.path.join(entry, file))
        elif entry.endswith(".glb"):
            files.append(entry)

    return {"files": sorted(files)}


@app.get("/outputs/{filename:path}")
def get_output_file(filename: str) -> FileResponse:
    """Download a stored GLB file.

    Args:
        filename: The filename or path to download (e.g., 'myimage_20241225_123456/myimage_textured.glb')

    Returns:
        The GLB file

    Raises:
        HTTPException: 404 if file not found
    """
    outputs_dir = "outputs"
    file_path = os.path.join(outputs_dir, filename)

    # Security: ensure the path doesn't escape outputs directory
    abs_outputs = os.path.abspath(outputs_dir)
    abs_file = os.path.abspath(file_path)
    if not abs_file.startswith(abs_outputs):
        raise HTTPException(status_code=404, detail="File not found")

    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        media_type="application/octet-stream",
        filename=os.path.basename(filename),
        headers={"Content-Disposition": f'attachment; filename="{os.path.basename(filename)}"'},
    )
