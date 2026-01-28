"""FastAPI application for Image to 3D GLB Conversion."""

import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

import gc
import glob
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import torch
import trimesh
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

# Type aliases for the pipelines (actual types are complex)
ShapePipeline = Any
TexturePipeline = Any
BackgroundRemoverType = Any


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

        # Apply torchvision fix if available
        try:
            from torchvision_fix import apply_fix
            apply_fix()
        except ImportError:
            pass
        except Exception:
            pass

        # Load shape generation pipeline
        model_path = 'tencent/Hunyuan3D-2.1'
        self._shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
        self._rembg = BackgroundRemover()

        # Load texture generation pipeline
        max_num_view = 4
        resolution = 512
        conf = Hunyuan3DPaintConfig(max_num_view, resolution)
        conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
        conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
        self._texture_pipeline = Hunyuan3DPaintPipeline(conf)

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

    def unload(self) -> None:
        """Unload pipelines and free GPU memory."""
        with self._lock:
            if self._shape_pipeline is not None:
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
                self.unload()

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


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """FastAPI lifespan context manager for startup/shutdown."""
    # Startup: start the inactivity checker
    pipeline_manager.start_checker()
    yield
    # Shutdown: stop checker and unload pipelines
    pipeline_manager.stop_checker()
    pipeline_manager.unload()


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


def _get_file_extension(filename: str) -> str:
    """Get lowercase file extension without the dot."""
    return os.path.splitext(filename)[1].lower().lstrip(".")


def _process_image_to_glb(
    image_path: str,
    shape_pipeline: ShapePipeline,
    texture_pipeline: TexturePipeline,
    rembg: BackgroundRemoverType,
) -> str:
    """Process an image through the 3D generation pipeline.

    Args:
        image_path: Path to the input image file
        shape_pipeline: Shape generation pipeline
        texture_pipeline: Texture generation pipeline
        rembg: Background remover

    Returns:
        Path to the generated textured GLB file
    """
    # Generate timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join("outputs", f"{input_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Output paths
    output_glb = os.path.join(output_dir, f"{input_name}.glb")
    output_textured_obj = os.path.join(output_dir, f"{input_name}_textured.obj")
    output_textured_glb = os.path.join(output_dir, f"{input_name}_textured.glb")

    # Image preprocessing
    loaded_image = Image.open(image_path)
    if loaded_image.mode == "RGB":
        image: Image.Image = rembg(loaded_image)
    else:
        image = loaded_image.convert("RGBA")

    # Shape generation
    mesh = shape_pipeline(image=image)[0]
    mesh.export(output_glb)

    # Texture generation
    output_mesh_path = texture_pipeline(
        mesh_path=output_glb,
        image_path=image,
        output_mesh_path=output_textured_obj,
        save_glb=False,
    )

    # Convert to GLB with trimesh
    mesh_textured = trimesh.load(output_mesh_path, force="mesh")
    mesh_textured.export(output_textured_glb)

    # Clean up intermediate files
    cleanup_patterns = [
        output_glb,
        output_textured_obj,
        output_mesh_path.replace(".obj", ".mtl"),
        output_mesh_path.replace(".obj", ".jpg"),
        output_mesh_path.replace(".obj", "_metallic.jpg"),
        output_mesh_path.replace(".obj", "_roughness.jpg"),
        os.path.join(output_dir, "white_mesh_remesh.obj"),
    ]
    for pattern in cleanup_patterns:
        for file in glob.glob(pattern):
            if os.path.exists(file):
                os.remove(file)

    return output_textured_glb


@app.post("/convert-image-to-3d")
def convert_image_to_3d(file: UploadFile = File(...)) -> FileResponse:
    """Convert an uploaded image to a textured 3D GLB model.

    Args:
        file: The image file to convert (png, jpg, jpeg, webp)

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

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(
        suffix=f".{extension}", delete=False
    ) as temp_file:
        temp_path = temp_file.name
        content = file.file.read()
        temp_file.write(content)

    try:
        # Get pipelines and process
        shape_pipeline, texture_pipeline, rembg = pipeline_manager.get_pipelines()
        output_path = _process_image_to_glb(
            temp_path, shape_pipeline, texture_pipeline, rembg
        )
    finally:
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
