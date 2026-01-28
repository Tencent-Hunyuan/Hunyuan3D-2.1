"""FastAPI application for Image to 3D GLB Conversion."""

import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

import gc
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI

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


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
