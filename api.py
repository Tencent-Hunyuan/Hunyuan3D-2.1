"""FastAPI application for Image to 3D GLB Conversion."""

from fastapi import FastAPI

app = FastAPI(
    title="Hunyuan3D-2.1 API",
    description="Convert images to textured 3D GLB models",
    version="1.0.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
