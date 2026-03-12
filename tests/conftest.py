"""Shared fixtures for the Hunyuan3D API test suite."""

import io
import os
import struct
import sys
import threading
import zlib
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hy3dshape"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hy3dpaint"))

from api import PreemptionManager, app, pipeline_manager, preemption  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_minimal_png(width: int = 2, height: int = 2) -> bytes:
    """Return a valid RGBA PNG file as bytes."""

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    header = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)  # 8-bit RGBA
    ihdr = _chunk(b"IHDR", ihdr_data)

    # Build raw scanlines (filter byte 0 + RGBA pixels)
    raw = b""
    for _ in range(height):
        raw += b"\x00" + b"\xff\x00\x00\xff" * width  # red pixels

    idat = _chunk(b"IDAT", zlib.compress(raw))
    iend = _chunk(b"IEND", b"")
    return header + ihdr + idat + iend


# ---------------------------------------------------------------------------
# Basic fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fresh_preemption_manager():
    """A fresh PreemptionManager with no global side-effects."""
    return PreemptionManager()


@pytest.fixture()
def png_bytes():
    return make_minimal_png()


@pytest.fixture()
def png_file(tmp_path, png_bytes):
    """Write a small PNG to disk and return the path."""
    p = tmp_path / "test_image.png"
    p.write_bytes(png_bytes)
    return str(p)


# ---------------------------------------------------------------------------
# Mock pipeline fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_mesh():
    """A mock trimesh mesh whose .export writes a real file."""
    mesh = MagicMock()

    def fake_export(path, *args, **kwargs):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"FAKE_GLB")

    mesh.export = fake_export
    return mesh


@pytest.fixture()
def mock_rembg():
    """Mock background remover that returns a PIL RGBA image."""
    from PIL import Image

    def rembg_fn(img):
        return img.convert("RGBA")

    return rembg_fn


@pytest.fixture()
def mock_shape_pipeline(mock_mesh):
    """Mock shape pipeline that respects callback and check_cancel kwargs."""
    def shape_fn(*args, **kwargs):
        callback = kwargs.get("callback")
        callback_steps = kwargs.get("callback_steps", 1)
        check_cancel = kwargs.get("check_cancel")

        # Simulate 5 diffusion steps
        for step in range(5):
            if callback is not None and step % callback_steps == 0:
                callback(step, None, None)
        if check_cancel is not None:
            check_cancel()
        return [mock_mesh]

    return MagicMock(side_effect=shape_fn)


@pytest.fixture()
def mock_texture_pipeline():
    """Mock texture pipeline that writes a dummy OBJ and returns the path."""
    def texture_fn(*args, **kwargs):
        check_cancel = kwargs.get("check_cancel")
        if check_cancel is not None:
            check_cancel()
        out = kwargs.get("output_mesh_path", "output.obj")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write("# fake OBJ\n")
        return out

    return MagicMock(side_effect=texture_fn)


@pytest.fixture()
def mock_trimesh_load(mock_mesh):
    return MagicMock(return_value=mock_mesh)


# ---------------------------------------------------------------------------
# TestClient fixture (fully mocked pipelines, isolated globals)
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(
    tmp_path,
    monkeypatch,
    mock_shape_pipeline,
    mock_texture_pipeline,
    mock_rembg,
    mock_trimesh_load,
):
    """Starlette TestClient with mocked pipelines and isolated state."""
    from starlette.testclient import TestClient

    monkeypatch.chdir(tmp_path)

    # Isolate PreemptionManager per test
    import api as api_module

    isolated_preemption = PreemptionManager()
    monkeypatch.setattr(api_module, "preemption", isolated_preemption)

    with patch.object(
        pipeline_manager,
        "get_pipelines",
        return_value=(mock_shape_pipeline, mock_texture_pipeline, mock_rembg),
    ), patch("api.trimesh.load", mock_trimesh_load):
        yield TestClient(app)
