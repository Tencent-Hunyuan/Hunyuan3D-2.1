"""End-to-end cancellation tests.

These tests POST an image to the HTTP API with the shape pipeline wired to
a real ``VectsetVAE.latents2mesh`` (tiny grid, fake decoders).  This verifies
the full signal chain:

    POST /cancel → PreemptionManager → cancel event → _check() →
    shape_pipeline(check_cancel=_check) → VectsetVAE.latents2mesh →
    volume_decoder / surface_extractor → PreemptedError → 409

The successful-completion test verifies that the same wiring produces a 200
with a GLB file when no cancellation occurs.
"""

import io
import os
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from api import PreemptedError, PreemptionManager, app, pipeline_manager
from tests.conftest import make_minimal_png
from tests.hy3dshape_bootstrap import VectsetVAE, Latent2MeshOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _upload(client, png_data=None, filename="test.png"):
    if png_data is None:
        png_data = make_minimal_png()
    return client.post(
        "/convert-image-to-3d",
        files={"file": (filename, io.BytesIO(png_data), "image/png")},
    )


def _make_vae(volume_decoder_fn, surface_extractor_fn):
    """Create a VectsetVAE with custom decoders (no ML weights needed)."""
    vae = VectsetVAE.__new__(VectsetVAE)
    vae.volume_decoder = volume_decoder_fn
    vae.geo_decoder = None  # unused by our fake decoders
    vae.surface_extractor = surface_extractor_fn
    return vae


def _dummy_surface_extractor(grid_logits, **kwargs):
    """Return a single-triangle mesh.  Picklable (runs in a subprocess)."""
    return [Latent2MeshOutput(
        mesh_v=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        mesh_f=np.array([[0, 1, 2]]),
    )]


def _fast_volume_decoder(latents, geo_decoder, check_cancel=None, **kwargs):
    """Complete immediately, returning a tiny grid."""
    if check_cancel is not None:
        check_cancel()
    return torch.zeros(1, 5, 5, 5)


def _make_shape_pipeline(vae, mock_mesh):
    """Return a callable that behaves like Hunyuan3DDiTFlowMatchingPipeline.

    Internally calls the real ``vae.latents2mesh`` so that ``check_cancel``
    flows through the production code path.
    """
    def shape_pipeline(*, image=None, callback=None, callback_steps=1,
                       check_cancel=None, **kwargs):
        # Simulate a few diffusion steps (fast, no ML)
        if callback:
            for i in range(3):
                if i % callback_steps == 0:
                    callback(i, None, None)

        # --- real cancellation plumbing under test ---
        vae.latents2mesh(
            torch.zeros(1, 4, 16),
            check_cancel=check_cancel,
            bounds=1.0,
            mc_level=0.0,
            num_chunks=50,
            octree_resolution=4,
        )

        return [mock_mesh]

    return shape_pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_mesh():
    mesh = MagicMock()
    def fake_export(path, *args, **kwargs):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"FAKE_GLB")
    mesh.export = fake_export
    return mesh


@pytest.fixture()
def mock_texture_pipeline():
    def texture_fn(*args, **kwargs):
        out = kwargs.get("output_mesh_path", "output.obj")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write("# fake OBJ\n")
        return out
    return MagicMock(side_effect=texture_fn)


@pytest.fixture()
def mock_rembg():
    def rembg_fn(img):
        return img.convert("RGBA")
    return rembg_fn


@pytest.fixture()
def mock_trimesh_load(mock_mesh):
    return MagicMock(return_value=mock_mesh)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestE2ESuccessfulCompletion:
    """POST image → real latents2mesh → 200 with GLB."""

    def test_returns_glb(
        self, tmp_path, monkeypatch, mock_mesh, mock_texture_pipeline,
        mock_rembg, mock_trimesh_load,
    ):
        monkeypatch.chdir(tmp_path)
        import api as api_module

        monkeypatch.setattr(api_module, "preemption", PreemptionManager())

        vae = _make_vae(_fast_volume_decoder, _dummy_surface_extractor)
        shape = _make_shape_pipeline(vae, mock_mesh)

        with patch.object(
            pipeline_manager, "get_pipelines",
            return_value=(shape, mock_texture_pipeline, mock_rembg),
        ), patch("api.trimesh.load", mock_trimesh_load):
            from starlette.testclient import TestClient
            client = TestClient(app)

            resp = _upload(client)

            assert resp.status_code == 200
            assert resp.headers["content-type"] == "application/octet-stream"
            assert len(resp.content) > 0


class TestE2ECancelDuringVolumeDecoding:
    """POST /cancel while latents2mesh is inside the volume decoder → 409."""

    def test_cancel_during_volume_decoding(
        self, tmp_path, monkeypatch, mock_mesh, mock_texture_pipeline,
        mock_rembg, mock_trimesh_load,
    ):
        monkeypatch.chdir(tmp_path)
        import api as api_module

        isolated_pm = PreemptionManager()
        monkeypatch.setattr(api_module, "preemption", isolated_pm)

        entered = threading.Event()
        proceed = threading.Event()

        def blocking_volume_decoder(latents, geo_decoder, check_cancel=None,
                                    **kwargs):
            entered.set()  # signal: we're inside volume decoding
            proceed.wait(timeout=10)
            if check_cancel is not None:
                check_cancel()  # raises PreemptedError if cancelled
            return torch.zeros(1, 5, 5, 5)

        vae = _make_vae(blocking_volume_decoder, _dummy_surface_extractor)
        shape = _make_shape_pipeline(vae, mock_mesh)

        with patch.object(
            pipeline_manager, "get_pipelines",
            return_value=(shape, mock_texture_pipeline, mock_rembg),
        ), patch("api.trimesh.load", mock_trimesh_load):
            from starlette.testclient import TestClient
            client = TestClient(app)

            result = [None]

            def do_convert():
                result[0] = _upload(client)

            t = threading.Thread(target=do_convert)
            t.start()

            # Wait until we're inside the volume decoder
            entered.wait(timeout=5)

            # Cancel via HTTP
            resp = client.post("/cancel")
            assert resp.status_code == 200
            assert resp.json()["status"] == "cancelled"

            # Unblock the volume decoder so it can call check_cancel
            proceed.set()
            start = time.monotonic()
            t.join(timeout=10)
            elapsed = time.monotonic() - start

            assert result[0] is not None
            assert result[0].status_code == 409
            assert elapsed < 3.0, (
                f"409 should arrive promptly after unblocking, took {elapsed:.2f}s"
            )


class TestE2ECancelDuringSurfaceExtraction:
    """POST /cancel while latents2mesh is inside surface extraction → 409."""

    def test_cancel_during_surface_extraction(
        self, tmp_path, monkeypatch, mock_mesh, mock_texture_pipeline,
        mock_rembg, mock_trimesh_load,
    ):
        monkeypatch.chdir(tmp_path)
        import api as api_module

        isolated_pm = PreemptionManager()
        monkeypatch.setattr(api_module, "preemption", isolated_pm)

        entered_extraction = threading.Event()

        def slow_surface_extractor(grid_logits, **kwargs):
            """Block for 10 s (runs in a subprocess via fork)."""
            # Signal the parent process that extraction has started.
            # We can't use threading.Event across processes, so we use
            # a file as a cross-process flag.
            flag_path = os.path.join(
                os.environ.get("TEST_TMP", "/tmp"),
                "_extraction_started",
            )
            with open(flag_path, "w") as f:
                f.write("1")
            time.sleep(10)
            return [Latent2MeshOutput(
                mesh_v=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                                dtype=np.float32),
                mesh_f=np.array([[0, 1, 2]]),
            )]

        vae = _make_vae(_fast_volume_decoder, slow_surface_extractor)
        shape = _make_shape_pipeline(vae, mock_mesh)

        flag = os.path.join(str(tmp_path), "_extraction_started")
        monkeypatch.setenv("TEST_TMP", str(tmp_path))

        with patch.object(
            pipeline_manager, "get_pipelines",
            return_value=(shape, mock_texture_pipeline, mock_rembg),
        ), patch("api.trimesh.load", mock_trimesh_load):
            from starlette.testclient import TestClient
            client = TestClient(app)

            result = [None]

            def do_convert():
                result[0] = _upload(client)

            t = threading.Thread(target=do_convert)
            t.start()

            # Wait for the surface-extraction subprocess to start
            for _ in range(100):
                if os.path.exists(flag):
                    break
                time.sleep(0.05)
            assert os.path.exists(flag), "Surface extraction subprocess never started"

            # Cancel via HTTP
            cancel_time = time.monotonic()
            resp = client.post("/cancel")
            assert resp.status_code == 200

            t.join(timeout=10)
            elapsed = time.monotonic() - cancel_time

            assert result[0] is not None
            assert result[0].status_code == 409
            # The surface extractor sleeps 10s; cancellation via subprocess
            # termination should respond well under that.
            assert elapsed < 3.0, (
                f"409 should arrive within ~1s of cancel, took {elapsed:.2f}s "
                f"(would be ~10s if subprocess termination is broken)"
            )
