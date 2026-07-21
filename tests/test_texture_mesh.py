"""Tests for the /texture-mesh endpoint and paint-only pipeline loading."""

import io
import threading
from unittest.mock import MagicMock, patch

import pytest

from api import PipelineManager, PreemptionManager, app, pipeline_manager
from tests.conftest import make_minimal_png

FAKE_GLB = b"FAKE_INPUT_GLB"


def _upload_texture_mesh(client, mesh_data=FAKE_GLB, mesh_filename="mesh.glb",
                         image_data=None, image_filename="ref.png", params=None):
    """POST a mesh + image to /texture-mesh."""
    if image_data is None:
        image_data = make_minimal_png()
    return client.post(
        "/texture-mesh",
        files={
            "mesh": (mesh_filename, io.BytesIO(mesh_data), "model/gltf-binary"),
            "image": (image_filename, io.BytesIO(image_data), "image/png"),
        },
        params=params or {},
    )


# ---------------------------------------------------------------------------
# TestClient fixture (texture-only pipelines mocked)
# ---------------------------------------------------------------------------

@pytest.fixture()
def texture_client(tmp_path, monkeypatch, mock_texture_pipeline, mock_rembg, mock_trimesh_load):
    """Starlette TestClient with get_texture_pipelines mocked."""
    from starlette.testclient import TestClient

    monkeypatch.chdir(tmp_path)

    import api as api_module

    isolated_preemption = PreemptionManager()
    monkeypatch.setattr(api_module, "preemption", isolated_preemption)

    with patch.object(
        pipeline_manager,
        "get_texture_pipelines",
        return_value=(mock_texture_pipeline, mock_rembg),
    ), patch("api.trimesh.load", mock_trimesh_load):
        yield TestClient(app)


# ---------------------------------------------------------------------------
# Endpoint validation + success
# ---------------------------------------------------------------------------

class TestTextureMeshEndpoint:
    def test_successful_200(self, texture_client, mock_texture_pipeline):
        resp = _upload_texture_mesh(texture_client)
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/octet-stream"
        # Paint pipeline was invoked on the uploaded mesh with the endpoint default.
        kwargs = mock_texture_pipeline.call_args.kwargs
        assert kwargs["use_remesh"] is False
        assert kwargs["save_glb"] is False
        assert kwargs["mesh_path"].endswith("mesh.glb")

    def test_invalid_mesh_extension_400(self, texture_client):
        resp = _upload_texture_mesh(texture_client, mesh_filename="mesh.stl")
        assert resp.status_code == 400

    def test_invalid_image_extension_400(self, texture_client):
        resp = _upload_texture_mesh(texture_client, image_filename="ref.txt")
        assert resp.status_code == 400

    def test_missing_mesh_422(self, texture_client):
        resp = texture_client.post(
            "/texture-mesh",
            files={"image": ("ref.png", io.BytesIO(make_minimal_png()), "image/png")},
        )
        assert resp.status_code == 422

    def test_missing_image_422(self, texture_client):
        resp = texture_client.post(
            "/texture-mesh",
            files={"mesh": ("mesh.glb", io.BytesIO(FAKE_GLB), "model/gltf-binary")},
        )
        assert resp.status_code == 422

    def test_config_overrides_applied(self, texture_client, mock_texture_pipeline):
        resp = _upload_texture_mesh(texture_client, params={
            "texture_steps": 5,
            "texture_views": 6,
            "texture_resolution": 768,
            "use_remesh": "true",
            "target_face_count": 123456,
            "fast_remesh": "true",
        })
        assert resp.status_code == 200
        config = mock_texture_pipeline.config
        assert config.texture_steps == 5
        assert config.max_selected_view_num == 6
        assert config.resolution == 768
        assert config.target_face_count == 123456
        assert config.fast_remesh is True
        assert mock_texture_pipeline.call_args.kwargs["use_remesh"] is True


# ---------------------------------------------------------------------------
# Paint-only pipeline loading
# ---------------------------------------------------------------------------

class TestPaintOnlyLoad:
    def _fake_texture_load(self, manager):
        def fake(check_cancel=None):
            manager._texture_pipeline = MagicMock()
            manager._rembg = MagicMock()
        return fake

    def test_texture_only_leaves_shape_unloaded(self):
        manager = PipelineManager()
        with patch.object(manager, "_load_texture_locked", side_effect=self._fake_texture_load(manager)) as load:
            texture, rembg = manager.get_texture_pipelines()
        load.assert_called_once()
        assert texture is manager._texture_pipeline
        assert rembg is manager._rembg
        assert manager._shape_pipeline is None

    def test_state_reports_loaded_after_paint_only_load(self):
        manager = PipelineManager()
        with patch.object(manager, "_load_texture_locked", side_effect=self._fake_texture_load(manager)):
            manager.get_texture_pipelines()
        loaded, last_usage = manager.status()
        assert loaded is True
        assert last_usage is not None

    def test_unload_after_paint_only_load_reports_unloaded(self):
        manager = PipelineManager()
        with patch.object(manager, "_load_texture_locked", side_effect=self._fake_texture_load(manager)):
            manager.get_texture_pipelines()
        assert manager.unload(reason="test") is True
        assert manager._texture_pipeline is None
        assert manager._rembg is None
        loaded, _ = manager.status()
        assert loaded is False

    def test_texture_load_not_repeated_when_loaded(self):
        manager = PipelineManager()
        with patch.object(manager, "_load_texture_locked", side_effect=self._fake_texture_load(manager)) as load:
            manager.get_texture_pipelines()
            manager.get_texture_pipelines()
        load.assert_called_once()


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

class TestTextureMeshCancellation:
    def test_cancel_during_processing_returns_409(
        self, tmp_path, monkeypatch, mock_rembg, mock_trimesh_load,
    ):
        monkeypatch.chdir(tmp_path)
        import api as api_module

        isolated_pm = PreemptionManager()
        monkeypatch.setattr(api_module, "preemption", isolated_pm)

        entered = threading.Event()
        proceed = threading.Event()

        def blocking_texture(*args, **kwargs):
            entered.set()
            proceed.wait(timeout=10)
            check_cancel = kwargs.get("check_cancel")
            if check_cancel:
                check_cancel()
            return kwargs.get("output_mesh_path", "output.obj")

        texture = MagicMock(side_effect=blocking_texture)

        with patch.object(
            pipeline_manager, "get_texture_pipelines",
            return_value=(texture, mock_rembg),
        ), patch("api.trimesh.load", mock_trimesh_load):
            from starlette.testclient import TestClient
            client = TestClient(app)

            result_holder = [None]

            def do_texture():
                result_holder[0] = _upload_texture_mesh(client)

            t = threading.Thread(target=do_texture)
            t.start()

            entered.wait(timeout=5)
            resp = client.post("/cancel")
            assert resp.status_code == 200

            proceed.set()
            t.join(timeout=10)

            assert result_holder[0] is not None
            assert result_holder[0].status_code == 409
