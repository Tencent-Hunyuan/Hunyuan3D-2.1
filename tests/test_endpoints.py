"""End-to-end tests for the HTTP endpoints."""

import io
import os
import threading
from unittest.mock import MagicMock, patch

import pytest

from api import PreemptedError, PreemptionManager, pipeline_manager, app
from tests.conftest import make_minimal_png


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _upload(client, png_data=None, filename="test.png", cancel_previous=False):
    """POST a file to /convert-image-to-3d."""
    if png_data is None:
        png_data = make_minimal_png()
    params = {}
    if cancel_previous:
        params["cancel_previous"] = "true"
    return client.post(
        "/convert-image-to-3d",
        files={"file": (filename, io.BytesIO(png_data), "image/png")},
        params=params,
    )


# ---------------------------------------------------------------------------
# Basic endpoints
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestListOutputs:
    def test_empty(self, client):
        resp = client.get("/outputs")
        assert resp.status_code == 200
        assert resp.json() == {"files": []}


class TestGetOutputFile:
    def test_not_found(self, client):
        resp = client.get("/outputs/nonexistent.glb")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Conversion endpoint
# ---------------------------------------------------------------------------

class TestConversion:
    def test_successful_200(self, client):
        resp = _upload(client)
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/octet-stream"

    def test_invalid_file_type_400(self, client):
        resp = _upload(client, png_data=b"not an image", filename="test.txt")
        assert resp.status_code == 400

    def test_missing_filename_rejects(self, client):
        resp = client.post(
            "/convert-image-to-3d",
            files={"file": ("", io.BytesIO(b"data"), "application/octet-stream")},
        )
        # FastAPI may reject with 422 (validation) or our code returns 400
        assert resp.status_code in (400, 422)


# ---------------------------------------------------------------------------
# Cancel endpoint
# ---------------------------------------------------------------------------

class TestCancelEndpoint:
    def test_cancel_idle(self, client):
        resp = client.post("/cancel")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "idle"


# ---------------------------------------------------------------------------
# Concurrent / preemption tests
# ---------------------------------------------------------------------------

def _make_blocking_shape(mock_mesh, entered, proceed):
    """Return a shape side_effect that blocks until *proceed* is set."""
    def blocking_shape(*args, **kwargs):
        entered.set()
        proceed.wait(timeout=10)
        check_cancel = kwargs.get("check_cancel")
        if check_cancel:
            check_cancel()
        return [mock_mesh]
    return blocking_shape


class TestCancelDuringProcessing:
    def test_returns_409(
        self, tmp_path, monkeypatch, mock_mesh, mock_texture_pipeline,
        mock_rembg, mock_trimesh_load,
    ):
        monkeypatch.chdir(tmp_path)
        import api as api_module

        isolated_pm = PreemptionManager()
        monkeypatch.setattr(api_module, "preemption", isolated_pm)

        entered = threading.Event()
        proceed = threading.Event()

        shape = MagicMock(side_effect=_make_blocking_shape(mock_mesh, entered, proceed))

        with patch.object(
            pipeline_manager, "get_pipelines",
            return_value=(shape, mock_texture_pipeline, mock_rembg),
        ), patch("api.trimesh.load", mock_trimesh_load):
            from starlette.testclient import TestClient
            client = TestClient(app)

            result_holder = [None]

            def do_convert():
                result_holder[0] = _upload(client)

            t = threading.Thread(target=do_convert)
            t.start()

            entered.wait(timeout=5)
            resp = client.post("/cancel")
            assert resp.status_code == 200

            proceed.set()
            t.join(timeout=10)

            assert result_holder[0] is not None
            assert result_holder[0].status_code == 409


class TestPreemptionCancelsFirst:
    def test_first_gets_409_second_gets_200(
        self, tmp_path, monkeypatch, mock_mesh, mock_texture_pipeline,
        mock_rembg, mock_trimesh_load,
    ):
        monkeypatch.chdir(tmp_path)
        import api as api_module

        isolated_pm = PreemptionManager()

        # Track when the second request has called begin() (which sets the
        # cancel event on the first request) so we don't release the first
        # request too early.
        cancel_set = threading.Event()
        original_begin = isolated_pm.begin

        def tracked_begin(*args, **kwargs):
            result = original_begin(*args, **kwargs)
            cancel_set.set()
            return result

        monkeypatch.setattr(isolated_pm, "begin", tracked_begin)
        monkeypatch.setattr(api_module, "preemption", isolated_pm)

        entered = threading.Event()
        proceed = threading.Event()
        blocking = _make_blocking_shape(mock_mesh, entered, proceed)

        call_count = [0]

        def shape_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return blocking(*args, **kwargs)
            # Second call: normal (fast)
            check_cancel = kwargs.get("check_cancel")
            if check_cancel:
                check_cancel()
            return [mock_mesh]

        shape = MagicMock(side_effect=shape_fn)

        with patch.object(
            pipeline_manager, "get_pipelines",
            return_value=(shape, mock_texture_pipeline, mock_rembg),
        ), patch("api.trimesh.load", mock_trimesh_load):
            from starlette.testclient import TestClient
            client = TestClient(app)

            result1 = [None]
            result2 = [None]

            def req1():
                result1[0] = _upload(client)

            def req2():
                result2[0] = _upload(client, cancel_previous=True)

            t1 = threading.Thread(target=req1)
            t1.start()
            entered.wait(timeout=5)

            # Reset cancel_set; the first begin() already fired it.
            cancel_set.clear()

            t2 = threading.Thread(target=req2)
            t2.start()

            # Wait for the second request's begin(cancel_previous=True) to
            # set the cancel event on the first request.  begin() blocks on
            # _processing.acquire(), but the cancel event swap happens before
            # that acquire, so tracked_begin fires cancel_set only AFTER the
            # acquire succeeds.  We must release the first request so the
            # second can acquire.  Release proceed *after* giving t2 a moment
            # to enter begin() and set the old cancel event.
            #
            # The sequence inside begin() is:
            #   1. acquire _lock  (fast)
            #   2. _cancel.set()  ← this is what preempts req1
            #   3. swap _cancel   (fast)
            #   4. release _lock  (fast)
            #   5. _processing.acquire()  ← blocks until req1 finishes
            #
            # So the cancel event is set before _processing.acquire blocks.
            # We just need t2 to have reached step 2 before we release req1.
            import time
            time.sleep(0.2)  # let t2 reach begin() and set the cancel event

            proceed.set()
            t1.join(timeout=10)
            t2.join(timeout=10)

            assert result1[0] is not None
            assert result1[0].status_code == 409
            assert result2[0] is not None
            assert result2[0].status_code == 200


class TestQueueModeBothSucceed:
    def test_both_200(
        self, tmp_path, monkeypatch, mock_mesh, mock_texture_pipeline,
        mock_rembg, mock_trimesh_load,
    ):
        monkeypatch.chdir(tmp_path)
        import api as api_module

        isolated_pm = PreemptionManager()
        monkeypatch.setattr(api_module, "preemption", isolated_pm)

        entered = threading.Event()
        proceed = threading.Event()

        call_count = [0]

        def shape_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                entered.set()
                proceed.wait(timeout=10)
            check_cancel = kwargs.get("check_cancel")
            if check_cancel:
                check_cancel()
            return [mock_mesh]

        shape = MagicMock(side_effect=shape_fn)

        with patch.object(
            pipeline_manager, "get_pipelines",
            return_value=(shape, mock_texture_pipeline, mock_rembg),
        ), patch("api.trimesh.load", mock_trimesh_load):
            from starlette.testclient import TestClient
            client = TestClient(app)

            result1 = [None]
            result2 = [None]

            def req1():
                result1[0] = _upload(client)

            def req2():
                result2[0] = _upload(client, cancel_previous=False)

            t1 = threading.Thread(target=req1)
            t1.start()
            entered.wait(timeout=5)

            t2 = threading.Thread(target=req2)
            t2.start()

            proceed.set()
            t1.join(timeout=10)
            t2.join(timeout=10)

            assert result1[0] is not None
            assert result1[0].status_code == 200
            assert result2[0] is not None
            assert result2[0].status_code == 200
