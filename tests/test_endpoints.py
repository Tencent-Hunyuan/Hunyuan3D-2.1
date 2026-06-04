"""End-to-end tests for the HTTP endpoints."""

import io
import os
import threading
import time
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


class TestState:
    """GET /state — load/VRAM report for the orchestrating memory manager."""

    def test_unloaded(self, client):
        resp = client.get("/state")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "idle"
        assert body["loaded"] is False
        assert body["last_activity_ts"] is None
        assert "vram_free_mb" in body
        assert "vram_total_mb" in body

    def test_loaded(self, client):
        # The client fixture stubs get_pipelines, so set the loaded state
        # directly on the manager.
        pipeline_manager._shape_pipeline = MagicMock()
        pipeline_manager._last_usage = time.time()
        try:
            body = client.get("/state").json()
            assert body["status"] == "ok"
            assert body["loaded"] is True
            assert body["last_activity_ts"] is not None
        finally:
            pipeline_manager._shape_pipeline = None
            pipeline_manager._last_usage = None


class TestKill:
    """POST /kill — self-exit contract for the orchestrating memory manager."""

    @pytest.fixture
    def mock_exit(self):
        import api as api_module

        with patch.object(api_module.os, "_exit") as mocked, \
                patch.object(api_module.time, "sleep"):
            yield mocked
        # Free the kill-once lock for other tests.
        if api_module._kill_once.locked():
            api_module._kill_once.release()

    def test_kill_responds_then_exits(self, client, mock_exit):
        resp = client.post("/kill")
        assert resp.status_code == 200
        assert resp.json()["status"] == "killing"
        # TestClient runs the BackgroundTask before returning.
        mock_exit.assert_called_once_with(137)

    def test_second_kill_reports_already_killing(self, client, mock_exit):
        client.post("/kill")
        resp = client.post("/kill")
        assert resp.json()["status"] == "already-killing"


class TestUnloadWaitsForSlot:
    """POST /unload must not delete pipelines under an in-flight request."""

    def test_unload_busy_503(self, client, monkeypatch):
        import api as api_module

        monkeypatch.setattr(api_module, "UNLOAD_WAIT_S", 0.05)
        # Occupy the processing slot like an in-flight request would.
        assert api_module.preemption.wait_idle(timeout=1.0)
        try:
            resp = client.post("/unload")
            assert resp.status_code == 503
            assert resp.json()["status"] == "busy"
        finally:
            api_module.preemption.end()

    def test_unload_idle_releases_slot(self, client):
        import api as api_module

        resp = client.post("/unload")
        assert resp.status_code == 200
        assert resp.json()["status"] in ("unloaded", "already_unloaded")
        # Slot must be free again afterwards.
        assert api_module.preemption.wait_idle(timeout=1.0)
        api_module.preemption.end()


class TestPageCacheDrop:
    """free_page_cache_if_needed — unified-memory survival helper."""

    @pytest.fixture
    def drop_calls(self, monkeypatch):
        """Stub _try_drop_caches and reset the cooldown; returns call log."""
        import api as api_module

        monkeypatch.setattr(api_module, "_page_cache_last_drop", 0.0)
        calls = []
        monkeypatch.setattr(api_module, "_try_drop_caches", lambda: calls.append(1) or True)
        return calls

    def test_noop_on_discrete_gpu(self, monkeypatch, drop_calls):
        import api as api_module

        monkeypatch.setattr(api_module, "_is_unified_memory", lambda: False)
        api_module.free_page_cache_if_needed()
        assert drop_calls == []

    def test_drops_when_memfree_low(self, monkeypatch, drop_calls):
        import api as api_module

        monkeypatch.setattr(api_module, "_is_unified_memory", lambda: True)
        monkeypatch.setattr(api_module, "_read_meminfo_kb", lambda: {"MemFree": 1024})
        api_module.free_page_cache_if_needed()
        assert drop_calls == [1]

    def test_skips_when_memfree_high(self, monkeypatch, drop_calls):
        import api as api_module

        monkeypatch.setattr(api_module, "_is_unified_memory", lambda: True)
        monkeypatch.setattr(
            api_module, "_read_meminfo_kb",
            lambda: {"MemFree": (api_module._PAGE_CACHE_RESERVE_GB + 1) * 1024 * 1024},
        )
        api_module.free_page_cache_if_needed()
        assert drop_calls == []
