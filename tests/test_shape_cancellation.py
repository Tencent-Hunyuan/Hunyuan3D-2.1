"""Tests for cancellation plumbing inside hy3dshape (volume decoding + surface extraction).

These tests verify that check_cancel is properly forwarded through the pipeline
and that cancellation is detected promptly. No GPU or ML models required.
"""

import threading
import time
from unittest.mock import MagicMock

import pytest
import torch

from api import PreemptedError
from tests.hy3dshape_bootstrap import VectsetVAE, VanillaVolumeDecoder


# ---------------------------------------------------------------------------
# VanillaVolumeDecoder tests
# ---------------------------------------------------------------------------

class TestVanillaVolumeDecoderCancellation:
    """Tests for check_cancel in VanillaVolumeDecoder.__call__."""

    def _make_decoder_and_geo(self):
        decoder = VanillaVolumeDecoder()

        def geo_decoder(queries, latents):
            batch_size = queries.shape[0]
            num_points = queries.shape[1]
            return torch.zeros(batch_size, num_points, 1)

        return decoder, geo_decoder

    def test_calls_check_cancel_per_chunk(self):
        decoder, geo_decoder = self._make_decoder_and_geo()
        check = MagicMock()

        # Small grid: octree_resolution=4 -> 5^3=125 points, chunks of 50 -> 3 chunks
        decoder(
            latents=torch.zeros(1, 4, 16),
            geo_decoder=geo_decoder,
            bounds=1.0,
            num_chunks=50,
            octree_resolution=4,
            enable_pbar=False,
            check_cancel=check,
        )

        assert check.call_count == 3, (
            f"Expected check_cancel called once per chunk (3 chunks), got {check.call_count}"
        )

    def test_raises_mid_loop(self):
        decoder, geo_decoder = self._make_decoder_and_geo()

        call_count = [0]

        def check_cancel():
            call_count[0] += 1
            if call_count[0] >= 2:
                raise PreemptedError("cancelled")

        with pytest.raises(PreemptedError):
            decoder(
                latents=torch.zeros(1, 4, 16),
                geo_decoder=geo_decoder,
                bounds=1.0,
                num_chunks=50,
                octree_resolution=4,
                enable_pbar=False,
                check_cancel=check_cancel,
            )

        assert call_count[0] == 2, "Should have raised on the 2nd chunk"

    def test_no_check_cancel_still_works(self):
        """Volume decoder works normally when check_cancel is None."""
        decoder, geo_decoder = self._make_decoder_and_geo()

        result = decoder(
            latents=torch.zeros(1, 4, 16),
            geo_decoder=geo_decoder,
            bounds=1.0,
            num_chunks=50,
            octree_resolution=4,
            enable_pbar=False,
        )

        assert result.shape[0] == 1  # batch size


# ---------------------------------------------------------------------------
# VectsetVAE.latents2mesh -- check_cancel forwarding
# ---------------------------------------------------------------------------

class TestLatents2MeshForwarding:
    """Verify that latents2mesh passes check_cancel to the volume decoder."""

    def test_forwards_check_cancel_to_volume_decoder(self):
        vae = VectsetVAE.__new__(VectsetVAE)

        received_kwargs = {}

        def fake_volume_decoder(latents, geo_decoder, **kwargs):
            received_kwargs.update(kwargs)
            return torch.zeros(1, 5, 5, 5)

        # Return a picklable value (result crosses a subprocess boundary).
        def fake_surface_extractor(grid_logits, **kwargs):
            return ["done"]

        vae.volume_decoder = fake_volume_decoder
        vae.geo_decoder = MagicMock()
        vae.surface_extractor = fake_surface_extractor

        sentinel = lambda: None  # noqa: E731
        vae.latents2mesh(
            torch.zeros(1, 4, 16),
            check_cancel=sentinel,
            bounds=1.0,
            mc_level=0.0,
            octree_resolution=4,
        )

        assert "check_cancel" in received_kwargs, (
            "check_cancel must be forwarded to volume_decoder"
        )
        assert received_kwargs["check_cancel"] is sentinel

    def test_check_cancel_not_in_surface_extractor_kwargs(self):
        """check_cancel should be popped from kwargs before reaching surface_extractor."""
        vae = VectsetVAE.__new__(VectsetVAE)

        def fake_volume_decoder(latents, geo_decoder, **kwargs):
            return torch.zeros(1, 5, 5, 5)

        # The extractor runs in a subprocess, so we cannot inspect kwargs
        # in-process.  Instead we have it echo its kwargs back through the
        # pipe as the return value and inspect them on the parent side.
        def fake_surface_extractor(grid_logits, **kwargs):
            return {"__kwargs_keys__": list(kwargs.keys())}

        vae.volume_decoder = fake_volume_decoder
        vae.geo_decoder = MagicMock()
        vae.surface_extractor = fake_surface_extractor

        result = vae.latents2mesh(
            torch.zeros(1, 4, 16),
            check_cancel=lambda: None,
            bounds=1.0,
            mc_level=0.0,
            octree_resolution=4,
        )

        assert "check_cancel" not in result["__kwargs_keys__"], (
            "check_cancel should be popped from kwargs, not passed to surface_extractor"
        )


# ---------------------------------------------------------------------------
# _cancellable_surface_extract
# ---------------------------------------------------------------------------

class TestCancellableSurfaceExtract:
    """Tests for VectsetVAE._cancellable_surface_extract."""

    def _make_vae(self, surface_fn):
        vae = VectsetVAE.__new__(VectsetVAE)
        vae.surface_extractor = surface_fn
        return vae

    def test_responds_within_1s_when_precancelled(self):
        """If check_cancel raises immediately, should return well under 1s."""

        def slow_extractor(grid_logits, **kwargs):
            time.sleep(10)
            return [MagicMock()]

        vae = self._make_vae(slow_extractor)

        def check_cancel():
            raise PreemptedError("cancelled")

        start = time.monotonic()
        with pytest.raises(PreemptedError):
            vae._cancellable_surface_extract(
                torch.zeros(1, 5, 5, 5),
                check_cancel,
                mc_level=0.0, bounds=1.0, octree_resolution=4,
            )
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"Should respond in <1s when pre-cancelled, took {elapsed:.2f}s"

    def test_detects_cancel_during_long_extraction(self):
        """Cancel set after 0.2s should be detected within ~1.5s total."""
        cancel_event = threading.Event()

        def slow_extractor(grid_logits, **kwargs):
            time.sleep(10)
            return [MagicMock()]

        vae = self._make_vae(slow_extractor)

        def check_cancel():
            if cancel_event.is_set():
                raise PreemptedError("cancelled")

        # Set cancel after 0.2s
        timer = threading.Timer(0.2, cancel_event.set)
        timer.start()

        start = time.monotonic()
        with pytest.raises(PreemptedError):
            vae._cancellable_surface_extract(
                torch.zeros(1, 5, 5, 5),
                check_cancel,
                mc_level=0.0, bounds=1.0, octree_resolution=4,
            )
        elapsed = time.monotonic() - start
        timer.cancel()

        assert elapsed < 2.0, f"Should detect cancel within ~1s, took {elapsed:.2f}s"

    def test_detects_cancel_despite_gil_hold(self):
        """Regression: a GIL-holding extractor must not block cancellation.

        ``time.sleep`` releases the GIL, so it cannot reproduce the real
        bug where ``measure.marching_cubes`` (a C call) holds the GIL for
        tens of seconds.  We simulate this by combining a Python busy-loop
        with ``sys.setswitchinterval(10)`` so the interpreter never
        voluntarily drops the GIL during the 3 s loop.

        With a ``ThreadPoolExecutor`` the main thread cannot acquire the
        GIL to run ``check_cancel`` and the test takes the full 3 s.
        With a subprocess the child has its own GIL, so the main process
        detects the cancel within ~0.5 s.
        """
        import sys

        cancel_event = threading.Event()

        def gil_holding_extractor(grid_logits, **kwargs):
            # Busy-wait for 3 s while holding the GIL.
            end = time.monotonic() + 3
            while time.monotonic() < end:
                pass
            return ["done"]

        vae = self._make_vae(gil_holding_extractor)

        def check_cancel():
            if cancel_event.is_set():
                raise PreemptedError("cancelled")

        old_interval = sys.getswitchinterval()
        sys.setswitchinterval(10)  # prevent voluntary GIL release
        timer = threading.Timer(0.3, cancel_event.set)
        timer.start()

        start = time.monotonic()
        try:
            with pytest.raises(PreemptedError):
                vae._cancellable_surface_extract(
                    torch.zeros(1, 5, 5, 5),
                    check_cancel,
                    mc_level=0.0, bounds=1.0, octree_resolution=4,
                )
            elapsed = time.monotonic() - start
        finally:
            sys.setswitchinterval(old_interval)
            timer.cancel()

        assert elapsed < 1.5, (
            f"Should detect cancel within ~1 s even with GIL-holding "
            f"extractor, took {elapsed:.2f} s (would be ~3 s with threads)"
        )

    def test_returns_result_on_success(self):
        """Without cancellation, returns the surface extractor's output."""
        # Result crosses a subprocess pickle boundary, so use an
        # equality check on a picklable value rather than identity.
        expected = {"vertices": [1, 2, 3], "faces": [4, 5, 6]}

        def fast_extractor(grid_logits, **kwargs):
            return expected

        vae = self._make_vae(fast_extractor)

        result = vae._cancellable_surface_extract(
            torch.zeros(1, 5, 5, 5),
            lambda: None,  # never cancels
            mc_level=0.0, bounds=1.0, octree_resolution=4,
        )

        assert result == expected

    def test_receives_cpu_tensor(self):
        """The surface extractor should receive a CPU tensor."""
        # The extractor runs in a subprocess, so we can't inspect
        # variables set in-process.  Instead, echo the device string
        # back through the pipe as the return value.
        def extractor(grid_logits, **kwargs):
            return {"device": str(grid_logits.device)}

        vae = self._make_vae(extractor)

        result = vae._cancellable_surface_extract(
            torch.zeros(1, 5, 5, 5),
            lambda: None,
            mc_level=0.0, bounds=1.0, octree_resolution=4,
        )

        assert result["device"] == "cpu", (
            f"Expected CPU tensor, got device={result['device']}"
        )
