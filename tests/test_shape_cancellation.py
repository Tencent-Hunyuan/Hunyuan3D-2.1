"""Tests for cancellation plumbing inside hy3dshape (volume decoding + surface extraction).

These tests verify that check_cancel is properly forwarded through the pipeline
and that cancellation is detected promptly. No GPU or ML models required.

The hy3dshape package has heavy transitive dependencies (diffusers, torchvision
CUDA builds, etc.) that may not be available in a test environment.  We bypass
the top-level ``__init__.py`` by wiring the package hierarchy ourselves so that
only the modules under test are actually loaded.
"""

import importlib
import importlib.util
import os
import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from api import PreemptedError

# ---------------------------------------------------------------------------
# Bootstrap: load only the hy3dshape modules we need, without triggering the
# heavy __init__.py that pulls in diffusers/torchvision/etc.
# ---------------------------------------------------------------------------

_HY3D_ROOT = str(Path(__file__).resolve().parent.parent / "hy3dshape" / "hy3dshape")
_AE_ROOT = os.path.join(_HY3D_ROOT, "models", "autoencoders")


def _ensure_package(name: str, path: str) -> types.ModuleType:
    """Register a bare package in sys.modules (no __init__.py executed)."""
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    mod.__path__ = [path]
    mod.__package__ = name
    sys.modules[name] = mod
    return mod


def _load_module(fqn: str, filepath: str) -> types.ModuleType:
    """Load a single .py file as *fqn* without running parent __init__ files."""
    if fqn in sys.modules:
        return sys.modules[fqn]
    spec = importlib.util.spec_from_file_location(fqn, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fqn] = mod
    spec.loader.exec_module(mod)
    return mod


def _bootstrap():
    """Wire the minimal package tree and load the modules under test."""
    # 1. bare packages (no code executed)
    _ensure_package("hy3dshape", _HY3D_ROOT)
    _ensure_package("hy3dshape.utils", os.path.join(_HY3D_ROOT, "utils"))
    _ensure_package("hy3dshape.models", os.path.join(_HY3D_ROOT, "models"))
    _ensure_package("hy3dshape.models.autoencoders", _AE_ROOT)

    # 2. utils – only load utils.py (logger, synchronize_timer); skip misc.py (needs omegaconf)
    utils_mod = _load_module(
        "hy3dshape.utils.utils",
        os.path.join(_HY3D_ROOT, "utils", "utils.py"),
    )
    # Re-export into the utils package so `from ...utils import logger` works
    utils_pkg = sys.modules["hy3dshape.utils"]
    for attr in ("logger", "synchronize_timer", "smart_load_model", "get_logger"):
        if hasattr(utils_mod, attr):
            setattr(utils_pkg, attr, getattr(utils_mod, attr))

    # 3. Stub heavy sibling modules that volume_decoders.py / model.py import
    #    but we never exercise in these tests.
    for stub_name in (
        "hy3dshape.models.autoencoders.attention_blocks",
        "hy3dshape.models.autoencoders.attention_processors",
    ):
        sys.modules.setdefault(stub_name, MagicMock())

    # 4. Load the modules we actually test
    _load_module(
        "hy3dshape.models.autoencoders.surface_extractors",
        os.path.join(_AE_ROOT, "surface_extractors.py"),
    )
    _load_module(
        "hy3dshape.models.autoencoders.volume_decoders",
        os.path.join(_AE_ROOT, "volume_decoders.py"),
    )
    _load_module(
        "hy3dshape.models.autoencoders.model",
        os.path.join(_AE_ROOT, "model.py"),
    )


_bootstrap()

from hy3dshape.models.autoencoders.volume_decoders import VanillaVolumeDecoder  # noqa: E402
from hy3dshape.models.autoencoders.model import VectsetVAE  # noqa: E402


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
