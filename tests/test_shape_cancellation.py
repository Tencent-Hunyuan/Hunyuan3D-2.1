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
