"""Unit tests for _process_image_to_glb."""

import os
import threading
from unittest.mock import MagicMock

import pytest

from api import PreemptedError, _process_image_to_glb


class TestSuccessfulCompletion:
    def test_returns_textured_glb_path(
        self, tmp_path, monkeypatch, png_file,
        mock_shape_pipeline, mock_texture_pipeline, mock_rembg,
        mock_mesh, mock_trimesh_load,
    ):
        monkeypatch.chdir(tmp_path)
        import api
        monkeypatch.setattr(api.trimesh, "load", mock_trimesh_load)

        result = _process_image_to_glb(
            png_file, mock_shape_pipeline, mock_texture_pipeline, mock_rembg,
        )
        assert result.endswith("_textured.glb")
        assert os.path.exists(result)
        mock_shape_pipeline.assert_called_once()
        mock_texture_pipeline.assert_called_once()


class TestCancelBeforeShape:
    def test_raises_preempted_before_shape_call(
        self, tmp_path, monkeypatch, png_file,
        mock_shape_pipeline, mock_texture_pipeline, mock_rembg,
    ):
        monkeypatch.chdir(tmp_path)
        cancel = threading.Event()
        cancel.set()  # already cancelled

        with pytest.raises(PreemptedError):
            _process_image_to_glb(
                png_file, mock_shape_pipeline, mock_texture_pipeline,
                mock_rembg, cancel,
            )
        mock_shape_pipeline.assert_not_called()


class TestCancelDuringShapeCallback:
    def test_raises_preempted_at_step(
        self, tmp_path, monkeypatch, png_file,
        mock_texture_pipeline, mock_rembg, mock_mesh,
    ):
        monkeypatch.chdir(tmp_path)
        cancel = threading.Event()

        def shape_fn(*args, **kwargs):
            callback = kwargs.get("callback")
            callback_steps = kwargs.get("callback_steps", 1)
            for step in range(5):
                if step == 2:
                    cancel.set()
                if callback and step % callback_steps == 0:
                    callback(step, None, None)  # should raise at step 2
            return [mock_mesh]

        shape = MagicMock(side_effect=shape_fn)

        with pytest.raises(PreemptedError):
            _process_image_to_glb(
                png_file, shape, mock_texture_pipeline,
                mock_rembg, cancel,
            )
        mock_texture_pipeline.assert_not_called()


class TestCancelAfterShapeBeforeTexture:
    def test_raises_at_checkpoint_after_shape(
        self, tmp_path, monkeypatch, png_file,
        mock_texture_pipeline, mock_rembg, mock_mesh,
    ):
        monkeypatch.chdir(tmp_path)
        cancel = threading.Event()

        def shape_fn(*args, **kwargs):
            # Complete shape generation, then set cancel before returning
            cancel.set()
            return [mock_mesh]

        shape = MagicMock(side_effect=shape_fn)

        with pytest.raises(PreemptedError):
            _process_image_to_glb(
                png_file, shape, mock_texture_pipeline,
                mock_rembg, cancel,
            )
        mock_texture_pipeline.assert_not_called()


class TestCancelAfterTexture:
    def test_raises_at_checkpoint_after_texture(
        self, tmp_path, monkeypatch, png_file,
        mock_shape_pipeline, mock_rembg, mock_mesh,
    ):
        monkeypatch.chdir(tmp_path)
        cancel = threading.Event()

        def texture_fn(*args, **kwargs):
            out = kwargs.get("output_mesh_path", "output.obj")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w") as f:
                f.write("# fake OBJ\n")
            cancel.set()
            return out

        texture = MagicMock(side_effect=texture_fn)

        with pytest.raises(PreemptedError):
            _process_image_to_glb(
                png_file, mock_shape_pipeline, texture,
                mock_rembg, cancel,
            )
