"""Bootstrap hy3dshape submodules without triggering heavy __init__.py imports.

The hy3dshape package's top-level __init__.py pulls in diffusers, torchvision
CUDA builds, etc. which may not be available in a test environment.  This
module wires the package hierarchy manually so only the autoencoder modules
(model, volume_decoders, surface_extractors) are loaded.

Usage::

    from tests.hy3dshape_bootstrap import VectsetVAE, VanillaVolumeDecoder, Latent2MeshOutput
"""

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

_HY3D_ROOT = str(Path(__file__).resolve().parent.parent / "hy3dshape" / "hy3dshape")
_AE_ROOT = os.path.join(_HY3D_ROOT, "models", "autoencoders")


def _ensure_package(name: str, path: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    mod.__path__ = [path]
    mod.__package__ = name
    sys.modules[name] = mod
    return mod


def _load_module(fqn: str, filepath: str) -> types.ModuleType:
    if fqn in sys.modules:
        return sys.modules[fqn]
    spec = importlib.util.spec_from_file_location(fqn, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fqn] = mod
    spec.loader.exec_module(mod)
    return mod


def bootstrap():
    """Wire the minimal package tree and load the modules under test."""
    _ensure_package("hy3dshape", _HY3D_ROOT)
    _ensure_package("hy3dshape.utils", os.path.join(_HY3D_ROOT, "utils"))
    _ensure_package("hy3dshape.models", os.path.join(_HY3D_ROOT, "models"))
    _ensure_package("hy3dshape.models.autoencoders", _AE_ROOT)

    utils_mod = _load_module(
        "hy3dshape.utils.utils",
        os.path.join(_HY3D_ROOT, "utils", "utils.py"),
    )
    utils_pkg = sys.modules["hy3dshape.utils"]
    for attr in ("logger", "synchronize_timer", "smart_load_model", "get_logger"):
        if hasattr(utils_mod, attr):
            setattr(utils_pkg, attr, getattr(utils_mod, attr))

    for stub_name in (
        "hy3dshape.models.autoencoders.attention_blocks",
        "hy3dshape.models.autoencoders.attention_processors",
    ):
        sys.modules.setdefault(stub_name, MagicMock())

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


bootstrap()

# Public re-exports
from hy3dshape.models.autoencoders.model import VectsetVAE  # noqa: E402
from hy3dshape.models.autoencoders.volume_decoders import VanillaVolumeDecoder  # noqa: E402
from hy3dshape.models.autoencoders.surface_extractors import Latent2MeshOutput  # noqa: E402
