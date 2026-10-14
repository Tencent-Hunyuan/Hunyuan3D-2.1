# Installing on Linux / Python 3.10

```bash
bash scripts/install_linux.sh
source env/bin/activate
python demo.py
```

The script is the whole procedure. The rest of this file explains what it works around, since
`requirements.txt` as published does not install on this platform.

## Verified configuration

Python 3.10, torch 2.5.1+cu124, diffusers 0.30.0, transformers 4.46.0, CUDA toolkit 12.6,
Quadro RTX 6000 (24 GB).

Measured peak VRAM, generating and texturing as separate stages:

| stage | peak VRAM | time |
|---|---|---|
| shape | 9.8 GiB | 94 s |
| texture (PBR, `max_num_view=6`, `resolution=512`) | 21.8 GiB | 77 s |

Holding both pipelines at once needs roughly the sum, which is where the README's "29 GB"
figure comes from. Release the shape pipeline before texturing and 24 GB is enough.

## What was fixed

**`bpy==4.0` has no wheel for this interpreter.** bpy publishes wheels for a single CPython
version per Blender release, and none exists for cp310 on Linux at any version, so
`pip install -r requirements.txt` fails outright. It is imported at module scope in
`DifferentiableRenderer/mesh_utils.py` but only used by `convert_obj_to_glb`, so the import is
now guarded and that function falls back to trimesh. The fallback packs the separate roughness
and metalness maps into one texture as glTF requires - roughness in green, metalness in blue -
so the exported GLB keeps full PBR materials. bpy is still used when present.

**`custom_rasterizer` fails to build.** Its `setup.py` imports torch, which pip's build
isolation hides, giving `ModuleNotFoundError: No module named 'torch'`. It needs
`--no-build-isolation`.

**`pytorch-lightning==1.9.5` needs `pkg_resources`**, which setuptools removed in version 81.
Without a pin, a fresh environment fails at import with `No module named 'pkg_resources'`.
`setuptools<81` is now pinned in `requirements.txt`.

**`deepspeed` was unpinned and is training-only.** pip backtracks through torch releases trying
to satisfy it, which can run for hours and ends up replacing the pinned torch. It moved to
`requirements-training.txt`.

**`cupy-cuda12x` is unused.** Nothing in the codebase imports cupy; it pulled roughly 2 GB of
CUDA 13 wheels. Removed. The full image to textured mesh pipeline runs without it.

**The mirror index URLs are now opt-in.** The two hardcoded `--extra-index-url` lines are much
slower than PyPI outside mainland China. Set `PIP_EXTRA_INDEX_URL` to use them.

**`open3d` is required**, despite no direct import - `trimesh.as_open3d` pulls it in during
remeshing. It is kept in `requirements.txt`.

## Note on `torchvision_fix`

`basicsr` imports `torchvision.transforms.functional_tensor`, removed in torchvision 0.17+.
The repo's own `torchvision_fix.py` patches this, but it only resolves if the repository root is
on `sys.path`. Running `python demo.py` from the repo root works; importing the pipeline from
elsewhere needs `sys.path.insert(0, "<repo root>")` before `apply_fix()`.
