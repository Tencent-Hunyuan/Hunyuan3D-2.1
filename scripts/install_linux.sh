#!/usr/bin/env bash
# Install Hunyuan3D-2.1 into a local virtualenv on Linux.
#
#   bash scripts/install_linux.sh            # creates ./env
#   VENV=/path/to/env bash scripts/install_linux.sh
#
# Needs: python3.10+, nvcc (for custom_rasterizer), a C++ toolchain, wget.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$REPO_DIR/env}"
cd "$REPO_DIR"

echo "==> virtualenv: $VENV"
[ -d "$VENV" ] || python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q -U pip wheel

echo "==> torch 2.5.1+cu124"
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Pinning torch in a constraints file stops unpinned dependencies from quietly
# upgrading it, which both breaks the CUDA extension ABI and sends pip backtracking
# through every torch release.
CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS"' EXIT
cat > "$CONSTRAINTS" <<'EOF'
torch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1
triton==3.1.0
EOF

echo "==> python dependencies"
pip install -r requirements.txt -c "$CONSTRAINTS"

# --no-build-isolation: custom_rasterizer's setup.py imports torch, which pip hides from
# the build environment by default ("ModuleNotFoundError: No module named 'torch'").
echo "==> custom_rasterizer (CUDA extension)"
pip install --no-build-isolation -e ./hy3dpaint/custom_rasterizer

echo "==> DifferentiableRenderer (C++ extension)"
(cd hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh)

echo "==> RealESRGAN weights"
mkdir -p hy3dpaint/ckpt
if [ ! -f hy3dpaint/ckpt/RealESRGAN_x4plus.pth ]; then
    wget -q --show-progress -O hy3dpaint/ckpt/RealESRGAN_x4plus.pth \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
fi

echo "==> verifying"
python - <<'EOF'
import sys
sys.path.insert(0, ".")
from torchvision_fix import apply_fix

apply_fix()
import torch, custom_rasterizer, open3d, basicsr, realesrgan  # noqa: F401

sys.path.insert(0, "hy3dpaint")
from DifferentiableRenderer.mesh_inpaint_processor import meshVerticeInpaint  # noqa: F401
from DifferentiableRenderer.mesh_utils import HAS_BPY

print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"bpy available: {HAS_BPY} (GLB export uses trimesh when False)")
print("install OK")
EOF

echo
echo "Done. Activate with:  source $VENV/bin/activate"
echo "Then try:             python demo.py"
