#!/bin/bash
# Setup script for Hunyuan3D-2.1 on NVIDIA DGX Spark
# Target: GB10 GPU (Blackwell sm_121), CUDA 13.0, aarch64, Python 3.12

set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=============================================="
echo "Hunyuan3D-2.1 Setup for DGX Spark"
echo "=============================================="
echo ""

# Check system
echo "System Info:"
echo "  Architecture: $(uname -m)"
echo "  Python: $(python3 --version)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "  GPU: Unable to query"
echo ""

# Check for required system dependencies
echo "Checking system dependencies..."

# Check for python3-dev (needed for building extensions)
if ! dpkg -s python3-dev >/dev/null 2>&1; then
    echo ""
    echo "ERROR: python3-dev is not installed."
    echo "Please install it with:"
    echo "  sudo apt install python3-dev"
    echo ""
    exit 1
fi
echo "  python3-dev: OK"

# Check for C++ compiler
if ! command -v g++ >/dev/null 2>&1; then
    echo ""
    echo "ERROR: g++ compiler is not installed."
    echo "Please install it with:"
    echo "  sudo apt install build-essential"
    echo ""
    exit 1
fi
echo "  g++ compiler: OK"

# Check for wget (needed for downloading models)
if ! command -v wget >/dev/null 2>&1; then
    echo ""
    echo "WARNING: wget is not installed. Model downloads may fail."
    echo "Install with: sudo apt install wget"
fi

echo ""

# Remove old venv if exists
if [ -d "venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf venv
fi

# Create new venv
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA 13.0 support
echo ""
echo "Installing PyTorch with CUDA 13.0 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Verify PyTorch CUDA
echo ""
echo "Verifying PyTorch installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA built version: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    x = torch.randn(3,3).cuda()
    print('GPU tensor test: PASSED')
else:
    print('WARNING: CUDA not available!')
    exit(1)
"

# Install build tools
echo ""
echo "Installing build tools..."
pip install ninja pybind11

# Install core ML packages
echo ""
echo "Installing core ML packages..."
pip install transformers diffusers accelerate pytorch-lightning huggingface-hub safetensors

# Install scientific computing
echo ""
echo "Installing scientific computing packages..."
pip install numpy scipy einops pandas

# Install computer vision packages
echo ""
echo "Installing computer vision packages..."
pip install opencv-python imageio scikit-image

# Install rembg and realesrgan (may need workarounds)
echo ""
echo "Installing image processing packages..."
pip install rembg || echo "WARNING: rembg installation failed, will try alternatives"
pip install basicsr || echo "WARNING: basicsr installation failed"
pip install realesrgan || echo "WARNING: realesrgan installation failed"

# Install 3D mesh processing
echo ""
echo "Installing 3D mesh processing packages..."
pip install trimesh pygltflib xatlas pillow fast-simplification
pip install pymeshlab || echo "WARNING: pymeshlab installation failed"
# Note: open3d is not available for aarch64, but it's not used in the codebase

# Install configuration management
echo ""
echo "Installing configuration packages..."
pip install omegaconf pyyaml configargparse

# Install web framework
echo ""
echo "Installing web framework packages..."
pip install gradio fastapi uvicorn

# Install utilities
echo ""
echo "Installing utility packages..."
pip install tqdm psutil pydantic

# Install GPU computing with CUDA 13
echo ""
echo "Installing cupy for CUDA 13..."
pip install cupy-cuda13x || echo "WARNING: cupy installation failed"

# Install additional ML packages
echo ""
echo "Installing additional ML packages..."
pip install timm torchmetrics torchdiffeq
pip install pythreejs || echo "WARNING: pythreejs installation failed"

# Install ONNX Runtime (may need special aarch64 handling)
echo ""
echo "Installing ONNX Runtime..."
pip install onnxruntime || pip install onnxruntime-gpu || echo "WARNING: onnxruntime installation failed"

# Skip bpy (Blender) - not available on PyPI, not needed for core functionality

# Skip deepspeed for now - may have CUDA 13 issues
# pip install deepspeed

echo ""
echo "=============================================="
echo "Installing custom extensions..."
echo "=============================================="

# Install custom rasterizer
# Note: Uses --no-build-isolation because setup.py imports torch at top level
# Note: TORCH_CUDA_ARCH_LIST set for Blackwell sm_121
echo ""
echo "Building custom rasterizer..."
cd hy3dpaint/custom_rasterizer
TORCH_CUDA_ARCH_LIST="12.1" pip install --no-build-isolation -e . || echo "WARNING: custom_rasterizer build failed"
cd ../..

# Compile DifferentiableRenderer
echo ""
echo "Building DifferentiableRenderer..."
cd hy3dpaint/DifferentiableRenderer
if [ -f "mesh_inpaint_processor.cpp" ]; then
    c++ -O3 -Wall -shared -std=c++11 -fPIC \
        $(python3-config --includes) \
        -I$(python -c "import pybind11; print(pybind11.get_include())") \
        -I$(python -c "import numpy; print(numpy.get_include())") \
        mesh_inpaint_processor.cpp \
        -o mesh_inpaint_processor$(python3-config --extension-suffix) || echo "WARNING: DifferentiableRenderer build failed"
fi
cd ../..

# Download RealESRGAN model if not present
echo ""
echo "Checking RealESRGAN model..."
if [ ! -f "hy3dpaint/ckpt/RealESRGAN_x4plus.pth" ]; then
    echo "Downloading RealESRGAN model..."
    mkdir -p hy3dpaint/ckpt
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt || echo "WARNING: Failed to download RealESRGAN model"
fi

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "To run the demo:"
echo "  source venv/bin/activate"
echo "  export TORCH_CUDA_ARCH_LIST=\"12.1a\""
echo "  python demo.py"
echo ""
echo "Or use: ./run_demo_spark.sh"
