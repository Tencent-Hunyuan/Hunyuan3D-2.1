#!/bin/bash
# Run demo on NVIDIA DGX Spark
# GB10 GPU (Blackwell sm_121), CUDA 13.0, aarch64

cd "$(dirname "${BASH_SOURCE[0]}")"
source venv/bin/activate

# Environment variables for Blackwell GPU
export TORCH_CUDA_ARCH_LIST="12.1a"
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# Memory management for unified memory architecture
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Suppress the sm_121 capability warning (it's safe to ignore)
export TORCH_CPP_LOG_LEVEL=ERROR

python demo.py "$@"
