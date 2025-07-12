# 🍏 Hunyuan3D-2.1 - macOS Local Usage Guide (Shape Only)

---

## ⚠️ Feature Limitation Notice

Due to macOS only supporting **Metal acceleration (MPS)** and **not CUDA**, Hunyuan3D-2.1 currently disables **texture generation** by default when running on macOS.

Reasons include:

- Texture functionality depends on CUDA-only modules such as `xatlas`, multi-view generation with `diffusers`, etc.
- macOS does **not support NVIDIA GPUs** or CUDA; Metal backend performance is limited.
- Texture-related libraries like `xatlas` and `bpy` are difficult to build or run on macOS.

👉 **Recommendation**: If you need full texture generation, please use **Linux or Windows with an NVIDIA GPU**.

---

## System Requirements

- **Operating System**: macOS 10.15 or later
- **Hardware**:
  - Apple Silicon chip (M1/M2/M3/M4)
  - 16GB+ RAM
  - 10GB+ free disk space
- **Software**:
  - [Anaconda](https://www.anaconda.com/download/success) environment manager recommended
  - Python 3.11–3.12 
  - Xcode Command Line Tools 
    ```bash
    xcode-select --install
    ```

---

## 📦 Installation Steps

### 1. Clone the repository

```bash
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
cd Hunyuan3D-2.1
```

### 2. Create and activate conda environment

```bash
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
cd Hunyuan3D-2.1
```

### 3. Install macOS-specific dependencies

```bash
pip install -r requirements_macos.txt
```

### 4. Download model 
Download the shape model files from Hugging Face:
📥 https://huggingface.co/tencent/Hunyuan3D-2.1/tree/main/hunyuan3d-dit-v2-1

---

## 🚀 Usage Instructions

### Run Demo (Shape Generation Only)

```bash
python demo_macos.py
```
### Gradio App 
```bash
python gradio_macos.py --device mps --low_vram_mode --disable_tex
```
Open your browser and go to: http://127.0.0.1:8080

### Code usage (untextured mesh generation)
```python
import sys
import torch
sys.path.insert(0, './hy3dshape')

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

device = torch.device("mps")

shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2.1',
    torch_dtype=torch.float32,
    device=device
)
shape_pipeline.to(device)

mesh_untextured = shape_pipeline(image='path/to/image.png', batch_size=1)[0]

mesh_untextured.export('demo.obj')
```
## Notes

- The script will automatically use Apple MPS if available, otherwise it falls back to CPU.
- It's recommended to disable texture (--disable_tex) to ensure compatibility and stability.
- If you encounter "python not found" errors, make sure the virtual environment is activated.