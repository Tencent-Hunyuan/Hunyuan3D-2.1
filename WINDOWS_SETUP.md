# Windows环境下安装和使用Hunyuan3D-2.1

本指南将帮助您在Windows环境下安装和使用Hunyuan3D-2.1。
本指南在RTX 4090 与 RTX 5080 推理成功，但显存需要比较多（至少32G显卡），只有等后续量化模型了。
## 系统需求

- Windows 10或Windows 11操作系统
- NVIDIA GPU (推荐32GB以上显存)
- CUDA 12.4+ (兼容PyTorch 2.5.1)
- Python 3.10+ (推荐3.10)
- Visual Studio 2019/2022 (用于编译C++组件)或MinGW-w64 (GCC)

## 安装步骤

### 1. 安装依赖软件

- **安装Python 3.10**：从[Python官网](https://www.python.org/downloads/windows/)下载并安装
- **安装CUDA工具包**：从[NVIDIA官网](https://developer.nvidia.com/cuda-downloads)下载并安装CUDA 12.4
- **安装Visual Studio 2019/2022**：安装时请选择"使用C++的桌面开发"工作负载
- **[推荐] 安装Conda**：从[Conda官网](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html)下载并安装Miniconda或Anaconda

### 2. 使用Conda创建隔离环境（推荐）

使用Conda创建隔离环境可以避免与系统Python环境产生冲突：

```bash
# 创建名为hunyuan3d的Python 3.10环境
conda create -n hunyuan3d python=3.10
# 激活环境
conda activate hunyuan3d
# 安装PyTorch (使用pip而非conda安装以确保使用正确的CUDA版本)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 
# 安装项目依赖
pip install -r requirements_win.txt
# 完成环境配置后，编译自定义组件
cd hy3dpaint/custom_rasterizer
call build_custom_rasterizer.bat
cd ../DifferentiableRenderer
call compile_mesh_painter.bat
cd ../..
```

注意：每次使用Hunyuan3D-2.1时，都需要先激活conda环境：
```bash
conda activate hunyuan3d
```

### 3. 手动安装步骤

如果上述方法不适合您，可以按照以下步骤手动完成安装：

#### 3.1 安装PyTorch和依赖项

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_win.txt
```

#### 3.2 编译自定义光栅化器

```bash
cd hy3dpaint/custom_rasterizer
build_custom_rasterizer.bat
cd ../..
```

#### 3.3 编译网格处理器

```bash
cd hy3dpaint/DifferentiableRenderer
compile_mesh_painter.bat
cd ../..
```

#### 4. 下载预训练模型

```bash
mkdir -p hy3dpaint/ckpt
# 使用浏览器下载模型并放置在指定目录。(或者用wget，在https://eternallybored.org/misc/wget/ 官网下载后放到安全的位置在设置环境变量)
# 下载链接: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
# 目标位置: hy3dpaint/ckpt/RealESRGAN_x4plus.pth
```

## 使用指南

### 启动Gradio应用

```bash
python demo.py
```

启动后，成功推理就代表没问题啦


### 通过Python代码使用

```python
import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

# 生成3D网格
shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')
mesh_untextured = shape_pipeline(image='assets/demo.png')[0]

# 生成纹理
paint_pipeline = Hunyuan3DPaintPipeline(Hunyuan3DPaintConfig(max_num_view=6, resolution=512))
mesh_textured = paint_pipeline(mesh_untextured, image_path='assets/demo.png')
```

