## 🤗 Get Started with Hunyuan3D 2.1 On MacOS

Hunyuan3D 2.1 supports Macos, Windows, Linux. You may follow the next steps to use Hunyuan3D 2.1 via for your Apple Silicon chip devices:

### Install Requirements
We test our model on an A100 GPU with Python 3.10 and PyTorch for Apple silicon, reference: https://developer.apple.com/metal/pytorch/.
```bash
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install py-cpuinfo
pip install -r requirements_for_apple_silicon.txt

cd hy3dpaint/custom_rasterizer
export FORCE_CPU=1
pip install .
cd ../..
cd hy3dpaint/DifferentiableRenderer
bash compile_mesh_painter_for_apple_silicon.sh
cd ../..

wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt
```

### Code Usage

We designed a diffusers-like API to use our shape generation model - Hunyuan3D-Shape and texture synthesis model -
Hunyuan3D-Paint.

You run this to test the demo.
```python
python demo_for_apple_silicon.py
```

```python
import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')
from textureGenPipeline import Hunyuan3DPaintPipeline
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

# let's generate a mesh first
shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')
mesh_untextured = shape_pipeline(image='assets/demo.png')[0]

paint_pipeline = Hunyuan3DPaintPipeline(Hunyuan3DPaintConfig(max_num_view=6, resolution=512))
mesh_textured = paint_pipeline(mesh_path, image_path='assets/demo.png')
```


### Gradio App

You could also host a [Gradio](https://www.gradio.app/) App in your own computer via:


```bash
python gradio_app.py \
  --model_path tencent/Hunyuan3D-2.1 \
  --subfolder hunyuan3d-dit-v2-1 \
  --texgen_model_path tencent/Hunyuan3D-2.1 \
  --device mps \
  --low_vram_mode
```
