import sys
import gc

sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

from PIL import Image
import torch

from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

try:
    from torchvision_fix import apply_fix

    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")

# shape
model_path = 'tencent/Hunyuan3D-2.1'


pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path, device=device)

image_path = 'assets/demo.png'
image = Image.open(image_path).convert("RGBA")
if image.mode == 'RGB':
    rembg = BackgroundRemover()
    image = rembg(image)

mesh = pipeline_shapegen(image=image, num_inference_steps=50)[0]
mesh.export('demo.glb')

# clean up
del pipeline_shapegen
del mesh
gc.collect()
torch.cuda.empty_cache()

# paint
max_num_view = 6  # can be 6 to 9
resolution = 512  # can be 768 or 512
conf = Hunyuan3DPaintConfig(max_num_view, resolution)
conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
conf.device = device

paint_pipeline = Hunyuan3DPaintPipeline(conf)

output_mesh_path = 'demo_textured.glb'
output_mesh_path = paint_pipeline(
    mesh_path="demo.glb",
    image_path='assets/demo.png',
    output_mesh_path=output_mesh_path
)
