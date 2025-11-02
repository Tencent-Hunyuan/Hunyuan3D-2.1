import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

import logging
from PIL import Image
from birefnet_rembg import BiRefNetRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

# --- Logging Setup ---
# Configure the logging module to display INFO level messages.
# This is necessary to see the detailed logs from the paint pipeline.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
)

from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")                                      
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

# # shape
# logging.info("--- Starting Shape Generation Stage ---")
# model_path = 'tencent/Hunyuan3D-2.1'
# pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)

# image_path = 'assets/chair_front.png'
# logging.info(f"Loading image from: {image_path}")
# image = Image.open(image_path).convert("RGBA")
# logging.info("Image is RGB, removing background...")
# rembg = BiRefNetRemover()
# image = rembg(image)

# logging.info("Running shape generation pipeline...")
# mesh = pipeline_shapegen(image=image)[0]
# mesh.export('demo.glb')
# logging.info("--- Shape Generation Finished. Saved to demo.glb ---")

# paint
logging.info("--- Starting Paint Generation Stage ---")
max_num_view = 9  # can be 6 to 9
resolution = 512  # can be 768 or 512
conf = Hunyuan3DPaintConfig(max_num_view, resolution)
conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
paint_pipeline = Hunyuan3DPaintPipeline(conf)

output_mesh_path = 'demo_textured.glb'
output_mesh_path = paint_pipeline(
    mesh_path = "assets/1K.glb", 
    image_path = 'assets/3.jpg',
    output_mesh_path = output_mesh_path
)
logging.info(f"--- Paint Generation Finished. Saved to {output_mesh_path} ---")
