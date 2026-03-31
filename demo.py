import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

import os
import time
import argparse
import glob
from datetime import datetime
import torch
import gc
import trimesh
from PIL import Image
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

# ========== ARGUMENT PARSING ==========
parser = argparse.ArgumentParser(description='Generate 3D model with texture from an image')
parser.add_argument('--input', '-i', type=str, default=None,
                    help='Path to input image')
parser.add_argument('--input-dir', '-d', type=str, default=None,
                    help='Path to directory containing input images (processes all .png, .jpg, .jpeg files)')
args = parser.parse_args()

# ========== INPUT VALIDATION ==========
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp')

if args.input and args.input_dir:
    parser.error("Cannot specify both --input and --input-dir")
elif args.input_dir:
    if not os.path.isdir(args.input_dir):
        parser.error(f"Input directory does not exist: {args.input_dir}")
    image_paths = sorted([
        os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ])
    if not image_paths:
        parser.error(f"No image files found in directory: {args.input_dir}")
    print(f"Found {len(image_paths)} images to process")
elif args.input:
    if not os.path.isfile(args.input):
        parser.error(f"Input file does not exist: {args.input}")
    image_paths = [args.input]
else:
    # Default to example image
    image_paths = ['assets/example_images/052.png']


def process_image(image_path, pipeline_shapegen, paint_pipeline, rembg):
    """Process a single image through shape and texture generation."""
    timings = {}
    
    # Generate timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join('save_dir', f'{input_name}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Output paths
    output_glb = os.path.join(output_dir, f'{input_name}.glb')
    output_textured_obj = os.path.join(output_dir, f'{input_name}_textured.obj')
    output_textured_glb = os.path.join(output_dir, f'{input_name}_textured.glb')

    # ========== IMAGE PREPROCESSING ==========
    print(f"Starting shape generation for {image_path}...")
    phase_start = time.time()
    image = Image.open(image_path)
    if image.mode == 'RGB':
        print("Image has no alpha channel, performing background removal...")
        image = rembg(image)
        print("Background removal complete")
    else:
        print("Image already has alpha channel, skipping background removal")
        image = image.convert("RGBA")
    timings['Image Preprocessing'] = time.time() - phase_start
    
    # Save background-removed image for debugging
    rembg_debug_path = os.path.join(output_dir, f'{input_name}_rembg.png')
    image.save(rembg_debug_path)
    print(f"Saved background-removed image to {rembg_debug_path}")

    # ========== SHAPE GENERATION ==========
    phase_start = time.time()
    mesh = pipeline_shapegen(image=image, num_inference_steps=25)[0]
    timings['Shape Generation'] = time.time() - phase_start

    phase_start = time.time()
    mesh.export(output_glb)
    timings['Shape Export'] = time.time() - phase_start
    print(f"Shape generation complete, saved to {output_glb}")

    # ========== TEXTURE GENERATION ==========
    print("Starting texture generation...")
    phase_start = time.time()
    output_mesh_path = paint_pipeline(
        mesh_path = output_glb, 
        image_path = image,  # Use background-removed image
        output_mesh_path = output_textured_obj,
        save_glb = False
    )
    timings['Texture Generation'] = time.time() - phase_start
    print(f"Texture generation complete, saved to {output_mesh_path}")

    # ========== CONVERT TO GLB WITH TRIMESH ==========
    print("Converting textured OBJ to GLB...")
    phase_start = time.time()
    try:
        mesh_textured = trimesh.load(output_mesh_path, force='mesh')
        mesh_textured.export(output_textured_glb)
        timings['GLB Conversion'] = time.time() - phase_start
        print(f"GLB export complete, saved to {output_textured_glb}")
        
        # Clean up intermediate files - keep only textured GLB
        print("Cleaning up intermediate files...")
        cleanup_patterns = [
            output_glb,  # untextured GLB
            output_textured_obj,  # textured OBJ
            output_mesh_path.replace('.obj', '.mtl'),  # MTL file
            output_mesh_path.replace('.obj', '.jpg'),  # base texture
            output_mesh_path.replace('.obj', '_metallic.jpg'),  # metallic map
            output_mesh_path.replace('.obj', '_roughness.jpg'),  # roughness map
            os.path.join(output_dir, 'white_mesh_remesh.obj'),  # temp mesh
        ]
        for pattern in cleanup_patterns:
            for file in glob.glob(pattern):
                if os.path.exists(file):
                    os.remove(file)
                    print(f"  Removed: {os.path.basename(file)}")
        print(f"Final output: {output_textured_glb}")
    except Exception as e:
        timings['GLB Conversion'] = time.time() - phase_start
        print(f"GLB conversion failed: {e}")
    
    return timings, output_textured_glb


# ========== TIMING ==========
all_timings = {}
total_start = time.time()

# ========== LOAD PIPELINES ==========
print("Loading pipelines...")
phase_start = time.time()
model_path = 'tencent/Hunyuan3D-2.1'
pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
pipeline_shapegen.enable_flashvdm(replace_vae=False, mc_algo='mc')
rembg = BackgroundRemover()
all_timings['Shape Pipeline Loading'] = time.time() - phase_start

phase_start = time.time()
max_num_view = 4  # Reduced from 6 for memory
resolution = 512
conf = Hunyuan3DPaintConfig(max_num_view, resolution)
conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
paint_pipeline = Hunyuan3DPaintPipeline(conf)
all_timings['Texture Pipeline Loading'] = time.time() - phase_start

# ========== PROCESS IMAGES ==========
results = []
for idx, image_path in enumerate(image_paths, 1):
    print(f"\n{'='*60}")
    print(f"Processing image {idx}/{len(image_paths)}: {image_path}")
    print('='*60)
    
    image_start = time.time()
    timings, output_path = process_image(image_path, pipeline_shapegen, paint_pipeline, rembg)
    image_time = time.time() - image_start
    
    results.append({
        'input': image_path,
        'output': output_path,
        'time': image_time,
        'timings': timings
    })

# ========== CLEANUP ==========
print("\nFreeing GPU memory...")
phase_start = time.time()
del pipeline_shapegen
del paint_pipeline
gc.collect()
torch.cuda.empty_cache()
all_timings['Memory Cleanup'] = time.time() - phase_start

# ========== TIMING SUMMARY ==========
total_time = time.time() - total_start
print("\n" + "="*60)
print("TIMING SUMMARY")
print("="*60)
for phase, duration in all_timings.items():
    print(f"{phase:.<40} {duration:>8.2f}s")
print("-"*60)
for result in results:
    name = os.path.basename(result['input'])
    print(f"{name:.<40} {result['time']:>8.2f}s")
print("-"*60)
print(f"{'TOTAL TIME':.<40} {total_time:>8.2f}s")
print("="*60)

if len(results) > 1:
    print(f"\nProcessed {len(results)} images successfully")
