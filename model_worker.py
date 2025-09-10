"""
Model worker for Hunyuan3D API server.
"""
import os
import time
import uuid
import base64
import trimesh
from io import BytesIO
from PIL import Image
import torch

# Apply torchvision compatibility fix before other imports
import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.utils import logger
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from hy3dpaint.convert_utils import create_glb_with_pbr_materials


def quick_convert_with_obj2gltf(obj_path: str, glb_path: str):
    textures = {
        'albedo': obj_path.replace('.obj', '.jpg'),
        'metallic': obj_path.replace('.obj', '_metallic.jpg'),
        'roughness': obj_path.replace('.obj', '_roughness.jpg')
        }
    create_glb_with_pbr_materials(obj_path, textures, glb_path)


def load_image_from_base64(image):
    """
    Load an image from base64 encoded string.
    
    Args:
        image (str): Base64 encoded image string
        
    Returns:
        PIL.Image: Loaded image
    """
    return Image.open(BytesIO(base64.b64decode(image)))


class ModelWorker:
    """
    Worker class for handling 3D model generation tasks.
    """
    
    def __init__(self,
                 model_path='tencent/Hunyuan3D-2.1',
                 subfolder='hunyuan3d-dit-v2-1',
                 device='cuda',
                 low_vram_mode=False,
                 worker_id=None,
                 model_semaphore=None,
                 save_dir='gradio_cache'):
        """
        Initialize the model worker.
        
        Args:
            model_path (str): Path to the shape generation model
            subfolder (str): Subfolder containing the model files
            device (str): Device to run the model on ('cuda' or 'cpu')
            low_vram_mode (bool): Whether to use low VRAM mode
            worker_id (str): Unique identifier for this worker
            model_semaphore: Semaphore for controlling model concurrency
            save_dir (str): Directory to save generated files
        """
        self.model_path = model_path
        self.worker_id = worker_id or str(uuid.uuid4())[:6]
        self.device = device
        self.low_vram_mode = low_vram_mode
        self.model_semaphore = model_semaphore
        self.save_dir = save_dir
        
        logger.info(f"Loading the model {model_path} on worker {self.worker_id} ...")

        # Initialize background remover
        self.rembg = BackgroundRemover()
        
        # Initialize shape generation pipeline (matching demo.py)
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
        
        # Initialize texture generation pipeline (matching demo.py)
        max_num_view = 6  # can be 6 to 9
        resolution = 512  # can be 768 or 512
        conf = Hunyuan3DPaintConfig(max_num_view, resolution)
        conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
        conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
        self.paint_pipeline = Hunyuan3DPaintPipeline(conf)
        # clean cache in save_dir
        for file in os.listdir(self.save_dir):
            os.remove(os.path.join(self.save_dir, file))
            
    def get_queue_length(self):
        """
        Get the current queue length for model processing.
        
        Returns:
            int: Number of tasks in the queue
        """
        if self.model_semaphore is None:
            return 0
        else:
            return (self.model_semaphore._value if hasattr(self.model_semaphore, '_value') else 0) + \
                   (len(self.model_semaphore._waiters) if hasattr(self.model_semaphore, '_waiters') and self.model_semaphore._waiters is not None else 0)

    def get_status(self):
        """
        Get the current status of the worker.
        
        Returns:
            dict: Status information including speed and queue length
        """
        return {
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate(self, uid, params):
        """
        Generate a 3D model from the given parameters.
        
        Args:
            uid: Unique identifier for this generation task
            params (dict): Generation parameters including image and options
            
        Returns:
            tuple: (file_path, uid) - Path to generated file and task ID
        """
        start_time = time.time()
        logger.info(f"Generating 3D model for uid: {uid}")
        # Handle input image
        if 'image' in params:
            image = params["image"]
            image = load_image_from_base64(image)
        else:
            raise ValueError("No input image provided")

        # Convert to RGBA and remove background if needed
        image = image.convert("RGBA")
        if image.mode == "RGB":
            image = self.rembg(image)

        # Generate mesh 
        try:
            mesh = self.pipeline(image=image)[0]
            logger.info("---Shape generation takes %s seconds ---" % (time.time() - start_time))
        except Exception as e:
            logger.error(f"Shape generation failed: {e}")
            raise ValueError(f"Failed to generate 3D mesh: {str(e)}")

        # Check if texture generation is requested
        generate_texture = params.get('texture', False)
        
        # Export initial mesh without texture
        initial_save_path = os.path.join(self.save_dir, f'{str(uid)}_initial.glb')
        mesh.export(initial_save_path)
        #Shape generation is done
        
        if generate_texture:
            # Generate textured mesh as obj ( as in demo )
            try:
                output_mesh_path_obj = os.path.join(self.save_dir, f'{str(uid)}_texturing.obj')
                textured_path_obj = self.paint_pipeline(
                    mesh_path=initial_save_path,
                    image_path=image,
                    output_mesh_path=output_mesh_path_obj,
                    save_glb=False            
                )
                logger.info("---Texture generation takes %s seconds ---" % (time.time() - start_time))
                logger.info(f"output_mesh_path: {output_mesh_path_obj} textured_path: {textured_path_obj}")

                # Convert textured OBJ to GLB using obj2gltf with PBR support
                print("convert textured OBJ to GLB")
                glb_path_textured = os.path.join(self.save_dir, f'{str(uid)}_texturing.glb')
                quick_convert_with_obj2gltf(textured_path_obj, glb_path_textured)
                # now rename glb_path to uid_textured.glb
                print("done.")
                final_save_path = os.path.join(self.save_dir, f'{str(uid)}_textured.glb')
                os.rename(glb_path_textured, final_save_path)
                print(f"final_save_path: {final_save_path}")

                
            except Exception as e:
                logger.error(f"Texture generation failed: {e}")
                # Fall back to untextured mesh if texture generation fails
                final_save_path = initial_save_path
                logger.warning(f"Using untextured mesh as fallback: {final_save_path}")
        else:
            # Skip texture generation, use initial mesh as final output
            logger.info("Skipping texture generation as requested")
            final_save_path = initial_save_path

        if self.low_vram_mode:
            torch.cuda.empty_cache()
            
        logger.info("---Total generation takes %s seconds ---" % (time.time() - start_time))
        # Alpha completion action calls the webhook to notify the client about the completion

        return final_save_path, uid

    @torch.inference_mode()
    def generate_multiview(self, uid, params):
        """
        Generate a 3D model from multiple view images.
        
        Args:
            uid: Unique identifier for this generation task
            params (dict): Generation parameters including images dict and options
            
        Returns:
            tuple: (file_path, uid) - Path to generated file and task ID
        """
        start_time = time.time()
        logger.info(f"Generating 3D model from multiple views for uid: {uid}")
        
        # Handle input images
        if 'images' in params:
            images_dict = params["images"]
            if not images_dict or len(images_dict) == 0:
                raise ValueError("No input images provided")
            
            # Convert base64 images to PIL Images
            images = {}
            for view_name, image_b64 in images_dict.items():
                if view_name in ['front', 'back', 'left', 'right']:
                    images[view_name] = load_image_from_base64(image_b64)
                else:
                    logger.warning(f"Unknown view name: {view_name}, skipping")
        else:
            raise ValueError("No input images provided")

        # Convert to RGBA and remove background if needed
        for view_name, image in images.items():
            image = image.convert("RGBA")
            if image.mode == "RGB":
                images[view_name] = self.rembg(image)

        # Generate mesh from multiple views using the same approach as gradio app
        try:
            # Set up generator with seed
            generator = torch.Generator()
            generator = generator.manual_seed(int(params.get('seed', 1234)))
            
            # Call pipeline with multiple views (same as gradio app)
            outputs = self.pipeline(
                image=images,
                num_inference_steps=params.get('num_inference_steps', 5),
                guidance_scale=params.get('guidance_scale', 5.0),
                generator=generator,
                octree_resolution=params.get('octree_resolution', 256),
                num_chunks=params.get('num_chunks', 8000),
                output_type='mesh'
            )
            
            # Extract mesh from outputs (same as gradio app)
            from hy3dshape.pipelines import export_to_trimesh
            mesh = export_to_trimesh(outputs)[0]
            
            logger.info("---Multiview shape generation takes %s seconds ---" % (time.time() - start_time))
        except Exception as e:
            logger.error(f"Multiview shape generation failed: {e}")
            raise ValueError(f"Failed to generate 3D mesh from multiple views: {str(e)}")

        # Check if texture generation is requested
        generate_texture = params.get('texture', False)
        
        # Export initial mesh without texture
        initial_save_path = os.path.join(self.save_dir, f'{str(uid)}_initial.glb')
        mesh.export(initial_save_path)
        
        if generate_texture:
            # Generate textured mesh as obj ( as in demo )
            try:
                output_mesh_path_obj = os.path.join(self.save_dir, f'{str(uid)}_texturing.obj')
                # Pass all available views to texture pipeline for better quality
                # The texture pipeline can handle multiple images and will use them for multiview texture generation
                textured_path_obj = self.paint_pipeline(
                    mesh_path=initial_save_path,
                    image_path=list(images.values()),  # Pass all views as a list
                    output_mesh_path=output_mesh_path_obj,
                    save_glb=False            
                )
                logger.info("---Multiview texture generation takes %s seconds ---" % (time.time() - start_time))
                logger.info(f"output_mesh_path: {output_mesh_path_obj} textured_path: {textured_path_obj}")

                # Convert textured OBJ to GLB using obj2gltf with PBR support
                print("convert textured OBJ to GLB")
                glb_path_textured = os.path.join(self.save_dir, f'{str(uid)}_texturing.glb')
                quick_convert_with_obj2gltf(textured_path_obj, glb_path_textured)
                # now rename glb_path to uid_textured.glb
                print("done.")
                final_save_path = os.path.join(self.save_dir, f'{str(uid)}_textured.glb')
                os.rename(glb_path_textured, final_save_path)
                print(f"final_save_path: {final_save_path}")

                
            except Exception as e:
                logger.error(f"Multiview texture generation failed: {e}")
                # Fall back to untextured mesh if texture generation fails
                final_save_path = initial_save_path
                logger.warning(f"Using untextured mesh as fallback: {final_save_path}")
        else:
            # Skip texture generation, use initial mesh as final output
            logger.info("Skipping texture generation as requested")
            final_save_path = initial_save_path

        if self.low_vram_mode:
            torch.cuda.empty_cache()
            
        logger.info("---Total multiview generation takes %s seconds ---" % (time.time() - start_time))

        # Alpha completion action calls the webhook to notify the client about the completion

        return final_save_path, uid 