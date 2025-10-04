#!/usr/bin/env python
"""
SageMaker serving wrapper for Hunyuan3D-2.1 API Server
Adapts the FastAPI server to work with SageMaker's requirements
"""

import os
import sys
import asyncio
import uuid
import base64
import io
import traceback

# Add paths for Hunyuan3D modules
sys.path.insert(0, '/workspace/Hunyuan3D-2.1')
sys.path.insert(0, '/workspace/Hunyuan3D-2.1/hy3dshape')
sys.path.insert(0, '/workspace/Hunyuan3D-2.1/hy3dpaint')

import torch
import trimesh
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image

# Import from root-level modules
from api_models import GenerationRequest
from logger_utils import build_logger
from constants import DEFAULT_SAVE_DIR
from model_worker import ModelWorker

# Import mesh processing functions
from hy3dshape.postprocessors import FloaterRemover, DegenerateFaceRemover, FaceReducer

# Configuration
SAVE_DIR = os.environ.get('SAGEMAKER_MODEL_DIR', DEFAULT_SAVE_DIR)
worker_id = str(uuid.uuid4())[:6]
logger = build_logger("sagemaker_server", f"{SAVE_DIR}/sagemaker_server.log")

# Global variables
worker = None
model_semaphore = None
floater_remover = None
degenerate_remover = None
face_reducer = None
model_initialized = False

# Create FastAPI app
app = FastAPI(
    title="Hunyuan3D-2.1 SageMaker Endpoint",
    description="SageMaker-compatible endpoint for Hunyuan3D-2.1",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def process_mesh(mesh: trimesh.Trimesh, max_faces: int = 40000) -> trimesh.Trimesh:
    """
    Process mesh by removing floaters, degenerate faces, and reducing face count.
    """
    logger.info("Processing mesh...")
    
    # Remove floating components
    mesh = floater_remover(mesh)
    
    # Remove degenerate faces
    mesh = degenerate_remover(mesh)
    
    # Reduce face count if needed
    if mesh.faces.shape[0] > max_faces:
        logger.info(f"Reducing faces from {mesh.faces.shape[0]} to {max_faces}...")
        mesh = face_reducer(mesh, max_facenum=max_faces)
    
    return mesh


def initialize_model():
    """Initialize the Hunyuan3D model and mesh processors"""
    global worker, model_semaphore, floater_remover, degenerate_remover, face_reducer, model_initialized
    
    try:
        logger.info("Initializing Hunyuan3D-2.1 model for SageMaker...")
        
        # Create save directory
        os.makedirs(SAVE_DIR, exist_ok=True)
        
        # Initialize mesh processors
        logger.info("Initializing mesh processors...")
        floater_remover = FloaterRemover()
        degenerate_remover = DegenerateFaceRemover()
        face_reducer = FaceReducer()
        
        # Initialize model semaphore
        model_semaphore = asyncio.Semaphore(5)
        
        # Get model configuration from environment or use defaults
        model_path = os.environ.get('MODEL_PATH', 'tencent/Hunyuan3D-2.1')
        subfolder = os.environ.get('MODEL_SUBFOLDER', 'hunyuan3d-dit-v2-1')
        device = os.environ.get('DEVICE', 'cuda')
        enable_flashvdm = os.environ.get('ENABLE_FLASHVDM', 'true').lower() == 'true'
        low_vram_mode = os.environ.get('LOW_VRAM_MODE', 'false').lower() == 'true'
        mc_algo = os.environ.get('MC_ALGO', 'mc')
        compile_model = os.environ.get('COMPILE_MODEL', 'false').lower() == 'true'
        
        logger.info(f"Model configuration:")
        logger.info(f"  - Model path: {model_path}")
        logger.info(f"  - Subfolder: {subfolder}")
        logger.info(f"  - Device: {device}")
        logger.info(f"  - FlashVDM: {enable_flashvdm}")
        logger.info(f"  - Low VRAM mode: {low_vram_mode}")
        
        # Initialize model worker
        logger.info("Initializing model worker (this may take a few minutes)...")
        worker = ModelWorker(
            model_path=model_path,
            subfolder=subfolder,
            device=device,
            low_vram_mode=low_vram_mode,
            worker_id=worker_id,
            model_semaphore=model_semaphore,
            save_dir=SAVE_DIR,
            mc_algo=mc_algo,
            enable_flashvdm=enable_flashvdm,
            compile=compile_model
        )
        
        model_initialized = True
        logger.info("Model initialization complete!")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        traceback.print_exc()
        model_initialized = False
        return False


@app.get("/ping")
async def ping():
    """
    SageMaker health check endpoint
    Returns 200 if model is loaded and ready
    """
    if model_initialized and worker is not None:
        return Response(status_code=200)
    else:
        return Response(status_code=503)


@app.post("/invocations")
async def invocations(request: Request):
    """
    SageMaker inference endpoint
    """
    file_path = None
    
    try:
        # Check if model is initialized
        if not model_initialized or worker is None:
            logger.error("Model not initialized")
            raise HTTPException(status_code=503, detail="Model not initialized")
        
        # Parse input JSON
        content_type = request.headers.get('content-type', '')
        if 'application/json' not in content_type:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported content type: {content_type}. Expected application/json"
            )
        
        input_data = await request.json()
        logger.info(f"Received inference request")
        logger.info(f"Request keys: {list(input_data.keys())}")
        
        # Validate required fields
        if 'image' not in input_data:
            logger.error("Missing 'image' field in request")
            raise HTTPException(
                status_code=400,
                detail="Missing required field: 'image' (base64 encoded)"
            )
        
        # Extract parameters
        image_b64 = input_data['image']
        remove_background = input_data.get('remove_background', True)
        texture = input_data.get('texture', False)
        seed = input_data.get('seed', 1234)
        octree_resolution = input_data.get('octree_resolution', 256)
        num_inference_steps = input_data.get('num_inference_steps', 5)
        guidance_scale = input_data.get('guidance_scale', 5.0)
        num_chunks = input_data.get('num_chunks', 8000)
        face_count = input_data.get('face_count', 40000)
        output_format = input_data.get('format', input_data.get('type', 'stl')).lower()
        
        # Validate output format
        if output_format not in ['stl', 'glb', 'obj']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid format: {output_format}. Must be 'stl', 'glb', or 'obj'"
            )
        
        logger.info(f"Processing request:")
        logger.info(f"  - image size: {len(image_b64)} bytes (base64)")
        logger.info(f"  - remove_background: {remove_background}")
        logger.info(f"  - texture: {texture}")
        logger.info(f"  - seed: {seed}")
        logger.info(f"  - face_count: {face_count}")
        logger.info(f"  - format: {output_format}")
        
        # Validate base64 image
        try:
            import base64 as b64
            image_bytes = b64.b64decode(image_b64)
            test_image = Image.open(io.BytesIO(image_bytes))
            logger.info(f"Validated image: size={test_image.size}, mode={test_image.mode}")
        except Exception as e:
            logger.error(f"Failed to decode/validate image: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image data: {str(e)}"
            )
        
        # Create GenerationRequest
        uid = uuid.uuid4()
        from api_models import GenerationRequest
        
        generation_request = GenerationRequest(
            image=image_b64,
            remove_background=remove_background,
            texture=texture,
            seed=seed,
            octree_resolution=octree_resolution,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_chunks=num_chunks,
            face_count=face_count,
            type=output_format if output_format in ['stl', 'glb'] else 'stl'
        )
        
        params = generation_request.dict()
        logger.info(f"Created GenerationRequest with fields: {list(params.keys())}")
        logger.info(f"Calling worker.generate() with uid={uid}")
        
        # Generate 3D model
        try:
            file_path, uid = worker.generate(uid, params)
            logger.info(f"Generation complete: {file_path}")
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            logger.error(f"Params used: {list(params.keys())}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Generation failed: {str(e)}"
            )
        
        # Load and process mesh
        logger.info("Loading and processing mesh...")
        mesh = trimesh.load(file_path)
        processed_mesh = process_mesh(mesh, max_faces=face_count)
        
        # For async inference: Upload files directly to S3
        # Get S3 output path from environment variable
        s3_output_path = os.environ.get('SAGEMAKER_OUTPUT_PATH', '')
        
        logger.info(f"S3 output path: {s3_output_path}")
        
        # Generate filenames
        glb_filename = f"{uid}.glb"
        stl_filename = f"{uid}.{output_format}"
        
        # Save files locally first
        local_output_dir = "/tmp/output"
        os.makedirs(local_output_dir, exist_ok=True)
        
        # 1. Save original GLB
        import shutil
        glb_local_path = os.path.join(local_output_dir, glb_filename)
        shutil.copy(file_path, glb_local_path)
        logger.info(f"Saved GLB locally: {glb_local_path}")
        
        # 2. Export processed mesh
        stl_local_path = os.path.join(local_output_dir, stl_filename)
        processed_mesh.export(stl_local_path)
        logger.info(f"Saved STL locally: {stl_local_path}")
        
        # 3. Upload to S3 if output path is available
        if s3_output_path:
            try:
                import boto3
                s3_client = boto3.client('s3')
                
                # Parse S3 path: s3://bucket/key/prefix
                s3_parts = s3_output_path.replace('s3://', '').split('/', 1)
                bucket = s3_parts[0]
                prefix = s3_parts[1] if len(s3_parts) > 1 else ''
                
                # Upload GLB
                glb_s3_key = f"{prefix}/{glb_filename}"
                s3_client.upload_file(glb_local_path, bucket, glb_s3_key)
                logger.info(f"Uploaded GLB to S3: s3://{bucket}/{glb_s3_key}")
                
                # Upload STL
                stl_s3_key = f"{prefix}/{stl_filename}"
                s3_client.upload_file(stl_local_path, bucket, stl_s3_key)
                logger.info(f"Uploaded STL to S3: s3://{bucket}/{stl_s3_key}")
                
                # Return metadata with S3 paths
                response = {
                    'success': True,
                    'format': output_format,
                    'faces': int(processed_mesh.faces.shape[0]),
                    'vertices': int(processed_mesh.vertices.shape[0]),
                    'worker_id': worker_id,
                    'glb_filename': glb_filename,
                    'stl_filename': stl_filename,
                    'glb_s3_path': f"s3://{bucket}/{glb_s3_key}",
                    'stl_s3_path': f"s3://{bucket}/{stl_s3_key}",
                    'message': f'Models uploaded to S3: {glb_filename} and {stl_filename}'
                }
                
            except Exception as e:
                logger.error(f"Failed to upload to S3: {e}")
                # Fallback to base64 if S3 upload fails
                with open(glb_local_path, 'rb') as f_glb:
                    glb_model_data = base64.b64encode(f_glb.read()).decode('utf-8')
                with open(stl_local_path, 'rb') as f_stl:
                    stl_model_data = base64.b64encode(f_stl.read()).decode('utf-8')
                response = {
                    'success': True,
                    'glb_model_data': glb_model_data,
                    'stl_model_data': stl_model_data,
                    'format': output_format,
                    'faces': int(processed_mesh.faces.shape[0]),
                    'vertices': int(processed_mesh.vertices.shape[0]),
                    'worker_id': worker_id,
                    'error': f'S3 upload failed: {str(e)}'
                }
        else:
            # No S3 path available, return base64
            logger.warning("No SAGEMAKER_OUTPUT_PATH found, returning base64")
            with open(glb_local_path, 'rb') as f_glb:
                    glb_model_data = base64.b64encode(f_glb.read()).decode('utf-8')
                with open(stl_local_path, 'rb') as f_stl:
                    stl_model_data = base64.b64encode(f_stl.read()).decode('utf-8')
            response = {
                'success': True,
                'glb_model_data': glb_model_data,
                'stl_model_data': stl_model_data,
                'format': output_format,
                'faces': int(processed_mesh.faces.shape[0]),
                'vertices': int(processed_mesh.vertices.shape[0]),
                'worker_id': worker_id
            }
        
        # Clean up local files
        try:
            os.remove(glb_local_path)
            os.remove(stl_local_path)
            logger.info("Cleaned up local temp files")
        except Exception as e:
            logger.warning(f"Failed to clean up local files: {e}")
        
        logger.info(f"Successfully generated model: {processed_mesh.faces.shape[0]} faces, {processed_mesh.vertices.shape[0]} vertices")
        
        return JSONResponse(content=response, status_code=200)
        
    except HTTPException:
        raise
    except torch.cuda.CudaError as e:
        error_msg = f"CUDA error: {str(e)}"
        logger.error(error_msg)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        error_msg = f"Unexpected error during inference: {str(e)}"
        logger.error(error_msg)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        # Always clean up the original GLB from gradio_cache
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up original GLB: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up GLB: {e}")


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Hunyuan3D-2.1 SageMaker Endpoint",
        "status": "running" if model_initialized else "initializing",
        "worker_id": worker_id,
        "endpoints": {
            "health": "/ping",
            "inference": "/invocations"
        }
    }


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info("Starting up SageMaker endpoint...")
    initialize_model()


if __name__ == '__main__':
    # Get port from environment (SageMaker uses 8080)
    port = int(os.environ.get('SAGEMAKER_BIND_TO_PORT', 8080))
    host = os.environ.get('SAGEMAKER_BIND_TO_HOST', '0.0.0.0')
    
    logger.info(f"Starting SageMaker inference server on {host}:{port}")
    
    # Start uvicorn server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        timeout_keep_alive=300  # 5 minutes timeout for long-running requests
    )