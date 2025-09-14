# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

"""
A model worker executes the model.
"""
import argparse
import asyncio
import base64
import logging
import os
import sys
import threading
import traceback
import uuid
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

# Import from root-level modules
from api_models import GenerationRequest, GenerationResponse, StatusResponse, HealthResponse
from logger_utils import build_logger
from constants import (
    SERVER_ERROR_MSG, DEFAULT_SAVE_DIR, API_TITLE, API_DESCRIPTION, 
    API_VERSION, API_CONTACT, API_LICENSE_INFO, API_TAGS_METADATA
)
from model_worker import ModelWorker

# Global variables
SAVE_DIR = DEFAULT_SAVE_DIR
worker_id = str(uuid.uuid4())[:6]
logger = build_logger("controller", f"{SAVE_DIR}/controller.log")

# Global worker and semaphore instances
worker = None
model_semaphore = None


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    contact=API_CONTACT,
    license_info=API_LICENSE_INFO,
    tags_metadata=API_TAGS_METADATA
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/generate", tags=["generation"])
async def generate_3d_model(request: GenerationRequest):
    """
    Generate a 3D model from an input image.
    
    This endpoint takes an image and generates a 3D model with optional textures.
    The generation process includes background removal, mesh generation, and optional texture mapping.
    
    Returns:
        FileResponse: The generated 3D model file (GLB or OBJ format)
    """
    logger.info("Worker generating...")
    
    # Convert Pydantic model to dict for compatibility
    params = request.dict()
    
    uid = uuid.uuid4()
    try:
        file_path, uid = worker.generate(uid, params)
        return FileResponse(file_path)
    except ValueError as e:
        traceback.print_exc()
        logger.error(f"Caught ValueError: {e}")
        ret = {
            "text": SERVER_ERROR_MSG,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)
    except torch.cuda.CudaError as e:
        logger.error(f"Caught torch.cuda.CudaError: {e}")
        ret = {
            "text": SERVER_ERROR_MSG,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)
    except Exception as e:
        logger.error(f"Caught Unknown Error: {e}")
        traceback.print_exc()
        ret = {
            "text": SERVER_ERROR_MSG,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)


@app.post("/send", response_model=GenerationResponse, tags=["generation"])
async def send_generation_task(request: GenerationRequest):
    """
    Send a 3D generation task to be processed asynchronously.
    
    This endpoint starts the generation process in the background and returns a task ID.
    Use the /status/{uid} endpoint to check the progress and retrieve the result.
    
    Returns:
        GenerationResponse: Contains the unique task identifier
    """
    logger.info("Worker send...")
    
    # Convert Pydantic model to dict for compatibility
    params = request.dict()
    
    uid = uuid.uuid4()
    try:
        threading.Thread(target=worker.generate, args=(uid, params,)).start()
        ret = {"uid": str(uid)}
        return JSONResponse(ret, status_code=200)
    except Exception as e:
        logger.error(f"Failed to start generation thread: {e}")
        ret = {"error": "Failed to start generation"}
        return JSONResponse(ret, status_code=500)


@app.get("/health", response_model=HealthResponse, tags=["status"])
async def health_check():
    """
    Health check endpoint to verify the service is running.
    
    Returns:
        HealthResponse: Service health status and worker identifier
    """
    return JSONResponse({"status": "healthy", "worker_id": worker_id}, status_code=200)


@app.get("/status/{uid}", response_model=StatusResponse, tags=["status"])
async def status(uid: str):
    """
    Check the status of a generation task.
    
    Args:
        uid: The unique identifier of the generation task
        
    Returns:
        StatusResponse: Current status of the task and result if completed
    """
    # Check for textured file first (preferred output)
    textured_file_path = os.path.join(SAVE_DIR, f'{uid}_textured.glb')
    initial_file_path = os.path.join(SAVE_DIR, f'{uid}_initial.glb')
    
    #print(f"Checking files: {textured_file_path} ({os.path.exists(textured_file_path)}), {initial_file_path} ({os.path.exists(initial_file_path)})")
    
    # If textured file exists, generation is complete
    if os.path.exists(textured_file_path):
        try:
            base64_str = base64.b64encode(open(textured_file_path, 'rb').read()).decode()
            response = {'status': 'completed', 'model_base64': base64_str}
            return JSONResponse(response, status_code=200)
        except Exception as e:
            logger.error(f"Error reading file {textured_file_path}: {e}")
            response = {'status': 'error', 'message': 'Failed to read generated file'}
            return JSONResponse(response, status_code=500)
    
    # If only initial file exists, texturing is in progress
    elif os.path.exists(initial_file_path):
        response = {'status': 'texturing'}
        return JSONResponse(response, status_code=200)
    
    # If no files exist, still processing
    else:
        response = {'status': 'processing'}
        return JSONResponse(response, status_code=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2.1')
    parser.add_argument("--subfolder", type=str, default='hunyuan3d-dit-v2-1')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument('--mc_algo', type=str, default='mc')
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument('--enable_flashvdm', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--low_vram_mode', action='store_true')
    parser.add_argument('--cache-path', type=str, default='./gradio_cache')
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # Update SAVE_DIR based on cache-path argument
    SAVE_DIR = args.cache_path
    os.makedirs(SAVE_DIR, exist_ok=True)
    

    model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)

    worker = ModelWorker(
        model_path=args.model_path, 
        subfolder=args.subfolder,
        device=args.device, 
        low_vram_mode=args.low_vram_mode,
        worker_id=worker_id,
        model_semaphore=model_semaphore,
        save_dir=SAVE_DIR,
        mc_algo=args.mc_algo,
        enable_flashvdm=args.enable_flashvdm,
        compile=args.compile
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
