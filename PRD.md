# PRD: Image to 3D GLB Conversion API

## Introduction

Create a FastAPI endpoint that converts a single image into a textured 3D GLB model using the Hunyuan3D-2.1 pipeline. The endpoint accepts image uploads, processes them synchronously, and returns the GLB file directly. Generated files are stored permanently. The ML pipelines are loaded lazily and automatically freed after 1 hour of inactivity to conserve GPU memory.

## Goals

- Provide a simple REST API endpoint for image-to-3D conversion
- Return textured GLB files directly in the response
- Store generated files permanently for later access
- Optimize GPU memory by unloading pipelines after 1 hour of inactivity
- Reuse existing `demo.py` logic for the conversion process

## User Stories

### US-001: Create FastAPI application structure
**Description:** As a developer, I need a basic FastAPI application setup so that I can add the conversion endpoint.

**Acceptance Criteria:**
- [x] Create `api.py` with FastAPI app instance
- [x] Add `/health` endpoint that returns `{"status": "ok"}`
- [x] App can be run with `uvicorn api:app`
- [x] Typecheck passes

### US-002: Implement pipeline manager with lazy loading
**Description:** As a system operator, I want the ML pipelines to load only when needed and unload after 1 hour of inactivity so that GPU memory is not wasted.

**Acceptance Criteria:**
- [x] Create `PipelineManager` class that lazily loads shape and texture pipelines
- [x] Track last usage timestamp
- [x] Background task checks every 5 minutes and unloads if inactive for 1 hour
- [x] `get_pipelines()` method returns loaded pipelines, loading them if needed
- [x] `unload()` method frees GPU memory (del pipelines, gc.collect, torch.cuda.empty_cache)
- [x] Thread-safe access to pipelines (use threading.Lock)
- [x] Typecheck passes

### US-003: Implement image-to-GLB conversion endpoint
**Description:** As an API user, I want to upload an image and receive a textured GLB file so that I can use 3D models in my application.

**Acceptance Criteria:**
- [x] POST `/convert-image-to-3d` endpoint accepts multipart file upload
- [x] Validates file is an image (png, jpg, jpeg, webp)
- [x] Returns 400 error for invalid file types
- [x] Processes image through shape generation pipeline
- [x] Processes mesh through texture generation pipeline
- [x] Returns GLB file with `application/octet-stream` content type
- [x] Returns appropriate filename in Content-Disposition header
- [x] Typecheck passes

### US-004: Add permanent file storage
**Description:** As an API user, I want generated GLB files stored permanently so that I can download them later.

**Acceptance Criteria:**
- [x] Generated files saved to `outputs/` directory with unique timestamp-based names
- [x] Add GET `/outputs/{filename}` endpoint to download stored files
- [x] Add GET `/outputs` endpoint to list all stored files
- [x] Returns 404 if file not found
- [x] Typecheck passes

### US-005: Add error handling and logging
**Description:** As a developer, I want proper error handling and logging so that I can diagnose issues in production.

**Acceptance Criteria:**
- [ ] Conversion errors return 500 with error message in JSON
- [ ] Log pipeline load/unload events
- [ ] Log each conversion request with input filename and processing time
- [ ] Log errors with stack traces
- [ ] Typecheck passes

## Non-Goals

- No authentication or API keys
- No rate limiting
- No async/job queue processing - requests block until complete
- No batch processing of multiple images
- No configurable output formats (GLB only)
- No web UI or Gradio interface

## Technical Considerations

- Reuse pipeline loading logic from `demo.py` lines 150-166
- Reuse `process_image()` conversion logic from `demo.py` lines 58-143
- Use `BackgroundTasks` or `asyncio` for the inactivity checker
- Pipeline loading takes ~30-60 seconds on first request
- Each conversion takes ~60-120 seconds depending on GPU
- Requires ~10-20GB VRAM when pipelines are loaded
