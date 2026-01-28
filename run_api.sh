#!/bin/bash
# Start the Hunyuan3D-2.1 FastAPI server

cd "$(dirname "${BASH_SOURCE[0]}")"
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create outputs directory if it doesn't exist
mkdir -p outputs

# Default host and port
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "Starting Hunyuan3D-2.1 API server on ${HOST}:${PORT}"
echo "Health check: http://${HOST}:${PORT}/health"
echo "API docs: http://${HOST}:${PORT}/docs"

uvicorn api:app --host "$HOST" --port "$PORT"
