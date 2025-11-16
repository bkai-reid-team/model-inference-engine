#!/bin/bash

echo "============================================"
echo "Ray Serve Multi-Model Docker Setup"
echo "============================================"
echo

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    echo "Please install Docker from: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "ERROR: Docker is not running"
    echo "Please start Docker and try again"
    exit 1
fi

# Check for NVIDIA Docker runtime (optional)
if ! docker run --rm --gpus all nvidia/cuda:13.0.1-cudnn-runtime-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "WARNING: NVIDIA Docker runtime not detected"
    echo "GPU acceleration may not work properly"
    echo
fi

echo "Building Docker image..."
docker compose build

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to build Docker image"
    exit 1
fi

echo
echo "Starting Ray Serve container..."
docker compose up -d

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to start container"
    exit 1
fi

echo
echo "============================================"
echo "Setup completed successfully!"
echo "============================================"
echo
echo "Services available at:"
echo "- Ray Serve API: http://localhost:8000"
echo "- Ray Dashboard: http://localhost:8265"
echo
echo "API Endpoints:"
echo "- POST http://localhost:8000/generate"
echo "- POST http://localhost:8000/embed"
echo "- POST http://localhost:8000/image_classification"
echo "- POST http://localhost:8000/efficientnet_b0"
echo "- POST http://localhost:8000/efficientnet_b4"
echo
echo "Testing commands:"
echo "- Health check: python health_check.py"
echo "- API tests: python test_api.py"
echo "- Integration tests: python test_integration.py"
echo
echo "To view logs: docker compose logs -f"
echo "To stop: docker compose down"
echo
