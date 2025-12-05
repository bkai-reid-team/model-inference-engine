# Use NVIDIA CUDA base image to support GPU
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies (Python 3.11)
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code 
COPY src/ ./src
COPY serve_app.py .
COPY serve_config.yaml .
COPY weights_manager.py .

# Check file exists in /app
RUN ls -al

# Create cache directories for models (root already has write permission)
RUN mkdir -p /root/.cache/huggingface \
    && mkdir -p /root/.cache/torch \
    && mkdir -p /root/.cache/transformers \
    && mkdir -p /root/.cache/sentence_transformers

# Expose port
EXPOSE 8000 8265 6379
