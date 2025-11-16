# Sử dụng NVIDIA CUDA base image để support GPU
FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies (Đã cập nhật để dùng Python 3.10)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first để leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
# Cần cài đặt Ray/Serve/PyTorch/vLLM/... tại đây
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code 
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && chown -R appuser:appuser /app

# Create cache directories for models (will be mounted from host)
# Cấu hình cache directories
RUN mkdir -p /home/appuser/.cache/huggingface \
    && mkdir -p /home/appuser/.cache/torch \
    && mkdir -p /home/appuser/.cache/transformers \
    && mkdir -p /home/appuser/.cache/sentence_transformers \
    && chown -R appuser:appuser /home/appuser/.cache

USER appuser

# Expose port
EXPOSE 8000 8265 6379

# Health check - Tăng start-period để cho phép LLM download và tải vào VRAM
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Default command: Đặt lệnh mặc định là một lệnh đơn giản, không triển khai Serve.
# Việc triển khai sẽ do 'command' trong docker-compose.yml kiểm soát.
# Sử dụng `CMD` đơn giản để dễ debug hơn.
CMD ["/usr/bin/python3.10", "-c", "import time; while True: time.sleep(1)"]
