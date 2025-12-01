# Base image with CUDA and TensorRT support
FROM nvcr.io/nvidia/tensorrt:23.08-py3

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir ultralytics==8.0.196 onnxruntime-gpu==1.16.3

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p outputs logs data

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python3", "src/main.py"]
