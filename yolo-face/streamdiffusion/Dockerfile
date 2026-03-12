# Base image with CUDA‑13/PyTorch preinstalled
FROM nvcr.io/nvidia/pytorch:25.09-py3

ENV DEBIAN_FRONTEND=noninteractive 
ENV TORCH_CUDA_ARCH_LIST="12.1" 

RUN apt-get update && apt-get install -y --no-install-recommends \
        make \
        wget \
        tar \
        build-essential \
        libgl1-mesa-dev \
        curl \
        unzip \
        git \
        python3-dev \
        python3-pip \
        libglib2.0-0 \
        libgtk2.0-dev \
        pkg-config \
    && apt clean && rm -rf /var/lib/apt/lists/* 

# Set the specific paths for the Grace-Blackwell ARM64 targets
ENV CPATH=$CPATH:/usr/local/cuda/targets/aarch64-linux/include
ENV LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/targets/aarch64-linux/lib
#ENV CXXFLAGS="-Wno-error=deprecated-declarations"

# Tell PyCUDA exactly where the root is
ENV CUDA_ROOT=/usr/local/cuda

RUN pip install --upgrade pip --no-cache-dir

COPY streamdiffusion/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Shared library
COPY common/ /app/

# Service source 
COPY streamdiffusion/ /app/
WORKDIR /app

CMD ["python3", "streamdiffusion_img2img.py"]