FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev ffmpeg libgl1 libsm6 libxext6 libxrender1 \
    build-essential git ca-certificates && \
    rm -rf /var/lib/apt/lists/*
# Create venv (keeps python deps isolated inside container)
ENV VENV=/opt/venv
RUN python3 -m venv $VENV
ENV PATH="$VENV/bin:$PATH"

# Upgrade pip and packaging tools
RUN python3 -m pip install --upgrade pip setuptools wheel --no-cache-dir

# Install CUDA-enabled PyTorch for CUDA 13.0 (cu130)
RUN python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 \
    torch torchvision torchaudio 

# Then install the remaining Python packages from requirements
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

# flash_attn requires torch at build time — install it before the rest,
# with --no-build-isolation so the already-installed torch is visible.
#ENV FLASH_ATTN_CUDA_ARCHS=130 
#ENV FLASH_ATTENTION_FORCE_BUILD="TRUE" 
#ENV FLASH_ATTENTION_FORCE_CXX11_ABI="FALSE" 
#ENV FLASH_ATTENTION_SKIP_CUDA_BUILD="FALSE" 
#RUN python3 -m pip install --no-cache-dir --no-build-isolation flash_attn>=2.7.4

WORKDIR /app

# Copy application files
COPY app.py /app/
COPY templates/ /app/templates/
COPY yolo*.pt /app/

# Download model weights (you'll need to provide the actual model)
# RUN wget <model_url> -O yolo26t.pt

# ---------------------------------------------------------------------------
# StreamDiffusionV2 — clone repo and install under /app/YOLO-FACE
# ---------------------------------------------------------------------------
RUN mkdir -p /app/YOLO-FACE && \
    git clone --depth 1 https://github.com/chenfengxu714/StreamDiffusionV2 \
        /app/YOLO-FACE/StreamDiffusionV2

# Install repo dependencies (torch/torchvision already installed above;
# --no-deps for torch-pinned lines avoids version conflicts)
RUN cd /app/YOLO-FACE/StreamDiffusionV2 

# Set env var so the service can locate the repo at runtime
ENV STREAMDIFFUSIONV2_ROOT=/app/YOLO-FACE/StreamDiffusionV2

EXPOSE 5000

CMD ["python3", "app.py"]
