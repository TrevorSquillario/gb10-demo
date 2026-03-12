FROM nvcr.io/nvidia/pytorch:25.09-py3

# Target GB10 (SM 12.1) so compiled CUDA extensions match the host GPU
ENV TORCH_CUDA_ARCH_LIST="12.1"
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_root

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel --no-cache-dir

ARG SAGEATTN_REF=main
RUN pip install --no-build-isolation "git+https://github.com/thu-ml/SageAttention@${SAGEATTN_REF}" || true

COPY stablediffusion/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app

# Shared library
COPY common/ /app/

# Service source
COPY stablediffusion/src/ /app/

CMD ["python3", "stablediffusion_app.py"]
