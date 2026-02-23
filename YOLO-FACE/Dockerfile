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
    torch torchvision torchaudio xformers 

# Then install the remaining Python packages from requirements
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app

# Copy application files
COPY app.py /app/
COPY templates/ /app/templates/
COPY yolo*.pt /app/

# Download model weights (you'll need to provide the actual model)
# RUN wget <model_url> -O yolo26t.pt

EXPOSE 5000

CMD ["python3", "app.py"]
