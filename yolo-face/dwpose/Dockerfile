FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libsm6 libxext6 libxrender1 git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel --no-cache-dir

COPY dwpose/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app

# Clone DWpose-TensorRT repo; the dwpose package and trt_utilities are imported
# directly from the clone via PYTHONPATH so they always match the installed TRT version.
RUN git clone --depth 1 https://github.com/yuvraj108c/Dwpose-Tensorrt /opt/Dwpose-Tensorrt

ENV PYTHONPATH="/opt/Dwpose-Tensorrt:${PYTHONPATH}"

# Shared library
COPY common/ /app/

# Service source
COPY dwpose/src/ /app/

CMD ["python3", "dwpose_app.py"]
