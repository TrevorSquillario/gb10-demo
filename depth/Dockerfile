FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libsm6 libxext6 libxrender1 git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel --no-cache-dir

# Clone Depth Anything V2 and add to Python path
RUN git clone https://github.com/DepthAnything/Depth-Anything-V2.git /opt/depth_anything_v2
RUN pip install --no-cache-dir -r /opt/depth_anything_v2/requirements.txt
ENV PYTHONPATH="/opt/depth_anything_v2:${PYTHONPATH}"

COPY depth/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app

# Shared library
COPY common/ /app/

# Service source
COPY depth/src/ /app/

CMD ["python3", "depth.py"]
