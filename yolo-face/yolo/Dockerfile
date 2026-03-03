FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel --no-cache-dir

COPY yolo/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app

# Shared library
COPY common/ /app/

# Model weights
COPY yolo/yolov12n-face.pt /app/models/yolov12n-face.pt

# Service source
COPY yolo/src/ /app/

ENV MODEL_PATH=/app/models/yolov12n-face.pt

CMD ["python3", "yolo_app.py"]
