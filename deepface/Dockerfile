FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip --no-cache-dir

COPY deepface/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app

# Shared library
COPY common/ /app/

# Service source
COPY deepface/src/ /app/

CMD ["python3", "deepface_app.py"]
