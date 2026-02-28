FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip --no-cache-dir

COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app

# Shared library
COPY common/ /app/

# Service source (app.py + templates/)
COPY api/src/ /app/

EXPOSE 5000

# 4 gevent workers — MJPEG streaming needs async I/O, not more processes
CMD ["gunicorn", "-w", "4", "-k", "gevent", "-b", "0.0.0.0:5000", \
     "--timeout", "120", "--keep-alive", "5", "app:app"]
