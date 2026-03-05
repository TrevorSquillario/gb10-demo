FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libsm6 libxext6 libxrender1 git \
    # EGL + GLES — required for VTK/PyVista headless GPU rendering
    libegl1 libgles2 libegl-mesa0 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel --no-cache-dir

# Clone Depth Anything V2 and add to Python path.
# metric_depth must come FIRST so that `from depth_anything_v2.dpt import
# DepthAnythingV2` resolves to the metric variant (metric_depth/depth_anything_v2/dpt.py)
# rather than the non-metric one (depth_anything_v2/dpt.py).  Using the wrong
# class with metric weights causes the model to silently output all-zeros.
RUN git clone https://github.com/DepthAnything/Depth-Anything-V2.git /opt/depth_anything_v2
RUN pip install --no-cache-dir -r /opt/depth_anything_v2/requirements.txt

ENV PYTHONPATH="/opt/depth_anything_v2:${PYTHONPATH}"
#ENV PYTHONPATH="/opt/depth_anything_v2/metric_depth:/opt/depth_anything_v2:${PYTHONPATH}"

COPY depth/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app

# Shared library
COPY common/ /app/

# Service source
COPY depth/src/ /app/

CMD ["python3", "depth.py"]
#CMD ["python3", "depth_metric.py"]
