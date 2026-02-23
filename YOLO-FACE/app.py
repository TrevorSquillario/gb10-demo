#!/usr/bin/env python3
"""
Flask web server for YOLO face detection/tracking video stream.
Streams processed video frames to a web browser with persistent face tracking.
Supports webcam, video files, and RTSP/HLS streams via ffmpeg.
Uses BoT-SORT tracker for high-accuracy face tracking with Re-ID support.
"""

import time
import os
import queue
import subprocess
import signal
import atexit
from threading import Lock
from collections import defaultdict

import cv2
import numpy as np
import torch
from flask import Flask, Response, render_template, send_file
import io

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator, colors

from types import SimpleNamespace

from services.deepface import DeepFaceService
from services.yolo import generate_frames as _service_generate_frames
from services.dwpose import generate_dwpose as _service_generate_dwpose
from services.depth_anything import generate_depth as _service_generate_depth
from services.stablediffusion import StableDiffusionService, generate_stablediffusion as _service_generate_stablediffusion
from dwpose import DWposeDetector
from depth_anything.util.transform import transform as depth_transform

app = Flask(__name__)

# Configuration from environment or defaults
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/yolo26n.pt")  # Face detection model
DEPTH_ENGINE_PATH = os.environ.get("DEPTH_ENGINE_PATH", "/app/models/tensorrt/depth-anything/v2_depth_anything_vitl-fp16.engine")  # TensorRT depth engine
DEPTH_GRAYSCALE = os.environ.get("DEPTH_GRAYSCALE", "0") == "1"  # Output grayscale instead of colormap
DETECT_STREAM_URL = os.environ.get("DETECT_STREAM_URL", None)  # HLS/HTTP stream URL (requires ffmpeg)
RTSP_LOCAL_URL = "rtsp://localhost:8554/test"  # Local RTSP endpoint for ffmpeg output
video_source = os.environ.get("VIDEO_SOURCE", "0")  # webcam index, file path, or rtsp:// URL

# Determine video source type
video_source_type = None
original_source = video_source

# Try to parse video_source as int for webcam device index
try:
    video_source = int(video_source)
    video_source_type = "webcam"
    LOGGER.info(f"Video source: USB Webcam (device {video_source})")
except ValueError:
    # Check if it's an RTSP stream
    if video_source.startswith("rtsp://"):
        video_source_type = "rtsp"
        LOGGER.info(f"Video source: RTSP stream ({video_source})")
    elif video_source.startswith("http://") or video_source.startswith("https://"):
        # HTTP/HTTPS URLs might be direct streams or need ffmpeg
        video_source_type = "http"
        LOGGER.info(f"Video source: HTTP stream ({video_source})")
    else:
        # Assume it's a local file path
        video_source_type = "file"
        LOGGER.info(f"Video source: Local file ({video_source})")
        if not os.path.exists(video_source):
            LOGGER.warning(f"File not found: {video_source}")

# CUDA settings
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda:0" if USE_CUDA else "cpu"

# Detection settings
show_fps = True
show_conf = False
show_stats = True  # Show people count and tracking stats
conf = 0.3
iou = 0.3
max_det = 50  # Increased for face detection
SKIP_FRAMES = int(os.environ.get("SKIP_FRAMES", "1"))  # Process every Nth frame

# Use BoT-SORT tracker for better face tracking with Re-ID
tracker = "botsort.yaml"
track_args = {
    "persist": True,  # Remember IDs across frames
    "verbose": False,
}

# If DETECT_STREAM_URL is set (HLS/HTTP that needs conversion), use ffmpeg to convert to local RTSP
if DETECT_STREAM_URL:
    original_source = video_source
    video_source = RTSP_LOCAL_URL
    video_source_type = "ffmpeg_rtsp"
    LOGGER.info(f"Using HLS/HTTP stream via ffmpeg: {DETECT_STREAM_URL} -> {RTSP_LOCAL_URL}")

# Global state
frame_lock = Lock()
ffmpeg_proc = None

# Latest JPEG crop per track_id for the UI panel
person_crops: dict[int, bytes] = {}
crops_lock = Lock()

# Person tracking statistics
tracked_people = set()  # Set of all unique person IDs seen
person_last_seen = defaultdict(int)  # Track when each person was last seen
current_people = set()  # People in current frame

# Initialize model
LOGGER.info("🚀 Initializing YOLO face detection model...")
LOGGER.info(f"Torch CUDA Version: {torch.version.cuda}")
model = YOLO(MODEL_PATH)

if USE_CUDA:
    LOGGER.info(f"Using CUDA on {DEVICE}")
    model.to(DEVICE)
else:
    LOGGER.info("Using CPU")

classes = model.names

# Initialize DWpose detector for skeleton/pose overlay
LOGGER.info("🦴 Initializing DWpose detector...")
dwpose_detector = DWposeDetector()


class DepthAnythingEngine:
    """TensorRT-backed Depth Anything inference engine for per-frame depth estimation.

    Uses torch for all memory management so pycuda is not required.
    """

    def __init__(self, engine_path: str):
        import tensorrt as trt

        self._trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        # Keep runtime alive — deserialised engine holds a reference to it
        self._runtime = trt.Runtime(logger)
        with open(engine_path, 'rb') as f:
            self._engine = self._runtime.deserialize_cuda_engine(f.read())

        self.context = self._engine.create_execution_context()
        input_shape = self.context.get_tensor_shape('input')
        self.output_shape = self.context.get_tensor_shape('output')

        input_size = int(trt.volume(input_shape))
        output_size = int(trt.volume(self.output_shape))

        # Allocate device tensors via torch — no pycuda required
        self._d_input = torch.zeros(input_size, dtype=torch.float32, device='cuda')
        self._d_output = torch.zeros(output_size, dtype=torch.float32, device='cuda')
        self._stream = torch.cuda.Stream()

    def __call__(self, frame_bgr: np.ndarray, grayscale: bool = False) -> np.ndarray:
        """Run depth estimation on a BGR frame and return a colourised BGR depth map."""
        orig_h, orig_w = frame_bgr.shape[:2]

        # Pre-process: same pipeline as load_image() but from an ndarray
        image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) / 255.0
        image = depth_transform({"image": image})["image"]  # C, H, W  float32
        image = image[None]  # B, C, H, W
        input_tensor = torch.from_numpy(image.ravel()).float()

        with torch.cuda.stream(self._stream):
            self._d_input.copy_(input_tensor)
            self.context.set_tensor_address('input', self._d_input.data_ptr())
            self.context.set_tensor_address('output', self._d_output.data_ptr())
            self.context.execute_async_v3(stream_handle=self._stream.cuda_stream)
        self._stream.synchronize()

        # Post-process
        depth = self._d_output.cpu().numpy()
        depth = np.reshape(depth, self.output_shape[2:])
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth = cv2.resize(depth, (orig_w, orig_h))

        if grayscale:
            return cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
        return cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)


# Initialise depth engine only when the engine file is present
depth_engine: DepthAnythingEngine | None = None
if os.path.exists(DEPTH_ENGINE_PATH):
    LOGGER.info(f"📐 Initializing Depth Anything TRT engine from {DEPTH_ENGINE_PATH}...")
    depth_engine = DepthAnythingEngine(DEPTH_ENGINE_PATH)
else:
    LOGGER.warning(f"Depth Anything engine not found at {DEPTH_ENGINE_PATH}; /depth_feed will be unavailable.")

# Facial attribute analysis — runs in background every ANALYSIS_INTERVAL seconds
ANALYSIS_INTERVAL = float(os.environ.get("ANALYSIS_INTERVAL", "15"))
deepface_service = DeepFaceService(interval=ANALYSIS_INTERVAL)
deepface_service.start()

# Stable Diffusion XL Turbo + ControlNet — optional, gated by SD_ENABLED env var
SD_ENABLED = os.environ.get("SD_ENABLED", "0") == "1"
sd_service: StableDiffusionService | None = None
if SD_ENABLED:
    LOGGER.info("🎨 Initializing Stable Diffusion XL Turbo + ControlNet pipeline...")
    sd_service = StableDiffusionService(
        controlnet_model=os.environ.get("CONTROLNET_MODEL", "xinsir/controlnet-openpose-sdxl-1.0"),
        sdxl_model=os.environ.get("SDXL_MODEL", "stabilityai/sdxl-turbo"),
    )
else:
    LOGGER.info("SD pipeline disabled (set SD_ENABLED=1 to enable)")

# Shared queue that DWpose worker pushes into and the SD worker pulls from
pose_queue: queue.Queue = queue.Queue(maxsize=4)


def start_ffmpeg():
    """Start ffmpeg to pull HLS/HTTP stream and publish to local RTSP."""
    global ffmpeg_proc
    if ffmpeg_proc is not None and ffmpeg_proc.poll() is None:
        return
    cmd = [
        "ffmpeg",
        "-re",
        "-i", DETECT_STREAM_URL,
        "-c", "copy",
        "-f", "rtsp",
        RTSP_LOCAL_URL,
    ]
    LOGGER.info(f"Starting ffmpeg: {' '.join(cmd)}")
    ffmpeg_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def stop_ffmpeg():
    """Stop the ffmpeg subprocess."""
    global ffmpeg_proc
    if ffmpeg_proc is None:
        return
    try:
        if ffmpeg_proc.poll() is None:
            LOGGER.info(f"Terminating ffmpeg (pid={ffmpeg_proc.pid})")
            ffmpeg_proc.terminate()
            ffmpeg_proc.wait(timeout=5)
    except Exception:
        try:
            ffmpeg_proc.kill()
        except Exception:
            pass
    ffmpeg_proc = None


# Ensure ffmpeg and deepface service are stopped on exit
def _cleanup():
    stop_ffmpeg()
    deepface_service.stop()


atexit.register(_cleanup)


def _handle_sigterm(signum, frame):
    stop_ffmpeg()
    raise SystemExit(0)


signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)


# ---------------------------------------------------------------------------
# Generator context objects — bundle config + shared mutable state so that
# the service-level generators in services/ can operate without importing app.
# ---------------------------------------------------------------------------
yolo_ctx = SimpleNamespace(
    # video source
    video_source=video_source,
    video_source_type=video_source_type,
    # model
    model=model,
    classes=classes,
    conf=conf,
    iou=iou,
    max_det=max_det,
    DEVICE=DEVICE,
    USE_CUDA=USE_CUDA,
    tracker=tracker,
    track_args=track_args,
    SKIP_FRAMES=SKIP_FRAMES,
    # stream / analysis
    DETECT_STREAM_URL=DETECT_STREAM_URL,
    ANALYSIS_INTERVAL=ANALYSIS_INTERVAL,
    deepface_service=deepface_service,
    # display flags
    show_fps=show_fps,
    show_conf=show_conf,
    show_stats=show_stats,
    # mutable scalar state (owned by ctx; routes read via yolo_ctx)
    frame_lock=frame_lock,
    latest_frame=None,
    frame_counter=0,
    last_deepface_time=0.0,
    # mutable container state (same objects as module-level globals below)
    tracked_people=tracked_people,
    person_last_seen=person_last_seen,
    current_people=current_people,
    person_crops=person_crops,
    crops_lock=crops_lock,
    # ffmpeg helpers
    start_ffmpeg=start_ffmpeg,
    stop_ffmpeg=stop_ffmpeg,
)

dwpose_ctx = SimpleNamespace(
    video_source=video_source,
    video_source_type=video_source_type,
    DETECT_STREAM_URL=DETECT_STREAM_URL,
    dwpose_detector=dwpose_detector,
    show_fps=show_fps,
    start_ffmpeg=start_ffmpeg,
    pose_queue=pose_queue,   # shared with sd_ctx
)

depth_ctx = SimpleNamespace(
    video_source=video_source,
    video_source_type=video_source_type,
    DETECT_STREAM_URL=DETECT_STREAM_URL,
    depth_engine=depth_engine,
    depth_grayscale=DEPTH_GRAYSCALE,
    show_fps=show_fps,
    start_ffmpeg=start_ffmpeg,
)

sd_ctx = SimpleNamespace(
    pose_queue=pose_queue,
    # Provide the DWpose context so generate_stablediffusion can start its own
    # capture thread.  This makes /sd_feed independent of /dwpose_feed.
    dwpose_ctx=dwpose_ctx,
    sd_service=sd_service,
    sd_prompt=os.environ.get("SD_PROMPT", "a person, cinematic lighting, photorealistic, 8k"),
    sd_negative_prompt=os.environ.get("SD_NEGATIVE_PROMPT", ""),
    sd_steps=int(os.environ.get("SD_STEPS", "4")),
    sd_guidance_scale=float(os.environ.get("SD_GUIDANCE_SCALE", "0.0")),
    sd_controlnet_scale=float(os.environ.get("SD_CONTROLNET_SCALE", "0.8")),
    sd_width=int(os.environ.get("SD_WIDTH", "1024")),
    sd_height=int(os.environ.get("SD_HEIGHT", "1024")),
    sd_show_side_by_side=os.environ.get("SD_SIDE_BY_SIDE", "0") == "1",
    show_fps=show_fps,
)


def get_center(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int]:
    """Calculate the center point of a bounding box."""
    return (x1 + x2) // 2, (y1 + y2) // 2


def generate_frames():
    """Yield MJPEG chunks with YOLO face detections — delegates to services/yolo.py."""
    return _service_generate_frames(yolo_ctx)


def generate_dwpose():
    """Yield MJPEG chunks with DWpose skeleton overlays — delegates to services/dwpose.py."""
    return _service_generate_dwpose(dwpose_ctx)


def generate_depth():
    """Yield MJPEG chunks with Depth Anything depth-map overlays — delegates to services/depth_anything.py."""
    return _service_generate_depth(depth_ctx)


def generate_stablediffusion():
    """Yield MJPEG chunks of SDXL Turbo images conditioned on DWpose — delegates to services/stablediffusion.py."""
    return _service_generate_stablediffusion(sd_ctx)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/dwpose_feed')
def dwpose_feed():
    """DWpose skeleton overlay streaming route."""
    return Response(generate_dwpose(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/depth_feed')
def depth_feed():
    """Depth Anything estimation streaming route."""
    if depth_engine is None:
        return "Depth Anything engine not loaded. Set DEPTH_ENGINE_PATH.", 503
    return Response(generate_depth(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/sd_feed')
def sd_feed():
    """SDXL Turbo + ControlNet streaming route (requires DWpose feed to be running)."""
    if sd_service is None:
        return "Stable Diffusion pipeline not loaded. Set SD_ENABLED=1.", 503
    return Response(generate_stablediffusion(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/attributes')
def attributes():
    """Return latest DeepFace facial attributes for all tracked people."""
    return deepface_service.get_all_attributes()


@app.route('/crop/<int:track_id>')
def crop(track_id: int):
    """Serve the latest JPEG crop for a tracked person."""
    with yolo_ctx.crops_lock:
        data = yolo_ctx.person_crops.get(track_id)
    if data is None:
        return '', 404
    return send_file(io.BytesIO(data), mimetype='image/jpeg')


@app.route('/stats')
def stats():
    """Return tracking statistics as JSON."""
    return {
        'current_people': len(yolo_ctx.current_people),
        'total_tracked': len(yolo_ctx.tracked_people),
        'tracked_ids': sorted(list(yolo_ctx.tracked_people)),
        'frame_count': yolo_ctx.frame_counter
    }


if __name__ == '__main__':
    LOGGER.info("🚀 Starting Face Tracking Flask server...")
    LOGGER.info("📊 Using BoT-SORT tracker with persist=True for Re-ID")
    LOGGER.info("Access the stream at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
