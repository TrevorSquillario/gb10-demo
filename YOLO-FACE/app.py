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
import logging

# ── Thread-pool limits ────────────────────────────────────────────────────────
# OpenCV, PyTorch, and ONNX Runtime each spin up one thread per CPU core by
# default.  On a 20-core host that's ~60 idle CPU threads before any app code
# runs.  Since all heavy inference is on the GPU, 2-4 threads per library is
# more than enough and drops the total thread count dramatically.
_CPU_THREADS = int(os.environ.get("APP_CPU_THREADS", "4"))
os.environ.setdefault("OMP_NUM_THREADS", str(_CPU_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_CPU_THREADS))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(_CPU_THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_CPU_THREADS))
# TensorFlow (used lazily by DeepFace) respects these before it initialises
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(_CPU_THREADS))
os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(_CPU_THREADS))
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np
import torch
from flask import Flask, Response, render_template, send_file

# Apply thread limits to libraries that ignore env vars at import time
cv2.setNumThreads(_CPU_THREADS)
torch.set_num_threads(_CPU_THREADS)
torch.set_num_interop_threads(_CPU_THREADS)
import io

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator, colors

from types import SimpleNamespace

from services.deepface import DeepFaceService
from services.yolo import generate_frames as _service_generate_frames
from services.dwpose import generate_dwpose as _service_generate_dwpose
from services.depth_anything import generate_depth as _service_generate_depth
from services.streamdiffusion_v2 import StreamDiffusionV2Service, run_streamdiffusionv2 as _service_run_sd2
from services.vlm import VlmService
from dwpose import DWposeDetector
from depth_anything.util.transform import transform as depth_transform

LOGGER.setLevel(logging.DEBUG)

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

# DWpose skeleton overlay — optional, gated by DWPOSE_ENABLED env var
DWPOSE_ENABLED = os.environ.get("DWPOSE_ENABLED", "0") == "1"
dwpose_detector: DWposeDetector | None = None
if DWPOSE_ENABLED:
    LOGGER.info("🦴 Initializing DWpose detector...")
    dwpose_detector = DWposeDetector()
else:
    LOGGER.info("DWpose detector disabled (set DWPOSE_ENABLED=1 to enable)")


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


# Depth Anything estimation — optional, gated by DEPTH_ENABLED env var
DEPTH_ENABLED = os.environ.get("DEPTH_ENABLED", "0") == "1"
depth_engine: DepthAnythingEngine | None = None
if DEPTH_ENABLED:
    if os.path.exists(DEPTH_ENGINE_PATH):
        LOGGER.info(f"📐 Initializing Depth Anything TRT engine from {DEPTH_ENGINE_PATH}...")
        depth_engine = DepthAnythingEngine(DEPTH_ENGINE_PATH)
    else:
        LOGGER.warning(f"Depth Anything engine not found at {DEPTH_ENGINE_PATH}; /depth_feed will be unavailable.")
else:
    LOGGER.info("Depth Anything disabled (set DEPTH_ENABLED=1 to enable)")

# Facial attribute analysis — runs in background every ANALYSIS_INTERVAL seconds
ANALYSIS_INTERVAL = float(os.environ.get("ANALYSIS_INTERVAL", "15"))
deepface_service = DeepFaceService(interval=ANALYSIS_INTERVAL)
deepface_service.start()

# VLM scene analysis — optional, gated by VLM_ENABLED env var
VLM_ENABLED = os.environ.get("VLM_ENABLED", "0") == "1"
vlm_service: VlmService | None = None

# StreamDiffusionV2 — optional, gated by SD2_ENABLED env var
SD2_ENABLED = os.environ.get("SD2_ENABLED", "0") == "1"
sd2_service: StreamDiffusionV2Service | None = None
if SD2_ENABLED:
    LOGGER.info("🎬 Initializing StreamDiffusionV2 pipeline...")
    sd2_service = StreamDiffusionV2Service(
        config_path=os.environ.get(
            "SD2_CONFIG_PATH",
            "/app/YOLO-FACE/StreamDiffusionV2/configs/wan_causal_dmd_v2v.yaml",
        ),
        checkpoint_folder=os.environ.get(
            "SD2_CHECKPOINT_FOLDER",
            "/app/YOLO-FACE/StreamDiffusionV2/ckpts/wan_causal_dmd_v2v",
        ),
    )
else:
    LOGGER.info("StreamDiffusionV2 pipeline disabled (set SD2_ENABLED=1 to enable)")

# VLM context and service startup
vlm_ctx = SimpleNamespace(
    video_source=video_source,
    video_source_type=video_source_type,
    DETECT_STREAM_URL=DETECT_STREAM_URL,
    vlm_prompt=os.environ.get("VLM_PROMPT", "Describe what you see in this image."),
    vlm_interval=float(os.environ.get("VLM_INTERVAL", "30")),
    vlm_frame_stride=int(os.environ.get("VLM_FRAME_STRIDE", "1")),
    vlm_max_new_tokens=int(os.environ.get("VLM_MAX_NEW_TOKENS", "256")),
)

if VLM_ENABLED:
    LOGGER.info("🤖 Initializing VLM scene analysis service...")
    vlm_service = VlmService(vlm_ctx)
    # vlm_service.start() is deferred until yolo_ctx (frame source) is ready
else:
    LOGGER.info("VLM service disabled (set VLM_ENABLED=1 to enable)")
    # Ensure attrs initialised even when disabled
    vlm_ctx.vlm_latest_result = None
    vlm_ctx.vlm_last_call_time = 0.0

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
    if vlm_service is not None:
        vlm_service.stop()


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
    latest_raw_frame=None,
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

# Wire VLM to read raw frames from the YOLO capture service
vlm_ctx.frame_source_ctx = yolo_ctx
if VLM_ENABLED and vlm_service is not None:
    vlm_service.start()

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

sd2_ctx = SimpleNamespace(
    sd2_service=sd2_service,
    sd2_input_video=os.environ.get("SD2_INPUT_VIDEO", None),
    sd2_prompt_file=os.environ.get("SD2_PROMPT_FILE", "/app/YOLO-FACE/StreamDiffusionV2/examples/original.mp4"),
    sd2_output_folder=os.environ.get("SD2_OUTPUT_FOLDER", "/tmp/sd2_output"),
    sd2_config_path=os.environ.get(
        "SD2_CONFIG_PATH",
        "/app/YOLO-FACE/StreamDiffusionV2/configs/wan_causal_dmd_v2v.yaml",
    ),
    sd2_checkpoint_folder=os.environ.get(
        "SD2_CHECKPOINT_FOLDER",
        "/app/YOLO-FACE/StreamDiffusionV2/ckpts/wan_causal_dmd_v2v",
    ),
    sd2_height=int(os.environ.get("SD2_HEIGHT", "480")),
    sd2_width=int(os.environ.get("SD2_WIDTH", "832")),
    sd2_fps=int(os.environ.get("SD2_FPS", "16")),
    sd2_step=int(os.environ.get("SD2_STEP", "2")),
    sd2_noise_scale=float(os.environ.get("SD2_NOISE_SCALE", "0.700")),
    sd2_num_frames=int(os.environ.get("SD2_NUM_FRAMES", "81")),
    sd2_target_fps=int(os.environ.get("SD2_TARGET_FPS")) if os.environ.get("SD2_TARGET_FPS") else None,
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


def generate_streamdiffusionv2():
    """Run SD2 inference then stream the output video as MJPEG frames."""
    result = _service_run_sd2(sd2_ctx)
    video_path = result.get("output_video", "")
    if not video_path or not os.path.exists(video_path):
        LOGGER.error(f"[SD2] Output video not found: {video_path}")
        return
    cap = cv2.VideoCapture(video_path)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Loop back to the start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            ret2, buf = cv2.imencode(".jpg", frame)
            if not ret2:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
    except GeneratorExit:
        pass
    finally:
        cap.release()

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
    if dwpose_detector is None:
        return "DWpose detector not loaded. Set DWPOSE_ENABLED=1.", 503
    return Response(generate_dwpose(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/depth_feed')
def depth_feed():
    """Depth Anything estimation streaming route."""
    if depth_engine is None:
        return "Depth Anything engine not loaded. Set DEPTH_ENGINE_PATH.", 503
    return Response(generate_depth(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/sd2_feed')
def sd2_feed():
    """StreamDiffusionV2 inference route — runs offline inference and streams the output video."""
    if sd2_service is None:
        return "StreamDiffusionV2 pipeline not loaded. Set SD2_ENABLED=1.", 503
    return Response(generate_streamdiffusionv2(),
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


@app.route('/vlm')
def vlm():
    """Return the latest VLM scene description as JSON."""
    return {'description': vlm_ctx.vlm_latest_result}


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
