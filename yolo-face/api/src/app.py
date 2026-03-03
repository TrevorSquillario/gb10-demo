"""
API gateway — serves all HTTP endpoints.

Zero inference code lives here.  Every route reads from Redis (frame bus,
hash keys, string keys) and formats/streams the response.

Routes
------
  GET  /                  Main HTML page
  GET  /video_feed        MJPEG stream from frames:yolo
  GET  /dwpose_feed       MJPEG stream from frames:dwpose
  GET  /sd_feed           MJPEG stream from frames:sd
  GET  /depth_feed        MJPEG stream from frames:depth
  GET  /attributes        JSON — DeepFace attributes per track_id
  GET  /stats             JSON — current/total people from latest detections
  GET  /crop/<track_id>   JPEG — latest face thumbnail
  GET  /vlm               JSON — latest VLM scene description
  GET  /gpu               JSON — per-GPU utilisation stats (via NVML)

Environment variables
---------------------
  REDIS_URL   Redis connection string.  Default: redis://redis:6379
  PORT        HTTP port.  Default: 5000
"""

import io
import json
import os
import subprocess

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, send_file

from logging_config import configure_logging, get_logger
from frame_bus import FrameBus, REDIS_URL

configure_logging()
log = get_logger(__name__)

# Optional NVML for GPU stats
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False
    log.warning("[API] pynvml not available — /gpu will use nvidia-smi fallback")

app = Flask(__name__, template_folder="templates")
bus = FrameBus()
r   = bus.redis()

PORT = int(os.environ.get("PORT", "5000"))

MJPEG_QUALITY = int(os.environ.get("MJPEG_QUALITY", "75"))


# ---------------------------------------------------------------------------
# MJPEG helper
# ---------------------------------------------------------------------------

def _mjpeg_stream(stream_name: str, seed_latest: bool = False):
    """Tail a Redis stream and yield MJPEG boundary chunks.

    When *seed_latest* is True the most-recent cached frame is emitted
    immediately before blocking for new entries, so slow-producing streams
    (e.g. SD with a long interval) don't leave the browser with a blank image.
    """
    if seed_latest:
        frame, _ = bus.latest(stream_name)
        if frame is not None:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY])
            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                )
    for frame, _ in bus.subscribe(stream_name):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY])
        if not ok:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        _mjpeg_stream("yolo"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/dwpose_feed")
def dwpose_feed():
    # Return 503 immediately if no dwpose frames are flowing
    frame, _ = bus.latest("dwpose")
    if frame is None:
        return "DWpose service not running (DWPOSE_ENABLED=1 required).", 503
    return Response(
        _mjpeg_stream("dwpose"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/sd_feed")
def sd_feed():
    # No availability guard — stream unconditionally and seed with the last
    # published frame so the browser gets something immediately.  The SD
    # service may take several minutes to warm up; the JS retry in the front
    # end handles the initial connection before any frame exists.
    return Response(
        _mjpeg_stream("sd", seed_latest=True),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/depth_feed")
def depth_feed():
    frame, _ = bus.latest("depth")
    if frame is None:
        return "Depth service not running (DEPTH_ENABLED=1 required).", 503
    return Response(
        _mjpeg_stream("depth"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/attributes")
def attributes():
    raw = r.hgetall("attrs")
    return jsonify({int(k): json.loads(v) for k, v in raw.items()})


@app.route("/stats")
def stats():
    raw_dets = r.hgetall("detections:latest")
    try:
        dets = {int(k): json.loads(v) for k, v in raw_dets.items()}
    except Exception:
        dets = {}
    total_raw  = r.get("stats:total")
    frames_raw = r.get("stats:frames")
    return jsonify({
        "current_people": len(dets),
        "tracked_ids":    sorted(dets.keys()),
        "total_tracked":  int(total_raw)  if total_raw  else 0,
        "frame_count":    int(frames_raw) if frames_raw else 0,
    })


@app.route("/crop/<int:track_id>")
def crop(track_id: int):
    data = r.get(f"crops:{track_id}")
    if data is None:
        return "", 404
    return send_file(io.BytesIO(data), mimetype="image/jpeg")


@app.route("/vlm")
def vlm():
    raw = r.get("vlm:latest")
    return jsonify({"description": raw.decode() if raw else None})


def _gpu_via_smi():
    """Query GPU stats via nvidia-smi CSV output. Returns list of dicts or None on failure."""
    try:
        fields = (
            "index,name,utilization.gpu,"
            "clocks.current.sm,power.draw.instant,temperature.gpu"
        )
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            timeout=3, stderr=subprocess.DEVNULL,
        ).decode()
        gpus = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            def _int(v):
                try: return int(v)
                except Exception: return None
            def _float(v):
                try: return round(float(v), 1)
                except Exception: return None
            gpus.append({
                "index":    _int(parts[0]),
                "name":     parts[1],
                "gpu_util": _int(parts[2]),
                "sm_clock": _int(parts[3]),
                "power":    _float(parts[4]),
                "temp":     _int(parts[5]),
            })
        return gpus or None
    except Exception:
        return None


@app.route("/gpu")
def gpu():
    """Return per-GPU utilisation stats — tries nvidia-smi first, then NVML."""
    # ── nvidia-smi path (most reliable across architectures) ─────────────────
    smi = _gpu_via_smi()
    if smi is not None:
        return jsonify({"gpus": smi, "source": "smi"})

    # ── NVML path (fallback) ──────────────────────────────────────────────────
    if not _NVML_OK:
        return jsonify({"gpus": []})
    try:
        count = pynvml.nvmlDeviceGetCount()
    except Exception as exc:
        return jsonify({"gpus": [], "error": str(exc)})

    result = []
    for i in range(count):
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
        except Exception:
            continue

        def _get(fn, *args, default=None):
            try:
                return fn(*args)
            except Exception:
                return default

        name_raw = _get(pynvml.nvmlDeviceGetName, h)
        if isinstance(name_raw, bytes):
            name_raw = name_raw.decode()
        util     = _get(pynvml.nvmlDeviceGetUtilizationRates, h)
        sm_clock = _get(pynvml.nvmlDeviceGetClockInfo, h, pynvml.NVML_CLOCK_SM)
        pwr_mw   = _get(pynvml.nvmlDeviceGetPowerUsage, h)
        temp     = _get(pynvml.nvmlDeviceGetTemperature, h, pynvml.NVML_TEMPERATURE_GPU)

        result.append({
            "index":    i,
            "name":     name_raw or f"GPU {i}",
            "gpu_util": util.gpu if util else None,
            "sm_clock": int(sm_clock) if sm_clock is not None else None,
            "power":    round(pwr_mw / 1000, 1) if pwr_mw is not None else None,
            "temp":     temp,
        })
    return jsonify({"gpus": result, "source": "nvml"})


@app.route("/healthz")
def healthz():
    try:
        r.ping()
        return jsonify({"status": "ok"})
    except Exception as exc:
        return jsonify({"status": "error", "detail": str(exc)}), 503


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info(f"[API] Starting on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
