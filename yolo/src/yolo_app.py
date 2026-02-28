"""
YOLO microservice — face detection + BoT-SORT tracking.

Subscribes to ``frames:raw`` on the Redis frame bus, runs YOLO inference on
every frame, then:

  * Publishes annotated frames to ``frames:yolo``.
  * Writes per-track detection metadata to the Redis Hash ``detections:latest``
    (TTL 5 s) so downstream services (DeepFace, API) can read without decoding
    the full frame.
  * Writes JPEG face crops to Redis keys ``crops:{track_id}`` (TTL 60 s)
    so the API can serve them directly.

Environment variables
---------------------
  MODEL_PATH          Path to YOLO weights file.
                      Default: /app/models/yolov12n-face.pt
  CONF                Detection confidence threshold.  Default: 0.3
  IOU                 NMS IoU threshold.  Default: 0.3
  MAX_DET             Max detections per frame.  Default: 50
  SKIP_FRAMES         Run inference every N frames.  Default: 1
  SHOW_FPS            Overlay FPS counter (1/0).  Default: 1
  SHOW_CONF           Overlay confidence scores (1/0).  Default: 0
  CROP_ENCODE_INTERVAL  Re-encode face thumbnail every N frames.  Default: 5
  MJPEG_QUALITY       JPEG quality 1-100.  Default: 75
  REDIS_URL           Redis connection string.
"""

import json
import os
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator, colors

from logging_config import configure_logging, get_logger
from frame_bus import FrameBus, REDIS_URL

configure_logging()
log = get_logger(__name__)

# ── Thread-pool limits ────────────────────────────────────────────────────────
_CPU_THREADS = int(os.environ.get("APP_CPU_THREADS", "4"))
os.environ.setdefault("OMP_NUM_THREADS", str(_CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_CPU_THREADS))
cv2.setNumThreads(_CPU_THREADS)
torch.set_num_threads(_CPU_THREADS)
torch.set_num_interop_threads(_CPU_THREADS)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH         = os.environ.get("MODEL_PATH", "/app/models/yolov12n-face.pt")
CONF               = float(os.environ.get("CONF", "0.3"))
IOU                = float(os.environ.get("IOU", "0.3"))
MAX_DET            = int(os.environ.get("MAX_DET", "50"))
SKIP_FRAMES        = int(os.environ.get("SKIP_FRAMES", "1"))
SHOW_FPS           = os.environ.get("SHOW_FPS", "1") == "1"
SHOW_CONF          = os.environ.get("SHOW_CONF", "0") == "1"
CROP_ENCODE_INTERVAL = int(os.environ.get("CROP_ENCODE_INTERVAL", "5"))
MJPEG_QUALITY      = int(os.environ.get("MJPEG_QUALITY", "75"))

DETECTION_KEY      = "detections:latest"
DETECTION_TTL      = 5      # seconds
CROP_TTL           = 60     # seconds


def _fps_overlay(im: np.ndarray, fps: int) -> None:
    text  = f"FPS: {fps}"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 1.0, 2
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    h, w = im.shape[:2]
    fx = w - tw - 15
    cv2.rectangle(im, (fx - 5, 25 - th - 5), (fx + tw + 5, 25 + bl), (255, 255, 255), -1)
    cv2.putText(im, text, (fx, 25), font, scale, (104, 31, 17), thick, cv2.LINE_AA)



def main() -> None:
    USE_CUDA = torch.cuda.is_available()
    DEVICE   = "cuda:0" if USE_CUDA else "cpu"

    log.info(f"[YOLO] Loading model from {MODEL_PATH} on {DEVICE}")
    model  = YOLO(MODEL_PATH)
    model.to(DEVICE)

    bus = FrameBus()
    r   = bus.redis()

    log.info(f"[YOLO] Subscribing to frames:raw  (redis={REDIS_URL})")

    tracked_people: set[int] = set()
    current_people: set[int] = set()
    frame_counter  = 0
    crop_encode_frame: dict[int, int] = {}

    fps_counter, fps_timer, fps_display = 0, time.time(), 0

    for raw_frame, _ in bus.subscribe("raw"):
        frame_counter += 1
        current_people = set()
        snapshot_detections: list[tuple] = []

        # ── Inference ───────────────────────────────────────────────────────
        results = model.track(
            raw_frame,
            conf=CONF,
            iou=IOU,
            max_det=MAX_DET,
            device=DEVICE,
            half=USE_CUDA,
            tracker="botsort.yaml",
            vid_stride=SKIP_FRAMES,
            persist=True,
            verbose=False,
        )

        annotator   = Annotator(raw_frame.copy())
        detections  = results[0].boxes.data if results[0].boxes is not None else []
        det_payload: dict[str, str] = {}

        for det in detections:
            track = det.tolist()
            if len(track) < 6:
                continue

            x1, y1, x2, y2 = map(int, track[:4])
            if len(track) == 6:
                conf_score, class_id, track_id = float(track[4]), int(track[5]), -1
            else:
                track_id   = int(track[4])
                conf_score = float(track[5])
                class_id   = int(track[6])

            if track_id >= 0:
                tracked_people.add(track_id)
                current_people.add(track_id)
                snapshot_detections.append((track_id, x1, y1, x2, y2))
                det_payload[str(track_id)] = json.dumps({
                    "bbox":  [x1, y1, x2, y2],
                    "score": conf_score,
                    "class": class_id,
                })

                # Throttled face-crop thumbnail
                h_im, w_im = raw_frame.shape[:2]
                cx1, cy1 = max(0, x1), max(0, y1)
                cx2, cy2 = min(w_im, x2), min(h_im, y2)
                last_enc  = crop_encode_frame.get(track_id, -CROP_ENCODE_INTERVAL)
                if (
                    (cx2 - cx1) > 10
                    and (cy2 - cy1) > 10
                    and (frame_counter - last_enc) >= CROP_ENCODE_INTERVAL
                ):
                    ok, buf = cv2.imencode(
                        ".jpg", raw_frame[cy1:cy2, cx1:cx2],
                        [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY],
                    )
                    if ok:
                        crop_encode_frame[track_id] = frame_counter
                        r.setex(f"crops:{track_id}", CROP_TTL, buf.tobytes())

            color      = colors(track_id if track_id >= 0 else 0, True)
            class_name = model.names.get(class_id, "face")
            label      = f"Person {track_id}" if track_id >= 0 else class_name
            if SHOW_CONF:
                label += f" {conf_score:.2f}"
            annotator.box_label([x1, y1, x2, y2], label=label, color=color)

        # Publish detections to Redis hash
        if det_payload:
            pipe = r.pipeline()
            pipe.delete(DETECTION_KEY)
            pipe.hset(DETECTION_KEY, mapping=det_payload)
            pipe.expire(DETECTION_KEY, DETECTION_TTL)
            pipe.execute()

        # ── Overlays ────────────────────────────────────────────────────────
        annotated = annotator.result()

        if SHOW_FPS:
            fps_counter += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_timer   = now
            _fps_overlay(annotated, fps_display)

        # Write running totals so the API /stats route can read them
        r.set("stats:total", len(tracked_people))
        r.set("stats:frames", frame_counter)

        # ── Publish annotated frame ──────────────────────────────────────────
        bus.publish("yolo", annotated, meta={
            "current":  len(current_people),
            "total":    len(tracked_people),
            "frame":    frame_counter,
        })


if __name__ == "__main__":
    main()
