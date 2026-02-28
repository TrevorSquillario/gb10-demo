"""
DeepFace microservice — facial attribute analysis.

Polls ``detections:latest`` for active track IDs, fetches the pre-encoded
JPEG crop written by the YOLO service at ``crops:{track_id}``, and writes
age/gender/emotion/race attributes to the Redis Hash ``attrs``.

No raw frame subscription is needed — YOLO already stores face crops so
this service only performs analysis, never re-crops.

Runs in its own process so TensorFlow's memory allocator never competes
with PyTorch CUDA contexts in the YOLO / DWpose / Depth containers.

Environment variables
---------------------
  ANALYSIS_INTERVAL     Minimum seconds between re-analysing the same face.
                        Default: 30
  DEEPFACE_CONFIDENCE   Minimum extract_faces() confidence to proceed.
                        Default: 0.9
  DEEPFACE_DETECTOR     Backend for extract_faces().  Default: retinaface
  ATTRS_TTL             Redis TTL for the attrs hash in seconds.  Default: 30
  POLL_INTERVAL         Seconds between polls when no new faces are ready.
                        Default: 1
  REDIS_URL             Redis connection string.
"""

import json
import os
import time

import cv2
import numpy as np

from logging_config import configure_logging, get_logger
from frame_bus import FrameBus

configure_logging()
log = get_logger(__name__)

ANALYSIS_INTERVAL    = float(os.environ.get("ANALYSIS_INTERVAL", "30"))
CONFIDENCE_THRESHOLD = float(os.environ.get("DEEPFACE_CONFIDENCE", "0.9"))
DETECTOR_BACKEND     = os.environ.get("DEEPFACE_DETECTOR", "retinaface")
ATTRS_TTL            = int(os.environ.get("ATTRS_TTL", "30"))
POLL_INTERVAL        = float(os.environ.get("POLL_INTERVAL", "15"))
MIN_CROP_PX          = 30

DETECTION_KEY = "detections:latest"
ATTRS_KEY     = "attrs"


def main() -> None:
    try:
        from deepface import DeepFace
    except ImportError:
        log.error("[DeepFace] deepface is not installed — exiting")
        return

    log.info(f"[DeepFace] Ready  interval={ANALYSIS_INTERVAL}s  detector={DETECTOR_BACKEND}")

    bus = FrameBus()
    r   = bus.redis()

    last_analyzed: dict[int, float] = {}

    while True:
        raw_dets = r.hgetall(DETECTION_KEY)
        if not raw_dets:
            time.sleep(POLL_INTERVAL)
            continue

        now  = time.time()
        pipe = r.pipeline()

        for tid_bytes in raw_dets:
            tid = int(tid_bytes)
            if now - last_analyzed.get(tid, 0) < ANALYSIS_INTERVAL:
                continue

            # Read the pre-encoded JPEG crop written by the YOLO service.
            crop_bytes = r.get(f"crops:{tid}")
            if not crop_bytes:
                continue

            buf  = np.frombuffer(crop_bytes, dtype=np.uint8)
            crop = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if crop is None:
                continue

            h, w = crop.shape[:2]
            if w < MIN_CROP_PX or h < MIN_CROP_PX:
                continue

            try:
                # ── Step 1: confidence pre-check ─────────────────────────────
                faces = DeepFace.extract_faces(
                    img_path=crop,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False,
                    align=True,
                )
                if not faces:
                    continue

                best = max(faces, key=lambda f: f.get("confidence", 0.0))
                best_confidence = float(best.get("confidence", 0.0))
                if best_confidence < CONFIDENCE_THRESHOLD:
                    log.debug(
                        f"[DeepFace] Person {tid} skipped — confidence {best_confidence:.2f} < {CONFIDENCE_THRESHOLD}"
                    )
                    continue

                # ── Step 2: attribute analysis on the aligned face ────────────
                # Use detector_backend="skip" so DeepFace does not run a second
                # detection pass; the crop was already validated above.
                face_f32 = best.get("face")
                analyze_input = (
                    cv2.cvtColor(
                        (face_f32 * 255).clip(0, 255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR,
                    )
                    if face_f32 is not None and face_f32.size > 0
                    else crop
                )

                result = DeepFace.analyze(
                    img_path=analyze_input,
                    actions=["age", "gender", "emotion", "race"],
                    detector_backend="skip",
                    enforce_detection=False,
                    silent=True,
                )
                attrs = result[0] if isinstance(result, list) else result

                pipe.hset(ATTRS_KEY, str(tid), json.dumps({
                    "age":        attrs.get("age"),
                    "gender":     attrs.get("dominant_gender"),
                    "emotion":    attrs.get("dominant_emotion"),
                    "race":       attrs.get("dominant_race"),
                    "confidence": round(best_confidence, 4),
                }))
                pipe.expire(ATTRS_KEY, ATTRS_TTL)
                last_analyzed[tid] = now
                log.info(
                    f"[DeepFace] Person {tid}: age={attrs.get('age')} "
                    f"gender={attrs.get('dominant_gender')} conf={best_confidence:.2f}"
                )

            except Exception as exc:
                log.debug(f"[DeepFace] Person {tid} skipped: {exc}")

        pipe.execute()
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
