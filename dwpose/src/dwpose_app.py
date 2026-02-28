"""
DWpose microservice — whole-body skeleton detection (TensorRT backend).

Subscribes to ``frames:raw``, runs DWpose-TensorRT skeleton detection, and
publishes the rendered canvas to ``frames:dwpose``.

Engine files must be pre-built with build_engine.py from the
yuvraj108c/Dwpose-Tensorrt repo and placed at the paths below.

Environment variables
---------------------
  TRT_DET_ENGINE   Path to yolox_l TensorRT engine.
                   Default: /app/models/tensorrt/dwpose/yolox_l.engine
  TRT_POSE_ENGINE  Path to dw-ll_ucoco_384 TensorRT engine.
                   Default: /app/models/tensorrt/dwpose/dw-ll_ucoco_384.engine
  SHOW_FPS         Overlay FPS counter (1/0).  Default: 1
  REDIS_URL        Redis connection string.
"""

import os
import time

import cv2

from logging_config import configure_logging, get_logger
from frame_bus import FrameBus

configure_logging()
log = get_logger(__name__)

SHOW_FPS        = os.environ.get("SHOW_FPS", "1") == "1"
TRT_DET_ENGINE  = os.environ.get("TRT_DET_ENGINE",  "/app/models/yolox_l.engine")
TRT_POSE_ENGINE = os.environ.get("TRT_POSE_ENGINE", "/app/models/dw-ll_ucoco_384.engine")

DET_ONNX  = os.environ.get("ONNX_DET_MODEL",  "/app/models/yolox_l.onnx")
POSE_ONNX = os.environ.get("ONNX_POSE_MODEL", "/app/models/dw-ll_ucoco_384.onnx")


def _ensure_engines() -> None:
    """Build TensorRT engines from ONNX if either engine file is missing."""
    if os.path.isfile(TRT_DET_ENGINE) and os.path.isfile(TRT_POSE_ENGINE):
        return

    log.info("[DWpose] TRT engines missing — building from ONNX (this may take several minutes)...")

    from export_trt import export_trt

    if not os.path.isfile(TRT_DET_ENGINE):
        log.info(f"[DWpose] Building det engine: {DET_ONNX} -> {TRT_DET_ENGINE}")
        export_trt(trt_path=TRT_DET_ENGINE, onnx_path=DET_ONNX, use_fp16=True)

    if not os.path.isfile(TRT_POSE_ENGINE):
        log.info(f"[DWpose] Building pose engine: {POSE_ONNX} -> {TRT_POSE_ENGINE}")
        export_trt(trt_path=TRT_POSE_ENGINE, onnx_path=POSE_ONNX, use_fp16=True)

    log.info("[DWpose] TRT engines built successfully.")


def _fps_overlay(im, fps: int) -> None:
    text = f"FPS: {fps}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 1.0, 2
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    h, w = im.shape[:2]
    fx = w - tw - 15
    cv2.rectangle(im, (fx - 5, 25 - th - 5), (fx + tw + 5, 25 + bl), (255, 255, 255), -1)
    cv2.putText(im, text, (fx, 25), font, scale, (104, 31, 17), thick, cv2.LINE_AA)


def main() -> None:
    _ensure_engines()

    # Propagate paths to Wholebody via env vars (read in wholebody.py)
    os.environ["TRT_DET_ENGINE"]  = TRT_DET_ENGINE
    os.environ["TRT_POSE_ENGINE"] = TRT_POSE_ENGINE

    log.info(f"[DWpose] Loading TRT engines  det={TRT_DET_ENGINE}  pose={TRT_POSE_ENGINE}")

    from dwpose import DWposeDetector
    detector = DWposeDetector()

    bus = FrameBus()
    log.info("[DWpose] Subscribing to frames:raw")

    fps_counter, fps_timer, fps_display = 0, time.time(), 0

    for raw_frame, _ in bus.subscribe("raw"):
        canvas = detector(
            image_np_hwc=raw_frame,
            show_body=True,
            show_face=True,
            show_hands=True,
        )

        if SHOW_FPS:
            fps_counter += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_timer   = now
            _fps_overlay(canvas, fps_display)

        bus.publish("dwpose", canvas)


if __name__ == "__main__":
    main()
