"""Mask microservice — background replacement for segmentation frames.

Subscribes to `frames:raw` and `frames:yolo_seg_tensor`.  When both a raw
frame and a mask are available, composes the persons over a replacement
background image and publishes the result to `frames:yolo_seg_bg`.  The mask
stream contains a uint8 grayscale frame (0/255) representing the union of all
person instances.

Environment variables
---------------------
  BG_IMAGE_PATH   Path to replacement background image.  Default
                  /app/models/background.jpg
"""

import os
import time

import cv2
import numpy as np

from logging_config import configure_logging, get_logger
from frame_bus import FrameBus, REDIS_URL

configure_logging()
log = get_logger(__name__)

# threading / performance
_CPU_THREADS = int(os.environ.get("APP_CPU_THREADS", "4"))
os.environ.setdefault("OMP_NUM_THREADS", str(_CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_CPU_THREADS))
cv2.setNumThreads(_CPU_THREADS)

BG_IMAGE_PATH = os.environ.get("BG_IMAGE_PATH", "/app/models/background.jpg")
SHOW_FPS       = os.environ.get("SHOW_FPS", "1") == "1"
# blur kernel size for mask feathering (must be odd)
BLUR_KSIZE     = int(os.environ.get("BLUR_KSIZE", "31"))


def _fps_overlay(im: np.ndarray, fps: int) -> None:
    text  = f"FPS: {fps}"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 1.0, 2
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    h, w = im.shape[:2]
    fx = w - tw - 15
    cv2.rectangle(im, (fx - 5, 25 - th - 5), (fx + tw + 5, 25 + bl), (255, 255, 255), -1)
    cv2.putText(im, text, (fx, 25), font, scale, (104, 31, 17), thick, cv2.LINE_AA)


def load_background() -> np.ndarray | None:
    if not os.path.isfile(BG_IMAGE_PATH):
        log.info(f"[MASK] no background file at {BG_IMAGE_PATH}")
        return None
    img = cv2.imread(BG_IMAGE_PATH)
    if img is None:
        log.warning(f"[MASK] failed to load background image {BG_IMAGE_PATH}")
        return None
    log.info(f"[MASK] loaded background image from {BG_IMAGE_PATH}")
    return img


def composite(raw_frame: np.ndarray, mask: np.ndarray, bg_img: np.ndarray) -> np.ndarray:
    """Return BGR uint8 composed frame using fast numpy alpha blending."""
    # Convert mask to 0.0-1.0 float32 and add a channel dimension (H, W, 1)
    alpha = (mask.astype(np.float32, copy=False) / 255.0)[..., np.newaxis]

    # Vectorized alpha blend
    out = raw_frame * alpha + bg_img * (1.0 - alpha)
    return out.astype(np.uint8, copy=False)


def main() -> None:
    bg_img_orig = load_background()
    bg_img = None
    bg_shape = None

    bus = FrameBus()
    r = bus.redis()

    log.info(f"[MASK] subscribing to frames:raw (redis={REDIS_URL})")

    fps_counter, fps_timer, fps_display = 0, time.time(), 0
    k = BLUR_KSIZE if BLUR_KSIZE % 2 == 1 else BLUR_KSIZE + 1

    for raw_frame, _ in bus.subscribe("raw"):
        if bg_img_orig is None:
            continue
            
        # Cache the resized background so we only pay this penalty once
        if bg_shape != raw_frame.shape:
            bg_img = cv2.resize(bg_img_orig, (raw_frame.shape[1], raw_frame.shape[0]))
            bg_shape = raw_frame.shape

        mask_frame, meta = bus.latest("yolo_seg_tensor")
        if mask_frame is None:
            continue

        # INSTANTLY copy or convert to release the FrameBus shared memory lock
        if mask_frame.ndim == 3 and mask_frame.shape[2] == 3:
            mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
        else:
            mask_frame = mask_frame.copy()

        # Soften/fade edges
        mask_frame = cv2.GaussianBlur(mask_frame, (k, k), 0)

        try:
            out = composite(raw_frame, mask_frame, bg_img)
            
            if SHOW_FPS:
                fps_counter += 1
                now = time.time()
                if now - fps_timer >= 1.0:
                    fps_display = fps_counter
                    fps_counter = 0
                    fps_timer = now
                _fps_overlay(out, fps_display)
                
            bus.publish("yolo_seg_bg", out, meta={"frame": meta.get("frame")})
        except Exception as e:
            log.error(f"[MASK] composition failed: {e}")
            continue


if __name__ == "__main__":
    main()