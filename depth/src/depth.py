"""
Depth Anything microservice — Depth Anything V2 monocular depth estimation.

Subscribes to ``frames:raw``, runs Depth Anything V2 on each frame, and
publishes colorised depth maps to ``frames:depth``.

Environment variables
---------------------
  DEPTH_MODEL         Encoder variant to use.
                      Options: vits, vitb, vitl, vitg
                      Default: vitl
  DEPTH_GRAYSCALE     Emit grayscale instead of INFERNO colormap (1/0).
                      Default: 0
  SHOW_FPS            Overlay FPS counter (1/0).  Default: 1
  REDIS_URL           Redis connection string.
"""

import os
import time

import cv2
import numpy as np
import torch

from logging_config import configure_logging, get_logger
from frame_bus import FrameBus

configure_logging()
log = get_logger(__name__)

DEPTH_MODEL     = os.environ.get("DEPTH_MODEL", "vits")  # vits | vitb | vitl | vitg
DEPTH_GRAYSCALE = os.environ.get("DEPTH_GRAYSCALE", "1") == "1"
SHOW_FPS        = os.environ.get("SHOW_FPS", "1") == "1"

_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

_HF_REPO = {
    "vits": "depth-anything/Depth-Anything-V2-Small",
    "vitb": "depth-anything/Depth-Anything-V2-Base",
    "vitl": "depth-anything/Depth-Anything-V2-Large",
    "vitg": "depth-anything/Depth-Anything-V2-Giant",
}


class DepthAnythingV2Wrapper:
    """Depth Anything V2 inference wrapper."""

    def __init__(self, encoder: str = "vitl"):
        from depth_anything_v2.dpt import DepthAnythingV2
        from huggingface_hub import hf_hub_download

        if encoder not in _MODEL_CONFIGS:
            raise ValueError(f"Unknown encoder '{encoder}'. Choose from: {list(_MODEL_CONFIGS)}")

        log.info(f"[Depth] Loading Depth Anything V2 model: {encoder}")
        model = DepthAnythingV2(**_MODEL_CONFIGS[encoder])
        ckpt = hf_hub_download(
            repo_id=_HF_REPO[encoder],
            filename=f"depth_anything_v2_{encoder}.pth",
        )
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self._model = model.to("cuda").eval()
        log.info("[Depth] Model ready")

    def __call__(self, frame_bgr: np.ndarray, grayscale: bool = False) -> np.ndarray:
        depth = self._model.infer_image(frame_bgr)  # HxW float32
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max > depth_min:
            depth = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth = np.zeros_like(depth, dtype=np.uint8)
        if grayscale:
            return cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
        return cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)


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
    log.info(f"[Depth] Initialising Depth Anything V2 ({DEPTH_MODEL})")
    model = DepthAnythingV2Wrapper(encoder=DEPTH_MODEL)

    bus = FrameBus()
    log.info("[Depth] Subscribing to frames:raw")

    fps_counter, fps_timer, fps_display = 0, time.time(), 0

    for raw_frame, _ in bus.subscribe("raw"):
        depth_frame = model(raw_frame, grayscale=DEPTH_GRAYSCALE)

        if SHOW_FPS:
            fps_counter += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_timer   = now
            _fps_overlay(depth_frame, fps_display)

        bus.publish("depth", depth_frame)


if __name__ == "__main__":
    main()
