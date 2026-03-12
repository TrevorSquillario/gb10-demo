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
import torch.nn.functional as F

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


# Target Output resolution
OUT_W, OUT_H = 1280, 720

class DepthAnythingV2Wrapper:
    def __init__(self, encoder: str = "vits"):
        from depth_anything_v2.dpt import DepthAnythingV2
        from huggingface_hub import hf_hub_download

        log.info(f"[Depth] Loading model: {encoder} in FP16 mode")
        model = DepthAnythingV2(**_MODEL_CONFIGS[encoder])
        ckpt = hf_hub_download(repo_id=_HF_REPO[encoder], filename=f"depth_anything_v2_{encoder}.pth")
        
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self._model = model.to(self._device).half().eval()
        
        # Pre-calculated normalization values
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self._device).half()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self._device).half()

        # Calculate inference size maintaining 16:9 aspect ratio
        # Must be multiples of 14. 518/14 = 37. 
        # For 16:9, we'll use 518 wide and 294 high (21*14).
        self.inf_w, self.inf_h = 518, 294 

    @torch.no_grad()
    def __call__(self, frame_bgr: np.ndarray, grayscale: bool = False) -> np.ndarray:
        # Resize to maintain 16:9 aspect ratio at inference
        image = cv2.resize(frame_bgr, (self.inf_w, self.inf_h))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        # Prepare Tensor (FP16)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self._device).half()
        tensor = (tensor - self.mean) / self.std

        # Inference
        depth = self._model(tensor)
        
        # Resize to 720p output resolution
        depth = F.interpolate(depth[:, None], (OUT_H, OUT_W), mode="bilinear", align_corners=True)[0, 0]
        
        # Normalization
        depth_min, depth_max = depth.min(), depth.max()
        depth = (depth - depth_min) / (depth_max - depth_min) * 255.0
        depth_np = depth.cpu().numpy().astype(np.uint8)

        if grayscale:
            return cv2.cvtColor(depth_np, cv2.COLOR_GRAY2BGR)
        return cv2.applyColorMap(depth_np, cv2.COLORMAP_INFERNO)

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
