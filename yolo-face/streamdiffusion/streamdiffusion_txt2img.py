import os
import sys
from typing import Literal, Dict, Optional

import fire
import cv2
import numpy as np
from PIL import Image
import time


#sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper import StreamDiffusionWrapper
from frame_bus import FrameBus

DEFAULT_PROMPT = os.environ.get(
    "SD_PROMPT",
    "A retro robot portrait with a thick glasses, smiling, blue background",
)
NEGATIVE_PROMPT = os.environ.get(
    "SD_NEGATIVE_PROMPT",
    "black and white, blurry, low resolution, pixelated, pixel art, low quality, low fidelity",
)

# Environment-overridable defaults
MODEL_ID_DEFAULT = os.environ.get("SD_MODEL", "stabilityai/sdxl-turbo")
SD_WIDTH = int(os.environ.get("SD_WIDTH", "512"))
SD_HEIGHT = int(os.environ.get("SD_HEIGHT", "512"))
SD_ACCEL = os.environ.get("SD_ACCEL", "tensorrt")
SD_SEED = int(os.environ.get("SD_SEED", "11"))
SD_DENOISING_BATCH = os.environ.get("SD_DENOISING_BATCH", "1") == "1"


def _fps_overlay(frame: np.ndarray, fps: int) -> None:
    text = f"SD FPS: {fps}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.6, 2
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(frame, (5, 25 - th - 5), (15 + tw, 25 + bl), (255, 255, 255), -1)
    cv2.putText(frame, text, (10, 25), font, scale, (0, 0, 0), thick)


def main(
    model_id_or_path: str | None = None,
    lora_dict: Optional[Dict[str, float]] = None,
    prompt: str | None = None,
    width: int | None = None,
    height: int | None = None,
    acceleration: Literal["none", "xformers", "tensorrt"] | None = None,
    use_denoising_batch: bool | None = None,
    seed: int | None = None,
):
    """
    Continuous text-to-image generator (no raw frame input).

    Produces images from `prompt`, saves to /output, and publishes to FrameBus `sd`.
    """

    # Resolve env-overridable defaults
    model_id_or_path = model_id_or_path or MODEL_ID_DEFAULT
    prompt = prompt or DEFAULT_PROMPT
    width = width or SD_WIDTH
    height = height or SD_HEIGHT
    acceleration = acceleration or SD_ACCEL
    use_denoising_batch = use_denoising_batch if use_denoising_batch is not None else SD_DENOISING_BATCH
    seed = seed if seed is not None else SD_SEED

    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=lora_dict,
        t_index_list=[0],
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=10,
        use_lcm_lora=False,
        vae_id=None,
        acceleration=acceleration,
        output_type="np",
        mode="txt2img",
        use_denoising_batch=use_denoising_batch,
        cfg_type="none",
        seed=seed,
        engine_dir="/app/models",
    )
    
    stream.prepare(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=2,
        guidance_scale=0.0,
    )

    # Use FrameBus for publishing generated frames (no raw input read)
    bus = FrameBus()

    # Initialize variables for FPS calculation
    frame_count = 0
    fps = 0
    fps_display = "FPS: 0"
    start_time = time.time()

    while True:
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time

        # Update FPS value once every second or after every 20 frames
        if elapsed_time >= 1.0 or frame_count == 20:
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            fps_display = f"FPS: {fps:.2f}"
            frame_count = 0
            start_time = time.time()

        output_image = stream()

        # Ensure the output is a uint8 HxWxC BGR numpy array and publish to FrameBus
        if isinstance(output_image, Image.Image):
            img_np = np.array(output_image)
        else:
            img_np = output_image

        # If float in [0,1], scale to [0,255]
        if isinstance(img_np, np.ndarray) and img_np.dtype in (np.float32, np.float64):
            img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

        # If image is RGB, convert to BGR for OpenCV / FrameBus
        if isinstance(img_np, np.ndarray) and img_np.ndim == 3 and img_np.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_np

        # Overlay FPS if possible
        try:
            if isinstance(img_bgr, np.ndarray) and img_bgr.ndim == 3:
                _fps_overlay(img_bgr, int(fps))
        except Exception:
            pass

        # Publish to frames:sd only
        try:
            bus.publish("sd", img_bgr, meta={"prompt": prompt})
        except Exception:
            pass
        # small sleep to avoid tight loop if generation is extremely fast
        time.sleep(0.01)


if __name__ == "__main__":
    fire.Fire(main)
