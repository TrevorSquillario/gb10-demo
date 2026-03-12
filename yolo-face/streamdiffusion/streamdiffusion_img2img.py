import os
import sys
from typing import Literal, Dict, Optional

import fire
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image
import time

#sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper import StreamDiffusionWrapper
from frame_bus import FrameBus

DEFAULT_PROMPT = os.environ.get(
    "SD_PROMPT",
    "Portrait of The Joker in a Halloween costume, face painting, with a glare pose, detailed, intricate, full of color, cinematic lighting, trending on ArtStation, 8K, hyperrealistic, focused, extreme details, Unreal Engine 5 cinematic, masterpiece",
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
SD_SEED = int(os.environ.get("SD_SEED", "2"))
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
    Process for generating images based on a prompt using a specified model.

    Parameters
    ----------    
    model_id_or_path : str
        The name of the model to use for image generation.
    lora_dict : Optional[Dict[str, float]], optional
        The lora_dict to load, by default None.
        Keys are the LoRA names and values are the LoRA scales.
        Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
    prompt : str
        The prompt to generate images from.
    width : int, optional
        The width of the image, by default 512.
    height : int, optional
        The height of the image, by default 512.
    acceleration : Literal["none", "xformers", "tensorrt"]
        The type of acceleration to use for image generation.
    use_denoising_batch : bool, optional
        Whether to use denoising batch or not, by default False.
    seed : int, optionalq
        The seed, by default 2. if -1, use random seed.
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
        t_index_list=[1],
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=10,
        use_lcm_lora=False,
        vae_id=None,
        acceleration=acceleration,
        output_type="np",
        mode="img2img",
        use_denoising_batch=use_denoising_batch,
        cfg_type="self",
        seed=seed,
        engine_dir="/app/models"
    )
    
    stream.prepare(
        prompt=prompt,
        negative_prompt= NEGATIVE_PROMPT,
        num_inference_steps=2,
        guidance_scale=0.0,
    )

    # Use FrameBus: read from frames:raw and publish to frames:sd
    bus = FrameBus()

    # Use the latest raw frame to determine center-crop geometry when available
    frame, _ = bus.latest("raw")
    if frame is None:
        # fallback geometry if no frame yet
        frame_width, frame_height = width, height
    else:
        frame_height, frame_width = frame.shape[:2]

    crop_size = min(frame_width, frame_height)
    mid_x, mid_y = int(frame_width / 2), int(frame_height / 2)
    cw2, ch2 = int(crop_size / 2), int(crop_size / 2)
   
    # Initialize variables for FPS calculation
    frame_count = 0
    fps = 0
    fps_display = "FPS: 0"
    start_time = time.time()

    while True:
        # Read latest raw frame from the bus
        frame, _ = bus.latest("raw")
        if frame is None:
            time.sleep(0.05)
            continue

        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time

        # Update FPS value once every second or after every 20 frames
        if elapsed_time >= 1.0 or frame_count == 20:
            fps = frame_count / elapsed_time
            fps_display = f"FPS: {fps:.2f}"
            frame_count = 0
            start_time = time.time()

        # Center crop the raw BGR frame
        cropped_img = frame[mid_y - ch2: mid_y + ch2, mid_x - cw2: mid_x + cw2]

        # Convert BGR -> RGB and create tensor directly (skip PIL round-trip)
        img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        # to tensor in [0,1], shape [C,H,W]
        input_tensor = TF.to_tensor(img_rgb)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        input_tensor = input_tensor.unsqueeze(0).to(device=device, dtype=dtype)
        # [0,1] -> [-1,1]
        input_tensor = input_tensor * 2.0 - 1.0
        # GPU resize to model size if needed (avoid CPU resize every frame)
        if (input_tensor.shape[2], input_tensor.shape[3]) != (height, width):
            input_tensor = F.interpolate(input_tensor, size=(height, width), mode="bilinear", align_corners=False)
        output_image = stream(image=input_tensor)

        # Ensure the output is a uint8 HxWxC BGR numpy array before writing
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

        # Publish to frames:sd
        try:
            bus.publish("sd", img_bgr, meta={"prompt": prompt})
        except Exception:
            pass

        # (no GUI) loop continues


if __name__ == "__main__":
    fire.Fire(main)
