"""
Stable Diffusion XL Turbo / SDXL-Lightning + ControlNet (DWpose) microservice.

Polls ``frames:dwpose`` every ``SD_INTERVAL`` seconds, runs SDXL conditioned
on the DWpose skeleton canvas, and publishes the generated image to
``frames:sd``.

Two backends are supported, selected via ``SD_BACKEND``:

  turbo     SDXL-Turbo + ControlNet (1-step, EulerAncestral).
  lightning ByteDance SDXL-Lightning UNet injected into the SDXL-base
            ControlNet pipeline (4-step, EulerDiscrete trailing).

The service sleeps between inferences so it does not contend with other GPU
workloads on the same device.

Environment variables
---------------------
  SD_BACKEND            Pipeline backend: "turbo" | "lightning". Default: turbo
  SD_INTERVAL           Seconds between inferences.             Default: 5
  SD_PROMPT             Positive text prompt.
  SD_NEGATIVE_PROMPT    Negative text prompt.                   Default: ""
  SD_STEPS              Inference steps. Defaults: turbo=1, lightning=4
  SD_GUIDANCE_SCALE     CFG scale. Distilled (Lightning) models work best
                        with 0.0; turbo needs > 1.0 for CFG/ControlNet.
                        Defaults: turbo=2.0, lightning=0.0
  SD_CONTROLNET_SCALE   ControlNet conditioning scale.          Default: 1.2
  SD_WIDTH              Output width  (pixels).                 Default: 1344
  SD_HEIGHT             Output height (pixels).                 Default: 768
  SD_SHOW_SIDE_BY_SIDE  Concat pose | generated (1/0).          Default: 0
  CONTROLNET_MODEL      HuggingFace ControlNet model ID.
                        Default: xinsir/controlnet-openpose-sdxl-1.0
  SDXL_MODEL            HuggingFace base model ID.
                        turbo default:     stabilityai/sdxl-turbo
                        lightning default: stabilityai/stable-diffusion-xl-base-1.0
  LIGHTNING_REPO        HF repo for Lightning weights.
                        Default: ByteDance/SDXL-Lightning
  LIGHTNING_CKPT        Safetensors filename inside LIGHTNING_REPO.
                        Default: sdxl_lightning_4step_unet.safetensors
  SHOW_FPS              Overlay FPS counter on output (1/0).   Default: 1
  REDIS_URL             Redis connection string.
"""

import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    UNet2DConditionModel,
    EulerDiscreteScheduler,
)
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from logging_config import configure_logging, get_logger
from frame_bus import FrameBus

configure_logging()
log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

SD_BACKEND          = os.environ.get("SD_BACKEND", "turbo").lower()  # "turbo" | "lightning"

# Per-backend defaults (each can still be overridden via env)
_IS_LIGHTNING       = SD_BACKEND == "lightning"
_DEFAULT_SDXL_MODEL = (
    "stabilityai/stable-diffusion-xl-base-1.0" if _IS_LIGHTNING
    else "stabilityai/sdxl-turbo"
)
_DEFAULT_STEPS          = 4 if _IS_LIGHTNING else 1
_DEFAULT_GUIDANCE_SCALE = 0.0 if _IS_LIGHTNING else 2.0  # distilled models skip CFG

SD_INTERVAL         = float(os.environ.get("SD_INTERVAL",         "5"))
SD_PROMPT           = os.environ.get("SD_PROMPT",           "a person, photorealistic, high quality")
SD_NEGATIVE_PROMPT  = os.environ.get("SD_NEGATIVE_PROMPT",  "")
SD_STEPS            = int(os.environ.get("SD_STEPS",            str(_DEFAULT_STEPS)))
SD_GUIDANCE_SCALE   = float(os.environ.get("SD_GUIDANCE_SCALE",   str(_DEFAULT_GUIDANCE_SCALE)))
SD_CONTROLNET_SCALE = float(os.environ.get("SD_CONTROLNET_SCALE", "1.2"))
SD_WIDTH            = int(os.environ.get("SD_WIDTH",            "1344"))
SD_HEIGHT           = int(os.environ.get("SD_HEIGHT",           "768"))
SD_SHOW_SIDE_BY_SIDE = os.environ.get("SD_SHOW_SIDE_BY_SIDE", "0") == "1"
CONTROLNET_MODEL    = os.environ.get("CONTROLNET_MODEL", "xinsir/controlnet-openpose-sdxl-1.0")
SDXL_MODEL          = os.environ.get("SDXL_MODEL",        _DEFAULT_SDXL_MODEL)
LIGHTNING_REPO      = os.environ.get("LIGHTNING_REPO",    "ByteDance/SDXL-Lightning")
LIGHTNING_CKPT      = os.environ.get("LIGHTNING_CKPT",    "sdxl_lightning_4step_unet.safetensors")
SHOW_FPS            = os.environ.get("SHOW_FPS", "1") == "1"

# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------

def _load_pipeline(device: str = "cuda", dtype: torch.dtype = torch.float16):
    log.info(f"[SD] Backend: {SD_BACKEND}")
    log.info(f"[SD] Loading ControlNet from {CONTROLNET_MODEL} ...")
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL,
        torch_dtype=dtype,
    ).to(device)

    if _IS_LIGHTNING:
        # ── SDXL-Lightning: inject custom UNet into the ControlNet pipeline ──
        log.info(
            f"[SD] Downloading SDXL-Lightning UNet  "
            f"repo={LIGHTNING_REPO}  ckpt={LIGHTNING_CKPT}"
        )
        unet = UNet2DConditionModel.from_config(
            SDXL_MODEL, subfolder="unet"
        ).to(device, dtype)
        unet.load_state_dict(
            load_file(hf_hub_download(LIGHTNING_REPO, LIGHTNING_CKPT), device=device)
        )

        log.info(f"[SD] Loading SDXL-Lightning pipeline from {SDXL_MODEL} ...")
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            SDXL_MODEL,
            unet=unet,
            controlnet=controlnet,
            torch_dtype=dtype,
            variant="fp16",
            use_safetensors=True,
        ).to(device)

        # Lightning requires trailing timestep spacing to match the distilled UNet
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
    else:
        # ── SDXL-Turbo ────────────────────────────────────────────────────────
        log.info(f"[SD] Loading SDXL Turbo pipeline from {SDXL_MODEL} ...")
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            SDXL_MODEL,
            controlnet=controlnet,
            torch_dtype=dtype,
            variant="fp16",
            use_safetensors=True,
        ).to(device)

        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )

    # ── Memory / performance optimisations ──────────────────────────────────
    pipe.enable_vae_tiling()  # reduces peak VRAM for non-square / widescreen decoding
    try:
        pipe.enable_xformers_memory_efficient_attention()
        log.info("[SD] xformers memory-efficient attention enabled")
    except Exception as exc:  # xformers not installed or not compatible
        log.warning(f"[SD] xformers unavailable, skipping: {exc}")

    log.info(
        f"[SD] Pipeline ready  backend={SD_BACKEND}  "
        f"steps={SD_STEPS}  guidance={SD_GUIDANCE_SCALE}  "
        f"size={SD_WIDTH}x{SD_HEIGHT}"
    )
    return pipe


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _infer(
    pipe,
    pose_image: Image.Image,
    generator: torch.Generator,
    prompt: str,
) -> np.ndarray:
    """Run one SDXL + ControlNet step. Returns BGR uint8 ndarray."""
    result = pipe(
        prompt=prompt,
        negative_prompt=SD_NEGATIVE_PROMPT,
        image=pose_image,
        num_inference_steps=SD_STEPS,
        guidance_scale=SD_GUIDANCE_SCALE,
        controlnet_conditioning_scale=SD_CONTROLNET_SCALE,
        width=SD_WIDTH,
        height=SD_HEIGHT,
        generator=generator,
    )
    rgb = np.array(result.images[0], dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Frame preparation: BGR ndarray → square-padded PIL (ControlNet expects PIL)
# ---------------------------------------------------------------------------

def _prepare_pose(pose_canvas: np.ndarray) -> Image.Image:
    """Letterbox to square, resize to SD output res, convert to PIL RGB."""
    pose_rgb = cv2.cvtColor(pose_canvas, cv2.COLOR_BGR2RGB)
    h, w = pose_rgb.shape[:2]
    side = max(h, w)
    pad_top  = (side - h) // 2
    pad_left = (side - w) // 2
    pose_sq = np.zeros((side, side, 3), dtype=np.uint8)
    pose_sq[pad_top:pad_top + h, pad_left:pad_left + w] = pose_rgb
    return Image.fromarray(pose_sq).resize((SD_WIDTH, SD_HEIGHT), Image.LANCZOS)


# ---------------------------------------------------------------------------
# FPS overlay
# ---------------------------------------------------------------------------

def _fps_overlay(frame: np.ndarray, fps: int) -> None:
    text  = f"SD FPS: {fps}"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.6, 2
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(frame, (5, 25 - th - 5), (15 + tw, 25 + bl), (255, 255, 255), -1)
    cv2.putText(frame, text, (10, 25), font, scale, (0, 0, 0), thick)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe   = _load_pipeline(device=device)
    generator = torch.Generator(device=device).manual_seed(42)

    bus = FrameBus()
    log.info(
        f"[SD] Starting inference loop  interval={SD_INTERVAL}s  "
        f"size={SD_WIDTH}x{SD_HEIGHT}  steps={SD_STEPS}"
    )

    fps_counter, fps_timer, fps_display = 0, time.time(), 0
    inference_count = 0

    while True:
        # ── Sleep between inferences to avoid saturating the GPU ─────────────
        time.sleep(SD_INTERVAL)

        # ── Grab the most-recent DWpose skeleton canvas ───────────────────────
        pose_canvas, _ = bus.latest("dwpose")
        if pose_canvas is None:
            log.warning("[SD] No dwpose frame available yet — skipping cycle.")
            continue

        # ── Build prompt: SD_PROMPT prefix + latest VLM description ──────────
        vlm_text = bus.vlm_description()
        if vlm_text:
            effective_prompt = f"{SD_PROMPT}, {vlm_text}"
            log.debug(f"[SD] Effective prompt: {effective_prompt[:120]}...")
        else:
            effective_prompt = SD_PROMPT

        # ── Prepare pose image ────────────────────────────────────────────────
        pose_pil = _prepare_pose(pose_canvas)

        # ── Inference ─────────────────────────────────────────────────────────
        t0 = time.time()
        generated_bgr = _infer(pipe, pose_pil, generator, effective_prompt)
        elapsed = time.time() - t0
        inference_count += 1
        log.info(f"[SD] Inference #{inference_count}  {elapsed:.2f}s")

        # ── Optional side-by-side: pose canvas | generated image ─────────────
        if SD_SHOW_SIDE_BY_SIDE:
            pose_resized = cv2.resize(pose_canvas, (SD_WIDTH, SD_HEIGHT))
            generated_bgr = np.concatenate([pose_resized, generated_bgr], axis=1)

        # ── FPS overlay ───────────────────────────────────────────────────────
        if SHOW_FPS:
            fps_counter += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_timer   = now
            _fps_overlay(generated_bgr, fps_display)

        # ── Publish to frames:sd ──────────────────────────────────────────────
        bus.publish("sd", generated_bgr, meta={"prompt": SD_PROMPT})


if __name__ == "__main__":
    main()
