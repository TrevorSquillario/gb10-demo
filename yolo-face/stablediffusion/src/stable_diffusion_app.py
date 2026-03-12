"""
Stable Diffusion XL SDXL-Lightning + ControlNet (DWpose) microservice.

Polls ``frames:dwpose`` every ``SD_INTERVAL`` seconds, runs SDXL conditioned
on the DWpose skeleton canvas, and publishes the generated image to
``frames:sd``.

Backend: SDXL-Lightning (only)

    lightning ByteDance SDXL-Lightning UNet injected into the SDXL-base
                        ControlNet pipeline (4-step, EulerDiscrete trailing).

The service sleeps between inferences so it does not contend with other GPU
workloads on the same device.

Environment variables
---------------------
    SD_BACKEND            Pipeline backend: "lightning" (only). Default: lightning
  SD_INTERVAL           Seconds between inferences.             Default: 5
  SD_PROMPT             Positive text prompt.
  SD_NEGATIVE_PROMPT    Negative text prompt.                   Default: ""
    SD_STEPS              Inference steps. Default: lightning=4
    SD_GUIDANCE_SCALE     CFG scale. Distilled (Lightning) models work best
                                                with 0.0. Default: 0.0
  SD_CONTROLNET_SCALE   ControlNet conditioning scale.          Default: 1.2
  SD_WIDTH              Output width  (pixels).                 Default: 1344
  SD_HEIGHT             Output height (pixels).                 Default: 768
  SD_SHOW_SIDE_BY_SIDE  Concat pose | generated (1/0).          Default: 0
  CONTROLNET_MODEL      HuggingFace ControlNet model ID.
                        Default: xinsir/controlnet-openpose-sdxl-1.0
    SDXL_MODEL            HuggingFace base model ID.
                                                default: stabilityai/stable-diffusion-xl-base-1.0
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
import torch._dynamo
from PIL import Image
import inspect
from diffusers import (
    T2IAdapter,
    StableDiffusionXLAdapterPipeline,
    UNet2DConditionModel,
    EulerDiscreteScheduler,
    AutoencoderTiny
)
from diffusers.models.attention_processor import AttnProcessor2_0
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from logging_config import configure_logging, get_logger
from frame_bus import FrameBus

configure_logging()
log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

SD_BACKEND          = os.environ.get("SD_BACKEND", "lightning").lower()

# Force Lightning-only operation
_IS_LIGHTNING = True
_DEFAULT_SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
_DEFAULT_STEPS = 2
_DEFAULT_GUIDANCE_SCALE = 0.0  # distilled models skip CFG

SD_INTERVAL         = float(os.environ.get("SD_INTERVAL",         "5"))
SD_PROMPT           = os.environ.get("SD_PROMPT",           "a person, photorealistic, high quality")
SD_NEGATIVE_PROMPT  = os.environ.get("SD_NEGATIVE_PROMPT",  "")
SD_STEPS            = int(os.environ.get("SD_STEPS",            str(_DEFAULT_STEPS)))
SD_GUIDANCE_SCALE   = float(os.environ.get("SD_GUIDANCE_SCALE",   str(_DEFAULT_GUIDANCE_SCALE)))
SD_CONTROLNET_SCALE = float(os.environ.get("SD_CONTROLNET_SCALE", "1.5"))
SD_WIDTH            = int(os.environ.get("SD_WIDTH",            "1344"))
SD_HEIGHT           = int(os.environ.get("SD_HEIGHT",           "768"))
SD_SHOW_SIDE_BY_SIDE = os.environ.get("SD_SHOW_SIDE_BY_SIDE", "0") == "1"
CONTROLNET_MODEL    = os.environ.get("CONTROLNET_MODEL", "TencentARC/t2i-adapter-openpose-sdxl-1.0")
SDXL_MODEL          = os.environ.get("SDXL_MODEL",        _DEFAULT_SDXL_MODEL)
LIGHTNING_REPO      = os.environ.get("LIGHTNING_REPO",    "ByteDance/SDXL-Lightning")
LIGHTNING_CKPT      = os.environ.get("LIGHTNING_CKPT",    "sdxl_lightning_2step_unet.safetensors")
SHOW_FPS            = os.environ.get("SHOW_FPS", "1") == "1"
TEACACHE_THRESHOLD  = float(os.environ.get("TEACACHE_THRESHOLD",  "0.1"))

class SageAttnProcessor:
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        # 1. Projections
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # 2. Manual 4D Reshape for SDXL
        # Query: (bs, seq, dim) -> (bs, seq, heads, head_dim) -> (bs, heads, seq, head_dim)
        head_dim = query.shape[-1] // attn.heads
        
        def reshape_heads(t):
            bsz, seq, _ = t.shape
            return t.view(bsz, seq, attn.heads, head_dim).transpose(1, 2)

        q = reshape_heads(query)
        k = reshape_heads(key)
        v = reshape_heads(value)

        # 3. SageAttention (Expects 4D: B, H, L, D)
        from sageattention import sageattn
        # Ensure it's contiguous and correct dtype for the kernel
        out = sageattn(q.contiguous(), k.contiguous(), v.contiguous(), is_causal=False)

        # 4. Reshape back: (bs, heads, seq, head_dim) -> (bs, seq, heads, head_dim) -> (bs, seq, dim)
        out = out.transpose(1, 2).reshape(hidden_states.shape[0], -1, attn.heads * head_dim)

        # 5. Final Output Projection
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        return out

# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------

def optimize_pipeline(pipe):
    # 1. Memory format optimization
    pipe.unet.to(memory_format=torch.channels_last)
    if hasattr(pipe, "adapter") and pipe.adapter is not None:
        pipe.adapter.to(memory_format=torch.channels_last)

    # 2. Apply SageAttention (UNet ONLY)
    sage_proc = SageAttnProcessor()
    # Ensure Dynamo skips the processor
    sage_proc.__call__ = torch._dynamo.disable(sage_proc.__call__)
    pipe.unet.set_attn_processor(sage_proc)

    # 3. Compile the models
    # Using max-autotune-no-cudagraphs for compatibility with graph breaks
    pipe.unet = torch.compile(pipe.unet, mode="max-autotune-no-cudagraphs")
    if hasattr(pipe, "adapter") and pipe.adapter is not None:
        pipe.adapter = torch.compile(pipe.adapter, mode="max-autotune-no-cudagraphs")
    
    return pipe

def _load_pipeline(device: str = "cuda", dtype: torch.dtype = torch.float16):
    log.info(f"[SD] Backend: {SD_BACKEND}")
    log.info("[SD] Loading ControlNet pipeline (no mask/inpaint) ...")

    log.info(f"[SD] Loading T2I adapter from {CONTROLNET_MODEL} ...")
    adapter = T2IAdapter.from_pretrained(
        CONTROLNET_MODEL,
        torch_dtype=dtype,
    ).to(device)

    # SDXL-Lightning: inject custom UNet into the ControlNet+Inpaint pipeline
    log.info(
        f"[SD] Downloading SDXL-Lightning UNet  repo={LIGHTNING_REPO}  ckpt={LIGHTNING_CKPT}"
    )
    unet = UNet2DConditionModel.from_config(
        SDXL_MODEL, subfolder="unet"
    ).to(device, dtype)
    unet.load_state_dict(
        load_file(hf_hub_download(LIGHTNING_REPO, LIGHTNING_CKPT), device=device)
    )

    log.info(f"[SD] Loading SDXL-Lightning Adapter pipeline from {SDXL_MODEL} ...")
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        SDXL_MODEL,
        adapter=adapter,
        unet=unet,
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True,
    ).to(device)

    # Debug: log callable signature and adapter info to verify expected kwargs
    try:
        sig = inspect.signature(pipe.__call__)
        log.info(f"[SD][DEBUG] Pipeline call signature: {sig}")
        params = list(sig.parameters.keys())
        log.info(f"[SD][DEBUG] Pipeline call params: {params}")
    except Exception as e:
        log.info(f"[SD][DEBUG] Failed to inspect pipeline signature: {e}")

    try:
        has_adapter = hasattr(pipe, "adapter") and pipe.adapter is not None
        log.info(f"[SD][DEBUG] pipe.adapter present={has_adapter} type={type(pipe.adapter) if has_adapter else None}")
    except Exception:
        pass

    # Lightning requires trailing timestep spacing to match the distilled UNet
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    # Tiny AutoEncoder for SD (TAESD) is a distilled VAE that decodes almost instantly with a negligible drop in pixel-perfect accuracy.
    pipe.vae = AutoencoderTiny.from_pretrained(
        "madebyollin/taesdxl", 
        torch_dtype=dtype
    ).to(device)

    pipe = optimize_pipeline(pipe)
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
    pose_image: Image.Image,  # The DWpose canvas
    generator: torch.Generator,
    prompt: str,
) -> np.ndarray:
    """Run one SDXL + T2I-Adapter step. Returns BGR uint8 ndarray."""
    start = time.perf_counter()
    
    # Setup kwargs for StableDiffusionXLAdapterPipeline
    kwargs = {
        "prompt": prompt,
        "negative_prompt": SD_NEGATIVE_PROMPT,
        "image": pose_image,  # T2I-Adapter expects the pose here
        "num_inference_steps": SD_STEPS,
        "guidance_scale": SD_GUIDANCE_SCALE,
        "adapter_conditioning_scale": SD_CONTROLNET_SCALE, # Use the mapped variable
        "adapter_conditioning_factor": 1.0,
        "generator": generator,
        "width": SD_WIDTH,
        "height": SD_HEIGHT,
    }

    result = pipe(**kwargs)
    
    end = time.perf_counter()
    log.info(f"[SD] inference took {end - start:.3f}s")
    
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
        #time.sleep(SD_INTERVAL)

        # Use SD_PROMPT only (do not append VLM description)
        effective_prompt = SD_PROMPT

        # Grab DWpose canvas for ControlNet conditioning
        pose_canvas, _ = bus.latest("dwpose")
        if pose_canvas is None:
            log.warning("[SD] No dwpose frame available yet — skipping cycle.")
            continue

        # Prepare pose PIL image (ControlNet expects a square, resized PIL)
        pose_pil = _prepare_pose(pose_canvas)

        # Log prepared sizes
        try:
            log.info(f"[SD] pose_pil.size={pose_pil.size}")
        except Exception:
            log.info("[SD] prepared PIL size logging failed")

        # ── Inference ─────────────────────────────────────────────────────
        t0 = time.time()
        try:
            generated_bgr = _infer(pipe, pose_pil, generator, effective_prompt)
        except Exception as exc:
            log.error(f"[SD] Inference failed: {exc}")
            continue
        elapsed = time.time() - t0
        inference_count += 1
        log.info(f"[SD] Inference #{inference_count}  {elapsed:.2f}s")
        try:
            gh, gw = generated_bgr.shape[:2]
            gch = generated_bgr.shape[2] if generated_bgr.ndim == 3 else 1
            log.info(f"[SD] generated image size={gw}x{gh} channels={gch}")
        except Exception:
            log.info("[SD] generated image shape logging failed")

        # Optional side-by-side: raw | generated
        if SD_SHOW_SIDE_BY_SIDE:
            raw_resized = cv2.resize(raw_frame, (SD_WIDTH, SD_HEIGHT))
            generated_bgr = np.concatenate([raw_resized, generated_bgr], axis=1)
            try:
                fh, fw = generated_bgr.shape[:2]
                fch = generated_bgr.shape[2] if generated_bgr.ndim == 3 else 1
                log.info(f"[SD] final published image size={fw}x{fh} channels={fch}")
            except Exception:
                log.info("[SD] final image shape logging failed")

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
        try:
            ph, pw = generated_bgr.shape[:2]
            log.info(f"[SD] publishing image size={pw}x{ph}")
        except Exception:
            log.info("[SD] publishing image shape logging failed")
        bus.publish("sd", generated_bgr, meta={"prompt": SD_PROMPT})


if __name__ == "__main__":
    main()
