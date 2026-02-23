"""
Stable Diffusion XL Turbo + ControlNet (DWpose) frame generator — Producer-Consumer pattern.

Architecture
------------
SD Inference thread  (GPU)
    Pulls pose_canvas frames from a shared pose_queue (fed by the DWpose worker)
    → resizes to SD output resolution → runs SDXL Turbo + ControlNet conditioning
    → optionally composites a side-by-side view → pushes BGR ndarray into out_queue.

Encoding thread   (CPU, the Flask generator)
    Pulls generated frames from out_queue → cv2.imencode JPEG → yields MJPEG
    boundary chunks to the HTTP response.

Expected ``ctx`` attributes
---------------------------
    pose_queue              queue.Queue   — shared with DWpose worker
    sd_service              StableDiffusionService instance
    sd_prompt               str
    sd_negative_prompt      str
    sd_steps                int    — 1-4 for SDXL Turbo
    sd_guidance_scale       float  — 0.0 for pure distillation (SDXL Turbo default)
    sd_controlnet_scale     float  — ControlNet conditioning scale (0.0–2.0)
    sd_width                int
    sd_height               int
    sd_show_side_by_side    bool   — concat pose canvas | generated image
    show_fps                bool
    dwpose_ctx              SimpleNamespace | None
                                — when provided, ``generate_stablediffusion``
                                  starts its own DWpose capture thread so the
                                  /sd_feed route works independently without a
                                  concurrent /dwpose_feed consumer.
"""

import queue
import threading
import time

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
)
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from ultralytics.utils import LOGGER

from services.dwpose import start_dwpose_background

_QUEUE_MAXSIZE = 2   # SD is slower than DWpose; shallow queue keeps frames fresh
_SENTINEL = None

_DEFAULT_CONTROLNET = "xinsir/controlnet-openpose-sdxl-1.0"
_DEFAULT_SDXL       = "stabilityai/sdxl-turbo"


# ---------------------------------------------------------------------------
# Queue helper (identical drop-oldest strategy used across all services)
# ---------------------------------------------------------------------------

def _queue_put(q: queue.Queue, item) -> None:
    """Non-blocking put; drops oldest entry when queue is full."""
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass


# ---------------------------------------------------------------------------
# Model loader — instantiate once at startup, pass via ctx.sd_service
# ---------------------------------------------------------------------------

class StableDiffusionService:
    """Loads and owns the SDXL Turbo + ControlNet pipeline."""

    def __init__(
        self,
        controlnet_model: str = _DEFAULT_CONTROLNET,
        sdxl_model: str = _DEFAULT_SDXL,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.device = device
        self.dtype = dtype

        LOGGER.info(f"[SD] Loading ControlNet from {controlnet_model} ...")
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_model,
            torch_dtype=dtype,
        ).to(device)

        LOGGER.info(f"[SD] Loading SDXL Turbo pipeline from {sdxl_model} ...")
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            sdxl_model,
            controlnet=self.controlnet,
            torch_dtype=dtype,
            variant="fp16",
            use_safetensors=True,
        ).to(device)

        # Euler Ancestral is the recommended scheduler for SDXL Turbo
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )

        # Memory optimisations safe on a GB10
        #self.pipe.enable_xformers_memory_efficient_attention()
        # Uncomment if VRAM constrained:
        # self.pipe.enable_sequential_cpu_offload()

        LOGGER.info("[SD] Pipeline ready.")

    @torch.inference_mode()
    def __call__(
        self,
        pose_image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        num_steps: int = 1,
        guidance_scale: float = 0.0,
        controlnet_scale: float = 1,
        width: int = 512,
        height: int = 512,
        generator: torch.Generator | None = None,
    ) -> np.ndarray:
        """Run one inference step.  Returns a BGR uint8 ndarray."""
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pose_image,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_scale,
            width=width,
            height=height,
            generator=generator,
        )
        rgb = np.array(result.images[0], dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# SD inference worker
# ---------------------------------------------------------------------------

def _sd_worker(
    ctx,
    pose_queue: queue.Queue,
    out_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """Inference thread: pull pose canvas → SDXL Turbo + ControlNet → push generated image."""
    fps_counter, fps_timer, fps_display = 0, time.time(), 0
    generator = torch.Generator(device=ctx.sd_service.device).manual_seed(42)

    try:
        while not stop_event.is_set():
            # ── Pull latest pose canvas ───────────────────────────────────────
            try:
                pose_canvas = pose_queue.get(timeout=5.0)
            except queue.Empty:
                LOGGER.warning("[SD] Timed out waiting for pose frame — is the DWpose feed running?")
                continue

            if pose_canvas is _SENTINEL:
                LOGGER.info("[SD] Received sentinel from pose queue — stopping.")
                break

            # ── BGR ndarray → PIL RGB (ControlNet expects PIL) ───────────────
            # Letterbox to square so joint positions are not distorted by
            # a non-uniform stretch before ControlNet conditioning.
            pose_rgb = cv2.cvtColor(pose_canvas, cv2.COLOR_BGR2RGB)
            h, w = pose_rgb.shape[:2]
            side = max(h, w)
            pad_top  = (side - h) // 2
            pad_left = (side - w) // 2
            pose_sq = np.zeros((side, side, 3), dtype=np.uint8)
            pose_sq[pad_top:pad_top + h, pad_left:pad_left + w] = pose_rgb
            pose_pil = Image.fromarray(pose_sq).resize(
                (ctx.sd_width, ctx.sd_height), Image.LANCZOS
            )

            # ── SDXL Turbo + ControlNet inference ────────────────────────────
            generated_bgr = ctx.sd_service(
                pose_image=pose_pil,
                prompt=ctx.sd_prompt,
                negative_prompt=ctx.sd_negative_prompt,
                num_steps=ctx.sd_steps,
                guidance_scale=ctx.sd_guidance_scale,
                controlnet_scale=ctx.sd_controlnet_scale,
                width=ctx.sd_width,
                height=ctx.sd_height,
                generator=generator,
            )

            # ── Optional side-by-side: pose canvas | generated image ─────────
            if ctx.sd_show_side_by_side:
                pose_resized = cv2.resize(pose_canvas, (ctx.sd_width, ctx.sd_height))
                generated_bgr = np.concatenate([pose_resized, generated_bgr], axis=1)

            # ── FPS overlay (inference rate) ──────────────────────────────────
            if ctx.show_fps:
                fps_counter += 1
                if time.time() - fps_timer >= 1.0:
                    fps_display = fps_counter
                    fps_counter = 0
                    fps_timer = time.time()

                fps_text = f"SD FPS: {fps_display}"
                fps_font = cv2.FONT_HERSHEY_SIMPLEX
                fps_scale, fps_thickness = 0.6, 2
                (tw, th), bl = cv2.getTextSize(fps_text, fps_font, fps_scale, fps_thickness)
                cv2.rectangle(
                    generated_bgr,
                    (10 - 5, 25 - th - 5),
                    (10 + tw + 5, 25 + bl),
                    (255, 255, 255),
                    -1,
                )
                cv2.putText(
                    generated_bgr, fps_text, (10, 25),
                    fps_font, fps_scale, (0, 0, 0), fps_thickness,
                )

            _queue_put(out_queue, generated_bgr)

    finally:
        out_queue.put(_SENTINEL)


# ---------------------------------------------------------------------------
# Public generator
# ---------------------------------------------------------------------------

def generate_stablediffusion(ctx):
    """Yield MJPEG-boundary chunks of SDXL Turbo images conditioned on DWpose.

    If ``ctx.dwpose_ctx`` is set, a private DWpose capture thread is started
    automatically so this route works without a concurrent ``/dwpose_feed``
    consumer.  Otherwise ``ctx.pose_queue`` must already be receiving frames
    from an external DWpose worker.

    Args:
        ctx: SimpleNamespace with all fields listed in the module docstring.
    """
    out_queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)
    stop_event = threading.Event()

    # Start an embedded DWpose worker when the caller provides its context.
    # This makes /sd_feed self-contained — no need for /dwpose_feed to be open.
    dwpose_thread = None
    if getattr(ctx, "dwpose_ctx", None) is not None:
        # Ensure dwpose_ctx forwards pose canvases into our shared pose_queue.
        ctx.dwpose_ctx.pose_queue = ctx.pose_queue
        dwpose_thread = start_dwpose_background(ctx.dwpose_ctx, stop_event)
        if dwpose_thread is None:
            LOGGER.error("[SD] Failed to start embedded DWpose worker — /sd_feed will produce no frames.")

    thread = threading.Thread(
        target=_sd_worker,
        args=(ctx, ctx.pose_queue, out_queue, stop_event),
        name="sd-inference",
        daemon=True,
    )
    thread.start()

    try:
        while True:
            try:
                frame = out_queue.get(timeout=10.0)
            except queue.Empty:
                LOGGER.warning("[SD] Encoder timed out waiting for generated frame — stream may have stalled")
                break

            if frame is _SENTINEL:
                break

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

    except GeneratorExit:
        pass

    finally:
        stop_event.set()
        thread.join(timeout=5.0)
        if dwpose_thread is not None:
            dwpose_thread.join(timeout=3.0)
        LOGGER.info("[SD] Session ended.")
