"""
VLM (Vision-Language Model) frame analysis service.

Downloads ``Qwen/Qwen2.5-VL-32B-Instruct`` from HuggingFace on first use and
analyses video frames on a background thread, storing plain-text results on
``ctx.vlm_latest_result``.

No video streaming is produced by this service.  Callers read
``ctx.vlm_latest_result`` directly (e.g. via a JSON API route).

Architecture
------------
A single background thread opens the video source, reads frames, and fires a
separate daemon thread for each VLM call so the capture loop is never blocked
by the (potentially slow) model.

Two gates control when the VLM is invoked:
  * **Frame stride** – only every ``vlm_frame_stride``-th frame is a candidate.
  * **Time gate** – at least ``vlm_interval`` seconds must have elapsed since
    the previous call.

Expected ``ctx`` attributes
---------------------------
Config (read-only):
    frame_source_ctx    object – context whose ``latest_raw_frame`` attribute is
                                 polled for new frames (e.g. ``yolo_ctx``)
    vlm_prompt          str   – prompt sent with every frame
                               (default: "Describe what you see in this image.")
    vlm_interval        float – minimum seconds between VLM calls (default 30)
    vlm_frame_stride    int   – process every N-th frame (default 1)
    vlm_max_new_tokens  int   – generation token budget (default 256)

Mutable state (written by this service):
    vlm_latest_result   str | None  – most-recent model output
    vlm_last_call_time  float       – epoch time of the last VLM call
"""

import threading
import time

import cv2
import numpy as np
from PIL import Image
from ultralytics.utils import LOGGER

#_MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"
_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# ---------------------------------------------------------------------------
# Model loader (lazy singleton)
# ---------------------------------------------------------------------------

_model     = None
_processor = None
_model_lock = threading.Lock()


def _load_model():
    """Load model + processor from HuggingFace (idempotent)."""
    global _model, _processor
    with _model_lock:
        if _model is not None:
            return _model, _processor

        LOGGER.info(f"[VLM] Loading {_MODEL_ID} – this may take a while …")

        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        _processor = AutoProcessor.from_pretrained(_MODEL_ID)
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            _MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        _model.eval()
        LOGGER.info("[VLM] Model loaded.")

    return _model, _processor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_vlm(frame_bgr: np.ndarray, prompt: str, max_new_tokens: int) -> str:
    """Run the VLM on a single BGR frame and return the generated text."""
    LOGGER.debug("[VLM] _run_vlm: importing qwen_vl_utils")
    try:
        from qwen_vl_utils import process_vision_info
        LOGGER.debug("[VLM] _run_vlm: qwen_vl_utils imported OK")
    except ImportError:
        LOGGER.warning("[VLM] _run_vlm: qwen_vl_utils not found — falling back to raw PIL path")
        process_vision_info = None

    LOGGER.debug("[VLM] _run_vlm: loading model")
    model, processor = _load_model()
    LOGGER.debug("[VLM] _run_vlm: model ready")

    pil_image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    LOGGER.debug(f"[VLM] _run_vlm: PIL image size={pil_image.size}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text",  "text": prompt},
            ],
        }
    ]

    LOGGER.debug("[VLM] _run_vlm: applying chat template")
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    LOGGER.debug(f"[VLM] _run_vlm: chat template applied (len={len(text)})")

    if process_vision_info is not None:
        LOGGER.debug("[VLM] _run_vlm: running process_vision_info")
        image_inputs, video_inputs = process_vision_info(messages)
        LOGGER.debug(f"[VLM] _run_vlm: process_vision_info done — {len(image_inputs) if image_inputs else 0} image(s)")
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    else:
        inputs = processor(
            text=[text],
            images=[pil_image],
            padding=True,
            return_tensors="pt",
        )

    import torch
    device = next(model.parameters()).device
    LOGGER.debug(f"[VLM] _run_vlm: moving inputs to device={device}")
    inputs = inputs.to(device)
    LOGGER.debug(f"[VLM] _run_vlm: input_ids shape={inputs.input_ids.shape}")

    LOGGER.debug(f"[VLM] _run_vlm: calling model.generate (max_new_tokens={max_new_tokens})")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    LOGGER.debug(f"[VLM] _run_vlm: generate done — output shape={generated_ids.shape}")

    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    LOGGER.debug("[VLM] _run_vlm: decoding output")
    result = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    LOGGER.debug(f"[VLM] _run_vlm: decode done — result length={len(result[0]) if result else 0}")
    return result[0] if result else ""


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _vlm_worker(ctx, stop_event: threading.Event) -> None:
    """Polling thread: sample raw frames from ctx.frame_source_ctx and dispatch VLM calls.

    Reads ``ctx.frame_source_ctx.latest_raw_frame`` (written by the YOLO capture
    service before any annotation) so that VLM never sees bounding boxes.
    """
    vlm_interval:   float = getattr(ctx, "vlm_interval",      30.0)
    frame_stride:   int   = max(1, getattr(ctx, "vlm_frame_stride", 1))
    prompt:         str   = getattr(ctx, "vlm_prompt",
                                   "Describe what you see in this image.")
    max_new_tokens: int   = getattr(ctx, "vlm_max_new_tokens", 256)

    frame_source         = ctx.frame_source_ctx
    frame_counter: int   = 0
    last_vlm_time: float = 0.0
    prev_frame           = None          # track object identity to detect new frames

    LOGGER.info("[VLM] Worker started — polling shared frame source.")

    while not stop_event.is_set():
        raw = getattr(frame_source, "latest_raw_frame", None)

        if raw is None or raw is prev_frame:
            time.sleep(0.01)
            continue

        prev_frame = raw
        frame_counter += 1
        now = time.time()

        if frame_counter % frame_stride == 0 and (now - last_vlm_time) >= vlm_interval:
            last_vlm_time          = now
            ctx.vlm_last_call_time = now

            LOGGER.info(
                f"[VLM] Dispatching inference on frame {frame_counter} "
                f"(interval {vlm_interval:.0f} s)"
            )

            candidate = raw.copy()

            def _infer(f=candidate, p=prompt, n=max_new_tokens):
                LOGGER.debug("[VLM] Inference thread started")
                try:
                    result = _run_vlm(f, p, n)
                    ctx.vlm_latest_result = result
                    if result:
                        LOGGER.info(
                            f"[VLM] Result: {result[:120]}{'…' if len(result) > 120 else ''}"
                        )
                    else:
                        LOGGER.warning("[VLM] Model returned an empty result")
                except BaseException as exc:
                    LOGGER.error(f"[VLM] Inference error ({type(exc).__name__}): {exc}", exc_info=True)
                finally:
                    LOGGER.debug("[VLM] Inference thread exiting")

            threading.Thread(target=_infer, daemon=True, name="vlm-infer").start()

    LOGGER.info("[VLM] Worker exited.")


# ---------------------------------------------------------------------------
# Public service class
# ---------------------------------------------------------------------------

class VlmService:
    """Runs VLM analysis on a video source in the background.

    Results are written to ``ctx.vlm_latest_result`` as plain text.
    No video frames are returned or buffered by this service.

    Parameters
    ----------
    ctx:
        A :class:`~types.SimpleNamespace` (or any object) carrying the
        attributes listed in this module's docstring.  The following mutable
        fields are **initialised** by the constructor if absent:

        * ``vlm_latest_result``  → ``None``
        * ``vlm_last_call_time`` → ``0.0``

    Example usage in ``app.py``::

        ctx.vlm_prompt       = "How many people are visible? Describe their actions."
        ctx.vlm_interval     = 30.0   # call VLM at most every 30 s
        ctx.vlm_frame_stride = 5      # only consider every 5th frame

        vlm_service = VlmService(ctx)
        vlm_service.start()

        # In a Flask route:
        # return jsonify({"description": ctx.vlm_latest_result})
    """

    def __init__(self, ctx) -> None:
        self.ctx = ctx
        if not hasattr(ctx, "vlm_latest_result"):
            ctx.vlm_latest_result = None
        if not hasattr(ctx, "vlm_last_call_time"):
            ctx.vlm_last_call_time = 0.0

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background capture+inference thread."""
        if self._thread and self._thread.is_alive():
            LOGGER.warning("[VLM] Service is already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=_vlm_worker,
            args=(self.ctx, self._stop_event),
            name="vlm-capture",
            daemon=True,
        )
        self._thread.start()
        LOGGER.info("[VLM] Service started.")

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        LOGGER.info("[VLM] Service stopped.")

    @property
    def latest_result(self) -> str | None:
        """The most recent VLM text output, or ``None`` if not yet available."""
        return self.ctx.vlm_latest_result
