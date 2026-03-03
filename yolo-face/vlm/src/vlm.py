"""
VLM microservice — Qwen2.5-VL scene description.

Subscribes to ``frames:raw``, and every ``VLM_INTERVAL`` seconds sends a
frame to ``Qwen/Qwen2.5-VL-7B-Instruct`` (or the configured model).  The
resulting plain-text description is stored in the Redis key ``vlm:latest``
for the API gateway to serve.

The model is loaded lazily on first invocation.

Environment variables
---------------------
  VLM_MODEL_ID        HuggingFace model ID.
                      Default: Qwen/Qwen2.5-VL-7B-Instruct
  VLM_PROMPT          Prompt sent with every frame.
                      Default: "Describe what you see in this image."
  VLM_INTERVAL        Seconds between inferences.  Default: 30
  VLM_MAX_NEW_TOKENS  Generation token budget.  Default: 256
  VLM_RESULT_TTL      Redis TTL for vlm:latest in seconds.  Default: 120
  REDIS_URL           Redis connection string.
"""

import os
import time

import cv2
import numpy as np
from PIL import Image

from logging_config import configure_logging, get_logger
from frame_bus import FrameBus

configure_logging()
log = get_logger(__name__)

VLM_MODEL_ID       = os.environ.get("VLM_MODEL_ID",      "Qwen/Qwen2.5-VL-7B-Instruct")
VLM_PROMPT         = os.environ.get("VLM_PROMPT",        "Describe what you see in this image.")
VLM_INTERVAL       = float(os.environ.get("VLM_INTERVAL",       "30"))
VLM_MAX_NEW_TOKENS = int(os.environ.get("VLM_MAX_NEW_TOKENS", "1024"))
VLM_RESULT_TTL     = int(os.environ.get("VLM_RESULT_TTL",     "120"))

VLM_KEY = "vlm:latest"

# ---------------------------------------------------------------------------
# Lazy model singleton
# ---------------------------------------------------------------------------

_model     = None
_processor = None


def _load_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    log.info(f"[VLM] Loading {VLM_MODEL_ID} — this may take a while …")
    _processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    _model.eval()
    log.info("[VLM] Model loaded.")
    return _model, _processor


def _infer(frame_bgr: np.ndarray) -> str:
    import torch
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        process_vision_info = None

    model, processor = _load_model()
    pil_image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text",  "text": VLM_PROMPT},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if process_vision_info is not None:
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
    else:
        inputs = processor(
            text=[text], images=[pil_image],
            padding=True, return_tensors="pt",
        )

    device = next(model.parameters()).device
    inputs = inputs.to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=VLM_MAX_NEW_TOKENS)

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    result = processor.batch_decode(trimmed, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    return result[0] if result else ""


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    bus = FrameBus()
    r   = bus.redis()

    log.info(f"[VLM] Polling frames:raw every {VLM_INTERVAL}s  model={VLM_MODEL_ID}")

    while True:
        time.sleep(VLM_INTERVAL)

        frame, _ = bus.latest("raw")
        if frame is None:
            log.debug("[VLM] No frame available yet, waiting…")
            continue

        log.info("[VLM] Running inference")
        try:
            description = _infer(frame)
            r.setex(VLM_KEY, VLM_RESULT_TTL, description)
            log.info(f"[VLM] Result ({len(description)} chars): {description[:120]}…")
        except Exception as exc:
            log.warning(f"[VLM] Inference failed: {exc}")


if __name__ == "__main__":
    main()
