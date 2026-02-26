"""
StreamDiffusionV2 Single-GPU Inference Service.

Wraps the ``streamv2v/inference.py`` pipeline from the StreamDiffusionV2 repo
(https://github.com/chenfengxu714/StreamDiffusionV2) and exposes it as a
Python service coherent with the rest of the YOLO-FACE service layer.

Architecture
------------
The ``StreamDiffusionV2Service`` class initialises and owns the
``SingleGPUInferencePipeline`` from the cloned repo.  It generates an output
video (.mp4) from an input video and a text prompt, delegating all heavy
lifting to the upstream library.

A thin ``generate_streamdiffusionv2`` generator is also provided so the result
can be streamed back over Flask as an MJPEG feed or simply as a JSON response
containing the output path.

Expected ``ctx`` attributes (for the generator)
------------------------------------------------
    sd2_service             StreamDiffusionV2Service instance
    sd2_input_video         str  — absolute path to the source .mp4
    sd2_prompt_file         str  — path to a text file containing the prompt
                                   (one prompt per line; only the first is used)
    sd2_output_folder       str  — directory where output_000.mp4 will be saved
    sd2_config_path         str  — path to the OmegaConf YAML config
                                   (default: <repo>/configs/wan_causal_dmd_v2v.yaml)
    sd2_checkpoint_folder   str  — path to checkpoint folder
                                   (default: /app/YOLO-FACE/StreamDiffusionV2/ckpts/wan_causal_dmd_v2v)
    sd2_height              int  — output height  (default: 480)
    sd2_width               int  — output width   (default: 832)
    sd2_fps                 int  — output FPS      (default: 16)
    sd2_step                int  — denoising steps (default: 2)
    sd2_noise_scale         float — noise scale    (default: 0.700)
    sd2_num_frames          int  — number of frames to generate (default: 81)
    sd2_target_fps          int | None — target realtime FPS for dynamic batching
"""

import logging
import os
import sys
import time
from types import SimpleNamespace

import torch

from ultralytics.utils import LOGGER

# ---------------------------------------------------------------------------
# Repo root — adjust if the clone location ever changes
# ---------------------------------------------------------------------------

_REPO_ROOT = os.environ.get(
    "STREAMDIFFUSIONV2_ROOT",
    "/app/YOLO-FACE/StreamDiffusionV2",
)


def _ensure_repo_on_path() -> None:
    """Prepend the StreamDiffusionV2 repo root to sys.path if necessary."""
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Default paths derived from the repo root
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG      = os.path.join(_REPO_ROOT, "configs", "wan_causal_dmd_v2v.yaml")
_DEFAULT_CKPT_FOLDER = os.path.join(_REPO_ROOT, "ckpts", "wan_causal_dmd_v2v")


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------

class StreamDiffusionV2Service:
    """Loads and owns the StreamDiffusionV2 single-GPU inference pipeline.

    Parameters
    ----------
    config_path:        Path to the OmegaConf YAML config file.
    checkpoint_folder:  Directory containing ``model.pt``.
    device:             Torch device string (default: ``"cuda"``).
    """

    def __init__(
        self,
        config_path: str = _DEFAULT_CONFIG,
        checkpoint_folder: str = _DEFAULT_CKPT_FOLDER,
        device: str = "cuda",
    ) -> None:
        _ensure_repo_on_path()

        # Lazy imports — the repo must be on sys.path first.
        from omegaconf import OmegaConf
        from streamv2v.inference import SingleGPUInferencePipeline

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        LOGGER.info(f"[SD2] Loading config from {config_path}")
        self.config = OmegaConf.load(config_path)

        LOGGER.info(f"[SD2] Initialising SingleGPUInferencePipeline on {self.device}")
        self.pipeline_manager = SingleGPUInferencePipeline(self.config, self.device)

        LOGGER.info(f"[SD2] Loading checkpoint from {checkpoint_folder}")
        self.pipeline_manager.load_model(checkpoint_folder)

        LOGGER.info("[SD2] StreamDiffusionV2 pipeline ready.")

    def run(
        self,
        input_video: str | None,
        prompt_file: str,
        output_folder: str,
        height: int = 480,
        width: int = 832,
        fps: int = 16,
        step: int = 2,
        noise_scale: float = 0.700,
        num_frames: int = 81,
        target_fps: int | None = None,
    ) -> str:
        """Run offline inference and return the path to the output video.

        Parameters
        ----------
        input_video:    Path to the source .mp4 (``None`` for text-to-video).
        prompt_file:    Path to a plain-text file of prompts (first line used).
        output_folder:  Directory where ``output_000.mp4`` will be written.
        height/width:   Output resolution.
        fps:            Output framerate written into the container.
        step:           Number of denoising steps (1–4 recommended).
        noise_scale:    Additive noise strength (0.0–1.0).
        num_frames:     Total number of frames to generate.
        target_fps:     Optional target realtime FPS; triggers dynamic batching.

        Returns
        -------
        Absolute path to the saved output video.
        """
        _ensure_repo_on_path()

        import numpy as np
        import torchvision
        import torchvision.transforms.functional as TF
        from causvid.data import TextDataset
        from einops import rearrange
        from omegaconf import OmegaConf

        # ── Merge per-call overrides into config ────────────────────────────
        overrides = dict(
            height=height,
            width=width,
            fps=fps,
            step=step,
            noise_scale=noise_scale,
            num_frames=num_frames,
        )
        if target_fps is not None:
            overrides["target_fps"] = target_fps

        config = OmegaConf.merge(self.config, OmegaConf.create(overrides))

        # Translate `step` → denoising_step_list (mirrors inference.py logic)
        if step <= 1:
            config.denoising_step_list = [700, 0]
        elif step == 2:
            config.denoising_step_list = [700, 500, 0]
        elif step == 3:
            config.denoising_step_list = [700, 600, 500, 0]
        else:
            config.denoising_step_list = [700, 600, 500, 400, 0]

        self.pipeline_manager.config = config
        # Propagate the step list into the underlying pipeline object
        self.pipeline_manager.pipeline.denoising_step_list = config.denoising_step_list

        # ── Load input video ─────────────────────────────────────────────────
        if input_video is not None:
            LOGGER.info(f"[SD2] Loading input video: {input_video}")
            video, _, _ = torchvision.io.read_video(input_video, output_format="TCHW")
            video = video[:num_frames]
            video = rearrange(video, "t c h w -> c t h w")

            # Resize each frame
            c, t, *_ = video.shape
            video = torch.stack(
                [TF.resize(video[:, i], (height, width), antialias=True) for i in range(t)],
                dim=1,
            )
            video = video.float() / 127.5 - 1.0
            input_video_tensor = video.unsqueeze(0).to(dtype=torch.bfloat16).to(self.device)
            LOGGER.info(f"[SD2] Input tensor shape: {input_video_tensor.shape}")
        else:
            input_video_tensor = None

        # ── Load prompts ─────────────────────────────────────────────────────
        dataset = TextDataset(prompt_file)
        prompts = [dataset[0]]

        # ── Infer chunk size & num_chunks ────────────────────────────────────
        chunk_size = 4 * self.pipeline_manager.pipeline.num_frame_per_block
        t_frames = (
            input_video_tensor.shape[2] if input_video_tensor is not None else num_frames
        )
        num_chunks = (t_frames - 1) // chunk_size
        num_steps  = len(self.pipeline_manager.pipeline.denoising_step_list)

        os.makedirs(output_folder, exist_ok=True)

        # ── Run inference ────────────────────────────────────────────────────
        LOGGER.info("[SD2] Starting inference …")
        t0 = time.time()

        torch.set_grad_enabled(False)
        self.pipeline_manager.run_inference(
            input_video_original=input_video_tensor,
            prompts=prompts,
            num_chunks=num_chunks,
            chunk_size=chunk_size,
            noise_scale=noise_scale,
            output_folder=output_folder,
            fps=fps,
            target_fps=target_fps,
            num_steps=num_steps,
        )

        output_path = os.path.join(output_folder, "output_000.mp4")
        LOGGER.info(f"[SD2] Inference done in {time.time() - t0:.1f}s → {output_path}")
        return output_path


# ---------------------------------------------------------------------------
# Public Flask-compatible response helper
# ---------------------------------------------------------------------------

def run_streamdiffusionv2(ctx) -> dict:
    """Run StreamDiffusionV2 inference and return a result dict.

    This is the entry point called from the Flask route handler.

    Args:
        ctx: SimpleNamespace with all fields listed in the module docstring.

    Returns:
        A dict with keys ``"output_video"`` (str path) and ``"elapsed"`` (float).
    """
    config_path = getattr(ctx, "sd2_config_path", _DEFAULT_CONFIG)
    checkpoint_folder = getattr(ctx, "sd2_checkpoint_folder", _DEFAULT_CKPT_FOLDER)

    t0 = time.time()
    output_path = ctx.sd2_service.run(
        input_video=getattr(ctx, "sd2_input_video", None),
        prompt_file=ctx.sd2_prompt_file,
        output_folder=ctx.sd2_output_folder,
        height=getattr(ctx, "sd2_height", 480),
        width=getattr(ctx, "sd2_width", 832),
        fps=getattr(ctx, "sd2_fps", 16),
        step=getattr(ctx, "sd2_step", 2),
        noise_scale=getattr(ctx, "sd2_noise_scale", 0.700),
        num_frames=getattr(ctx, "sd2_num_frames", 81),
        target_fps=getattr(ctx, "sd2_target_fps", None),
    )

    return {
        "output_video": output_path,
        "elapsed": round(time.time() - t0, 2),
    }
