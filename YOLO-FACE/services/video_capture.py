"""
Centralised helper for opening a cv2.VideoCapture with retries.

Each generator (YOLO, DWpose, Depth) shares this single entry point so that
backend selection / retry logic lives in exactly one place.
"""

import os
import time

import cv2
from ultralytics.utils import LOGGER


def open_video_capture(
    video_source,
    video_source_type: str,
    label: str = "",
    max_retries: int | None = None,
) -> cv2.VideoCapture | None:
    """Open a cv2.VideoCapture with retries, choosing backend by source type.

    Args:
        video_source:      Integer device index, file path, or URL.
        video_source_type: One of ``"webcam"``, ``"rtsp"``, ``"ffmpeg_rtsp"``,
                           ``"http"``, ``"file"``.
        label:             Short tag prepended to log messages (e.g. ``"YOLO"``).
        max_retries:       How many attempts to make before giving up.
                           Defaults to 15 for streaming sources and 3 for files.

    Returns:
        An opened :class:`cv2.VideoCapture` instance, or ``None`` if every
        attempt failed.
    """
    prefix = f"[{label}] " if label else ""

    if max_retries is None:
        max_retries = 15 if video_source_type in ("rtsp", "ffmpeg_rtsp", "http") else 3

    cap: cv2.VideoCapture | None = None

    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            time.sleep(1.0)

        LOGGER.info(f"{prefix}Opening video source (attempt {attempt}/{max_retries})")

        if video_source_type in ("rtsp", "ffmpeg_rtsp"):
            cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)

        elif video_source_type == "webcam":
            if os.name == "posix":
                try:
                    cap = cv2.VideoCapture(video_source, cv2.CAP_V4L2)
                except Exception:
                    cap = cv2.VideoCapture(video_source)
            else:
                cap = cv2.VideoCapture(video_source)

        elif video_source_type == "http":
            cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap = cv2.VideoCapture(video_source)

        else:  # "file" or anything else
            cap = cv2.VideoCapture(video_source)

        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            LOGGER.info(f"{prefix}Video source opened: {width}x{height} @ {fps:.1f} FPS")
            return cap

        # Release the unusable capture and try again
        try:
            cap.release()
        except Exception:
            pass
        cap = None
        LOGGER.warning(
            f"{prefix}Video source not ready (attempt {attempt}/{max_retries}), retrying..."
        )

    LOGGER.error(f"{prefix}Failed to open video source after {max_retries} attempts")
    return None
