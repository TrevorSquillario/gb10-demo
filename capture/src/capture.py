"""
Capture microservice — reads a video source and publishes raw frames to the
Redis frame bus (stream ``frames:raw``).

Keeping capture in its own process means:
  * No GIL competition with YOLO / DWpose / Depth inference workers.
  * A single source of truth for the raw frame; all downstream services
    subscribe rather than each opening their own VideoCapture handle.
  * Reconnect logic lives in one place.

Environment variables
---------------------
  VIDEO_SOURCE        Device index (int), file path, or stream URL.
                      Default: ``"0"`` (first USB webcam).
  CAPTURE_FPS         Target publish rate in frames per second.
                      Default: ``30``.
  REDIS_URL           Redis connection string.
                      Default: ``redis://redis:6379``.
"""

import os
import sys
import time

import cv2
from logging_config import configure_logging, get_logger
from frame_bus import FrameBus, REDIS_URL


# Ensure logging is configured to stdout for container environments
configure_logging()

# Module logger
LOGGER = get_logger(__name__)

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


# ---------------------------------------------------------------------------
# Main capture loop
# ---------------------------------------------------------------------------

def _parse_video_source(raw: str) -> tuple:
    """Return (video_source, video_source_type) from a raw string/int."""
    try:
        return int(raw), "webcam"
    except ValueError:
        pass
    if raw.startswith("rtsp://"):
        return raw, "rtsp"
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw, "http"
    return raw, "file"


def main() -> None:
    raw_source = os.environ.get("VIDEO_SOURCE", "0")
    video_source, video_source_type = _parse_video_source(raw_source)
    target_fps = float(os.environ.get("CAPTURE_FPS", "30"))
    interval   = 1.0 / target_fps

    LOGGER.info(
        f"[Capture] source={video_source!r} type={video_source_type} "
        f"target_fps={target_fps} redis={REDIS_URL}"
    )

    bus = FrameBus()

    while True:
        cap = open_video_capture(video_source, video_source_type, label="Capture")
        if cap is None:
            LOGGER.error("[Capture] Could not open video source — retrying in 5 s")
            time.sleep(5)
            continue

        LOGGER.info("[Capture] Publishing frames to frames:raw")
        try:
            while True:
                t0 = time.monotonic()
                ok, frame = cap.read()
                if not ok:
                    if video_source_type == "file":
                        LOGGER.info("[Capture] File ended, looping")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    LOGGER.warning("[Capture] Stream read failed — reconnecting")
                    break

                bus.publish("raw", frame)

                elapsed = time.monotonic() - t0
                sleep   = interval - elapsed
                if sleep > 0:
                    time.sleep(sleep)
        finally:
            cap.release()


if __name__ == "__main__":
    main()
