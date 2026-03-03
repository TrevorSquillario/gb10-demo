"""
Frame bus — shared Redis Streams transport layer.

Each microservice publishes or subscribes to named streams on a Redis server.
Frames are JPEG-encoded before storage so the bus is backend-agnostic (the
consumer receives a decoded BGR ndarray regardless of which service produced
the frame).

Stream names
------------
  frames:raw     Raw frames from the capture service
  frames:yolo    Annotated frames from the YOLO detection service
  frames:dwpose  Skeleton-overlay frames from the DWpose service
  frames:depth   Depth-map frames from the Depth Anything service

Key/hash names (non-stream)
---------------------------
  detections:latest   Redis Hash  {track_id: JSON-encoded detection dict}
  crops:{track_id}    Redis String  JPEG bytes for a face thumbnail
  attrs               Redis Hash  {track_id: JSON-encoded DeepFace attributes}
  vlm:latest          Redis String  Latest VLM scene description (plain text)

Environment variables
---------------------
  REDIS_URL       Full Redis URL (default: redis://redis:6379)
  FRAME_BUS_MAXLEN  Ring-buffer depth per stream (default: 10)
  FRAME_JPEG_QUALITY  JPEG quality 1-100 (default: 75)
"""

import os
import time
import msgpack
import numpy as np
import cv2
from redis import Redis

REDIS_URL    = os.environ.get("REDIS_URL", "redis://redis:6379")
_MAXLEN      = int(os.environ.get("FRAME_BUS_MAXLEN", "10"))
_JPEG_QUALITY = int(os.environ.get("FRAME_JPEG_QUALITY", "75"))

STREAMS: dict[str, str] = {
    "raw":    "frames:raw",
    "yolo":   "frames:yolo",
    "dwpose": "frames:dwpose",
    "depth":  "frames:depth",
    "sd":     "frames:sd",
}

VLM_KEY = "vlm:latest"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _encode_frame(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def _decode_frame(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# FrameBus
# ---------------------------------------------------------------------------

class FrameBus:
    """Thin wrapper around Redis Streams for frame publishing/consuming."""

    def __init__(self, url: str = REDIS_URL):
        self._r: Redis = Redis.from_url(url, decode_responses=False)

    # ------------------------------------------------------------------ write

    def publish(
        self,
        stream: str,
        frame: np.ndarray,
        meta: dict | None = None,
    ) -> None:
        """Publish *frame* to *stream*.

        Args:
            stream: Logical stream name (key of :data:`STREAMS`).
            frame:  BGR ndarray.
            meta:   Optional dict; serialised with msgpack and stored in the
                    ``meta`` field so consumers can read it without decoding
                    the full frame.
        """
        key = STREAMS[stream]
        payload: dict = {
            b"frame": _encode_frame(frame),
            b"ts":    str(time.time()).encode(),
        }
        if meta:
            payload[b"meta"] = msgpack.dumps(meta)
        self._r.xadd(key, payload, maxlen=_MAXLEN, approximate=True)

    # ------------------------------------------------------------------ read

    def latest(self, stream: str) -> tuple[np.ndarray | None, dict]:
        """Return the most-recent frame (non-blocking), or ``(None, {})``."""
        key = STREAMS[stream]
        entries = self._r.xrevrange(key, count=1)
        if not entries:
            return None, {}
        _, fields = entries[0]
        frame = _decode_frame(fields[b"frame"])
        meta  = msgpack.loads(fields[b"meta"]) if b"meta" in fields else {}
        return frame, meta

    def subscribe(
        self,
        stream: str,
        block_ms: int = 100,
        start_id: str = "$",
    ):
        """Generator that yields ``(frame, meta)`` as new entries arrive.

        Blocks up to *block_ms* ms waiting for the next entry so consumer
        loops stay tight without busy-waiting.

        Args:
            stream:   Logical stream name.
            block_ms: Redis XREAD block timeout in milliseconds.
            start_id: Redis stream ID to read from. ``"$"`` means only new
                      entries published after this call.  Pass ``"0"`` to
                      replay from the beginning of the ring buffer.
        """
        last_id = start_id
        key     = STREAMS[stream]
        while True:
            results = self._r.xread({key: last_id}, count=1, block=block_ms)
            if not results:
                continue
            _, entries = results[0]
            for entry_id, fields in entries:
                last_id = entry_id
                try:
                    frame = _decode_frame(fields[b"frame"])
                    meta  = msgpack.loads(fields[b"meta"]) if b"meta" in fields else {}
                    yield frame, meta
                except Exception:
                    # Corrupt/partial entry — skip silently
                    continue

    # ---------------------------------------------------------------- helpers

    def vlm_description(self) -> str | None:
        """Return the latest VLM scene description, or ``None`` if unavailable."""
        raw = self._r.get(VLM_KEY)
        return raw.decode() if raw else None

    def redis(self) -> Redis:
        """Expose the underlying Redis client for ad-hoc operations."""
        return self._r
