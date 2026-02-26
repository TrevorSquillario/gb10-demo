"""
DWpose skeleton-overlay frame generator — Producer-Consumer pattern.

Architecture
------------
Inference thread  (GPU)
    Reads frames from cap → runs DWpose skeleton detection → draws FPS overlay
    → pushes rendered BGR canvas into a bounded Queue.

Encoding thread   (CPU, the Flask generator)
    Pulls canvas frames from the Queue → cv2.imencode JPEG → yields MJPEG
    boundary chunks to the HTTP response.

The Queue is bounded (``_QUEUE_MAXSIZE``) and uses a drop-oldest strategy so the
inference thread never blocks and the HTTP consumer always receives the most
recent frame.

Expected ``ctx`` attributes
---------------------------
    video_source        int | str
    video_source_type   str
    DETECT_STREAM_URL   str | None
    dwpose_detector     DWposeDetector instance
    show_fps            bool
    start_ffmpeg        callable() → None
"""

import queue
import threading
import time

import cv2
from ultralytics.utils import LOGGER

from services.video_capture import open_video_capture

_QUEUE_MAXSIZE = 4
_SENTINEL = None


def _queue_put(q: queue.Queue, item) -> None:
    """Non-blocking put that drops the oldest entry when the queue is full."""
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass


def _dwpose_worker(
    ctx,
    cap: cv2.VideoCapture,
    frame_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """Inference thread: DWpose skeleton detection + overlay → push canvas into queue."""
    fps_counter, fps_timer, fps_display = 0, time.time(), 0

    try:
        while not stop_event.is_set():
            success, frame = cap.read()
            if not success:
                if ctx.video_source_type == "file":
                    LOGGER.info("[DWpose] Video file ended, looping...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    LOGGER.warning(f"[DWpose] {ctx.video_source_type} stream failed")
                    break

            # ── GPU inference ── returns a rendered H×W×3 BGR canvas ──────────
            pose_canvas = ctx.dwpose_detector(
                image_np_hwc=frame,
                show_body=True,
                show_face=True,
                show_hands=True,
            )

            # ── FPS overlay (measured at inference rate) ─────────────────────────
            if ctx.show_fps:
                fps_counter += 1
                if time.time() - fps_timer >= 1.0:
                    fps_display = fps_counter
                    fps_counter = 0
                    fps_timer = time.time()

                fps_text = f"FPS: {fps_display}"
                fps_font = cv2.FONT_HERSHEY_SIMPLEX
                fps_scale, fps_thickness = 1.0, 2
                (tw, th), bl = cv2.getTextSize(fps_text, fps_font, fps_scale, fps_thickness)
                h_frame, w_frame = pose_canvas.shape[:2]
                fx = w_frame - tw - 15
                cv2.rectangle(
                    pose_canvas, (fx - 5, 25 - th - 5), (fx + tw + 5, 25 + bl), (255, 255, 255), -1
                )
                cv2.putText(
                    pose_canvas, fps_text, (fx, 25), fps_font, fps_scale,
                    (104, 31, 17), fps_thickness, cv2.LINE_AA,
                )

            # ── Push to encoder queue (drop-oldest when full) ─────────────────
            _queue_put(frame_queue, pose_canvas)
            # ── Forward to any downstream consumer (e.g. SD ControlNet) ─────
            if hasattr(ctx, "pose_queue") and ctx.pose_queue is not None:
                _queue_put(ctx.pose_queue, pose_canvas)
    finally:
        cap.release()
        frame_queue.put(_SENTINEL)


def start_dwpose_background(ctx, stop_event: threading.Event) -> "threading.Thread | None":
    """Start a background DWpose worker that pushes pose canvases into ``ctx.pose_queue``.

    Unlike :func:`generate_dwpose`, this function does **not** yield MJPEG frames —
    it is intended for consumers such as the SD ControlNet pipeline that only need the
    pose canvas from their own video capture, without a second HTTP client on
    ``/dwpose_feed``.

    The worker runs in a daemon thread and will stop when *stop_event* is set or the
    video source is exhausted.

    Args:
        ctx: A SimpleNamespace with at minimum:
             ``video_source``, ``video_source_type``, ``DETECT_STREAM_URL``,
             ``dwpose_detector``, ``show_fps``, ``start_ffmpeg``, ``pose_queue``.
        stop_event: Signal the thread to stop by setting this event.

    Returns:
        The started :class:`~threading.Thread`, or ``None`` if the video source
        could not be opened.
    """
    if ctx.DETECT_STREAM_URL:
        ctx.start_ffmpeg()
        time.sleep(2)

    cap = open_video_capture(ctx.video_source, ctx.video_source_type, label="DWpose-SD")
    if cap is None:
        return None

    # A small sink queue is required by _dwpose_worker but discarded here —
    # the relevant output goes directly to ctx.pose_queue.
    _sink: queue.Queue = queue.Queue(maxsize=2)

    thread = threading.Thread(
        target=_dwpose_worker,
        args=(ctx, cap, _sink, stop_event),
        name="dwpose-sd-bg",
        daemon=True,
    )
    thread.start()
    LOGGER.info("[DWpose-SD] Background worker started.")
    return thread


def generate_dwpose(ctx):
    """Yield MJPEG-boundary chunks with DWpose skeleton overlays.

    Starts a background inference thread and encodes frames from its output queue.

    Args:
        ctx: A :class:`~types.SimpleNamespace` carrying configuration and
             dependencies (see module docstring for field list).

    Yields nothing and returns immediately when ``ctx.dwpose_detector`` is ``None``.
    """
    if ctx.dwpose_detector is None:
        LOGGER.error("[DWpose] Detector not loaded; cannot stream DWpose feed.")
        return

    if ctx.DETECT_STREAM_URL:
        ctx.start_ffmpeg()
        time.sleep(2)

    cap = open_video_capture(ctx.video_source, ctx.video_source_type, label="DWpose")
    if cap is None:
        return

    frame_queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_dwpose_worker,
        args=(ctx, cap, frame_queue, stop_event),
        name="dwpose-inference",
        daemon=True,
    )
    thread.start()

    try:
        while True:
            try:
                frame = frame_queue.get(timeout=5.0)
            except queue.Empty:
                LOGGER.warning("[DWpose] Encoder timed out waiting for frame — stream may have stalled")
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
        thread.join(timeout=3.0)
