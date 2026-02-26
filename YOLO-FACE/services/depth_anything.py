"""
Depth Anything TensorRT frame generator — Producer-Consumer pattern.

Architecture
------------
Inference thread  (GPU via TensorRT + torch.cuda.Stream)
    Reads frames from cap → runs depth estimation asynchronously on its own
    CUDA stream → draws FPS overlay → pushes the colourised depth map (BGR
    ndarray) into a bounded Queue.

Encoding thread   (CPU, the Flask generator)
    Pulls depth-map frames from the Queue → cv2.imencode JPEG → yields MJPEG
    boundary chunks to the HTTP response.

The Queue is bounded (``_QUEUE_MAXSIZE``) and uses a drop-oldest strategy so the
inference thread never blocks and the HTTP consumer always receives the most
recent frame.

Expected ``ctx`` attributes
---------------------------
    video_source        int | str
    video_source_type   str
    DETECT_STREAM_URL   str | None
    depth_engine        DepthAnythingEngine instance (or None)
    depth_grayscale     bool
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


def _depth_worker(
    ctx,
    cap: cv2.VideoCapture,
    frame_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """Inference thread: TRT depth estimation + overlay → push depth map into queue.

    Uses the engine's existing ``torch.cuda.Stream`` for async GPU execution so
    cap.read() CPU time overlaps with GPU inference.
    """
    fps_counter, fps_timer, fps_display = 0, time.time(), 0

    try:
        while not stop_event.is_set():
            success, frame = cap.read()
            if not success:
                if ctx.video_source_type == "file":
                    LOGGER.info("[Depth] Video file ended, looping...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    LOGGER.warning(f"[Depth] {ctx.video_source_type} stream failed")
                    break

            # ── GPU inference (TensorRT, async via engine's cuda.Stream) ──────
            depth_frame = ctx.depth_engine(frame, grayscale=ctx.depth_grayscale)

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
                h_frame, w_frame = depth_frame.shape[:2]
                fx = w_frame - tw - 15
                cv2.rectangle(
                    depth_frame, (fx - 5, 25 - th - 5), (fx + tw + 5, 25 + bl), (255, 255, 255), -1
                )
                cv2.putText(
                    depth_frame, fps_text, (fx, 25), fps_font, fps_scale,
                    (104, 31, 17), fps_thickness, cv2.LINE_AA,
                )

            # ── Push to encoder queue (drop-oldest when full) ─────────────────
            _queue_put(frame_queue, depth_frame)

    finally:
        cap.release()
        frame_queue.put(_SENTINEL)


def generate_depth(ctx):
    """Yield MJPEG-boundary chunks with Depth Anything depth-map overlays.

    Starts a background TensorRT inference thread and encodes frames from its
    output queue.

    Args:
        ctx: A :class:`~types.SimpleNamespace` carrying configuration and
             dependencies (see module docstring for field list).

    Yields nothing and returns immediately when ``ctx.depth_engine`` is ``None``.
    """
    if ctx.depth_engine is None:
        LOGGER.error("[Depth] Engine not loaded; cannot stream depth feed.")
        return

    if ctx.DETECT_STREAM_URL:
        ctx.start_ffmpeg()
        time.sleep(2)

    cap = open_video_capture(ctx.video_source, ctx.video_source_type, label="Depth")
    if cap is None:
        return

    frame_queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_depth_worker,
        args=(ctx, cap, frame_queue, stop_event),
        name="depth-inference",
        daemon=True,
    )
    thread.start()

    try:
        while True:
            try:
                frame = frame_queue.get(timeout=5.0)
            except queue.Empty:
                LOGGER.warning("[Depth] Encoder timed out waiting for frame — stream may have stalled")
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
