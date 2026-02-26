"""
YOLO face-detection / tracking frame generator.

The public entry-point is :func:`generate_frames`.  Callers pass a single
:class:`~types.SimpleNamespace` context object (``ctx``) that carries both
static configuration and the mutable state shared with ``app.py``.

Expected ``ctx`` attributes
---------------------------
Config (read-only from this module's perspective):
    model               YOLO model instance
    classes             dict[int, str] — class names
    conf / iou / max_det / DEVICE / USE_CUDA / tracker / track_args / SKIP_FRAMES
    DETECT_STREAM_URL   str | None
    ANALYSIS_INTERVAL   float
    show_fps / show_conf / show_stats  bool
    start_ffmpeg        callable
    stop_ffmpeg         callable

Source info:
    video_source        int | str
    video_source_type   str

Mutable state (updated in place so app.py routes can read live values):
    frame_counter       int   (incremented each frame)
    latest_frame        np.ndarray | None
    last_deepface_time  float
    tracked_people      set[int]
    person_last_seen    defaultdict[int, int]
    current_people      set[int]
    person_crops        dict[int, bytes]
    crops_lock          threading.Lock
    frame_lock          threading.Lock
    deepface_service    DeepFaceService
"""

import os
import queue
import threading
import time

import cv2
import numpy as np
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator, colors

from services.video_capture import open_video_capture

# Inference frames waiting to be JPEG-encoded.  A shallow queue keeps latency
# low; drop-oldest means the encoder always sees the freshest frame.
_QUEUE_MAXSIZE = 4
# Sentinel placed in the queue by the inference thread to signal it has exited.
_SENTINEL = None

# Re-encode each tracked person's thumbnail only every N inference frames.
# Larger values reduce per-frame CPU/alloc overhead; 5 is imperceptible in the UI.
_CROP_ENCODE_INTERVAL = int(os.environ.get("CROP_ENCODE_INTERVAL", "5"))

# JPEG quality for both the MJPEG stream and person-crop thumbnails (1-100).
# 75 is visually near-lossless for video and ~half the byte-size of the
# OpenCV default (95), reducing encoder CPU and network bandwidth.
_MJPEG_QUALITY = int(os.environ.get("MJPEG_QUALITY", "75"))


def _queue_put(q: queue.Queue, item) -> None:
    """Non-blocking put that drops the oldest entry when the queue is full."""
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()   # discard oldest frame
            except queue.Empty:
                pass             # another thread raced us — retry


def _yolo_worker(
    ctx,
    cap: cv2.VideoCapture,
    frame_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """Inference thread: GPU inference + overlay → push annotated frame into queue."""
    fps_counter, fps_timer, fps_display = 0, time.time(), 0
    # Track the last frame index at which each person's thumbnail was JPEG-encoded.
    _crop_encode_frame: dict[int, int] = {}

    try:
        while not stop_event.is_set():
            success, im = cap.read()
            if not success:
                if ctx.video_source_type == "file":
                    LOGGER.info("[YOLO] Video file ended, looping...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    LOGGER.warning(f"[YOLO] {ctx.video_source_type} stream failed")
                    break

            ctx.frame_counter += 1
            ctx.current_people = set()
            snapshot_detections: list[tuple[int, int, int, int, int]] = []

            # ── Store raw (pre-annotation) frame for other services ───────────────────────
            # One copy serves latest_raw_frame, DeepFace submission, AND clean
            # crop thumbnails — never more than one full-frame allocation per tick.
            raw_im = im.copy()
            with ctx.frame_lock:
                ctx.latest_raw_frame = raw_im

            # ── GPU inference ────────────────────────────────────────────────────────────
            results = ctx.model.track(
                im,
                conf=ctx.conf,
                iou=ctx.iou,
                max_det=ctx.max_det,
                device=ctx.DEVICE,
                half=ctx.USE_CUDA,
                tracker=ctx.tracker,
                vid_stride=ctx.SKIP_FRAMES,
                **ctx.track_args,
            )

            annotator = Annotator(im)
            detections = results[0].boxes.data if results[0].boxes is not None else []

            for det in detections:
                track = det.tolist()
                if len(track) < 6:
                    continue

                x1, y1, x2, y2 = map(int, track[:4])
                # Track format: [x1,y1,x2,y2,track_id,score,class]
                if len(track) == 6:
                    conf_score = float(track[4])
                    class_id  = int(track[5])
                    track_id  = -1
                else:
                    track_id  = int(track[4])
                    conf_score = float(track[5])
                    class_id  = int(track[6])

                if track_id >= 0:
                    ctx.tracked_people.add(track_id)
                    ctx.person_last_seen[track_id] = ctx.frame_counter
                    ctx.current_people.add(track_id)
                    snapshot_detections.append((track_id, x1, y1, x2, y2))

                    # Store latest JPEG crop for the UI panel (throttled).
                    # Crop from raw_im so thumbnails show clean faces without
                    # bounding-box annotation painted on them.
                    h_im, w_im = raw_im.shape[:2]
                    cx1, cy1 = max(0, x1), max(0, y1)
                    cx2, cy2 = min(w_im, x2), min(h_im, y2)
                    last_enc = _crop_encode_frame.get(track_id, -_CROP_ENCODE_INTERVAL)
                    if (
                        (cx2 - cx1) > 10
                        and (cy2 - cy1) > 10
                        and (ctx.frame_counter - last_enc) >= _CROP_ENCODE_INTERVAL
                    ):
                        crop_bgr = raw_im[cy1:cy2, cx1:cx2]
                        ok, buf = cv2.imencode(
                            ".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, _MJPEG_QUALITY]
                        )
                        if ok:
                            _crop_encode_frame[track_id] = ctx.frame_counter
                            with ctx.crops_lock:
                                ctx.person_crops[track_id] = buf.tobytes()

                    if ctx.frame_counter % 30 == 0:
                        LOGGER.info(
                            f"[YOLO] Tracking Person ID: {track_id} at [{x1},{y1},{x2},{y2}]"
                        )

                color = colors(track_id, True)
                annotator.get_txt_color(color)
                class_name = ctx.classes.get(class_id, "person")

                label = f"{class_name}"
                if track_id >= 0:
                    label = f"Person ID {track_id}"
                if ctx.show_conf:
                    label += f" ({conf_score:.2f})"

                annotator.box_label([x1, y1, x2, y2], label=label, color=color)

            # ── DeepFace background analysis ───────────────────────────────────────
            # Pass raw_im (pre-annotation) so the face crops are clean.
            # raw_im is already a copy so submit() stores it directly.
            now = time.time()
            if now - ctx.last_deepface_time >= ctx.ANALYSIS_INTERVAL and snapshot_detections:
                ctx.deepface_service.submit(raw_im, snapshot_detections)
                ctx.last_deepface_time = now

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
                h_frame, w_frame = im.shape[:2]
                fx = w_frame - tw - 15
                cv2.rectangle(im, (fx - 5, 25 - th - 5), (fx + tw + 5, 25 + bl), (255, 255, 255), -1)
                cv2.putText(im, fps_text, (fx, 25), fps_font, fps_scale, (104, 31, 17), fps_thickness, cv2.LINE_AA)

            # ── Stats overlay ─────────────────────────────────────────────────────
            if ctx.show_stats:
                stats_y = 50
                stats_font = cv2.FONT_HERSHEY_SIMPLEX
                stats_scale, stats_thickness = 0.5, 1

                stats_text = f"Current: {len(ctx.current_people)}"
                (tw, th), bl = cv2.getTextSize(stats_text, stats_font, stats_scale, stats_thickness)
                cv2.rectangle(im, (10 - 5, stats_y - th - 5), (10 + tw + 5, stats_y + bl), (255, 255, 255), -1)
                cv2.putText(im, stats_text, (10, stats_y), stats_font, stats_scale, (104, 31, 17), stats_thickness, cv2.LINE_AA)

                stats_y += 25
                total_text = f"Total: {len(ctx.tracked_people)}"
                (tw, th), bl = cv2.getTextSize(total_text, stats_font, stats_scale, stats_thickness)
                cv2.rectangle(im, (10 - 5, stats_y - th - 5), (10 + tw + 5, stats_y + bl), (255, 255, 255), -1)
                cv2.putText(im, total_text, (10, stats_y), stats_font, stats_scale, (104, 31, 17), stats_thickness, cv2.LINE_AA)

            # ── Share latest frame with snapshot routes ────────────────────────
            # No copy needed: cap.read() allocates a fresh buffer each call so
            # `im` is not reused, and nothing writes to it after this point.
            with ctx.frame_lock:
                ctx.latest_frame = im

            # ── Push to encoder queue (drop-oldest when full) ─────────────────
            _queue_put(frame_queue, im)

    finally:
        cap.release()
        frame_queue.put(_SENTINEL)  # wake up the encoder so it can exit cleanly


def generate_frames(ctx):
    """Yield MJPEG-boundary chunks with YOLO face detections and BoT-SORT tracking.

    Starts a background inference thread that pushes annotated frames into a
    queue; this function pulls from the queue, JPEG-encodes on the CPU, and
    yields multipart HTTP chunks.

    Args:
        ctx: A :class:`~types.SimpleNamespace` carrying all configuration and
             shared mutable state (see module docstring for field list).
    """
    if ctx.DETECT_STREAM_URL:
        ctx.start_ffmpeg()
        time.sleep(2)

    cap = open_video_capture(ctx.video_source, ctx.video_source_type, label="YOLO")
    if cap is None:
        ctx.stop_ffmpeg()
        return

    frame_queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_yolo_worker,
        args=(ctx, cap, frame_queue, stop_event),
        name="yolo-inference",
        daemon=True,
    )
    thread.start()

    try:
        while True:
            try:
                frame = frame_queue.get(timeout=5.0)
            except queue.Empty:
                LOGGER.warning("[YOLO] Encoder timed out waiting for frame — stream may have stalled")
                break

            if frame is _SENTINEL:
                break  # inference thread signalled it has exited

            ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _MJPEG_QUALITY])
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
        ctx.stop_ffmpeg()
        LOGGER.info(f"[YOLO] Session ended. Total unique people tracked: {len(ctx.tracked_people)}")
        LOGGER.info(f"[YOLO] Person IDs: {sorted(ctx.tracked_people)}")
