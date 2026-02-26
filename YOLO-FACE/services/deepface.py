"""
DeepFace facial attribute analysis service.

Runs in a background worker thread. Every ANALYSIS_INTERVAL seconds the main loop
submits a (frame, detections) snapshot via submit(). The worker crops each tracked
person, calls DeepFace.analyze(), and stores the results keyed by track_id so that
the annotation loop can overlay attributes (age, gender, emotion, race) in real time.
"""

import os
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np

from ultralytics.utils import LOGGER

# Minimum crop dimensions to skip tiny/distant detections
MIN_CROP_PX = 30

# Minimum face-detection confidence from extract_faces() to proceed with analysis.
# Faces below this score are treated as uncertain and skipped.
CONFIDENCE_THRESHOLD = float(os.environ.get("DEEPFACE_CONFIDENCE", "0.9"))

# Backend used by extract_faces() for the confidence pre-check.
# retinaface is accurate but heavier; opencv / ssd are faster alternatives.
DETECTOR_BACKEND = os.environ.get("DEEPFACE_DETECTOR", "retinaface")

# DeepFace actions to run on every crop
DEEPFACE_ACTIONS = ["age", "gender", "emotion", "race"]


class DeepFaceService:
    """Background service that runs DeepFace attribute analysis on person crops.

    Usage:
        svc = DeepFaceService(interval=15.0)
        svc.start()
        # Inside frame loop every N seconds:
        svc.submit(frame, [(track_id, x1, y1, x2, y2), ...])
        # To read results:
        attrs = svc.get_attributes(track_id)   # dict or None
        all_attrs = svc.get_all_attributes()   # {track_id: dict}
        svc.stop()
    """

    def __init__(self, interval: float = 15.0):
        self.interval = interval
        self._results: dict[int, dict] = {}
        self._results_lock = threading.Lock()
        # Queue capacity 1 — we only care about the latest snapshot
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._worker, name="deepface-worker", daemon=True)
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background analysis thread."""
        if self._running:
            return
        self._running = True
        self._thread.start()
        LOGGER.info("DeepFace analysis service started (interval=%ss)", self.interval)

    def stop(self) -> None:
        """Signal the worker to stop and join."""
        self._running = False
        try:
            self._queue.put_nowait(None)  # Unblock blocking get
        except queue.Full:
            pass
        self._thread.join(timeout=5)
        LOGGER.info("DeepFace analysis service stopped")

    def submit(self, frame: np.ndarray, detections: list[tuple[int, int, int, int, int]]) -> None:
        """Submit a snapshot for analysis.

        Args:
            frame: BGR frame from OpenCV.
            detections: list of (track_id, x1, y1, x2, y2) for each tracked person.

        The queue holds at most one item; a new submit() replaces any pending
        unprocessed snapshot so analysis always uses the most recent frame.
        """
        if not self._running or not detections:
            return
        # Caller must pass a pre-copied frame (yolo worker passes raw_im which
        # is already its own allocation); storing a reference avoids a redundant
        # full-frame copy in this hot path.
        snapshot = (frame, list(detections))
        # Replace any pending snapshot with the freshest one
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(snapshot)
        except queue.Full:
            pass

    def get_attributes(self, track_id: int) -> Optional[dict]:
        """Return the latest DeepFace attributes for a track ID, or None."""
        with self._results_lock:
            return self._results.get(track_id)

    def get_all_attributes(self) -> dict[int, dict]:
        """Return a copy of all stored attributes keyed by track_id."""
        with self._results_lock:
            return dict(self._results)

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        try:
            from deepface import DeepFace  # lazy import — avoids TF startup at module load
        except ImportError:
            LOGGER.error("deepface is not installed. Run: pip install deepface")
            return

        LOGGER.info("DeepFace worker ready")

        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:
                break

            frame, detections = item
            h, w = frame.shape[:2]

            for track_id, x1, y1, x2, y2 in detections:
                # Clamp to frame bounds
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(w, int(x2))
                y2 = min(h, int(y2))

                crop_w = x2 - x1
                crop_h = y2 - y1
                if crop_w < MIN_CROP_PX or crop_h < MIN_CROP_PX:
                    LOGGER.debug(
                        "Skipping Person %d crop (%dx%d) — too small", track_id, crop_w, crop_h
                    )
                    continue

                crop = frame[y1:y2, x1:x2]

                try:
                    t0 = time.time()

                    # ── Step 1: confidence pre-check ──────────────────────────────
                    # extract_faces() returns face regions with a confidence score.
                    # We pick the highest-confidence face in the crop; if it falls
                    # below CONFIDENCE_THRESHOLD we skip this person entirely so only
                    # clearly-detected faces appear in the side panel.
                    faces = DeepFace.extract_faces(
                        img_path=crop,
                        detector_backend=DETECTOR_BACKEND,
                        enforce_detection=False,  # return empty list instead of raising
                        align=True,
                    )

                    if not faces:
                        LOGGER.debug("No faces extracted from crop for Person %d", track_id)
                        continue

                    best_face = max(faces, key=lambda f: f.get("confidence", 0.0))
                    best_confidence = float(best_face.get("confidence", 0.0))

                    if best_confidence < CONFIDENCE_THRESHOLD:
                        LOGGER.debug(
                            "Skipping Person %d — face confidence %.2f < %.2f",
                            track_id, best_confidence, CONFIDENCE_THRESHOLD,
                        )
                        continue

                    # ── Step 2: full attribute analysis ───────────────────────────
                    # Reuse the aligned face image from extract_faces so that
                    # analyze() skips a second expensive detector pass on the
                    # full crop.  best_face["face"] is float32 RGB [0,1].
                    face_f32 = best_face.get("face")
                    if face_f32 is not None and face_f32.size > 0:
                        aligned_bgr = cv2.cvtColor(
                            (face_f32 * 255).clip(0, 255).astype(np.uint8),
                            cv2.COLOR_RGB2BGR,
                        )
                        analyze_input = aligned_bgr
                    else:
                        analyze_input = crop

                    results = DeepFace.analyze(
                        img_path=analyze_input,
                        actions=DEEPFACE_ACTIONS,
                        detector_backend=DETECTOR_BACKEND,
                        enforce_detection=False,
                        silent=True,
                    )

                    elapsed = time.time() - t0

                    if not results:
                        continue

                    r = results[0]
                    attrs = {
                        "age": r.get("age"),
                        "gender": r.get("dominant_gender"),
                        "emotion": r.get("dominant_emotion"),
                        "race": r.get("dominant_race"),
                        "confidence": round(best_confidence, 4),
                        "analyzed_at": time.time(),
                    }
                    with self._results_lock:
                        self._results[track_id] = attrs

                    LOGGER.info(
                        "DeepFace Person %d → age=%s gender=%s emotion=%s race=%s conf=%.2f (%.2fs)",
                        track_id,
                        attrs["age"],
                        attrs["gender"],
                        attrs["emotion"],
                        attrs["race"],
                        best_confidence,
                        elapsed,
                    )

                except Exception as exc:
                    LOGGER.warning("DeepFace analysis failed for Person %d: %s", track_id, exc)
