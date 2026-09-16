"""Attendance session orchestration.

This is the module that turns per-frame recognitions into attendance rows.
Two rules protect the record from the noise inherent in frame-by-frame
classification:

**Temporal voting** - a student is only marked after *N* accepted matches
(``consecutive_hits_required``).  A single lucky frame is never enough, which
is the cheapest available defence against a false positive marking the wrong
person present.

**Idempotent marking** - the database enforces one row per
(session, student), so a student who stays in frame for a thousand frames
still produces exactly one attendance record.

Students recognised after ``late_after_minutes`` from the session start are
recorded with status ``LATE`` rather than ``PRESENT``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from .config import AppConfig
from .database import AttendanceDatabase
from .detector import FaceDetector, draw_boxes
from .logger import get_logger
from .preprocessing import FacePreprocessor
from .recognizer import FaceRecognizer, Prediction
from .video_source import BaseSource

logger = get_logger(__name__)


@dataclass
class SessionStats:
    """Everything the CLI needs to summarise a finished session."""

    session_id: str
    frames_processed: int = 0
    faces_detected: int = 0
    accepted: int = 0
    rejected: int = 0
    marked: List[str] = field(default_factory=list)
    already_present: int = 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "session_id": self.session_id,
            "frames_processed": self.frames_processed,
            "faces_detected": self.faces_detected,
            "accepted_matches": self.accepted,
            "rejected_matches": self.rejected,
            "students_marked": self.marked,
        }


class AttendanceService:
    """Runs a recognition session end to end."""

    def __init__(
        self,
        config: AppConfig,
        database: AttendanceDatabase,
        detector: FaceDetector,
        preprocessor: FacePreprocessor,
        recognizer: FaceRecognizer,
    ):
        self.config = config
        self.db = database
        self.detector = detector
        self.preprocessor = preprocessor
        self.recognizer = recognizer
        self._hits: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    def process_frame(
        self, frame: np.ndarray, session_id: str, frame_index: int, stats: SessionStats
    ) -> List[tuple]:
        """Detect, recognise and (when confirmed) mark everyone in one frame.

        Returns annotation tuples ``(box, label, matched)`` for optional
        on-screen drawing.
        """
        annotations: List[tuple] = []
        boxes = self.detector.detect(frame)
        stats.faces_detected += len(boxes)

        for box in boxes:
            try:
                chip = self.preprocessor.process(frame, box)
            except ValueError as exc:
                logger.debug("Skipping face at %s: %s", box, exc)
                continue

            prediction = self.recognizer.predict(chip)
            if prediction.accepted and prediction.student_id:
                stats.accepted += 1
                self._register_hit(prediction, session_id, frame_index, stats)
            else:
                stats.rejected += 1
                self.db.log_recognition(
                    session_id, None, prediction.distance, "REJECTED", frame_index
                )
            annotations.append((box, prediction.label(), prediction.accepted))

        stats.frames_processed += 1
        return annotations

    def _register_hit(
        self,
        prediction: Prediction,
        session_id: str,
        frame_index: int,
        stats: SessionStats,
    ) -> None:
        """Apply temporal voting and mark the student once confirmed."""
        student_id = prediction.student_id or ""
        self.db.log_recognition(
            session_id, student_id, prediction.distance, "ACCEPTED", frame_index
        )
        if self.db.is_marked(session_id, student_id):
            stats.already_present += 1
            return

        self._hits[student_id] += 1
        required = self.config.attendance.consecutive_hits_required
        if self._hits[student_id] < required:
            logger.debug(
                "%s: %d/%d confirmations", student_id, self._hits[student_id], required
            )
            return

        status = self._status_for(session_id)
        if self.db.mark_attendance(
            session_id, student_id, prediction.confidence, status
        ):
            stats.marked.append(student_id)
            logger.info(
                "Marked %s (%s) as %s - confidence %.1f",
                student_id, prediction.name or "?", status, prediction.confidence,
            )

    def _status_for(self, session_id: str) -> str:
        """PRESENT, or LATE once the grace period has elapsed."""
        started = self.db.get_session_start(session_id)
        if started is None:
            return "PRESENT"
        elapsed_minutes = (datetime.now() - started).total_seconds() / 60.0
        return (
            "LATE"
            if elapsed_minutes > self.config.attendance.late_after_minutes
            else "PRESENT"
        )

    # ------------------------------------------------------------------
    def run_session(
        self,
        source: BaseSource,
        session_id: str,
        course_code: str = "CV-COURSE",
        display: bool = False,
        source_label: str = "",
    ) -> SessionStats:
        """Consume every frame from *source*, marking attendance as it goes."""
        self.db.start_session(session_id, course_code, source_label or str(source))
        stats = SessionStats(session_id=session_id)
        self._hits.clear()

        window = f"Attendance - {session_id} (press q to quit)"
        try:
            for frame_index, frame in source.frames():
                annotations = self.process_frame(frame, session_id, frame_index, stats)
                if display:
                    import cv2

                    cv2.imshow(window, draw_boxes(frame, annotations))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("Stopped by user")
                        break
        except KeyboardInterrupt:
            logger.info("Interrupted - saving what has been recorded so far")
        finally:
            if display:
                import cv2

                cv2.destroyAllWindows()
            source.release()
            self.db.end_session(session_id)

        logger.info(
            "Session %s finished: %d frames, %d faces, %d marked",
            session_id, stats.frames_processed, stats.faces_detected, len(stats.marked),
        )
        return stats

    # ------------------------------------------------------------------
    def identify_image(self, frame: np.ndarray) -> List[Prediction]:
        """Recognise every face in a single image without touching the DB."""
        results: List[Prediction] = []
        for box in self.detector.detect(frame):
            chip = self.preprocessor.process(frame, box)
            results.append(self.recognizer.predict(chip))
        return results


def default_session_id(course_code: str, when: Optional[datetime] = None) -> str:
    """Build a readable, collision-free session id such as ``CV101-20260916-0930``."""
    moment = when or datetime.now()
    return f"{course_code}-{moment:%Y%m%d-%H%M}"
