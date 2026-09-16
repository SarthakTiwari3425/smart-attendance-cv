"""Face detection.

Two interchangeable strategies implement the same :class:`FaceDetector`
interface:

``HaarCascadeDetector``
    Viola-Jones cascade of Haar-like features - the classical, CPU-only
    detector shipped with OpenCV.  Used for webcam frames, videos and
    photographs.

``PassthroughDetector``
    Treats the whole image as a single face.  Needed for benchmark datasets
    (ORL/AT&T, Yale, the bundled synthetic set) that are already cropped, and
    for unit tests that must run without cascade files.

Selecting the backend through a factory keeps the rest of the pipeline free
of ``if backend == ...`` branches.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Protocol, Tuple

import cv2
import numpy as np

from .config import DetectionConfig
from .exceptions import DetectionError
from .logger import get_logger

logger = get_logger(__name__)

Box = Tuple[int, int, int, int]


class FaceDetector(Protocol):
    """Interface every detection strategy must satisfy."""

    def detect(self, image: np.ndarray) -> List[Box]:
        """Return bounding boxes ``(x, y, w, h)`` for all faces found."""
        ...


class HaarCascadeDetector:
    """Viola-Jones face detector backed by an OpenCV cascade XML."""

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.cascade = load_cascade(config.cascade)

    def detect(self, image: np.ndarray) -> List[Box]:
        if image is None or image.size == 0:
            raise DetectionError("Cannot detect faces in an empty image")
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        min_size = self.config.min_face_size
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.config.scale_factor,
            minNeighbors=self.config.min_neighbors,
            minSize=(min_size, min_size),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) == 0:
            return []
        # Largest faces first: the front row of a classroom matters most, and
        # the cap protects the per-frame latency budget.
        boxes = sorted(
            (tuple(int(v) for v in face) for face in faces),
            key=lambda b: b[2] * b[3],
            reverse=True,
        )
        return boxes[: self.config.max_faces_per_frame]


class PassthroughDetector:
    """Returns the full image as one face (pre-cropped datasets)."""

    def __init__(self, config: DetectionConfig | None = None):
        self.config = config or DetectionConfig(backend="none")

    def detect(self, image: np.ndarray) -> List[Box]:
        if image is None or image.size == 0:
            raise DetectionError("Cannot detect faces in an empty image")
        height, width = image.shape[:2]
        return [(0, 0, int(width), int(height))]


def load_cascade(name_or_path: str) -> cv2.CascadeClassifier:
    """Load a cascade by absolute path or by name from OpenCV's data folder."""
    candidate = Path(name_or_path)
    path = candidate if candidate.is_file() else Path(cv2.data.haarcascades) / name_or_path
    if not path.is_file():
        raise DetectionError(
            f"Cascade file not found: {name_or_path}. "
            f"Looked in {cv2.data.haarcascades}"
        )
    cascade = cv2.CascadeClassifier(str(path))
    if cascade.empty():
        raise DetectionError(f"Cascade file could not be parsed: {path}")
    logger.debug("Loaded cascade %s", path.name)
    return cascade


def build_detector(config: DetectionConfig) -> FaceDetector:
    """Factory returning the detection strategy named in the configuration."""
    backend = config.backend.lower()
    if backend == "haar":
        return HaarCascadeDetector(config)
    if backend == "none":
        return PassthroughDetector(config)
    raise DetectionError(f"Unknown detection backend: {config.backend!r}")


def draw_boxes(frame: np.ndarray, annotations) -> np.ndarray:
    """Draw labelled rectangles on a copy of *frame* (used by ``--display``)."""
    canvas = frame.copy()
    for box, label, matched in annotations:
        x, y, w, h = box
        color = (0, 170, 0) if matched else (0, 0, 200)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            canvas, label, (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )
    return canvas
