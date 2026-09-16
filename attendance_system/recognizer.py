"""Face recognition models.

Three classical recognisers from ``cv2.face`` are supported behind one
interface:

=============  ======================================================
Algorithm      How it works
=============  ======================================================
``lbph``       Local Binary Pattern histograms compared per grid cell.
               Robust to illumination, trains incrementally, and works
               with few samples - the default.
``eigen``      PCA on the training set; projects faces onto the top
               eigenvectors ("eigenfaces") and compares in that space.
``fisher``     LDA on top of PCA; maximises between-class scatter and
               needs at least two students.
=============  ======================================================

All three return a *distance* (lower = more similar), so a probe is accepted
only when ``distance <= distance_threshold``.  This explicit rejection
threshold is what lets the system answer "unknown person" instead of forcing
every face onto the nearest enrolled student.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import RecognitionConfig
from .dataset import DatasetSplit
from .exceptions import ModelNotTrainedError, AttendanceSystemError
from .logger import get_logger

logger = get_logger(__name__)

MODEL_FILENAME = "face_model.yml"
LABELS_FILENAME = "labels.json"


@dataclass
class Prediction:
    """Outcome of comparing one face chip against the trained model."""

    student_id: Optional[str]
    name: Optional[str]
    distance: float
    accepted: bool

    @property
    def confidence(self) -> float:
        """Distance mapped to a 0-100 'confidence' for human-readable output."""
        return round(max(0.0, 100.0 - self.distance), 2)

    def label(self) -> str:
        if self.accepted and self.student_id:
            return f"{self.student_id} ({self.confidence:.0f}%)"
        return "Unknown"


def build_model(config: RecognitionConfig):
    """Factory for the underlying OpenCV recogniser object."""
    if not hasattr(cv2, "face"):
        raise AttendanceSystemError(
            "cv2.face is unavailable. Install opencv-contrib-python "
            "(pip install opencv-contrib-python)."
        )
    algorithm = config.algorithm.lower()
    if algorithm == "lbph":
        return cv2.face.LBPHFaceRecognizer_create(
            radius=config.lbph_radius,
            neighbors=config.lbph_neighbors,
            grid_x=config.lbph_grid_x,
            grid_y=config.lbph_grid_y,
        )
    if algorithm == "eigen":
        return cv2.face.EigenFaceRecognizer_create(num_components=config.num_components)
    if algorithm == "fisher":
        return cv2.face.FisherFaceRecognizer_create(num_components=config.num_components)
    raise AttendanceSystemError(f"Unknown algorithm: {config.algorithm!r}")


class FaceRecognizer:
    """Train, persist, load and query a classical face-recognition model."""

    def __init__(self, config: RecognitionConfig):
        self.config = config
        self.model = build_model(config)
        self.label_map: Dict[int, str] = {}
        self.names: Dict[str, str] = {}
        self._trained = False

    # ------------------------------------------------------------------
    @property
    def is_trained(self) -> bool:
        return self._trained

    def train(self, data: DatasetSplit) -> None:
        """Fit the model on a labelled dataset."""
        if len(data) == 0:
            raise ModelNotTrainedError("Cannot train on an empty dataset")
        distinct = len(set(data.labels))
        if self.config.algorithm == "fisher" and distinct < 2:
            raise AttendanceSystemError(
                "Fisherfaces needs at least two enrolled students; "
                "use --algorithm lbph for a single student."
            )
        images = [np.asarray(img, dtype=np.uint8) for img in data.images]
        labels = np.asarray(data.labels, dtype=np.int32)
        self.model.train(images, labels)
        self.label_map = dict(data.label_map)
        self.names = dict(data.names)
        self._trained = True
        logger.info(
            "Trained %s on %d samples / %d students",
            self.config.algorithm, len(images), distinct,
        )

    def update(self, data: DatasetSplit) -> None:
        """Incrementally add samples (LBPH only) without a full retrain."""
        if self.config.algorithm != "lbph":
            raise AttendanceSystemError("Only LBPH supports incremental update()")
        if not self._trained:
            return self.train(data)
        self.model.update(
            [np.asarray(i, dtype=np.uint8) for i in data.images],
            np.asarray(data.labels, dtype=np.int32),
        )
        self.label_map.update(data.label_map)
        self.names.update(data.names)

    # ------------------------------------------------------------------
    def predict(self, chip: np.ndarray) -> Prediction:
        """Classify one normalised face chip."""
        if not self._trained:
            raise ModelNotTrainedError(
                "Model is not trained. Run 'python main.py train' first."
            )
        label, distance = self.model.predict(np.asarray(chip, dtype=np.uint8))
        student_id = self.label_map.get(int(label))
        accepted = bool(
            student_id is not None and distance <= self.config.distance_threshold
        )
        return Prediction(
            student_id=student_id if accepted else None,
            name=self.names.get(student_id or "", None) if accepted else None,
            distance=float(distance),
            accepted=accepted,
        )

    def predict_raw(self, chip: np.ndarray) -> Tuple[int, float]:
        """Return the raw ``(label, distance)`` pair - used by evaluation."""
        if not self._trained:
            raise ModelNotTrainedError("Model is not trained")
        label, distance = self.model.predict(np.asarray(chip, dtype=np.uint8))
        return int(label), float(distance)

    def predict_batch(self, chips: List[np.ndarray]) -> List[Prediction]:
        return [self.predict(chip) for chip in chips]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, model_dir: str | Path) -> Path:
        """Write the model plus its label map to *model_dir*."""
        if not self._trained:
            raise ModelNotTrainedError("Refusing to save an untrained model")
        directory = Path(model_dir)
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / MODEL_FILENAME
        self.model.write(str(model_path))
        (directory / LABELS_FILENAME).write_text(
            json.dumps(
                {
                    "algorithm": self.config.algorithm,
                    "label_map": {str(k): v for k, v in self.label_map.items()},
                    "names": self.names,
                    "distance_threshold": self.config.distance_threshold,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Model saved to %s", model_path)
        return model_path

    @classmethod
    def load(cls, model_dir: str | Path, config: RecognitionConfig) -> "FaceRecognizer":
        """Restore a previously trained model."""
        directory = Path(model_dir)
        model_path = directory / MODEL_FILENAME
        labels_path = directory / LABELS_FILENAME
        if not model_path.is_file() or not labels_path.is_file():
            raise ModelNotTrainedError(
                f"No trained model in {directory}. Run 'python main.py train' first."
            )
        meta = json.loads(labels_path.read_text(encoding="utf-8"))
        stored_algorithm = meta.get("algorithm", config.algorithm)
        if stored_algorithm != config.algorithm:
            logger.warning(
                "Model on disk was trained with %s; loading it as %s",
                stored_algorithm, stored_algorithm,
            )
            config = RecognitionConfig(**{**config.__dict__, "algorithm": stored_algorithm})
        recognizer = cls(config)
        recognizer.model.read(str(model_path))
        recognizer.label_map = {int(k): v for k, v in meta["label_map"].items()}
        recognizer.names = meta.get("names", {})
        recognizer._trained = True
        logger.info(
            "Loaded %s model with %d students", stored_algorithm, len(recognizer.label_map)
        )
        return recognizer
