"""Enrolment and dataset management.

The on-disk layout is deliberately simple and human-inspectable::

    data/dataset/
        S001_Asha_Verma/
            S001_000.png
            S001_001.png
        S002_Rahul_Nair/
            ...

Folder name encodes ``<student_id>_<name>``, so the dataset can be rebuilt
from disk alone even if the database is lost.  Samples are stored already
normalised (grayscale, aligned, CLAHE, fixed size), which makes training
fast and keeps train/inference pre-processing identical.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .detector import FaceDetector
from .exceptions import DatasetError
from .logger import get_logger
from .preprocessing import FacePreprocessor

logger = get_logger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".tif", ".tiff"}
_SAFE = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass
class DatasetSplit:
    """A materialised set of face chips with integer labels."""

    images: List[np.ndarray]
    labels: List[int]
    label_map: Dict[int, str]          # numeric label -> student_id
    names: Dict[str, str]              # student_id -> display name

    def __len__(self) -> int:
        return len(self.images)

    def student_ids(self) -> List[str]:
        return sorted(set(self.label_map.values()))


def folder_name(student_id: str, name: str) -> str:
    """Build a filesystem-safe folder name for a student."""
    return f"{_SAFE.sub('', student_id)}_{_SAFE.sub('_', name.strip())}"


def parse_folder(folder: Path) -> Tuple[str, str]:
    """Inverse of :func:`folder_name`; returns ``(student_id, name)``."""
    stem = folder.name
    if "_" not in stem:
        return stem, stem
    student_id, _, raw_name = stem.partition("_")
    return student_id, raw_name.replace("_", " ") or student_id


class DatasetManager:
    """Creates, extends and loads the enrolled-face dataset."""

    def __init__(
        self,
        dataset_dir: str | Path,
        detector: FaceDetector,
        preprocessor: FacePreprocessor,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.detector = detector
        self.preprocessor = preprocessor
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Enrolment
    # ------------------------------------------------------------------
    def student_dir(self, student_id: str, name: str) -> Path:
        """Return (creating if needed) the folder holding a student's samples."""
        for existing in self.dataset_dir.iterdir():
            if existing.is_dir() and parse_folder(existing)[0] == student_id:
                return existing
        path = self.dataset_dir / folder_name(student_id, name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_sample(self, student_id: str, name: str, chip: np.ndarray) -> Path:
        """Persist one normalised face chip and return its path."""
        target = self.student_dir(student_id, name)
        index = len(list(target.glob("*.png")))
        path = target / f"{_SAFE.sub('', student_id)}_{index:03d}.png"
        if not cv2.imwrite(str(path), chip):
            raise DatasetError(f"Failed to write sample to {path}")
        return path

    def enroll_from_images(
        self, student_id: str, name: str, source_dir: str | Path, limit: int = 0
    ) -> int:
        """Enrol a student from a folder of photographs.

        Each image is scanned for a face; the largest detection is cropped,
        normalised and stored.  Returns the number of samples added.
        """
        source = Path(source_dir)
        if not source.is_dir():
            raise DatasetError(f"Not a directory: {source}")
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise DatasetError(f"No images found in {source}")
        added = 0
        for path in files:
            if limit and added >= limit:
                break
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                logger.warning("Skipping unreadable image %s", path.name)
                continue
            boxes = self.detector.detect(image)
            if not boxes:
                logger.warning("No face detected in %s - skipped", path.name)
                continue
            chip = self.preprocessor.crop_chip(image, boxes[0])
            if self.preprocessor.is_blurry(chip):
                logger.warning("%s rejected: too blurry", path.name)
                continue
            self.save_sample(student_id, name, chip)
            added += 1
        if added == 0:
            raise DatasetError(
                f"No usable faces found in {source}. "
                "Try clearer, front-facing photos or --detector none for cropped faces."
            )
        logger.info("Enrolled %d samples for %s (%s)", added, student_id, name)
        return added

    def enroll_from_camera(
        self, student_id: str, name: str, source, samples: int, display: bool = False
    ) -> int:
        """Capture *samples* face chips from a live camera / video source."""
        added = 0
        for frame_index, frame in source.frames():
            boxes = self.detector.detect(frame)
            if not boxes:
                continue
            chip = self.preprocessor.crop_chip(frame, boxes[0])
            if self.preprocessor.is_blurry(chip):
                continue
            self.save_sample(student_id, name, chip)
            added += 1
            logger.info("Captured sample %d/%d", added, samples)
            if display:
                from .detector import draw_boxes

                cv2.imshow(
                    "Enrolment - press q to stop",
                    draw_boxes(frame, [(boxes[0], f"{added}/{samples}", True)]),
                )
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if added >= samples:
                break
        if display:
            cv2.destroyAllWindows()
        return added

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self, min_samples: int = 1) -> DatasetSplit:
        """Load every stored sample into memory as a labelled dataset."""
        images: List[np.ndarray] = []
        labels: List[int] = []
        label_map: Dict[int, str] = {}
        names: Dict[str, str] = {}
        next_label = 0

        folders = sorted(p for p in self.dataset_dir.iterdir() if p.is_dir())
        if not folders:
            raise DatasetError(
                f"Dataset is empty: {self.dataset_dir}. Run 'enroll' or 'demo' first."
            )

        for folder in folders:
            student_id, name = parse_folder(folder)
            files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            if len(files) < min_samples:
                logger.warning(
                    "%s has %d samples (< %d) - excluded from training",
                    student_id, len(files), min_samples,
                )
                continue
            label = next_label
            next_label += 1
            label_map[label] = student_id
            names[student_id] = name
            for file in files:
                chip = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
                if chip is None:
                    logger.warning("Unreadable sample %s", file)
                    continue
                images.append(self.preprocessor.normalize(chip))
                labels.append(label)

        if not images:
            raise DatasetError(
                "No student has enough samples to train on. "
                f"Need at least {min_samples} images per student."
            )
        logger.info(
            "Loaded %d samples across %d students", len(images), len(label_map)
        )
        return DatasetSplit(images, labels, label_map, names)

    def counts(self) -> Dict[str, int]:
        """Return ``{student_id: sample_count}`` straight from disk."""
        result: Dict[str, int] = {}
        for folder in sorted(p for p in self.dataset_dir.iterdir() if p.is_dir()):
            student_id, _ = parse_folder(folder)
            result[student_id] = sum(
                1 for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
            )
        return result

    def delete_student(self, student_id: str) -> bool:
        for folder in self.dataset_dir.iterdir():
            if folder.is_dir() and parse_folder(folder)[0] == student_id:
                shutil.rmtree(folder)
                return True
        return False


def stratified_split(
    data: DatasetSplit, test_ratio: float = 0.3, seed: int = 42
) -> Tuple[DatasetSplit, DatasetSplit]:
    """Split per class so every student appears in both train and test sets."""
    if not 0.0 < test_ratio < 1.0:
        raise DatasetError("test_ratio must be between 0 and 1")
    rng = np.random.default_rng(seed)
    by_label: Dict[int, List[int]] = {}
    for index, label in enumerate(data.labels):
        by_label.setdefault(label, []).append(index)

    train_idx: List[int] = []
    test_idx: List[int] = []
    for label, indices in by_label.items():
        order = rng.permutation(len(indices))
        shuffled = [indices[i] for i in order]
        n_test = max(1, int(round(len(shuffled) * test_ratio)))
        n_test = min(n_test, len(shuffled) - 1)   # always keep >=1 for training
        test_idx.extend(shuffled[:n_test])
        train_idx.extend(shuffled[n_test:])

    def subset(indices: List[int]) -> DatasetSplit:
        return DatasetSplit(
            images=[data.images[i] for i in indices],
            labels=[data.labels[i] for i in indices],
            label_map=dict(data.label_map),
            names=dict(data.names),
        )

    return subset(sorted(train_idx)), subset(sorted(test_idx))
