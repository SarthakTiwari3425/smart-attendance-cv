"""Image pre-processing for face recognition.

The recogniser (LBPH / Eigenfaces / Fisherfaces) is sensitive to scale,
illumination and in-plane rotation, so every face crop passes through the
same normalisation chain before it is either stored as a training sample or
compared against the model:

    BGR frame -> grayscale -> optional eye-based alignment -> CLAHE
              -> fixed-size resize -> uint8 face chip

Keeping this in one place guarantees that training and inference see
identically processed images, which is the single most common source of
accuracy loss in classical face-recognition pipelines.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import PreprocessConfig
from .logger import get_logger

logger = get_logger(__name__)

Box = Tuple[int, int, int, int]  # (x, y, w, h)


class FacePreprocessor:
    """Normalises face crops so that training and inference match."""

    def __init__(self, config: PreprocessConfig, eye_cascade: Optional[cv2.CascadeClassifier] = None):
        self.config = config
        self._eye_cascade = eye_cascade
        self._clahe = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=(config.clahe_grid, config.clahe_grid),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert a BGR/BGRA image to single-channel grayscale."""
        if image is None or image.size == 0:
            raise ValueError("Cannot pre-process an empty image")
        if image.ndim == 2:
            return image
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def equalize(self, gray: np.ndarray) -> np.ndarray:
        """Flatten illumination differences between enrolment and classroom."""
        if self.config.use_clahe:
            return self._clahe.apply(gray)
        return cv2.equalizeHist(gray)

    def resize(self, gray: np.ndarray) -> np.ndarray:
        """Resize to the fixed chip size expected by the recogniser."""
        target = (self.config.face_width, self.config.face_height)
        interpolation = (
            cv2.INTER_AREA
            if gray.shape[0] > target[1] or gray.shape[1] > target[0]
            else cv2.INTER_CUBIC
        )
        return cv2.resize(gray, target, interpolation=interpolation)

    def sharpness(self, gray: np.ndarray) -> float:
        """Variance of the Laplacian - a cheap focus/motion-blur measure."""
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def is_blurry(self, gray: np.ndarray) -> bool:
        """Reject frames too blurred to produce a reliable LBP histogram."""
        return self.sharpness(gray) < self.config.blur_rejection_threshold

    def align(self, gray_face: np.ndarray) -> np.ndarray:
        """Rotate the crop so that the eye line is horizontal.

        Uses a Haar eye cascade; if fewer than two eyes are found the crop is
        returned unchanged, which is the safe behaviour for profile or
        low-resolution faces.
        """
        if not self.config.align_eyes or self._eye_cascade is None:
            return gray_face
        eyes: Sequence[Box] = self._eye_cascade.detectMultiScale(
            gray_face, scaleFactor=1.1, minNeighbors=6,
            minSize=(max(8, gray_face.shape[1] // 10),) * 2,
        )
        if len(eyes) < 2:
            return gray_face
        # Keep the two largest detections and order them left-to-right.
        eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
        left, right = sorted(eyes, key=lambda e: e[0])
        left_c = (left[0] + left[2] / 2.0, left[1] + left[3] / 2.0)
        right_c = (right[0] + right[2] / 2.0, right[1] + right[3] / 2.0)
        dy = right_c[1] - left_c[1]
        dx = right_c[0] - left_c[0]
        if abs(dx) < 1e-6:
            return gray_face
        angle = float(np.degrees(np.arctan2(dy, dx)))
        if abs(angle) < 1.0 or abs(angle) > 45.0:
            return gray_face                  # nothing to fix / bogus detection
        center = ((left_c[0] + right_c[0]) / 2.0, (left_c[1] + right_c[1]) / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            gray_face, matrix, (gray_face.shape[1], gray_face.shape[0]),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )

    def denoise(self, gray: np.ndarray) -> np.ndarray:
        """Suppress sensor noise before LBP codes are computed.

        Local Binary Patterns compare a pixel with its immediate neighbours,
        so single-pixel noise flips bits and corrupts the histogram.  A small
        Gaussian kernel measurably improved rank-1 accuracy in our own
        evaluation (see ``docs/`` and the report), which is why it is on by
        default.
        """
        k = int(self.config.denoise_ksize)
        if k < 3:
            return gray
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(gray, (k, k), 0)

    # -- the two stages the rest of the project uses -------------------
    def crop_chip(self, image: np.ndarray, box: Optional[Box] = None) -> np.ndarray:
        """Stage 1 (storage form): grayscale -> crop -> align -> resize.

        This is what gets written to the dataset folder: a geometrically
        normalised but photometrically untouched face.
        """
        gray = self.to_grayscale(image)
        if box is not None:
            gray = crop(gray, box)
        gray = self.align(gray)
        return self.resize(gray).astype(np.uint8, copy=False)

    def normalize(self, chip: np.ndarray) -> np.ndarray:
        """Stage 2 (model input): denoise -> CLAHE -> resize.

        Applied identically to training samples loaded from disk and to live
        frames, which is what keeps the two distributions comparable.
        """
        gray = self.to_grayscale(chip)
        gray = self.denoise(gray)
        gray = self.equalize(gray)
        return self.resize(gray).astype(np.uint8, copy=False)

    def process(self, image: np.ndarray, box: Optional[Box] = None) -> np.ndarray:
        """Full inference chain: :meth:`crop_chip` followed by :meth:`normalize`."""
        return self.normalize(self.crop_chip(image, box))


def crop(image: np.ndarray, box: Box, margin: float = 0.0) -> np.ndarray:
    """Crop *box* from *image*, clamped to the image bounds.

    A positive *margin* (fraction of the box size) includes some context
    around the face, which helps the eye cascade during alignment.
    """
    x, y, w, h = box
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid bounding box: {box}")
    dx, dy = int(w * margin), int(h * margin)
    x0 = max(0, x - dx)
    y0 = max(0, y - dy)
    x1 = min(image.shape[1], x + w + dx)
    y1 = min(image.shape[0], y + h + dy)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Bounding box {box} lies outside the image")
    return image[y0:y1, x0:x1]
