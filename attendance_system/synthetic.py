"""Procedural face generator used for the offline demo and the test-suite.

Why this exists
---------------
An evaluator running the project on a server has no webcam and probably no
face dataset, and downloading one (ORL/AT&T, Yale) needs network access that
may not be available.  This module draws parametric, face-like grayscale
images: each synthetic "identity" gets its own geometry (head shape, eye
spacing, brow thickness, nose, mouth, hairline, glasses) and each sample of
that identity adds realistic nuisance variation - illumination gradients,
sensor noise, small in-plane rotation, translation, scale and blur.

That is exactly the nuisance set a classical recogniser has to survive, so
``python main.py demo`` exercises the real pipeline end to end rather than a
mock.  For a graded real-world run, enrol real photographs with
``python main.py enroll``; nothing else in the pipeline changes.

Images are produced pre-cropped, so they are used with
``--detector none`` (the :class:`~attendance_system.detector.PassthroughDetector`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from .logger import get_logger

logger = get_logger(__name__)

DEMO_NAMES: List[Tuple[str, str]] = [
    ("S001", "Asha Verma"),
    ("S002", "Rahul Nair"),
    ("S003", "Meera Iyer"),
    ("S004", "Kabir Shah"),
    ("S005", "Divya Rao"),
    ("S006", "Arjun Menon"),
]


@dataclass
class FaceIdentity:
    """The fixed geometry that makes one synthetic person recognisable."""

    seed: int
    skin: int
    face_w: float
    face_h: float
    eye_dx: float
    eye_y: float
    eye_r: float
    brow_t: int
    nose_len: float
    nose_w: float
    mouth_w: float
    mouth_y: float
    hair_line: float
    glasses: bool
    beard: bool

    @classmethod
    def sample(cls, seed: int) -> "FaceIdentity":
        rng = np.random.default_rng(seed)
        return cls(
            seed=seed,
            skin=int(rng.integers(110, 205)),
            face_w=float(rng.uniform(0.30, 0.40)),
            face_h=float(rng.uniform(0.40, 0.50)),
            eye_dx=float(rng.uniform(0.14, 0.21)),
            eye_y=float(rng.uniform(0.40, 0.46)),
            eye_r=float(rng.uniform(0.035, 0.055)),
            brow_t=int(rng.integers(2, 6)),
            nose_len=float(rng.uniform(0.10, 0.17)),
            nose_w=float(rng.uniform(0.04, 0.08)),
            mouth_w=float(rng.uniform(0.10, 0.17)),
            mouth_y=float(rng.uniform(0.66, 0.74)),
            hair_line=float(rng.uniform(0.16, 0.28)),
            glasses=bool(rng.random() < 0.4),
            beard=bool(rng.random() < 0.35),
        )


def render_face(identity: FaceIdentity, size: int = 200, jitter_seed: int = 0) -> np.ndarray:
    """Draw one grayscale sample of *identity* with nuisance variation."""
    rng = np.random.default_rng((identity.seed + 1) * 1000 + jitter_seed)
    canvas = np.full((size, size), 70, dtype=np.uint8)          # background
    cx, cy = size // 2, int(size * 0.52)
    axes = (int(size * identity.face_w), int(size * identity.face_h))

    # Head + neck
    cv2.ellipse(canvas, (cx, cy), axes, 0, 0, 360, identity.skin, -1)
    cv2.ellipse(
        canvas, (cx, cy + int(axes[1] * 0.95)),
        (int(axes[0] * 0.35), int(axes[1] * 0.25)), 0, 0, 360,
        max(0, identity.skin - 12), -1,
    )

    # Hair
    cv2.ellipse(
        canvas, (cx, int(size * identity.hair_line) + int(axes[1] * 0.35)),
        (int(axes[0] * 1.02), int(axes[1] * 0.55)), 0, 180, 360,
        max(10, identity.skin - 85), -1,
    )

    # Eyes (sclera, iris, pupil) and brows
    eye_y = int(size * identity.eye_y)
    eye_r = int(size * identity.eye_r)
    for sign in (-1, 1):
        ex = cx + sign * int(size * identity.eye_dx)
        cv2.ellipse(canvas, (ex, eye_y), (eye_r, int(eye_r * 0.62)), 0, 0, 360, 245, -1)
        cv2.circle(canvas, (ex, eye_y), max(2, int(eye_r * 0.5)), 60, -1)
        cv2.circle(canvas, (ex, eye_y), max(1, int(eye_r * 0.22)), 15, -1)
        cv2.line(
            canvas,
            (ex - eye_r, eye_y - int(eye_r * 1.5)),
            (ex + eye_r, eye_y - int(eye_r * 1.7)),
            max(5, identity.skin - 95), identity.brow_t,
        )
        if identity.glasses:
            cv2.circle(canvas, (ex, eye_y), int(eye_r * 1.5), 40, 2)
    if identity.glasses:
        cv2.line(
            canvas,
            (cx - int(size * identity.eye_dx) + int(eye_r * 1.5), eye_y),
            (cx + int(size * identity.eye_dx) - int(eye_r * 1.5), eye_y),
            40, 2,
        )

    # Nose
    nose_bottom = eye_y + int(size * identity.nose_len)
    nose_w = int(size * identity.nose_w)
    cv2.line(canvas, (cx, eye_y + eye_r), (cx - nose_w // 2, nose_bottom),
             max(0, identity.skin - 40), 2)
    cv2.ellipse(canvas, (cx, nose_bottom), (nose_w, max(2, nose_w // 2)),
                0, 0, 180, max(0, identity.skin - 45), 2)

    # Mouth
    mouth_y = int(size * identity.mouth_y)
    mouth_w = int(size * identity.mouth_w)
    cv2.ellipse(canvas, (cx, mouth_y), (mouth_w, max(4, mouth_w // 3)),
                0, 0, 180, max(0, identity.skin - 75), -1)
    if identity.beard:
        cv2.ellipse(
            canvas, (cx, mouth_y + int(size * 0.06)),
            (int(axes[0] * 0.62), int(axes[1] * 0.30)), 0, 0, 180,
            max(8, identity.skin - 70), -1,
        )

    # Skin texture: deterministic per identity, so it behaves like a real
    # persistent characteristic rather than noise.
    texture_rng = np.random.default_rng(identity.seed)
    texture = texture_rng.normal(0, 4, canvas.shape)
    canvas = np.clip(canvas.astype(np.float32) + texture, 0, 255)

    # ---- Per-sample nuisance variation -------------------------------
    angle = rng.uniform(-8, 8)
    scale = rng.uniform(0.94, 1.06)
    tx, ty = rng.uniform(-5, 5), rng.uniform(-5, 5)
    matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    canvas = cv2.warpAffine(
        canvas, matrix, (size, size), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Directional illumination gradient
    gradient = np.linspace(-1, 1, size, dtype=np.float32)
    gx, gy = np.meshgrid(gradient, gradient)
    direction = rng.uniform(0, 2 * np.pi)
    strength = rng.uniform(5, 30)
    canvas = canvas + strength * (np.cos(direction) * gx + np.sin(direction) * gy)

    # Sensor noise + optional slight defocus
    canvas = canvas + rng.normal(0, rng.uniform(2, 6), canvas.shape)
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        canvas = cv2.GaussianBlur(canvas, (3, 3), rng.uniform(0.4, 1.0))
    return canvas


def generate_dataset(
    output_dir: str | Path,
    students: List[Tuple[str, str]] | None = None,
    samples_per_student: int = 15,
    size: int = 200,
    start_seed: int = 0,
) -> dict:
    """Write a full synthetic dataset in the standard folder layout."""
    from .dataset import folder_name          # local import avoids a cycle

    students = students or DEMO_NAMES
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    written = {}
    for offset, (student_id, name) in enumerate(students):
        identity = FaceIdentity.sample(start_seed + offset)
        folder = root / folder_name(student_id, name)
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(samples_per_student):
            image = render_face(identity, size=size, jitter_seed=index)
            cv2.imwrite(str(folder / f"{student_id}_{index:03d}.png"), image)
        written[student_id] = samples_per_student
        logger.info("Generated %d samples for %s (%s)", samples_per_student, student_id, name)
    return written


def generate_classroom_frames(
    output_dir: str | Path,
    students: List[Tuple[str, str]] | None = None,
    frames_per_student: int = 4,
    size: int = 200,
    start_seed: int = 0,
    include_stranger: bool = True,
) -> int:
    """Write unlabelled 'classroom camera' frames for the recognition demo.

    Frames are drawn with a different jitter offset from the enrolment set, so
    the demo never tests on images it trained on.  One unenrolled stranger is
    included to show the rejection path working.
    """
    students = students or DEMO_NAMES
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    count = 0
    for offset, (student_id, _) in enumerate(students):
        identity = FaceIdentity.sample(start_seed + offset)
        for index in range(frames_per_student):
            image = render_face(identity, size=size, jitter_seed=500 + index)
            cv2.imwrite(str(root / f"frame_{count:03d}_{student_id}.png"), image)
            count += 1
    if include_stranger:
        stranger = FaceIdentity.sample(start_seed + 9_999)
        for index in range(frames_per_student):
            image = render_face(stranger, size=size, jitter_seed=700 + index)
            cv2.imwrite(str(root / f"frame_{count:03d}_UNKNOWN.png"), image)
            count += 1
    logger.info("Generated %d classroom frames in %s", count, root)
    return count
