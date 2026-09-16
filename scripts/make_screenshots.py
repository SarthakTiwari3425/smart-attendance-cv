#!/usr/bin/env python3
"""Produce the figures used in the README and the project report.

Creates:
  docs/images/sample_faces.png     - enrolment montage (synthetic cohort)
  docs/images/pipeline_stages.png  - one face through each pre-processing stage
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attendance_system.config import PreprocessConfig          # noqa: E402
from attendance_system.preprocessing import FacePreprocessor    # noqa: E402
from attendance_system.synthetic import DEMO_NAMES, FaceIdentity, render_face  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"


def label(image: np.ndarray, text: str) -> np.ndarray:
    canvas = cv2.copyMakeBorder(image, 0, 22, 0, 0, cv2.BORDER_CONSTANT, value=255)
    cv2.putText(canvas, text, (4, canvas.shape[0] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1, cv2.LINE_AA)
    return canvas


def sample_grid() -> None:
    rows = []
    for offset, (student_id, name) in enumerate(DEMO_NAMES):
        identity = FaceIdentity.sample(offset)
        row = [label(render_face(identity, jitter_seed=j), f"{student_id} s{j}")
               for j in range(4)]
        rows.append(np.hstack(row))
    grid = np.vstack(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "sample_faces.png"), grid)


def pipeline_stages() -> None:
    processor = FacePreprocessor(PreprocessConfig(align_eyes=False))
    raw = render_face(FaceIdentity.sample(0), jitter_seed=3)
    stages = [
        label(raw, "1. input"),
        label(processor.crop_chip(raw), "2. crop+resize"),
        label(processor.denoise(processor.crop_chip(raw)), "3. denoise"),
        label(processor.normalize(raw), "4. CLAHE (model input)"),
    ]
    cv2.imwrite(str(OUT / "pipeline_stages.png"), np.hstack(stages))


if __name__ == "__main__":
    sample_grid()
    pipeline_stages()
    print(f"Figures written to {OUT}")
