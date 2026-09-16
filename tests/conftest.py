"""Shared pytest fixtures.

Every fixture works offline and without a camera: face data comes from the
synthetic generator, and the database lives in a temporary folder.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attendance_system.config import AppConfig           # noqa: E402
from attendance_system.database import AttendanceDatabase  # noqa: E402
from attendance_system.dataset import DatasetManager      # noqa: E402
from attendance_system.detector import build_detector     # noqa: E402
from attendance_system.preprocessing import FacePreprocessor  # noqa: E402
from attendance_system.synthetic import DEMO_NAMES, generate_dataset  # noqa: E402


@pytest.fixture
def config(tmp_path) -> AppConfig:
    cfg = AppConfig()
    cfg.detection.backend = "none"
    cfg.preprocess.align_eyes = False
    cfg.paths.dataset_dir = str(tmp_path / "dataset")
    cfg.paths.database_path = str(tmp_path / "attendance.db")
    cfg.paths.model_dir = str(tmp_path / "models")
    cfg.paths.report_dir = str(tmp_path / "reports")
    cfg.attendance.consecutive_hits_required = 1
    return cfg


@pytest.fixture
def db(config) -> AttendanceDatabase:
    database = AttendanceDatabase(config.paths.database_path)
    yield database
    database.close()


@pytest.fixture
def dataset_manager(config) -> DatasetManager:
    generate_dataset(
        config.paths.dataset_dir, students=DEMO_NAMES[:3], samples_per_student=8
    )
    detector = build_detector(config.detection)
    preprocessor = FacePreprocessor(config.preprocess)
    return DatasetManager(config.paths.dataset_dir, detector, preprocessor)
