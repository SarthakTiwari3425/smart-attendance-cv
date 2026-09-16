"""Tests for session orchestration, reporting, configuration and the CLI."""

import numpy as np
import pytest

from attendance_system.attendance_service import AttendanceService, default_session_id
from attendance_system.config import AppConfig, RecognitionConfig
from attendance_system.dataset import stratified_split
from attendance_system.detector import build_detector
from attendance_system.evaluation import evaluate, save_report, sweep_thresholds
from attendance_system.exceptions import ConfigurationError, VideoSourceError
from attendance_system.preprocessing import FacePreprocessor
from attendance_system.recognizer import FaceRecognizer
from attendance_system.reporting import export_csv, render_table, session_report
from attendance_system.video_source import ImageFolderSource, open_source


@pytest.fixture
def service(config, db, dataset_manager):
    data = dataset_manager.load(min_samples=5)
    recognizer = FaceRecognizer(RecognitionConfig(distance_threshold=1e9))
    recognizer.train(data)
    for student_id, name in data.names.items():
        db.add_student(student_id, name)
    return AttendanceService(
        config,
        db,
        build_detector(config.detection),
        FacePreprocessor(config.preprocess),
        recognizer,
    ), data


# ------------------------------------------------------------- service
def test_session_marks_each_student_once(service, db, config):
    svc, data = service
    from attendance_system.attendance_service import SessionStats

    db.start_session("T1", "CV101")
    stats = SessionStats(session_id="T1")
    for index, image in enumerate(data.images[:10]):
        frame = np.stack([image] * 3, axis=-1)          # fake a BGR frame
        svc.process_frame(frame, "T1", index, stats)
    marked = {r.student_id for r in db.session_attendance("T1")}
    assert marked
    assert len(db.session_attendance("T1")) == len(marked)   # no duplicates


def test_temporal_voting_delays_marking(config, db, dataset_manager):
    """With two confirmations required, one frame must not be enough."""
    config.attendance.consecutive_hits_required = 2
    data = dataset_manager.load(min_samples=5)
    recognizer = FaceRecognizer(RecognitionConfig(distance_threshold=1e9))
    recognizer.train(data)
    for student_id, name in data.names.items():
        db.add_student(student_id, name)
    svc = AttendanceService(
        config, db, build_detector(config.detection),
        FacePreprocessor(config.preprocess), recognizer,
    )
    from attendance_system.attendance_service import SessionStats

    db.start_session("T2", "CV101")
    stats = SessionStats(session_id="T2")
    frame = np.stack([data.images[0]] * 3, axis=-1)
    svc.process_frame(frame, "T2", 0, stats)
    assert db.session_attendance("T2") == []       # one hit: still unconfirmed
    svc.process_frame(frame, "T2", 1, stats)
    assert len(db.session_attendance("T2")) == 1   # second hit confirms


def test_unknown_face_is_not_marked(config, db, dataset_manager):
    from attendance_system.attendance_service import SessionStats

    data = dataset_manager.load(min_samples=5)
    recognizer = FaceRecognizer(RecognitionConfig(distance_threshold=0.001))
    recognizer.train(data)
    svc = AttendanceService(
        config, db, build_detector(config.detection),
        FacePreprocessor(config.preprocess), recognizer,
    )
    db.start_session("T3", "CV101")
    stats = SessionStats(session_id="T3")
    svc.process_frame(np.stack([data.images[0]] * 3, -1), "T3", 0, stats)
    assert db.session_attendance("T3") == []
    assert stats.rejected == 1


def test_run_session_over_an_image_folder(service, db, config, tmp_path):
    import cv2

    svc, data = service
    folder = tmp_path / "frames"
    folder.mkdir()
    for index, image in enumerate(data.images[:6]):
        cv2.imwrite(str(folder / f"f{index}.png"), image)
    stats = svc.run_session(ImageFolderSource(folder), "T4", "CV101")
    assert stats.frames_processed == 6
    assert stats.faces_detected == 6
    assert len(db.session_attendance("T4")) == len(set(stats.marked))


def test_default_session_id_format():
    from datetime import datetime

    assert default_session_id("CV101", datetime(2026, 9, 16, 9, 30)) == "CV101-20260916-0930"


# ------------------------------------------------------------ reporting
def test_render_table_aligns_columns():
    table = render_table(["A", "BB"], [("x", "yyyy")])
    assert table.splitlines()[0].startswith("A ")
    assert "yyyy" in table


def test_render_table_handles_no_rows():
    assert render_table(["A"], []) == "(no records)"


def test_session_report_lists_present_and_absent(db):
    db.add_student("S001", "Asha")
    db.add_student("S002", "Rahul")
    db.start_session("R1", "CV101")
    db.mark_attendance("R1", "S001", 85.0)
    text = session_report(db, "R1")
    assert "Asha" in text and "Rahul" in text
    assert "Present: 1/2" in text


def test_export_csv_writes_a_header_and_rows(db, tmp_path):
    db.add_student("S001", "Asha")
    db.start_session("R2", "CV101")
    db.mark_attendance("R2", "S001", 85.0)
    path = export_csv(db.session_attendance("R2"), tmp_path / "out.csv")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("session_id,student_id")
    assert len(lines) == 2


# ----------------------------------------------------------- evaluation
def test_evaluation_produces_metrics_and_artefacts(dataset_manager, tmp_path):
    data = dataset_manager.load(min_samples=5)
    train, test = stratified_split(data, 0.3, seed=3)
    recognizer = FaceRecognizer(RecognitionConfig(distance_threshold=1e9))
    recognizer.train(train)
    result = evaluate(recognizer, test, n_train=len(train))
    assert 0.0 <= result.accuracy <= 1.0
    assert len(result.confusion) == result.n_classes
    assert sum(sum(row) for row in result.confusion) == len(test)
    written = save_report(result, tmp_path / "reports")
    assert written["json"].exists() and written["csv"].exists()


def test_threshold_sweep_is_monotonic_in_accept_rate():
    records = [(0, 0, 10.0), (0, 1, 50.0), (1, 1, 20.0)]
    sweep = sweep_thresholds(records, steps=5)
    rates = [row["accept_rate"] for row in sweep]
    assert rates == sorted(rates)
    assert rates[-1] == 1.0


# ---------------------------------------------------- config & sources
def test_config_rejects_unknown_section():
    with pytest.raises(ConfigurationError):
        AppConfig.from_dict({"nonsense": {"a": 1}})


def test_config_rejects_unknown_key():
    with pytest.raises(ConfigurationError):
        AppConfig.from_dict({"recognition": {"not_a_key": 1}})


def test_config_validation_catches_bad_algorithm():
    config = AppConfig()
    config.recognition.algorithm = "deep-magic"
    with pytest.raises(ConfigurationError):
        config.validate()


def test_config_file_round_trip(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "log_level: DEBUG\n"
        "recognition:\n  algorithm: eigen\n  distance_threshold: 42.5\n"
        "detection:\n  backend: none\n",
        encoding="utf-8",
    )
    config = AppConfig.load(path)
    assert config.log_level == "DEBUG"
    assert config.recognition.algorithm == "eigen"
    assert config.recognition.distance_threshold == 42.5
    assert config.detection.backend == "none"


def test_missing_source_raises_a_helpful_error():
    with pytest.raises(VideoSourceError):
        open_source("/definitely/not/here")


def test_open_source_picks_image_folder(tmp_path):
    import cv2

    cv2.imwrite(str(tmp_path / "a.png"), np.zeros((10, 10), dtype=np.uint8))
    assert isinstance(open_source(str(tmp_path)), ImageFolderSource)


def test_cli_runs_init_and_students(tmp_path, monkeypatch):
    """Smoke-test the CLI wiring end to end with a temporary project root."""
    from attendance_system import cli

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"paths:\n"
        f"  dataset_dir: {tmp_path / 'dataset'}\n"
        f"  database_path: {tmp_path / 'db.sqlite'}\n"
        f"  model_dir: {tmp_path / 'models'}\n"
        f"  report_dir: {tmp_path / 'reports'}\n"
        f"  log_file: {tmp_path / 'log.txt'}\n",
        encoding="utf-8",
    )
    assert cli.main(["--config", str(config_path), "init"]) == 0
    assert cli.main(["--config", str(config_path), "students"]) == 0
