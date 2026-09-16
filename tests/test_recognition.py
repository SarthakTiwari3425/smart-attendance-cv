"""Tests for dataset handling and the recognition model.

These are the accuracy-critical paths, so the assertions check real
behaviour (the model actually identifies held-out samples) rather than just
that functions return without raising.
"""

import numpy as np
import pytest

from attendance_system.config import RecognitionConfig
from attendance_system.dataset import (
    DatasetManager,
    folder_name,
    parse_folder,
    stratified_split,
)
from attendance_system.exceptions import DatasetError, ModelNotTrainedError
from attendance_system.recognizer import FaceRecognizer


# ----------------------------------------------------------------- dataset
def test_folder_name_round_trip():
    name = folder_name("S001", "Asha Verma")
    assert name == "S001_Asha_Verma"
    from pathlib import Path

    student_id, display = parse_folder(Path(name))
    assert student_id == "S001"
    assert display == "Asha Verma"


def test_folder_name_strips_unsafe_characters():
    assert "/" not in folder_name("S/001", "A..B")


def test_loading_dataset_yields_labelled_samples(dataset_manager):
    data = dataset_manager.load(min_samples=5)
    assert len(data) == 24                     # 3 students x 8 samples
    assert len(data.label_map) == 3
    assert len(data.images) == len(data.labels)
    assert data.images[0].shape == (200, 200)


def test_load_rejects_students_with_too_few_samples(dataset_manager):
    with pytest.raises(DatasetError):
        dataset_manager.load(min_samples=99)


def test_empty_dataset_raises(tmp_path, config):
    from attendance_system.detector import build_detector
    from attendance_system.preprocessing import FacePreprocessor

    manager = DatasetManager(
        tmp_path / "empty",
        build_detector(config.detection),
        FacePreprocessor(config.preprocess),
    )
    with pytest.raises(DatasetError):
        manager.load()


def test_stratified_split_keeps_every_class_in_both_halves(dataset_manager):
    data = dataset_manager.load(min_samples=5)
    train, test = stratified_split(data, test_ratio=0.25, seed=7)
    assert set(train.labels) == set(test.labels) == set(data.labels)
    assert len(train) + len(test) == len(data)


def test_split_ratio_is_rejected_when_out_of_range(dataset_manager):
    data = dataset_manager.load(min_samples=5)
    with pytest.raises(DatasetError):
        stratified_split(data, test_ratio=1.5)


def test_counts_match_files_on_disk(dataset_manager):
    assert set(dataset_manager.counts().values()) == {8}


# -------------------------------------------------------------- recognizer
def test_prediction_before_training_raises():
    recognizer = FaceRecognizer(RecognitionConfig())
    with pytest.raises(ModelNotTrainedError):
        recognizer.predict(np.zeros((200, 200), dtype=np.uint8))


def test_lbph_identifies_held_out_samples(dataset_manager):
    data = dataset_manager.load(min_samples=5)
    train, test = stratified_split(data, test_ratio=0.25, seed=11)
    recognizer = FaceRecognizer(RecognitionConfig(distance_threshold=1e9))
    recognizer.train(train)
    correct = sum(
        recognizer.predict_raw(image)[0] == label
        for image, label in zip(test.images, test.labels)
    )
    assert correct / len(test) >= 0.8


def test_far_away_face_is_rejected_by_threshold(dataset_manager):
    """A tight threshold must produce 'Unknown' rather than a forced match."""
    data = dataset_manager.load(min_samples=5)
    recognizer = FaceRecognizer(RecognitionConfig(distance_threshold=0.01))
    recognizer.train(data)
    stranger = np.random.default_rng(99).integers(0, 255, (200, 200), dtype=np.uint8)
    prediction = recognizer.predict(stranger)
    assert prediction.accepted is False
    assert prediction.student_id is None
    assert prediction.label() == "Unknown"


def test_confidence_is_bounded(dataset_manager):
    data = dataset_manager.load(min_samples=5)
    recognizer = FaceRecognizer(RecognitionConfig(distance_threshold=1e9))
    recognizer.train(data)
    prediction = recognizer.predict(data.images[0])
    assert 0.0 <= prediction.confidence <= 100.0


def test_model_round_trips_through_disk(dataset_manager, tmp_path):
    data = dataset_manager.load(min_samples=5)
    recognizer = FaceRecognizer(RecognitionConfig(distance_threshold=1e9))
    recognizer.train(data)
    before = recognizer.predict_raw(data.images[0])
    recognizer.save(tmp_path / "model")

    restored = FaceRecognizer.load(tmp_path / "model", RecognitionConfig())
    assert restored.label_map == recognizer.label_map
    assert restored.predict_raw(data.images[0]) == pytest.approx(before, rel=1e-6)


def test_loading_a_missing_model_raises(tmp_path):
    with pytest.raises(ModelNotTrainedError):
        FaceRecognizer.load(tmp_path / "nothing", RecognitionConfig())


def test_saving_an_untrained_model_raises(tmp_path):
    with pytest.raises(ModelNotTrainedError):
        FaceRecognizer(RecognitionConfig()).save(tmp_path / "model")


def test_threshold_calibration_lands_near_genuine_distances(dataset_manager):
    from attendance_system.evaluation import calibrate_threshold

    data = dataset_manager.load(min_samples=5)
    train, test = stratified_split(data, test_ratio=0.3, seed=5)
    recognizer = FaceRecognizer(RecognitionConfig(distance_threshold=1e9))
    recognizer.train(train)
    threshold = calibrate_threshold(recognizer, test)
    genuine = [
        recognizer.predict_raw(image)[1]
        for image, label in zip(test.images, test.labels)
        if recognizer.predict_raw(image)[0] == label
    ]
    assert threshold > float(np.mean(genuine))
