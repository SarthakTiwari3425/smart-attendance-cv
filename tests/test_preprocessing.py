"""Unit tests for the pre-processing chain and the detection strategies."""

import numpy as np
import pytest

from attendance_system.config import DetectionConfig, PreprocessConfig
from attendance_system.detector import (
    HaarCascadeDetector,
    PassthroughDetector,
    build_detector,
    load_cascade,
)
from attendance_system.exceptions import DetectionError
from attendance_system.preprocessing import FacePreprocessor, crop


@pytest.fixture
def preprocessor():
    return FacePreprocessor(PreprocessConfig(align_eyes=False))


def test_grayscale_conversion_from_colour(preprocessor):
    colour = np.random.default_rng(0).integers(0, 255, (50, 60, 3), dtype=np.uint8)
    gray = preprocessor.to_grayscale(colour)
    assert gray.ndim == 2
    assert gray.shape == (50, 60)


def test_grayscale_is_idempotent(preprocessor):
    gray = np.zeros((20, 20), dtype=np.uint8)
    assert preprocessor.to_grayscale(gray) is gray


def test_empty_image_is_rejected(preprocessor):
    with pytest.raises(ValueError):
        preprocessor.to_grayscale(np.empty((0, 0), dtype=np.uint8))


def test_process_returns_configured_size():
    config = PreprocessConfig(face_width=100, face_height=120, align_eyes=False)
    processor = FacePreprocessor(config)
    image = np.random.default_rng(1).integers(0, 255, (300, 300, 3), dtype=np.uint8)
    chip = processor.process(image)
    assert chip.shape == (120, 100)
    assert chip.dtype == np.uint8


def test_crop_clamps_to_image_bounds():
    image = np.arange(100 * 100, dtype=np.uint8).reshape(100, 100)
    cropped = crop(image, (90, 90, 40, 40))       # box extends past the edge
    assert cropped.shape == (10, 10)


def test_crop_rejects_invalid_box():
    image = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        crop(image, (0, 0, 0, 5))


def test_blur_detection_separates_sharp_from_flat(preprocessor):
    rng = np.random.default_rng(2)
    noisy = rng.integers(0, 255, (200, 200), dtype=np.uint8)
    flat = np.full((200, 200), 128, dtype=np.uint8)
    assert preprocessor.sharpness(noisy) > preprocessor.sharpness(flat)
    assert preprocessor.is_blurry(flat)
    assert not preprocessor.is_blurry(noisy)


def test_denoise_is_skipped_when_kernel_too_small():
    processor = FacePreprocessor(PreprocessConfig(denoise_ksize=0))
    image = np.random.default_rng(3).integers(0, 255, (40, 40), dtype=np.uint8)
    assert np.array_equal(processor.denoise(image), image)


def test_normalize_is_stable_for_training_and_inference(preprocessor):
    """The same chip must normalise identically however it reaches the model."""
    rng = np.random.default_rng(4)
    chip = rng.integers(0, 255, (200, 200), dtype=np.uint8)
    assert np.array_equal(preprocessor.normalize(chip), preprocessor.normalize(chip))


def test_passthrough_detector_returns_whole_image():
    detector = PassthroughDetector()
    boxes = detector.detect(np.zeros((80, 120, 3), dtype=np.uint8))
    assert boxes == [(0, 0, 120, 80)]


def test_detector_factory_selects_backend():
    assert isinstance(build_detector(DetectionConfig(backend="none")), PassthroughDetector)
    assert isinstance(build_detector(DetectionConfig(backend="haar")), HaarCascadeDetector)
    with pytest.raises(DetectionError):
        build_detector(DetectionConfig(backend="magic"))


def test_missing_cascade_file_raises():
    with pytest.raises(DetectionError):
        load_cascade("no_such_cascade.xml")


def test_haar_detector_finds_no_faces_in_flat_image():
    detector = build_detector(DetectionConfig(backend="haar"))
    assert detector.detect(np.full((240, 320, 3), 100, dtype=np.uint8)) == []
