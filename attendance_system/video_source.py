"""Frame sources.

The recognition pipeline does not care where frames come from, so three
sources implement one tiny interface (``frames()`` yielding
``(index, frame)``):

* :class:`CameraSource`    - a live webcam, by device index
* :class:`VideoFileSource` - any video file OpenCV can decode
* :class:`ImageFolderSource` - a folder of stills, which is what makes the
  whole project runnable head-less on a machine with no camera

``open_source`` picks the right one from a command-line string, so
``--source 0``, ``--source clip.mp4`` and ``--source photos/`` all work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np

from .exceptions import VideoSourceError
from .logger import get_logger

logger = get_logger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".tif", ".tiff"}
Frame = Tuple[int, np.ndarray]


class BaseSource:
    """Common context-manager behaviour for all frame sources."""

    stride: int = 1

    def frames(self) -> Iterator[Frame]:          # pragma: no cover - interface
        raise NotImplementedError

    def release(self) -> None:
        pass

    def __enter__(self) -> "BaseSource":
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()


class CameraSource(BaseSource):
    """Live webcam capture."""

    def __init__(self, device: int = 0, max_frames: int = 0, stride: int = 1):
        self.capture = cv2.VideoCapture(device)
        if not self.capture.isOpened():
            raise VideoSourceError(
                f"Could not open camera {device}. "
                "Check that a webcam is connected, or use --source <folder|video>."
            )
        self.max_frames = max_frames
        self.stride = max(1, stride)

    def frames(self) -> Iterator[Frame]:
        index = 0
        while True:
            ok, frame = self.capture.read()
            if not ok:
                logger.warning("Camera returned no frame; stopping")
                break
            if index % self.stride == 0:
                yield index, frame
            index += 1
            if self.max_frames and index >= self.max_frames:
                break

    def release(self) -> None:
        self.capture.release()


class VideoFileSource(BaseSource):
    """Decode frames from a video file."""

    def __init__(self, path: str | Path, max_frames: int = 0, stride: int = 1):
        self.path = Path(path)
        if not self.path.is_file():
            raise VideoSourceError(f"Video file not found: {self.path}")
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise VideoSourceError(f"OpenCV cannot decode {self.path}")
        self.max_frames = max_frames
        self.stride = max(1, stride)

    def frames(self) -> Iterator[Frame]:
        index = 0
        emitted = 0
        while True:
            ok, frame = self.capture.read()
            if not ok:
                break
            if index % self.stride == 0:
                yield index, frame
                emitted += 1
                if self.max_frames and emitted >= self.max_frames:
                    break
            index += 1

    def release(self) -> None:
        self.capture.release()


class ImageFolderSource(BaseSource):
    """Iterate over still images in a folder (head-less friendly)."""

    def __init__(self, path: str | Path, max_frames: int = 0, stride: int = 1):
        self.path = Path(path)
        if not self.path.exists():
            raise VideoSourceError(f"Path not found: {self.path}")
        if self.path.is_file():
            self.files = [self.path]
        else:
            self.files = sorted(
                p for p in self.path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
            )
        if not self.files:
            raise VideoSourceError(f"No images found under {self.path}")
        self.stride = max(1, stride)
        if max_frames:
            self.files = self.files[: max_frames * self.stride]

    def frames(self) -> Iterator[Frame]:
        for index, file in enumerate(self.files):
            if index % self.stride:
                continue
            image = cv2.imread(str(file), cv2.IMREAD_COLOR)
            if image is None:
                logger.warning("Skipping unreadable image %s", file)
                continue
            yield index, image

    def __len__(self) -> int:
        return len(self.files)


def open_source(spec: str, max_frames: int = 0, stride: int = 1) -> BaseSource:
    """Resolve a ``--source`` string into a concrete frame source.

    ``"0"``/``"1"`` -> webcam index, an existing directory -> image folder,
    an image file -> single frame, anything else that exists -> video file.
    """
    spec = str(spec)
    if spec.isdigit():
        return CameraSource(int(spec), max_frames=max_frames, stride=stride)
    path = Path(spec)
    if not path.exists():
        raise VideoSourceError(
            f"Source not found: {spec}. Pass a camera index (0), a folder of "
            "images, or a video file."
        )
    if path.is_dir() or path.suffix.lower() in IMAGE_SUFFIXES:
        return ImageFolderSource(path, max_frames=max_frames, stride=stride)
    return VideoFileSource(path, max_frames=max_frames, stride=stride)
