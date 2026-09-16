"""Project-specific exception hierarchy.

Every failure mode the CLI can report maps to one of these, which keeps the
top-level error handling in :mod:`attendance_system.cli` short and explicit.
"""


class AttendanceSystemError(Exception):
    """Base class for all errors raised by this project."""


class ConfigurationError(AttendanceSystemError):
    """Raised when config values are missing, unknown or invalid."""


class DatasetError(AttendanceSystemError):
    """Raised for problems with the enrolled face dataset."""


class DetectionError(AttendanceSystemError):
    """Raised when a detector cannot be initialised or run."""


class ModelNotTrainedError(AttendanceSystemError):
    """Raised when recognition is attempted before a model exists."""


class StorageError(AttendanceSystemError):
    """Raised for database / persistence failures."""


class VideoSourceError(AttendanceSystemError):
    """Raised when a webcam, video file or image folder cannot be opened."""
