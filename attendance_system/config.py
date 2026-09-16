"""Configuration handling for the attendance system.

All tunable parameters live here so that no magic numbers are scattered
through the pipeline.  Values are loaded from ``config.yaml`` when present
and can be overridden from the command line.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict

from .exceptions import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class PathConfig:
    """Filesystem locations used by the project."""

    dataset_dir: str = "data/dataset"
    database_path: str = "data/attendance.db"
    model_dir: str = "models"
    report_dir: str = "reports"
    log_file: str = "logs/attendance.log"

    def resolve(self, root: Path) -> "PathConfig":
        """Return a copy with every path made absolute relative to *root*."""
        return PathConfig(
            dataset_dir=str((root / self.dataset_dir).resolve()),
            database_path=str((root / self.database_path).resolve()),
            model_dir=str((root / self.model_dir).resolve()),
            report_dir=str((root / self.report_dir).resolve()),
            log_file=str((root / self.log_file).resolve()),
        )


@dataclass
class DetectionConfig:
    """Haar-cascade face detection parameters."""

    backend: str = "haar"           # "haar" | "none"
    cascade: str = "haarcascade_frontalface_default.xml"
    eye_cascade: str = "haarcascade_eye.xml"
    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_face_size: int = 60
    max_faces_per_frame: int = 10


@dataclass
class PreprocessConfig:
    """Normalisation applied to every detected face before recognition."""

    face_width: int = 200
    face_height: int = 200
    use_clahe: bool = True
    denoise_ksize: int = 5           # Gaussian kernel; 0 disables denoising
    clahe_clip_limit: float = 2.0
    clahe_grid: int = 8
    align_eyes: bool = True
    blur_rejection_threshold: float = 25.0   # variance of Laplacian


@dataclass
class RecognitionConfig:
    """Face recogniser parameters."""

    algorithm: str = "lbph"         # "lbph" | "eigen" | "fisher"
    lbph_radius: int = 1
    lbph_neighbors: int = 8
    lbph_grid_x: int = 8
    lbph_grid_y: int = 8
    num_components: int = 0          # eigen/fisher; 0 = keep all
    # LBPH returns a *distance*: smaller is better.  A probe whose distance
    # exceeds this threshold is reported as "unknown".
    distance_threshold: float = 75.0
    min_samples_per_student: int = 5


@dataclass
class AttendanceConfig:
    """Business rules for marking attendance."""

    consecutive_hits_required: int = 3
    cooldown_seconds: int = 300      # do not re-mark a student within a session
    frame_stride: int = 1            # process every Nth frame of a video
    late_after_minutes: int = 10     # status becomes LATE after this


@dataclass
class AppConfig:
    """Root configuration object."""

    paths: PathConfig = field(default_factory=PathConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    attendance: AttendanceConfig = field(default_factory=AttendanceConfig)
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        """Build a config from a nested dictionary, validating key names."""
        known = {
            "paths": PathConfig,
            "detection": DetectionConfig,
            "preprocess": PreprocessConfig,
            "recognition": RecognitionConfig,
            "attendance": AttendanceConfig,
        }
        kwargs: Dict[str, Any] = {}
        for section, value in (data or {}).items():
            if section == "log_level":
                kwargs["log_level"] = str(value)
                continue
            if section not in known:
                raise ConfigurationError(f"Unknown configuration section: {section!r}")
            if not isinstance(value, dict):
                raise ConfigurationError(f"Section {section!r} must be a mapping")
            dataclass_type = known[section]
            valid_fields = {f for f in dataclass_type.__dataclass_fields__}
            unknown = set(value) - valid_fields
            if unknown:
                raise ConfigurationError(
                    f"Unknown keys in section {section!r}: {sorted(unknown)}"
                )
            kwargs[section] = dataclass_type(**value)
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "AppConfig":
        """Load configuration from a YAML (or JSON) file.

        Falls back to built-in defaults when the file is missing, so the
        project runs out of the box on a clean checkout.
        """
        config_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not config_path.exists():
            if path:                     # explicitly requested but missing
                raise ConfigurationError(f"Config file not found: {config_path}")
            return cls()
        text = config_path.read_text(encoding="utf-8")
        data = _parse_config_text(text, config_path)
        config = cls.from_dict(data)
        config.validate()
        return config

    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Sanity-check values that would otherwise fail deep in the pipeline."""
        if self.recognition.algorithm not in {"lbph", "eigen", "fisher"}:
            raise ConfigurationError(
                f"Unsupported algorithm: {self.recognition.algorithm!r}"
            )
        if self.detection.backend not in {"haar", "none"}:
            raise ConfigurationError(
                f"Unsupported detection backend: {self.detection.backend!r}"
            )
        if self.preprocess.face_width <= 0 or self.preprocess.face_height <= 0:
            raise ConfigurationError("Face size must be positive")
        if self.recognition.distance_threshold <= 0:
            raise ConfigurationError("distance_threshold must be positive")
        if self.attendance.consecutive_hits_required < 1:
            raise ConfigurationError("consecutive_hits_required must be >= 1")

    def resolved(self, root: Path | None = None) -> "AppConfig":
        """Return a copy whose paths are absolute."""
        root = root or PROJECT_ROOT
        clone = AppConfig.from_dict(json.loads(json.dumps(asdict(self))))
        clone.paths = self.paths.resolve(root)
        return clone

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parse_config_text(text: str, source: Path) -> Dict[str, Any]:
    """Parse YAML if PyYAML is installed, else fall back to a tiny parser.

    The fallback keeps the project runnable with zero optional dependencies;
    it understands the flat ``section:`` / ``  key: value`` layout used by
    ``config.yaml``.
    """
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except ImportError:
        pass

    data: Dict[str, Any] = {}
    section: Dict[str, Any] | None = None
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith((" ", "\t")):
            key = line.rstrip(":").strip()
            if not line.endswith(":"):
                data[key] = _coerce(line.split(":", 1)[1].strip())
                section = None
                continue
            section = {}
            data[key] = section
        else:
            if section is None:
                raise ConfigurationError(
                    f"{source}:{lineno}: indented key outside of a section"
                )
            if ":" not in line:
                raise ConfigurationError(f"{source}:{lineno}: expected 'key: value'")
            key, value = line.split(":", 1)
            section[key.strip()] = _coerce(value.strip())
    return data


def _coerce(value: str) -> Any:
    """Convert a scalar string from the config file to a Python value."""
    lowered = value.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if lowered in {"null", "none", ""}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip('"\'')
