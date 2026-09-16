"""Centralised logging configuration.

A rotating file handler gives the non-functional "logging & monitoring"
requirement a concrete implementation, while the console handler keeps the
CLI readable.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"


def setup_logging(log_file: str | Path | None = None, level: str = "INFO") -> None:
    """Configure the root logger exactly once per process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    console = logging.StreamHandler(stream=sys.stderr)
    console.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
    root.addHandler(console)

    if log_file:
        path = Path(log_file)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                path, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
            )
            file_handler.setFormatter(logging.Formatter(_FORMAT))
            root.addHandler(file_handler)
        except OSError as exc:            # read-only fs, permissions, ...
            root.warning("File logging disabled (%s)", exc)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name)
