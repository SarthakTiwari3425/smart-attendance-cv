"""Allows ``python -m attendance_system ...`` as an alternative to main.py."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
