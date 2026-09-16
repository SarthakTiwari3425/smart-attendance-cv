#!/usr/bin/env python3
"""Single entry point for the Smart Attendance System.

Usage examples
--------------
    python main.py demo                       # full offline walkthrough
    python main.py enroll --id S001 --name "Asha Verma" --source photos/asha
    python main.py train
    python main.py recognize --source 0 --course CV101
    python main.py report --session CV101-20260916-0930 --csv
    python main.py evaluate --compare
"""

import sys

from attendance_system.cli import main

if __name__ == "__main__":
    sys.exit(main())
