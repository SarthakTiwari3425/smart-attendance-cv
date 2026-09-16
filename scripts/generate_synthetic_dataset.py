#!/usr/bin/env python3
"""Generate the synthetic face dataset used by the offline demo.

Useful when you want the demo data without running the whole ``demo``
command, e.g. to inspect the images or to benchmark a different algorithm::

    python scripts/generate_synthetic_dataset.py --out data/dataset --samples 20
    python main.py --detector none train
    python main.py --detector none evaluate --compare
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attendance_system.logger import setup_logging          # noqa: E402
from attendance_system.synthetic import (                   # noqa: E402
    DEMO_NAMES,
    generate_classroom_frames,
    generate_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/dataset", help="dataset folder")
    parser.add_argument("--frames-out", default="data/demo_frames",
                        help="folder for unlabelled 'classroom camera' frames")
    parser.add_argument("--students", type=int, default=len(DEMO_NAMES))
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--size", type=int, default=200)
    args = parser.parse_args()

    setup_logging(None, "INFO")
    students = DEMO_NAMES[: max(2, min(args.students, len(DEMO_NAMES)))]
    written = generate_dataset(args.out, students, args.samples, args.size)
    frames = generate_classroom_frames(args.frames_out, students, 3, args.size)
    print(f"Wrote {sum(written.values())} enrolment samples to {args.out}")
    print(f"Wrote {frames} classroom frames to {args.frames_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
