"""Command-line interface.

Every capability of the project is reachable from a terminal with no GUI:

    init       create folders and the database
    demo       generate a synthetic cohort, train, recognise and report
    enroll     register a student from photos or a camera
    train      fit the recogniser on the enrolled dataset
    recognize  run an attendance session over a camera / video / image folder
    report     print or export attendance
    evaluate   measure accuracy, per-class metrics and threshold behaviour
    students   list or remove enrolled students

``--display`` is opt-in everywhere, so the default behaviour is head-less.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .attendance_service import AttendanceService, default_session_id
from .config import AppConfig, PROJECT_ROOT
from .database import AttendanceDatabase
from .dataset import DatasetManager, stratified_split
from .detector import build_detector
from .evaluation import calibrate_threshold, evaluate, save_report
from .exceptions import AttendanceSystemError
from .logger import get_logger, setup_logging
from .preprocessing import FacePreprocessor
from .recognizer import FaceRecognizer
from .reporting import export_session_csv, session_report, summary_report, render_table
from .video_source import open_source

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="attendance",
        description="Smart Attendance System - classical computer-vision face recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", help="path to config.yaml", default=None)
    parser.add_argument(
        "--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument(
        "--detector", default=None, choices=["haar", "none"],
        help="face detector backend ('none' = images are already cropped faces)",
    )
    parser.add_argument(
        "--algorithm", default=None, choices=["lbph", "eigen", "fisher"],
        help="recognition algorithm",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="max accepted distance; larger = more permissive",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create data folders and an empty database")

    demo = sub.add_parser(
        "demo", help="end-to-end offline demo (no camera or dataset needed)"
    )
    demo.add_argument("--students", type=int, default=6)
    demo.add_argument("--samples", type=int, default=15)
    demo.add_argument("--keep", action="store_true", help="keep any existing dataset")

    enroll = sub.add_parser("enroll", help="register a student's face samples")
    enroll.add_argument("--id", required=True, dest="student_id")
    enroll.add_argument("--name", required=True)
    enroll.add_argument("--email", default=None)
    enroll.add_argument(
        "--source", default="0",
        help="camera index, a folder of photos, or a video file",
    )
    enroll.add_argument("--samples", type=int, default=20)
    enroll.add_argument("--display", action="store_true")

    train = sub.add_parser("train", help="train the recogniser on enrolled faces")
    train.add_argument("--test-ratio", type=float, default=0.25)
    train.add_argument(
        "--no-calibrate", action="store_true",
        help="keep the configured threshold instead of calibrating it",
    )

    recognize = sub.add_parser("recognize", help="run an attendance session")
    recognize.add_argument("--source", default="0")
    recognize.add_argument("--session", default=None, help="session id")
    recognize.add_argument("--course", default="CV101")
    recognize.add_argument("--max-frames", type=int, default=0)
    recognize.add_argument("--stride", type=int, default=1)
    recognize.add_argument("--display", action="store_true")
    recognize.add_argument("--export-csv", action="store_true")

    report = sub.add_parser("report", help="print or export attendance")
    report.add_argument("--session", default=None)
    report.add_argument("--summary", action="store_true")
    report.add_argument("--list-sessions", action="store_true")
    report.add_argument("--csv", action="store_true", help="also export a CSV")

    evaluate_cmd = sub.add_parser("evaluate", help="measure recognition accuracy")
    evaluate_cmd.add_argument("--test-ratio", type=float, default=0.3)
    evaluate_cmd.add_argument("--seed", type=int, default=42)
    evaluate_cmd.add_argument(
        "--compare", action="store_true", help="evaluate all three algorithms"
    )

    students = sub.add_parser("students", help="manage enrolled students")
    students.add_argument("--remove", default=None, metavar="STUDENT_ID")

    return parser


# ----------------------------------------------------------------------
# Wiring
# ----------------------------------------------------------------------
def load_config(args: argparse.Namespace) -> AppConfig:
    """Load config.yaml then apply command-line overrides."""
    config = AppConfig.load(args.config).resolved(PROJECT_ROOT)
    if args.detector:
        config.detection.backend = args.detector
    if args.algorithm:
        config.recognition.algorithm = args.algorithm
    if args.threshold is not None:
        config.recognition.distance_threshold = args.threshold
    if args.log_level:
        config.log_level = args.log_level
    config.validate()
    return config


def build_components(config: AppConfig):
    """Construct detector, preprocessor and dataset manager from config."""
    from .detector import load_cascade

    eye_cascade = None
    if config.preprocess.align_eyes and config.detection.backend == "haar":
        try:
            eye_cascade = load_cascade(config.detection.eye_cascade)
        except AttendanceSystemError as exc:
            logger.warning("Eye alignment disabled: %s", exc)
    detector = build_detector(config.detection)
    preprocessor = FacePreprocessor(config.preprocess, eye_cascade=eye_cascade)
    manager = DatasetManager(config.paths.dataset_dir, detector, preprocessor)
    return detector, preprocessor, manager


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------
def cmd_init(args, config: AppConfig) -> int:
    for path in (
        config.paths.dataset_dir,
        config.paths.model_dir,
        config.paths.report_dir,
        Path(config.paths.database_path).parent,
    ):
        Path(path).mkdir(parents=True, exist_ok=True)
    with AttendanceDatabase(config.paths.database_path):
        pass
    print("Initialised:")
    print(f"  dataset  : {config.paths.dataset_dir}")
    print(f"  database : {config.paths.database_path}")
    print(f"  models   : {config.paths.model_dir}")
    print(f"  reports  : {config.paths.report_dir}")
    return 0


def cmd_enroll(args, config: AppConfig) -> int:
    detector, preprocessor, manager = build_components(config)
    with AttendanceDatabase(config.paths.database_path) as db:
        db.add_student(args.student_id, args.name, args.email)
        spec = str(args.source)
        if spec.isdigit():
            source = open_source(spec)
            added = manager.enroll_from_camera(
                args.student_id, args.name, source, args.samples, args.display
            )
            source.release()
        else:
            path = Path(spec)
            if path.is_dir():
                added = manager.enroll_from_images(
                    args.student_id, args.name, path, limit=args.samples
                )
            else:
                source = open_source(spec)
                added = manager.enroll_from_camera(
                    args.student_id, args.name, source, args.samples, args.display
                )
                source.release()
        total = manager.counts().get(args.student_id, added)
        db.set_sample_count(args.student_id, total)
    print(f"Enrolled {added} new sample(s) for {args.student_id} ({args.name}).")
    print(f"Total samples on file: {total}")
    print("Next step: python main.py train")
    return 0


def cmd_train(args, config: AppConfig) -> int:
    _, _, manager = build_components(config)
    data = manager.load(min_samples=config.recognition.min_samples_per_student)
    recognizer = FaceRecognizer(config.recognition)

    threshold = config.recognition.distance_threshold
    if not args.no_calibrate and len(set(data.labels)) >= 2:
        train_split, holdout = stratified_split(data, args.test_ratio)
        probe = FaceRecognizer(config.recognition)
        probe.train(train_split)
        threshold = calibrate_threshold(probe, holdout)
        recognizer.config.distance_threshold = threshold

    recognizer.train(data)                     # final model uses every sample
    model_path = recognizer.save(config.paths.model_dir)

    with AttendanceDatabase(config.paths.database_path) as db:
        for student_id, count in manager.counts().items():
            if db.get_student(student_id) is None:
                db.add_student(student_id, data.names.get(student_id, student_id))
            db.set_sample_count(student_id, count)

    print(f"Trained {config.recognition.algorithm.upper()} on {len(data)} samples "
          f"from {len(set(data.labels))} students.")
    print(f"Rejection threshold: {threshold:.2f}"
          f"{'' if args.no_calibrate else ' (auto-calibrated)'}")
    print(f"Model saved to {model_path}")
    return 0


def cmd_recognize(args, config: AppConfig) -> int:
    detector, preprocessor, _ = build_components(config)
    recognizer = FaceRecognizer.load(config.paths.model_dir, config.recognition)
    if args.threshold is not None:
        recognizer.config.distance_threshold = args.threshold

    session_id = args.session or default_session_id(args.course)
    stride = max(args.stride, config.attendance.frame_stride)
    source = open_source(args.source, max_frames=args.max_frames, stride=stride)

    with AttendanceDatabase(config.paths.database_path) as db:
        service = AttendanceService(config, db, detector, preprocessor, recognizer)
        stats = service.run_session(
            source, session_id, args.course, display=args.display,
            source_label=str(args.source),
        )
        print()
        print(session_report(db, session_id))
        print()
        print(render_table(
            ["Frames", "Faces", "Accepted", "Rejected", "Newly marked"],
            [(stats.frames_processed, stats.faces_detected, stats.accepted,
              stats.rejected, len(stats.marked))],
        ))
        if args.export_csv:
            path = export_session_csv(db, session_id, config.paths.report_dir)
            print(f"\nCSV written to {path}")
    return 0


def cmd_report(args, config: AppConfig) -> int:
    with AttendanceDatabase(config.paths.database_path) as db:
        if args.list_sessions:
            rows = db.list_sessions()
            print(render_table(
                ["Session", "Course", "Started", "Ended", "Source"],
                [(r["session_id"], r["course_code"], r["started_at"],
                  r["ended_at"] or "-", r["source"] or "-") for r in rows],
            ))
            return 0
        if args.summary or not args.session:
            print(summary_report(db))
            if not args.session:
                return 0
        print()
        print(session_report(db, args.session))
        if args.csv:
            path = export_session_csv(db, args.session, config.paths.report_dir)
            print(f"\nCSV written to {path}")
    return 0


def cmd_evaluate(args, config: AppConfig) -> int:
    _, _, manager = build_components(config)
    data = manager.load(min_samples=2)
    train_split, test_split = stratified_split(data, args.test_ratio, seed=args.seed)

    algorithms = ["lbph", "eigen", "fisher"] if args.compare else [config.recognition.algorithm]
    rows = []
    results = []
    for algorithm in algorithms:
        settings = config.recognition.__class__(**{**config.recognition.__dict__,
                                                   "algorithm": algorithm})
        recognizer = FaceRecognizer(settings)
        try:
            recognizer.train(train_split)
        except AttendanceSystemError as exc:
            logger.warning("Skipping %s: %s", algorithm, exc)
            continue
        threshold = calibrate_threshold(recognizer, test_split)
        result = evaluate(recognizer, test_split, threshold=threshold, n_train=len(train_split))
        results.append(result)
        rows.append((
            algorithm.upper(),
            f"{result.accuracy * 100:.2f}%",
            f"{result.accuracy_with_threshold * 100:.2f}%",
            f"{threshold:.1f}",
            f"{result.mean_distance_correct:.1f}",
            f"{result.mean_distance_wrong:.1f}",
        ))

    print(render_table(
        ["Algorithm", "Rank-1 acc", "Acc @ threshold", "Threshold",
         "Mean dist (correct)", "Mean dist (wrong)"],
        rows,
    ))
    for result in results:
        # One folder per algorithm, so a --compare run does not overwrite itself.
        target = (
            Path(config.paths.report_dir) / result.algorithm
            if len(results) > 1
            else Path(config.paths.report_dir)
        )
        written = save_report(result, target)
        if result is results[0]:
            print()
            print(render_table(
                ["Student", "Support", "Precision", "Recall", "F1"],
                [(m.student_id, m.support, f"{m.precision:.2f}", f"{m.recall:.2f}",
                  f"{m.f1:.2f}") for m in result.per_class],
            ))
            print()
        print(f"Artefacts for {result.algorithm.upper()}:")
        for kind, path in written.items():
            print(f"  {kind:4s} -> {path}")
    return 0


def cmd_students(args, config: AppConfig) -> int:
    _, _, manager = build_components(config)
    with AttendanceDatabase(config.paths.database_path) as db:
        if args.remove:
            removed_db = db.remove_student(args.remove)
            removed_files = manager.delete_student(args.remove)
            print(
                f"Removed {args.remove}: database={removed_db}, samples={removed_files}"
            )
            print("Re-run 'train' so the model forgets this student.")
            return 0
        students = db.list_students()
        counts = manager.counts()
        print(render_table(
            ["Student ID", "Name", "Samples (disk)", "Enrolled at"],
            [(s.student_id, s.name, counts.get(s.student_id, 0), s.enrolled_at)
             for s in students],
        ))
    return 0


def cmd_demo(args, config: AppConfig) -> int:
    """Generate data, train, recognise and report - one command, no camera."""
    import shutil

    from .synthetic import DEMO_NAMES, generate_classroom_frames, generate_dataset

    config.detection.backend = "none"          # synthetic faces are pre-cropped
    config.preprocess.align_eyes = False
    students = DEMO_NAMES[: max(2, min(args.students, len(DEMO_NAMES)))]

    dataset_dir = Path(config.paths.dataset_dir)
    if dataset_dir.exists() and not args.keep:
        shutil.rmtree(dataset_dir)
    frames_dir = Path(config.paths.dataset_dir).parent / "demo_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)

    print("STEP 1/5  Generating a synthetic cohort")
    generate_dataset(dataset_dir, students=students, samples_per_student=args.samples)
    generate_classroom_frames(frames_dir, students=students, frames_per_student=3)

    print("STEP 2/5  Training the recogniser")
    train_args = argparse.Namespace(test_ratio=0.3, no_calibrate=False)
    cmd_train(train_args, config)

    print("\nSTEP 3/5  Evaluating on a held-out split")
    evaluate_args = argparse.Namespace(test_ratio=0.3, seed=42, compare=True)
    cmd_evaluate(evaluate_args, config)

    print("\nSTEP 4/5  Running an attendance session over the demo frames")
    session_id = default_session_id("DEMO")
    recognize_args = argparse.Namespace(
        source=str(frames_dir), session=session_id, course="DEMO",
        max_frames=0, stride=1, display=False, export_csv=True, threshold=None,
    )
    cmd_recognize(recognize_args, config)

    print("\nSTEP 5/5  Cumulative summary")
    report_args = argparse.Namespace(
        session=None, summary=True, list_sessions=False, csv=False
    )
    cmd_report(report_args, config)
    print(
        "\nDemo complete. The unenrolled 'stranger' frames should appear as "
        "rejections, not as attendance rows."
    )
    return 0


COMMANDS = {
    "init": cmd_init,
    "demo": cmd_demo,
    "enroll": cmd_enroll,
    "train": cmd_train,
    "recognize": cmd_recognize,
    "report": cmd_report,
    "evaluate": cmd_evaluate,
    "students": cmd_students,
}


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point used by ``main.py`` and ``python -m attendance_system``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args)
    except AttendanceSystemError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    setup_logging(config.paths.log_file, config.log_level)
    handler = COMMANDS[args.command]
    try:
        return handler(args, config)
    except AttendanceSystemError as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:                        # pragma: no cover
        logger.exception("Unexpected failure: %s", exc)
        return 1


if __name__ == "__main__":                          # pragma: no cover
    raise SystemExit(main())
