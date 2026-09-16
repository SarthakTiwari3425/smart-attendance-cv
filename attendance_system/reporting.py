"""Attendance reporting.

Turns rows from the database into the two artefacts a teacher actually
wants: a readable console table and a CSV that opens in Excel.  All output
goes through :func:`render_table`, so column widths stay consistent and
nothing depends on a third-party table library.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

from .database import AttendanceDatabase, AttendanceRecord
from .logger import get_logger

logger = get_logger(__name__)


def render_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a plain-text table that lines up in any terminal."""
    if not rows:
        return "(no records)"
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    line = "-+-".join("-" * w for w in widths)
    out = [" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)), line]
    for row in rows:
        out.append(" | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
    return "\n".join(out)


def session_report(db: AttendanceDatabase, session_id: str) -> str:
    """Present + absent listing for one session."""
    present = db.session_attendance(session_id)
    absent = db.absentees(session_id)
    total = len(present) + len(absent)
    percentage = (100.0 * len(present) / total) if total else 0.0

    blocks = [
        f"Attendance report - session {session_id}",
        f"Generated {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        render_table(
            ["Student ID", "Name", "Status", "Confidence", "Marked at"],
            [
                (r.student_id, r.name, r.status, f"{r.confidence:.1f}", r.marked_at)
                for r in present
            ],
        ),
        "",
        f"Present: {len(present)}/{total} ({percentage:.1f}%)",
    ]
    if absent:
        blocks += [
            "",
            "Absent:",
            render_table(
                ["Student ID", "Name"], [(s.student_id, s.name) for s in absent]
            ),
        ]
    return "\n".join(blocks)


def summary_report(db: AttendanceDatabase) -> str:
    """Attendance percentage per student across all sessions."""
    rows = db.attendance_percentage()
    return "\n".join(
        [
            "Cumulative attendance summary",
            "",
            render_table(
                ["Student ID", "Name", "Attended", "Sessions", "Percent"],
                [(r[0], r[1], r[2], r[3], f"{r[4]:.1f}%") for r in rows],
            ),
        ]
    )


def export_csv(
    records: List[AttendanceRecord], output_path: str | Path
) -> Path:
    """Write attendance records to CSV and return the path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["session_id", "student_id", "name", "status", "confidence", "marked_at"]
        )
        for record in records:
            writer.writerow(
                [
                    record.session_id,
                    record.student_id,
                    record.name,
                    record.status,
                    f"{record.confidence:.2f}",
                    record.marked_at,
                ]
            )
    logger.info("Wrote %d rows to %s", len(records), path)
    return path


def export_session_csv(
    db: AttendanceDatabase, session_id: str, report_dir: str | Path
) -> Path:
    """Convenience wrapper: dump one session to ``<report_dir>/<session>.csv``."""
    records = db.session_attendance(session_id)
    safe = session_id.replace("/", "-")
    return export_csv(records, Path(report_dir) / f"attendance_{safe}.csv")
