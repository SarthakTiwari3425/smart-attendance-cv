"""SQLite persistence layer (repository pattern).

The schema has four tables:

* ``students``        - one row per enrolled person
* ``sessions``        - one row per class/attendance session
* ``attendance``      - the authoritative record; unique per (session, student)
* ``recognition_log`` - every recognition decision, including rejections,
                        which makes the system auditable and supports
                        threshold tuning after the fact.

Foreign keys are enforced and all writes go through parameterised queries,
which addresses the security non-functional requirement (no SQL injection is
possible even though student IDs come from the command line).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from .exceptions import StorageError
from .logger import get_logger

logger = get_logger(__name__)

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS students (
    student_id   TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    email        TEXT,
    sample_count INTEGER NOT NULL DEFAULT 0,
    enrolled_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    course_code TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    source      TEXT
);

CREATE TABLE IF NOT EXISTS attendance (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    student_id  TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    marked_at   TEXT NOT NULL,
    confidence  REAL NOT NULL,
    status      TEXT NOT NULL DEFAULT 'PRESENT',
    UNIQUE (session_id, student_id)
);

CREATE TABLE IF NOT EXISTS recognition_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    student_id  TEXT,
    distance    REAL NOT NULL,
    decision    TEXT NOT NULL,
    frame_index INTEGER NOT NULL DEFAULT 0,
    logged_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_attendance_session ON attendance(session_id);
CREATE INDEX IF NOT EXISTS idx_attendance_student ON attendance(student_id);
CREATE INDEX IF NOT EXISTS idx_log_session ON recognition_log(session_id);
"""


@dataclass
class Student:
    student_id: str
    name: str
    email: Optional[str] = None
    sample_count: int = 0
    enrolled_at: str = ""


@dataclass
class AttendanceRecord:
    session_id: str
    student_id: str
    name: str
    marked_at: str
    confidence: float
    status: str


class AttendanceDatabase:
    """Thin, explicit repository over SQLite.

    Usable as a context manager::

        with AttendanceDatabase(path) as db:
            db.add_student("S001", "Asha")
    """

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            self.conn = sqlite3.connect(self.db_path)
        except sqlite3.Error as exc:                      # pragma: no cover
            raise StorageError(f"Cannot open database {self.db_path}: {exc}") from exc
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        logger.debug("Database ready at %s", self.db_path)

    # ------------------------------------------------------------------
    def __enter__(self) -> "AttendanceDatabase":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        try:
            self.conn.close()
        except sqlite3.Error:                             # pragma: no cover
            pass

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Cursor]:
        """Transactional cursor: commit on success, roll back on failure."""
        cursor = self.conn.cursor()
        try:
            yield cursor
            self.conn.commit()
        except sqlite3.Error as exc:
            self.conn.rollback()
            raise StorageError(str(exc)) from exc
        finally:
            cursor.close()

    # ------------------------------------------------------------------
    # Students
    # ------------------------------------------------------------------
    def add_student(
        self, student_id: str, name: str, email: Optional[str] = None
    ) -> Student:
        """Insert a student, or update the name/email if the ID already exists."""
        student_id = student_id.strip()
        name = name.strip()
        if not student_id or not name:
            raise StorageError("student_id and name are required")
        now = datetime.now().isoformat(timespec="seconds")
        with self._write() as cur:
            cur.execute(
                """
                INSERT INTO students (student_id, name, email, sample_count, enrolled_at)
                VALUES (?, ?, ?, 0, ?)
                ON CONFLICT(student_id) DO UPDATE SET
                    name = excluded.name,
                    email = COALESCE(excluded.email, students.email)
                """,
                (student_id, name, email, now),
            )
        return self.get_student(student_id)                # type: ignore[return-value]

    def get_student(self, student_id: str) -> Optional[Student]:
        row = self.conn.execute(
            "SELECT * FROM students WHERE student_id = ?", (student_id,)
        ).fetchone()
        return _to_student(row) if row else None

    def list_students(self) -> List[Student]:
        rows = self.conn.execute(
            "SELECT * FROM students ORDER BY student_id"
        ).fetchall()
        return [_to_student(r) for r in rows]

    def set_sample_count(self, student_id: str, count: int) -> None:
        with self._write() as cur:
            cur.execute(
                "UPDATE students SET sample_count = ? WHERE student_id = ?",
                (int(count), student_id),
            )

    def remove_student(self, student_id: str) -> bool:
        with self._write() as cur:
            cur.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------
    def start_session(
        self, session_id: str, course_code: str, source: str = "webcam"
    ) -> str:
        now = datetime.now().isoformat(timespec="seconds")
        with self._write() as cur:
            cur.execute(
                """
                INSERT INTO sessions (session_id, course_code, started_at, source)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO NOTHING
                """,
                (session_id, course_code, now, source),
            )
        return session_id

    def end_session(self, session_id: str) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self._write() as cur:
            cur.execute(
                "UPDATE sessions SET ended_at = ? WHERE session_id = ?",
                (now, session_id),
            )

    def get_session_start(self, session_id: str) -> Optional[datetime]:
        row = self.conn.execute(
            "SELECT started_at FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return datetime.fromisoformat(row["started_at"]) if row else None

    def list_sessions(self) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC"
        ).fetchall()

    # ------------------------------------------------------------------
    # Attendance
    # ------------------------------------------------------------------
    def mark_attendance(
        self,
        session_id: str,
        student_id: str,
        confidence: float,
        status: str = "PRESENT",
        marked_at: Optional[datetime] = None,
    ) -> bool:
        """Record a student as present.

        Returns ``True`` if a new row was written, ``False`` if the student was
        already marked for this session (idempotent by design - a student who
        walks past the camera ten times is still one attendance row).
        """
        timestamp = (marked_at or datetime.now()).isoformat(timespec="seconds")
        with self._write() as cur:
            cur.execute(
                """
                INSERT INTO attendance (session_id, student_id, marked_at, confidence, status)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id, student_id) DO NOTHING
                """,
                (session_id, student_id, timestamp, float(confidence), status),
            )
            return cur.rowcount > 0

    def is_marked(self, session_id: str, student_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM attendance WHERE session_id = ? AND student_id = ?",
            (session_id, student_id),
        ).fetchone()
        return row is not None

    def log_recognition(
        self,
        session_id: str,
        student_id: Optional[str],
        distance: float,
        decision: str,
        frame_index: int = 0,
    ) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self._write() as cur:
            cur.execute(
                """
                INSERT INTO recognition_log
                    (session_id, student_id, distance, decision, frame_index, logged_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, student_id, float(distance), decision, int(frame_index), now),
            )

    def session_attendance(self, session_id: str) -> List[AttendanceRecord]:
        rows = self.conn.execute(
            """
            SELECT a.session_id, a.student_id, s.name, a.marked_at, a.confidence, a.status
            FROM attendance a
            JOIN students s ON s.student_id = a.student_id
            WHERE a.session_id = ?
            ORDER BY a.marked_at
            """,
            (session_id,),
        ).fetchall()
        return [AttendanceRecord(**dict(r)) for r in rows]

    def absentees(self, session_id: str) -> List[Student]:
        rows = self.conn.execute(
            """
            SELECT * FROM students
            WHERE student_id NOT IN (
                SELECT student_id FROM attendance WHERE session_id = ?
            )
            ORDER BY student_id
            """,
            (session_id,),
        ).fetchall()
        return [_to_student(r) for r in rows]

    def attendance_between(self, start: str, end: str) -> List[AttendanceRecord]:
        rows = self.conn.execute(
            """
            SELECT a.session_id, a.student_id, s.name, a.marked_at, a.confidence, a.status
            FROM attendance a
            JOIN students s ON s.student_id = a.student_id
            WHERE date(a.marked_at) BETWEEN date(?) AND date(?)
            ORDER BY a.marked_at
            """,
            (start, end),
        ).fetchall()
        return [AttendanceRecord(**dict(r)) for r in rows]

    def attendance_percentage(self) -> List[tuple]:
        """Per-student attendance rate across every recorded session."""
        total = self.conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"]
        rows = self.conn.execute(
            """
            SELECT s.student_id, s.name, COUNT(a.id) AS attended
            FROM students s
            LEFT JOIN attendance a ON a.student_id = s.student_id
            GROUP BY s.student_id, s.name
            ORDER BY s.student_id
            """
        ).fetchall()
        return [
            (
                r["student_id"],
                r["name"],
                r["attended"],
                total,
                round(100.0 * r["attended"] / total, 1) if total else 0.0,
            )
            for r in rows
        ]

    def purge_older_than(self, days: int) -> int:
        """Data-retention helper: delete logs older than *days* (privacy NFR)."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
        with self._write() as cur:
            cur.execute("DELETE FROM recognition_log WHERE logged_at < ?", (cutoff,))
            return cur.rowcount


def _to_student(row: sqlite3.Row) -> Student:
    return Student(
        student_id=row["student_id"],
        name=row["name"],
        email=row["email"],
        sample_count=row["sample_count"],
        enrolled_at=row["enrolled_at"],
    )


def ensure_students(db: AttendanceDatabase, students: Iterable[tuple]) -> None:
    """Bulk helper used by the demo/seed paths."""
    for student_id, name in students:
        db.add_student(student_id, name)
