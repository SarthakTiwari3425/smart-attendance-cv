"""Tests for the SQLite persistence layer."""

from datetime import datetime, timedelta

import pytest

from attendance_system.database import AttendanceDatabase
from attendance_system.exceptions import StorageError


def test_add_and_fetch_student(db):
    db.add_student("S001", "Asha Verma", "asha@example.edu")
    student = db.get_student("S001")
    assert student is not None
    assert student.name == "Asha Verma"
    assert student.email == "asha@example.edu"


def test_re_enrolling_updates_instead_of_duplicating(db):
    db.add_student("S001", "Asha V")
    db.add_student("S001", "Asha Verma")
    assert len(db.list_students()) == 1
    assert db.get_student("S001").name == "Asha Verma"


def test_blank_identifiers_are_rejected(db):
    with pytest.raises(StorageError):
        db.add_student("   ", "No Id")
    with pytest.raises(StorageError):
        db.add_student("S002", "")


def test_attendance_is_idempotent_per_session(db):
    db.add_student("S001", "Asha")
    db.start_session("CV101-A", "CV101")
    assert db.mark_attendance("CV101-A", "S001", 88.0) is True
    assert db.mark_attendance("CV101-A", "S001", 91.0) is False   # already present
    assert len(db.session_attendance("CV101-A")) == 1


def test_same_student_can_attend_two_sessions(db):
    db.add_student("S001", "Asha")
    for session in ("CV101-A", "CV101-B"):
        db.start_session(session, "CV101")
        assert db.mark_attendance(session, "S001", 80.0) is True


def test_absentees_are_the_complement_of_present(db):
    db.add_student("S001", "Asha")
    db.add_student("S002", "Rahul")
    db.start_session("CV101-A", "CV101")
    db.mark_attendance("CV101-A", "S001", 80.0)
    absent_ids = [s.student_id for s in db.absentees("CV101-A")]
    assert absent_ids == ["S002"]


def test_foreign_key_blocks_unknown_student():
    db = AttendanceDatabase(":memory:")
    db.conn.execute("PRAGMA foreign_keys = ON")
    db.start_session("CV101-A", "CV101")
    with pytest.raises(StorageError):
        db.mark_attendance("CV101-A", "GHOST", 90.0)
    db.close()


def test_attendance_percentage_counts_all_sessions(db):
    db.add_student("S001", "Asha")
    db.add_student("S002", "Rahul")
    db.start_session("A", "CV101")
    db.start_session("B", "CV101")
    db.mark_attendance("A", "S001", 90.0)
    db.mark_attendance("B", "S001", 90.0)
    db.mark_attendance("A", "S002", 90.0)
    rows = {r[0]: r[4] for r in db.attendance_percentage()}
    assert rows["S001"] == 100.0
    assert rows["S002"] == 50.0


def test_recognition_log_records_rejections(db):
    db.start_session("A", "CV101")
    db.log_recognition("A", None, 120.4, "REJECTED", 7)
    rows = db.conn.execute("SELECT * FROM recognition_log").fetchall()
    assert len(rows) == 1
    assert rows[0]["decision"] == "REJECTED"


def test_purge_removes_only_old_log_rows(db):
    db.add_student("S001", "Asha")
    db.start_session("A", "CV101")
    db.log_recognition("A", "S001", 30.0, "ACCEPTED", 1)
    old = (datetime.now() - timedelta(days=90)).isoformat(timespec="seconds")
    db.conn.execute(
        "INSERT INTO recognition_log (session_id, student_id, distance, decision,"
        " frame_index, logged_at) VALUES ('A','S001',30.0,'ACCEPTED',2,?)", (old,)
    )
    db.conn.commit()
    assert db.purge_older_than(30) == 1
    assert len(db.conn.execute("SELECT * FROM recognition_log").fetchall()) == 1


def test_removing_a_student_cascades_to_attendance(db):
    db.conn.execute("PRAGMA foreign_keys = ON")
    db.add_student("S001", "Asha")
    db.start_session("A", "CV101")
    db.mark_attendance("A", "S001", 90.0)
    assert db.remove_student("S001") is True
    assert db.session_attendance("A") == []
