"""
database/db_manager.py
──────────────────────
Handles all SQLite database operations:
  - Creating tables (students, attendance logs)
  - Inserting / querying students
  - Marking attendance (with duplicate prevention)
  - Fetching reports and summaries

SQLite is used because it requires zero server setup —
the entire database lives in a single .db file.
"""

import sqlite3
import os
from datetime import date, datetime

# ── Path to the SQLite database file (created automatically) ──────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "attendance.db")


def get_connection():
    """
    Open and return a connection to the SQLite database.
    'check_same_thread=False' is needed for Flask's multi-threaded context.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row   # Rows behave like dicts (access by column name)
    return conn


def init_db():
    """
    Create all required tables if they don't already exist.
    Call this once at project startup.

    Tables:
      students     – stores student info + serialized face encoding
      attendance   – one record per student per day, includes attention score
    """
    conn = get_connection()
    cursor = conn.cursor()

    # ── students table ──────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id  TEXT    UNIQUE NOT NULL,   -- e.g. "STU001"
            name        TEXT    NOT NULL,
            encoding    BLOB    NOT NULL,           -- pickled numpy array (128 floats)
            registered_on TEXT  DEFAULT (date('now'))
        )
    """)

    # ── attendance table ────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id      TEXT    NOT NULL,
            name            TEXT    NOT NULL,
            date            TEXT    NOT NULL,       -- "YYYY-MM-DD"
            time            TEXT    NOT NULL,       -- "HH:MM:SS"
            attention_score INTEGER DEFAULT 100,    -- 0-100 (100 = fully attentive)
            status          TEXT    DEFAULT 'Attentive',  -- 'Attentive' / 'Distracted'
            UNIQUE(student_id, date)               -- prevent duplicate attendance
        )
    """)

    conn.commit()
    conn.close()
    print("[DB] Database initialized successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# STUDENT OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def add_student(student_id: str, name: str, encoding_bytes: bytes) -> bool:
    """
    Insert a new student record.
    encoding_bytes = pickle.dumps(numpy_array)  — stored as binary BLOB.

    Returns True on success, False if student_id already exists.
    """
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO students (student_id, name, encoding) VALUES (?, ?, ?)",
            (student_id, name, encoding_bytes)
        )
        conn.commit()
        print(f"[DB] Student '{name}' ({student_id}) registered.")
        return True
    except sqlite3.IntegrityError:
        print(f"[DB] Student ID '{student_id}' already exists.")
        return False
    finally:
        conn.close()


def get_all_students() -> list:
    """Return all student records as a list of Row objects."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM students ORDER BY name").fetchall()
    conn.close()
    return rows


def get_student_by_id(student_id: str):
    """Fetch a single student row by their student_id."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM students WHERE student_id = ?", (student_id,)
    ).fetchone()
    conn.close()
    return row


# ─────────────────────────────────────────────────────────────────────────────
# ATTENDANCE OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def mark_attendance(student_id: str, name: str,
                    attention_score: int = 100, status: str = "Attentive") -> bool:
    """
    Mark a student as present for today.
    The UNIQUE(student_id, date) constraint ensures no duplicate entries.

    Returns True if attendance was newly marked, False if already marked today.
    """
    today = date.today().isoformat()           # "YYYY-MM-DD"
    now   = datetime.now().strftime("%H:%M:%S") # "HH:MM:SS"

    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO attendance (student_id, name, date, time, attention_score, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (student_id, name, today, now, attention_score, status))
        conn.commit()
        print(f"[ATTENDANCE] ✅ {name} marked Present at {now}")
        return True
    except sqlite3.IntegrityError:
        # Already marked today — silently ignore
        return False
    finally:
        conn.close()


def update_attention(student_id: str, attention_score: int, status: str):
    """
    Update today's attention score and status for an already-marked student.
    Called periodically during a session to keep the score fresh.
    """
    today = date.today().isoformat()
    conn  = get_connection()
    conn.execute("""
        UPDATE attendance
        SET attention_score = ?, status = ?
        WHERE student_id = ? AND date = ?
    """, (attention_score, status, student_id, today))
    conn.commit()
    conn.close()


def get_attendance_today() -> list:
    """Fetch all attendance records for today."""
    today = date.today().isoformat()
    conn  = get_connection()
    rows  = conn.execute("""
        SELECT * FROM attendance WHERE date = ? ORDER BY time
    """, (today,)).fetchall()
    conn.close()
    return rows


def get_attendance_all() -> list:
    """Fetch all attendance records (all dates), newest first."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM attendance ORDER BY date DESC, time DESC"
    ).fetchall()
    conn.close()
    return rows


def get_summary_today() -> dict:
    """
    Return a summary dict for today:
      { total_students, present, absent, attentive, distracted }
    """
    today      = date.today().isoformat()
    conn       = get_connection()

    total      = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    present    = conn.execute(
        "SELECT COUNT(*) FROM attendance WHERE date = ?", (today,)
    ).fetchone()[0]
    attentive  = conn.execute(
        "SELECT COUNT(*) FROM attendance WHERE date = ? AND status = 'Attentive'",
        (today,)
    ).fetchone()[0]
    distracted = conn.execute(
        "SELECT COUNT(*) FROM attendance WHERE date = ? AND status = 'Distracted'",
        (today,)
    ).fetchone()[0]

    conn.close()
    return {
        "total_students": total,
        "present":        present,
        "absent":         total - present,
        "attentive":      attentive,
        "distracted":     distracted,
        "date":           today,
    }
