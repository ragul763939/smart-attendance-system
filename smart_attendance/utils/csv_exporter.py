"""
utils/csv_exporter.py
──────────────────────
Export attendance data from SQLite to CSV files.

Generated files are saved in the reports/ directory.
"""

import csv
import os
from datetime import date, datetime
from database.db_manager import get_attendance_today, get_attendance_all

# ── Output directory for CSV reports ─────────────────────────────────────
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

# Column headers for the CSV file
CSV_HEADERS = ["Student ID", "Name", "Date", "Time",
               "Attention Score", "Status"]


def _ensure_reports_dir():
    """Create the reports/ directory if it doesn't exist."""
    os.makedirs(REPORTS_DIR, exist_ok=True)


def export_today() -> str:
    """
    Export today's attendance to a CSV file.

    Returns:
        Full path to the generated CSV file.
    """
    _ensure_reports_dir()
    today    = date.today().isoformat()
    filename = f"attendance_{today}.csv"
    filepath = os.path.join(REPORTS_DIR, filename)

    records  = get_attendance_today()
    _write_csv(filepath, records)
    print(f"[CSV] Today's report saved: {filepath}")
    return filepath


def export_all() -> str:
    """
    Export all-time attendance records to a single CSV file.

    Returns:
        Full path to the generated CSV file.
    """
    _ensure_reports_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"attendance_all_{timestamp}.csv"
    filepath  = os.path.join(REPORTS_DIR, filename)

    records   = get_attendance_all()
    _write_csv(filepath, records)
    print(f"[CSV] Full report saved: {filepath}")
    return filepath


def export_date_range(start_date: str, end_date: str) -> str:
    """
    Export attendance records between two dates (inclusive).

    Args:
        start_date : "YYYY-MM-DD"
        end_date   : "YYYY-MM-DD"

    Returns:
        Full path to the generated CSV file.
    """
    from database.db_manager import get_connection

    _ensure_reports_dir()
    filename = f"attendance_{start_date}_to_{end_date}.csv"
    filepath = os.path.join(REPORTS_DIR, filename)

    conn    = get_connection()
    records = conn.execute("""
        SELECT * FROM attendance
        WHERE date BETWEEN ? AND ?
        ORDER BY date, time
    """, (start_date, end_date)).fetchall()
    conn.close()

    _write_csv(filepath, records)
    print(f"[CSV] Range report saved: {filepath}")
    return filepath


def _write_csv(filepath: str, records: list):
    """
    Write records to a CSV file.

    Args:
        filepath : Destination file path
        records  : List of sqlite3.Row objects
    """
    with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(CSV_HEADERS)   # Header row

        for row in records:
            writer.writerow([
                row["student_id"],
                row["name"],
                row["date"],
                row["time"],
                row["attention_score"],
                row["status"],
            ])
