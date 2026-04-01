"""
app.py
───────
Flask web dashboard for the Smart Attendance System.

Routes:
  GET  /                   → Main dashboard (today's summary + attendance)
  GET  /students           → List of all registered students
  GET  /report             → Full attendance history with filters
  GET  /api/summary        → JSON: today's summary stats
  GET  /api/attendance     → JSON: today's attendance records
  POST /api/export/today   → Export today's CSV, returns file
  POST /api/export/all     → Export all-time CSV, returns file

Run:
    python app.py
    Then open: http://localhost:5000
"""

from flask import Flask, render_template, jsonify, send_file, request
from database.db_manager import (
    init_db, get_summary_today, get_attendance_today,
    get_attendance_all, get_all_students
)
from utils.csv_exporter import export_today, export_all
import os

app = Flask(__name__)

# ── Initialize DB when app starts ─────────────────────────────────────────
init_db()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """
    Main dashboard page.
    Passes today's summary and attendance records to the template.
    """
    summary    = get_summary_today()
    attendance = get_attendance_today()

    # Convert sqlite3.Row objects to plain dicts for Jinja2 templating
    attendance_list = [dict(row) for row in attendance]

    return render_template("index.html",
                           summary=summary,
                           attendance=attendance_list)


@app.route("/students")
def students():
    """Students registration page — shows all registered students."""
    all_students = [dict(row) for row in get_all_students()]
    # Don't pass encoding blob to template (it's binary data)
    for s in all_students:
        s.pop("encoding", None)
    return render_template("students.html", students=all_students)


@app.route("/report")
def report():
    """Full attendance report page with all historical records."""
    all_records = [dict(row) for row in get_attendance_all()]
    return render_template("report.html", records=all_records)


# ─────────────────────────────────────────────────────────────────────────────
# JSON API ROUTES (used by dashboard JS for live updates)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/summary")
def api_summary():
    """Return today's attendance summary as JSON."""
    return jsonify(get_summary_today())


@app.route("/api/attendance")
def api_attendance():
    """Return today's attendance records as JSON."""
    records = [dict(row) for row in get_attendance_today()]
    return jsonify(records)


@app.route("/api/students")
def api_students():
    """Return all students as JSON (without encoding blob)."""
    students_list = []
    for row in get_all_students():
        s = dict(row)
        s.pop("encoding", None)
        students_list.append(s)
    return jsonify(students_list)


# ─────────────────────────────────────────────────────────────────────────────
# CSV EXPORT ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/export/today")
def api_export_today():
    """Generate and serve today's attendance CSV."""
    filepath = export_today()
    return send_file(filepath, as_attachment=True,
                     download_name=os.path.basename(filepath))


@app.route("/api/export/all")
def api_export_all():
    """Generate and serve all-time attendance CSV."""
    filepath = export_all()
    return send_file(filepath, as_attachment=True,
                     download_name=os.path.basename(filepath))


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Smart Attendance Dashboard")
    print("  Open: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
