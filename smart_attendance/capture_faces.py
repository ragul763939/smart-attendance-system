"""
capture_faces.py
─────────────────
Script to register new students into the system.

Run this BEFORE starting the attendance runner.
Each student needs to be registered once — their face
encoding is saved to the database permanently.

Usage:
    python capture_faces.py
"""

from database.db_manager import init_db
from face_recognition_module.encoder import capture_and_encode


def register_student():
    """
    Interactive CLI to register a student via webcam face capture.
    """
    print("=" * 50)
    print("  SMART ATTENDANCE SYSTEM – Student Registration")
    print("=" * 50)

    # ── Get student details from user ────────────────────────────────────
    name       = input("\nEnter student name   : ").strip()
    student_id = input("Enter student ID     : ").strip().upper()

    if not name or not student_id:
        print("[ERROR] Name and ID cannot be empty.")
        return

    print(f"\nRegistering: {name} ({student_id})")
    print("The webcam will open. Position your face inside the rectangle.")

    # ── Capture and encode face ──────────────────────────────────────────
    success = capture_and_encode(student_id, name)

    if success:
        print(f"\n✅ Student '{name}' registered successfully!")
    else:
        print(f"\n❌ Registration failed for '{name}'.")


def register_multiple():
    """
    Register multiple students in one session.
    Type 'done' when finished.
    """
    init_db()   # Ensure tables exist
    print("\n[INFO] Type 'done' as student name to finish.\n")

    while True:
        name = input("Student name (or 'done'): ").strip()
        if name.lower() == "done":
            print("\n[INFO] Registration session ended.")
            break

        student_id = input(f"Student ID for {name}: ").strip().upper()
        if not student_id:
            print("[WARN] ID cannot be empty. Try again.")
            continue

        capture_and_encode(student_id, name)
        print(f"\n{'─' * 40}\n")


if __name__ == "__main__":
    import sys

    init_db()   # Initialize database on first run

    # If "--multi" flag passed, register multiple students
    if "--multi" in sys.argv:
        register_multiple()
    else:
        register_student()
