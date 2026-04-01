"""
face_recognition_module/encoder.py
────────────────────────────────────
Generates and stores 128-dimensional face encodings.

A face encoding is a list of 128 numbers that uniquely describe
a person's face — like a numerical fingerprint.

Steps:
  1. Capture a photo of a student's face via webcam
  2. Run it through face_recognition.face_encodings()
  3. Serialize (pickle) the numpy array → store as BLOB in DB
"""

import cv2
import face_recognition
import pickle
import numpy as np
from database.db_manager import add_student


def capture_and_encode(student_id: str, name: str) -> bool:
    """
    Opens the webcam, lets the user press 'S' to capture a face,
    generates the encoding, and saves it to the database.

    Controls:
      S  – Snap / capture current frame
      Q  – Quit without saving

    Returns True if encoding was saved successfully.
    """
    print(f"\n[ENCODER] Opening webcam for: {name} ({student_id})")
    print("  → Look at the camera and press [S] to capture, [Q] to quit.\n")

    cap = cv2.VideoCapture(0)   # 0 = default webcam

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return False

    encoding_saved = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from webcam.")
            break

        # ── Show a live preview with instructions overlay ───────────────
        display = frame.copy()
        cv2.putText(display, f"Student: {name}  ID: {student_id}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "Press [S] to Capture | [Q] to Quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # Draw a guide rectangle in the center to help student position face
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        cv2.rectangle(display,
                      (cx - 120, cy - 150), (cx + 120, cy + 150),
                      (255, 255, 0), 2)

        cv2.imshow("Register Student – Face Capture", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') or key == ord('S'):
            # ── Try to extract face encoding from the current frame ──────
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Locate all faces in the image
            face_locations = face_recognition.face_locations(rgb_frame)

            if len(face_locations) == 0:
                print("[WARN] No face detected in frame. Please reposition.")
                cv2.putText(display, "NO FACE DETECTED!", (cx - 100, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.imshow("Register Student – Face Capture", display)
                cv2.waitKey(1000)
                continue

            if len(face_locations) > 1:
                print("[WARN] Multiple faces detected. Please ensure only one person is in frame.")
                continue

            # Generate the 128-number encoding for the single detected face
            encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            encoding  = encodings[0]  # numpy array of shape (128,)

            # Serialize numpy array to bytes for database storage
            encoding_bytes = pickle.dumps(encoding)

            # Save to database
            success = add_student(student_id, name, encoding_bytes)
            if success:
                print(f"[ENCODER] ✅ Face encoding saved for {name}.")
                encoding_saved = True
                # Show success message on screen
                cv2.putText(display, "CAPTURED! Press Q to exit.",
                            (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
                cv2.imshow("Register Student – Face Capture", display)
                cv2.waitKey(2000)
            break

        elif key == ord('q') or key == ord('Q'):
            print("[ENCODER] Capture cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return encoding_saved


def load_all_encodings() -> tuple:
    """
    Load all student face encodings from the database.

    Returns:
        known_encodings : list of numpy arrays  (128-d each)
        known_ids       : list of student_id strings
        known_names     : list of name strings
    """
    from database.db_manager import get_all_students

    students        = get_all_students()
    known_encodings = []
    known_ids       = []
    known_names     = []

    for student in students:
        # Deserialize the BLOB back into a numpy array
        encoding = pickle.loads(student["encoding"])
        known_encodings.append(encoding)
        known_ids.append(student["student_id"])
        known_names.append(student["name"])

    print(f"[ENCODER] Loaded {len(known_encodings)} face encoding(s) from database.")
    return known_encodings, known_ids, known_names
