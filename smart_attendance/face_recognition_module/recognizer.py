"""
face_recognition_module/recognizer.py
──────────────────────────────────────
Compares detected faces in a live video frame against known encodings.

How face recognition works (simplified):
  1. Detect face locations in the current frame
  2. Compute 128-d encodings for each detected face
  3. Compare each unknown encoding to all known encodings
     using Euclidean distance (face_recognition.compare_faces)
  4. The closest match below a tolerance threshold = recognized student
"""

import face_recognition
import numpy as np
import cv2


# ── Tolerance: lower = stricter matching (0.6 is a good default) ──────────
RECOGNITION_TOLERANCE = 0.50


def recognize_faces(frame: np.ndarray,
                    known_encodings: list,
                    known_ids: list,
                    known_names: list,
                    scale: float = 0.5) -> list:
    """
    Detect and identify all faces in a single video frame.

    Args:
        frame           : BGR image from OpenCV (numpy array)
        known_encodings : list of 128-d numpy arrays (from database)
        known_ids       : list of student ID strings
        known_names     : list of student name strings
        scale           : resize factor for speed (0.5 = half resolution)

    Returns:
        A list of dicts, one per detected face:
        {
            "student_id" : str or "Unknown",
            "name"       : str or "Unknown",
            "location"   : (top, right, bottom, left) in original frame coords,
            "confidence" : float (0.0 – 1.0, higher = more confident),
        }
    """
    results = []

    # ── Step 1: Downscale for faster processing ───────────────────────────
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    rgb_small   = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # ── Step 2: Locate faces in the small frame ───────────────────────────
    # "hog" model is faster; use "cnn" for higher accuracy (needs GPU)
    face_locations = face_recognition.face_locations(rgb_small, model="hog")

    if not face_locations:
        return results   # No faces detected

    # ── Step 3: Compute encodings for all detected faces ─────────────────
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    # ── Step 4: Match each detected face against known encodings ──────────
    for face_encoding, location in zip(face_encodings, face_locations):

        name       = "Unknown"
        student_id = "Unknown"
        confidence = 0.0

        if known_encodings:
            # Compare against all known faces; returns list of True/False
            matches = face_recognition.compare_faces(
                known_encodings, face_encoding, tolerance=RECOGNITION_TOLERANCE
            )

            # Compute Euclidean distances (lower = better match)
            distances = face_recognition.face_distance(known_encodings, face_encoding)

            best_idx = int(np.argmin(distances))   # Index of closest match

            if matches[best_idx]:
                # Convert distance to a 0-1 confidence score
                confidence = round(1 - distances[best_idx], 2)
                name       = known_names[best_idx]
                student_id = known_ids[best_idx]

        # ── Scale location back to original frame coordinates ─────────────
        top, right, bottom, left = location
        top    = int(top    / scale)
        right  = int(right  / scale)
        bottom = int(bottom / scale)
        left   = int(left   / scale)

        results.append({
            "student_id": student_id,
            "name":       name,
            "location":   (top, right, bottom, left),
            "confidence": confidence,
        })

    return results


def draw_face_boxes(frame: np.ndarray, recognized: list) -> np.ndarray:
    """
    Draw bounding boxes and name labels on the frame for each recognized face.

    Green box  = recognized student
    Red box    = unknown person
    """
    for person in recognized:
        top, right, bottom, left = person["location"]
        name       = person["name"]
        student_id = person["student_id"]
        confidence = person["confidence"]

        # Choose color: green for known, red for unknown
        color = (0, 200, 0) if name != "Unknown" else (0, 0, 220)

        # Draw rectangle around face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Draw filled label background
        cv2.rectangle(frame,
                      (left, bottom - 35), (right, bottom),
                      color, cv2.FILLED)

        # Print name + confidence on label
        label = f"{name} ({int(confidence * 100)}%)" if name != "Unknown" else "Unknown"
        cv2.putText(frame, label,
                    (left + 4, bottom - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1)

    return frame
