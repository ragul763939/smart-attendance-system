"""
attendance_runner.py
─────────────────────
Main real-time detection loop.

This script:
  1. Loads all known face encodings from the database
  2. Opens the webcam
  3. For each frame:
     a. Detects and recognizes all faces
     b. Marks attendance (first time seen today)
     c. Analyzes eye status (open/closed)
     d. Estimates head pose (looking forward?)
     e. Computes attention score per student
     f. Updates attention in the database every N seconds
     g. Draws all results on the frame
  4. Press Q to quit

Usage:
    python attendance_runner.py
"""

import cv2
import time
from datetime import datetime

# ── Local module imports ──────────────────────────────────────────────────
from database.db_manager         import init_db, mark_attendance, update_attention
from face_recognition_module.encoder    import load_all_encodings
from face_recognition_module.recognizer import recognize_faces, draw_face_boxes
from behavior_analysis.eye_detector     import load_detector_and_predictor, get_eye_status, draw_eye_landmarks
from behavior_analysis.head_pose        import estimate_head_pose, draw_head_pose_axes
from behavior_analysis.attention_classifier import get_tracker, reset_all_trackers
from utils.helpers import draw_status_overlay, draw_attention_badge, resize_frame

# ── How often (seconds) to update attention score in DB ──────────────────
ATTENTION_UPDATE_INTERVAL = 10   # every 10 seconds

# ── Webcam index (0 = default camera) ────────────────────────────────────
CAMERA_INDEX = 0


def run_attendance():
    """
    Main loop: open webcam, detect faces, mark attendance, analyze behavior.
    """
    print("\n" + "=" * 55)
    print("  SMART ATTENDANCE SYSTEM – Real-Time Detection")
    print("=" * 55)

    # ── 1. Initialize database ───────────────────────────────────────────
    init_db()

    # ── 2. Load face encodings ───────────────────────────────────────────
    known_encodings, known_ids, known_names = load_all_encodings()
    if not known_encodings:
        print("[WARN] No students registered yet.")
        print("       Run: python capture_faces.py  first.")
        return

    # ── 3. Load dlib detector + landmark predictor ───────────────────────
    try:
        detector, predictor = load_detector_and_predictor()
        behavior_enabled    = True
    except FileNotFoundError as e:
        print(f"[WARN] {e}")
        print("[WARN] Behavior analysis disabled (landmark model missing).")
        detector = predictor = None
        behavior_enabled    = False

    # ── 4. Reset attention trackers for this session ─────────────────────
    reset_all_trackers()

    # ── 5. Open webcam ───────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check camera index.")
        return

    print(f"\n[INFO] Webcam opened. Session started at {datetime.now().strftime('%H:%M:%S')}")
    print("[INFO] Press [Q] to quit.\n")

    # ── Timing variables ─────────────────────────────────────────────────
    session_start       = time.time()
    last_attention_save = time.time()
    fps_counter         = 0
    fps_timer           = time.time()
    current_fps         = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from webcam.")
            break

        # Flip horizontally for mirror-like display (natural feel)
        frame = cv2.flip(frame, 1)

        # Resize for consistent display
        frame = resize_frame(frame, width=960)

        # ── FPS calculation ───────────────────────────────────────────────
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            fps_timer   = time.time()

        # ── Session elapsed time string ───────────────────────────────────
        elapsed     = int(time.time() - session_start)
        session_str = f"{elapsed // 60:02d}:{elapsed % 60:02d}"

        # ── A. Face recognition ───────────────────────────────────────────
        recognized = recognize_faces(frame, known_encodings,
                                     known_ids, known_names)

        # ── B. Attendance marking + behavior analysis ─────────────────────
        for person in recognized:
            sid       = person["student_id"]
            name      = person["name"]
            location  = person["location"]  # (top, right, bottom, left)

            # Mark attendance (ignored if already marked today)
            if sid != "Unknown":
                mark_attendance(sid, name)

            # ── Behavior analysis (if landmark model is available) ────────
            eyes_open       = True
            looking_forward = True
            landmarks       = None

            if behavior_enabled and sid != "Unknown":
                try:
                    eye_data  = get_eye_status(frame, location,
                                               detector, predictor)
                    pose_data = estimate_head_pose(frame, eye_data["landmarks"])

                    eyes_open       = eye_data["eyes_open"]
                    looking_forward = pose_data["looking_forward"]
                    landmarks       = eye_data["landmarks"]

                    # Draw eye and pose overlays
                    frame = draw_eye_landmarks(frame, landmarks)
                    frame = draw_head_pose_axes(frame, landmarks, pose_data)

                except Exception:
                    pass   # Silently skip if face is partially visible

            # ── Update attention tracker ──────────────────────────────────
            if sid != "Unknown":
                tracker = get_tracker(sid)
                tracker.update(eyes_open, looking_forward)
                score, label = tracker.get_status()

                # Draw attention badge above face box
                top, right, bottom, left = location
                frame = draw_attention_badge(frame, left, top, score, label)

                # Persist attention score to DB periodically
                now = time.time()
                if now - last_attention_save >= ATTENTION_UPDATE_INTERVAL:
                    update_attention(sid, score, label)
                    last_attention_save = now

        # ── C. Draw face boxes with names ─────────────────────────────────
        frame = draw_face_boxes(frame, recognized)

        # ── D. HUD overlay ────────────────────────────────────────────────
        frame = draw_status_overlay(frame, current_fps,
                                    len(recognized), session_str)

        # ── E. Show frame ─────────────────────────────────────────────────
        cv2.imshow("Smart Attendance System  |  Press Q to quit", frame)

        # ── F. Exit on Q key ──────────────────────────────────────────────
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] Session ended by user.")
            break

    # ── Final attention save before exit ─────────────────────────────────
    print("[INFO] Saving final attention scores...")
    # (trackers already have latest data — one last save)

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam released. Goodbye!\n")


if __name__ == "__main__":
    run_attendance()
