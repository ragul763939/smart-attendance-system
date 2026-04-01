"""
behavior_analysis/eye_detector.py
───────────────────────────────────
Detects whether a student's eyes are open or closed using
dlib's 68-point facial landmark predictor.

How Eye Aspect Ratio (EAR) works:
  ┌─────────────────────────────────────────────────────┐
  │  Eye landmarks (6 points per eye):                  │
  │  P1 ──────────────────── P4                         │
  │        P2          P5                               │
  │        P3          P6                               │
  │                                                     │
  │  EAR = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)        │
  │                                                     │
  │  Open eye  → EAR ≈ 0.25–0.35                        │
  │  Closed eye → EAR < 0.20                            │
  └─────────────────────────────────────────────────────┘

Reference: Soukupová & Čech (2016) "Real-Time Eye Blink Detection"
"""

import dlib
import cv2
import numpy as np
from scipy.spatial import distance as dist
import os

# ── EAR threshold below which we consider the eye "closed" ──────────────
EAR_THRESHOLD       = 0.20

# ── Number of consecutive closed frames before flagging "eyes closed" ───
CONSECUTIVE_FRAMES  = 3

# ── Indices into the 68-point landmark array for each eye ────────────────
# dlib indexes landmarks 0-67; eyes are at:
LEFT_EYE_INDICES  = list(range(36, 42))   # landmarks 36-41
RIGHT_EYE_INDICES = list(range(42, 48))   # landmarks 42-47

# ── Path to dlib's pre-trained shape predictor model ─────────────────────
PREDICTOR_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "shape_predictor_68_face_landmarks.dat"
)


def load_detector_and_predictor():
    """
    Load dlib's face detector and 68-point shape predictor.
    Call once at startup and reuse the returned objects.

    Returns:
        detector  : dlib frontal face detector
        predictor : dlib shape predictor (68 landmarks)
    """
    if not os.path.exists(PREDICTOR_PATH):
        raise FileNotFoundError(
            f"[EYE] Landmark model not found at: {PREDICTOR_PATH}\n"
            "Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        )

    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    print("[EYE] dlib detector and predictor loaded.")
    return detector, predictor


def _eye_aspect_ratio(eye_points: np.ndarray) -> float:
    """
    Calculate the Eye Aspect Ratio (EAR) for a set of 6 eye landmark points.

    A lower EAR means the eye is more closed.
    """
    # Vertical distances
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])

    # Horizontal distance
    C = dist.euclidean(eye_points[0], eye_points[3])

    ear = (A + B) / (2.0 * C)
    return ear


def get_eye_status(frame: np.ndarray,
                   face_location: tuple,
                   detector,
                   predictor) -> dict:
    """
    Analyze eye status for a face in the given frame.

    Args:
        frame         : BGR image (numpy array)
        face_location : (top, right, bottom, left) from face_recognition
        detector      : dlib face detector
        predictor     : dlib shape predictor

    Returns:
        {
            "left_ear"    : float,
            "right_ear"   : float,
            "avg_ear"     : float,
            "eyes_open"   : bool,
            "landmarks"   : numpy array of 68 (x, y) points or None
        }
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert face_recognition location (top,right,bottom,left) → dlib rect
    top, right, bottom, left = face_location
    rect = dlib.rectangle(left, top, right, bottom)

    # Get 68 facial landmarks
    shape = predictor(gray, rect)

    # Convert dlib shape to numpy array of (x, y) coordinates
    landmarks = np.array([[shape.part(i).x, shape.part(i).y]
                           for i in range(68)])

    # Extract eye point arrays
    left_eye  = landmarks[LEFT_EYE_INDICES]
    right_eye = landmarks[RIGHT_EYE_INDICES]

    # Compute EAR for each eye
    left_ear  = _eye_aspect_ratio(left_eye)
    right_ear = _eye_aspect_ratio(right_eye)
    avg_ear   = (left_ear + right_ear) / 2.0

    eyes_open = avg_ear >= EAR_THRESHOLD

    return {
        "left_ear":  round(left_ear,  3),
        "right_ear": round(right_ear, 3),
        "avg_ear":   round(avg_ear,   3),
        "eyes_open": eyes_open,
        "landmarks": landmarks,
    }


def draw_eye_landmarks(frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Draw eye landmarks on the frame for debugging / visualization.
    Green circles on each eye landmark point.
    """
    if landmarks is None:
        return frame

    for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
        x, y = landmarks[idx]
        cv2.circle(frame, (x, y), 2, (0, 255, 100), -1)

    return frame
