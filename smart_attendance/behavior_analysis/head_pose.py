"""
behavior_analysis/head_pose.py
───────────────────────────────
Estimates head pose (yaw, pitch, roll) using OpenCV's solvePnP.

How head pose estimation works:
  1. Use dlib to get 6 key facial landmarks (nose tip, chin, eyes, mouth corners)
  2. Map those to a generic 3D face model (known real-world coordinates)
  3. Use cv2.solvePnP to find the rotation vector that explains
     the 2D→3D projection
  4. Convert rotation vector to Euler angles (yaw, pitch, roll)

  Yaw   → left/right head turn   (looking away from screen)
  Pitch → up/down head tilt
  Roll  → head tilt sideways

A student looking straight at the screen has yaw ≈ 0°, pitch ≈ 0°.
"""

import cv2
import numpy as np


# ── Thresholds for "looking away" ─────────────────────────────────────────
YAW_THRESHOLD   = 25   # degrees left or right
PITCH_THRESHOLD = 20   # degrees up or down


# ── 3D model points of a generic human face (in mm, arbitrary origin) ────
# These correspond to: nose tip, chin, left eye corner, right eye corner,
# left mouth corner, right mouth corner
MODEL_POINTS_3D = np.array([
    (0.0,   0.0,    0.0),     # Nose tip
    (0.0,  -330.0, -65.0),    # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0,  170.0, -135.0),  # Right eye right corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0,  -150.0, -125.0), # Right mouth corner
], dtype=np.float64)

# dlib 68-point indices for the 6 landmarks above
LANDMARK_INDICES = [30, 8, 36, 45, 48, 54]


def estimate_head_pose(frame: np.ndarray,
                       landmarks: np.ndarray) -> dict:
    """
    Estimate head orientation from 68-point facial landmarks.

    Args:
        frame     : BGR image (used to get frame dimensions for camera matrix)
        landmarks : (68, 2) numpy array from dlib shape predictor

    Returns:
        {
            "yaw"          : float (degrees, + = right turn, - = left turn),
            "pitch"        : float (degrees, + = looking up, - = looking down),
            "roll"         : float (degrees),
            "looking_forward": bool,
            "direction"    : str  ("Forward" / "Left" / "Right" / "Up" / "Down"),
        }
    """
    if landmarks is None:
        return _unknown_pose()

    h, w = frame.shape[:2]

    # ── Camera intrinsic matrix (approximated from frame size) ───────────
    focal_length  = w                    # focal length ≈ image width
    center        = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0,            center[0]],
        [0,            focal_length, center[1]],
        [0,            0,            1        ]
    ], dtype=np.float64)

    # No lens distortion assumed
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # ── Extract the 6 2D landmark points ─────────────────────────────────
    image_points_2d = np.array(
        [landmarks[i] for i in LANDMARK_INDICES],
        dtype=np.float64
    )

    # ── Solve for rotation & translation ─────────────────────────────────
    success, rotation_vec, translation_vec = cv2.solvePnP(
        MODEL_POINTS_3D,
        image_points_2d,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return _unknown_pose()

    # ── Convert rotation vector → rotation matrix → Euler angles ──────────
    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)

    # Decompose into Euler angles using projection matrix decomposition
    pose_matrix = cv2.hconcat([rotation_matrix, translation_vec])
    _, _, _, _, _, _, euler_angles = cv2.decomposeHomographyMat(
        pose_matrix[:, :3], camera_matrix
    )

    # Simpler manual extraction from rotation matrix (more reliable)
    yaw   = float(np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])))
    pitch = float(np.degrees(np.arctan2(-rotation_matrix[2, 0],
                                         np.sqrt(rotation_matrix[2, 1]**2 +
                                                 rotation_matrix[2, 2]**2))))
    roll  = float(np.degrees(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])))

    # ── Classify direction ────────────────────────────────────────────────
    direction        = "Forward"
    looking_forward  = True

    if abs(yaw) > YAW_THRESHOLD:
        direction       = "Right" if yaw > 0 else "Left"
        looking_forward = False
    elif pitch > PITCH_THRESHOLD:
        direction       = "Up"
        looking_forward = False
    elif pitch < -PITCH_THRESHOLD:
        direction       = "Down"
        looking_forward = False

    return {
        "yaw":             round(yaw,   1),
        "pitch":           round(pitch, 1),
        "roll":            round(roll,  1),
        "looking_forward": looking_forward,
        "direction":       direction,
    }


def draw_head_pose_axes(frame: np.ndarray,
                        landmarks: np.ndarray,
                        pose: dict) -> np.ndarray:
    """
    Draw a text overlay showing head direction on the frame.
    """
    if landmarks is None or not pose:
        return frame

    direction = pose.get("direction", "?")
    yaw       = pose.get("yaw",   0)
    pitch     = pose.get("pitch", 0)

    color = (0, 200, 0) if pose.get("looking_forward") else (0, 100, 255)

    nose_x, nose_y = int(landmarks[30][0]), int(landmarks[30][1])

    cv2.putText(frame,
                f"Head: {direction} | Y:{yaw:.0f} P:{pitch:.0f}",
                (nose_x - 80, nose_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return frame


def _unknown_pose() -> dict:
    """Return a neutral pose dict when detection fails."""
    return {
        "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
        "looking_forward": True, "direction": "Unknown"
    }
