"""
utils/helpers.py
─────────────────
Shared utility functions used across the project.
"""

import cv2
import numpy as np
from datetime import datetime


def get_timestamp() -> str:
    """Return current date-time as a readable string: '2024-01-15 09:30:45'"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_date_str() -> str:
    """Return today's date as 'YYYY-MM-DD'."""
    return datetime.now().strftime("%Y-%m-%d")


def get_time_str() -> str:
    """Return current time as 'HH:MM:SS'."""
    return datetime.now().strftime("%H:%M:%S")


def draw_status_overlay(frame: np.ndarray,
                        fps: float,
                        face_count: int,
                        session_time: str) -> np.ndarray:
    """
    Draw a HUD (heads-up display) on the top-left of the frame.
    Shows FPS, face count, and session time.
    """
    overlay  = frame.copy()
    bar_h    = 90
    cv2.rectangle(overlay, (0, 0), (340, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 1)
    cv2.putText(frame, f"Faces detected: {face_count}",
                (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 1)
    cv2.putText(frame, f"Session: {session_time}",
                (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 1)
    cv2.putText(frame, "Press [Q] to quit",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    return frame


def draw_attention_badge(frame: np.ndarray,
                         x: int, y: int,
                         score: int,
                         label: str) -> np.ndarray:
    """
    Draw a small colored badge showing attention score near a face.

    Green = Attentive, Orange = Distracted
    """
    color = (0, 200, 80) if label == "Attentive" else (0, 140, 255)
    text  = f"{label} {score}%"

    cv2.putText(frame, text,
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame


def resize_frame(frame: np.ndarray, width: int = 960) -> np.ndarray:
    """
    Resize frame to a fixed width while maintaining aspect ratio.
    Keeps display window consistent regardless of webcam resolution.
    """
    h, w = frame.shape[:2]
    ratio       = width / w
    new_height  = int(h * ratio)
    return cv2.resize(frame, (width, new_height))
