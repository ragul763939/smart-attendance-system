"""
behavior_analysis/attention_classifier.py
───────────────────────────────────────────
Combines eye status + head pose into a single attention score and label.

Scoring logic (per frame):
  +50 pts  : eyes are open
  +50 pts  : head is looking forward

  Score 80–100 → "Attentive"
  Score  0–79  → "Distracted"

Per-student rolling average is maintained using a simple deque buffer
to smooth out brief blinks or head turns.
"""

from collections import deque

# ── Size of the rolling window (number of recent frames to average) ───────
WINDOW_SIZE = 30   # ~1 second at 30 fps


class AttentionTracker:
    """
    Tracks attention score for a single student over time.

    Usage:
        tracker = AttentionTracker()
        tracker.update(eyes_open=True, looking_forward=True)
        score, label = tracker.get_status()
    """

    def __init__(self):
        # Deque automatically discards oldest entries beyond maxlen
        self._scores = deque(maxlen=WINDOW_SIZE)

    def update(self, eyes_open: bool, looking_forward: bool) -> None:
        """
        Add a new per-frame attention score.
        Called once per video frame per detected student.
        """
        score = 0
        if eyes_open:
            score += 50
        if looking_forward:
            score += 50
        self._scores.append(score)

    def get_status(self) -> tuple:
        """
        Compute the rolling average score and determine label.

        Returns:
            (attention_score: int 0-100,  label: str "Attentive" | "Distracted")
        """
        if not self._scores:
            return 100, "Attentive"   # Default: assume attentive if no data

        avg_score = int(sum(self._scores) / len(self._scores))
        label     = "Attentive" if avg_score >= 70 else "Distracted"
        return avg_score, label

    def reset(self):
        """Clear the score history (e.g., when a new session starts)."""
        self._scores.clear()


# ── Module-level registry: one tracker per student_id ────────────────────
_trackers: dict[str, AttentionTracker] = {}


def get_tracker(student_id: str) -> AttentionTracker:
    """Return (or create) an AttentionTracker for the given student."""
    if student_id not in _trackers:
        _trackers[student_id] = AttentionTracker()
    return _trackers[student_id]


def reset_all_trackers():
    """Clear all trackers (call at the start of each session)."""
    _trackers.clear()
