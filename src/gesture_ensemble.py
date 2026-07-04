"""Gesture recognition: rules + custom ML."""

from __future__ import annotations

import numpy as np

from src.gesture_classifier import (
    ALL_LABELS,
    MODE_ALL,
    MODE_SPELL,
    MODE_WORDS,
    classify_gesture,
    gesture_to_text,
    is_single_char,
)
from src.ml_gesture_model import MLGestureClassifier


class EnsembleGestureRecognizer:
    """
    Combines rule-based and custom ML recognizers and picks the best match by confidence.
    Also smooths predictions over a short landmark window.
    """

    SMOOTH_WINDOW = 5

    def __init__(self) -> None:
        self._ml = MLGestureClassifier()
        self._history: list[tuple[str, float, str]] = []

    def close(self) -> None:
        return None

    def recognize(
        self,
        landmarks: list[tuple] | None,
        handedness: str,
        frame_bgr: np.ndarray | None,
        mode: str = MODE_ALL,
    ) -> tuple[str | None, str, float]:
        """
        Returns (gesture_id, category, confidence).
        category: phrase | letter | number | none
        """
        candidates: list[tuple[str, float, str]] = []

        if landmarks:
            rule_id, rule_cat = classify_gesture(landmarks, handedness, mode=mode)
            if rule_id:
                candidates.append((rule_id, 0.62, rule_cat))

            ml_id, ml_conf = self._ml.predict(landmarks, handedness)
            if ml_id:
                ml_cat = _category_for(ml_id)
                candidates.append((ml_id, ml_conf, ml_cat))

        if not candidates:
            self._history.clear()
            return None, "none", 0.0

        # Mode-based boost
        for i, (gid, conf, cat) in enumerate(candidates):
            if mode == MODE_SPELL and cat in ("letter", "number"):
                candidates[i] = (gid, conf + 0.08, cat)
            elif mode == MODE_WORDS and cat == "phrase":
                candidates[i] = (gid, conf + 0.08, cat)

        best = max(candidates, key=lambda c: c[1])
        self._history.append(best)
        if len(self._history) > self.SMOOTH_WINDOW:
            self._history.pop(0)

        # Temporal voting for stability
        if len(self._history) >= 3:
            ids = [h[0] for h in self._history]
            most_common = max(set(ids), key=ids.count)
            avg_conf = float(np.mean([h[1] for h in self._history if h[0] == most_common]))
            cat = next(h[2] for h in reversed(self._history) if h[0] == most_common)
            return most_common, cat, avg_conf

        return best[0], best[2], best[1]


def _category_for(gesture_id: str) -> str:
    if len(gesture_id) == 1 and gesture_id.isalpha():
        return "letter"
    if len(gesture_id) == 1 and gesture_id.isdigit():
        return "number"
    return "phrase"


def result_to_text(gesture_id: str | None) -> str | None:
    return gesture_to_text(gesture_id)


def is_letter_or_number(gesture_id: str | None, category: str) -> bool:
    if category in ("letter", "number"):
        return True
    text = gesture_to_text(gesture_id)
    return is_single_char(text)
