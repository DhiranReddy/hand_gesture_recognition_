"""Hand landmark detection using MediaPipe and OpenCV."""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


class HandTracker:
    """Wraps MediaPipe Hands for real-time landmark extraction."""

    def __init__(self, max_hands: int = 1, detection_confidence: float = 0.7):
        self._use_solutions = hasattr(mp, "solutions")
        if not self._use_solutions:
            raise RuntimeError("MediaPipe Hands solutions are required for landmark extraction.")

        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=0.6,
        )

    def process(
        self,
        frame_bgr: np.ndarray,
        *,
        draw_skeleton: bool = True,
    ) -> list[dict]:
        """Return normalized landmark lists for each detected hand."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        hands: list[dict] = []

        if not results.multi_hand_landmarks:
            return hands

        h, w = frame_bgr.shape[:2]
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness or [],
        ):
            label = handedness.classification[0].label if handedness else "Unknown"
            landmarks = [
                (lm.x * w, lm.y * h, lm.z)
                for lm in hand_landmarks.landmark
            ]
            hands.append({"label": label, "landmarks": landmarks})

            if draw_skeleton:
                self._mp_draw.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    self._mp_draw.DrawingSpec(color=(0, 180, 255), thickness=2),
                )

        return hands

    def close(self) -> None:
        self._hands.close()
