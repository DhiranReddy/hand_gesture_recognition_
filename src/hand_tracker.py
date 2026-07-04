"""Hand landmark detection using MediaPipe and OpenCV."""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from src.constants import (
    INDEX_DIP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_DIP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_DIP,
    PINKY_PIP,
    PINKY_TIP,
    RING_DIP,
    RING_PIP,
    RING_TIP,
    THUMB_IP,
    THUMB_TIP,
    WRIST,
)
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

    def _draw_skeleton(self, frame_bgr: np.ndarray, landmarks: list[tuple]) -> None:
        for start, end in HAND_CONNECTIONS:
            p1 = tuple(np.array(landmarks[start][:2], dtype=np.int32))
            p2 = tuple(np.array(landmarks[end][:2], dtype=np.int32))
            cv2.line(frame_bgr, p1, p2, (0, 180, 255), 2)

        for x, y, _ in landmarks:
            cv2.circle(frame_bgr, (int(x), int(y)), 3, (0, 255, 0), -1)

    def close(self) -> None:
        self._hands.close()


def finger_extended(landmarks: list[tuple], tip: int, pip: int, dip: int) -> bool:
    """Check if a finger is extended using tip-to-pip distance vs pip-to-dip."""
    tip_pt = np.array(landmarks[tip][:2])
    pip_pt = np.array(landmarks[pip][:2])
    dip_pt = np.array(landmarks[dip][:2])
    return np.linalg.norm(tip_pt - pip_pt) > np.linalg.norm(pip_pt - dip_pt) * 0.85


def thumb_extended(landmarks: list[tuple], handedness: str) -> bool:
    """Thumb extension depends on hand orientation."""
    tip = np.array(landmarks[THUMB_TIP][:2])
    ip = np.array(landmarks[THUMB_IP][:2])
    wrist = np.array(landmarks[WRIST][:2])

    tip_dist = np.linalg.norm(tip - wrist)
    ip_dist = np.linalg.norm(ip - wrist)

    if handedness == "Right":
        return tip_dist > ip_dist * 1.1 and tip[0] > ip[0]
    return tip_dist > ip_dist * 1.1 and tip[0] < ip[0]


def get_finger_states(landmarks: list[tuple], handedness: str) -> list[bool]:
    """Return [thumb, index, middle, ring, pinky] extension states."""
    return [
        thumb_extended(landmarks, handedness),
        finger_extended(landmarks, INDEX_TIP, INDEX_PIP, INDEX_DIP),
        finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_DIP),
        finger_extended(landmarks, RING_TIP, RING_PIP, RING_DIP),
        finger_extended(landmarks, PINKY_TIP, PINKY_PIP, PINKY_DIP),
    ]
