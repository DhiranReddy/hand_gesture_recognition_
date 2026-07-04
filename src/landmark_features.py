"""Normalize hand landmarks into ML feature vectors."""

from __future__ import annotations

import numpy as np

from src.constants import MIDDLE_MCP, WRIST


def landmarks_to_features(landmarks: list[tuple], handedness: str) -> np.ndarray:
    """
    Convert 21 hand landmarks to a fixed-size feature vector.
    Wrist-centered, scale-normalized, handedness-aware.
    """
    pts = np.array([[lm[0], lm[1], lm[2]] for lm in landmarks], dtype=np.float32)
    wrist = pts[WRIST]
    palm = np.linalg.norm(pts[MIDDLE_MCP] - wrist)
    if palm < 1e-6:
        palm = 1.0

    centered = (pts - wrist) / palm

    # Mirror x for consistent left/right features.
    if handedness == "Left":
        centered[:, 0] *= -1

    # Pairwise fingertip distances add shape detail.
    tips = centered[[4, 8, 12, 16, 20]]
    tip_dists = []
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            tip_dists.append(np.linalg.norm(tips[i] - tips[j]))

    return np.concatenate([centered.flatten(), np.array(tip_dists, dtype=np.float32)])
