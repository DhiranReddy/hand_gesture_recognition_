"""Hand geometry helpers for precise gesture classification."""

from __future__ import annotations

import numpy as np

from src.constants import (
    INDEX_DIP,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_DIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_DIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_DIP,
    RING_MCP,
    RING_PIP,
    RING_TIP,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
)

FingerState = str  # "extended", "bent", "curled"


def _pt(landmarks: list[tuple], idx: int) -> np.ndarray:
    return np.array(landmarks[idx][:2], dtype=float)


def palm_size(landmarks: list[tuple]) -> float:
    return float(np.linalg.norm(_pt(landmarks, WRIST) - _pt(landmarks, MIDDLE_MCP)))


def _distance(a: tuple | np.ndarray, b: tuple | np.ndarray) -> float:
    return float(np.linalg.norm(np.array(a[:2]) - np.array(b[:2])))


def finger_curl_state(
    landmarks: list[tuple],
    tip: int,
    pip: int,
    dip: int,
    mcp: int,
) -> FingerState:
    """Classify a finger as extended, bent (hooked), or curled."""
    tip_pt = _pt(landmarks, tip)
    pip_pt = _pt(landmarks, pip)
    dip_pt = _pt(landmarks, dip)
    mcp_pt = _pt(landmarks, mcp)
    wrist_pt = _pt(landmarks, WRIST)

    tip_to_wrist = np.linalg.norm(tip_pt - wrist_pt)
    mcp_to_wrist = np.linalg.norm(mcp_pt - wrist_pt)
    tip_to_pip = np.linalg.norm(tip_pt - pip_pt)
    pip_to_dip = np.linalg.norm(pip_pt - dip_pt)

    if tip_to_pip < pip_to_dip * 0.65:
        return "curled"
    if tip_to_wrist > mcp_to_wrist * 1.35 and tip_to_pip > pip_to_dip * 0.9:
        return "extended"
    if tip_to_pip < pip_to_dip * 0.95:
        return "bent"
    return "curled"


def thumb_state(landmarks: list[tuple], handedness: str) -> FingerState:
    tip = _pt(landmarks, THUMB_TIP)
    ip = _pt(landmarks, THUMB_IP)
    mcp = _pt(landmarks, THUMB_MCP)
    wrist = _pt(landmarks, WRIST)
    index_mcp = _pt(landmarks, INDEX_MCP)

    tip_dist = np.linalg.norm(tip - wrist)
    ip_dist = np.linalg.norm(ip - wrist)

    if tip_dist < ip_dist * 0.95:
        return "curled"

    # Thumb extended outward
    if handedness == "Right":
        outward = tip[0] > mcp[0] + 5
    else:
        outward = tip[0] < mcp[0] - 5

    if outward and tip_dist > ip_dist * 1.05:
        return "extended"

    # Thumb alongside palm (ASL A, T, etc.)
    if _distance(tip, index_mcp) < palm_size(landmarks) * 0.45:
        return "bent"

    return "curled"


def get_detailed_finger_states(landmarks: list[tuple], handedness: str) -> list[FingerState]:
    return [
        thumb_state(landmarks, handedness),
        finger_curl_state(landmarks, INDEX_TIP, INDEX_PIP, INDEX_DIP, INDEX_MCP),
        finger_curl_state(landmarks, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_MCP),
        finger_curl_state(landmarks, RING_TIP, RING_PIP, RING_DIP, RING_MCP),
        finger_curl_state(landmarks, PINKY_TIP, PINKY_PIP, PINKY_DIP, PINKY_MCP),
    ]


def count_extended(states: list[FingerState]) -> int:
    return sum(1 for s in states[1:] if s == "extended")


def fingertips_pinch(landmarks: list[tuple], a: int, b: int, ratio: float = 0.28) -> bool:
    palm = palm_size(landmarks)
    if palm < 1:
        return False
    return _distance(landmarks[a], landmarks[b]) < palm * ratio


def all_fingertips_touch(landmarks: list[tuple], ratio: float = 0.38) -> bool:
    tips = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
    palm = palm_size(landmarks)
    if palm < 1:
        return False
    center = np.mean([_pt(landmarks, t) for t in tips], axis=0)
    return all(_distance(landmarks[t], center) < palm * ratio for t in tips)


def index_middle_together(landmarks: list[tuple]) -> bool:
    palm = palm_size(landmarks)
    return _distance(landmarks[INDEX_TIP], landmarks[MIDDLE_TIP]) < palm * 0.22


def index_middle_spread(landmarks: list[tuple]) -> bool:
    palm = palm_size(landmarks)
    return _distance(landmarks[INDEX_TIP], landmarks[MIDDLE_TIP]) > palm * 0.35


def index_middle_crossed(landmarks: list[tuple]) -> bool:
    index_tip = _pt(landmarks, INDEX_TIP)
    middle_tip = _pt(landmarks, MIDDLE_TIP)
    return abs(index_tip[1] - middle_tip[1]) < palm_size(landmarks) * 0.15 and (
        _distance(landmarks[INDEX_TIP], landmarks[MIDDLE_TIP]) < palm_size(landmarks) * 0.3
    )


def pointing_horizontal(landmarks: list[tuple], handedness: str) -> bool:
    """Index finger points sideways (ASL G)."""
    index_tip = _pt(landmarks, INDEX_TIP)
    index_mcp = _pt(landmarks, INDEX_MCP)
    wrist = _pt(landmarks, WRIST)
    vertical_span = abs(index_tip[1] - wrist[1])
    horizontal_span = abs(index_tip[0] - index_mcp[0])
    return horizontal_span > vertical_span * 1.1 and horizontal_span > palm_size(landmarks) * 0.5


def thumb_between_index_middle(landmarks: list[tuple]) -> bool:
    thumb = _pt(landmarks, THUMB_TIP)
    index_mcp = _pt(landmarks, INDEX_MCP)
    middle_mcp = _pt(landmarks, MIDDLE_MCP)
    mid_x = (index_mcp[0] + middle_mcp[0]) / 2
    mid_y = (index_mcp[1] + middle_mcp[1]) / 2
    return _distance(thumb, (mid_x, mid_y)) < palm_size(landmarks) * 0.35
