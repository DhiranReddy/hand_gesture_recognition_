"""ASL number signs 0-9."""

from __future__ import annotations

from src.constants import INDEX_TIP, MIDDLE_TIP, PINKY_TIP, THUMB_TIP
from src.finger_geometry import (
    all_fingertips_touch,
    count_extended,
    get_detailed_finger_states,
    index_middle_spread,
    index_middle_together,
    fingertips_pinch,
)


def classify_number(landmarks: list[tuple], handedness: str) -> str | None:
    """Return digit 0-9 ID or None."""
    s = get_detailed_finger_states(landmarks, handedness)
    thumb, index, middle, ring, pinky = s
    ext = count_extended(s)

    # 0 — O shape
    if all_fingertips_touch(landmarks):
        return "0"

    # 1 — index only
    if index == "extended" and ext == 1 and thumb != "extended":
        return "1"

    # 2 — V sign (index + middle spread)
    if index == "extended" and middle == "extended" and ext == 2 and index_middle_spread(landmarks):
        return "2"

    # 3 — index + middle + ring OR thumb + index + middle
    if index == "extended" and middle == "extended" and ring == "extended" and pinky != "extended" and thumb != "extended":
        return "3"

    # 4 — four fingers, thumb tucked
    if index == "extended" and middle == "extended" and ring == "extended" and pinky == "extended" and thumb != "extended":
        return "4"

    # 5 — all five extended
    if all(f == "extended" for f in s):
        return "5"

    # 6 — thumb and pinky touch, others curled (ASL 6)
    if fingertips_pinch(landmarks, THUMB_TIP, PINKY_TIP, ratio=0.4) and middle == "curled" and ring == "curled":
        return "6"

    # 7 — thumb touches middle tip, index extended
    if index == "extended" and fingertips_pinch(landmarks, THUMB_TIP, MIDDLE_TIP, ratio=0.35):
        return "7"

    # 8 — thumb touches index tip, middle extended
    if middle == "extended" and fingertips_pinch(landmarks, THUMB_TIP, INDEX_TIP, ratio=0.35):
        return "8"

    # 9 — F shape: thumb+index pinch, others up
    if fingertips_pinch(landmarks, THUMB_TIP, INDEX_TIP) and middle == "extended" and ring == "extended" and pinky == "extended":
        return "9"

    # 3 alternate: thumb + index + middle
    if thumb == "extended" and index == "extended" and middle == "extended" and ring == "curled" and pinky == "curled":
        return "3"

    return None
