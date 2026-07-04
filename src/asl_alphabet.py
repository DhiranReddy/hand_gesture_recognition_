"""ASL alphabet (A-Z) classification from hand landmarks."""

from __future__ import annotations

from src.constants import INDEX_TIP, MIDDLE_TIP, RING_TIP, THUMB_TIP
from src.finger_geometry import (
    all_fingertips_touch,
    count_extended,
    get_detailed_finger_states,
    index_middle_crossed,
    index_middle_spread,
    index_middle_together,
    palm_size,
    pointing_horizontal,
    thumb_between_index_middle,
    fingertips_pinch,
)


def classify_letter(landmarks: list[tuple], handedness: str) -> str | None:
    """Return A-Z letter ID or None."""
    s = get_detailed_finger_states(landmarks, handedness)
    thumb, index, middle, ring, pinky = s
    ext = count_extended(s)

    # O — all fingertips touch (also 0)
    if all_fingertips_touch(landmarks):
        return "O"

    # I — pinky only extended
    if pinky == "extended" and ext == 1 and thumb != "extended":
        return "I"

    # J — same static pose as I (motion forms J in ASL)
    # Handled as I; users spell with motion. Static fallback: I.

    # Y — thumb + pinky extended (shaka)
    if thumb == "extended" and pinky == "extended" and index != "extended" and middle != "extended" and ring != "extended":
        return "Y"

    # L — thumb + index extended at right angle
    if thumb == "extended" and index == "extended" and middle != "extended" and ring != "extended" and pinky != "extended":
        return "L"

    # V — index + middle spread (peace sign)
    if index == "extended" and middle == "extended" and ring != "extended" and pinky != "extended" and index_middle_spread(landmarks):
        return "V"

    # U — index + middle together pointing up
    if index == "extended" and middle == "extended" and ring != "extended" and pinky != "extended" and index_middle_together(landmarks):
        return "U"

    # R — index and middle crossed
    if index in ("extended", "bent") and middle in ("extended", "bent") and index_middle_crossed(landmarks):
        return "R"

    # K — index + middle up in V, thumb between at base
    if index == "extended" and middle == "extended" and thumb_between_index_middle(landmarks):
        return "K"

    # W — index, middle, ring extended
    if index == "extended" and middle == "extended" and ring == "extended" and pinky != "extended":
        return "W"

    # B — four fingers extended, thumb tucked
    if index == "extended" and middle == "extended" and ring == "extended" and pinky == "extended" and thumb != "extended":
        return "B"

    # F — thumb+index pinch, other three extended
    if fingertips_pinch(landmarks, THUMB_TIP, INDEX_TIP) and middle == "extended" and ring == "extended" and pinky == "extended":
        return "F"

    # D — index up, others curled, thumb touches curled fingers
    if index == "extended" and ext == 1 and thumb in ("bent", "curled") and middle == "curled" and ring == "curled" and pinky == "curled":
        return "D"

    # G — index points horizontally
    if index == "extended" and pointing_horizontal(landmarks, handedness) and middle == "curled" and ring == "curled" and pinky == "curled":
        return "G"

    # H — index + middle extended horizontally together
    if index == "extended" and middle == "extended" and index_middle_together(landmarks) and pointing_horizontal(landmarks, handedness):
        return "H"

    # X — index bent/hooked, others curled
    if index == "bent" and middle == "curled" and ring == "curled" and pinky == "curled":
        return "X"

    # C — curved partial extension (2-3 fingers bent/extended, not full B)
    if ext >= 2 and ext <= 3 and thumb != "extended" and not (index == "extended" and middle == "extended" and ring == "extended" and pinky == "extended"):
        bent_count = sum(1 for f in s[1:] if f == "bent")
        if bent_count >= 1 or (ext == 3 and pinky != "extended"):
            return "C"

    # A — fist, thumb alongside (not extended up)
    if ext == 0 and thumb in ("bent", "curled") and all(f == "curled" for f in s[1:]):
        return "A"

    # S — tight fist, thumb wrapped over
    if ext == 0 and thumb == "curled":
        return "S"

    # T — thumb pops between index and middle in partial fist
    if ext == 0 and thumb == "bent" and thumb_between_index_middle(landmarks):
        return "T"

    # E — fingers curled, thumb tucked over (all curled including thumb bent)
    if all(f in ("curled", "bent") for f in s[1:]) and thumb == "bent" and ext == 0:
        return "E"

    # M — thumb under three fingers (thumb curled, three fingers bent over)
    if thumb == "curled" and index == "bent" and middle == "bent" and ring == "bent" and pinky == "curled":
        return "M"

    # N — thumb under two fingers
    if thumb == "curled" and index == "bent" and middle == "bent" and ring == "curled" and pinky == "curled":
        return "N"

    # P — like K but pointing downward (index/middle down)
    if index == "extended" and middle == "extended" and thumb_between_index_middle(landmarks):
        index_tip_y = landmarks[INDEX_TIP][1]
        wrist_y = landmarks[0][1]
        if index_tip_y > wrist_y:
            return "P"

    # Q — thumb + index pointing down/side
    if thumb == "extended" and index == "extended" and middle == "curled" and ring == "curled" and pinky == "curled":
        if landmarks[INDEX_TIP][1] > landmarks[0][1]:
            return "Q"

    # Z — index extended (static; motion draws Z in ASL)
    if index == "extended" and ext == 1 and middle == "curled" and ring == "curled" and pinky == "curled" and thumb != "extended":
        return "Z"

    return None
