"""Unified gesture classification: phrases, letters, and numbers."""

from __future__ import annotations

from src.asl_alphabet import classify_letter
from src.asl_numbers import classify_number
from src.common_phrases import PHRASE_LABELS, classify_phrase

# Recognition mode: what to prioritize when shapes overlap.
MODE_WORDS = "words"      # Common phrases for fast conversation
MODE_SPELL = "spell"      # A-Z and 0-9 fingerspelling
MODE_ALL = "all"          # Try everything; prefer most specific match


LETTER_LABELS: dict[str, str] = {chr(c): chr(c) for c in range(ord("A"), ord("Z") + 1)}
NUMBER_LABELS: dict[str, str] = {str(i): str(i) for i in range(10)}

ALL_LABELS: dict[str, str] = {**PHRASE_LABELS, **LETTER_LABELS, **NUMBER_LABELS}


def classify_gesture(
    landmarks: list[tuple],
    handedness: str,
    mode: str = MODE_ALL,
) -> tuple[str | None, str]:
    """
    Classify hand pose. Returns (gesture_id, category).
    category is one of: phrase, letter, number, none
    """
    phrase = classify_phrase(landmarks, handedness)
    number = classify_number(landmarks, handedness)
    letter = classify_letter(landmarks, handedness)

    if mode == MODE_WORDS:
        if phrase:
            return phrase, "phrase"
        if letter:
            return letter, "letter"
        if number:
            return number, "number"
        return None, "none"

    if mode == MODE_SPELL:
        if letter:
            return letter, "letter"
        if number:
            return number, "number"
        if phrase:
            return phrase, "phrase"
        return None, "none"

    # MODE_ALL — unique gestures first, then spell, then phrases
    unique_phrases = {
        "LOVE", "OK", "HELP", "FOOD", "WATER", "BATHROOM", "EMERGENCY",
        "QUESTION", "DONT_UNDERSTAND", "UNDERSTAND", "CAN_YOU_HEAR", "CAN_YOU_SEE",
    }
    if phrase and phrase in unique_phrases:
        return phrase, "phrase"

    if letter:
        return letter, "letter"
    if number:
        return number, "number"
    if phrase:
        return phrase, "phrase"

    return None, "none"


def gesture_to_text(gesture_id: str | None) -> str | None:
    if gesture_id is None:
        return None
    known = ALL_LABELS.get(gesture_id)
    if known:
        return known
    # Fallback allows dataset-defined custom IDs like NEED_HELP.
    return gesture_id.replace("_", " ").title()


def is_single_char(text: str | None) -> bool:
    return bool(text) and len(text) == 1
