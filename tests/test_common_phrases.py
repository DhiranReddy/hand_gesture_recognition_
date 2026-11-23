"""Tests for common phrase recognition rules."""

from __future__ import annotations

import pytest

from src.common_phrases import classify_phrase
from src.gesture_classifier import MODE_ALL, MODE_SPELL, MODE_WORDS, classify_gesture
from tests.conftest import OPEN_PALM

FINGERS = {
    "thumb": (4, 2),
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18),
}


def fold_fingers(hand: list[tuple[float, float, float]], names: list[str]) -> list[tuple[float, float, float]]:
    out = list(hand)
    for name in names:
        tip, pip = FINGERS[name]
        px, py, pz = out[pip]
        out[tip] = (px, py + 26.0, pz)
    return out


def raise_fingers(*keep: str) -> list[tuple[float, float, float]]:
    to_fold = [n for n in FINGERS if n not in keep]
    return fold_fingers(OPEN_PALM, to_fold)


def thumbs_up() -> list[tuple[float, float, float]]:
    hand = fold_fingers(OPEN_PALM, ["index", "middle", "ring", "pinky"])
    hand[4] = (400.0, 300.0, 0.0)
    hand[3] = (415.0, 335.0, 0.0)
    hand[2] = (428.0, 372.0, 0.0)
    return hand


def thumbs_down() -> list[tuple[float, float, float]]:
    hand = thumbs_up()
    hand[4] = (400.0, 560.0, 0.0)
    hand[3] = (415.0, 528.0, 0.0)
    hand[2] = (428.0, 492.0, 0.0)
    return hand


def fingers_together(hand: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    out = list(hand)
    for i, x in ((8, 466.0), (7, 468.0), (6, 470.0), (12, 474.0), (11, 476.0), (10, 478.0)):
        y, z = out[i][1], out[i][2]
        out[i] = (x, y, z)
    return out


def sideways(hand: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    out = []
    for x, y, z in hand:
        out.append((480.0 + -(y - 460.0), 460.0 + (x - 480.0) * 0.35, z))
    return out


def test_yes_no_gestures():
    assert classify_phrase(thumbs_up(), "Left") == "YES"
    assert classify_phrase(thumbs_down(), "Left") == "NO"


def test_love_and_help_gestures():
    assert classify_phrase(raise_fingers("thumb", "index", "pinky"), "Left") == "LOVE"
    assert classify_phrase(raise_fingers("index", "pinky"), "Left") == "HELP"


def test_hello_and_please_gestures():
    assert classify_phrase(OPEN_PALM, "Left") == "HELLO"
    assert classify_phrase(fingers_together(OPEN_PALM), "Left") == "PLEASE"


def test_water_gesture():
    assert classify_phrase(raise_fingers("thumb", "index", "middle", "ring"), "Left") == "WATER"


def test_you_and_thank_you_gestures():
    assert classify_phrase(sideways(raise_fingers("index")), "Left") == "YOU"
    assert classify_phrase(sideways(fingers_together(OPEN_PALM)), "Left") == "THANK_YOU"


def test_fist_does_not_trigger_phrase(fist):
    # A closed fist should NOT be detected as PLEASE or any phrase (it is letter S/A)
    assert classify_phrase(fist, "Left") is None


def test_classify_gesture_modes():
    # In MODE_ALL, unique phrases like YES, NO, HELLO take precedence
    res, cat = classify_gesture(thumbs_up(), "Left", mode=MODE_ALL)
    assert res == "YES"
    assert cat == "phrase"

    res, cat = classify_gesture(OPEN_PALM, "Left", mode=MODE_ALL)
    assert res == "HELLO"
    assert cat == "phrase"

    # In MODE_SPELL, letters take precedence over phrases that overlap with letters
    res, cat = classify_gesture(OPEN_PALM, "Left", mode=MODE_SPELL)
    assert cat in ("letter", "number")
