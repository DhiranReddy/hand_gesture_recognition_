"""Shared fixtures for the gesture test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# A neutral open palm in pixel space, wrist at the bottom and fingers rising.
# Landmark order follows MediaPipe: wrist, thumb(1-4), index(5-8), middle(9-12),
# ring(13-16), pinky(17-20).
OPEN_PALM = [
    (480.0, 460.0, 0.0),
    (440.0, 435.0, 0.005), (412.0, 405.0, 0.008), (392.0, 378.0, 0.010), (376.0, 352.0, 0.012),
    (452.0, 372.0, 0.002), (444.0, 330.0, 0.004), (440.0, 302.0, 0.006), (437.0, 278.0, 0.008),
    (480.0, 364.0, 0.000), (480.0, 318.0, 0.002), (480.0, 288.0, 0.004), (480.0, 262.0, 0.006),
    (508.0, 370.0, -0.002), (514.0, 326.0, -0.001), (518.0, 298.0, 0.001), (521.0, 274.0, 0.003),
    (534.0, 384.0, -0.004), (546.0, 350.0, -0.003), (554.0, 328.0, -0.002), (560.0, 308.0, -0.001),
]


@pytest.fixture
def open_palm() -> list[tuple[float, float, float]]:
    return list(OPEN_PALM)


@pytest.fixture
def fist() -> list[tuple[float, float, float]]:
    """Curled hand: fingertips pulled back toward the palm."""
    hand = list(OPEN_PALM)
    for tip, pip in ((4, 2), (8, 6), (12, 10), (16, 14), (20, 18)):
        px, py, pz = hand[pip]
        hand[tip] = (px, py + 26.0, pz)
    return hand
