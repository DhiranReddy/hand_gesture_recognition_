"""Feature extraction invariants.

These properties are what make the model usable across people, cameras, and
distances. If any of them break, accuracy degrades in ways that are very hard
to diagnose from the live app.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.landmark_features import landmarks_to_features


def test_feature_vector_length(open_palm):
    assert landmarks_to_features(open_palm, "Right").shape == (73,)


def test_translation_invariant(open_palm):
    """Moving the hand across the frame must not change the features."""
    shifted = [(x + 137.0, y - 64.0, z) for x, y, z in open_palm]
    np.testing.assert_allclose(
        landmarks_to_features(open_palm, "Right"),
        landmarks_to_features(shifted, "Right"),
        atol=1e-5,
    )


def test_scale_invariant(open_palm):
    """A hand twice as close to the camera must produce the same features."""
    wx, wy, wz = open_palm[0]
    scaled = [
        (wx + (x - wx) * 2.0, wy + (y - wy) * 2.0, wz + (z - wz) * 2.0)
        for x, y, z in open_palm
    ]
    np.testing.assert_allclose(
        landmarks_to_features(open_palm, "Right"),
        landmarks_to_features(scaled, "Right"),
        atol=1e-5,
    )


def test_handedness_mirrors_x_only(open_palm):
    right = landmarks_to_features(open_palm, "Right")
    left = landmarks_to_features(open_palm, "Left")

    coords_r = right[:63].reshape(21, 3)
    coords_l = left[:63].reshape(21, 3)

    np.testing.assert_allclose(coords_l[:, 0], -coords_r[:, 0], atol=1e-6)
    np.testing.assert_allclose(coords_l[:, 1:], coords_r[:, 1:], atol=1e-6)
    # Mirroring is a rigid reflection, so pairwise tip distances are unchanged.
    np.testing.assert_allclose(left[63:], right[63:], atol=1e-6)


def test_wrist_is_origin(open_palm):
    coords = landmarks_to_features(open_palm, "Right")[:63].reshape(21, 3)
    np.testing.assert_allclose(coords[0], np.zeros(3), atol=1e-6)


def test_degenerate_palm_does_not_divide_by_zero():
    """A collapsed hand must not produce inf/nan and poison the model."""
    collapsed = [(100.0, 100.0, 0.0)] * 21
    features = landmarks_to_features(collapsed, "Right")
    assert np.all(np.isfinite(features))


def test_fist_and_palm_are_distinguishable(open_palm, fist):
    palm_features = landmarks_to_features(open_palm, "Right")
    fist_features = landmarks_to_features(fist, "Right")
    assert np.linalg.norm(palm_features - fist_features) > 0.5


@pytest.mark.parametrize("handedness", ["Right", "Left"])
def test_output_is_float32_finite(open_palm, handedness):
    features = landmarks_to_features(open_palm, handedness)
    assert features.dtype == np.float32
    assert np.all(np.isfinite(features))
