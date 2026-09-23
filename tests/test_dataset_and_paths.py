"""Dataset round-tripping, validation, and profile path handling."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.custom_dataset import (
    DatasetError,
    append_sample,
    canonical_label,
    load_dataset,
)
from src.gesture_classifier import gesture_to_text, is_single_char
from src.profile_paths import (
    DEFAULT_PROFILE,
    LEGACY_DATASET_PATH,
    LEGACY_MODEL_PATH,
    dataset_path_for_profile,
    model_path_for_profile,
    profile_dir,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("need help", "NEED_HELP"),
        ("  Thank   You  ", "THANK_YOU"),
        ("call_doctor", "CALL_DOCTOR"),
        ("a", "A"),
    ],
)
def test_canonical_label(raw, expected):
    assert canonical_label(raw) == expected


def write_samples(path, label, count, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        landmarks = rng.uniform(0, 400, size=(21, 3))
        append_sample(
            path,
            label=label,
            handedness="Right",
            landmarks=[tuple(p) for p in landmarks],
        )


def test_append_then_load_round_trip(tmp_path):
    dataset = tmp_path / "gesture_landmarks.jsonl"
    write_samples(dataset, "NEED_HELP", 25, seed=1)
    write_samples(dataset, "CALL_DOCTOR", 25, seed=2)

    X, y, counts, path = load_dataset(dataset, min_samples_per_label=20)

    assert X.shape == (50, 73)
    assert path == dataset
    assert counts == {"NEED_HELP": 25, "CALL_DOCTOR": 25}
    assert set(y) == {"NEED_HELP", "CALL_DOCTOR"}


def test_labels_below_threshold_are_dropped(tmp_path):
    dataset = tmp_path / "d.jsonl"
    write_samples(dataset, "KEEP_A", 25, seed=1)
    write_samples(dataset, "KEEP_B", 25, seed=2)
    write_samples(dataset, "TOO_FEW", 3, seed=3)

    _, y, counts, _ = load_dataset(dataset, min_samples_per_label=20)
    assert "TOO_FEW" not in counts
    assert "TOO_FEW" not in set(y)


def test_missing_dataset_raises(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path / "absent.jsonl")


def test_single_label_dataset_raises(tmp_path):
    dataset = tmp_path / "d.jsonl"
    write_samples(dataset, "ONLY", 30)
    with pytest.raises(DatasetError, match="at least 2 labels"):
        load_dataset(dataset, min_samples_per_label=20)


def test_malformed_json_reports_line_number(tmp_path):
    dataset = tmp_path / "d.jsonl"
    write_samples(dataset, "GOOD", 2)
    with dataset.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    with pytest.raises(DatasetError, match="line 3"):
        load_dataset(dataset, min_samples_per_label=1)


def test_wrong_landmark_count_rejected(tmp_path):
    dataset = tmp_path / "d.jsonl"
    dataset.write_text(
        json.dumps({"label": "X", "handedness": "Right", "landmarks": [[0, 0, 0]] * 20})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="21 landmarks"):
        load_dataset(dataset, min_samples_per_label=1)


def test_blank_lines_are_skipped(tmp_path):
    dataset = tmp_path / "d.jsonl"
    write_samples(dataset, "A_LABEL", 21, seed=4)
    write_samples(dataset, "B_LABEL", 21, seed=5)
    with dataset.open("a", encoding="utf-8") as handle:
        handle.write("\n\n")
    X, _, counts, _ = load_dataset(dataset, min_samples_per_label=20)
    assert X.shape[0] == 42
    assert counts == {"A_LABEL": 21, "B_LABEL": 21}


# ---------------------------------------------------------------- profiles


def test_default_profile_uses_legacy_paths():
    assert model_path_for_profile(DEFAULT_PROFILE) == LEGACY_MODEL_PATH
    assert dataset_path_for_profile(DEFAULT_PROFILE) == LEGACY_DATASET_PATH


def test_named_profile_is_isolated():
    path = dataset_path_for_profile("Alice")
    assert path.parent.name == "alice"
    assert path != LEGACY_DATASET_PATH


@pytest.mark.parametrize("hostile", ["../../etc", "a/../../b", "..", "//root", "x\\y"])
def test_profile_names_cannot_escape_the_profiles_root(hostile):
    """Profile names reach the filesystem, so traversal must be impossible."""
    resolved = profile_dir(hostile).resolve()
    root = (profile_dir("anything").parent).resolve()
    assert root in resolved.parents or resolved == root


def test_empty_profile_name_falls_back_to_default():
    assert model_path_for_profile("!!!") == LEGACY_MODEL_PATH


# ------------------------------------------------------------------ labels


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("A", "A"),
        ("7", "7"),
        # Curated phrases win over title-casing the raw id.
        ("THANK_YOU", "Thank you"),
        ("LOVE", "I love you"),
        ("DONT_UNDERSTAND", "I don't understand"),
        # Unknown ids (custom datasets) fall back to title case.
        ("NEED_HELP", "Need Help"),
    ],
)
def test_gesture_to_text(label, expected):
    assert gesture_to_text(label) == expected


def test_gesture_to_text_none():
    assert gesture_to_text(None) is None


def test_is_single_char():
    assert is_single_char("A")
    assert not is_single_char("Thank You")
    assert not is_single_char(None)
