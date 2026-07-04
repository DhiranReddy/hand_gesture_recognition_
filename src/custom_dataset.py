"""Utilities for storing and loading custom landmark datasets."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.landmark_features import landmarks_to_features

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "gesture_landmarks.jsonl"


class DatasetError(RuntimeError):
    """Raised when the custom dataset is missing or invalid."""


def canonical_label(label: str) -> str:
    """Normalize labels to uppercase IDs used by the classifier."""
    return "_".join(label.strip().upper().split())


def _parse_landmarks(value: Any) -> list[tuple[float, float, float]]:
    if not isinstance(value, list) or len(value) != 21:
        raise DatasetError("Each sample must include 21 landmarks.")

    parsed: list[tuple[float, float, float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 3:
            raise DatasetError("Each landmark must be a 3-value list [x, y, z].")
        parsed.append((float(point[0]), float(point[1]), float(point[2])))
    return parsed


def append_sample(
    dataset_path: Path,
    *,
    label: str,
    handedness: str,
    landmarks: list[tuple[float, float, float]],
) -> None:
    """Append one labeled sample to the dataset."""
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": canonical_label(label),
        "handedness": handedness,
        "landmarks": [[float(x), float(y), float(z)] for x, y, z in landmarks],
    }
    with dataset_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def load_dataset(
    dataset_path: Path | None = None,
    *,
    min_samples_per_label: int = 20,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], Path]:
    """Load dataset from JSONL and return feature matrix, labels, and counts."""
    path = dataset_path or DATASET_PATH
    if not path.exists():
        raise DatasetError(
            f"Dataset not found at {path}. Use capture_gesture_dataset.py to collect samples."
        )

    X: list[np.ndarray] = []
    y: list[str] = []
    counts: Counter[str] = Counter()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"Invalid JSON at line {line_number}: {exc}") from exc

            label_raw = str(record.get("label", "")).strip()
            if not label_raw:
                raise DatasetError(f"Missing label at line {line_number}.")
            label = canonical_label(label_raw)

            handedness = str(record.get("handedness", "Right")).strip() or "Right"
            handedness = handedness if handedness in {"Left", "Right"} else "Right"
            landmarks = _parse_landmarks(record.get("landmarks"))

            X.append(landmarks_to_features(landmarks, handedness))
            y.append(label)
            counts[label] += 1

    if not X:
        raise DatasetError(f"Dataset at {path} has no samples.")

    kept = {label for label, count in counts.items() if count >= min_samples_per_label}
    if len(kept) < 2:
        raise DatasetError(
            "Need at least 2 labels with enough samples. "
            f"Current counts: {dict(counts)}"
        )

    if kept != set(counts):
        filtered_X: list[np.ndarray] = []
        filtered_y: list[str] = []
        for features, label in zip(X, y):
            if label in kept:
                filtered_X.append(features)
                filtered_y.append(label)
        X = filtered_X
        y = filtered_y

    return np.array(X, dtype=np.float32), np.array(y), dict(Counter(y)), path
