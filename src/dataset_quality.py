"""Quality checks for custom gesture landmark datasets."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.custom_dataset import DATASET_PATH, DatasetError, canonical_label
from src.landmark_features import landmarks_to_features


class DatasetQualityError(RuntimeError):
    """Raised when dataset quality checks cannot run."""


def _parse_landmarks(value: Any) -> list[tuple[float, float, float]]:
    if not isinstance(value, list) or len(value) != 21:
        raise DatasetError("Each sample must include 21 landmarks.")

    points: list[tuple[float, float, float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 3:
            raise DatasetError("Each landmark must be a 3-value list [x, y, z].")
        points.append((float(point[0]), float(point[1]), float(point[2])))
    return points


def _load_feature_rows(dataset_path: Path) -> tuple[list[str], np.ndarray]:
    if not dataset_path.exists():
        raise DatasetQualityError(f"Dataset not found at {dataset_path}")

    labels: list[str] = []
    features: list[np.ndarray] = []

    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetQualityError(f"Invalid JSON at line {line_number}: {exc}") from exc

            label = canonical_label(str(row.get("label", "")).strip())
            if not label:
                raise DatasetQualityError(f"Missing label at line {line_number}")

            handedness = str(row.get("handedness", "Right")).strip() or "Right"
            handedness = handedness if handedness in {"Left", "Right"} else "Right"

            try:
                landmarks = _parse_landmarks(row.get("landmarks"))
            except DatasetError as exc:
                raise DatasetQualityError(f"Invalid landmarks at line {line_number}: {exc}") from exc

            labels.append(label)
            features.append(landmarks_to_features(landmarks, handedness))

    if not labels:
        raise DatasetQualityError(f"Dataset at {dataset_path} has no samples")

    return labels, np.array(features, dtype=np.float32)


def analyze_dataset(
    dataset_path: Path | None = None,
    *,
    min_samples_per_label: int = 20,
    duplicate_decimals: int = 4,
    duplicate_ratio_warn: float = 0.35,
    low_variance_warn: float = 0.015,
) -> dict[str, Any]:
    """Return quality metrics and issues for a custom gesture dataset."""
    path = dataset_path or DATASET_PATH
    labels, feature_matrix = _load_feature_rows(path)
    counts = Counter(labels)

    by_label_indices: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label_indices[label].append(idx)

    duplicate_rates: dict[str, float] = {}
    low_variance_scores: dict[str, float] = {}
    high_duplicate_labels: list[str] = []
    low_variance_labels: list[str] = []

    for label, indices in by_label_indices.items():
        block = feature_matrix[np.array(indices)]
        rounded = np.round(block, duplicate_decimals)
        unique_count = int(np.unique(rounded, axis=0).shape[0])
        duplicate_ratio = 1.0 - (unique_count / len(indices))
        duplicate_rates[label] = float(duplicate_ratio)
        if duplicate_ratio >= duplicate_ratio_warn:
            high_duplicate_labels.append(label)

        score = float(np.mean(np.std(block, axis=0)))
        low_variance_scores[label] = score
        if score <= low_variance_warn:
            low_variance_labels.append(label)

    labels_below_min = sorted([label for label, count in counts.items() if count < min_samples_per_label])

    imbalance_ratio = 1.0
    if counts:
        imbalance_ratio = float(max(counts.values()) / max(1, min(counts.values())))

    issues: list[str] = []
    if len(counts) < 2:
        issues.append("Need at least 2 labels for meaningful classifier training.")
    if labels_below_min:
        issues.append(
            "Labels below minimum samples "
            f"({min_samples_per_label}): {', '.join(labels_below_min)}"
        )
    if high_duplicate_labels:
        issues.append(
            "High near-duplicate rate in labels: "
            f"{', '.join(sorted(high_duplicate_labels))}. "
            "Capture more pose/angle variation."
        )
    if low_variance_labels:
        issues.append(
            "Low feature variance in labels: "
            f"{', '.join(sorted(low_variance_labels))}. "
            "Move slightly and vary distance/rotation while capturing."
        )
    if imbalance_ratio >= 4.0:
        issues.append(
            f"Strong class imbalance detected (max/min sample ratio: {imbalance_ratio:.2f})."
        )

    return {
        "dataset_path": str(path),
        "sample_count": int(len(labels)),
        "label_count": int(len(counts)),
        "label_counts": dict(counts),
        "imbalance_ratio": imbalance_ratio,
        "labels_below_min": labels_below_min,
        "duplicate_rates": duplicate_rates,
        "low_variance_scores": low_variance_scores,
        "issues": issues,
        "healthy": not issues,
    }
