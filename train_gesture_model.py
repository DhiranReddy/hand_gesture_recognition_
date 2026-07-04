#!/usr/bin/env python3
"""Train or retrain the custom gesture MLP model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from src.custom_dataset import DatasetError
from src.dataset_quality import DatasetQualityError, analyze_dataset
from src.ml_gesture_model import train_and_save
from src.profile_paths import (
    DEFAULT_PROFILE,
    dataset_path_for_profile,
    model_path_for_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train custom gesture model from JSONL landmarks dataset")
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help="Profile name for personalized datasets/models (default: default)",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Override dataset path (otherwise derived from --profile)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Override model output path (otherwise derived from --profile)",
    )
    parser.add_argument(
        "--min-samples-per-label",
        type=int,
        default=20,
        help="Minimum samples per class to include in training",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Validation split ratio used for evaluation output",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for deterministic train/validation split",
    )
    parser.add_argument(
        "--balance-target-per-label",
        type=int,
        default=220,
        help="Upsample minority classes to this many training samples (capped by largest class)",
    )
    parser.add_argument(
        "--augment-noise-std",
        type=float,
        default=0.08,
        help="Feature-space Gaussian noise std used for synthetic augmented samples",
    )
    parser.add_argument(
        "--quality-check",
        action="store_true",
        help="Run dataset quality checks before training",
    )
    parser.add_argument(
        "--strict-quality",
        action="store_true",
        help="Fail training when quality issues are detected (implies --quality-check)",
    )
    parser.add_argument(
        "--show-report",
        action="store_true",
        help="Print per-label precision/recall report JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path or dataset_path_for_profile(args.profile)
    model_path = args.model_path or model_path_for_profile(args.profile)

    if args.strict_quality:
        args.quality_check = True

    if args.quality_check:
        try:
            quality = analyze_dataset(
                dataset_path,
                min_samples_per_label=args.min_samples_per_label,
            )
        except DatasetQualityError as exc:
            print(f"Dataset quality check failed: {exc}")
            sys.exit(1)

        issues = quality.get("issues", [])
        print(f"Quality check: labels={quality.get('label_count')} samples={quality.get('sample_count')}")
        if issues:
            print("Quality warnings:")
            for issue in issues:
                print(f"- {issue}")
            if args.strict_quality:
                print("Aborting due to --strict-quality.")
                sys.exit(1)
        else:
            print("Quality check passed: no issues detected.")

    try:
        path, metadata = train_and_save(
            model_path,
            dataset_path=dataset_path,
            min_samples_per_label=args.min_samples_per_label,
            test_size=args.test_size,
            random_state=args.random_state,
            balance_target_per_label=args.balance_target_per_label,
            augment_noise_std=args.augment_noise_std,
        )
    except DatasetError as exc:
        print(f"Training failed: {exc}")
        print(f"Expected dataset file: {dataset_path}")
        print("Run capture_gesture_dataset.py to record samples first.")
        sys.exit(1)

    print(f"Model saved to {path}")
    print(f"Dataset: {metadata.get('dataset_path')}")
    print(f"Samples: {metadata.get('sample_count')}")
    print(f"Labels: {len(metadata.get('label_counts', {}))}")
    print(
        "Balanced train samples: "
        f"{metadata.get('train_samples_before_balance', 0)}"
        f" -> {metadata.get('train_samples_after_balance', 0)}"
    )
    print(f"Augmented samples created: {metadata.get('augmented_samples_created', 0)}")
    print(f"Validation accuracy: {metadata.get('test_accuracy', 0.0):.2%}")

    if args.show_report:
        report = metadata.get("classification_report", {})
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
