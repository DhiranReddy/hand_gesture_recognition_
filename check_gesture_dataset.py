#!/usr/bin/env python3
"""Run quality checks on the custom gesture dataset."""

from __future__ import annotations

import argparse
import json
import sys

from src.dataset_quality import DatasetQualityError, analyze_dataset
from src.profile_paths import DEFAULT_PROFILE, dataset_path_for_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check custom gesture dataset quality")
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help="Profile name for personalized datasets (default: default)",
    )
    parser.add_argument(
        "--min-samples-per-label",
        type=int,
        default=20,
        help="Minimum recommended samples per class",
    )
    parser.add_argument(
        "--duplicate-decimals",
        type=int,
        default=4,
        help="Decimal precision used to detect near-duplicates",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when quality issues are found",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full report as JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = dataset_path_for_profile(args.profile)
    try:
        report = analyze_dataset(
            dataset_path,
            min_samples_per_label=args.min_samples_per_label,
            duplicate_decimals=args.duplicate_decimals,
        )
    except DatasetQualityError as exc:
        print(f"Dataset quality check failed: {exc}")
        return 1

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Dataset: {report['dataset_path']}")
        print(f"Samples: {report['sample_count']}")
        print(f"Labels: {report['label_count']}")
        print(f"Imbalance ratio (max/min): {report['imbalance_ratio']:.2f}")
        print("Per-label counts:")
        for label, count in sorted(report["label_counts"].items()):
            print(f"  - {label}: {count}")

        issues = report.get("issues", [])
        if issues:
            print("Issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("No issues detected.")

    if args.strict and not report.get("healthy", False):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
