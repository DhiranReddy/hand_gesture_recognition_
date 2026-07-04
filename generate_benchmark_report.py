#!/usr/bin/env python3
"""Generate benchmark metrics and confusion matrix for a profile model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.custom_dataset import DatasetError, load_dataset
from src.profile_paths import (
    DEFAULT_PROFILE,
    dataset_path_for_profile,
    model_path_for_profile,
    reports_dir_for_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark report for a trained profile model")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="Profile name")
    parser.add_argument("--min-samples-per-label", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _save_confusion_matrix(path: Path, cm: np.ndarray, labels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), max(6, len(labels) * 0.45)))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    threshold = cm.max() * 0.65 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            color = "white" if val > threshold else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()

    dataset_path = dataset_path_for_profile(args.profile)
    model_path = model_path_for_profile(args.profile)
    reports_dir = reports_dir_for_profile(args.profile)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Train a profile model first using train_gesture_model.py")
        return 1

    try:
        X, y, _, _ = load_dataset(dataset_path=dataset_path, min_samples_per_label=args.min_samples_per_label)
    except DatasetError as exc:
        print(f"Dataset load failed: {exc}")
        return 1

    data = joblib.load(model_path)
    model = data.get("model")
    encoder = data.get("encoder")
    if model is None or encoder is None:
        print(f"Model bundle at {model_path} is invalid.")
        return 1

    known = set(str(label) for label in encoder.classes_)
    keep_mask = np.array([label in known for label in y], dtype=bool)
    X = X[keep_mask]
    y = y[keep_mask]
    if len(y) < 2:
        print("Not enough known-label samples to benchmark this model.")
        return 1

    y_enc = encoder.transform(y)
    stratify = y_enc if len(np.unique(y_enc)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )
    _ = X_train, y_train

    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))

    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(encoder.classes_)),
        target_names=list(encoder.classes_),
        zero_division=0,
        output_dict=True,
    )

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(encoder.classes_)))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    cm_path = reports_dir / f"confusion_matrix_{timestamp}.png"
    report_path = reports_dir / f"benchmark_{timestamp}.md"
    history_path = reports_dir / "benchmark_history.jsonl"

    _save_confusion_matrix(cm_path, cm, list(encoder.classes_))

    summary = {
        "timestamp": timestamp,
        "profile": args.profile,
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "accuracy": accuracy,
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "weighted_f1": float(report.get("weighted avg", {}).get("f1-score", 0.0)),
        "sample_count_eval": int(len(y_test)),
        "confusion_matrix": str(cm_path),
    }

    previous = None
    if history_path.exists():
        with history_path.open("r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]
        if lines:
            previous = json.loads(lines[-1])

    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary) + "\n")

    delta_line = "No previous benchmark found."
    if previous:
        acc_delta = summary["accuracy"] - float(previous.get("accuracy", 0.0))
        f1_delta = summary["macro_f1"] - float(previous.get("macro_f1", 0.0))
        delta_line = (
            f"Accuracy delta vs previous: {acc_delta:+.2%}; "
            f"Macro F1 delta: {f1_delta:+.2%}."
        )

    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# Benchmark Report ({timestamp})\n\n")
        handle.write(f"- Profile: {args.profile}\n")
        handle.write(f"- Model: {model_path}\n")
        handle.write(f"- Dataset: {dataset_path}\n")
        handle.write(f"- Eval samples: {len(y_test)}\n")
        handle.write(f"- Accuracy: {accuracy:.2%}\n")
        handle.write(f"- Macro F1: {summary['macro_f1']:.3f}\n")
        handle.write(f"- Weighted F1: {summary['weighted_f1']:.3f}\n")
        handle.write(f"- {delta_line}\n\n")
        handle.write(f"Confusion matrix: {cm_path.name}\n\n")
        handle.write("## Per-label Metrics\n\n")
        handle.write("| Label | Precision | Recall | F1 | Support |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for label in encoder.classes_:
            stats = report.get(label, {})
            handle.write(
                f"| {label} | {stats.get('precision', 0.0):.3f} | {stats.get('recall', 0.0):.3f} | "
                f"{stats.get('f1-score', 0.0):.3f} | {int(stats.get('support', 0))} |\n"
            )

    print(f"Benchmark report: {report_path}")
    print(f"Confusion matrix: {cm_path}")
    print(delta_line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
