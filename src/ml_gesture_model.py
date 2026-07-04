"""MLP gesture classifier trained on custom hand landmark datasets."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from src.custom_dataset import DatasetError, load_dataset
from src.landmark_features import landmarks_to_features

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "gesture_mlp.joblib"


def _balance_and_augment_training_set(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    target_per_label: int,
    noise_std: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, int], int]:
    """Upsample minority classes with noisy feature augmentations."""
    rng = np.random.default_rng(random_state)

    class_ids, class_counts = np.unique(y_train, return_counts=True)
    before = {str(int(cid)): int(count) for cid, count in zip(class_ids, class_counts)}

    if len(class_counts) == 0:
        return X_train, y_train, before, before, 0

    effective_target = min(int(target_per_label), int(np.max(class_counts)))
    if effective_target <= 0:
        return X_train, y_train, before, before, 0

    X_chunks: list[np.ndarray] = [X_train]
    y_chunks: list[np.ndarray] = [y_train]
    created = 0

    for class_id, class_count in zip(class_ids, class_counts):
        need = int(effective_target - class_count)
        if need <= 0:
            continue

        rows = X_train[y_train == class_id]
        picks = rng.integers(0, len(rows), size=need)
        selected = rows[picks]

        feature_std = np.std(rows, axis=0)
        scaled_noise = rng.normal(0.0, noise_std, size=selected.shape) * np.maximum(feature_std, 0.05)
        augmented = selected + scaled_noise.astype(np.float32)

        X_chunks.append(augmented)
        y_chunks.append(np.full(need, class_id, dtype=y_train.dtype))
        created += need

    X_balanced = np.vstack(X_chunks).astype(np.float32)
    y_balanced = np.concatenate(y_chunks)

    after_ids, after_counts = np.unique(y_balanced, return_counts=True)
    after = {str(int(cid)): int(count) for cid, count in zip(after_ids, after_counts)}
    return X_balanced, y_balanced, before, after, created

def _build_classifier(random_state: int = 42) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(192, 96),
        activation="relu",
        max_iter=700,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.12,
        learning_rate_init=8e-4,
    )


def train_and_save(
    model_path: Path | None = None,
    *,
    min_samples_per_label: int = 20,
    test_size: float = 0.2,
    random_state: int = 42,
    balance_target_per_label: int = 220,
    augment_noise_std: float = 0.08,
) -> tuple[Path, dict[str, Any]]:
    path = model_path or MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    X, y, counts, dataset_path = load_dataset(
        min_samples_per_label=min_samples_per_label,
    )
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    stratify = y_enc if len(np.unique(y_enc)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    X_train_balanced, y_train_balanced, balance_before, balance_after, augmented_count = (
        _balance_and_augment_training_set(
            X_train,
            y_train,
            target_per_label=balance_target_per_label,
            noise_std=augment_noise_std,
            random_state=random_state,
        )
    )

    clf = _build_classifier(random_state=random_state)
    clf.fit(X_train_balanced, y_train_balanced)

    y_pred = clf.predict(X_test)
    accuracy = float(np.mean(y_pred == y_test))
    class_report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(encoder.classes_)),
        target_names=list(encoder.classes_),
        zero_division=0,
        output_dict=True,
    )

    metadata: dict[str, Any] = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "min_samples_per_label": int(min_samples_per_label),
        "sample_count": int(len(y)),
        "label_counts": counts,
        "test_size": float(test_size),
        "balance_target_per_label": int(balance_target_per_label),
        "augment_noise_std": float(augment_noise_std),
        "train_samples_before_balance": int(len(y_train)),
        "train_samples_after_balance": int(len(y_train_balanced)),
        "augmented_samples_created": int(augmented_count),
        "balance_before": balance_before,
        "balance_after": balance_after,
        "test_accuracy": accuracy,
        "classification_report": class_report,
    }

    joblib.dump({"model": clf, "encoder": encoder, "metadata": metadata}, path)
    return path, metadata


class MLGestureClassifier:
    """Loads trained MLP and predicts gesture from landmarks."""

    MIN_CONFIDENCE = 0.56
    MIN_MARGIN = 0.10

    def __init__(self, model_path: Path | None = None):
        self._model = None
        self._encoder = None
        self._metadata: dict[str, Any] = {}
        self._path = model_path or MODEL_PATH
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            try:
                train_and_save(self._path)
            except DatasetError as exc:
                print(f"Custom gesture model not found and no dataset available: {exc}")
                return
        try:
            data = joblib.load(self._path)
        except Exception as exc:
            print(f"Could not load gesture model ({exc}).")
            return
        self._model = data["model"]
        self._encoder = data["encoder"]
        self._metadata = data.get("metadata", {})

    def predict(self, landmarks: list[tuple], handedness: str) -> tuple[str | None, float]:
        if self._model is None or self._encoder is None:
            return None, 0.0

        features = landmarks_to_features(landmarks, handedness)
        probs = self._model.predict_proba(features.reshape(1, -1))[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        sorted_probs = np.sort(probs)
        margin = float(sorted_probs[-1] - sorted_probs[-2]) if len(sorted_probs) > 1 else confidence

        if confidence < self.MIN_CONFIDENCE or margin < self.MIN_MARGIN:
            return None, confidence

        return str(self._encoder.inverse_transform([idx])[0]), confidence

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)
