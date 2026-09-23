"""
Pickle-free model persistence.

joblib serialises with pickle, which executes arbitrary code on load. This
project encourages sharing model and profile files, so a downloaded bundle
would be remote code execution. Models are therefore stored as plain JSON:
weights, the standardiser, the label list and metadata.

The loader reimplements the forward pass in numpy rather than rebuilding an
sklearn estimator. That keeps the file independent of the scikit-learn version
that produced it, and makes this the single reference for the maths the browser
also implements in web/src/classifier.js.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

FORMAT = "dense-mlp-v2"
MODEL_SUFFIX = ".json"


class ModelError(RuntimeError):
    """Raised when a model file is missing, malformed, or an unknown format."""


def dataset_fingerprint(labels: np.ndarray) -> str:
    """
    Identify the dataset a split was computed against.

    Hashing the label sequence is enough: train_test_split's output depends on
    the number of rows and their order, both of which this captures, and it
    stays stable regardless of float formatting in the landmarks.
    """
    digest = hashlib.sha256()
    for label in labels:
        digest.update(str(label).encode("utf-8"))
        digest.update(b"\x00")
    return f"sha256:{digest.hexdigest()[:32]}"


class DenseMLP:
    """Feed-forward network with optional input standardisation."""

    def __init__(
        self,
        layers: list[tuple[np.ndarray, np.ndarray]],
        classes: list[str],
        *,
        mean: np.ndarray | None = None,
        scale: np.ndarray | None = None,
    ) -> None:
        self.layers = layers
        self.classes_ = np.array(classes)
        self.mean = mean
        self.scale = scale

    @property
    def n_features_in_(self) -> int:
        return int(self.layers[0][0].shape[0])

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            return X
        return (X - self.mean) / self.scale

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        activations = self._standardise(np.asarray(X, dtype=np.float64))
        for index, (weights, bias) in enumerate(self.layers):
            activations = activations @ weights + bias
            if index < len(self.layers) - 1:
                np.maximum(activations, 0.0, out=activations)

        shifted = activations - activations.max(axis=1, keepdims=True)
        np.exp(shifted, out=shifted)
        return shifted / shifted.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def _extract(estimator) -> tuple[list, np.ndarray | None, np.ndarray | None]:
    """Pull weights and any standardiser out of a fitted sklearn pipeline."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    mean = scale = None
    net = estimator
    if isinstance(estimator, Pipeline):
        for _, step in estimator.steps:
            if isinstance(step, StandardScaler):
                mean, scale = step.mean_, step.scale_
            elif isinstance(step, MLPClassifier):
                net = step
    if not isinstance(net, MLPClassifier):
        raise ModelError(f"Expected an MLPClassifier, got {type(net).__name__}")
    if net.activation != "relu" or net.out_activation_ != "softmax":
        raise ModelError(
            f"Only relu/softmax is supported, got {net.activation}/{net.out_activation_}"
        )
    return list(zip(net.coefs_, net.intercepts_)), mean, scale


def save_model(
    path: Path,
    estimator,
    classes: list[str],
    metadata: dict[str, Any],
) -> Path:
    path = path.with_suffix(MODEL_SUFFIX)
    path.parent.mkdir(parents=True, exist_ok=True)

    layers, mean, scale = _extract(estimator)
    payload = {
        "format": FORMAT,
        "n_features": int(layers[0][0].shape[0]),
        "labels": [str(c) for c in classes],
        "standardizer": (
            None
            if mean is None
            else {
                "mean": np.asarray(mean, dtype=np.float64).round(8).tolist(),
                "scale": np.asarray(scale, dtype=np.float64).round(8).tolist(),
            }
        ),
        "layers": [
            {
                "weights": np.asarray(w, dtype=np.float64).round(8).tolist(),
                "bias": np.asarray(b, dtype=np.float64).round(8).tolist(),
            }
            for w, b in layers
        ],
        "metadata": metadata,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_model(path: Path) -> tuple[DenseMLP, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise ModelError(f"Model not found: {path}")
    if path.suffix == ".joblib":
        raise ModelError(
            f"{path} is a pickle. Convert it with tools/migrate_model.py — "
            "loading pickles is unsafe and no longer supported."
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelError(f"{path} is not valid JSON: {exc}") from exc

    if payload.get("format") != FORMAT:
        raise ModelError(f"Unsupported model format: {payload.get('format')!r}")

    standardizer = payload.get("standardizer")
    model = DenseMLP(
        [
            (
                np.asarray(layer["weights"], dtype=np.float64),
                np.asarray(layer["bias"], dtype=np.float64),
            )
            for layer in payload["layers"]
        ],
        payload["labels"],
        mean=None if not standardizer else np.asarray(standardizer["mean"]),
        scale=None if not standardizer else np.asarray(standardizer["scale"]),
    )
    return model, payload.get("metadata", {})
