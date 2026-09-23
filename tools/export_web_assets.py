#!/usr/bin/env python3
"""
Export training artifacts into browser-loadable JSON.

The web app runs inference client-side, so it needs the MLP weights and a
compact English frequency list rather than the joblib bundle.

    python tools/export_web_assets.py
    python tools/export_web_assets.py --profile myname
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WEB_ASSETS = ROOT / "web" / "public" / "model"

# Trimmed from pyspellchecker's 160k list — enough to correct fingerspelling
# without shipping a multi-megabyte dictionary to every visitor.
WORDLIST_SIZE = 20000

# Words worth having regardless of corpus rank, because they carry the
# conversations this app exists for.
PRIORITY_WORDS = (
    "hello hi hey goodbye bye please thanks thank sorry yes no help water food "
    "bathroom emergency doctor nurse hospital pain medicine ambulance love "
    "understand question answer wait stop start again more less good bad fine "
    "okay name meet zoom teams call mute unmute hear see today tomorrow "
    "yesterday morning night now later here there family friend home work "
    "school hungry thirsty tired cold hot sick hurt happy sad"
).split()


def export_model(model_path: Path, out_path: Path) -> dict[str, Any]:
    """Republish the trained model for the browser, adding display names."""
    from src.gesture_classifier import ALL_LABELS
    from src.model_io import load_model

    model, metadata = load_model(model_path)

    payload = {
        "format": "dense-mlp-v2",
        "n_features": int(model.n_features_in_),
        "labels": [str(c) for c in model.classes_],
        # Whole table, not just this model's labels, so a retrained model
        # with more phrases renders correctly without re-exporting.
        "display_names": dict(ALL_LABELS),
        "standardizer": (
            None
            if model.mean is None
            else {
                "mean": np.asarray(model.mean, dtype=np.float64).round(8).tolist(),
                "scale": np.asarray(model.scale, dtype=np.float64).round(8).tolist(),
            }
        ),
        "layers": [
            {
                "weights": np.asarray(w, dtype=np.float64).round(8).tolist(),
                "bias": np.asarray(b, dtype=np.float64).round(8).tolist(),
            }
            for w, b in model.layers
        ],
        "source_metadata": metadata,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def export_wordlist(out_path: Path, size: int = WORDLIST_SIZE) -> int:
    """Emit the most frequent English words as a rank-ordered array."""
    src = (
        Path(__import__("spellchecker").__file__).parent
        / "resources"
        / "en.json.gz"
    )
    if not src.exists():
        raise SystemExit(f"pyspellchecker resource not found at {src}")

    with gzip.open(src, "rt", encoding="utf-8") as handle:
        freqs: dict[str, int] = json.load(handle)

    ranked = sorted(freqs.items(), key=lambda kv: -kv[1])
    words = [w for w, _ in ranked if w.isalpha()][:size]

    known = set(words)
    words.extend(w for w in PRIORITY_WORDS if w not in known)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(words), encoding="utf-8")
    return len(words)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export web inference assets")
    parser.add_argument("--profile", default="default", help="Profile whose model to export")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=WEB_ASSETS,
        help="Directory to write JSON assets into",
    )
    parser.add_argument("--wordlist-size", type=int, default=WORDLIST_SIZE)
    return parser.parse_args()


def main() -> int:
    import sys

    sys.path.insert(0, str(ROOT))
    from src.profile_paths import model_path_for_profile

    args = parse_args()
    model_path = model_path_for_profile(args.profile)
    if not model_path.exists():
        print(f"No model at {model_path}. Train one first with train_gesture_model.py.")
        return 1

    payload = export_model(model_path, args.out_dir / "gesture_mlp.json")
    shapes = " -> ".join(
        [str(payload["n_features"])]
        + [str(len(layer["bias"])) for layer in payload["layers"]]
    )
    print(f"Model  : {args.out_dir / 'gesture_mlp.json'}")
    print(f"         {shapes}  ({len(payload['labels'])} labels)")
    if not payload["source_metadata"]:
        print("         warning: model has no training metadata (unknown accuracy)")

    count = export_wordlist(args.out_dir / "words.json", args.wordlist_size)
    print(f"Words  : {args.out_dir / 'words.json'} ({count} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
