#!/usr/bin/env python3
"""
Convert a legacy .joblib model to the pickle-free JSON format.

Loading a .joblib executes whatever is pickled inside it, so only run this on a
file you produced yourself.

    python tools/migrate_model.py models/gesture_mlp.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model_io import save_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert .joblib to .json")
    parser.add_argument("path", type=Path, help="Legacy .joblib model")
    parser.add_argument("--out", type=Path, help="Destination (default: same name, .json)")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the prompt confirming you trust this file",
    )
    args = parser.parse_args()

    if not args.path.exists():
        print(f"Not found: {args.path}")
        return 1

    if not args.yes:
        print(f"{args.path} is a pickle; loading it runs code it contains.")
        if input("Only continue if you created it. Continue? [y/N] ").strip().lower() != "y":
            return 1

    import joblib  # imported late so the unsafe path is explicit

    bundle = joblib.load(args.path)
    model, encoder = bundle.get("model"), bundle.get("encoder")
    if model is None or encoder is None:
        print("Not a recognised model bundle.")
        return 1

    metadata = dict(bundle.get("metadata") or {})
    metadata["migrated_from"] = str(args.path)

    out = save_model(
        args.out or args.path.with_suffix(".json"),
        model,
        [str(c) for c in encoder.classes_],
        metadata,
    )
    print(f"Wrote {out}")
    print("Delete the .joblib once you have verified the JSON works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
