#!/usr/bin/env python3
"""
Report whether a trained gesture model is usable.

A classifier that collapses onto one label looks fine by every smoke test but
is useless in the app, so measure it directly: probe with many perturbed hands
and report the prediction spread.

    python tools/diagnose_model.py [--profile NAME]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.landmark_features import landmarks_to_features  # noqa: E402
from src.profile_paths import model_path_for_profile  # noqa: E402
from tests.conftest import OPEN_PALM  # noqa: E402

N_PROBES = 3000
# Only flag severe collapse. The probes are perturbed open palms, not a uniform
# sample of hand shapes, so a healthy model legitimately never emits some labels
# for them — that alone is not evidence of a problem.
COLLAPSE_SHARE = 0.40
MIN_TEST_ACCURACY = 0.70


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose a gesture model")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--probes", type=int, default=N_PROBES)
    args = parser.parse_args()

    path = model_path_for_profile(args.profile)
    if not path.exists():
        print(f"No model at {path}.")
        return 1

    bundle = joblib.load(path)
    clf, encoder = bundle["model"], bundle["encoder"]
    rng = np.random.default_rng(0)

    base = np.array(OPEN_PALM, dtype=float)
    rows = []
    for _ in range(args.probes):
        hand = base.copy()
        hand[:, :2] += rng.normal(0, 55, size=(21, 2))
        hand[:, 0] += rng.normal(0, 120)
        hand[:, 1] += rng.normal(0, 90)
        rows.append(landmarks_to_features([tuple(p) for p in hand], "Right"))

    probs = clf.predict_proba(np.array(rows, dtype=np.float32))
    counts = Counter(encoder.classes_[i] for i in probs.argmax(1))
    top_label, top_count = counts.most_common(1)[0]
    share = top_count / len(probs)
    ordered = np.sort(probs, axis=1)
    never = [c for c in encoder.classes_ if c not in counts]

    metadata = bundle.get("metadata") or {}
    accuracy = metadata.get("test_accuracy")

    print(f"model                  : {path}")
    print(f"labels                 : {len(encoder.classes_)}")
    if accuracy is not None:
        print(f"held-out accuracy      : {accuracy:.1%}  "
              f"({metadata.get('sample_count', '?')} samples)")
    else:
        print("held-out accuracy      : unknown (model has no training metadata)")

    print(f"\nsynthetic probes       : {len(probs)}")
    print(f"distinct predictions   : {len(counts)}")
    print(f"most common label      : {top_label} ({share:.1%} of probes)")
    print(f"mean top probability   : {ordered[:, -1].mean():.3f}")
    if never:
        print(f"unseen on probes ({len(never)}) : {' '.join(map(str, never))}")
        print("  (probes are perturbed open palms, so this is weak evidence)")

    collapsed = share > COLLAPSE_SHARE
    if accuracy is None:
        healthy = not collapsed
        verdict = ("NO ACCURACY DATA — probes look reasonable" if healthy
                   else "MODEL IS DEGENERATE — do not rely on it")
    else:
        healthy = accuracy >= MIN_TEST_ACCURACY and not collapsed
        verdict = ("MODEL LOOKS USABLE" if healthy
                   else "MODEL IS DEGENERATE — do not rely on it")

    if collapsed:
        print(f"\ncollapse: {share:.0%} of probes map to one label "
              f"(limit {COLLAPSE_SHARE:.0%})")

    print(f"\n{verdict}")
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
