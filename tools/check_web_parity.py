#!/usr/bin/env python3
"""
Verify the browser pipeline reproduces the Python one.

Generates random hands, runs them through the Python feature extractor and
sklearn model, then runs the same inputs through the JS modules under node and
compares. A silent divergence here would make the exported weights meaningless,
so this is the gate before any web deploy.

    python tools/check_web_parity.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.landmark_features import landmarks_to_features  # noqa: E402
from src.model_io import load_model  # noqa: E402
from src.ml_gesture_model import MLGestureClassifier  # noqa: E402

WEB = ROOT / "web"
N_CASES = 200
FRAME_W, FRAME_H = 960, 540
FEATURE_TOL = 1e-4
PROB_TOL = 1e-4

DRIVER = """
import { landmarksToFeatures } from "%(features)s";
import { GestureClassifier } from "%(classifier)s";
import { readFileSync } from "node:fs";

const cases = JSON.parse(readFileSync(process.argv[2], "utf8"));
const spec = JSON.parse(readFileSync(process.argv[3], "utf8"));
const clf = new GestureClassifier(spec);

const out = cases.map((c) => {
  const lm = c.landmarks.map(([x, y, z]) => ({ x, y, z }));
  const features = Array.from(
    landmarksToFeatures(lm, c.handedness, c.width, c.height),
  );
  const { label, confidence } = clf.predict(lm, c.handedness, c.width, c.height);
  return { features, label, confidence };
});
console.log(JSON.stringify(out));
"""


def build_cases(rng: np.random.Generator) -> list[dict]:
    """
    Random hands in MediaPipe's normalized 0-1 space.

    This is what the JS extractor consumes; the Python side takes the same hands
    already scaled to pixels, matching how each tracker hands data over.
    """
    cases = []
    for i in range(N_CASES):
        base = rng.uniform([0.2, 0.2, -0.1], [0.8, 0.8, 0.1], size=(21, 3))
        cases.append(
            {
                "landmarks": base.tolist(),
                "handedness": "Left" if i % 2 else "Right",
                "width": FRAME_W,
                "height": FRAME_H,
            }
        )
    return cases


def to_pixel_space(case: dict) -> list[tuple[float, float, float]]:
    """Mirror what HandTracker.process emits: x/y in pixels, z left normalized."""
    return [
        (x * case["width"], y * case["height"], z) for x, y, z in case["landmarks"]
    ]


def main() -> int:
    if shutil.which("node") is None:
        print("node not found — install Node.js to run the parity check.")
        return 2

    model_json = WEB / "public" / "model" / "gesture_mlp.json"
    if not model_json.exists():
        print(f"Missing {model_json}. Run tools/export_web_assets.py first.")
        return 1

    rng = np.random.default_rng(7)
    cases = build_cases(rng)

    clf, _ = load_model(ROOT / "models" / "gesture_mlp.json")

    py_features = []
    py_labels: list[str | None] = []
    py_conf: list[float] = []
    for case in cases:
        feats = landmarks_to_features(to_pixel_space(case), case["handedness"])
        py_features.append(feats)

        probs = clf.predict_proba(feats.reshape(1, -1))[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        ordered = np.sort(probs)
        margin = float(ordered[-1] - ordered[-2])
        gated = (
            conf >= MLGestureClassifier.MIN_CONFIDENCE
            and margin >= MLGestureClassifier.MIN_MARGIN
        )
        py_labels.append(str(clf.classes_[idx]) if gated else None)
        py_conf.append(conf)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cases_file = tmp / "cases.json"
        cases_file.write_text(json.dumps(cases), encoding="utf-8")

        driver = tmp / "driver.mjs"
        driver.write_text(
            DRIVER
            % {
                "features": (WEB / "src" / "features.js").as_uri(),
                "classifier": (WEB / "src" / "classifier.js").as_uri(),
            },
            encoding="utf-8",
        )

        proc = subprocess.run(
            ["node", str(driver), str(cases_file), str(model_json)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print("node driver failed:\n" + proc.stderr)
            return 1
        js_results = json.loads(proc.stdout)

    max_feature_delta = 0.0
    max_prob_delta = 0.0
    label_mismatches = 0

    for i, (py_feat, js) in enumerate(zip(py_features, js_results)):
        delta = float(np.max(np.abs(py_feat - np.array(js["features"], dtype=np.float32))))
        max_feature_delta = max(max_feature_delta, delta)
        max_prob_delta = max(max_prob_delta, abs(py_conf[i] - js["confidence"]))
        if py_labels[i] != js["label"]:
            label_mismatches += 1
            if label_mismatches <= 5:
                print(f"  case {i}: python={py_labels[i]!r} js={js['label']!r}")

    print(f"cases                 : {len(cases)}")
    print(f"max feature delta     : {max_feature_delta:.3e}  (tol {FEATURE_TOL:.0e})")
    print(f"max confidence delta  : {max_prob_delta:.3e}  (tol {PROB_TOL:.0e})")
    print(f"label mismatches      : {label_mismatches}")

    ok = (
        max_feature_delta <= FEATURE_TOL
        and max_prob_delta <= PROB_TOL
        and label_mismatches == 0
    )
    print("\nPARITY OK" if ok else "\nPARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
