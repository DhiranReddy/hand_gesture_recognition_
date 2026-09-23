#!/usr/bin/env python3
"""
Verify the JS rule classifiers match the Python ones.

Covers finger-state geometry plus classify_letter and classify_number. The
phrase rules are deliberately NOT compared: the browser ports only the subset
the model also knows, because the Python table contains unreachable duplicate
branches that are not worth carrying over.

    python tools/check_rules_parity.py
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

from src.asl_alphabet import classify_letter  # noqa: E402
from src.asl_numbers import classify_number  # noqa: E402
from src.finger_geometry import get_detailed_finger_states  # noqa: E402
from tests.conftest import OPEN_PALM  # noqa: E402

WEB = ROOT / "web" / "src"
N_CASES = 600

DRIVER = """
import { fingerStates } from "%(geometry)s";
import { classifyLetter, classifyNumber } from "%(rules)s";
import { readFileSync } from "node:fs";

const cases = JSON.parse(readFileSync(process.argv[2], "utf8"));
const out = cases.map((c) => {
  const lm = c.landmarks.map(([x, y]) => ({ x, y }));
  return {
    states: fingerStates(lm, c.handedness),
    letter: classifyLetter(lm, c.handedness),
    number: classifyNumber(lm, c.handedness),
  };
});
console.log(JSON.stringify(out));
"""


def build_cases(rng: np.random.Generator) -> list[dict]:
    """
    Plausible hands, not uniform noise.

    Random points almost never satisfy the rules, so nothing would be compared.
    Perturbing a real hand — whole-finger curls plus joint jitter — exercises the
    branches that actually fire in use.
    """
    base = np.array(OPEN_PALM, dtype=float)[:, :2]
    groups = [(1, 4), (5, 8), (9, 12), (13, 16), (17, 20)]
    cases: list[dict] = []

    for i in range(N_CASES):
        hand = base.copy()
        for start, end in groups:
            roll = rng.random()
            if roll < 0.4:  # curl this finger toward the palm
                pip = hand[start + 1]
                for j in range(start, end + 1):
                    pull = (j - start + 1) / 4.0
                    hand[j] = hand[j] * (1 - pull) + (pip + [0, 26]) * pull
            elif roll < 0.55:  # half-bend
                for j in range(start + 2, end + 1):
                    hand[j] += [0, 16]
        hand += rng.normal(0, 7, size=hand.shape)
        hand[:, 0] += rng.normal(0, 40)
        hand[:, 1] += rng.normal(0, 30)

        cases.append(
            {
                "landmarks": hand.tolist(),
                "handedness": "Left" if i % 3 == 0 else "Right",
            }
        )
    return cases


def main() -> int:
    if shutil.which("node") is None:
        print("node not found — install Node.js to run the parity check.")
        return 2

    rng = np.random.default_rng(11)
    cases = build_cases(rng)

    py = []
    for case in cases:
        lm = [(x, y, 0.0) for x, y in case["landmarks"]]
        py.append(
            {
                "states": get_detailed_finger_states(lm, case["handedness"]),
                "letter": classify_letter(lm, case["handedness"]),
                "number": classify_number(lm, case["handedness"]),
            }
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cases_file = tmp / "cases.json"
        cases_file.write_text(json.dumps(cases), encoding="utf-8")

        driver = tmp / "driver.mjs"
        driver.write_text(
            DRIVER
            % {
                "geometry": (WEB / "geometry.js").as_uri(),
                "rules": (WEB / "rules.js").as_uri(),
            },
            encoding="utf-8",
        )

        proc = subprocess.run(
            ["node", str(driver), str(cases_file)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print("node driver failed:\n" + proc.stderr)
            return 1
        js = json.loads(proc.stdout)

    mismatches = {"states": 0, "letter": 0, "number": 0}
    shown = 0
    for i, (a, b) in enumerate(zip(py, js)):
        for field in mismatches:
            if a[field] != b[field]:
                mismatches[field] += 1
                if shown < 6:
                    print(f"  case {i} {field}: python={a[field]!r} js={b[field]!r}")
                    shown += 1

    fired = sum(1 for r in py if r["letter"] or r["number"])
    print(f"cases                : {len(cases)}")
    print(f"cases where a rule fired : {fired}")
    for field, count in mismatches.items():
        print(f"{field + ' mismatches':<24} : {count}")

    ok = not any(mismatches.values())
    print("\nRULES PARITY OK" if ok else "\nRULES PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
