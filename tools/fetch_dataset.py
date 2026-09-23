#!/usr/bin/env python3
"""
Build a landmark training set from a public ASL image dataset.

Downloads an image dataset from Hugging Face, runs MediaPipe over every image,
and writes the 21 landmarks per detected hand to the project's JSONL format —
the same shape capture_gesture_dataset.py produces, so train_gesture_model.py
consumes it unchanged.

    python tools/fetch_dataset.py                   # full run, ~5 min
    python tools/fetch_dataset.py --limit 500       # quick smoke test
    python tools/fetch_dataset.py --append          # add to an existing file

Licensing: the default dataset declares no license on its dataset card. It is
fine for local experiments, but confirm terms before shipping a model trained on
it. Pass --dataset to use a source you have cleared.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_REPO = "Marxulia/asl_sign_languages_alphabets_v03"
DEFAULT_FILE = "data/train-00000-of-00001.parquet"
CACHE_DIR = Path(
    os.environ.get("TMPDIR", "/tmp")
) / "signal-datasets"

# Quiet MediaPipe's C++ logging before it is imported.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def download(repo: str, filename: str, force: bool = False) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = CACHE_DIR / f"{repo.replace('/', '__')}__{Path(filename).name}"
    if target.exists() and not force:
        print(f"cached   : {target} ({target.stat().st_size / 1e6:.0f} MB)")
        return target

    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"
    print(f"fetching : {url}")
    with urllib.request.urlopen(url, timeout=300) as response:
        total = int(response.headers.get("content-length", 0))
        done = 0
        with target.open("wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
                done += len(chunk)
                if total:
                    pct = done / total * 100
                    print(f"\r           {pct:5.1f}%  {done/1e6:.0f}/{total/1e6:.0f} MB",
                          end="", flush=True)
        print()
    return target


def label_names(table) -> list[str] | None:
    """Recover class names from the parquet's HF metadata, if present."""
    meta = (table.schema.metadata or {}).get(b"huggingface")
    if not meta:
        return None
    try:
        info = json.loads(meta.decode("utf-8"))
        feature = info["info"]["features"]["label"]
        return list(feature["names"])
    except (KeyError, ValueError, TypeError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a landmark dataset from images")
    parser.add_argument("--dataset", default=DEFAULT_REPO, help="Hugging Face dataset id")
    parser.add_argument("--file", default=DEFAULT_FILE, help="Parquet path inside the repo")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "gesture_landmarks.jsonl")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process an evenly spaced sample of this many images (for smoke tests)",
    )
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--append", action="store_true", help="Keep existing samples")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("pyarrow is required: pip install pyarrow")
        return 1

    import cv2  # noqa: F401  (imported for its side effect on some builds)
    import mediapipe as mp
    import numpy as np
    from PIL import Image

    path = download(args.dataset, args.file, args.force_download)
    table = pq.read_table(path)
    if "image" not in table.column_names or "label" not in table.column_names:
        print(f"Unexpected columns: {table.column_names}")
        return 1

    names = label_names(table) or [str(i) for i in range(26)]
    images = table.column("image")
    labels = table.column("label")
    # Rows are grouped by label, so a prefix would only cover the first class.
    # An evenly spaced stride keeps a --limit run representative.
    if args.limit and args.limit < table.num_rows:
        step = table.num_rows / args.limit
        order = [int(i * step) for i in range(args.limit)]
    else:
        order = list(range(table.num_rows))
    total = len(order)
    print(f"images   : {total}  classes: {len(names)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"

    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=args.min_confidence,
    )

    kept: Counter[str] = Counter()
    missed: Counter[str] = Counter()

    with args.out.open(mode, encoding="utf-8") as handle:
        for seen, i in enumerate(order, start=1):
            label = names[labels[i].as_py()]
            record = images[i].as_py()
            try:
                image = Image.open(io.BytesIO(record["bytes"])).convert("RGB")
            except Exception:
                missed[label] += 1
                continue

            frame = np.array(image)
            height, width = frame.shape[:2]
            result = hands.process(frame)

            if not result.multi_hand_landmarks:
                missed[label] += 1
            else:
                hand = result.multi_hand_landmarks[0]
                handedness = "Right"
                if result.multi_handedness:
                    handedness = result.multi_handedness[0].classification[0].label
                # Pixel x/y with MediaPipe's normalized z, matching HandTracker.
                points = [
                    [lm.x * width, lm.y * height, lm.z] for lm in hand.landmark
                ]
                handle.write(
                    json.dumps(
                        {"label": label, "handedness": handedness, "landmarks": points}
                    )
                    + "\n"
                )
                kept[label] += 1

            if seen % 250 == 0 or seen == total:
                done = sum(kept.values())
                print(f"\r  {seen}/{total}  kept {done}  "
                      f"({done/seen:.0%} detected)", end="", flush=True)
    print()
    hands.close()

    print(f"\nwrote    : {args.out}")
    print(f"samples  : {sum(kept.values())} of {total} "
          f"({sum(kept.values())/total:.1%} detected)")

    thin = [(n, kept[n]) for n in names if kept[n] < 60]
    if thin:
        print("thin classes (<60): " + ", ".join(f"{n}={c}" for n, c in thin))
    print("per-class: " + " ".join(f"{n}:{kept[n]}" for n in names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
