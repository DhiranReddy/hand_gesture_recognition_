#!/usr/bin/env python3
"""Capture custom hand-gesture landmark samples into a JSONL dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.custom_dataset import DATASET_PATH, append_sample, canonical_label
from src.hand_tracker import HandTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture custom gesture landmarks")
    parser.add_argument("--label", required=True, help="Gesture label, e.g. HELLO or NEED_HELP")
    parser.add_argument(
        "--output",
        type=Path,
        default=DATASET_PATH,
        help="Output dataset JSONL path",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=200,
        help="Target number of samples to collect",
    )
    parser.add_argument(
        "--auto-every",
        type=int,
        default=3,
        help="Capture every N frames while auto mode is enabled",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    label = canonical_label(args.label)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Check camera permissions.")
        return 1

    tracker = HandTracker(max_hands=1)
    count = 0
    frame_count = 0
    auto_mode = False

    print(f"Capturing label: {label}")
    print(f"Target samples: {args.target}")
    print("Keys: C=capture once  A=toggle auto capture  Q=quit")

    try:
        while count < args.target:
            ok, frame = cap.read()
            if not ok:
                print("Error: Could not read camera frame.")
                break

            frame = cv2.flip(frame, 1)
            hands = tracker.process(frame, draw_skeleton=True)
            sample_ready = bool(hands)

            should_capture = False
            if auto_mode and sample_ready:
                frame_count += 1
                if frame_count % max(args.auto_every, 1) == 0:
                    should_capture = True

            if should_capture:
                hand = hands[0]
                append_sample(
                    args.output,
                    label=label,
                    handedness=hand["label"],
                    landmarks=hand["landmarks"],
                )
                count += 1

            status = f"Label: {label}  Samples: {count}/{args.target}  Auto: {'ON' if auto_mode else 'OFF'}"
            cv2.rectangle(frame, (10, 10), (940, 70), (20, 20, 20), -1)
            cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 240, 255), 2)
            cv2.putText(
                frame,
                "C capture | A auto | Q quit",
                (20, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 210, 225),
                1,
            )
            cv2.imshow("Capture Gesture Dataset", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("a"):
                auto_mode = not auto_mode
            if key == ord("c") and sample_ready:
                hand = hands[0]
                append_sample(
                    args.output,
                    label=label,
                    handedness=hand["label"],
                    landmarks=hand["landmarks"],
                )
                count += 1

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()

    print(f"Saved {count} samples for {label} into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
