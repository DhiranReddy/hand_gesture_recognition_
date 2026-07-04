#!/usr/bin/env python3
"""Guided profile calibration by capturing multiple gesture labels in one session."""

from __future__ import annotations

import argparse

import cv2

from src.custom_dataset import append_sample, canonical_label
from src.hand_tracker import HandTracker
from src.profile_paths import DEFAULT_PROFILE, dataset_path_for_profile

DEFAULT_LABELS = [
    "HELLO",
    "THANK_YOU",
    "HELP",
    "YES",
    "NO",
    "PLEASE",
    "YOU",
    "A",
    "E",
    "I",
    "O",
    "U",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guided calibration for a profile")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="Profile name")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=DEFAULT_LABELS,
        help="Gesture labels to calibrate in order",
    )
    parser.add_argument(
        "--per-label",
        type=int,
        default=90,
        help="Samples to capture per label",
    )
    parser.add_argument(
        "--auto-every",
        type=int,
        default=3,
        help="Capture every N frames in auto mode",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = [canonical_label(label) for label in args.labels if label.strip()]
    if not labels:
        print("No labels provided.")
        return 1

    dataset_path = dataset_path_for_profile(args.profile)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Check camera permissions.")
        return 1

    tracker = HandTracker(max_hands=1)
    auto_mode = True
    frame_count = 0

    current_index = 0
    counts = {label: 0 for label in labels}

    print(f"Profile calibration: {args.profile}")
    print(f"Dataset: {dataset_path}")
    print(f"Labels: {', '.join(labels)}")
    print("Keys: C=capture once  A=toggle auto  N=next label  S=skip label  Q=quit")

    try:
        while current_index < len(labels):
            label = labels[current_index]

            ok, frame = cap.read()
            if not ok:
                print("Error: Failed to read camera frame.")
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
                    dataset_path,
                    label=label,
                    handedness=hand["label"],
                    landmarks=hand["landmarks"],
                )
                counts[label] += 1

            if counts[label] >= args.per_label:
                current_index += 1
                continue

            panel_text = [
                f"Profile: {args.profile}",
                f"Label {current_index + 1}/{len(labels)}: {label}",
                f"Progress: {counts[label]}/{args.per_label}",
                f"Auto: {'ON' if auto_mode else 'OFF'}",
                "C capture | A auto | N next | S skip | Q quit",
            ]
            cv2.rectangle(frame, (10, 10), (650, 130), (18, 18, 18), -1)
            y = 36
            for line in panel_text:
                cv2.putText(
                    frame,
                    line,
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62 if y <= 54 else 0.5,
                    (220, 240, 255),
                    2 if y <= 54 else 1,
                    cv2.LINE_AA,
                )
                y += 22

            cv2.imshow("Profile Calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("a"):
                auto_mode = not auto_mode
            if key == ord("n"):
                current_index += 1
            if key == ord("s"):
                current_index += 1
            if key == ord("c") and sample_ready:
                hand = hands[0]
                append_sample(
                    dataset_path,
                    label=label,
                    handedness=hand["label"],
                    landmarks=hand["landmarks"],
                )
                counts[label] += 1

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()

    print("Calibration summary:")
    for label in labels:
        print(f"- {label}: {counts.get(label, 0)} samples")
    print(f"Saved to {dataset_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
