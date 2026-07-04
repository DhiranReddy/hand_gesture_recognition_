#!/usr/bin/env python3
"""
Hand Gesture Caption App
------------------------
Assistive communication via hand gestures with live captions.
ML ensemble + auto-spacing + spell correction for video calls.
"""

from __future__ import annotations

import argparse
import sys

import cv2

from src.caption_renderer import draw_caption_bar
from src.conference_ui import (
    copy_to_clipboard,
    draw_compact_overlay,
    draw_conference_caption_bar,
    draw_gesture_reference,
    draw_hold_progress,
)
from src.constants import (
    CONFERENCE_HOLD_FRAMES,
    COOLDOWN_FRAMES,
    HOLD_FRAMES,
)
from src.custom_dataset import append_sample, canonical_label
from src.gesture_classifier import MODE_ALL, MODE_SPELL, MODE_WORDS
from src.gesture_ensemble import EnsembleGestureRecognizer, result_to_text
from src.hand_tracker import HandTracker
from src.profile_paths import DEFAULT_PROFILE, dataset_path_for_profile, model_path_for_profile
from src.word_processor import WordProcessor

MODE_CYCLE = [MODE_ALL, MODE_WORDS, MODE_SPELL]
MODE_NAMES = {MODE_ALL: "CUSTOM+ALL", MODE_WORDS: "CUSTOM+WORDS", MODE_SPELL: "CUSTOM+SPELL"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hand gesture captions for video calls")
    parser.add_argument("--conference", action="store_true", help="Conference mode UI")
    parser.add_argument("--caption-only", action="store_true", help="Open caption strip window")
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help="Profile name for personalized model and dataset",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Check camera permissions.")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    profile_model_path = model_path_for_profile(args.profile)
    profile_dataset_path = dataset_path_for_profile(args.profile)

    tracker = HandTracker(max_hands=1)
    recognizer = EnsembleGestureRecognizer(model_path=profile_model_path, dataset_path=profile_dataset_path)
    words = WordProcessor()

    recognition_mode = MODE_ALL
    conference_mode = args.conference
    show_caption_window = args.caption_only
    show_skeleton = not conference_mode
    show_guide = False
    copied_flash = 0

    current_gesture: str | None = None
    hold_count = 0
    cooldown = 0
    last_added = ""
    ml_confidence = 0.0
    feedback_flash = 0
    relabel_mode = False
    relabel_text = ""
    relabel_landmarks: list[tuple] | None = None
    relabel_handedness = "Right"

    cv2.namedWindow("Hand Gesture Captions", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand Gesture Captions", 960, 640)
    if show_caption_window:
        cv2.namedWindow("Captions for Meet/Zoom", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Captions for Meet/Zoom", 960, 160)

    print("Hand Gesture Caption App — custom ML mode enabled.")
    print(f"Profile: {args.profile}")
    print(f"Model: {profile_model_path}")
    print(f"Feedback dataset: {profile_dataset_path}")
    print("Auto-spacing + spell correction active.")
    print("Keys: M=mode  T=caption window  V=copy  H=hide mesh  G=guide  K=save sample  R=relabel sample  Q=quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)
            hands = tracker.process(frame, draw_skeleton=show_skeleton)

            detected_id: str | None = None
            category = "none"
            landmarks = hands[0]["landmarks"] if hands else None
            handedness = hands[0]["label"] if hands else "Right"

            detected_id, category, ml_confidence = recognizer.recognize(
                landmarks,
                handedness,
                frame,
                mode=recognition_mode,
            )

            detected_text = result_to_text(detected_id)
            hold_required = CONFERENCE_HOLD_FRAMES if conference_mode else HOLD_FRAMES

            if cooldown > 0:
                cooldown -= 1
            if copied_flash > 0:
                copied_flash -= 1
            if feedback_flash > 0:
                feedback_flash -= 1

            if not detected_id:
                if words.on_idle_frame():
                    last_added = ""

            if detected_id and detected_id == current_gesture:
                hold_count += 1
                words.set_preview(detected_text or "")
            else:
                current_gesture = detected_id
                hold_count = 0
                if detected_text:
                    words.set_preview(detected_text)
                else:
                    words.clear_preview()

            progress = hold_count / hold_required if detected_id else 0.0
            if hold_count >= hold_required and detected_text and cooldown == 0:
                if detected_text != last_added:
                    words.add_gesture(detected_text, category)
                    last_added = detected_text
                    cooldown = COOLDOWN_FRAMES
                    hold_count = 0
                    words.clear_preview()

            hint = detected_text or ""
            if ml_confidence > 0:
                hint = f"{hint} ({ml_confidence:.0%})" if hint else ""

            draw_compact_overlay(
                frame,
                show_skeleton=show_skeleton,
                mode_label=MODE_NAMES[recognition_mode],
                conference_mode=conference_mode,
            )
            if show_guide:
                draw_gesture_reference(frame)
            if detected_text and hold_count < hold_required:
                draw_hold_progress(frame, progress, f"Hold: {detected_text}")

            if feedback_flash > 0:
                cv2.putText(
                    frame,
                    "Feedback sample saved",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (100, 255, 150),
                    2,
                    cv2.LINE_AA,
                )

            if relabel_mode:
                cv2.rectangle(frame, (12, frame.shape[0] - 95), (540, frame.shape[0] - 10), (18, 18, 18), -1)
                cv2.rectangle(frame, (12, frame.shape[0] - 95), (540, frame.shape[0] - 10), (90, 180, 220), 1)
                cv2.putText(
                    frame,
                    "Correction mode: type label and press Enter",
                    (22, frame.shape[0] - 66),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (220, 235, 245),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Label: {relabel_text or '_'}",
                    (22, frame.shape[0] - 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.64,
                    (110, 255, 220),
                    2,
                    cv2.LINE_AA,
                )

            output = draw_caption_bar(
                frame,
                words.display_transcript,
                preview=words.preview,
                gesture_hint=hint,
                mode_label=MODE_NAMES[recognition_mode],
                buffer=words.buffer,
                correction=words.last_correction,
            )
            cv2.imshow("Hand Gesture Captions", output)

            if show_caption_window:
                caption_strip = draw_conference_caption_bar(
                    words.display_transcript,
                    width=960,
                    preview=words.preview,
                    mode_label=MODE_NAMES[recognition_mode],
                    copied_flash=copied_flash,
                )
                cv2.imshow("Captions for Meet/Zoom", caption_strip)

            key = cv2.waitKey(1) & 0xFF

            if relabel_mode:
                if key in (13, 10):
                    if relabel_text and relabel_landmarks:
                        append_sample(
                            profile_dataset_path,
                            label=canonical_label(relabel_text),
                            handedness=relabel_handedness,
                            landmarks=[
                                (float(x), float(y), float(z)) for x, y, z in relabel_landmarks
                            ],
                        )
                        feedback_flash = 50
                        print(f"Saved correction sample: {canonical_label(relabel_text)}")
                    relabel_mode = False
                    relabel_text = ""
                    relabel_landmarks = None
                    continue
                if key == 27:
                    relabel_mode = False
                    relabel_text = ""
                    relabel_landmarks = None
                    continue
                if key in (8, 127):
                    relabel_text = relabel_text[:-1]
                    continue
                if 32 <= key <= 126:
                    ch = chr(key)
                    if ch.isalnum() or ch == "_" or ch == " ":
                        relabel_text += ch
                    continue

            if key == ord("q"):
                break
            if key == ord(" "):
                words.add_space()
            if key in (8, 127):
                words.backspace()
            if key == ord("c"):
                words.clear()
                last_added = ""
            if key == ord("v"):
                if copy_to_clipboard(words.display_transcript):
                    copied_flash = 45
                    print("Transcript copied to clipboard.")
            if key == ord("m"):
                idx = MODE_CYCLE.index(recognition_mode)
                recognition_mode = MODE_CYCLE[(idx + 1) % len(MODE_CYCLE)]
                print(f"Recognition mode: {MODE_NAMES[recognition_mode]}")
            if key == ord("t"):
                show_caption_window = not show_caption_window
                if show_caption_window:
                    cv2.namedWindow("Captions for Meet/Zoom", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Captions for Meet/Zoom", 960, 160)
                else:
                    cv2.destroyWindow("Captions for Meet/Zoom")
            if key == ord("h"):
                show_skeleton = not show_skeleton
            if key == ord("g"):
                show_guide = not show_guide
            if key == ord("f"):
                conference_mode = not conference_mode
                if conference_mode:
                    show_skeleton = False
            if key == ord("k") and landmarks and detected_id:
                append_sample(
                    profile_dataset_path,
                    label=detected_id,
                    handedness=handedness,
                    landmarks=[(float(x), float(y), float(z)) for x, y, z in landmarks],
                )
                feedback_flash = 50
                print(f"Saved feedback sample from prediction: {detected_id}")
            if key == ord("r") and landmarks:
                relabel_mode = True
                relabel_text = detected_id or ""
                relabel_handedness = handedness
                relabel_landmarks = [(float(x), float(y), float(z)) for x, y, z in landmarks]
                print("Correction mode enabled. Type label and press Enter.")

    finally:
        recognizer.close()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()

    if words.transcript.strip():
        print(f"\nFinal transcript: {words.transcript.strip()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
