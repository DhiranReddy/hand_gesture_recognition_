"""Caption bar rendering and transcript management."""

from __future__ import annotations

import cv2
import numpy as np

from src.constants import (
    CAPTION_BAR_HEIGHT,
    COLOR_CYAN,
    COLOR_DARK_BG,
    COLOR_PREVIEW,
    COLOR_WHITE,
)


class CaptionManager:
    """Builds and stores the spoken transcript from recognized gestures."""

    def __init__(self) -> None:
        self.transcript = ""
        self.preview = ""

    def add_word(self, word: str) -> None:
        if not word:
            return
        if self.transcript and not self.transcript.endswith(" "):
            self.transcript += " "
        self.transcript += word

    def add_letter(self, letter: str) -> None:
        self.transcript += letter

    def set_preview(self, text: str) -> None:
        self.preview = text

    def clear_preview(self) -> None:
        self.preview = ""

    def backspace(self) -> None:
        self.transcript = self.transcript[:-1]

    def clear(self) -> None:
        self.transcript = ""
        self.preview = ""

    def add_space(self) -> None:
        if self.transcript and not self.transcript.endswith(" "):
            self.transcript += " "


def _wrap_text(text: str, max_chars: int = 60) -> list[str]:
    if not text:
        return [""]
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[-2:] if len(lines) > 2 else lines


def draw_caption_bar(
    frame: np.ndarray,
    transcript: str,
    preview: str = "",
    gesture_hint: str = "",
    mode_label: str = "ALL",
    buffer: str = "",
    correction: str = "",
) -> np.ndarray:
    """Draw a caption panel below the camera feed."""
    h, w = frame.shape[:2]
    bar = np.full((CAPTION_BAR_HEIGHT, w, 3), COLOR_DARK_BG, dtype=np.uint8)

    cv2.rectangle(bar, (0, 0), (w - 1, CAPTION_BAR_HEIGHT - 1), (60, 60, 60), 1)

    cv2.putText(
        bar,
        f"LIVE CAPTIONS  |  {mode_label}  |  auto-space + autocorrect",
        (16, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        COLOR_CYAN,
        1,
        cv2.LINE_AA,
    )

    lines = _wrap_text(transcript, max_chars=int(w / 12))
    y = 52
    for line in lines:
        cv2.putText(
            bar,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            COLOR_WHITE,
            2,
            cv2.LINE_AA,
        )
        y += 32

    if buffer:
        cv2.putText(
            bar,
            f"spelling: {buffer}_",
            (16, CAPTION_BAR_HEIGHT - 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_PREVIEW,
            1,
            cv2.LINE_AA,
        )

    if correction:
        cv2.putText(
            bar,
            f"corrected: {correction}",
            (w // 2, CAPTION_BAR_HEIGHT - 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_CYAN,
            1,
            cv2.LINE_AA,
        )

    if preview:
        preview_text = f"Hold to add: {preview}"
        cv2.putText(
            bar,
            preview_text,
            (16, CAPTION_BAR_HEIGHT - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_PREVIEW,
            1,
            cv2.LINE_AA,
        )

    if gesture_hint:
        cv2.putText(
            bar,
            gesture_hint,
            (w - 220, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_CYAN,
            1,
            cv2.LINE_AA,
        )

    return np.vstack([frame, bar])
