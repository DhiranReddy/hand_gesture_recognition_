"""Conferencing-friendly UI for Zoom, Meet, and Teams."""

from __future__ import annotations

import subprocess
import sys

import cv2
import numpy as np

from src.constants import (
    CAPTION_BAR_HEIGHT,
    COLOR_BLACK,
    COLOR_CYAN,
    COLOR_DARK_BG,
    COLOR_GREEN,
    COLOR_PREVIEW,
    COLOR_WHITE,
    CONFERENCE_CAPTION_HEIGHT,
)


def copy_to_clipboard(text: str) -> bool:
    """Copy transcript to system clipboard."""
    if not text.strip():
        return False
    try:
        if sys.platform == "darwin":
            proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
            proc.communicate(text.encode("utf-8"))
            return proc.returncode == 0
        if sys.platform.startswith("linux"):
            proc = subprocess.Popen(
                ["xclip", "-selection", "clipboard"],
                stdin=subprocess.PIPE,
            )
            proc.communicate(text.encode("utf-8"))
            return proc.returncode == 0
        if sys.platform == "win32":
            proc = subprocess.Popen(
                ["clip"],
                stdin=subprocess.PIPE,
                shell=True,
            )
            proc.communicate(text.encode("utf-16le"))
            return proc.returncode == 0
    except (FileNotFoundError, OSError):
        return False
    return False


def _wrap_text(text: str, max_chars: int) -> list[str]:
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
    return lines[-3:] if len(lines) > 3 else lines


def draw_conference_caption_bar(
    transcript: str,
    width: int = 960,
    preview: str = "",
    mode_label: str = "ALL",
    copied_flash: int = 0,
) -> np.ndarray:
    """Large caption strip designed for screen-share / overlay positioning."""
    h = CONFERENCE_CAPTION_HEIGHT
    bar = np.full((h, width, 3), COLOR_DARK_BG, dtype=np.uint8)
    cv2.rectangle(bar, (0, 0), (width - 1, h - 1), (70, 70, 70), 2)

    cv2.putText(
        bar,
        "LIVE CAPTIONS",
        (16, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        COLOR_CYAN,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        bar,
        f"Mode: {mode_label}",
        (width - 160, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_GREEN,
        1,
        cv2.LINE_AA,
    )

    lines = _wrap_text(transcript, max_chars=int(width / 11))
    y = 58
    for line in lines:
        cv2.putText(
            bar,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            COLOR_WHITE,
            2,
            cv2.LINE_AA,
        )
        y += 34

    if preview:
        cv2.putText(
            bar,
            f"Hold: {preview}",
            (20, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_PREVIEW,
            1,
            cv2.LINE_AA,
        )

    if copied_flash > 0:
        cv2.putText(
            bar,
            "Copied to clipboard!",
            (width // 2 - 120, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_GREEN,
            1,
            cv2.LINE_AA,
        )

    return bar


def draw_compact_overlay(
    frame: np.ndarray,
    *,
    show_skeleton: bool,
    mode_label: str,
    conference_mode: bool,
) -> None:
    """Minimal on-video hints for calls — keeps face/hand area clean."""
    h, w = frame.shape[:2]

    if conference_mode:
        cv2.rectangle(frame, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"Conference Mode | {mode_label} | T=captions  C=copy  H=hide mesh  M=mode",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            COLOR_GREEN,
            1,
            cv2.LINE_AA,
        )
    else:
        hints = [
            f"Mode: {mode_label} (M to switch)",
            "Hold gesture | SPACE=space | C=clear | V=copy | T=caption window",
        ]
        y = 22
        for hint in hints:
            cv2.putText(frame, hint, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_GREEN, 1, cv2.LINE_AA)
            y += 18


def draw_hold_progress(frame: np.ndarray, progress: float, label: str) -> None:
    if progress <= 0:
        return
    h, w = frame.shape[:2]
    bar_w = 220
    bar_h = 14
    x = w - bar_w - 16
    y = 48 if h > 400 else 40
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), COLOR_BLACK, -1)
    fill = int(bar_w * min(progress, 1.0))
    cv2.rectangle(frame, (x, y), (x + fill, y + bar_h), COLOR_GREEN, -1)
    cv2.putText(frame, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WHITE, 1, cv2.LINE_AA)


def draw_gesture_reference(frame: np.ndarray) -> None:
    """Quick reference overlay for letters and common words."""
    h, w = frame.shape[:2]
    panel_w = 280
    panel = np.full((h, panel_w, 3), (20, 20, 20), dtype=np.uint8)
    lines = [
        "GESTURE GUIDE",
        "",
        "FINGERSPELL A-Z",
        "A=fist  B=flat hand",
        "C=curve  D=index up",
        "F=OK+3 up  I=pinky",
        "L=thumb+index  O=O-shape",
        "V=peace  Y=shaka",
        "",
        "NUMBERS 0-9",
        "0=O  1=index  2=V",
        "3=3 fingers  4=flat",
        "5=open palm",
        "",
        "QUICK WORDS",
        "Open palm=Hello",
        "Fist=Please",
        "ILY=I love you",
        "Thumb up=Yes",
        "",
        "Press G to hide",
    ]
    y = 24
    for line in lines:
        color = COLOR_CYAN if line == "GESTURE GUIDE" else COLOR_WHITE
        size = 0.48 if line else 0.3
        if line:
            cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)
        y += 18 if line else 8

    frame[:, w - panel_w :] = panel
