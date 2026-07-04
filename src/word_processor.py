"""Auto-spacing and spell correction for fingerspelled words."""

from __future__ import annotations

from spellchecker import SpellChecker


class WordProcessor:
    """
    Buffers letters into words, auto-corrects on pause, and auto-inserts spaces.
    Phrase gestures are added as whole words immediately.
    """

    PAUSE_FRAMES = 22
    MIN_WORD_LENGTH = 2

    def __init__(self) -> None:
        self.transcript = ""
        self.buffer = ""
        self.preview = ""
        self.pause_frames = 0
        self.last_correction = ""
        self._spell = SpellChecker(distance=2)

        # Boost common conversational words during calls.
        for word in (
            "hello", "hi", "help", "please", "thank", "thanks", "water", "food",
            "yes", "no", "sorry", "love", "understand", "emergency", "doctor",
            "bathroom", "pain", "name", "meet", "zoom", "hear", "see", "wait",
            "good", "fine", "today", "tomorrow", "question", "answer", "mute",
        ):
            self._spell.word_frequency.load_words([word])

    @property
    def display_transcript(self) -> str:
        """Transcript shown in captions (committed text + live buffer)."""
        if self.buffer:
            return f"{self.transcript}{self.buffer}"
        return self.transcript

    def set_preview(self, text: str) -> None:
        self.preview = text

    def clear_preview(self) -> None:
        self.preview = ""

    def on_idle_frame(self) -> bool:
        """Call when no gesture is held. Returns True if buffer was committed."""
        if not self.buffer:
            self.pause_frames = 0
            return False

        self.pause_frames += 1
        if self.pause_frames >= self.PAUSE_FRAMES:
            self._flush_buffer()
            return True
        return False

    def add_gesture(self, text: str, category: str) -> None:
        """Add recognized gesture text from ML/rules."""
        self.pause_frames = 0

        if category in ("letter", "number") or (len(text) == 1 and text.isalnum()):
            self.buffer += text.upper() if text.isalpha() else text
            return

        self._flush_buffer()
        self._append_word(text)

    def _flush_buffer(self) -> None:
        if not self.buffer:
            return

        raw = self.buffer
        corrected = self._autocorrect(raw)
        self.last_correction = corrected if corrected.lower() != raw.lower() else ""
        self._append_word(corrected)
        self.buffer = ""
        self.pause_frames = 0

    def _autocorrect(self, word: str) -> str:
        if len(word) < self.MIN_WORD_LENGTH:
            return word

        lower = word.lower()
        if lower in self._spell:
            return lower

        suggestion = self._spell.correction(lower)
        if suggestion and suggestion != lower:
            return suggestion
        return lower

    def _append_word(self, word: str) -> None:
        if not word:
            return
        if self.transcript and not self.transcript.endswith(" "):
            self.transcript += " "
        self.transcript += word

    def add_space(self) -> None:
        self._flush_buffer()
        if self.transcript and not self.transcript.endswith(" "):
            self.transcript += " "

    def backspace(self) -> None:
        if self.buffer:
            self.buffer = self.buffer[:-1]
        elif self.transcript:
            self.transcript = self.transcript[:-1]

    def clear(self) -> None:
        self.transcript = ""
        self.buffer = ""
        self.preview = ""
        self.pause_frames = 0
        self.last_correction = ""

    def force_flush(self) -> None:
        """Commit buffered letters immediately (e.g. manual space)."""
        self._flush_buffer()
