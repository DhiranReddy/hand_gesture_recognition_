"""Fingerspelling buffer behaviour: buffering, auto-space, and autocorrect."""

from __future__ import annotations

import pytest

from src.word_processor import WordProcessor


@pytest.fixture
def words() -> WordProcessor:
    return WordProcessor()


def spell(processor: WordProcessor, text: str) -> None:
    for char in text:
        processor.add_gesture(char, "letter")


def test_letters_buffer_before_commit(words):
    spell(words, "HEL")
    assert words.buffer == "HEL"
    assert words.transcript == ""
    assert words.display_transcript == "HEL"


def test_pause_commits_buffer(words):
    spell(words, "HELP")
    for _ in range(WordProcessor.PAUSE_FRAMES):
        committed = words.on_idle_frame()
    assert committed is True
    assert words.buffer == ""
    assert words.transcript == "help"


def test_idle_without_buffer_is_noop(words):
    assert words.on_idle_frame() is False
    assert words.transcript == ""


def test_autocorrect_fixes_dropped_letter(words):
    spell(words, "WATR")
    words.force_flush()
    assert words.transcript == "water"
    assert words.last_correction == "water"


def test_correct_spelling_reports_no_correction(words):
    spell(words, "HELP")
    words.force_flush()
    assert words.transcript == "help"
    assert words.last_correction == ""


def test_phrase_flushes_pending_letters(words):
    spell(words, "HI")
    words.add_gesture("Thank you", "phrase")
    assert words.buffer == ""
    assert words.transcript.startswith("hi ")
    assert words.transcript.endswith("Thank you")


def test_numbers_stay_verbatim(words):
    for char in "2024":
        words.add_gesture(char, "number")
    words.force_flush()
    assert "2024" in words.transcript


def test_backspace_prefers_buffer(words):
    spell(words, "HELP")
    words.backspace()
    assert words.buffer == "HEL"
    assert words.transcript == ""


def test_backspace_falls_through_to_transcript(words):
    spell(words, "HI")
    words.force_flush()
    before = words.transcript
    words.backspace()
    assert words.transcript == before[:-1]


def test_single_letter_is_not_autocorrected(words):
    spell(words, "A")
    words.force_flush()
    assert words.transcript == "A"


def test_clear_resets_everything(words):
    spell(words, "HELP")
    words.force_flush()
    words.clear()
    assert (words.transcript, words.buffer, words.preview) == ("", "", "")


def test_words_are_space_separated(words):
    spell(words, "HELP")
    words.force_flush()
    spell(words, "ME")
    words.force_flush()
    assert words.transcript == "help me"


def test_add_space_commits_pending_buffer(words):
    spell(words, "HELP")
    words.add_space()
    assert words.buffer == ""
    assert words.transcript == "help "


def test_double_letter_buffering(words):
    spell(words, "HELLO")
    assert words.buffer == "HELLO"
    words.force_flush()
    assert words.transcript == "hello"


def test_numbers_never_autocorrect_to_i(words):
    for num in ["10", "12", "42", "2026"]:
        words.clear()
        for ch in num:
            words.add_gesture(ch, "number")
        words.force_flush()
        assert words.transcript == num
