import test from "node:test";
import assert from "node:assert/strict";
import { WordProcessor, SpellCorrector } from "../src/words.js";

const SAMPLE_WORDS = [
  "the", "you", "i", "to", "a", "and", "hello", "help", "water", "please",
  "yes", "no", "good", "meet", "look", "will", "call", "all", "food", "book",
];

test("SpellCorrector corrects common typos", () => {
  const corrector = new SpellCorrector(SAMPLE_WORDS);
  assert.equal(corrector.correct("helpp"), "help");
  assert.equal(corrector.correct("watr"), "water");
  assert.equal(corrector.correct("pleas"), "please");
});

test("SpellCorrector leaves numbers untouched (does not convert digits to 'i')", () => {
  const corrector = new SpellCorrector(SAMPLE_WORDS);
  assert.equal(corrector.correct("10"), "10");
  assert.equal(corrector.correct("12"), "12");
  assert.equal(corrector.correct("42"), "42");
  assert.equal(corrector.correct("99"), "99");
  assert.equal(corrector.correct("2026"), "2026");
});

test("WordProcessor buffers letters and auto-spaces on idle", () => {
  const corrector = new SpellCorrector(SAMPLE_WORDS);
  const wp = new WordProcessor(corrector, { pauseMs: 200 });

  wp.addGesture("H", "letter");
  wp.addGesture("E", "letter");
  wp.addGesture("L", "letter");
  wp.addGesture("P", "letter");

  assert.equal(wp.buffer, "HELP");
  assert.equal(wp.transcript, "");
  assert.equal(wp.displayTranscript, "HELP");

  // Before pause timeout, onIdle does not commit
  assert.equal(wp.onIdle(100), false);
  assert.equal(wp.buffer, "HELP");

  // After pause timeout, onIdle commits & spell-checks
  assert.equal(wp.onIdle(350), true);
  assert.equal(wp.buffer, "");
  assert.equal(wp.transcript, "help");
});

test("WordProcessor correctly handles double letters in words", () => {
  const corrector = new SpellCorrector(SAMPLE_WORDS);
  const wp = new WordProcessor(corrector, { pauseMs: 200 });

  wp.addGesture("H", "letter");
  wp.addGesture("E", "letter");
  wp.addGesture("L", "letter");
  wp.addGesture("L", "letter");
  wp.addGesture("O", "letter");

  assert.equal(wp.buffer, "HELLO");
  wp.flush();
  assert.equal(wp.transcript, "hello");
});

test("WordProcessor preserves numbers verbatim", () => {
  const corrector = new SpellCorrector(SAMPLE_WORDS);
  const wp = new WordProcessor(corrector, { pauseMs: 200 });

  wp.addGesture("4", "number");
  wp.addGesture("2", "number");
  wp.flush();

  assert.equal(wp.transcript, "42");
});

test("WordProcessor supports undo", () => {
  const corrector = new SpellCorrector(SAMPLE_WORDS);
  const wp = new WordProcessor(corrector, { pauseMs: 200 });

  wp.addGesture("H", "letter");
  wp.addGesture("I", "letter");
  assert.equal(wp.buffer, "HI");

  assert.equal(wp.undo(), true);
  assert.equal(wp.buffer, "H");

  assert.equal(wp.undo(), true);
  assert.equal(wp.buffer, "");

  assert.equal(wp.undo(), false);
});
