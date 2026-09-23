/**
 * Fusion policy: which source wins, and when the bundled model is allowed in.
 *
 *   node --test "web/test/*.test.mjs"
 *
 * Rule correctness itself is covered by tools/check_rules_parity.py, which
 * compares the JS rules against the Python originals over 600 hands.
 */

import test from "node:test";
import assert from "node:assert/strict";

const store = new Map();
globalThis.localStorage = {
  getItem: (k) => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
  removeItem: (k) => store.delete(k),
  clear: () => store.clear(),
};

const { Recognizer } = await import("../src/recognizer.js");
const { Calibration } = await import("../src/calibration.js");
const { landmarksToFeatures } = await import("../src/features.js");

const W = 960;
const H = 540;

// A neutral open hand in pixel space; fingers rise from a wrist at the bottom.
const OPEN = [
  [480, 460], [440, 435], [412, 405], [392, 378], [376, 352],
  [452, 372], [444, 330], [440, 302], [437, 278],
  [480, 364], [480, 318], [480, 288], [480, 262],
  [508, 370], [514, 326], [518, 298], [521, 274],
  [534, 384], [546, 350], [554, 328], [560, 308],
];

const norm = (hand) => hand.map(([x, y]) => ({ x: x / W, y: y / H, z: 0 }));

/** Pull the given fingertips back toward their PIP joint. */
function curl(hand, pairs) {
  const out = hand.map((p) => [...p]);
  for (const [tip, pip] of pairs) {
    out[tip] = [out[pip][0], out[pip][1] + 26];
  }
  return out;
}

const ALL_FINGERS = [[4, 2], [8, 6], [12, 10], [16, 14], [20, 18]];
const FIST = curl(OPEN, ALL_FINGERS);
const INDEX_UP = curl(OPEN, [[4, 2], [12, 10], [16, 14], [20, 18]]);
const PEACE = curl(OPEN, [[4, 2], [16, 14], [20, 18]]);

function make({ calibration = null, classifier = null } = {}) {
  return new Recognizer({ classifier, calibration });
}

test("rules recognise distinct shapes as distinct labels", () => {
  const r = make();
  const seen = new Map();
  for (const [name, hand] of [
    ["open", OPEN], ["fist", FIST], ["index", INDEX_UP], ["peace", PEACE],
  ]) {
    r.reset();
    const result = r.recognizeFrame(norm(hand), "Right", W, H);
    assert.ok(result.label, `${name} should be recognised`);
    assert.equal(result.source, "rules");
    seen.set(name, result.label);
  }
  // The failure this replaced was every shape collapsing to one label.
  assert.equal(new Set(seen.values()).size, seen.size, `distinct: ${[...seen]}`);
});

test("the bundled model is ignored unless explicitly enabled", () => {
  const calls = [];
  const classifier = {
    predict: (...args) => {
      calls.push(args);
      return { label: "HELP", confidence: 0.99 };
    },
    setMode() {},
    reset() {},
  };

  // A shape no rule matches, so the model is the only thing that could answer.
  const flat = norm(OPEN.map(([x]) => [x, 300]));

  const off = make({ classifier });
  assert.equal(off.recognizeFrame(flat, "Right", W, H).label, null);
  assert.equal(calls.length, 0, "model must not even be consulted");

  const on = make({ classifier });
  on.useModel = true;
  const result = on.recognizeFrame(flat, "Right", W, H);
  assert.equal(result.label, "HELP");
  assert.equal(result.source, "model");
});

test("an enabled model still has to clear a high confidence bar", () => {
  const classifier = {
    predict: () => ({ label: "HELP", confidence: 0.7 }),
    setMode() {},
    reset() {},
  };
  const r = make({ classifier });
  r.useModel = true;
  const flat = norm(OPEN.map(([x]) => [x, 300]));
  assert.equal(r.recognizeFrame(flat, "Right", W, H).label, null);
});

test("your own samples outrank the rules", () => {
  store.clear();
  const calibration = new Calibration();
  const hand = norm(PEACE);

  // Teach the peace shape as something else entirely.
  const features = landmarksToFeatures(hand, "Right", W, H);
  for (let i = 0; i < 12; i += 1) calibration.add("THANK_YOU", features);

  const bare = make();
  const byRule = bare.recognizeFrame(hand, "Right", W, H);
  assert.equal(byRule.source, "rules");
  assert.notEqual(byRule.label, "THANK_YOU");

  const tuned = make({ calibration });
  const mine = tuned.recognizeFrame(hand, "Right", W, H);
  assert.equal(mine.label, "THANK_YOU");
  assert.equal(mine.source, "you");
});

test("mode filters which labels may be returned", () => {
  const r = make();
  const hand = norm(INDEX_UP);

  r.setMode("all");
  const all = r.recognizeFrame(hand, "Right", W, H);
  assert.ok(all.label);

  // Index-up resolves to a digit/letter, so Words mode must reject it.
  r.setMode("words");
  const words = r.recognizeFrame(hand, "Right", W, H);
  assert.ok(words.label === null || words.label.length > 1);

  r.setMode("letters");
  const letters = r.recognizeFrame(hand, "Right", W, H);
  assert.ok(letters.label === null || letters.label.length === 1);
});

test("category matches the label kind", () => {
  const r = make();
  const result = r.recognizeFrame(norm(PEACE), "Right", W, H);
  const expected = /^[0-9]$/.test(result.label)
    ? "number"
    : result.label.length === 1
      ? "letter"
      : "phrase";
  assert.equal(result.category, expected);
});

test("no hand clears the smoothing window", () => {
  const r = make();
  for (let i = 0; i < 4; i += 1) r.recognize(norm(PEACE), "Right", W, H);
  assert.ok(r.history.length > 0);

  const empty = r.recognize(null, "Right", W, H);
  assert.equal(empty.label, null);
  assert.equal(r.history.length, 0);
});

test("smoothing holds a steady reading across frames", () => {
  const r = make();
  let last = null;
  for (let i = 0; i < 6; i += 1) {
    last = r.recognize(norm(PEACE), "Right", W, H);
  }
  assert.ok(last.label);
  assert.ok(last.confidence > 0);
});
