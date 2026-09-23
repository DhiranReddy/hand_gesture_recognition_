/**
 * Word and phrase gestures.
 *
 *   node --test "web/test/*.test.mjs"
 *
 * Each phrase should be reachable from a hand a person could actually make,
 * and the six built on unclaimed shapes must not collide with each other.
 *
 * The fixture is a left hand: thumbState() decides "extended" by which side of
 * the knuckles the thumb sits on, and in this layout it sits at lower x.
 */

import test from "node:test";
import assert from "node:assert/strict";

const { classifyPhrase, classifyByRules, classifyLetter } = await import(
  "../src/rules.js"
);

const HAND = "Left";

// wrist, thumb(1-4), index(5-8), middle(9-12), ring(13-16), pinky(17-20)
const BASE = [
  [480, 460],
  [440, 435], [412, 405], [392, 378], [376, 352],
  [452, 372], [444, 330], [440, 302], [437, 278],
  [480, 364], [480, 318], [480, 288], [480, 262],
  [508, 370], [514, 326], [518, 298], [521, 274],
  [534, 384], [546, 350], [554, 328], [560, 308],
];

const FINGERS = {
  thumb: [4, 2], index: [8, 6], middle: [12, 10], ring: [16, 14], pinky: [20, 18],
};

const pts = (hand) => hand.map(([x, y]) => ({ x, y }));

/** Fold the named fingers back toward their knuckle. */
function fold(hand, names) {
  const out = hand.map((p) => [...p]);
  for (const name of names) {
    const [tip, pip] = FINGERS[name];
    out[tip] = [out[pip][0], out[pip][1] + 26];
  }
  return out;
}

/** Keep only the named fingers extended. */
const raise = (...keep) =>
  fold(BASE, Object.keys(FINGERS).filter((n) => !keep.includes(n)));

function thumbsUp() {
  const hand = fold(BASE, ["index", "middle", "ring", "pinky"]);
  hand[4] = [400, 300];
  hand[3] = [415, 335];
  hand[2] = [428, 372];
  return hand;
}

function thumbsDown() {
  const hand = thumbsUp().map((p) => [...p]);
  hand[4] = [400, 560];
  hand[3] = [415, 528];
  hand[2] = [428, 492];
  return hand;
}

/** Bring index and middle together, so an open hand reads as "together". */
function fingersTogether(hand) {
  const out = hand.map((p) => [...p]);
  for (const [i, x] of [[8, 466], [7, 468], [6, 470], [12, 474], [11, 476], [10, 478]]) {
    out[i][0] = x;
  }
  return out;
}

/** Rotate about the wrist so the hand points sideways. */
const sideways = (hand) =>
  hand.map(([x, y]) => [480 + -(y - 460), 460 + (x - 480) * 0.35]);

const CASES = {
  YES: thumbsUp(),
  NO: thumbsDown(),
  LOVE: raise("thumb", "index", "pinky"),
  HELP: raise("index", "pinky"),
  HELLO: BASE,
  PLEASE: fingersTogether(BASE),
  WATER: raise("thumb", "index", "middle", "ring"),
  YOU: sideways(raise("index")),
  THANK_YOU: sideways(fingersTogether(BASE)),
};

for (const [label, hand] of Object.entries(CASES)) {
  test(`${label} is recognised from its gesture`, () => {
    assert.equal(classifyPhrase(pts(hand), HAND), label);
  });
}

test("every phrase gesture maps to a different phrase", () => {
  const seen = new Set(Object.keys(CASES).map((l) => classifyPhrase(pts(CASES[l]), HAND)));
  assert.equal(seen.size, Object.keys(CASES).length, [...seen].join(", "));
});

test("thumb direction is the only difference between yes and no", () => {
  assert.equal(classifyPhrase(pts(thumbsUp()), HAND), "YES");
  assert.equal(classifyPhrase(pts(thumbsDown()), HAND), "NO");
});

test("the thumb is what separates I-love-you from help", () => {
  // Same two fingers raised; adding the thumb changes the meaning.
  assert.equal(classifyPhrase(pts(raise("index", "pinky")), HAND), "HELP");
  assert.equal(classifyPhrase(pts(raise("thumb", "index", "pinky")), HAND), "LOVE");
});

test("spread waves hello, together asks please", () => {
  assert.equal(classifyPhrase(pts(BASE), HAND), "HELLO");
  assert.equal(classifyPhrase(pts(fingersTogether(BASE)), HAND), "PLEASE");
});

test("thumbs up, thumbs down and I-love-you claim shapes no letter wants", () => {
  // These three are the only genuinely unambiguous ones, so they must read
  // correctly whatever mode the app is in.
  for (const label of ["YES", "NO", "LOVE"]) {
    assert.equal(
      classifyLetter(pts(CASES[label]), HAND),
      null,
      `${label} should not collide with a letter`,
    );
    assert.equal(classifyByRules(pts(CASES[label]), HAND, "all").label, label);
    assert.equal(classifyByRules(pts(CASES[label]), HAND, "words").label, label);
  }
});

test("the phrases that do overlap a letter still win outside Letters mode", () => {
  // Horns also satisfies the loose rule for C; an open palm is also 5. Both
  // are intended, and priority is what resolves them.
  assert.equal(classifyLetter(pts(CASES.HELP), HAND), "C");
  assert.equal(classifyByRules(pts(CASES.HELP), HAND, "all").label, "HELP");
  assert.equal(classifyByRules(pts(CASES.HELP), HAND, "letters").label, "C");
});

test("phrases win in mixed mode, letters win in Letters mode", () => {
  const open = pts(BASE);
  assert.equal(classifyByRules(open, HAND, "all").label, "HELLO");

  const asLetter = classifyByRules(open, HAND, "letters");
  assert.equal(asLetter.category, "number"); // an open palm is the digit 5
  assert.notEqual(asLetter.label, "HELLO");
});

test("Words mode never returns a single character", () => {
  for (const hand of Object.values(CASES)) {
    const result = classifyByRules(pts(hand), HAND, "words");
    if (result) assert.ok(result.label.length > 1, `got ${result.label}`);
  }
});
