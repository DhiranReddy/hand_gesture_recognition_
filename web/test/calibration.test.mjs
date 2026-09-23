/**
 * Calibration: k-NN behaviour plus the dataset round-trip.
 *
 *   node --test "web/test/*.test.mjs"
 *
 * The module persists through localStorage, which Node has no notion of, so a
 * minimal in-memory stand-in is installed before importing it.
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

const { Calibration, MIN_SAMPLES, MAX_SAMPLES } = await import(
  "../src/calibration.js"
);
const { landmarksToFeatures } = await import("../src/features.js");

// Samples are stored in pixel space, so features are built with width/height 1.
const featuresOf = (hand, handedness = "Right") =>
  landmarksToFeatures(hand, handedness, 1, 1);

/** A distinct but hand-shaped landmark set, seeded for repeatability. */
function hand(seed) {
  return Array.from({ length: 21 }, (_, i) => ({
    x: 300 + Math.sin(seed * 7.3 + i * 0.7) * 90,
    y: 300 + Math.cos(seed * 5.1 + i * 0.9) * 90,
    z: 0,
  }));
}

function jitter(h, amount, rng = Math.random) {
  return h.map((p) => ({
    x: p.x + (rng() - 0.5) * amount,
    y: p.y + (rng() - 0.5) * amount,
    z: p.z,
  }));
}

function fresh() {
  store.clear();
  return new Calibration();
}

test("no prediction before any samples exist", () => {
  const c = fresh();
  assert.equal(c.predict(featuresOf(hand(1))), null);
  assert.equal(c.isActive, false);
});

test("a label below MIN_SAMPLES is ignored", () => {
  const c = fresh();
  const a = hand(1);
  for (let i = 0; i < MIN_SAMPLES - 1; i += 1) c.add("A", jitter(a, 2));

  assert.equal(c.countFor("A"), MIN_SAMPLES - 1);
  assert.deepEqual(c.trainedLabels, []);
  assert.equal(c.predict(featuresOf(a)), null);
});

test("matches the label whose samples are nearest", () => {
  const c = fresh();
  const a = hand(1);
  const b = hand(9);
  for (let i = 0; i < 12; i += 1) {
    c.add("A", jitter(a, 2));
    c.add("B", jitter(b, 2));
  }

  assert.equal(c.predict(featuresOf(jitter(a, 3))).label, "A");
  assert.equal(c.predict(featuresOf(jitter(b, 3))).label, "B");
});

test("rejects a shape unlike anything recorded", () => {
  const c = fresh();
  const a = hand(1);
  for (let i = 0; i < 12; i += 1) c.add("A", jitter(a, 2));
  assert.equal(c.predict(featuresOf(hand(42))), null);
});

test("pruning does not change which label wins", () => {
  // The distance loop exits early once a row cannot beat the current worst
  // neighbour; that must never turn into a wrong (partial) distance.
  const c = fresh();
  const labels = ["A", "B", "C", "D"];
  labels.forEach((label, index) => {
    const centre = hand(index * 3 + 1);
    for (let i = 0; i < 10; i += 1) c.add(label, jitter(centre, 2));
  });

  for (const [index, label] of labels.entries()) {
    const probe = featuresOf(jitter(hand(index * 3 + 1), 3));
    assert.equal(c.predict(probe).label, label, `probe for ${label}`);
  }
});

test("samples per label are capped, keeping the newest", () => {
  const c = fresh();
  const a = hand(1);
  for (let i = 0; i < MAX_SAMPLES + 10; i += 1) c.add("A", jitter(a, 2));
  assert.equal(c.countFor("A"), MAX_SAMPLES);
});

test("samples survive a reload and clear() removes them", () => {
  const c = fresh();
  const a = hand(1);
  for (let i = 0; i < 12; i += 1) c.add("A", jitter(a, 2));

  const reloaded = new Calibration();
  assert.equal(reloaded.countFor("A"), 12);
  assert.equal(reloaded.predict(featuresOf(jitter(a, 3))).label, "A");

  reloaded.clear("A");
  assert.equal(new Calibration().countFor("A"), 0);
});

test("unreadable storage degrades to empty rather than throwing", () => {
  store.clear();
  store.set("signal.calibration.v2", "{not json");
  const c = new Calibration();
  assert.equal(c.total, 0);
  assert.equal(c.isActive, false);
});

/* ------------------------------------------------- dataset round-trip */

test("export produces the trainer's JSONL shape", () => {
  const c = fresh();
  for (let i = 0; i < 3; i += 1) c.add("NEED_HELP", jitter(hand(2), 2), "Left");

  const lines = c.toJSONL().trim().split("\n");
  assert.equal(lines.length, 3);

  const record = JSON.parse(lines[0]);
  assert.deepEqual(Object.keys(record).sort(), ["handedness", "label", "landmarks"]);
  assert.equal(record.label, "NEED_HELP");
  assert.equal(record.handedness, "Left");
  assert.equal(record.landmarks.length, 21);
  assert.equal(record.landmarks[0].length, 3);
  assert.ok(record.landmarks.every((p) => p.every(Number.isFinite)));
});

test("import restores samples that then match", () => {
  const source = fresh();
  const a = hand(1);
  for (let i = 0; i < 12; i += 1) source.add("A", jitter(a, 2));
  const exported = source.toJSONL();

  const target = fresh();
  assert.equal(target.fromJSONL(exported), 12);
  assert.equal(target.countFor("A"), 12);
  assert.equal(target.predict(featuresOf(jitter(a, 3))).label, "A");
});

test("import skips malformed lines instead of failing", () => {
  const c = fresh();
  const good = JSON.stringify({
    label: "A",
    handedness: "Right",
    landmarks: hand(1).map((p) => [p.x, p.y, p.z]),
  });
  const text = [
    good,
    "",
    "{not json",
    JSON.stringify({ label: "B", landmarks: [[0, 0, 0]] }), // wrong length
    JSON.stringify({ landmarks: hand(2).map((p) => [p.x, p.y, p.z]) }), // no label
    good,
  ].join("\n");

  assert.equal(c.fromJSONL(text), 2);
  assert.equal(c.countFor("A"), 2);
  assert.equal(c.countFor("B"), 0);
});

test("an empty calibration exports nothing", () => {
  assert.equal(fresh().toJSONL(), "");
});
