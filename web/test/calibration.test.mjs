/**
 * Calibration k-NN behaviour.
 *
 *   node --test web/test/
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

const vec = (seed) =>
  Float32Array.from({ length: 73 }, (_, i) => Math.sin(seed * 13.7 + i * 0.31));

const jitter = (v, amount, rng = Math.random) =>
  Float32Array.from(v, (x) => x + (rng() - 0.5) * amount);

function fresh() {
  store.clear();
  return new Calibration();
}

test("no prediction before any samples exist", () => {
  const c = fresh();
  assert.equal(c.predict(vec(1)), null);
  assert.equal(c.isActive, false);
});

test("a label below MIN_SAMPLES is ignored", () => {
  const c = fresh();
  const a = vec(1);
  for (let i = 0; i < MIN_SAMPLES - 1; i += 1) c.add("A", jitter(a, 0.02));

  assert.equal(c.countFor("A"), MIN_SAMPLES - 1);
  assert.deepEqual(c.trainedLabels, []);
  assert.equal(c.predict(a), null);
});

test("matches the label whose samples are nearest", () => {
  const c = fresh();
  const a = vec(1);
  const b = vec(9);
  for (let i = 0; i < 12; i += 1) {
    c.add("A", jitter(a, 0.02));
    c.add("B", jitter(b, 0.02));
  }

  assert.equal(c.predict(jitter(a, 0.03)).label, "A");
  assert.equal(c.predict(jitter(b, 0.03)).label, "B");
});

test("rejects a shape unlike anything recorded", () => {
  const c = fresh();
  const a = vec(1);
  for (let i = 0; i < 12; i += 1) c.add("A", jitter(a, 0.02));

  // Far away in feature space: better to defer to the model than guess.
  assert.equal(c.predict(vec(42)), null);
});

test("pruning does not change which label wins", () => {
  // The distance loop exits early once a row cannot beat the current worst
  // neighbour; that must never turn into a wrong (partial) distance.
  const c = fresh();
  const labels = ["A", "B", "C", "D"];
  labels.forEach((label, index) => {
    const centre = vec(index * 3 + 1);
    for (let i = 0; i < 10; i += 1) c.add(label, jitter(centre, 0.02));
  });

  for (const [index, label] of labels.entries()) {
    const probe = jitter(vec(index * 3 + 1), 0.03);
    assert.equal(c.predict(probe).label, label, `probe for ${label}`);
  }
});

test("confidence rises when neighbours agree", () => {
  const c = fresh();
  const a = vec(1);
  for (let i = 0; i < 15; i += 1) c.add("A", jitter(a, 0.01));

  const result = c.predict(jitter(a, 0.01));
  assert.equal(result.label, "A");
  assert.equal(result.confidence, 1); // every neighbour is an A
  assert.ok(result.distance >= 0);
});

test("samples per label are capped, keeping the newest", () => {
  const c = fresh();
  const a = vec(1);
  for (let i = 0; i < MAX_SAMPLES + 10; i += 1) c.add("A", jitter(a, 0.02));
  assert.equal(c.countFor("A"), MAX_SAMPLES);
});

test("samples survive a reload and clear() removes them", () => {
  const c = fresh();
  const a = vec(1);
  for (let i = 0; i < 12; i += 1) c.add("A", jitter(a, 0.02));

  const reloaded = new Calibration();
  assert.equal(reloaded.countFor("A"), 12);
  assert.equal(reloaded.predict(jitter(a, 0.03)).label, "A");

  reloaded.clear("A");
  assert.equal(new Calibration().countFor("A"), 0);
});

test("unreadable storage degrades to empty rather than throwing", () => {
  store.clear();
  store.set("signal.calibration.v1", "{not json");
  const c = new Calibration();
  assert.equal(c.total, 0);
  assert.equal(c.isActive, false);
});
