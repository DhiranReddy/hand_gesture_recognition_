/**
 * Personal calibration: a k-nearest-neighbour layer over samples you record.
 *
 * Samples are stored as raw landmarks in pixel space — the same thing
 * capture_gesture_dataset.py writes — so a recording session doubles as a
 * training set that train_gesture_model.py can consume via toJSONL().
 *
 * Feature vectors are derived once on load and cached in memory: k-NN needs
 * them every frame, and recomputing hundreds of them per frame would not keep
 * up with the camera.
 *
 * Landmarks are stored already scaled to pixels, so features are built with a
 * width/height of 1 — the extractor then passes the values through untouched,
 * exactly as the Python pipeline does with its pixel-space tuples.
 */

import { landmarksToFeatures } from "./features.js?v=10";

const STORAGE_KEY = "signal.calibration.v2";
const LEGACY_KEYS = ["signal.calibration.v1"];
const PRECISION = 3;

/** Below this distance a recorded sample counts as "the same shape". */
const MATCH_DISTANCE = 1.45;
/** Neighbours consulted per prediction. */
const K = 5;
/** Fewer samples than this for a label and it is ignored — too noisy to trust. */
export const MIN_SAMPLES = 5;
/** Beyond this per label, extra samples add cost without accuracy. */
export const MAX_SAMPLES = 40;

const round = (v) => Number(v.toFixed(PRECISION));

export class Calibration {
  constructor() {
    /** @type {Map<string, {lm: number[][], h: string}[]>} */
    this.samples = new Map();
    /** @type {Map<string, Float32Array[]>} derived, never persisted */
    this.features = new Map();
    this.#load();
  }

  #load() {
    try {
      // v1 stored feature vectors, which cannot be turned back into landmarks.
      for (const key of LEGACY_KEYS) localStorage.removeItem(key);

      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      for (const [label, rows] of Object.entries(JSON.parse(raw))) {
        if (!Array.isArray(rows) || !rows.length) continue;
        const valid = rows.filter((r) => Array.isArray(r?.lm) && r.lm.length === 21);
        if (valid.length) this.samples.set(label, valid);
      }
      for (const label of this.samples.keys()) this.#rebuild(label);
    } catch {
      /* Corrupt or blocked storage — start empty rather than break the app. */
    }
  }

  #rebuild(label) {
    const rows = this.samples.get(label) || [];
    this.features.set(
      label,
      rows.map((row) =>
        landmarksToFeatures(
          row.lm.map(([x, y, z]) => ({ x, y, z })),
          row.h,
          1,
          1,
        ),
      ),
    );
  }

  #save() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(Object.fromEntries(this.samples)));
      return true;
    } catch {
      // Quota exceeded is realistic at 40 samples across many labels.
      return false;
    }
  }

  /**
   * @param {string} label
   * @param {{x:number,y:number,z:number}[]} pixels 21 landmarks in pixel space
   * @param {string} handedness as reported by the detector
   */
  add(label, pixels, handedness = "Right") {
    const rows = this.samples.get(label) || [];
    if (rows.length >= MAX_SAMPLES) rows.shift();
    rows.push({
      lm: pixels.map((p) => [round(p.x), round(p.y), round(p.z ?? 0)]),
      h: handedness === "Left" ? "Left" : "Right",
    });
    this.samples.set(label, rows);
    this.#rebuild(label);
    return this.#save();
  }

  countFor(label) {
    return this.samples.get(label)?.length || 0;
  }

  get counts() {
    return Object.fromEntries([...this.samples].map(([l, r]) => [l, r.length]));
  }

  /** Labels with enough samples to be used for matching. */
  get trainedLabels() {
    return [...this.samples]
      .filter(([, rows]) => rows.length >= MIN_SAMPLES)
      .map(([label]) => label);
  }

  get total() {
    let n = 0;
    for (const rows of this.samples.values()) n += rows.length;
    return n;
  }

  get isActive() {
    return this.trainedLabels.length > 0;
  }

  clear(label) {
    if (label) {
      this.samples.delete(label);
      this.features.delete(label);
    } else {
      this.samples.clear();
      this.features.clear();
    }
    this.#save();
  }

  /**
   * Nearest-neighbour match against recorded samples.
   * @param {Float32Array} features
   * @returns {{label: string, confidence: number, distance: number}|null}
   */
  predict(features) {
    if (!this.isActive) return null;

    /** @type {{label: string, d: number}[]} */
    const near = [];
    for (const [label, rows] of this.features) {
      if ((this.samples.get(label)?.length || 0) < MIN_SAMPLES) continue;
      for (const row of rows) {
        // Prune against whichever bound is tighter: the match cutoff, or the
        // worst neighbour currently held. Squared to avoid a sqrt per row.
        const worst = near.length === K ? near[K - 1].d : MATCH_DISTANCE;
        const limit = Math.min(worst, MATCH_DISTANCE) ** 2;

        let sum = 0;
        let pruned = false;
        for (let i = 0; i < row.length; i += 1) {
          const delta = features[i] - row[i];
          sum += delta * delta;
          if (sum > limit) {
            // Partial sums understate distance, so this row must be dropped
            // outright rather than recorded with a bogus value.
            pruned = true;
            break;
          }
        }
        if (pruned) continue;

        near.push({ label, d: Math.sqrt(sum) });
        near.sort((a, b) => a.d - b.d);
        if (near.length > K) near.pop();
      }
    }

    if (!near.length) return null;

    // Distance-weighted vote: closer samples count for more.
    const votes = new Map();
    let totalWeight = 0;
    for (const { label, d } of near) {
      const weight = 1 / (d + 0.05);
      votes.set(label, (votes.get(label) || 0) + weight);
      totalWeight += weight;
    }

    let best = null;
    let bestWeight = 0;
    for (const [label, weight] of votes) {
      if (weight > bestWeight) {
        best = label;
        bestWeight = weight;
      }
    }

    return { label: best, confidence: bestWeight / totalWeight, distance: near[0].d };
  }

  /**
   * Serialise to the project's training format — one JSON object per line,
   * identical to what capture_gesture_dataset.py appends.
   */
  toJSONL() {
    const lines = [];
    for (const [label, rows] of this.samples) {
      for (const row of rows) {
        lines.push(
          JSON.stringify({ label, handedness: row.h, landmarks: row.lm }),
        );
      }
    }
    return lines.join("\n") + (lines.length ? "\n" : "");
  }

  /** Merge samples from an exported JSONL file. */
  fromJSONL(text) {
    let added = 0;
    for (const line of text.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        const record = JSON.parse(trimmed);
        const lm = record.landmarks;
        if (!Array.isArray(lm) || lm.length !== 21 || !record.label) continue;
        const rows = this.samples.get(record.label) || [];
        if (rows.length >= MAX_SAMPLES) continue;
        rows.push({
          lm: lm.map((p) => [round(+p[0]), round(+p[1]), round(+p[2] || 0)]),
          h: record.handedness === "Left" ? "Left" : "Right",
        });
        this.samples.set(record.label, rows);
        added += 1;
      } catch {
        /* Skip malformed lines rather than abandoning the whole import. */
      }
    }
    for (const label of this.samples.keys()) this.#rebuild(label);
    this.#save();
    return added;
  }
}
