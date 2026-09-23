/**
 * Personal calibration: a k-nearest-neighbour layer over samples you record.
 *
 * The shipped MLP was trained on somebody else's hands, and its training set is
 * missing, so its accuracy on any given person is unknown and unimprovable by
 * retraining. This sidesteps that: record a handful of examples of a gesture as
 * *you* make it, and matching falls back to your own examples whenever they are
 * a close match.
 *
 * It reuses the same 73-d feature vector as the MLP, so the two are directly
 * comparable and calibration costs no extra work per frame.
 */

const STORAGE_KEY = "signal.calibration.v1";
const PRECISION = 4;

/** Below this distance a recorded sample counts as "the same shape". */
const MATCH_DISTANCE = 1.45;
/** Neighbours consulted per prediction. */
const K = 5;
/** Fewer samples than this for a label and it is ignored — too noisy to trust. */
export const MIN_SAMPLES = 5;
/** Beyond this per label, extra samples add cost without accuracy. */
export const MAX_SAMPLES = 40;

function round(value) {
  return Number(value.toFixed(PRECISION));
}

export class Calibration {
  constructor() {
    /** @type {Map<string, number[][]>} */
    this.samples = new Map();
    this.#load();
  }

  #load() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      for (const [label, rows] of Object.entries(JSON.parse(raw))) {
        if (Array.isArray(rows) && rows.length) this.samples.set(label, rows);
      }
    } catch {
      /* Corrupt or blocked storage — start empty rather than break the app. */
    }
  }

  #save() {
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify(Object.fromEntries(this.samples)),
      );
      return true;
    } catch {
      // Quota exceeded is realistic at ~40 samples x 47 labels.
      return false;
    }
  }

  add(label, features) {
    const rows = this.samples.get(label) || [];
    if (rows.length >= MAX_SAMPLES) rows.shift();
    rows.push(Array.from(features, round));
    this.samples.set(label, rows);
    return this.#save();
  }

  countFor(label) {
    return this.samples.get(label)?.length || 0;
  }

  get counts() {
    return Object.fromEntries(
      [...this.samples].map(([label, rows]) => [label, rows.length]),
    );
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
    if (label) this.samples.delete(label);
    else this.samples.clear();
    this.#save();
  }

  /**
   * Nearest-neighbour match against recorded samples.
   * @returns {{label: string, confidence: number, distance: number}|null}
   *   null when nothing recorded is close enough to be worth trusting.
   */
  predict(features) {
    if (!this.isActive) return null;

    /** @type {{label: string, d: number}[]} */
    const near = [];
    for (const [label, rows] of this.samples) {
      if (rows.length < MIN_SAMPLES) continue;
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

    return {
      label: best,
      confidence: bestWeight / totalWeight,
      distance: near[0].d,
    };
  }

  toJSON() {
    return { version: 1, samples: Object.fromEntries(this.samples) };
  }

  /** Replace everything with a previously exported payload. */
  load(payload) {
    if (!payload || payload.version !== 1 || !payload.samples) return false;
    this.samples = new Map(Object.entries(payload.samples));
    return this.#save();
  }
}
