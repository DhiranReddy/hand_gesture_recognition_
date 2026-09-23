/**
 * Dense MLP inference + temporal smoothing.
 *
 * Ports src/ml_gesture_model.py (forward pass, confidence gating) and the
 * voting half of src/gesture_ensemble.py. The rule-based classifiers are not
 * ported: the trained model already covers all 47 labels, and running two
 * disagreeing recognizers in the browser buys inconsistency, not accuracy.
 */

import { landmarksToFeatures, FEATURE_LENGTH } from "./features.js?v=13";

const MIN_CONFIDENCE = 0.56;
const MIN_MARGIN = 0.1;
const SMOOTH_WINDOW = 5;
const MIN_VOTES = 3;

export const MODE_ALL = "all";
export const MODE_WORDS = "words";
export const MODE_LETTERS = "letters";

function relu(v) {
  for (let i = 0; i < v.length; i += 1) {
    if (v[i] < 0) v[i] = 0;
  }
  return v;
}

function softmax(v) {
  let max = -Infinity;
  for (let i = 0; i < v.length; i += 1) {
    if (v[i] > max) max = v[i];
  }
  let sum = 0;
  for (let i = 0; i < v.length; i += 1) {
    v[i] = Math.exp(v[i] - max);
    sum += v[i];
  }
  for (let i = 0; i < v.length; i += 1) v[i] /= sum;
  return v;
}

/** Column-major-free dense layer: out[j] = sum_i(in[i] * W[i][j]) + b[j]. */
function dense(input, layer) {
  const { weights, bias } = layer;
  const out = new Float64Array(bias.length);
  out.set(bias);
  for (let i = 0; i < input.length; i += 1) {
    const x = input[i];
    if (x === 0) continue;
    const row = weights[i];
    for (let j = 0; j < out.length; j += 1) {
      out[j] += x * row[j];
    }
  }
  return out;
}

export class GestureClassifier {
  constructor(spec) {
    if (spec.format !== "dense-mlp-v2") {
      throw new Error(`Unsupported model format: ${spec.format}`);
    }
    if (spec.n_features !== FEATURE_LENGTH) {
      throw new Error(
        `Model expects ${spec.n_features} features, extractor produces ${FEATURE_LENGTH}.`,
      );
    }
    this.labels = spec.labels;
    this.layers = spec.layers;
    // Inputs are standardised before the first layer when the trainer fitted a
    // StandardScaler. Skipping this would silently wreck every prediction.
    this.mean = spec.standardizer ? Float64Array.from(spec.standardizer.mean) : null;
    this.scale = spec.standardizer ? Float64Array.from(spec.standardizer.scale) : null;
    this.metadata = spec.source_metadata || {};
    this.history = [];
    // Registering the exported phrase table keeps display text identical to the
    // Python app ("I don't understand", not "Dont Understand").
    registerDisplayNames(spec.display_names);
    this.setMode(MODE_ALL);
  }

  static async load(url) {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Could not load model (${response.status}) from ${url}`);
    }
    return new GestureClassifier(await response.json());
  }

  /**
   * Restrict which labels can be returned. Narrowing to letters or words makes
   * fingerspelling far more reliable, because visually similar shapes across
   * categories (2 vs V, O vs 0) stop competing.
   */
  setMode(mode) {
    this.mode = mode;
    this.allowed = this.labels.map((label) => {
      if (mode === MODE_WORDS) return label.length > 1;
      if (mode === MODE_LETTERS) return label.length === 1;
      return true;
    });
    this.reset();
  }

  /** Whether the current mode permits this label. */
  isAllowed(label) {
    const index = this.labels.indexOf(label);
    return index !== -1 && this.allowed[index];
  }

  /** Raw single-frame prediction with no smoothing. */
  predict(landmarks, handedness, width, height) {
    let activations = landmarksToFeatures(landmarks, handedness, width, height);
    if (this.mean) {
      const standardised = new Float64Array(activations.length);
      for (let i = 0; i < activations.length; i += 1) {
        standardised[i] = (activations[i] - this.mean[i]) / this.scale[i];
      }
      activations = standardised;
    }
    for (let i = 0; i < this.layers.length; i += 1) {
      activations = dense(activations, this.layers[i]);
      if (i < this.layers.length - 1) relu(activations);
    }
    const probs = softmax(activations);

    let best = -1;
    let second = -1;
    for (let i = 0; i < probs.length; i += 1) {
      if (!this.allowed[i]) continue;
      if (best === -1 || probs[i] > probs[best]) {
        second = best;
        best = i;
      } else if (second === -1 || probs[i] > probs[second]) {
        second = i;
      }
    }
    if (best === -1) return { label: null, confidence: 0 };

    const confidence = probs[best];
    const margin = second === -1 ? confidence : confidence - probs[second];
    if (confidence < MIN_CONFIDENCE || margin < MIN_MARGIN) {
      return { label: null, confidence };
    }
    return { label: this.labels[best], confidence };
  }

  /**
   * Prediction stabilised across a short window, so a single noisy frame
   * cannot commit a wrong letter.
   */
  recognize(landmarks, handedness, width, height) {
    if (!landmarks) {
      this.history.length = 0;
      return { label: null, confidence: 0 };
    }

    const frame = this.predict(landmarks, handedness, width, height);
    if (!frame.label) {
      this.history.length = 0;
      return frame;
    }

    this.history.push(frame);
    if (this.history.length > SMOOTH_WINDOW) this.history.shift();
    if (this.history.length < MIN_VOTES) return frame;

    const tally = new Map();
    for (const entry of this.history) {
      tally.set(entry.label, (tally.get(entry.label) || 0) + 1);
    }
    let winner = frame.label;
    let bestCount = 0;
    for (const [label, count] of tally) {
      if (count > bestCount) {
        winner = label;
        bestCount = count;
      }
    }

    let sum = 0;
    let n = 0;
    for (const entry of this.history) {
      if (entry.label === winner) {
        sum += entry.confidence;
        n += 1;
      }
    }
    return { label: winner, confidence: sum / n };
  }

  reset() {
    this.history.length = 0;
  }
}

/** Human-readable text per label, populated from the exported phrase table. */
const DISPLAY_NAMES = new Map();

export function registerDisplayNames(names) {
  if (!names) return;
  for (const [label, text] of Object.entries(names)) {
    DISPLAY_NAMES.set(label, text);
  }
}

/**
 * Turn a model label into display text, mirroring gesture_to_text() in
 * src/gesture_classifier.py: known labels use the curated phrase, and anything
 * else (a custom dataset id) falls back to title case.
 */
export function labelToText(label) {
  if (!label) return null;
  const known = DISPLAY_NAMES.get(label);
  if (known) return known;
  if (label.length === 1) return label;
  return label
    .split("_")
    .map((part) => part.charAt(0) + part.slice(1).toLowerCase())
    .join(" ");
}

export function categoryFor(label) {
  if (!label) return "none";
  if (label.length === 1 && /[A-Z]/.test(label)) return "letter";
  if (label.length === 1 && /[0-9]/.test(label)) return "number";
  return "phrase";
}
