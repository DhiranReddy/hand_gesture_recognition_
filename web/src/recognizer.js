/**
 * Fuses the three sources of evidence into one reading.
 *
 * Priority, highest first:
 *   1. your calibration samples — recorded from your own hand
 *   2. the trained MLP          — letters only, when confident
 *   3. the geometric rules      — everything else, and the fallback
 *
 * The model covers A-Z and scores 90% on a held-out split of 8411 real
 * photographs (tools/fetch_dataset.py builds that set). Digits and phrases are
 * not in its vocabulary at all, so the rules own those outright and also catch
 * anything the model is unsure about.
 *
 * Which of the two is better on a given webcam is genuinely unknown — the
 * model's accuracy is measured on posed photos, a different distribution from
 * live video — so the model stays switchable rather than assumed correct.
 */

import { landmarksToFeatures } from "./features.js?v=9";
import { classifyByRules } from "./rules.js?v=9";

const SMOOTH_WINDOW = 5;
const MIN_VOTES = 3;

/** Deterministic geometry earns more trust than a probabilistic guess. */
const RULE_CONFIDENCE = 0.9;
/** Personal samples must clear this before they override the rules. */
const PERSONAL_MIN = 0.6;
/** The model must be clearly confident before it outranks the rules. */
const MODEL_MIN = 0.7;

function categoryFor(label) {
  if (!label) return "none";
  if (label.length === 1 && /[A-Z]/.test(label)) return "letter";
  if (label.length === 1 && /[0-9]/.test(label)) return "number";
  return "phrase";
}

export class Recognizer {
  /**
   * @param {object} options
   * @param {import("./classifier.js").GestureClassifier} options.classifier
   * @param {import("./calibration.js").Calibration} options.calibration
   */
  constructor({ classifier, calibration }) {
    this.classifier = classifier;
    this.calibration = calibration;
    this.useModel = true;
    this.mode = "all";
    this.history = [];
  }

  setMode(mode) {
    this.mode = mode;
    this.classifier?.setMode(mode);
    this.reset();
  }

  reset() {
    this.history.length = 0;
    this.classifier?.reset();
  }

  #allowed(label) {
    if (!label) return false;
    if (this.mode === "words") return label.length > 1;
    if (this.mode === "letters") return label.length === 1;
    return true;
  }

  /**
   * @param {{x:number,y:number,z:number}[]|null} landmarks normalized 0-1, as
   *   the detector reports them — the same convention the feature extractor and
   *   the model use. Pixel coordinates for the geometric rules are derived here.
   * @param {string} handedness as reported by the detector
   * @returns {{label: string|null, confidence: number, category: string,
   *            source: "you"|"rules"|"model"|null}}
   */
  recognize(landmarks, handedness, width, height) {
    if (!landmarks) {
      this.reset();
      return { label: null, confidence: 0, category: "none", source: null };
    }

    const frame = this.#single(landmarks, handedness, width, height);
    if (!frame.label) {
      this.history.length = 0;
      return frame;
    }

    this.history.push(frame);
    if (this.history.length > SMOOTH_WINDOW) this.history.shift();
    if (this.history.length < MIN_VOTES) return frame;

    // Majority vote over the window, so one bad frame cannot commit a letter.
    const tally = new Map();
    for (const entry of this.history) {
      tally.set(entry.label, (tally.get(entry.label) || 0) + 1);
    }
    let winner = frame.label;
    let best = 0;
    for (const [label, count] of tally) {
      if (count > best) {
        winner = label;
        best = count;
      }
    }

    const agreeing = this.history.filter((e) => e.label === winner);
    const confidence =
      agreeing.reduce((sum, e) => sum + e.confidence, 0) / agreeing.length;

    return {
      label: winner,
      confidence,
      category: categoryFor(winner),
      source: agreeing[agreeing.length - 1].source,
    };
  }

  /** One frame, no smoothing. Exposed for testing. */
  #single(landmarks, handedness, width, height) {
    const none = { label: null, confidence: 0, category: "none", source: null };

    if (this.calibration?.isActive) {
      const features = landmarksToFeatures(landmarks, handedness, width, height);
      const personal = this.calibration.predict(features);
      if (personal && personal.confidence >= PERSONAL_MIN && this.#allowed(personal.label)) {
        return {
          label: personal.label,
          confidence: personal.confidence,
          category: categoryFor(personal.label),
          source: "you",
        };
      }
    }

    if (this.useModel && this.classifier) {
      const guess = this.classifier.predict(landmarks, handedness, width, height);
      if (guess.label && guess.confidence >= MODEL_MIN && this.#allowed(guess.label)) {
        return {
          label: guess.label,
          confidence: guess.confidence,
          category: categoryFor(guess.label),
          source: "model",
        };
      }
    }

    // The rules work in pixel space; everything else stays normalized. They
    // also cover digits and phrases, which are outside the model's vocabulary.
    const pixels = landmarks.map((p) => ({ x: p.x * width, y: p.y * height }));
    const rule = classifyByRules(pixels, handedness, this.mode);
    if (rule && this.#allowed(rule.label)) {
      return {
        label: rule.label,
        confidence: RULE_CONFIDENCE,
        category: rule.category,
        source: "rules",
      };
    }

    return none;
  }

  /** Test seam: single-frame result without the smoothing window. */
  recognizeFrame(landmarks, handedness, width, height) {
    return this.#single(landmarks, handedness, width, height);
  }
}
