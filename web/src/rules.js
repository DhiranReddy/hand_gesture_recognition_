/**
 * Rule-based ASL recognition — a port of src/asl_alphabet.py, asl_numbers.py
 * and the phrase rules in common_phrases.py.
 *
 * This exists because the shipped MLP is degenerate: probed across thousands of
 * hand poses it answers HELP for 61% of them and never predicts 20 of its 47
 * labels at all. The rules are deterministic geometry, so they neither depend on
 * that model's lost training set nor collapse onto one class.
 *
 * Branch order is load-bearing — the first match wins, exactly as in Python.
 */

import {
  INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP, THUMB_TIP,
  allFingertipsTouch,
  countExtended,
  fingerStates,
  fingertipsPinch,
  indexMiddleCrossed,
  indexMiddleSpread,
  indexMiddleTogether,
  palmSize,
  pointingHorizontal,
  thumbBetweenIndexMiddle,
  WRIST,
} from "./geometry.js?v=11";

/** @returns {string|null} A-Z */
export function classifyLetter(lm, handedness) {
  const s = fingerStates(lm, handedness);
  const [thumb, index, middle, ring, pinky] = s;
  const ext = countExtended(s);

  if (allFingertipsTouch(lm)) return "O";

  if (pinky === "extended" && ext === 1 && thumb !== "extended") return "I";

  if (
    thumb === "extended" && pinky === "extended" &&
    index !== "extended" && middle !== "extended" && ring !== "extended"
  ) return "Y";

  if (
    thumb === "extended" && index === "extended" &&
    middle !== "extended" && ring !== "extended" && pinky !== "extended"
  ) return "L";

  if (
    index === "extended" && middle === "extended" &&
    ring !== "extended" && pinky !== "extended" && indexMiddleSpread(lm)
  ) return "V";

  if (
    index === "extended" && middle === "extended" &&
    ring !== "extended" && pinky !== "extended" && indexMiddleTogether(lm)
  ) return "U";

  if (
    (index === "extended" || index === "bent") &&
    (middle === "extended" || middle === "bent") &&
    indexMiddleCrossed(lm)
  ) return "R";

  if (index === "extended" && middle === "extended" && thumbBetweenIndexMiddle(lm)) return "K";

  if (
    index === "extended" && middle === "extended" &&
    ring === "extended" && pinky !== "extended"
  ) return "W";

  if (
    index === "extended" && middle === "extended" &&
    ring === "extended" && pinky === "extended" && thumb !== "extended"
  ) return "B";

  if (
    fingertipsPinch(lm, THUMB_TIP, INDEX_TIP) &&
    middle === "extended" && ring === "extended" && pinky === "extended"
  ) return "F";

  if (
    index === "extended" && ext === 1 &&
    (thumb === "bent" || thumb === "curled") &&
    middle === "curled" && ring === "curled" && pinky === "curled"
  ) return "D";

  if (
    index === "extended" && pointingHorizontal(lm) &&
    middle === "curled" && ring === "curled" && pinky === "curled"
  ) return "G";

  if (
    index === "extended" && middle === "extended" &&
    indexMiddleTogether(lm) && pointingHorizontal(lm)
  ) return "H";

  if (index === "bent" && middle === "curled" && ring === "curled" && pinky === "curled") {
    return "X";
  }

  if (
    ext >= 2 && ext <= 3 && thumb !== "extended" &&
    !(index === "extended" && middle === "extended" &&
      ring === "extended" && pinky === "extended")
  ) {
    const bentCount = s.slice(1).filter((f) => f === "bent").length;
    if (bentCount >= 1 || (ext === 3 && pinky !== "extended")) return "C";
  }

  if (
    ext === 0 && (thumb === "bent" || thumb === "curled") &&
    s.slice(1).every((f) => f === "curled")
  ) return "A";

  if (ext === 0 && thumb === "curled") return "S";
  if (ext === 0 && thumb === "bent" && thumbBetweenIndexMiddle(lm)) return "T";

  if (
    s.slice(1).every((f) => f === "curled" || f === "bent") &&
    thumb === "bent" && ext === 0
  ) return "E";

  if (
    thumb === "curled" && index === "bent" && middle === "bent" &&
    ring === "bent" && pinky === "curled"
  ) return "M";

  if (
    thumb === "curled" && index === "bent" && middle === "bent" &&
    ring === "curled" && pinky === "curled"
  ) return "N";

  if (index === "extended" && middle === "extended" && thumbBetweenIndexMiddle(lm)) {
    if (lm[INDEX_TIP].y > lm[WRIST].y) return "P";
  }

  if (
    thumb === "extended" && index === "extended" &&
    middle === "curled" && ring === "curled" && pinky === "curled"
  ) {
    if (lm[INDEX_TIP].y > lm[WRIST].y) return "Q";
  }

  if (
    index === "extended" && ext === 1 &&
    middle === "curled" && ring === "curled" && pinky === "curled" &&
    thumb !== "extended"
  ) return "Z";

  return null;
}

/** @returns {string|null} 0-9 */
export function classifyNumber(lm, handedness) {
  const s = fingerStates(lm, handedness);
  const [thumb, index, middle, ring, pinky] = s;
  const ext = countExtended(s);

  if (allFingertipsTouch(lm)) return "0";
  if (index === "extended" && ext === 1 && thumb !== "extended") return "1";
  if (index === "extended" && middle === "extended" && ext === 2 && indexMiddleSpread(lm)) {
    return "2";
  }
  if (
    index === "extended" && middle === "extended" && ring === "extended" &&
    pinky !== "extended" && thumb !== "extended"
  ) return "3";
  if (
    index === "extended" && middle === "extended" &&
    ring === "extended" && pinky === "extended" && thumb !== "extended"
  ) return "4";
  if (s.every((f) => f === "extended")) return "5";
  if (
    fingertipsPinch(lm, THUMB_TIP, PINKY_TIP, 0.4) &&
    middle === "curled" && ring === "curled"
  ) return "6";
  if (index === "extended" && fingertipsPinch(lm, THUMB_TIP, MIDDLE_TIP, 0.35)) return "7";
  if (middle === "extended" && fingertipsPinch(lm, THUMB_TIP, INDEX_TIP, 0.35)) return "8";
  if (
    fingertipsPinch(lm, THUMB_TIP, INDEX_TIP) &&
    middle === "extended" && ring === "extended" && pinky === "extended"
  ) return "9";
  if (
    thumb === "extended" && index === "extended" && middle === "extended" &&
    ring === "curled" && pinky === "curled"
  ) return "3";

  return null;
}

/**
 * Phrase gestures, limited to the labels the shipped model also knows so the
 * two recognisers share a vocabulary. The Python table has ~60 entries, several
 * of which are unreachable duplicates; porting those would import the bugs.
 */
export function classifyPhrase(lm, handedness) {
  const s = fingerStates(lm, handedness);
  const [thumb, index, middle, ring, pinky] = s;
  const ext = countExtended(s);
  const palm = palmSize(lm);

  if (
    thumb === "extended" && index === "extended" && pinky === "extended" &&
    middle === "curled" && ring === "curled"
  ) return "LOVE";

  if (thumb === "extended" && ext === 0) {
    return lm[THUMB_TIP].y < lm[WRIST].y ? "YES" : "NO";
  }

  if (
    fingertipsPinch(lm, THUMB_TIP, INDEX_TIP) &&
    middle === "extended" && ring === "extended" && pinky === "extended"
  ) return "OK";

  if (ext === 0 && thumb !== "extended") return "PLEASE";

  if (
    thumb === "extended" && index === "extended" && middle === "extended" &&
    ring === "curled" && pinky === "curled"
  ) return "HELP";

  if (index === "extended" && ext === 1 && thumb !== "extended") return "YOU";

  if (
    thumb === "extended" && index === "extended" &&
    middle === "extended" && ring === "extended" && pinky === "curled"
  ) return "WATER";

  const tips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP];
  let cx = 0;
  let cy = 0;
  for (const t of tips) {
    cx += lm[t].x;
    cy += lm[t].y;
  }
  const centre = { x: cx / tips.length, y: cy / tips.length };
  if (
    thumb === "curled" &&
    tips.every((t) => Math.hypot(lm[t].x - centre.x, lm[t].y - centre.y) < palm * 0.5)
  ) return "FOOD";

  if (s.every((f) => f === "extended")) return "HELLO";

  if (
    index === "extended" && middle === "extended" &&
    ring === "extended" && pinky === "extended" && thumb !== "extended"
  ) return "THANK_YOU";

  return null;
}

export const MODE_ALL = "all";
export const MODE_WORDS = "words";
export const MODE_LETTERS = "letters";

/** Phrases distinctive enough to outrank a letter in mixed mode. */
const UNIQUE_PHRASES = new Set(["LOVE", "OK", "HELP", "FOOD", "WATER"]);

/**
 * Combined rule verdict, mirroring classify_gesture() in gesture_classifier.py.
 * @returns {{label: string, category: string}|null}
 */
export function classifyByRules(lm, handedness, mode = MODE_ALL) {
  const phrase = classifyPhrase(lm, handedness);
  const number = classifyNumber(lm, handedness);
  const letter = classifyLetter(lm, handedness);

  const asPhrase = phrase ? { label: phrase, category: "phrase" } : null;
  const asLetter = letter ? { label: letter, category: "letter" } : null;
  const asNumber = number ? { label: number, category: "number" } : null;

  if (mode === MODE_WORDS) return asPhrase || asLetter || asNumber;
  if (mode === MODE_LETTERS) return asLetter || asNumber || asPhrase;

  if (phrase && UNIQUE_PHRASES.has(phrase)) return asPhrase;
  return asLetter || asNumber || asPhrase;
}
