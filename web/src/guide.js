/**
 * Reference text for how each gesture is formed.
 *
 * Sourced from the README's ASL tables so the in-app guide teaches the shape
 * rather than just listing label names.
 */

export const HAND_SHAPES = {
  A: "Fist with the thumb resting along the side",
  B: "Flat hand, fingers straight up, thumb tucked across the palm",
  C: "Fingers and thumb curved into a C",
  D: "Index finger up, other fingers curled to meet the thumb",
  E: "Fingers curled down, thumb tucked beneath them",
  F: "Thumb and index touching, the other three fingers up",
  G: "Index finger pointing sideways, thumb parallel",
  H: "Index and middle together, pointing sideways",
  I: "Pinky up, all other fingers closed",
  J: "Pinky up, then trace a J — motion letter",
  K: "Index and middle up in a V, thumb between them",
  L: "Thumb and index at a right angle, an L shape",
  M: "Thumb tucked under three folded fingers",
  N: "Thumb tucked under two folded fingers",
  O: "All fingertips meeting the thumb in an O",
  P: "Like K, rotated to point downward",
  Q: "Like G, rotated to point downward",
  R: "Index and middle crossed over each other",
  S: "Closed fist, thumb crossing the front",
  T: "Thumb poking up between index and middle",
  U: "Index and middle up, held together",
  V: "Index and middle up, spread apart",
  W: "Index, middle and ring up, spread",
  X: "Index finger bent into a hook",
  Y: "Thumb and pinky out, middle fingers closed",
  Z: "Index finger up, then trace a Z — motion letter",
  0: "Fingertips meeting the thumb in a round O",
  1: "Index finger up",
  2: "Index and middle up, spread",
  3: "Thumb, index and middle up",
  4: "Four fingers up, thumb folded across the palm",
  5: "Open palm, all fingers spread",
  6: "Pinky touching the thumb, three fingers up",
  7: "Ring finger touching the thumb",
  8: "Middle finger touching the thumb",
  9: "Index finger touching the thumb",
};

/**
 * Word and phrase gestures, described from the shapes the rule classifier in
 * src/common_phrases.py looks for. These say how the gesture is meant to be
 * formed — useful to learn from, and the closest thing to documentation while
 * the model's original training set is missing.
 */
export const PHRASE_SHAPES = {
  HELLO: "Open palm, all five fingers extended",
  HI: "Fingers bent in a small wave, pinky out",
  GOODBYE: "Thumb, index and middle out; ring and pinky folded",
  THANK_YOU: "Flat hand, four fingers up, thumb tucked in",
  PLEASE: "Closed fist with the thumb held flat against the fingers",
  SORRY: "Fist with the index and thumb bent inward",
  YES: "Fist with the thumb pointing up",
  NO: "Fist with the thumb pointing down",
  OK: "Thumb and index pinched into a ring, other three fingers up",
  LOVE: "Thumb, index and pinky extended; middle and ring folded",
  HELP: "Thumb, index and middle extended; ring and pinky folded",
  YOU: "Index finger pointing forward, thumb closed",
  ME: "Four fingers together pointing in, thumb folded",
  WATER: "Four fingers extended with the pinky folded in",
  FOOD: "Fingertips gathered together, thumb tucked",
  BATHROOM: "Closed fist with the thumb bent forward",
  EMERGENCY: "Four fingers up and spread apart",
  GOOD: "Fist with the thumb up",
  BAD: "Fist with the thumb down",
  HUNGRY: "Fingers curved into a C against the body",
  QUESTION: "Index finger hooked, thumb out",
  ONE_MOMENT: "Index finger raised, thumb closed",
  CAN_YOU_HEAR: "Three fingers up, pinky folded, thumb bent",
};

/** Letters whose true ASL form requires movement the static model cannot see. */
export const MOTION_LETTERS = new Set(["J", "Z"]);

/** Shapes that differ by only one finger, so they are easy to confuse. */
export const CONFUSABLE = new Set(["HELLO", "THANK_YOU", "WATER", "EMERGENCY"]);

export function shapeFor(label) {
  return HAND_SHAPES[label] || PHRASE_SHAPES[label] || null;
}

/**
 * Everything the app can recognise, independent of the model's vocabulary.
 * The model covers A-Z only; digits and phrases come from the rules, so the
 * guide and the calibration picker must not be built from model labels alone.
 */
export const VOCABULARY = [
  ...Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i)),
  ...Array.from({ length: 10 }, (_, i) => String(i)),
  "HELLO", "YES", "NO", "PLEASE", "THANK_YOU", "HELP",
  "LOVE", "OK", "WATER", "FOOD", "YOU",
];

export function groupLabels(labels) {
  return [
    { title: "Letters", items: labels.filter((l) => /^[A-Z]$/.test(l)) },
    { title: "Numbers", items: labels.filter((l) => /^[0-9]$/.test(l)) },
    { title: "Words & phrases", items: labels.filter((l) => l.length > 1) },
  ].filter((group) => group.items.length > 0);
}
