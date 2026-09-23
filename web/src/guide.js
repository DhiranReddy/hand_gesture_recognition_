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
  YES: "Thumb up, fingers closed — a thumbs-up",
  NO: "Thumb down, fingers closed — a thumbs-down",
  HELLO: "Open hand held up, fingers spread — a wave",
  PLEASE: "Open hand held up with the fingers together",
  THANK_YOU: "Flat hand, fingers together, moving out from the chin",
  HELP: "Index and pinky out, thumb closed — horns",
  LOVE: "Thumb, index and pinky out together — I-love-you",
  OK: "Thumb and index in a ring, other three fingers up",
  YOU: "Index finger pointing sideways, at the person you mean",
  WATER: "Three fingers up with the thumb out — a W for water",
  FOOD: "Fingertips gathered to a point, as if lifting food",
};

/** Why each phrase gesture is the shape it is, shown in the guide. */
export const PHRASE_REASONS = {
  YES: "The universal thumbs-up. No letter or digit uses it, so it reads in any mode.",
  NO: "The same hand rotated down — the opposite sign for the opposite word, and equally unambiguous.",
  HELLO: "An open hand mid-wave. Fingers spread is hello; together is please.",
  PLEASE: "An open hand held flat and still, the way you hold it out to ask.",
  THANK_YOU: "The ASL sign: a flat hand leaving the chin, so it points sideways.",
  HELP: "Horns — two fingers up like a distress signal. Shares a shape with C, so use Words mode if C keeps winning.",
  LOVE: "The I-L-Y sign: I, L and Y in one hand. The thumb is what separates it from Help, and nothing else claims it.",
  OK: "The universal OK ring. It shares a shape with F and 9, so it wins only outside Letters mode.",
  YOU: "Pointing at the person. Sideways, so it is not confused with the number one.",
  WATER: "A W for water, with the thumb out so it is not the letter W.",
  FOOD: "Fingertips pinched as if lifting food to your mouth.",
};


/** Letters whose true ASL form requires movement the static model cannot see. */
export const MOTION_LETTERS = new Set(["J", "Z"]);

/** Phrase shapes a letter or digit also wants; these need Words or Letters mode. */
export const CONFUSABLE = new Set([
  "HELLO", "PLEASE", "YOU", "THANK_YOU", "WATER",
]);

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
