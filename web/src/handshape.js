/**
 * Parametric hand diagrams for the gesture guide.
 *
 * Every shape is described with the same vocabulary the classifiers use —
 * per-finger extended/bent/curled plus a thumb position — and drawn from that
 * description. One renderer covers all 36 letters and digits, so a shape and
 * its picture cannot drift apart.
 *
 * Shapes follow canonical ASL. Where the recogniser is known to disagree or to
 * be unable to tell two signs apart, that is recorded in COLLISIONS and shown
 * in the guide rather than hidden.
 */

const E = "e"; // extended
const B = "b"; // bent
const C = "c"; // curled

/**
 * f: index, middle, ring, pinky states.
 * t: thumb position — up | side | across | tucked | between | out |
 *    pinch-index | pinch-middle | pinch-ring | pinch-pinky
 * o: orientation — up (default) | right | down
 * s: spread index and middle apart.
 * x: cross index over middle.
 * special: draw a distinct outline instead of the standard hand.
 * motion: arrow path drawn over the hand.
 */
export const SHAPES = {
  A: { f: "cccc", t: "side" },
  B: { f: "eeee", t: "across" },
  C: { f: "bbbb", t: "out", special: "curve" },
  D: { f: "eccc", t: "pinch-middle" },
  E: { f: "cccc", t: "across" },
  F: { f: "ceee", t: "pinch-index" },
  G: { f: "eccc", t: "out", o: "right" },
  H: { f: "eecc", t: "tucked", o: "right" },
  I: { f: "ccce", t: "side" },
  J: { f: "ccce", t: "side", motion: "j" },
  K: { f: "eecc", t: "between", s: true },
  L: { f: "eccc", t: "up" },
  M: { f: "bbbc", t: "tucked" },
  N: { f: "bbcc", t: "tucked" },
  O: { f: "bbbb", t: "out", special: "ring" },
  P: { f: "eecc", t: "between", s: true, o: "down" },
  Q: { f: "eccc", t: "out", o: "down" },
  R: { f: "eecc", t: "tucked", x: true },
  S: { f: "cccc", t: "across" },
  T: { f: "cccc", t: "between" },
  U: { f: "eecc", t: "tucked" },
  V: { f: "eecc", t: "tucked", s: true },
  W: { f: "eeec", t: "tucked" },
  X: { f: "bccc", t: "side" },
  Y: { f: "ccce", t: "out" },
  Z: { f: "eccc", t: "tucked", motion: "z" },

  0: { f: "bbbb", t: "out", special: "ring" },
  1: { f: "eccc", t: "tucked" },
  2: { f: "eecc", t: "tucked", s: true },
  3: { f: "eecc", t: "up", s: true },
  4: { f: "eeee", t: "across" },
  5: { f: "eeee", t: "out" },
  6: { f: "eeec", t: "pinch-pinky" },
  7: { f: "eece", t: "pinch-ring" },
  8: { f: "ecee", t: "pinch-middle" },
  9: { f: "ceee", t: "pinch-index" },
};

/**
 * Signs the static recogniser cannot separate, or where the shipped rules
 * disagree with canonical ASL. Surfaced in the guide so a user who sees the
 * "wrong" letter understands why.
 */
export const COLLISIONS = {
  O: { with: ["0"], why: "O and zero are the same static shape." },
  0: { with: ["O"], why: "Zero and O are the same static shape." },
  V: { with: ["2"], why: "V and two are the same static shape." },
  2: { with: ["V"], why: "Two and V are the same static shape." },
  F: { with: ["9"], why: "F and nine are the same static shape." },
  9: { with: ["F"], why: "Nine and F are the same static shape." },
  1: { with: ["Z", "D"], why: "One, Z and D all show a single raised index finger." },
  Z: { with: ["1"], why: "Z is drawn in the air; held still it reads as one." },
  J: { with: ["I"], why: "J is drawn in the air; held still it reads as I." },
  I: { with: ["J"], why: "I and a still J are the same shape." },
  7: { with: ["8"], why: "The shipped rules pinch the wrong finger for 7 and 8, so they can swap." },
  8: { with: ["7"], why: "The shipped rules pinch the wrong finger for 7 and 8, so they can swap." },
  M: { with: ["N", "S"], why: "M, N and S differ only by how many fingers cover the thumb." },
  N: { with: ["M", "S"], why: "M, N and S differ only by how many fingers cover the thumb." },
  A: { with: ["S", "E"], why: "A, S and E are all closed fists with different thumb placement." },
  S: { with: ["A", "E", "M", "N"], why: "S, A and E are all closed fists with different thumb placement." },
  E: { with: ["A", "S"], why: "E, A and S are all closed fists with different thumb placement." },
  U: { with: ["V"], why: "U and V differ only by whether the two fingers are spread." },
};

/* ------------------------------------------------------------- geometry */

const SVG_NS = "http://www.w3.org/2000/svg";
const BASE_Y = 78;

const FINGERS = [
  { key: "index", x: 45, len: 40, w: 11 },
  { key: "middle", x: 58.5, len: 46, w: 11.5 },
  { key: "ring", x: 72, len: 41, w: 11 },
  { key: "pinky", x: 84, len: 31, w: 9.5 },
];

const SCALE = { e: 1, b: 0.5, c: 0.19 };

const THUMB = {
  up: { x: 30, y: 74, len: 34, angle: -18, w: 12 },
  out: { x: 28, y: 84, len: 32, angle: -52, w: 12 },
  side: { x: 32, y: 86, len: 26, angle: -12, w: 12 },
  across: { x: 34, y: 96, len: 30, angle: 68, w: 11 },
  tucked: { x: 34, y: 96, len: 18, angle: 40, w: 11 },
  between: { x: 46, y: 80, len: 20, angle: 8, w: 10 },
};

function node(name, attrs = {}) {
  const element = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) {
    element.setAttribute(key, String(value));
  }
  return element;
}

function capsule(x, y, length, width, angle = 0) {
  const rect = node("rect", {
    x: x - width / 2,
    y: y - length,
    width,
    height: length,
    rx: width / 2,
    fill: "var(--hand-fill)",
    stroke: "var(--hand-line)",
    "stroke-width": 2,
  });
  if (angle) rect.setAttribute("transform", `rotate(${angle} ${x} ${y})`);
  return rect;
}

function fingertipPoint(finger, state, spread) {
  const length = finger.len * SCALE[state];
  let x = finger.x;
  if (spread && finger.key === "index") x -= 6;
  if (spread && finger.key === "middle") x += 6;
  return { x, y: BASE_Y - length };
}

/**
 * Build the SVG for one gesture.
 * @param {string} label letter or digit
 * @returns {SVGElement|null}
 */
export function renderHand(label) {
  const shape = SHAPES[label];
  if (!shape) return null;

  const svg = node("svg", {
    viewBox: "0 0 120 150",
    class: "hand-svg",
    role: "img",
    "aria-label": `Hand shape for ${label}`,
  });

  const rotation = shape.o === "right" ? -72 : shape.o === "down" ? 168 : 0;
  const group = node("g", rotation ? { transform: `rotate(${rotation} 60 90)` } : {});

  // Palm
  group.append(
    node("rect", {
      x: 37,
      y: BASE_Y - 6,
      width: 54,
      height: 50,
      rx: 17,
      fill: "var(--hand-fill)",
      stroke: "var(--hand-line)",
      "stroke-width": 2,
    }),
  );

  if (shape.special === "ring") {
    // Fingertips meeting the thumb: one open circle reads better than four stubs.
    group.append(
      node("circle", {
        cx: 60,
        cy: 52,
        r: 24,
        fill: "none",
        stroke: "var(--hand-line)",
        "stroke-width": 11,
        "stroke-linecap": "round",
        opacity: 0.92,
      }),
    );
  } else if (shape.special === "curve") {
    group.append(
      node("path", {
        d: "M86 26 A32 32 0 1 0 86 82",
        fill: "none",
        stroke: "var(--hand-line)",
        "stroke-width": 11,
        "stroke-linecap": "round",
        opacity: 0.92,
      }),
    );
  } else {
    const states = shape.f.split("");
    FINGERS.forEach((finger, i) => {
      const state = states[i] || C;
      const length = finger.len * SCALE[state];
      let x = finger.x;
      if (shape.s && finger.key === "index") x -= 6;
      if (shape.s && finger.key === "middle") x += 6;

      // Crossing has to overlap, not just lean, or R reads as U.
      let angle = 0;
      if (shape.x && finger.key === "index") {
        angle = 26;
        x += 5;
      }
      if (shape.x && finger.key === "middle") {
        angle = -20;
        x -= 6;
      }

      group.append(capsule(x, BASE_Y, length, finger.w, angle));
    });
  }

  // Thumb
  const pinch = shape.t.startsWith("pinch-") ? shape.t.slice(6) : null;
  if (pinch) {
    const target = FINGERS.find((f) => f.key === pinch);
    const states = shape.f.split("");
    const index = FINGERS.indexOf(target);
    const tip = fingertipPoint(target, states[index] || C, shape.s);
    group.append(
      node("path", {
        d: `M30 92 Q30 ${tip.y + 14} ${tip.x - 3} ${tip.y + 4}`,
        fill: "none",
        stroke: "var(--hand-line)",
        "stroke-width": 11,
        "stroke-linecap": "round",
      }),
      node("circle", {
        cx: tip.x - 1,
        cy: tip.y + 3,
        r: 9,
        fill: "none",
        stroke: "var(--hand-accent)",
        "stroke-width": 3,
      }),
    );
  } else {
    const spec = THUMB[shape.t] || THUMB.side;
    group.append(capsule(spec.x, spec.y, spec.len, spec.w, spec.angle));
  }

  svg.append(group);

  if (shape.motion) {
    svg.append(
      node("path", {
        d:
          shape.motion === "j"
            ? "M92 40 L92 74 Q92 92 72 92"
            : "M74 30 L104 30 L74 62 L104 62",
        fill: "none",
        stroke: "var(--hand-accent)",
        "stroke-width": 3,
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
        "stroke-dasharray": "5 5",
      }),
    );
  }

  return svg;
}

export function hasDiagram(label) {
  return Boolean(SHAPES[label]);
}
