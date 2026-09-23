/**
 * Hand geometry helpers — a port of src/finger_geometry.py.
 *
 * All of it works in 2D pixel space (the Python original slices [:2] off every
 * landmark), so z is ignored here too. Thresholds are copied verbatim: they are
 * the calibration of the whole rule-based recogniser, and nudging one silently
 * changes which letters fire.
 */

export const WRIST = 0;
export const THUMB_CMC = 1, THUMB_MCP = 2, THUMB_IP = 3, THUMB_TIP = 4;
export const INDEX_MCP = 5, INDEX_PIP = 6, INDEX_DIP = 7, INDEX_TIP = 8;
export const MIDDLE_MCP = 9, MIDDLE_PIP = 10, MIDDLE_DIP = 11, MIDDLE_TIP = 12;
export const RING_MCP = 13, RING_PIP = 14, RING_DIP = 15, RING_TIP = 16;
export const PINKY_MCP = 17, PINKY_PIP = 18, PINKY_DIP = 19, PINKY_TIP = 20;

/** Landmarks arrive as {x, y} in pixels. */
const px = (lm, i) => lm[i];

function dist(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

export function palmSize(lm) {
  return dist(px(lm, WRIST), px(lm, MIDDLE_MCP));
}

/** extended | bent | curled */
export function fingerCurlState(lm, tip, pip, dip, mcp) {
  const tipToWrist = dist(px(lm, tip), px(lm, WRIST));
  const mcpToWrist = dist(px(lm, mcp), px(lm, WRIST));
  const tipToPip = dist(px(lm, tip), px(lm, pip));
  const pipToDip = dist(px(lm, pip), px(lm, dip));

  if (tipToPip < pipToDip * 0.65) return "curled";
  if (tipToWrist > mcpToWrist * 1.35 && tipToPip > pipToDip * 0.9) return "extended";
  if (tipToPip < pipToDip * 0.95) return "bent";
  return "curled";
}

export function thumbState(lm, handedness) {
  const tip = px(lm, THUMB_TIP);
  const ip = px(lm, THUMB_IP);
  const mcp = px(lm, THUMB_MCP);
  const wrist = px(lm, WRIST);
  const indexMcp = px(lm, INDEX_MCP);

  const tipDist = dist(tip, wrist);
  const ipDist = dist(ip, wrist);

  if (tipDist < ipDist * 0.95) return "curled";

  const outward =
    handedness === "Right" ? tip.x > mcp.x + 5 : tip.x < mcp.x - 5;
  if (outward && tipDist > ipDist * 1.05) return "extended";

  if (dist(tip, indexMcp) < palmSize(lm) * 0.45) return "bent";
  return "curled";
}

/** [thumb, index, middle, ring, pinky] */
export function fingerStates(lm, handedness) {
  return [
    thumbState(lm, handedness),
    fingerCurlState(lm, INDEX_TIP, INDEX_PIP, INDEX_DIP, INDEX_MCP),
    fingerCurlState(lm, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_MCP),
    fingerCurlState(lm, RING_TIP, RING_PIP, RING_DIP, RING_MCP),
    fingerCurlState(lm, PINKY_TIP, PINKY_PIP, PINKY_DIP, PINKY_MCP),
  ];
}

/** Extended count excludes the thumb, matching count_extended(). */
export function countExtended(states) {
  let n = 0;
  for (let i = 1; i < states.length; i += 1) {
    if (states[i] === "extended") n += 1;
  }
  return n;
}

export function fingertipsPinch(lm, a, b, ratio = 0.28) {
  const palm = palmSize(lm);
  if (palm < 1) return false;
  return dist(px(lm, a), px(lm, b)) < palm * ratio;
}

export function allFingertipsTouch(lm, ratio = 0.38) {
  const tips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP];
  const palm = palmSize(lm);
  if (palm < 1) return false;

  let cx = 0;
  let cy = 0;
  for (const t of tips) {
    cx += px(lm, t).x;
    cy += px(lm, t).y;
  }
  const centre = { x: cx / tips.length, y: cy / tips.length };
  return tips.every((t) => dist(px(lm, t), centre) < palm * ratio);
}

export function indexMiddleTogether(lm) {
  return dist(px(lm, INDEX_TIP), px(lm, MIDDLE_TIP)) < palmSize(lm) * 0.22;
}

export function indexMiddleSpread(lm) {
  return dist(px(lm, INDEX_TIP), px(lm, MIDDLE_TIP)) > palmSize(lm) * 0.35;
}

export function indexMiddleCrossed(lm) {
  const palm = palmSize(lm);
  const indexTip = px(lm, INDEX_TIP);
  const middleTip = px(lm, MIDDLE_TIP);
  return (
    Math.abs(indexTip.y - middleTip.y) < palm * 0.15 &&
    dist(indexTip, middleTip) < palm * 0.3
  );
}

export function pointingHorizontal(lm) {
  const indexTip = px(lm, INDEX_TIP);
  const indexMcp = px(lm, INDEX_MCP);
  const wrist = px(lm, WRIST);
  const verticalSpan = Math.abs(indexTip.y - wrist.y);
  const horizontalSpan = Math.abs(indexTip.x - indexMcp.x);
  return horizontalSpan > verticalSpan * 1.1 && horizontalSpan > palmSize(lm) * 0.5;
}

export function thumbBetweenIndexMiddle(lm) {
  const thumb = px(lm, THUMB_TIP);
  const indexMcp = px(lm, INDEX_MCP);
  const middleMcp = px(lm, MIDDLE_MCP);
  const mid = { x: (indexMcp.x + middleMcp.x) / 2, y: (indexMcp.y + middleMcp.y) / 2 };
  return dist(thumb, mid) < palmSize(lm) * 0.35;
}
