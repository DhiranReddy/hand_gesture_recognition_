/**
 * Landmark -> feature vector, ported from src/landmark_features.py.
 *
 * The layout must stay byte-for-byte compatible with the Python trainer or the
 * exported weights are meaningless: 63 wrist-centred coordinates followed by 10
 * pairwise fingertip distances.
 *
 * One inherited quirk is load-bearing: x and y are scaled to pixels while z is
 * left in MediaPipe's normalized units. It is dimensionally odd, but the model
 * was trained that way, so the browser has to reproduce it.
 */

const WRIST = 0;
const MIDDLE_MCP = 9;
const TIP_INDICES = [4, 8, 12, 16, 20];

export const FEATURE_LENGTH = 73;

/**
 * @param {Array<{x:number,y:number,z:number}>} landmarks 21 normalized landmarks
 * @param {string} handedness "Left" or "Right"
 * @param {number} width  frame width in pixels
 * @param {number} height frame height in pixels
 * @returns {Float32Array} length-73 feature vector
 */
export function landmarksToFeatures(landmarks, handedness, width, height) {
  const pts = new Float32Array(63);
  for (let i = 0; i < 21; i += 1) {
    pts[i * 3] = landmarks[i].x * width;
    pts[i * 3 + 1] = landmarks[i].y * height;
    pts[i * 3 + 2] = landmarks[i].z;
  }

  const wx = pts[WRIST * 3];
  const wy = pts[WRIST * 3 + 1];
  const wz = pts[WRIST * 3 + 2];

  const dx = pts[MIDDLE_MCP * 3] - wx;
  const dy = pts[MIDDLE_MCP * 3 + 1] - wy;
  const dz = pts[MIDDLE_MCP * 3 + 2] - wz;
  let palm = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (palm < 1e-6) palm = 1.0;

  const mirror = handedness === "Left" ? -1 : 1;

  const out = new Float32Array(FEATURE_LENGTH);
  for (let i = 0; i < 21; i += 1) {
    out[i * 3] = (((pts[i * 3] - wx) / palm) * mirror);
    out[i * 3 + 1] = (pts[i * 3 + 1] - wy) / palm;
    out[i * 3 + 2] = (pts[i * 3 + 2] - wz) / palm;
  }

  let cursor = 63;
  for (let i = 0; i < TIP_INDICES.length; i += 1) {
    for (let j = i + 1; j < TIP_INDICES.length; j += 1) {
      const a = TIP_INDICES[i] * 3;
      const b = TIP_INDICES[j] * 3;
      const ex = out[a] - out[b];
      const ey = out[a + 1] - out[b + 1];
      const ez = out[a + 2] - out[b + 2];
      out[cursor] = Math.sqrt(ex * ex + ey * ey + ez * ez);
      cursor += 1;
    }
  }

  return out;
}
