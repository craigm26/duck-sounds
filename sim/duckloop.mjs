// The control loop, ported from DuckKit's golden-tested Swift.
// Observation layout, action scaling and the low-pass are NOT invented here:
// they come from duckkit-constants.json, exported from the package whose
// forward pass is proved against onnxruntime to 1e-4.
import fs from 'node:fs';

export const C = JSON.parse(fs.readFileSync(new URL('./duckkit-constants.json', import.meta.url), 'utf8'));
export const POLICY_JOINTS = C.jointNames.filter(n => n !== 'mouth');
export const HOME = C.homePose.filter((_, i) => C.jointNames[i] !== 'mouth');
export const LO = C.rangeLo.filter((_, i) => C.jointNames[i] !== 'mouth');
export const HI = C.rangeHi.filter((_, i) => C.jointNames[i] !== 'mouth');
// Head joints are the four the runtime filters harder: neck_pitch..head_roll.
const HEAD = new Set(['neck_pitch', 'head_pitch', 'head_yaw', 'head_roll']);
export const ALPHA = POLICY_JOINTS.map(n => (HEAD.has(n) ? C.alphaHead : C.alphaLegs));

/** world -z expressed in the trunk frame, from the free joint quaternion. */
export function projectedGravity(q) {
  const [w, x, y, z] = q;
  // R^T * (0,0,-1)
  return [
    -(2 * (x * z - w * y)),
    -(2 * (y * z + w * x)),
    -(1 - 2 * (x * x + y * y)),
  ];
}

/** The 61-float observation, in DuckKit's verified order. */
export function buildObs(gyro, grav, jpos, jvel, lastAction, cmd) {
  const o = new Float32Array(61);
  let i = 0;
  for (let k = 0; k < 3; k++) o[i++] = gyro[k];
  for (let k = 0; k < 3; k++) o[i++] = grav[k];
  for (let k = 0; k < 14; k++) o[i++] = jpos[k] - HOME[k];
  for (let k = 0; k < 14; k++) o[i++] = jvel[k];
  for (let k = 0; k < 14; k++) o[i++] = lastAction[k];
  for (let k = 0; k < 13; k++) o[i++] = cmd[k];
  return o;
}

/** action -> joint targets: scale from home, low-pass, clamp to travel. */
export function gaitTargets(action, previous) {
  const scaled = new Array(14);
  for (let k = 0; k < 14; k++) scaled[k] = HOME[k] + C.actionScale * action[k];
  const filtered = new Array(14);
  for (let k = 0; k < 14; k++) {
    filtered[k] = previous ? previous[k] + ALPHA[k] * (scaled[k] - previous[k]) : scaled[k];
  }
  return filtered.map((v, k) => Math.min(Math.max(v, LO[k]), HI[k]));
}

/** The 13-value command block. */
export function command({ vx = 0, vy = 0, vyaw = 0, head = [0, 0, 0, 0], bodyZ = 0, bodyRoll = 0, bodyPitch = 0 } = {}) {
  return [vx, vy, vyaw, head[0], head[1], head[2], head[3], 0, 0, bodyZ, bodyRoll, bodyPitch, 0];
}
