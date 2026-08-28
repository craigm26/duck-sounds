// The control loop, ported from DuckKit's golden-tested Swift.
//
// The observation layout, the action scaling and the low-pass are NOT invented
// here: they arrive as `constants`, exported from the package whose forward
// pass is proved against onnxruntime to 1e-4. Same file runs in Node (for the
// headless walk test) and in the browser.
export function makeLoop(C) {
  const POLICY = C.jointNames.map((n, i) => i).filter(i => C.jointNames[i] !== 'mouth');
  const pick = a => POLICY.map(i => a[i]);
  const HOME = pick(C.homePose), LO = pick(C.rangeLo), HI = pick(C.rangeHi);
  const HEAD = new Set(['neck_pitch', 'head_pitch', 'head_yaw', 'head_roll']);
  const ALPHA = POLICY.map(i => (HEAD.has(C.jointNames[i]) ? C.alphaHead : C.alphaLegs));

  /** world −z expressed in the trunk frame, from the free-joint quaternion. */
  function projectedGravity([w, x, y, z]) {
    return [-(2 * (x * z - w * y)), -(2 * (y * z + w * x)), -(1 - 2 * (x * x + y * y))];
  }

  /** The 61-float observation, in DuckKit's verified order. */
  function buildObs(gyro, grav, jpos, jvel, lastAction, cmd) {
    const o = new Float32Array(61);
    let i = 0;
    for (let k = 0; k < 3; k++) o[i++] = gyro[k];
    for (let k = 0; k < 3; k++) o[i++] = grav[k];
    for (let k = 0; k < 14; k++) o[i++] = jpos[k] - HOME[k];
    for (let k = 0; k < 14; k++) o[i++] = jvel[k];
    for (let k = 0; k < 14; k++) o[i++] = lastAction[k];
    for (let k = 0; k < 13; k++) o[i++] = cmd[k];
    if (i !== 61) throw new Error('observation layout drifted from 61');
    return o;
  }

  /** action → joint targets: scale from home, low-pass, clamp to travel. */
  function gaitTargets(action, previous) {
    const out = new Array(14);
    for (let k = 0; k < 14; k++) {
      const scaled = HOME[k] + C.actionScale * action[k];
      const filtered = previous ? previous[k] + ALPHA[k] * (scaled - previous[k]) : scaled;
      out[k] = Math.min(Math.max(filtered, LO[k]), HI[k]);
    }
    return out;
  }

  /** The 13-value command block. */
  function command({ vx = 0, vy = 0, vyaw = 0, head = [0, 0, 0, 0],
                     bodyZ = 0, bodyRoll = 0, bodyPitch = 0 } = {}) {
    return [vx, vy, vyaw, head[0], head[1], head[2], head[3], 0, 0, bodyZ, bodyRoll, bodyPitch, 0];
  }

  return { C, HOME, LO, HI, ALPHA, projectedGravity, buildObs, gaitTargets, command };
}
