// The four pieces of arithmetic that BOTH reward transcriptions need, in ONE
// copy.
//
// WHY THIS FILE EXISTS. `duckbench-core.mjs` transcribes the six terms of
// `microduck_velocity_env_cfg.py` for /tune; `chase_score.mjs` transcribes the
// nine computable terms of `microduck_ball_kick_env_cfg.py` for /chase. The two
// configs are different rewards, but three of their terms are the SAME term —
// `upright`, `body_ang_vel` and `action_rate_l2` — and the first two are built
// out of exactly these functions. A second copy of `gravityXYSquared` in the
// ball module would be a second transcription of one formula, which is how two
// benches come to report the same term under the same name with different
// numbers and both look plausible. There is one copy, and both import it.
//
// NOTHING HERE KNOWS WHICH CONFIG IS ASKING. These are quaternion identities
// and MuJoCo's free-joint convention; the weights, the variances and the term
// names live with the config that declares them.
//
// The comments below are the ones these functions carried in
// `duckbench-core.mjs`, moved with them rather than left behind.

/** `|projected gravity xy|²` — zero upright, 1 on its side. RunMetrics's own. */
export function gravityXYSquared([w, x, y, z]) {
  const gx = -2 * (x * z + w * y), gy = -2 * (y * z - w * x);
  return gx * gx + gy * gy;
}

/** Rotate a body-frame vector into the world. RunMetrics's own. */
export function rotate([w, x, y, z], v) {
  const tx = 2 * (y * v[2] - z * v[1]);
  const ty = 2 * (z * v[0] - x * v[2]);
  const tz = 2 * (x * v[1] - y * v[0]);
  return [v[0] + w * tx + (y * tz - z * ty),
          v[1] + w * ty + (z * tx - x * tz),
          v[2] + w * tz + (x * ty - y * tx)];
}

/** The inverse: a WORLD vector into the trunk's frame, by the conjugate. */
export function unrotate(q, v) { return rotate([q[0], -q[1], -q[2], -q[3]], v); }

/**
 * THE TRUNK'S TWIST IN ITS OWN FRAME, which is the frame the clip format
 * stores and the frame every tracking term is written against.
 *
 * MuJoCo keeps a free joint's LINEAR velocity in the world and its ANGULAR
 * velocity in the body, so exactly one of the two has to be rotated, and
 * rotating the wrong one produces a plausible number for a duck that is
 * walking due north and a wrong one for every other heading. `body_ang_vel`
 * then wants the WORLD-frame angular velocity, so it rotates back out — the
 * two terms genuinely live in different frames and mjlab reads them that way.
 */
export function twistOf(root, qvel) {
  const q = [root[3], root[4], root[5], root[6]];
  const linear = unrotate(q, [qvel[0], qvel[1], qvel[2]]);
  return [linear[0], linear[1], linear[2], qvel[3], qvel[4], qvel[5]];
}

/**
 * The yaw of a free joint's quaternion, in radians.
 *
 * ONE COPY BECAUSE IT IS READ TWICE AND MUST NOT DIFFER. `POST /ball` uses it
 * to place a ball at a bearing; `chase_score.mjs` uses the SAME yaw, at the
 * same instant, as Pollen's frozen `kick_dir` and as the axis `ballTravel_mm`
 * is projected onto. Two spellings of the same atan2 would put the ball on one
 * line and score the travel along another.
 */
export function yawOf([w, x, y, z]) {
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}
