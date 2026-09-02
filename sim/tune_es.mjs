// SPIKE 2 — a (1+λ)-ES over a per-joint gain and trim, folded into the policy,
// and the only question it exists to answer: does the curve move beyond noise?
//
// WHAT IS BEING SEARCHED. Twenty numbers: a multiplier in [0.7, 1.3] and an
// additive trim in [−0.05, 0.05] rad on each of the TEN LEG SLOTS. The four
// head slots are held at gain 1, offset 0 and are not searched — the head is
// driven by a pose command that rides in the observation, so trimming it is
// trimming against a command rather than against a gait, and the velocity
// config's own `pose` term deliberately excludes those joints for the same
// reason. Fourteen slots exist; ten of them are the search space.
//
// THE CANDIDATE IS FOLDED, NOT BOLTED ON. Every candidate becomes an actual
// network — `diag(gain)·W` and `gain ⊙ b + offset` on the last 14×128 Gemm —
// so the thing being scored is the thing that would be written to a file, and
// the previous-action block the network reads back on the next tick carries the
// folded output rather than a raw one. duckkit's `DuckPolicyWriterFoldTests`
// proves the two are the same to 1e-5 at every state of a 200-tick closed run;
// this file does not have to rely on that, and does not.
//
// THE REWARD IS POLLEN'S, MINUS WHAT THIS PLANT CANNOT ANSWER. Six terms of
// `microduck_velocity_env_cfg`, at the weights `RunMetrics` reads out of that
// config, computed per tick and averaged over the episode. Everything else in
// that config is refused BY NAME below with the reason, because a shorter list
// is not a better one.
//
// AND THE ANSWER IS GUARDED AGAINST FARMING. Five of the six terms can be
// maximised by a duck that stands still and does nothing: upright is perfect,
// pose is perfect, action_rate is zero, body_ang_vel is zero, and only the
// tracking terms object. That is not a hypothetical — `PolicyBlend` in
// StudioKit has the receipt: a 75% blend scored a perfect 16 of 16 on "ends
// standing" while travelling two millimetres. So its `wentInertRatherThanFalling`
// check is reused verbatim here as a hard rejection, travel is printed beside
// every reward, and the winner is re-scored on drop heights the search never
// saw.
import { makeRig, median } from './rollout_robotd.mjs';
import fs from 'node:fs';

const C = JSON.parse(fs.readFileSync(new URL('./duckkit-constants.json', import.meta.url), 'utf8'));
const POLICY_JOINTS = C.jointNames.filter(n => n !== 'mouth');
const HEAD_SLOTS = [5, 6, 7, 8];
const LEG_SLOTS = [...Array(14).keys()].filter(k => !HEAD_SLOTS.includes(k));

// ── the reward ───────────────────────────────────────────────────────────────

/**
 * Terms of `microduck_velocity_env_cfg.py` that this plant CANNOT answer, with
 * the reason, printed every run. `RunMetrics.Task.unevaluable` says the same
 * thing about a recording; this says it about `scene.mjb`, which is a different
 * claim and in one case a different reason.
 */
const REFUSED = [
  ['air_time', 'no foot-contact sensor in scene.mjb (its six sensors are orientation, '
             + 'angular-velocity, imu_ang_vel, imu_lin_vel, imu_accel, root_angmom)'],
  ['foot_clearance', 'reads the foot sites against the contact sensor; the sensor is not there'],
  ['foot_swing_height', 'same sensor'],
  ['foot_slip', 'same sensor'],
  ['self_collisions', 'no collision sensor in this plant'],
  ['dof_pos_limits', 'scores against soft limits — a fraction of travel that neither duckkit '
                   + 'nor this repo ships, so the fraction would have to be invented'],
  ['angular_momentum', 'the plant DOES carry root_angmom, so this one is refused for a '
                     + 'different reason: RunMetrics does not carry its weight, and picking '
                     + 'one would be inventing a reward rather than reading Pollen\'s'],
];

/** `|projected gravity xy|²` — zero upright, 1 on its side. RunMetrics's own. */
function gravityXYSquared(q) {
  const [w, x, y, z] = q;
  const gx = -2 * (x * z + w * y), gy = -2 * (y * z - w * x);
  return gx * gx + gy * gy;
}

/** Rotate a body-frame vector into the world. RunMetrics's own. */
function rotate(q, v) {
  const [w, x, y, z] = q;
  const tx = 2 * (y * v[2] - z * v[1]);
  const ty = 2 * (z * v[0] - x * v[2]);
  const tz = 2 * (x * v[1] - y * v[0]);
  return [v[0] + w * tx + (y * tz - z * ty),
          v[1] + w * ty + (z * tx - x * tz),
          v[2] + w * tz + (x * ty - y * tx)];
}

/** Rotate a WORLD vector into the trunk's frame — the inverse of the above. */
function unrotate(q, v) { return rotate([q[0], -q[1], -q[2], -q[3]], v); }

/** `variable_posture`'s per-joint tolerance. RunMetrics's `legStd`, verbatim. */
function legStd(name, standing) {
  if (name.includes('hip_yaw')) return standing ? 0.1 : 0.3;
  if (name.includes('hip_roll')) return 0.05;
  if (name.includes('hip_pitch')) return standing ? 0.15 : 0.4;
  if (name.includes('knee')) return standing ? 0.15 : 0.4;
  if (name.includes('ankle')) return standing ? 0.1 : 0.25;
  return null;
}

const HOME14 = POLICY_JOINTS.map(n => C.homePose[C.jointNames.indexOf(n)]);

/**
 * The six evaluable terms, per tick, averaged over the episode.
 *
 * WHERE EACH VELOCITY LIVES, because getting this wrong is silent. MuJoCo's
 * free joint keeps LINEAR velocity in the world frame and ANGULAR velocity in
 * the body's own — and mjlab's terms want `base_lin_vel` and `base_ang_vel` in
 * the BODY frame for tracking, and the WORLD frame for `body_ang_vel`. So the
 * linear velocity is rotated into the trunk here and the angular one is rotated
 * out of it, and neither is used raw.
 */
function reward(trace, command) {
  const [cvx, cvy, cvyaw] = command;
  const speed = Math.hypot(cvx, cvy) + Math.abs(cvyaw);
  const standing = speed < 0.01;
  let upright = 0, linear = 0, angular = 0, pose = 0, angVel = 0, rate = 0;
  for (let i = 0; i < trace.length; i++) {
    const f = trace[i];
    const q = [f.root[3], f.root[4], f.root[5], f.root[6]];
    upright += Math.exp(-gravityXYSquared(q) / 0.05);

    const v = unrotate(q, [f.vel[0], f.vel[1], f.vel[2]]);   // world -> trunk
    const w = [f.vel[3], f.vel[4], f.vel[5]];                // already trunk frame
    const ex = cvx - v[0], ey = cvy - v[1];
    linear += Math.exp(-(ex * ex + ey * ey + v[2] * v[2]) / 0.1);
    const ez = cvyaw - w[2];
    angular += Math.exp(-(ez * ez + w[0] * w[0] + w[1] * w[1]) / 0.5);

    const world = rotate(q, w);                              // trunk -> world
    angVel += world[0] * world[0] + world[1] * world[1];

    let sum = 0, count = 0;
    for (let k = 0; k < 14; k++) {
      const std = legStd(POLICY_JOINTS[k], standing);
      if (std === null) continue;
      const d = f.joints[k] - HOME14[k];
      sum += (d * d) / (std * std);
      count++;
    }
    pose += count ? Math.exp(-sum / count) : 0;

    if (i > 0) {
      const a = f.action, b = trace[i - 1].action;
      for (let k = 0; k < 14; k++) { const d = a[k] - b[k]; rate += d * d; }
    }
  }
  const n = trace.length;
  const terms = {
    upright: upright / n,
    track_linear_velocity: linear / n,
    track_angular_velocity: angular / n,
    pose: pose / n,
    body_ang_vel: angVel / n,
    action_rate_l2: rate / Math.max(n - 1, 1),
  };
  // The weights are the config's, as RunMetrics reads them: action_rate_l2 is
  // the RAMP END (−1.0), because that is what the trained policy lived under.
  const total = 2.0 * terms.upright
              + 2.0 * terms.track_linear_velocity
              + 2.0 * terms.track_angular_velocity
              + 1.0 * terms.pose
              - 0.05 * terms.body_ang_vel
              - 1.0 * terms.action_rate_l2;
  return { total, terms };
}

// ── the search ───────────────────────────────────────────────────────────────

/** mulberry32 — a seeded generator, so a run is reproducible by its seed. */
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const gauss = r => {
  const u = Math.max(r(), 1e-12), v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};

const GAIN_LO = 0.7, GAIN_HI = 1.3, OFF = 0.05;
const GAIN_SIGMA = 0.05, OFF_SIGMA = 0.008;
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

/** A candidate is 14 gains and 14 offsets; head slots are pinned. */
function identity() {
  return { gain: new Array(14).fill(1), offset: new Array(14).fill(0) };
}
function mutate(parent, r) {
  const child = { gain: [...parent.gain], offset: [...parent.offset] };
  for (const k of LEG_SLOTS) {
    child.gain[k] = clamp(child.gain[k] + GAIN_SIGMA * gauss(r), GAIN_LO, GAIN_HI);
    child.offset[k] = clamp(child.offset[k] + OFF_SIGMA * gauss(r), -OFF, OFF);
  }
  return child;
}

// ── the run ──────────────────────────────────────────────────────────────────

const SECONDS = 6;
const LAMBDA = 6;
const GENERATIONS = 15;
const SEEDS = [1, 2];
// The pipeline being tuned is the ROBOT's, because that is what a tuned file
// would be shipped into: action_scale 0.9 (robotd's for walking) and the
// low-pass at 0.5 / 0.7. action_scale is NOT part of the search — it stays a
// robotd config key, which is the same rule the fold itself obeys.
const PIPELINE = { scale: 0.9, alphas: [0.5, 0.7], filter: true };
// Three drop heights to score on, eight held out. Spike 1 measured up to
// 273 mm of travel spread across this range for one unchanged setting, so a
// single drop is not a measurement of a policy.
const TRAIN_DROPS = [0.1210, 0.1250, 0.1290];
const HELD_OUT = [0.1200, 0.1215, 0.1230, 0.1245, 0.1260, 0.1275, 0.1285, 0.1300];

const rig = await makeRig();
const timing = {};

async function score(candidate, drops) {
  const rewards = [], travels = [];
  let standing = 0;
  for (const drop of drops) {
    const run = await rig.episode({
      seconds: SECONDS, drop, vx: 0.5, ...PIPELINE, timing,
      gain: candidate.gain, offset: candidate.offset });
    rewards.push(reward(run.trace, run.command).total);
    travels.push(run.travelled);
    if (run.standing) standing++;
  }
  return { reward: median(rewards), travel: median(travels), standing, of: drops.length,
           rewards, travels };
}

console.log('SPIKE 2 — (1+6)-ES over 10 leg gains and 10 leg trims, folded per candidate');
console.log(`plant ${rig.plant}, ${rig.timestep} s timestep, ${rig.substeps} substeps `
          + `per ${rig.tickHz} Hz tick`);
console.log(`pipeline: action_scale ${PIPELINE.scale}, low-pass ${PIPELINE.alphas[0]}/`
          + `${PIPELINE.alphas[1]} — robotd's, and action_scale is not searched`);
console.log(`episodes: ${SECONDS} s at ${rig.tickHz} Hz, scored on drops `
          + `${TRAIN_DROPS.join(', ')} m`);
console.log('');
console.log('reward = 2.0·upright + 2.0·track_linear_velocity + 2.0·track_angular_velocity');
console.log('       + 1.0·pose − 0.05·body_ang_vel − 1.0·action_rate_l2');
console.log('         (microduck_velocity_env_cfg weights, as RunMetrics reads them;');
console.log('          action_rate_l2 at its RAMP END, −1.0, which is what the policy trained under)');
console.log('');
console.log('refused, by name:');
for (const [name, why] of REFUSED) console.log(`  ${name.padEnd(18)} ${why}`);
console.log('');

// The yardstick: the unmodified network, on both sets of drops.
const baseline = await score(identity(), TRAIN_DROPS);
const baselineHeld = await score(identity(), HELD_OUT);
console.log(`baseline (gain 1, offset 0): reward ${baseline.reward.toFixed(4)}, `
          + `travel ${(baseline.travel * 1000).toFixed(0)} mm, standing `
          + `${baseline.standing}/${baseline.of}`);
console.log(`baseline on the 8 held-out drops: reward ${baselineHeld.reward.toFixed(4)}, `
          + `travel ${(baselineHeld.travel * 1000).toFixed(0)} mm, standing `
          + `${baselineHeld.standing}/${baselineHeld.of}`);
// THE NOISE FLOOR, MEASURED RATHER THAN ASSUMED: the spread of the SAME
// candidate's reward across drop heights. A generation-to-generation gain
// smaller than this is not a gain.
const floor = Math.max(...baselineHeld.rewards) - Math.min(...baselineHeld.rewards);
console.log(`noise floor — the unchanged network's own reward spread across `
          + `${HELD_OUT.length} drops: ${floor.toFixed(4)}`);
console.log('');

/**
 * `PolicyBlend.wentInertRatherThanFalling`, reused as the farming guard: it
 * stayed on its feet and stopped doing the thing. Rejected outright rather than
 * scored, because five of the six reward terms pay for standing still.
 */
function wentInert(result, yardstick) {
  return yardstick >= 0.05
      && result.standing > result.of / 2
      && result.travel < yardstick * 0.25;
}

const curves = [];
for (const seed of SEEDS) {
  const r = rng(seed);
  let parent = identity();
  let parentScore = baseline;
  const curve = [];
  const started = Date.now();
  for (let g = 1; g <= GENERATIONS; g++) {
    let bestChild = null, bestScore = null;
    let inert = 0;
    for (let c = 0; c < LAMBDA; c++) {
      const child = mutate(parent, r);
      const s = await score(child, TRAIN_DROPS);
      if (wentInert(s, baseline.travel)) { inert++; continue; }
      if (!bestScore || s.reward > bestScore.reward) { bestChild = child; bestScore = s; }
    }
    if (bestScore && bestScore.reward > parentScore.reward) {
      parent = bestChild; parentScore = bestScore;
    }
    curve.push({ g, reward: parentScore.reward, travel: parentScore.travel, inert });
    console.log(`seed ${seed} gen ${String(g).padStart(2)}  reward `
              + `${parentScore.reward.toFixed(4)}  travel `
              + `${(parentScore.travel * 1000).toFixed(0)} mm  `
              + `(${inert} of ${LAMBDA} children rejected as inert)`);
  }
  const held = await score(parent, HELD_OUT);
  const wall = (Date.now() - started) / 1000;
  curves.push({ seed, curve, parent, parentScore, held, wall });
  console.log(`seed ${seed} winner on the 8 HELD-OUT drops: reward ${held.reward.toFixed(4)} `
            + `(baseline ${baselineHeld.reward.toFixed(4)}), travel `
            + `${(held.travel * 1000).toFixed(0)} mm (baseline `
            + `${(baselineHeld.travel * 1000).toFixed(0)} mm), standing ${held.standing}/${held.of}`);
  console.log(`seed ${seed} wall time ${wall.toFixed(0)} s`);
  console.log('');
}

// ── the verdict ──────────────────────────────────────────────────────────────

console.log('gen   ' + SEEDS.map(s => `seed ${s} reward   travel`).join('   '));
for (let g = 0; g < GENERATIONS; g++) {
  console.log(`${String(g + 1).padStart(3)}   ` + curves.map(c =>
    `${c.curve[g].reward.toFixed(4)}   ${(c.curve[g].travel * 1000).toFixed(0).padStart(4)} mm`
  ).join('      '));
}
console.log('');

let moved = 0;
for (const c of curves) {
  const gainOnTrain = c.parentScore.reward - baseline.reward;
  const gainOnHeld = c.held.reward - baselineHeld.reward;
  console.log(`seed ${c.seed}: +${gainOnTrain.toFixed(4)} on the drops it searched, `
            + `${gainOnHeld >= 0 ? '+' : ''}${gainOnHeld.toFixed(4)} on the eight it did not `
            + `(noise floor ${floor.toFixed(4)})`);
  if (gainOnHeld > floor) moved++;
}
console.log('');
console.log(moved === SEEDS.length
  ? `VERDICT: the curve moved beyond noise on ${moved}/${SEEDS.length} seeds, and the gain `
    + 'survived on drop heights the search never saw.'
  : moved > 0
    ? `VERDICT: ${moved} of ${SEEDS.length} seeds beat the noise floor on held-out drops. `
      + 'One seed is not a result.'
    : 'VERDICT: no seed beat the noise floor on held-out drops. What the search found on the '
      + 'drops it scored did not survive contact with drops it did not.');
console.log('');

const perTick = ns => (ns / 1e6 / timing.ticks).toFixed(3);
console.log(`${timing.ticks} control ticks measured: physics ${perTick(timing.physicsNs)} ms/tick, `
          + `policyforward ${perTick(timing.policyNs)} ms/tick (Pi 5, Node ${process.versions.node})`);
console.log(`one 6 s episode is ${SECONDS * rig.tickHz + 25} ticks; real time for that is `
          + `${(SECONDS * 1000).toFixed(0)} ms, and this Pi takes about `
          + `${(((timing.physicsNs + timing.policyNs) / 1e6 / timing.ticks) * (SECONDS * rig.tickHz + 25)).toFixed(0)} ms.`);
console.log('THE PHONE IS UNMEASURED. Nothing in this run touched an iPhone. What can be said is '
          + 'the split: policyforward is the part a phone would also have to do, and it is '
          + `${(100 * timing.policyNs / (timing.physicsNs + timing.policyNs)).toFixed(0)}% of the `
          + 'per-tick cost here, in plain JS on a Pi 5. The physics half is MuJoCo WASM, which is '
          + 'not what a phone would run. Deriving a phone number from these two would be '
          + 'arithmetic on a machine nobody benchmarked.');
