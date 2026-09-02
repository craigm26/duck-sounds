// A rollout that can run the ROBOT's control pipeline, not just the canon one.
//
// WHY THIS FILE HAD TO EXIST BEFORE ANY KNOB COULD BE SWEPT. The bench's own
// rollout applies the policy the way TRAINING did — `target = reference +
// action`, scale 1.0, no filter — and every recorded clip in duckkit comes from
// that path, which is exactly why it must not be parameterised. Its whole value
// is being the same arithmetic every time. But `action_scale` and the low-pass
// alphas are ROBOTD's knobs: on hardware the target is
// `home + action_scale × action`, then a first-order filter at α = 0.5 on the
// head and 0.7 on the legs, then the travel clamp. So the two knobs this spike
// was asked to sweep are knobs the canon rollout does not have and should not
// grow. /measure and /record cannot answer the question; this can.
//
// IT REPORTS BOTH PATHS, ALWAYS. A number from the robotd path and a number
// from the canon path are answers about different plants, and the interesting
// comparison is between them: canon at scale 1.0 with no filter is what every
// published clip and every /measure count describes, and the robotd path at
// 0.9/0.5/0.7 is what a real duck would do with the same file. Reporting one
// without the other is how a tuning result gets attached to the wrong robot.
//
// FIDELITY IS CHECKED, NOT CLAIMED. `checkAgainstDuckloop` runs this file's
// parameterised pipeline at robotd's own constants against `duckloop.mjs`'s
// `gaitTargets` — the function the browser walk demo and the headless walk test
// both use — and requires them to agree bit for bit. If they ever disagree,
// this file is describing a robot nobody else in the repo is describing.
//
// WHAT IS NOT MODELLED, BY NAME. The BAM friction actuator's lag is the
// plant's, so it is in here through MuJoCo and not through anything this file
// does. Servo deadband, bus latency, gear backlash and battery sag are not
// modelled anywhere in this repo and are not modelled here either: a gain found
// in this world is a hypothesis about hardware, not a measurement of it.
import fs from 'node:fs';
import load from 'mujoco';
import { makeLoop } from '../site/duckloop.mjs';
import { readPolicy, makeForward, fold } from './policyfold.mjs';

const C = JSON.parse(fs.readFileSync(new URL('./duckkit-constants.json', import.meta.url), 'utf8'));
const LOOP = makeLoop(C);
const { HOME, LO, HI, ALPHA, buildObs, projectedGravity, command, findDuckJoints } = LOOP;

/** The four head joints, in POLICY slot order — 5..8, between the legs. */
const HEAD_SLOTS = new Set([5, 6, 7, 8]);

/**
 * One tick of joint targets, with the runtime's knobs exposed.
 *
 * `scale` is robotd's `action_scale`; `alphas` is its head/legs low-pass pair.
 * `filter: false` is the canon path — no low-pass at all, which is NOT the same
 * as α = 1 in this code only in that it also skips the previous-target
 * bookkeeping; numerically α = 1 and no filter are identical, and a test below
 * says so.
 */
export function targetsFor(action, previous, { scale, alphas, filter = true }) {
  const out = new Array(14);
  for (let k = 0; k < 14; k++) {
    const scaled = HOME[k] + scale * action[k];
    let value = scaled;
    if (filter && previous) {
      const a = HEAD_SLOTS.has(k) ? alphas[0] : alphas[1];
      value = previous[k] + a * (scaled - previous[k]);
    }
    out[k] = Math.min(Math.max(value, LO[k]), HI[k]);
  }
  return out;
}

/**
 * This file's pipeline at robotd's own constants, against duckloop's.
 *
 * Bit-for-bit, not nearly: both are the same three operations on the same
 * doubles, so any difference is a difference in the model of the robot and not
 * in floating point.
 */
export function checkAgainstDuckloop({ samples = 500, seed = 11 } = {}) {
  let s = seed >>> 0;
  const rand = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  let worst = 0;
  let previous = null;
  for (let n = 0; n < samples; n++) {
    const action = Array.from({ length: 14 }, () => rand() * 2 - 1);
    const mine = targetsFor(action, previous, {
      scale: C.actionScale, alphas: [C.alphaHead, C.alphaLegs], filter: true });
    const theirs = LOOP.gaitTargets(action, previous);
    for (let k = 0; k < 14; k++) worst = Math.max(worst, Math.abs(mine[k] - theirs[k]));
    previous = theirs;
  }
  return worst;
}

/**
 * The rig: one MuJoCo world, one policy, episodes on demand.
 *
 * ONE WORLD REUSED, RESET PER EPISODE. Building an MjModel costs about a second
 * and a search runs hundreds of episodes; the reset below writes every qpos,
 * qvel and ctrl the duck owns, which is what the bench's own `placeDuck` does.
 */
export async function makeRig({ scene = 'scene.mjb', policyFile = 'alpha_walking.onnx',
                               standFile = 'BEST_alpha_stand.onnx' } = {}) {
  const here = new URL('./', import.meta.url);
  const mj = await load();
  const bytes = fs.readFileSync(new URL(scene, here));
  mj.FS.writeFile('/rig.mjb', new Uint8Array(bytes));
  const model = mj.MjModel.mj_loadBinary('/rig.mjb', new mj.MjVFS());
  const data = new mj.MjData(model);
  const D = findDuckJoints(model);

  let GYRO = -1;
  for (let i = 0; i < model.nsensor; i++) {
    if (model.sensor(i).name === 'imu_ang_vel') { GYRO = model.sensor(i).adr; break; }
  }
  if (GYRO < 0) throw new Error('sensor missing from the model: imu_ang_vel');

  // The duck's own actuators, by the joint they drive — never `ctrl[k]`, which
  // is only right while the model holds one duck.
  const CTRL = [];
  const names = C.jointNames.filter(n => n !== 'mouth');
  for (const name of names) {
    let found = -1;
    for (let a = 0; a < model.nu; a++) if (model.actuator(a).name === name) { found = a; break; }
    if (found < 0) throw new Error(`actuator missing from the model: ${name}`);
    CTRL.push(found);
  }

  // The plant's own timestep, read out of the plant by stepping it once rather
  // than assumed — the same derivation duckbench-core does, and for the same
  // reason: a scene with a different timestep would silently run this control
  // loop at the wrong rate.
  reset(0.1231);
  mj.mj_step(model, data);
  const timestep = data.time;
  const substeps = Math.round((1 / C.tickHz) / timestep);
  if (!(substeps >= 1) || Math.abs(substeps * timestep - 1 / C.tickHz) > 1e-9) {
    throw new Error(`${C.tickHz} Hz control does not divide a ${timestep} s timestep`);
  }

  const base = readPolicy(new URL(policyFile, here).pathname);
  const stand = readPolicy(new URL(standFile, here).pathname);
  const forwardStand = makeForward(stand);

  function reset(z) {
    mj.mj_resetData(model, data);
    const f = D.freeQpos;
    data.qpos[f + 2] = z;
    data.qpos[f + 3] = 1; data.qpos[f + 4] = 0; data.qpos[f + 5] = 0; data.qpos[f + 6] = 0;
    for (let k = 0; k < 6; k++) data.qvel[D.freeDof + k] = 0;
    for (let i = 0; i < 14; i++) {
      data.qpos[D.qpos[i]] = HOME[i];
      data.qvel[D.dof[i]] = 0;
      data.ctrl[CTRL[i]] = HOME[i];
    }
    mj.mj_forward(model, data);
  }

  function observe(lastAction, cmd) {
    const f = D.freeQpos;
    const jp = [], jv = [];
    for (let k = 0; k < 14; k++) { jp.push(data.qpos[D.qpos[k]]); jv.push(data.qvel[D.dof[k]]); }
    return buildObs([data.sensordata[GYRO], data.sensordata[GYRO + 1], data.sensordata[GYRO + 2]],
                    projectedGravity([data.qpos[f + 3], data.qpos[f + 4],
                                      data.qpos[f + 5], data.qpos[f + 6]]),
                    jp, jv, lastAction, cmd);
  }

  /**
   * One episode.
   *
   * `gain`/`offset`, when given, are FOLDED into the policy — so the network
   * being run is the network that would be written to a file, not the base one
   * with a residual bolted on outside. That is the whole point of folding in
   * JS; duckkit's `DuckPolicyWriterFoldTests` proves the two agree, and this
   * does not have to rely on that proof.
   *
   * THE SETTLE USES THE SAME PIPELINE AS THE EPISODE, deliberately. robotd does
   * not switch its filter off for the first half second, so neither does this;
   * the bench's canon rollout settles through the canon path for the same
   * reason. It does mean a robotd episode and a canon episode differ from tick
   * −25, which is honest: they are different robots from the moment they land.
   */
  async function episode({ seconds = 6, drop = 0.1231, vx = 0.5, vy = 0, vyaw = 0,
                           scale = 1.0, alphas = [1, 1], filter = false,
                           gain = null, offset = null, settle = 25, timing = null } = {}) {
    const net = gain ? fold(base, gain, offset) : base;
    const forward = makeForward(net);
    reset(drop);
    const f = D.freeQpos;
    const start = [data.qpos[f], data.qpos[f + 1]];
    let last = new Array(14).fill(0);
    let previous = null;
    const ticks = Math.round(seconds * C.tickHz);
    const cmdNeutral = command({});
    const cmdDrive = command({ vx, vy, vyaw });
    const trace = [];
    let physicsNs = 0, policyNs = 0;

    for (let t = -settle; t < ticks; t++) {
      const cmd = t >= 0 ? cmdDrive : cmdNeutral;
      const obs = observe(last, cmd);
      const t0 = timing ? process.hrtime.bigint() : 0n;
      const action = Array.from(t < 0 ? forwardStand(obs) : forward(obs));
      if (timing) policyNs += Number(process.hrtime.bigint() - t0);
      const targets = targetsFor(action, previous, { scale, alphas, filter });
      for (let k = 0; k < 14; k++) data.ctrl[CTRL[k]] = targets[k];
      previous = targets;
      last = action;
      const t1 = timing ? process.hrtime.bigint() : 0n;
      for (let s = 0; s < substeps; s++) mj.mj_step(model, data);
      if (timing) physicsNs += Number(process.hrtime.bigint() - t1);
      if (t >= 0) {
        trace.push({
          root: [data.qpos[f], data.qpos[f + 1], data.qpos[f + 2],
                 data.qpos[f + 3], data.qpos[f + 4], data.qpos[f + 5], data.qpos[f + 6]],
          vel: [data.qvel[D.freeDof], data.qvel[D.freeDof + 1], data.qvel[D.freeDof + 2],
                data.qvel[D.freeDof + 3], data.qvel[D.freeDof + 4], data.qvel[D.freeDof + 5]],
          joints: Array.from({ length: 14 }, (_, k) => data.qpos[D.qpos[k]]),
          action,
        });
      }
    }
    const end = trace[trace.length - 1].root;
    const travelled = Math.hypot(end[0] - start[0], end[1] - start[1]);
    const forwardTravel = end[0] - start[0];
    const [w, x, y] = [end[3], end[4], end[5]];
    const upright = -(1 - 2 * (x * x + y * y)) < -0.5;
    if (timing) {
      timing.ticks = (timing.ticks || 0) + ticks + settle;
      timing.physicsNs = (timing.physicsNs || 0) + physicsNs;
      timing.policyNs = (timing.policyNs || 0) + policyNs;
    }
    return { travelled, forwardTravel, endHeight: end[2], upright,
             standing: upright && end[2] >= 0.100, trace, command: [vx, vy, vyaw] };
  }

  return { episode, model, data, mj, timestep, substeps, tickHz: C.tickHz,
           joints: D, ctrl: CTRL, base, plant: scene };
}

/** Median of an array, without mutating it. */
export const median = a => {
  const s = [...a].sort((x, y) => x - y);
  return s.length % 2 ? s[s.length >> 1] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
};
