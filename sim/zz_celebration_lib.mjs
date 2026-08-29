// zz_celebration_lib.mjs — verify an authored celebration track in physics.
//
// Modelled on measure_success.mjs's rollout path: settle 25 ticks on
// BEST_alpha_stand, then KEEP running BEST_alpha_stand (zero command) and add
// the track's smoothstep-interpolated offsets at blend 1.0 to the policy
// targets, clamped to travel. Domain randomisation is Pollen's own set
// (drop 0.12-0.13 m, foot friction x0.7-1.3, trunk COM +/-3 mm, one push
// +/-0.3 m/s in x and y). ONE DELIBERATE DIFFERENCE from measure_success: its
// push interval of 3-6 s falls entirely PAST the end of a 2.0-2.6 s clip, so
// no push would ever land. Here the push tick is drawn uniformly inside the
// motion (0.3 s .. end-0.3 s) so every rollout takes exactly one real shove.
//
// PASS per rollout = ends standing: projected gravity z < -0.5 AND trunk
// z >= 0.100, both AVERAGED over the final 15 ticks. Also tracked: the peak
// per-joint rate (|qvel| of the 14 policy joints, rad/s) over the whole
// recorded motion, across all rollouts.
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
import { clearStairs, findStairJoints } from '../site/stairs.js';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);
const DT = 1 / C.tickHz;

const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const model = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model), ADDR = findStairJoints(model);
let GYRO = 0;
for (let i = 0; i < model.nsensor; i++) {
  if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;
}
const STAND = 'BEST_alpha_stand.onnx';
const session = await ort.InferenceSession.create('./' + STAND);

function resetDuck(z) {
  mj.mj_resetData(model, data);
  clearStairs(data, ADDR);
  data.qpos[D.freeQpos + 2] = z;
  data.qpos[D.freeQpos + 3] = 1;
  for (let i = 0; i < 14; i++) { data.qpos[D.qpos[i]] = HOME[i]; data.ctrl[i] = HOME[i]; }
  mj.mj_forward(model, data);
}
const quat = () => [data.qpos[D.freeQpos + 3], data.qpos[D.freeQpos + 4],
                    data.qpos[D.freeQpos + 5], data.qpos[D.freeQpos + 6]];

/** Interpolate an authored keyframe track, exactly as record_intents.mjs does. */
function poseAt(track, time) {
  if (time <= 0) return HOME.slice();
  let pt = 0, pp = HOME;
  for (const f of track) {
    if (time <= f.t) {
      const u = (time - pt) / Math.max(f.t - pt, 1e-9), s = u * u * (3 - 2 * u);
      return f.pose.map((v, k) => pp[k] + (v - pp[k]) * s);
    }
    pt = f.t; pp = f.pose;
  }
  return track[track.length - 1].pose.slice();
}

// measure_success.mjs's deterministic generator, same seed, so a rerun agrees.
let seed = 0x2f6e2b1;
function rand() {
  seed ^= seed << 13; seed >>>= 0;
  seed ^= seed >> 17;
  seed ^= seed << 5;  seed >>>= 0;
  return seed / 0x100000000;
}
const between = (lo, hi) => lo + rand() * (hi - lo);

const FOOT_GEOMS = [];
for (let g = 0; g < model.ngeom; g++) {
  if (/foot/i.test(model.geom(g).name || '')) FOOT_GEOMS.push(g);
}
const BASE_FRICTION = FOOT_GEOMS.map(g => model.geom_friction[g * 3]);
const TRUNK_BODY = (() => {
  for (let b = 0; b < model.nbody; b++) if (model.body(b).name === 'trunk_base') return b;
  return -1;
})();
const BASE_COM = [model.body_ipos[TRUNK_BODY * 3], model.body_ipos[TRUNK_BODY * 3 + 1],
                  model.body_ipos[TRUNK_BODY * 3 + 2]];
function restoreModel() {
  for (let i = 0; i < FOOT_GEOMS.length; i++) {
    model.geom_friction[FOOT_GEOMS[i] * 3] = BASE_FRICTION[i];
  }
  for (let k = 0; k < 3; k++) model.body_ipos[TRUNK_BODY * 3 + k] = BASE_COM[k];
}

function posture(z, up) {
  if (up > 0.5) return 'inverted';
  if (up > -0.5) return 'toppled';
  if (z >= 0.100) return 'standing';
  if (z >= 0.075) return 'crouched';
  if (z >= 0.052) return 'seated';
  return 'fallen';
}

async function rollout(track, seconds, blend) {
  const friction = between(0.7, 1.3);
  for (let i = 0; i < FOOT_GEOMS.length; i++) {
    model.geom_friction[FOOT_GEOMS[i] * 3] = BASE_FRICTION[i] * friction;
  }
  for (let k = 0; k < 3; k++) {
    model.body_ipos[TRUNK_BODY * 3 + k] = BASE_COM[k] + between(-0.003, 0.003);
  }
  resetDuck(between(0.12, 0.13));

  const ticks = Math.round(seconds * C.tickHz);
  // One push, guaranteed to land inside the motion (see header).
  const pushAt = Math.round(between(0.3, seconds - 0.3) * C.tickHz);
  const pushX = between(-0.3, 0.3), pushY = between(-0.3, 0.3);
  let lastAction = new Array(14).fill(0);
  const tail = [];
  let peakRate = 0;

  for (let t = -25; t < ticks; t++) {
    if (t === pushAt) { data.qvel[D.freeDof] += pushX; data.qvel[D.freeDof + 1] += pushY; }
    const jp = [], jv = [];
    for (let k = 0; k < 14; k++) { jp.push(data.qpos[D.qpos[k]]); jv.push(data.qvel[D.dof[k]]); }
    const cmd = command({});   // zero command throughout: the stand keeps standing
    const obs = buildObs([data.sensordata[GYRO], data.sensordata[GYRO + 1], data.sensordata[GYRO + 2]],
                         projectedGravity(quat()), jp, jv, lastAction, cmd);
    const out = await session.run({ obs: new ort.Tensor('float32', obs, [1, 61]) });
    lastAction = Array.from(out.actions.data);

    const offsets = t >= 0 ? poseAt(track, t * DT) : null;
    for (let k = 0; k < 14; k++) {
      const base = HOME[k] + lastAction[k];
      const v = offsets ? base + (offsets[k] - HOME[k]) * blend : base;
      data.ctrl[k] = Math.min(Math.max(v, LO[k]), HI[k]);
    }
    for (let s = 0; s < 4; s++) mj.mj_step(model, data);
    if (!Number.isFinite(data.qpos[D.freeQpos + 2])) return null;
    if (t >= 0) {
      for (let k = 0; k < 14; k++) {
        const r = Math.abs(data.qvel[D.dof[k]]);
        if (r > peakRate) peakRate = r;
      }
    }
    if (t >= ticks - 15) {
      const q = quat();
      tail.push([data.qpos[D.freeQpos + 2], projectedGravity(q)[2]]);
    }
  }
  return { tail, peakRate };
}

/** offsets keyed by policy-joint index -> track of absolute poses. */
export function trackFrom(frames, scale = 1) {
  return frames.map(([t, off]) => ({
    t,
    pose: HOME.map((h, k) => h + (off[k] ?? 0) * scale),
  }));
}

export async function verify(name, frames, seconds, scale = 1) {
  seed = 0x2f6e2b1;   // fresh dice per verification, same for every motion
  const track = trackFrom(frames, scale);
  const N = 16;
  let pass = 0, unstable = 0, peakRate = 0;
  const endings = {}, rows = [];
  let worst = null;   // rollout with the lowest average tail z among uprights, or any non-standing
  for (let i = 0; i < N; i++) {
    const r = await rollout(track, seconds, 1.0);
    restoreModel();
    if (!r) { unstable++; rows.push(`  #${i}: NaN state`); continue; }
    const z = r.tail.reduce((a, x) => a + x[0], 0) / r.tail.length;
    const g = r.tail.reduce((a, x) => a + x[1], 0) / r.tail.length;
    const p = posture(z, g);
    const ok = g < -0.5 && z >= 0.100;
    if (ok) pass++;
    endings[p] = (endings[p] ?? 0) + 1;
    if (r.peakRate > peakRate) peakRate = r.peakRate;
    // keep the worst: any failure beats any success; among same class, lowest z
    if (!worst || (worst.ok && !ok) || (worst.ok === ok && z < worst.z)) {
      worst = { ok, z, g, p, i };
    }
    rows.push(`  #${String(i).padStart(2)}: tail z ${z.toFixed(4)}  grav_z ${g.toFixed(3)}  ${p}${ok ? '' : '  FAIL'}  peak ${r.peakRate.toFixed(2)} rad/s`);
  }
  console.log(`\n=== ${name}${scale !== 1 ? ` (offsets x${scale})` : ''} ===`);
  console.log(rows.join('\n'));
  console.log(`PASS ${pass}/${N} (need >=14)   unstable ${unstable}`);
  console.log(`peak per-joint rate ${peakRate.toFixed(2)} rad/s (limit 13)`);
  console.log(`endings ${JSON.stringify(endings)}`);
  console.log(`worst final: rollout #${worst.i}  tail z ${worst.z.toFixed(4)}  grav_z ${worst.g.toFixed(3)}  ${worst.p}`);
  return { pass, N, peakRate, worst, endings, unstable };
}
