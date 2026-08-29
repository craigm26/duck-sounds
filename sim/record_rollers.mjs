// Record the roller policy gliding on Pollen's rollers plant — the skating
// counterpart of record.mjs, for the phone's ghost to wear when it is on
// wheels.
//
// WHAT THIS IS AND IS NOT. BEST_roller.onnx is Pollen's roller velocity
// policy, run on scene_physics_rollers.xml (robot_allcollisions_rollers.xml,
// byte-identical to microduck_rl develop, through build_rollers.py). A glide is
// not a stride: there is no periodic gait to trim to, so each clip is a fixed
// window after the policy has settled, and the wheels' turning is NOT in the
// pose — it is passive, and the renderer rolls them from distance covered.
// The plant's training-parameter rebuild (per-actuator forceranges) is still
// pending, so the speeds here are the older rollers scene's, as the soccer
// capabilities already say.
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
import { clearStairs, findStairJoints } from '../site/stairs.js';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/r.mjb', new Uint8Array(fs.readFileSync('scene-rollers.mjb')));
const model = mj.MjModel.mj_loadBinary('/r.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model), ADDR = findStairJoints(model);
let GYRO = 0;
for (let i = 0; i < model.nsensor; i++) {
  if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;
}
const session = await ort.InferenceSession.create('./BEST_roller.onnx');
const inputName = session.inputNames[0];
const MOUTH = C.jointNames.indexOf('mouth');

// PARK THE PROPS. scene-rollers.mjb is the soccer scene: a ball at
// (0.55, 0.10), three blocks, two cones and the stairs, all in a duck's
// path. The first recording glided straight into the ball — three of five
// clips were recorded striking it. Every free body that is not the robot
// goes far away and the stairs are cleared, exactly as record_intents.mjs
// does for its own scene.
function parkProps() {
  let parked = 0;
  for (let j = 0; j < model.njnt; j++) {
    if (model.jnt_type[j] !== 0) continue;               // mjJNT_FREE
    const adr = model.jnt_qposadr[j], dof = model.jnt_dofadr[j];
    if (adr === D.freeQpos) continue;                    // the duck
    data.qpos[adr] = -5 - parked * 1.5; data.qpos[adr + 1] = -5; data.qpos[adr + 2] = 0.2;
    data.qpos[adr + 3] = 1; data.qpos[adr + 4] = 0; data.qpos[adr + 5] = 0; data.qpos[adr + 6] = 0;
    for (let k = 0; k < 6; k++) data.qvel[dof + k] = 0;
    parked++;
  }
  if (ADDR) clearStairs(data, ADDR);
  return parked;
}

function reset() {
  mj.mj_resetData(model, data);
  parkProps();
  for (let i = 0; i < 14; i++) { data.qpos[D.qpos[i]] = HOME[i]; data.ctrl[i] = HOME[i]; }
  mj.mj_forward(model, data);
}

// The training path, like record.mjs: target = HOME + action, scale 1.0
// (what BEST_roller.onnx declares), no filter.
let clamped = 0;
async function capture(opts, seconds, settleTicks) {
  reset(); clamped = 0;
  let lastAction = new Array(14).fill(0);
  const frames = [];
  const total = Math.round(seconds * C.tickHz) + settleTicks;
  for (let t = 0; t < total; t++) {
    const cmd = command(t >= settleTicks ? opts : {});
    const f = D.freeQpos;
    const q = [data.qpos[f + 3], data.qpos[f + 4], data.qpos[f + 5], data.qpos[f + 6]];
    const jpos = [], jvel = [];
    for (let k = 0; k < 14; k++) { jpos.push(data.qpos[D.qpos[k]]); jvel.push(data.qvel[D.dof[k]]); }
    const obs = buildObs([data.sensordata[GYRO], data.sensordata[GYRO + 1], data.sensordata[GYRO + 2]],
                         projectedGravity(q), jpos, jvel, lastAction, cmd);
    const out = await session.run({ [inputName]: new ort.Tensor('float32', obs, [1, 61]) });
    lastAction = Array.from(out[session.outputNames[0]].data);
    for (let k = 0; k < 14; k++) {
      data.ctrl[k] = Math.min(Math.max(HOME[k] + lastAction[k], LO[k]), HI[k]);
    }
    for (let s = 0; s < 4; s++) mj.mj_step(model, data);   // 200 Hz physics under the 50 Hz policy, as record.mjs
    // THE FRAME IS THE POST-STEP STATE — joints, root and quaternion all
    // read after the physics, as record.mjs does. The first version paired
    // pre-step joints with a post-step root, so every clip's legs lagged its
    // travel by one tick.
    const after = [];
    for (let k = 0; k < 14; k++) after.push(data.qpos[D.qpos[k]]);
    const qAfter = [data.qpos[f + 3], data.qpos[f + 4], data.qpos[f + 5], data.qpos[f + 6]];
    // MuJoCo's joint limits are soft: a hip driven hard into its stop can sit
    // a few thousandths past the range. The servo's travel is the truth for
    // a render, so the recording is clamped there, and says how often.
    const joints = after.map((v, k) => {
      const c = Math.min(Math.max(v, LO[k]), HI[k]);
      if (c !== v) clamped++;
      return c;
    });
    joints.splice(MOUTH, 0, 0);
    frames.push({ joints, root: [data.qpos[f], data.qpos[f + 1], data.qpos[f + 2]], quat: qAfter });
  }
  const fell = data.qpos[D.freeQpos + 2] < 0.06;
  return { frames, fell, clamped };
}

// name, command, seconds after settle, settle ticks (the command is applied
// AFTER the settle, so the first second of every clip is the policy taking
// up the command from a standstill; the window starts after that).
const CLIPS = [
  ['skate_stand', { vx: 0 },                 5, 50],
  ['skate',       { vx: 0.45 },              6, 50],
  ['skate_fast',  { vx: 0.8 },               6, 50],
  ['skate_back',  { vx: -0.4 },              6, 50],
  ['skate_turn',  { vx: 0.4, vyaw: 0.6 },    6, 50],
];
const WINDOW = 250; // 5 s at 50 Hz, taken from the END of the run: the steady glide

// THE SWIZZLE IS A GAIT AFTER ALL. Measured on the first recording: every
// moving clip is a clean ~0.62 s cycle of hip-yaw/knee/ankle at ~0.7 rad
// amplitude (autocorrelation 0.85, 0.04 rad per tick — smooth, not jitter).
// So the window is cut to whole cycles at the seam that closes best, like
// record.mjs does for walking, and the renderer can loop it.
function trimToCycles(frames) {
  const J = frames[0].joints.length, n = frames.length;
  const hipYaw = frames.map(f => f.joints[0]);
  const mean = hipYaw.reduce((a, v) => a + v, 0) / n;
  const x = hipYaw.map(v => v - mean), den = x.reduce((a, v) => a + v * v, 0) || 1e-12;
  let best = { c: 0, lag: 0 };
  for (let lag = 10; lag < 125; lag++) {
    let c = 0; for (let i = 0; i + lag < n; i++) c += x[i] * x[i + lag];
    c /= den; if (c > best.c) best = { c, lag };
  }
  if (best.c < 0.5) return { frames, period: n, quality: best.c, seam: 0 };
  const P = best.lag, kMax = Math.floor((n - P) / P);
  // The seam is judged by its WORST joint, not a sum — one hip snapping 0.4
  // rad at the loop point is what the eye sees — and fewer cycles are
  // traded for a cleaner one.
  let seam = Infinity, start = 0, k = kMax;
  for (let kk = kMax; kk >= Math.max(3, kMax - 2); kk--) {
    for (let s = 0; s + kk * P < n; s++) {
      let d = 0;
      for (let j = 0; j < J; j++) d = Math.max(d, Math.abs(frames[s].joints[j] - frames[s + kk * P].joints[j]));
      if (d < seam) { seam = d; start = s; k = kk; }
    }
  }
  return { frames: frames.slice(start, start + k * P), period: P, quality: best.c, seam };
}

// EVERY RUN IS A MERGE. This recorder owns the rollers clips: it drops the
// stale ones and leaves the walker's alone, and record.mjs does the reverse,
// so neither run can silently delete the other's work.
const out = JSON.parse(fs.readFileSync('duck-trajectories.json', 'utf8'));
for (const [name, clip] of Object.entries(out.clips)) if (clip.variant === 'rollers') delete out.clips[name];
for (const [name, opts, secs, settle] of CLIPS) {
  const { frames: all, fell, clamped } = await capture(opts, secs, settle);
  const { frames, period, quality, seam } = trimToCycles(all.slice(all.length - WINDOW));
  const first = frames[0], last = frames[frames.length - 1];
  const dx = last.root[0] - first.root[0], dy = last.root[1] - first.root[1];
  const yawOf = q => Math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3]));
  // UNWRAPPED. A turn clip covers more than half a revolution, and the
  // difference of two wrapped headings is then off by 2π in either direction;
  // summing per-tick deltas gives the yaw the robot actually turned through.
  let dyaw = 0;
  for (let i = 1; i < frames.length; i++) {
    let d = yawOf(frames[i].quat) - yawOf(frames[i - 1].quat);
    if (d > Math.PI) d -= 2 * Math.PI; else if (d < -Math.PI) d += 2 * Math.PI;
    dyaw += d;
  }
  const r = v => Math.round(v * 10000) / 10000;
  const secsWindow = frames.length / C.tickHz;
  // How much the legs actually do while gliding — the honest answer to
  // "zero related servo movements".
  const travel = frames[0].joints.map((_, j) =>
    frames.reduce((a, f, i) => i ? a + Math.abs(f.joints[j] - frames[i - 1].joints[j]) : 0, 0));
  const busiest = travel.map((v, j) => [v, C.jointNames[j]]).sort((a, b) => b[0] - a[0]).slice(0, 3);
  out.clips[name] = {
    period, quality: r(quality),
    deltaX: r(dx), deltaY: r(dy), deltaYaw: r(dyaw),
    height: r(frames.reduce((a, f) => a + f.root[2], 0) / frames.length),
    // Rounded to 4 decimals for the file, then clamped AGAIN: rounding
    // −0.38397 to −0.3840 carries a joint sitting on its stop 3e-5 past it.
    frames: frames.map(f => f.joints.map((v, k) => Math.min(Math.max(r(v), C.rangeLo[k]), C.rangeHi[k]))),
    variant: 'rollers',
  };
  console.log(`CLIP ${name.padEnd(11)} ${frames.length} ticks (period ${period}, autocorr ${quality.toFixed(2)}, seam ${seam.toFixed(3)})  Δ(${r(dx)}, ${r(dy)}) m in ${secsWindow}s = ${(Math.hypot(dx, dy) / secsWindow).toFixed(3)} m/s  Δyaw ${r(dyaw)} rad  z ${r(last.root[2])}  ${fell ? 'FELL' : 'upright'}  clamped ${clamped} samples  busiest joints: ${busiest.map(([v, n]) => `${n} ${v.toFixed(2)} rad`).join(', ')}`);
}
fs.writeFileSync('duck-trajectories.json', JSON.stringify(out));
console.log('wrote duck-trajectories.json with', Object.keys(out.clips).join(', '));
