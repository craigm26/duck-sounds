// Record real walking motion for on-device replay.
//
// DuckSimulation cannot walk (it has no contact), and MuJoCo is not going into
// a zero-dependency Swift package. So the motion is recorded HERE, from the
// real policy in real physics, and replayed on the phone. What an AR ghost
// draws is then still the trained network's walk — it just was not computed on
// the device.
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const { HOME, buildObs, gaitTargets, projectedGravity, command } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/scene.xml', fs.readFileSync('scene.xml', 'utf8'));
const model = mj.MjModel.mj_loadXML('/scene.xml');
const data = new mj.MjData(model);
const session = await ort.InferenceSession.create('./alpha_walking.onnx');
const inputName = session.inputNames[0];
const MOUTH = C.jointNames.indexOf('mouth');

function reset() {
  mj.mj_resetData(model, data);
  data.qpos[2] = 0.1231; data.qpos[3] = 1;
  for (let i = 0; i < 14; i++) { data.qpos[7 + i] = HOME[i]; data.ctrl[i] = HOME[i]; }
  mj.mj_forward(model, data);
}

async function capture(opts, seconds) {
  reset();
  const cmd = command(opts);
  let lastAction = new Array(14).fill(0), previous = null;
  const frames = [];
  for (let t = 0; t < Math.round(seconds * C.tickHz); t++) {
    const q = [data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]];
    const jpos = [], jvel = [];
    for (let k = 0; k < 14; k++) { jpos.push(data.qpos[7 + k]); jvel.push(data.qvel[6 + k]); }
    const obs = buildObs([data.sensordata[0], data.sensordata[1], data.sensordata[2]], projectedGravity(q), jpos, jvel, lastAction, cmd);
    const out = await session.run({ [inputName]: new ort.Tensor('float32', obs, [1, 61]) });
    lastAction = Array.from(out[session.outputNames[0]].data);
    previous = gaitTargets(lastAction, previous);
    for (let k = 0; k < 14; k++) data.ctrl[k] = previous[k];
    for (let s = 0; s < 4; s++) mj.mj_step(model, data);
    // The 15-joint pose in DuckKit's order, plus where the body actually is.
    const joints = [];
    let n = 0;
    for (let i = 0; i < 15; i++) joints.push(i === MOUTH ? 0 : data.qpos[7 + n++]);
    frames.push({
      joints,
      root: [data.qpos[0], data.qpos[1], data.qpos[2]],
      quat: [data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]],
    });
  }
  return frames;
}

/** Trim to a whole number of strides so the clip loops without a hitch. */
function trimToStride(frames, settleTicks) {
  const body = frames.slice(settleTicks);
  const ankle = body.map(f => f.joints[4]);
  const mean = ankle.reduce((a, b) => a + b, 0) / ankle.length;
  const c = ankle.map(v => v - mean);
  const denom = c.reduce((a, v) => a + v * v, 0) || 1e-12;
  let best = -2, lag = 0;
  for (let L = 8; L < Math.floor(c.length / 2); L++) {
    let s = 0;
    for (let k = 0; k + L < c.length; k++) s += c[k] * c[k + L];
    s /= denom;
    if (s > best) { best = s; lag = L; }
  }
  const strides = Math.max(1, Math.floor(body.length / lag));
  return { frames: body.slice(0, strides * lag), period: lag, strides, quality: best };
}

// Only the commands this actuator model reproduces faithfully. A stiffness
// sweep showed no setting that both stays upright AND turns symmetrically:
// kp 14 turns correctly in both directions and falls over; kp 9 is stable but
// only turns left. Rather than ship a clip of something the policy is not
// really doing, right-turn is produced by MIRRORING turn_left on the device.
const CLIPS = [
  ['stand',     { vx: 0 },              6, 100],
  ['walk',      { vx: 0.15 },          10, 150],
  ['walk_fast', { vx: 0.20 },          10, 150],
  ['turn_left', { vx: 0, vyaw: 0.8 },  10, 150],
];

const out = { hz: C.tickHz, joints: C.jointNames, clips: {} };
for (const [name, opts, secs, settle] of CLIPS) {
  const raw = await capture(opts, secs);
  const { frames, period, quality } = trimToStride(raw, settle);
  const first = frames[0], last = frames[frames.length - 1];
  // Root motion is stored as a per-loop DELTA so a looping walk keeps travelling.
  const dx = last.root[0] - first.root[0], dy = last.root[1] - first.root[1];
  const yawOf = q => Math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3]));
  const dyaw = yawOf(last.quat) - yawOf(first.quat);
  const r = v => Math.round(v * 10000) / 10000;
  out.clips[name] = {
    period, quality: r(quality),
    deltaX: r(dx), deltaY: r(dy), deltaYaw: r(dyaw),
    height: r(frames.reduce((a, f) => a + f.root[2], 0) / frames.length),
    frames: frames.map(f => f.joints.map(r)),
  };
  console.log(`CLIP ${name.padEnd(11)} ${String(frames.length).padStart(4)} ticks  period ${period}  autocorr ${quality.toFixed(3)}  Δ(${r(dx)}, ${r(dy)}) m  Δyaw ${r(dyaw)} rad`);
}
fs.writeFileSync('duck-trajectories.json', JSON.stringify(out));
console.log('WROTE duck-trajectories.json', (fs.statSync('duck-trajectories.json').size / 1024).toFixed(1), 'KB');
