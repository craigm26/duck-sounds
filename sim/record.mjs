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
const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);
const mj = await load();
// THE CANON PLANT — the same compiled scene the intent recorder uses, carrying
// training's solver, friction, contact priority and torque ceiling. The first
// version of this recorder ran a third, hand-tuned plant (a kp-9 stiffness
// sweep) that was neither the training plant nor the hardware, and every AR
// ghost walked with its artefacts.
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const model = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model);
let GYRO = 0;
for (let i = 0; i < model.nsensor; i++) {
  if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;
}
const session = await ort.InferenceSession.create('./alpha_walking.onnx');
const settling = await ort.InferenceSession.create('./BEST_alpha_stand.onnx');
const inputName = session.inputNames[0];
const MOUTH = C.jointNames.indexOf('mouth');

function reset() {
  mj.mj_resetData(model, data);
  // Addressed by name, not by fixed offsets: the canon scene carries stairs
  // and props ahead of the duck in qpos, so 2/3/7+i land on the wrong joints.
  data.qpos[D.freeQpos + 2] = 0.1231; data.qpos[D.freeQpos + 3] = 1;
  for (let i = 0; i < 14; i++) { data.qpos[D.qpos[i]] = HOME[i]; data.ctrl[i] = HOME[i]; }
  mj.mj_forward(model, data);
}

async function capture(opts, seconds) {
  reset();
  let lastAction = new Array(14).fill(0), previous = null;
  const frames = [];
  // A STANDING SETTLE BEFORE THE WALK TAKES OVER, exactly as the intent
  // recorder does. Without it the walk policy meets the drop-bounce with its
  // command already applied, and on the canon plant that kicked turn_left into
  // a sustained WRONG-WAY spin (−0.53 rad/s under a +1.0 command); with the
  // settle the same command turns +0.17 rad/s, the correct sign.
  for (let t = -25; t < Math.round(seconds * C.tickHz); t++) {
    const cmd = command(t >= 0 ? opts : {});
    const q = [data.qpos[D.freeQpos + 3], data.qpos[D.freeQpos + 4],
               data.qpos[D.freeQpos + 5], data.qpos[D.freeQpos + 6]];
    const jpos = [], jvel = [];
    for (let k = 0; k < 14; k++) { jpos.push(data.qpos[D.qpos[k]]); jvel.push(data.qvel[D.dof[k]]); }
    const obs = buildObs([data.sensordata[GYRO], data.sensordata[GYRO + 1], data.sensordata[GYRO + 2]],
                         projectedGravity(q), jpos, jvel, lastAction, cmd);
    const runner = t < 0 ? settling : session;
    const out = await runner.run({ [inputName]: new ort.Tensor('float32', obs, [1, 61]) });
    lastAction = Array.from(out[session.outputNames[0]].data);
    // THE TRAINING PATH: target = HOME + action at scale 1.0, no filter —
    // what mjlab drove during training and what every .onnx declares
    // (action_scale = 1.0). The first version ran gaitTargets, robotd's
    // HARDWARE path (0.9 + low-pass), so the shipped ghost was an emulation of
    // the robot on an invented plant rather than a replay of the policy.
    for (let k = 0; k < 14; k++) {
      data.ctrl[k] = Math.min(Math.max(HOME[k] + lastAction[k], LO[k]), HI[k]);
    }
    for (let s = 0; s < 4; s++) mj.mj_step(model, data);
    if (t < 0) continue;
    // The 15-joint pose in DuckKit's order, plus where the body actually is.
    const joints = [];
    let n = 0;
    for (let i = 0; i < 15; i++) joints.push(i === MOUTH ? 0 : data.qpos[D.qpos[n++]]);
    frames.push({
      joints,
      root: [data.qpos[D.freeQpos], data.qpos[D.freeQpos + 1], data.qpos[D.freeQpos + 2]],
      quat: q.slice(),
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
  // THE SEAM DECIDES, NOT JUST THE AUTOCORRELATION. A walking stride is two
  // steps, and the ankle signal can correlate almost as well at ONE step —
  // half a stride — as at a full one. Trim to a half-stride and the loop's
  // last frame hands back to its first on the OPPOSITE leg: measured, a
  // 0.33 rad knee jump at the seam, once per loop, on a ghost that is
  // otherwise smooth. So both candidate periods are tried and the one whose
  // seam actually closes wins, with a few ticks of slack at the end to land
  // on the best joining frame.
  // The seam is tuned by sliding the START, never by clipping the end short:
  // the replayer loops modulo the clip length, so the length must stay an
  // exact number of periods — a test on the consumer side asserts it.
  const seamError = (start, end) => {
    let worst = 0;
    for (let j = 0; j < body[start].joints.length; j++) {
      worst = Math.max(worst, Math.abs(body[end - 1].joints[j] - body[start].joints[j]));
    }
    return worst;
  };
  const MAX_SHIFT = 6;
  let bestChoice = null;
  for (const L of [lag, lag * 2]) {
    if (L >= body.length - MAX_SHIFT) continue;
    const strides = Math.max(1, Math.floor((body.length - MAX_SHIFT) / L));
    for (let shift = 0; shift <= MAX_SHIFT; shift++) {
      const end = shift + strides * L;
      if (end > body.length) continue;
      const err = seamError(shift, end);
      if (!bestChoice || err < bestChoice.err) {
        bestChoice = { start: shift, end, period: L, strides, err };
      }
    }
  }
  return { frames: body.slice(bestChoice.start, bestChoice.end),
           period: bestChoice.period, strides: bestChoice.strides,
           quality: best, seam: bestChoice.err };
}

// COMMANDS THE POLICY ACTUALLY TRACKS. On the canon plant alpha_walking has a
// low-command dead band: at vx 0.1-0.2 it marches in place (measured 1-2 mm/s),
// and only from about 0.25 does commanded velocity produce real travel
// (0.25 -> 0.106 m/s, 0.35 -> 0.150 m/s, both measured over 8 s). The old
// 0.15/0.20 commands were recorded inside that dead band on a plant stiff
// enough to move anyway — which is why the old clips travelled at all.
// Right-turn is still MIRRORED from turn_left on the device rather than
// recorded, so both directions are exactly as symmetric as the mirror.
const CLIPS = [
  ['stand',     { vx: 0 },              6, 100],
  ['walk',      { vx: 0.25 },          10, 150],
  ['walk_fast', { vx: 0.35 },          10, 150],
  ['turn_left', { vx: 0, vyaw: 1.0 },  10, 150],
];

// A MERGE, NOT A REWRITE: the rollers clips (record_rollers.mjs, variant
// 'rollers') survive a walker run; only the walker's own clips are replaced.
const out = { hz: C.tickHz, joints: C.jointNames, clips: {} };
if (fs.existsSync('duck-trajectories.json')) {
  const previous = JSON.parse(fs.readFileSync('duck-trajectories.json', 'utf8')).clips;
  for (const [name, clip] of Object.entries(previous)) if (clip.variant === 'rollers') out.clips[name] = clip;
}
for (const [name, opts, secs, settle] of CLIPS) {
  const raw = await capture(opts, secs);
  const { frames, period, quality, seam } = trimToStride(raw, settle);
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
  console.log(`CLIP ${name.padEnd(11)} ${String(frames.length).padStart(4)} ticks  period ${period}  autocorr ${quality.toFixed(3)}  seam ${seam.toFixed(3)}  Δ(${r(dx)}, ${r(dy)}) m  Δyaw ${r(dyaw)} rad`);
}
fs.writeFileSync('duck-trajectories.json', JSON.stringify(out));
console.log('WROTE duck-trajectories.json', (fs.statSync('duck-trajectories.json').size / 1024).toFixed(1), 'KB');
