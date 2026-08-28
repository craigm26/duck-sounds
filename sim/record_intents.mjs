// Record every intent from real physics, for replay on a phone.
//
// WHY RECORD AT ALL. `DuckSimulation` cannot run these policies live: the
// network locks its phase to CONTACT through the gyro, projected gravity and
// joint velocities, and on a device all three are constants. Measured, closing
// that loop gives a 25 Hz flip-flop or a slam into the travel stops. So the
// motion happens HERE, in MuJoCo, driven by the real trained networks, and the
// phone replays it. What an AR duck shows is still the trained policy's motion;
// the physics simply happened earlier and somewhere else.
//
// WHY THIS IS NOT `record.mjs`. That one records the WALKING policy under
// different velocity commands and trims each result to a whole stride by
// autocorrelation, because a gait loops. Almost nothing here loops: a kick
// happens once, a roll happens once, sitting down ends sat. Trimming a kick to
// its "period" would be meaningless. These are one-shots, and they carry a
// start pose and an end pose rather than a stride.
//
// THE TWO FAMILIES ARE DRIVEN DIFFERENTLY, and that is the whole shape of this
// file. Seven intents swap in a DIFFERENT policy file and let it run. Six are
// authored keyframe tracks that ride ON TOP of the standing policy as offsets,
// which is how they were searched in the first place — the policy keeps its
// balance while the track reaches somewhere. Both end up as the same thing on
// disk: joint angles per tick, plus where the trunk got to.
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
import { clearStairs, findStairJoints, layoutStairs, STAIR_Y } from '../site/stairs.js';
// step_up ships its SEARCHED PARAMETERS rather than an exported track, and the
// browser builds the track from them at load. Importing the same builder means
// the recording is the motion a visitor actually sees, not a re-derivation of
// it that could drift.
import { buildTrack } from '../site/intent.mjs';

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

const sessions = new Map();
async function policy(file) {
  if (!sessions.has(file)) sessions.set(file, await ort.InferenceSession.create('./' + file));
  return sessions.get(file);
}

function resetDuck({ x = 0, y = 0, z = 0.1231 } = {}) {
  mj.mj_resetData(model, data);
  clearStairs(data, ADDR);
  data.qpos[D.freeQpos] = x;
  data.qpos[D.freeQpos + 1] = y;
  data.qpos[D.freeQpos + 2] = z;
  data.qpos[D.freeQpos + 3] = 1;
  for (let i = 0; i < 14; i++) { data.qpos[D.qpos[i]] = HOME[i]; data.ctrl[i] = HOME[i]; }
  mj.mj_forward(model, data);
}

const quat = () => [data.qpos[D.freeQpos + 3], data.qpos[D.freeQpos + 4],
                    data.qpos[D.freeQpos + 5], data.qpos[D.freeQpos + 6]];

/** Yaw from the trunk quaternion, radians. */
function yaw() {
  const [w, x, y, z] = quat();
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}

/** Interpolate an authored keyframe track, exactly as the browser does. */
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

/**
 * Run one intent and return its frames.
 *
 * `settle` ticks run first and are DISCARDED: a duck dropped into the scene
 * bounces for a moment, and a clip that starts mid-bounce starts with a lurch
 * the robot never performs.
 */
async function capture(spec) {
  const session = await policy(spec.policy);
  if (!spec.continueFrom) resetDuck(spec.start);
  if (spec.stairs) layoutStairs(data, ADDR, spec.stairs);

  let lastAction = new Array(14).fill(0);
  const frames = [], roots = [];
  const settle = spec.continueFrom ? 0 : (spec.settle ?? 25);
  const ticks = Math.round(spec.seconds * C.tickHz);

  for (let t = -settle; t < ticks; t++) {
    if (spec.stairs) layoutStairs(data, ADDR, spec.stairs);
    const jp = [], jv = [];
    for (let k = 0; k < 14; k++) { jp.push(data.qpos[D.qpos[k]]); jv.push(data.qvel[D.dof[k]]); }

    // The command block is a clock for ground_pick and a flag for sitstand, so
    // it is rebuilt per tick rather than hoisted.
    //
    // NEUTRAL DURING THE SETTLE. The settle exists to let the drop-bounce die,
    // and feeding the intent's own command through it starts the motion before
    // recording does: `sit` was already down to 90 mm on its first recorded
    // frame, so the clip opened halfway through sitting and a phone replaying
    // it would show a duck that teleports into a crouch.
    const cmd = command(spec.command && t >= 0 ? spec.command(t * DT) : {});
    const obs = buildObs([data.sensordata[GYRO], data.sensordata[GYRO + 1], data.sensordata[GYRO + 2]],
                         projectedGravity(quat()), jp, jv, lastAction, cmd);
    const out = await session.run({ obs: new ort.Tensor('float32', obs, [1, 61]) });
    lastAction = Array.from(out.actions.data);

    const offsets = spec.track && t >= 0 ? poseAt(spec.track, t * DT) : null;
    for (let k = 0; k < 14; k++) {
      const base = HOME[k] + lastAction[k];
      const v = offsets ? base + (offsets[k] - HOME[k]) * spec.blend : base;
      data.ctrl[k] = Math.min(Math.max(v, LO[k]), HI[k]);
    }
    for (let s = 0; s < 4; s++) mj.mj_step(model, data);

    if (t >= 0) {
      const angles = [];
      for (let k = 0; k < 14; k++) angles.push(+data.qpos[D.qpos[k]].toFixed(5));
      frames.push(angles);
      roots.push([+data.qpos[D.freeQpos].toFixed(5),
                  +data.qpos[D.freeQpos + 1].toFixed(5),
                  +data.qpos[D.freeQpos + 2].toFixed(5),
                  +yaw().toFixed(5)]);
    }
  }
  return { frames, roots, upright: projectedGravity(quat())[2] < -0.9 };
}

// ── what to record ────────────────────────────────────────────────────────
//
// Durations come from the browser's own intent table, converted from its step
// counts at 50 Hz, with a little tail so a clip ends settled rather than
// mid-motion.
const STAND = 'BEST_alpha_stand.onnx';
const authored = name => {
  const j = JSON.parse(fs.readFileSync(`../site/intent-${name}.json`, 'utf8'));
  const track = j.keyframes ?? buildTrack(j.params, HOME);
  const blend = j.blend ?? j.params.blend;
  return { track, blend, policy: STAND, seconds: track[track.length - 1].t + 1.2 };
};

const SPECS = [
  { id: 'hold',        policy: STAND,                     seconds: 2.0 },
  { id: 'kick_left',   policy: 'ball_kick_left.onnx',     seconds: 1.4 },
  { id: 'kick_right',  policy: 'ball_kick_right.onnx',    seconds: 1.4 },
  { id: 'ground_pick', policy: 'alpha_ground_pick.onnx',  seconds: 2.8,
    // The command slots carry a clock, not a velocity: cos and sin of progress
    // through a 4 s period. Feeding a velocity here makes the duck try to walk.
    command: t => ({ vx: Math.cos(2 * Math.PI * t / 4.0), vy: Math.sin(2 * Math.PI * t / 4.0) }) },
  { id: 'roulade',     policy: 'roulade.onnx',            seconds: 3.0 },
  { id: 'sit',         policy: 'BEST_alpha_sitstand.onnx', seconds: 3.0, command: () => ({ vx: 1 }) },
  // MUST follow `sit`, and the recorder relies on it: asked to stand up from a
  // duck that is already standing, the policy correctly does nothing, and the
  // clip would be three seconds of a robot not moving.
  { id: 'stand',       policy: 'BEST_alpha_sitstand.onnx', seconds: 3.0,
    command: () => ({ vx: 0 }), continueFrom: 'sit' },
  ...['stepup', 'lever', 'riser', 'climb', 'backroll', 'wallflip'].map(name => ({
    id: { stepup: 'step_up', lever: 'lever_up', riser: 'riser_up',
          climb: 'climb', backroll: 'back_roll', wallflip: 'wall_flip' }[name],
    ...authored(name),
  })),
];

const out = { hz: C.tickHz, joints: C.jointNames.filter(n => n !== 'mouth'), clips: {} };
for (const spec of SPECS) {
  const r = await capture(spec);
  const first = r.roots[0], last = r.roots[r.roots.length - 1];
  out.clips[spec.id] = {
    frames: r.frames,
    height: first[2],
    deltaX: +(last[0] - first[0]).toFixed(5),
    deltaY: +(last[1] - first[1]).toFixed(5),
    deltaYaw: +(last[3] - first[3]).toFixed(5),
    endsUpright: r.upright,
    policy: spec.policy,
    authored: !!spec.track,
  };
  console.log(`CLIP ${spec.id.padEnd(12)} ${String(r.frames.length).padStart(4)} ticks  `
    + `z ${first[2].toFixed(3)}→${last[2].toFixed(3)}  `
    + `Δ(${(last[0]-first[0]).toFixed(3)}, ${(last[1]-first[1]).toFixed(3)}) m  `
    + `Δyaw ${(last[3]-first[3]).toFixed(3)}  upright=${r.upright}`);
}
fs.writeFileSync('duck-intent-clips.json', JSON.stringify(out));
console.log(`wrote ${Object.keys(out.clips).length} clips`);
