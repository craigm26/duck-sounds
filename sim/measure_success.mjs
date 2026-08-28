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
import { clearStairs, findStairJoints, layoutStairs, STAIR_Y, STEP_HALF_DEPTH,
         STEP_HALF_HEIGHT, STAIR_HALF_WIDTH } from '../site/stairs.js';
import { NEEDS, STAGE_STAIRS, stagingFor, targetFor } from '../site/intent-specs.js';
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

/**
 * A world vector expressed in the trunk's own frame: q^-1 * v * q.
 *
 * NEEDED BECAUSE THE TWO HALVES OF A FREE JOINT'S VELOCITY LIVE IN DIFFERENT
 * FRAMES. MuJoCo stores a free joint's linear velocity in the GLOBAL frame and
 * its angular velocity in the BODY frame. mjlab's velocity-tracking rewards
 * both read body-frame quantities (`root_link_lin_vel_b`, `root_link_ang_vel_b`),
 * so the angular half is already right and the linear half has to be rotated.
 * Recording the raw global vector instead would make every tracking number
 * wrong by exactly the robot's heading — largest at the moment it turns, which
 * is when the number matters.
 */
function intoBody(q, v) {
  const [w, x, y, z] = q;
  // t = 2 * (q_vec x v); v' = v + w*t + q_vec x t, then conjugated for the
  // inverse rotation, which is the same expression with the vector part negated.
  const cx = -x, cy = -y, cz = -z;
  const tx = 2 * (cy * v[2] - cz * v[1]);
  const ty = 2 * (cz * v[0] - cx * v[2]);
  const tz = 2 * (cx * v[1] - cy * v[0]);
  return [v[0] + w * tx + (cy * tz - cz * ty),
          v[1] + w * ty + (cz * tx - cx * tz),
          v[2] + w * tz + (cx * ty - cy * tx)];
}

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
  // THE SETTLE STANDS; THE INTENT'S POLICY ONLY TAKES OVER AT t=0.
  //
  // Running the spec's own policy through the settle starts the motion before
  // recording does. It was already fixed once for the command — `sit` opened
  // halfway down — and the policy is the same bug wearing different clothes:
  // roulade tips over regardless of what the command says, so its 25 settle
  // ticks had it already rolling and the clip opened INVERTED. The browser
  // does not work that way either; sim.js swaps the intent's session in at
  // fire time, over a duck that was standing.
  const settling = await policy(STAND);
  if (!spec.continueFrom) resetDuck(spec.start);
  if (spec.stairs) layoutStairs(data, ADDR, spec.stairs);

  let lastAction = new Array(14).fill(0);
  const frames = [], roots = [];
  // WHAT THE POLICY EMITTED AND WHAT IT WAS ASKED FOR, alongside what the robot
  // then did. Without these three the recording can only be described
  // kinematically: Pollen's reward terms are written against the ACTION
  // (`action_rate_l2`), against the COMMAND (`track_linear_velocity`,
  // `track_angular_velocity`) and against the base TWIST, none of which can be
  // recovered from joint positions afterwards. A clip that stores only qpos
  // forces an app to either invent those numbers or leave the whole reward
  // panel empty, and the first of those is much worse than the second.
  const actions = [], commands = [], twists = [];
  let netYaw = 0, lastYaw = null;
  const settle = spec.continueFrom ? 0 : (spec.settle ?? 25);
  const ticks = Math.round(spec.seconds * C.tickHz);
  const tail = [];

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
    const runner = t < 0 ? settling : session;
    const out = await runner.run({ obs: new ort.Tensor('float32', obs, [1, 61]) });
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
      const q = quat();
      roots.push([data.qpos[D.freeQpos], data.qpos[D.freeQpos + 1],
                  data.qpos[D.freeQpos + 2], q[0], q[1], q[2], q[3]]);
      // The action that PRODUCED this frame, not the one that follows it: the
      // policy ran on the state before `mj_step`, so pairing it with the pose
      // after is what makes `action_rate_l2` a difference between consecutive
      // decisions rather than an off-by-one.
      actions.push(lastAction.map(v => +v.toFixed(5)));
      // Only the three velocity slots. The command block is thirteen wide and
      // the other ten are one-hot flags that never move within a clip; storing
      // them per tick would triple the file to record a constant.
      commands.push([+cmd[0].toFixed(5), +cmd[1].toFixed(5), +cmd[2].toFixed(5)]);
      const lin = intoBody(q, [data.qvel[D.freeDof], data.qvel[D.freeDof + 1],
                               data.qvel[D.freeDof + 2]]);
      twists.push([+lin[0].toFixed(5), +lin[1].toFixed(5), +lin[2].toFixed(5),
                   +data.qvel[D.freeDof + 3].toFixed(5),
                   +data.qvel[D.freeDof + 4].toFixed(5),
                   +data.qvel[D.freeDof + 5].toFixed(5)]);
      // Unwrapped, summed per tick. `atan2(last) - atan2(first)` is ambiguous
      // the moment a move turns more than half a circle: roulade came out at
      // -3.272 rad, outside (-pi, pi], and the true rotation could equally have
      // been +3.011. A cursor folding that number sends the duck the wrong way.
      const y = yaw();
      if (lastYaw !== null) {
        let d = y - lastYaw;
        while (d > Math.PI) d -= 2 * Math.PI;
        while (d < -Math.PI) d += 2 * Math.PI;
        netYaw += d;
      }
      lastYaw = y;
    }
  }
  return { frames, roots, netYaw, actions, commands, twists };
}

// ── what to record ────────────────────────────────────────────────────────
//
// Durations come from the browser's own intent table, converted from its step
// counts at 50 Hz, with a little tail so a clip ends settled rather than
// mid-motion.
const STAND = 'BEST_alpha_stand.onnx';
/**
 * An authored move, WITH the world it was searched against.
 *
 * The first cut of this returned only the keyframes, the blend and a duration —
 * so `capture` never saw a `stairs` key, `resetDuck` ran `clearStairs`, and five
 * stair moves were recorded playing into thin air on a bare floor 1.5 m from the
 * wall the flip needs. Every one came out at exactly hold's flat-floor height of
 * 0.11622 m, and step_up turned 0.9 rad over a move that does not turn. That is
 * a duck stumbling, not a stunt, and `endsUpright` was true for all of them, so
 * nothing caught it.
 *
 * The search metadata is right there in the JSON — `approach`, `gap`, `side` —
 * and the browser reads it: sim.js replaces the driving command with
 * `{vx: approach}` for the whole move. Threading it here is what makes the
 * recording the motion the move actually is.
 */
// This table used to live here — which meant the BROWSER had no idea any of it
// existed, and pressing a move with nothing in front of the duck played a track
// authored against a step that was not there. It now lives in
// site/intent-specs.js and both sides import it. Heights: each move is staged
// against the tallest step it has been MEASURED to clear, because recording
// against one it cannot climb produces a faceplant — honest, and useless as a
// clip.

const authored = (name, id) => {
  const j = JSON.parse(fs.readFileSync(`../site/intent-${name}.json`, 'utf8'));
  const track = j.keyframes ?? buildTrack(j.params, HOME);
  const params = j.params ?? {};
  const blend = j.blend ?? params.blend;
  const approach = j.approach ?? params.approach ?? 0;
  const gap = j.gap ?? params.gap ?? 0.06;
  const spec = {
    track, blend, approach,
    // step_up declares NO policy, so the browser's `intent.session || session`
    // falls through to whatever is driving — alpha_walking on legs. The other
    // five name BEST_alpha_stand explicitly.
    policy: j.policy ?? 'alpha_walking.onnx',
    seconds: track[track.length - 1].t + 1.2,
    // The move's own forward velocity replaces the driving command for its
    // whole duration, exactly as sim.js:139 does.
    command: () => ({ vx: approach }),
  };
  const staging = stagingFor(id, j);
  if (staging && staging.kind === 'stair') {
    spec.stairs = { ...STAGE_STAIRS };
    spec.start = targetFor(staging, spec.stairs);
  } else if (staging && staging.kind === 'wall') {
    spec.start = targetFor(staging, spec.stairs);
  }
  return spec;
};

/**
 * Policies this project did not train.
 *
 * The whole point of a shareable intent: somebody else's .onnx, run in the same
 * harness, recorded the same way. `headspin` came from outside — it inverts
 * within half a second (projected gravity z goes from -1.00 to +0.75) and holds
 * a headstand for as long as you let it run.
 *
 * They are recorded identically to Pollen's own, and marked by their FINGERPRINT
 * rather than by which list they appear in: `DuckOfficialPolicies` answers
 * "released" or "unrecognised" from the weights, so a clip carries where it came
 * from in a form a recipient can check instead of a label they must believe.
 */
const COMMUNITY = [
  { id: 'headspin', policy: 'headspin.onnx', seconds: 4.0,
    credit: 'shared by another Microduck owner' },
];

const SPECS = [
  { id: 'hold',        policy: STAND,                     seconds: 2.0 },
  // KICK_STEPS 25 then POST_KICK_LOCK_STEPS 20 — 45 ticks, 0.9 s. The first
  // cut recorded 1.4 s, nearly three times the kick, so the clip was mostly a
  // duck standing still after one.
  { id: 'kick_left',   policy: 'ball_kick_left.onnx',     seconds: 0.9 },
  { id: 'kick_right',  policy: 'ball_kick_right.onnx',    seconds: 0.9 },
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
  ...['stepup', 'lever', 'riser', 'climb', 'backroll', 'wallflip'].map(name => {
    const id = { stepup: 'step_up', lever: 'lever_up', riser: 'riser_up',
                 climb: 'climb', backroll: 'back_roll', wallflip: 'wall_flip' }[name];
    return { id, ...authored(name, id) };
  }),
  ...COMMUNITY,
];

// ─────────────────────────────────────────────────────────────────────────────
// HOW OFTEN DOES IT ACTUALLY WORK?
//
// A recording is one run. "Ends toppled" tells you what happened that time and
// nothing about whether it happens every time, and a panel that printed a
// success rate from a single clip would be printing 0% or 100% and calling it a
// measurement. So this runs each intent many times under randomised conditions
// and counts.
//
// THE RANDOMISATION IS POLLEN'S OWN, read out of microduck_velocity_env_cfg.py
// rather than invented, because a robustness number is only meaningful against
// a stated distribution:
//
//   reset_base pose_range z .... (0.12, 0.13) m      — the drop height
//   foot_friction ranges ....... (0.7, 1.3)          — how grippy the footpad is
//   push_robot velocity_range .. (-0.3, +0.3) m/s    — a shove in x and y
//   push interval .............. (3.0, 6.0) s, so a 4 s clip gets at most one
//   randomize_com .............. ±0.003 m on the trunk
//
// The seed is fixed, so two runs of this script agree and a change in the
// numbers means a change in the robot rather than a change in the dice.
const ROLLOUTS = +(process.env.ROLLOUTS || 12);

// A small deterministic generator. Math.random would make every run disagree
// with the last one by a few percent, which is indistinguishable from a
// regression.
let seed = 0x2f6e2b1;
function rand() {
  seed ^= seed << 13; seed >>>= 0;
  seed ^= seed >> 17;
  seed ^= seed << 5;  seed >>>= 0;
  return seed / 0x100000000;
}
const between = (lo, hi) => lo + rand() * (hi - lo);

// The footpad geoms, so friction can be randomised where Pollen randomise it.
const FOOT_GEOMS = [];
for (let g = 0; g < model.ngeom; g++) {
  const name = model.geom(g).name || '';
  if (/foot/i.test(name)) FOOT_GEOMS.push(g);
}
const BASE_FRICTION = FOOT_GEOMS.map(g => model.geom_friction[g * 3]);
const TRUNK_BODY = (() => {
  for (let b = 0; b < model.nbody; b++) if (model.body(b).name === 'trunk_base') return b;
  return -1;
})();
const BASE_COM = TRUNK_BODY >= 0
  ? [model.body_ipos[TRUNK_BODY * 3], model.body_ipos[TRUNK_BODY * 3 + 1],
     model.body_ipos[TRUNK_BODY * 3 + 2]]
  : null;

/**
 * WHAT COUNTS AS SUCCESS, stated per intent rather than assumed.
 *
 * "Ends upright" is right for a kick and wrong for `sit`, which succeeds by
 * ending seated, and wrong for a stair move, which succeeds only by ending ON
 * the step — a duck standing perfectly upright at the bottom has failed. Every
 * criterion below is a measurement of the final state against a threshold that
 * is written down, so a rate can be read as "how often did it do THAT".
 */
// What each clip was RECORDED doing, so a rollout can be asked the second
// question as well as the first.
const RECORDED = JSON.parse(fs.readFileSync('duck-intent-clips.json', 'utf8')).clips;

/// The recorder's own classifier, copied here rather than re-derived: gravity
/// decides first, because a duck balanced on its head is lower than a fallen
/// one and is not fallen.
function posture(z, up) {
  if (up > 0.5) return 'inverted';
  if (up > -0.5) return 'toppled';
  if (z >= 0.100) return 'standing';
  if (z >= 0.075) return 'crouched';
  if (z >= 0.052) return 'seated';
  return 'fallen';
}

/// OVER A WINDOW, never a single tick — the same rule the recorder follows.
/// The headspin holds gravity between +0.56 and +0.93 for its whole run and was
/// once labelled "toppled" because the one tick sampled sat on the boundary.
function postureOf(tail) {
  const z = tail.reduce((a, r) => a + r[2], 0) / tail.length;
  const g = tail.reduce((a, r) => a + projectedGravity(r.slice(3))[2], 0) / tail.length;
  return posture(z, g);
}

function verdictFor(spec) {
  if (spec.id === 'sit') {
    return { text: 'ends seated, trunk between 52 and 100 mm',
             ok: r => upright(r) && r[2] >= 0.052 && r[2] < 0.100 };
  }
  if (spec.stairs) {
    const target = spec.stairs.count * spec.stairs.rise;
    return {
      text: `ends standing on the flight, trunk at least ${Math.round(target * 1000)} mm up`,
      // ON the step, not beside it: the trunk has to be as high as the top of
      // the flight plus a standing robot's own height, less a tolerance.
      ok: r => upright(r) && r[2] >= 0.100 + target - 0.004,
    };
  }
  // A CLIP WITH NO STATED GOAL IS JUDGED AGAINST ITS OWN RECORDING. The
  // community headspin ENDS ON ITS HEAD on purpose, and scoring it by "ends
  // standing" measured the opposite of what it is for — 0/16, for a motion
  // that did exactly what its author intended. Nothing here knows another
  // owner's intent, so the only criterion available is the one the recording
  // itself demonstrates.
  const recorded = RECORDED[spec.id]?.endsIn;
  if (recorded && recorded !== 'standing') {
    return { text: `ends ${recorded}, as the recording did — this motion has no stated goal here`,
             ok: r => posture(r[2], projectedGravity(r.slice(3))[2]) === recorded };
  }
  return { text: 'ends standing, trunk at least 100 mm up',
           ok: r => upright(r) && r[2] >= 0.100 };
}

// Upright by GRAVITY, not by height — a duck balanced on its head is low and
// not fallen, and one lying flat is at a plausible height and is not standing.
const upright = r => projectedGravity(r.slice(3))[2] < -0.5;

async function rollout(spec) {
  const session = await policy(spec.policy);
  const settling = await policy(STAND);

  // Domain randomisation, applied before the reset so the settle happens under
  // the conditions the run will be judged in.
  const friction = between(0.7, 1.3);
  for (let i = 0; i < FOOT_GEOMS.length; i++) {
    model.geom_friction[FOOT_GEOMS[i] * 3] = BASE_FRICTION[i] * friction;
  }
  if (BASE_COM) {
    for (let k = 0; k < 3; k++) {
      model.body_ipos[TRUNK_BODY * 3 + k] = BASE_COM[k] + between(-0.003, 0.003);
    }
  }
  const dropHeight = between(0.12, 0.13);

  const start = { ...(spec.start || {}), z: dropHeight };
  if (!spec.continueFrom) resetDuck(start);
  if (spec.stairs) layoutStairs(data, ADDR, spec.stairs);

  const ticks = Math.round(spec.seconds * C.tickHz);
  const tail = [];
  const settle = spec.continueFrom ? 0 : (spec.settle ?? 25);
  // At most one push in a clip this short, at Pollen's own interval.
  const pushAt = Math.round(between(3.0, 6.0) * C.tickHz);
  let lastAction = new Array(14).fill(0);

  for (let t = -settle; t < ticks; t++) {
    if (spec.stairs) layoutStairs(data, ADDR, spec.stairs);
    if (t === pushAt) {
      data.qvel[D.freeDof] += between(-0.3, 0.3);
      data.qvel[D.freeDof + 1] += between(-0.3, 0.3);
    }
    const jp = [], jv = [];
    for (let k = 0; k < 14; k++) { jp.push(data.qpos[D.qpos[k]]); jv.push(data.qvel[D.dof[k]]); }
    const cmd = command(spec.command && t >= 0 ? spec.command(t * DT) : {});
    const obs = buildObs([data.sensordata[GYRO], data.sensordata[GYRO + 1], data.sensordata[GYRO + 2]],
                         projectedGravity(quat()), jp, jv, lastAction, cmd);
    const runner = t < 0 ? settling : session;
    const out = await runner.run({ obs: new ort.Tensor('float32', obs, [1, 61]) });
    lastAction = Array.from(out.actions.data);

    const offsets = spec.track && t >= 0 ? poseAt(spec.track, t * DT) : null;
    for (let k = 0; k < 14; k++) {
      const base = HOME[k] + lastAction[k];
      const v = offsets ? base + (offsets[k] - HOME[k]) * spec.blend : base;
      data.ctrl[k] = Math.min(Math.max(v, LO[k]), HI[k]);
    }
    for (let s = 0; s < 4; s++) mj.mj_step(model, data);
    // A NaN state is a failed rollout, not a crash. MuJoCo can produce one on
    // an extreme contact impulse, which is exactly why Pollen's config carries
    // a `nan_state` termination.
    if (!Number.isFinite(data.qpos[D.freeQpos + 2])) return null;
    if (t >= ticks - 15) {
      const q = quat();
      tail.push([data.qpos[D.freeQpos], data.qpos[D.freeQpos + 1],
                 data.qpos[D.freeQpos + 2], q[0], q[1], q[2], q[3]]);
    }
  }
  return tail;
}

// Restore whatever the model shipped with, so one intent's dice do not carry
// into the next one's baseline.
function restoreModel() {
  for (let i = 0; i < FOOT_GEOMS.length; i++) {
    model.geom_friction[FOOT_GEOMS[i] * 3] = BASE_FRICTION[i];
  }
  if (BASE_COM) {
    for (let k = 0; k < 3; k++) model.body_ipos[TRUNK_BODY * 3 + k] = BASE_COM[k];
  }
}

const results = {};
for (const spec of SPECS) {
  // `stand` continues from `sit`, so it has no start state of its own to
  // randomise and cannot be rolled out independently.
  if (spec.continueFrom) {
    console.log(`SKIP ${spec.id} — continues from ${spec.continueFrom}`);
    continue;
  }
  const verdict = verdictFor(spec);
  const recordedEnd = RECORDED[spec.id]?.endsIn ?? null;
  let ok = 0, same = 0, unstable = 0;
  const heights = [], endings = {};
  for (let i = 0; i < ROLLOUTS; i++) {
    const tail = await rollout(spec);
    restoreModel();
    if (!tail || !tail.length) { unstable++; continue; }
    const final = tail[tail.length - 1];
    heights.push(final[2]);
    const ending = postureOf(tail);
    endings[ending] = (endings[ending] ?? 0) + 1;
    if (verdict.ok(final)) ok++;
    if (ending === recordedEnd) same++;
  }
  heights.sort((a, b) => a - b);
  results[spec.id] = {
    rollouts: ROLLOUTS,
    unstable,
    // TWO DIFFERENT QUESTIONS, and conflating them is how a corpus of
    // recordings starts lying about itself. `achieves` asks whether the move
    // did what it is FOR — a stair move that ends upright on the floor has
    // failed, however tidily it is standing. `repeats` asks only whether it
    // did again what it did the day it was recorded, which is what says
    // whether the clip on file is representative or a lucky take.
    achieves: ok,
    criterion: verdict.text,
    repeats: same,
    recordedEnding: recordedEnd,
    endings,
    medianHeight: heights.length ? +heights[heights.length >> 1].toFixed(4) : null,
    worstHeight: heights.length ? +heights[0].toFixed(4) : null,
  };
  console.log(`${spec.id.padEnd(12)} achieves ${String(ok).padStart(3)}/${ROLLOUTS}`
    + `  repeats ${String(same).padStart(3)}/${ROLLOUTS}`
    + `  median z ${results[spec.id].medianHeight}  ${verdict.text}`);
}

fs.writeFileSync('intent-success.json', JSON.stringify({
  format: 'duck-intent-success/1',
  rollouts: ROLLOUTS,
  seed: '0x2f6e2b1',
  randomisation: {
    source: 'pollen-robotics/microduck_rl · microduck_velocity_env_cfg.py',
    dropHeightMetres: [0.12, 0.13],
    footFrictionScale: [0.7, 1.3],
    pushMetresPerSecond: [-0.3, 0.3],
    pushIntervalSeconds: [3.0, 6.0],
    trunkCentreOfMassMetres: [-0.003, 0.003],
  },
  intents: results,
}, null, 2));
console.log(`wrote ${Object.keys(results).length} intents`);
