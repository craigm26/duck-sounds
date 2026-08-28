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
  let netYaw = 0, lastYaw = null;
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
  return { frames, roots, netYaw };
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
const AUTHORED_WORLD = {
  // Heights each move was last MEASURED to clear, strictly. Recording against a
  // step it cannot climb produces a faceplant, which is honest but useless as a
  // clip; recording against the height it does clear shows the move working.
  step_up:   { rise: 0.010 },
  lever_up:  { rise: 0.010 },
  riser_up:  { rise: 0.010 },
  climb:     { rise: 0.010 },
  back_roll: null,            // flat floor, no prop — it is a roll
  wall_flip: { wall: true },  // needs the arena wall, not a stair
};

const authored = (name, id) => {
  const j = JSON.parse(fs.readFileSync(`../site/intent-${name}.json`, 'utf8'));
  const track = j.keyframes ?? buildTrack(j.params, HOME);
  const params = j.params ?? {};
  const blend = j.blend ?? params.blend;
  const approach = j.approach ?? params.approach ?? 0;
  const gap = j.gap ?? params.gap ?? 0.06;
  const world = AUTHORED_WORLD[id];
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
  if (world && world.rise !== undefined) {
    spec.stairs = { count: 4, rise: world.rise, run: 0.28, start: 0.12 };
    spec.start = { x: 0.12 - 0.07 - gap, y: STAIR_Y + (j.side ?? 0) };
  } else if (world && world.wall) {
    // Facing the arena wall at y = 1.5 with half-thickness 0.025.
    spec.start = { x: 0, y: 1.5 - 0.025 - 0.05 - gap };
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

/**
 * Move a clip so it starts at the origin, facing along +x.
 *
 * MANDATORY, NOT COSMETIC. A recorded root carries wherever the duck happened
 * to be in MuJoCo. `stand` is captured with `continueFrom: 'sit'`, so it begins
 * 74 mm and 0.033 rad away from where `sit` began; the stair moves start at
 * y = 1.305, beside a wall. Replayed as recorded, firing an intent teleports
 * the duck across the room. De-origined, a clip is droppable anywhere on an AR
 * floor, and "frame 0 is at the origin" becomes something a decoder can assert
 * rather than something a comment hopes for.
 */
/**
 * What posture a trunk height AND ORIENTATION mean together.
 *
 * Height alone is not enough, and the corpus proves it: the community headspin
 * clip sits at 0.048 m, lower than a fallen duck, because it is balanced on its
 * head ON PURPOSE. Classifying by height called that "fallen", which is the
 * same species of wrong as the hardcoded labels this replaced — a confident
 * claim contradicted by the recording it describes.
 *
 * So gravity decides first. Projected gravity z is -1 when the trunk is upright
 * and +1 when it is upside down; past +0.5 the duck is inverted whatever its
 * height, and that is a posture rather than a failure. Only once it is the
 * right way up does height separate standing (hold settles at 0.116) from
 * seated (sit settles at 0.059), with the midpoint between them.
 */
/**
 * Posture over a WINDOW, never a single tick.
 *
 * The headspin holds gravity z between +0.56 and +0.93 for its whole run, and
 * was still classified "toppled" because the one frame sampled — the last —
 * happened to sit on the +0.5 boundary. A single tick of a dynamic motion is
 * noise, and a label taken from noise is a label that flickers between
 * recordings of the same clip.
 */
function postureOver(roots, from, to) {
  const slice = roots.slice(from, to);
  const z = slice.reduce((a, r) => a + r[2], 0) / slice.length;
  const g = slice.reduce((a, r) => a + projectedGravity(r.slice(3))[2], 0) / slice.length;
  return posture(z, g);
}

function posture(z, up) {
  if (up > 0.5) return 'inverted';
  if (up > -0.5) return 'toppled';    // on its side: neither upright nor over
  if (z >= 0.100) return 'standing';
  if (z >= 0.075) return 'crouched';
  if (z >= 0.052) return 'seated';
  return 'fallen';
}

function deOrigin(roots) {
  const [x0, y0] = roots[0];
  const q0 = [roots[0][3], roots[0][4], roots[0][5], roots[0][6]];
  // Only the YAW is removed. Taking out the whole orientation would stand a
  // duck that starts mid-roll upright and destroy the motion.
  const yaw0 = Math.atan2(2 * (q0[0] * q0[3] + q0[1] * q0[2]),
                          1 - 2 * (q0[2] * q0[2] + q0[3] * q0[3]));
  const c = Math.cos(-yaw0), s2 = Math.sin(-yaw0);
  const inv = [Math.cos(-yaw0 / 2), 0, 0, Math.sin(-yaw0 / 2)];
  const mul = (a, b) => [
    a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
    a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
    a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
    a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]];
  return roots.map(r => {
    const dx = r[0] - x0, dy = r[1] - y0;
    const q = mul(inv, [r[3], r[4], r[5], r[6]]);
    return [+(dx * c - dy * s2).toFixed(5), +(dx * s2 + dy * c).toFixed(5), +r[2].toFixed(5),
            +q[0].toFixed(5), +q[1].toFixed(5), +q[2].toFixed(5), +q[3].toFixed(5)];
  });
}

const out = {
  format: 'duck-intent-clips/2',
  hz: C.tickHz,
  joints: C.jointNames.filter(n => n !== 'mouth'),
  clips: {},
};
for (const spec of SPECS) {
  const r = await capture(spec);
  const roots = deOrigin(r.roots);
  const last = roots[roots.length - 1];
  out.clips[spec.id] = {
    frames: r.frames,
    // Per frame: x, y, z, then the trunk quaternion (w, x, y, z). NOT a yaw
    // scalar — a duck that rolls or flips has an orientation a single angle
    // cannot carry, and roulade and back_roll both do.
    roots,
    // Unwrapped total rotation. The one summary kept, because it cannot be
    // recovered from the last quaternion: that only gives an angle modulo 2pi.
    netYaw: +r.netYaw.toFixed(5),
    // `hold` is the only clip that loops. Everything else happens once.
    loops: spec.id === 'hold',
    policy: spec.policy,
    authored: !!spec.track,
    // MEASURED FROM THE TRUNK, NOT ASSERTED. These were hardcoded from the
    // clip's own id — `stand` starts seated, `sit` ends seated, everything else
    // standing — which is a label describing what the clip was MEANT to do. A
    // review measured the corpus and found them fabricated: step_up ends at
    // 49 mm having fallen over, and was labelled "standing" because its id is
    // not "sit". A posture claim next to a recording that contradicts it is
    // worse than no claim, because something downstream will chain on it.
    startsFrom: postureOver(roots, 0, 10),
    endsIn: postureOver(roots, Math.max(0, roots.length - 15), roots.length),
    // Whose policy this is. Absent for Pollen's own; the app resolves the
    // real answer from the policy's fingerprint rather than from this string.
    ...(spec.credit ? { credit: spec.credit } : {}),
  };
  console.log(`CLIP ${spec.id.padEnd(12)} ${String(r.frames.length).padStart(4)} ticks  `
    + `z ${roots[0][2].toFixed(3)}\u2192${last[2].toFixed(3)}  `
    + `\u0394(${last[0].toFixed(3)}, ${last[1].toFixed(3)}) m  netYaw ${r.netYaw.toFixed(3)}`);
}
fs.writeFileSync('duck-intent-clips.json', JSON.stringify(out));
console.log(`wrote ${Object.keys(out.clips).length} clips`);
