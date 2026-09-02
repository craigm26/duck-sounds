// The duck bench's core: physics, the control loop, and the answers — with no
// idea what machine it is on.
//
// WHY THIS IS ITS OWN FILE. Every measurement Duck Studio shows came from a
// bench on a desk, reached over Tailscale, because an iPhone has no MuJoCo and
// no onnxruntime. That sentence is a build decision, not a law: MuJoCo compiles
// to WebAssembly and a 197,774-parameter MLP is four matrix multiplies. What
// actually stopped the phone was that the bench and Node were the same file —
// `fs.readFileSync` for the plant, `os.cpus()` for the core count,
// `onnxruntime-node` for the forward pass, `http.createServer` for the door.
// None of that is physics.
//
// So this file holds the physics and takes everything else as `env`. Node fills
// `env` with fs, os, crypto and onnxruntime; a browser fills it with fetch,
// `navigator.hardwareConcurrency`, `crypto.subtle` and a forward pass over the
// canonical parameter bytes. The plant, the control loop, the observation, the
// endpoints and every sentence they return are the same code in both, which is
// the only reason a number measured on a phone is comparable to one measured on
// the desk.
//
// IT IS A MOVE, AND IT WAS PROVED TO BE ONE. `bench_parity.mjs` replays a fixed
// script of sixty requests and compares every leaf of every answer; the split
// was accepted only when the old file and this one differed in nothing but the
// two /health fields that were deliberately added. A refactor of a file whose
// output is trajectories cannot be reviewed by reading it.
import { makeLoop } from './duckloop.mjs';
// THE ONE PIECE OF INFERENCE THE CORE OWNS, AND ONLY FOR /tune. Everything
// else asks the shell for a session, because a shell may have onnxruntime and a
// browser may not. A fold cannot be asked for that way: onnxruntime runs a
// graph and will not let you change one, so scoring a per-joint gain and trim
// means holding the parameters and multiplying them here. Both shells can reach
// this file — it is 60 lines of arithmetic with no machine in it.
import { loadParameters, foldParameters, forward, FLOAT_COUNT } from './policyforward.mjs';
// THE STAIRS, AND THE ONE EPISODE THAT SCORES A CLIMB ON THEM.
//
// `climb_score.mjs` is not a second opinion about anything: it IS the episode
// climb/rig3.mjs's scoreSaved() runs and the one climb/robust.mjs's 14-cell
// grid is decided on, imported rather than transcribed, so /climb answers the
// number the audit published instead of a number that looks like it.
// climb_score.mjs in turn imports `./stairs.js`, `./climb_event.mjs` and
// `./climb_servo.mjs`, which are re-export shims in sim/ and the real files in
// the phone bundle's flat assets/ — the same trick `./duckloop.mjs` uses, and
// the reason both make_phonebench scripts had to learn four new filenames.
import { makeClimbRig, criteria as climbCriteria, checkBounds as climbCheckBounds,
         checkIntent as climbCheckIntent, optsOf as climbOptsOf,
         intentHashPayload, intentIsolate, intentStepCount, gridCells,
         reachedFlight as climbReachedFlight,
         CRITERION_SENTENCE, UPRIGHT_TAIL_MIN, CLEAR_BONUS, CEILING_ABOVE,
         RISER_X as CLIMB_RISER_X, LATERAL as CLIMB_LATERAL,
         DECLARED_BOUNDS as CLIMB_BOUNDS } from './climb_score.mjs';

/**
 * A bench, over the machine `env` describes.
 *
 * env = {
 *   sceneName:   string — the plant to load, by asset name
 *   mujoco:      the instantiated MuJoCo module (npm `mujoco` in Node,
 *                mujoco.js in a page). NOT loaded here: the two shells get it
 *                by entirely different routes and neither import works in the
 *                other.
 *   readAsset:   (name) -> Uint8Array | Promise<Uint8Array>
 *   listPolicies:() -> [string] — the whole allow-list, by the name /policy
 *                takes. SYNCHRONOUS, because /health reports it inline.
 *   sha256:      (bytes) -> hex | Promise<hex>
 *   cores:       number
 *   makeSession: (bytes, name) -> { run(Float32Array) -> Float32Array,
 *                                   reference?: [number] } | Promise<same>
 *   scratch:     { has(name), get(name), set(name, bytes), delete?(name) }
 *   host:        { kind: 'desk' | 'phone', device: string, engine: string }
 * }
 */
// ── Pollen's reward, as this plant can answer it ─────────────────────────
//
// WHOSE ARITHMETIC THIS IS. Every formula below is a transcription of
// `RunMetrics.swift` in StudioKit, which is itself read out of
// `microduck_velocity_env_cfg.py` in pollen-robotics/microduck_rl. Not one of
// them is invented here, and the two transcriptions are held together by a
// shared fixture rather than by care: `sim/tune_parity.mjs` and
// `TuneTraceParityTests.swift` compute the six terms from the SAME fifty-tick
// trace and must agree to 1e-9. That gate is the only reason this file is
// allowed to hold a second copy of the reward at all — two transcriptions of
// one config is otherwise exactly how a search comes to optimise a hill the
// rest of the app cannot see, with both numbers looking plausible.
//
// THE WEIGHTS ARE NOT HERE, ON PURPOSE. /tune answers per-tick MEANS and the
// client weighs them (`DuckTuner.terms`). A bench that returned one weighted
// number could change what a reward means without anything upstream noticing.

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
 * `variable_posture`'s per-joint tolerance, radians — RunMetrics's `legStd`.
 * `null` for the four joints the config's regex drops: the neck and head are
 * driven by a pose command that rides in the observation, so pulling them
 * home as well would teach the policy to ignore it.
 */
export function legStd(name, standing) {
  if (name.includes('hip_yaw')) return standing ? 0.1 : 0.3;
  if (name.includes('hip_roll')) return 0.05;
  if (name.includes('hip_pitch')) return standing ? 0.15 : 0.4;
  if (name.includes('knee')) return standing ? 0.15 : 0.4;
  if (name.includes('ankle')) return standing ? 0.1 : 0.25;
  return null;
}

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
 * The six terms of `microduck_velocity_env_cfg` this plant can answer, as
 * per-tick SUMS plus the tick counts they are divided by.
 *
 * SUMS AND NOT MEANS, because /tune pools several episodes into one figure
 * and a mean of means over episodes of unequal length is not the per-tick
 * mean of anything. `action_rate_l2` has its own count: it differences
 * consecutive decisions, so an episode of n ticks contributes n−1 of them.
 */
export function rewardSums(trace, jointNames, home) {
  const s = { upright: 0, track_linear_velocity: 0, track_angular_velocity: 0,
              pose: 0, body_ang_vel: 0, action_rate_l2: 0 };
  for (let i = 0; i < trace.length; i++) {
    const f = trace[i];
    const q = [f.root[3], f.root[4], f.root[5], f.root[6]];
    const t = twistOf(f.root, f.qvel);
    const [cvx, cvy, cvyaw] = f.command;

    // upright = exp(−|projected gravity xy|² / std²), std² = 0.05 in this config.
    s.upright += Math.exp(-gravityXYSquared(q) / 0.05);

    // The commanded twist against the actual one, in the trunk's frame.
    const ex = cvx - t[0], ey = cvy - t[1];
    s.track_linear_velocity += Math.exp(-(ex * ex + ey * ey + t[2] * t[2]) / 0.1);
    const ez = cvyaw - t[5];
    s.track_angular_velocity += Math.exp(-(ez * ez + t[3] * t[3] + t[4] * t[4]) / 0.5);

    // body_ang_vel = ωx² + ωy² in the WORLD frame — yaw is deliberately free.
    const world = rotate(q, [t[3], t[4], t[5]]);
    s.body_ang_vel += world[0] * world[0] + world[1] * world[1];

    // `variable_posture`: the tolerance loosens the moment a velocity is
    // commanded, and the threshold is the config's own 0.01.
    const speed = Math.hypot(cvx, cvy) + Math.abs(cvyaw);
    const standing = speed < 0.01;
    let sum = 0, count = 0;
    for (let k = 0; k < 14; k++) {
      const std = legStd(jointNames[k], standing);
      if (std === null) continue;
      const d = f.joints[k] - home[k];
      sum += (d * d) / (std * std);
      count++;
    }
    s.pose += count ? Math.exp(-sum / count) : 0;

    // action_rate_l2 = Σ(aₜ − aₜ₋₁)² over the network's own fourteen outputs.
    if (i > 0) {
      const a = f.action, b = trace[i - 1].action;
      for (let k = 0; k < 14; k++) { const d = a[k] - b[k]; s.action_rate_l2 += d * d; }
    }
  }
  return { sums: s, ticks: trace.length, rateTicks: Math.max(trace.length - 1, 1) };
}

/** Start of the driven span to the end of it, in the plane. `readTravel`'s number. */
export function netDisplacement(trace) {
  if (trace.length < 2) return 0;
  const a = trace[0].root, b = trace[trace.length - 1].root;
  return Math.hypot(b[0] - a[0], b[1] - a[1]);
}

/**
 * HOW FAR IT GOT IN THE DIRECTION IT WAS TOLD TO GO.
 *
 * WHY NOT JUST THE DISTANCE. A duck that falls over travels; a duck that
 * turns in a circle travels; a duck driven backwards by a bad trim travels.
 * The number a search needs beside a reward is the one that catches a
 * candidate which has quietly stopped WALKING, and only a signed projection
 * onto the commanded direction is that number — `hypot` calls a metre
 * backwards a metre of progress.
 *
 * It is RunMetrics's "Forward — along the heading it started on", with the
 * heading taken at the first DRIVEN tick (the settle is over by then) and the
 * direction taken from the schedule rather than assumed to be +x: a schedule
 * that commands vy is asking the duck to walk sideways, and a projection onto
 * the wrong axis would score that as standing still.
 *
 * A schedule with no linear command at all has no direction to project onto,
 * so the plain net displacement is reported and the answer says so.
 */
export function travelledAlongCommand(trace) {
  if (trace.length < 2) return 0;
  let sx = 0, sy = 0;
  for (const f of trace) { sx += f.command[0]; sy += f.command[1]; }
  const magnitude = Math.hypot(sx, sy);
  if (!(magnitude > 1e-12)) return netDisplacement(trace);
  const ux = sx / magnitude, uy = sy / magnitude;
  const a = trace[0].root, b = trace[trace.length - 1].root;
  // The heading the driven span started on, out of the quaternion.
  const [w, x, y, z] = [a[3], a[4], a[5], a[6]];
  const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  const cos = Math.cos(yaw), sin = Math.sin(yaw);
  const wx = ux * cos - uy * sin, wy = ux * sin + uy * cos;
  return (b[0] - a[0]) * wx + (b[1] - a[1]) * wy;
}

/**
 * Finite AND plausible. A diverged MuJoCo state yields enormous DOUBLES —
 * measured at 6.8e37 — which `Number.isFinite` accepts happily. Past a
 * thousand is not a duck in any pose. Shared by the training capture and the
 * /tune trace, so the two cannot disagree about what a diverged tick is.
 */
export const sane = v => Number.isFinite(v) && Math.abs(v) < 1000;

/**
 * The residual a search may ask this bench to fold, transcribed from
 * DuckTuner.TuningVector in Microduck Studio. Kept as numbers so nothing
 * retypes them; a test on that side pins the same three.
 */
export const TUNE_ENVELOPE = { gainLower: 0.7, gainUpper: 1.3, offsetLimit: 0.05 };

/** Every term this bench knows how to compute, in the config's own order. */
export const TUNE_TERMS = ['upright', 'track_linear_velocity', 'track_angular_velocity',
                    'pose', 'body_ang_vel', 'action_rate_l2'];

/**
 * The terms of the velocity config this PLANT cannot answer, by name and with
 * the reason — asked for, and refused rather than dropped.
 *
 * A SHORTER LIST IS NOT A BETTER ONE. `RunMetrics.Task.unevaluable` says
 * something similar about a RECORDING; this says it about `scene.mjb`, which
 * is a different claim and in one case a different reason: the plant does
 * carry `root_angmom`, so `angular_momentum` is refused not because the
 * sensor is missing but because nothing in this family reads that term's
 * weight out of the config, and picking one would be inventing a reward.
 */
export const TUNE_REFUSALS = new Map([
  ['air_time', 'reads the foot-contact sensor, and this plant has none: its six sensors are '
             + 'orientation, angular-velocity, imu_ang_vel, imu_lin_vel, imu_accel, root_angmom'],
  ['foot_clearance', 'reads the foot sites against the contact sensor; the sensor is not there'],
  ['foot_swing_height', 'the same sensor'],
  ['foot_slip', 'the same sensor'],
  ['self_collisions', 'no collision sensor in this plant'],
  ['dof_pos_limits', 'scores against soft limits — a fraction of the model\'s travel that '
                   + 'neither duckkit nor this bench ships, so the fraction would have to be '
                   + 'invented'],
  ['angular_momentum', 'the plant DOES carry root_angmom, so this one is refused for a '
                     + 'different reason: nothing here reads its weight out of the config, and '
                     + 'picking one would be inventing a reward rather than reading Pollen\'s'],
]);

export async function makeBench(env) {
  const C = JSON.parse(new TextDecoder().decode(await env.readAsset('duckkit-constants.json')));
  const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);
  /**
   * THE POLICY THAT DOES THE STANDING, RESOLVED BY ROLE RATHER THAN BY FILENAME.
   *
   * `ensureStanding` gates /reset, /intent, /stop, /state, /record, /measure and
   * /perform on this one file, so a bench that cannot find it can do nothing at
   * all. It was pinned to `BEST_alpha_stand.onnx` — the name a training run left
   * on the artefact — while the app bundles the identical network under its ROLE
   * name, `alpha_stand.onnx`. A bench assembled from the app's own assets
   * therefore booted, answered /health, and then refused every other endpoint
   * with `unknown policy: BEST_alpha_stand.onnx`, which reads like a missing
   * policy rather than like a naming convention.
   *
   * Role name first, training name second, and /health says which was found.
   */
  const STAND_TRIED = ['alpha_stand.onnx', 'BEST_alpha_stand.onnx'];
  const STAND = (() => {
    const known = catalogue();
    const found = STAND_TRIED.find(name => known.has(name));
    if (!found) {
      throw new Error(`this bench has no standing policy: it needs one of ${STAND_TRIED.join(' or ')}`);
    }
    return found;
  })();
  // The settle every recorded clip opens with — half a second under the standing
  // policy, so the duck is on its feet before anything asks it to move. The live
  // world opens with the same one, or /intent's first tick would be steering a
  // duck that is still falling the 1 mm from its drop height.
  const SETTLE_TICKS = 25;

  const mj = env.mujoco;
  // WHICH WORLD. The canon plant is scene.mjb and every recorded clip in duckkit
  // claims to come from it, so it is the default and adding bodies to it is not
  // on. A bench that wants something to pick up asks for a different scene:
  //   DUCKBENCH_SCENE=scene_grasp.mjb node duckbench.mjs
  const SCENE = env.sceneName;
  const SCENE_BYTES = await env.readAsset(SCENE);
  mj.FS.writeFile('/s.mjb', new Uint8Array(SCENE_BYTES));
  // AND WHICH WORLD IT ACTUALLY WAS, SAID OUT LOUD IN EVERY ANSWER THAT CARRIES
  // A MEASUREMENT. A caller that keeps a result — Duck Studio keeps them beside
  // the draft that caused them, forever — has to be able to say which plant
  // produced it, and it can only say what this bench tells it. Until now
  // /perform and /record told it nothing, so the app wrote a placeholder of its
  // own and then printed the placeholder as though it were a fact about a world.
  // A filename alone is not enough either: `sim/scene.mjb` and `site/scene.mjb`
  // share a name and differ in bytes (see PLANT.md), and it was that pair that
  // made this a bug rather than a nicety. So the digest goes with the name.
  const PLANT = SCENE.slice(SCENE.lastIndexOf('/') + 1);
  const PLANT_DIGEST = await env.sha256(SCENE_BYTES);
  const model = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
  const data = new mj.MjData(model);

  /** The fourteen joints a policy drives, in the order the observation wants them. */
  const DUCK_JOINTS = C.jointNames.filter(n => n !== 'mouth');

  const namedIndex = (count, read, name) => {
    for (let i = 0; i < count; i++) if (read(i) === name) return i;
    return -1;
  };

  /**
   * EVERY DUCK IN THIS WORLD, FOUND BY WALKING THE MODEL.
   *
   * A duck is a body whose name is `trunk_base` or ends in `_trunk_base` and
   * which carries a free joint, and everything else about it — its fourteen
   * joints, its fourteen actuators, its gyro — is that body's prefix followed by
   * the name the single-duck scene uses. That prefix is not decoration: every
   * name in MJCF is a global, so a second copy of the duck subtree collides with
   * the first on all thirty-odd of them, and prefixing is how
   * `build_multiduck.py` gets N of them into one model at all.
   *
   * WALKED RATHER THAN CONFIGURED for the reason GRASPABLES is walked: a scene
   * with three ducks in it says so because the ducks are there. Nothing has to be
   * told twice, `scene.mjb` keeps answering with exactly one duck named `duck`,
   * and a caller that never heard of a second one reads the same answers it
   * always did.
   *
   * A MISSING PIECE IS FATAL HERE RATHER THAN SILENT LATER. A trunk whose gyro
   * cannot be found used to fall back on sensor address 0, which means feeding a
   * policy some other sensor's three numbers as its angular velocity — a lie that
   * produces a plausible-looking rollout. Boot is the place to say so.
   */
  function discoverDucks() {
    const found = [];
    for (let b = 0; b < model.nbody; b++) {
      const body = model.body(b).name;
      if (body !== 'trunk_base' && !body.endsWith('_trunk_base')) continue;
      const prefix = body.slice(0, body.length - 'trunk_base'.length);
      // `trunk_base` alone is the canon scene's duck and has no prefix; it is
      // named `duck` so that every answer can name a duck even there.
      const name = prefix ? prefix.slice(0, -1) : 'duck';
      let freeQpos = -1, freeDof = -1;
      for (let j = 0; j < model.njnt; j++) {
        if (model.jnt_type[j] !== 0 || model.jnt_bodyid[j] !== b) continue;
        freeQpos = model.jnt_qposadr[j]; freeDof = model.jnt_dofadr[j];
        break;
      }
      if (freeQpos < 0) throw new Error(`${body} has no free joint: it is bolted to the world`);
      const qpos = [], dof = [], ctrl = [];
      for (const joint of DUCK_JOINTS) {
        const j = namedIndex(model.njnt, i => model.jnt(i).name, prefix + joint);
        if (j < 0) throw new Error(`joint missing from the model: ${prefix}${joint}`);
        qpos.push(model.jnt_qposadr[j]); dof.push(model.jnt_dofadr[j]);
        const a = namedIndex(model.nu, i => model.actuator(i).name, prefix + joint);
        if (a < 0) throw new Error(`actuator missing from the model: ${prefix}${joint}`);
        ctrl.push(a);
      }
      const gyro = namedIndex(model.nsensor, i => model.sensor(i).name, prefix + 'imu_ang_vel');
      if (gyro < 0) throw new Error(`sensor missing from the model: ${prefix}imu_ang_vel`);
      // Where this duck starts: the free joint's own qpos0, which is the `pos`
      // the MJCF gave its trunk. `r4` is declared further down and this runs at
      // load, so the rounding is done longhand rather than reaching into the
      // temporal dead zone.
      const spawn = [0, 1, 2].map(k => Math.round(model.qpos0[freeQpos + k] * 10000) / 10000);
      found.push({ name, prefix, spawn, gyro: model.sensor(gyro).adr,
                   joints: { qpos, dof, freeQpos, freeDof }, ctrl });
    }
    if (!found.length) throw new Error('this world has no duck in it');
    return found;
  }
  const DUCKS = discoverDucks();
  const DUCK_NAMES = DUCKS.map(d => d.name);
  /** Every duck's root address, so nothing else in the world mistakes one for a prop. */
  const DUCK_ROOTS = new Set(DUCKS.map(d => d.joints.freeQpos));

  // THE TWO FINDERS ARE PINNED TO EACH OTHER AT BOOT. `findDuckJoints` is
  // duckloop's, shared with the browser and with every other runner in this
  // directory, and it knows only about an unprefixed duck; `discoverDucks` above
  // is the generalisation. Where both apply — the canon one-duck scene — they
  // must agree, or the multi-duck path has quietly started driving different
  // joints than the recorded corpus came from.
  {
    const plain = DUCKS.find(d => d.prefix === '');
    if (plain) {
      const canon = findDuckJoints(model);
      const same = canon.freeQpos === plain.joints.freeQpos
                && canon.freeDof === plain.joints.freeDof
                && canon.qpos.every((v, i) => v === plain.joints.qpos[i])
                && canon.dof.every((v, i) => v === plain.joints.dof[i]);
      if (!same) throw new Error('discoverDucks disagrees with duckloop findDuckJoints about the duck');
    }
  }

  // THE BALL IS THE ONLY OTHER THING WORTH ADDRESSING IN THIS WORLD. It is
  // Pollen's own (radius 0.05, condim 6 so it actually decelerates), and a
  // steering loop needs to put it somewhere and then be scored on reaching it.
  const BALL = (() => {
    for (let j = 0; j < model.njnt; j++) {
      if (model.jnt_type[j] !== 0) continue;                 // mjJNT_FREE
      const adr = model.jnt_qposadr[j];
      if (DUCK_ROOTS.has(adr)) continue;                     // a duck, not a prop
      if (model.body(model.jnt_bodyid[j]).name === 'ball') return { adr, dof: model.jnt_dofadr[j] };
    }
    return null;
  })();
  const BALL_RADIUS = 0.05;

  /**
   * Every free body in this world that is not the duck and not the ball — the
   * things a fetch or a drag is about.
   *
   * FOUND BY WALKING THE MODEL, not by a list kept in step with the XML. A scene
   * with a broom in it says so because the broom is there; nothing has to be
   * told twice, and a scene without one reports an empty list rather than a lie.
   */
  const GRASPABLES = (() => {
    const found = [];
    for (let j = 0; j < model.njnt; j++) {
      if (model.jnt_type[j] !== 0) continue;                 // mjJNT_FREE
      const adr = model.jnt_qposadr[j];
      // A SECOND DUCK IS NOT A GRASPABLE. Its trunk is a free body like any
      // other and would otherwise be listed as something to pick up, which is
      // both wrong and — since /health publishes this list — a claim about the
      // world that the world does not support.
      if (DUCK_ROOTS.has(adr)) continue;
      const body = model.jnt_bodyid[j];
      const name = model.body(body).name;
      if (name === 'ball') continue;                         // it has its own door
      // `r4` is declared further down and this runs at load, so the rounding is
      // done longhand rather than reaching into the temporal dead zone.
      const mass = Math.round(model.body_mass[body] * 10000) / 10000;
      found.push({ name, adr, dof: model.jnt_dofadr[j], body, mass });
    }
    return found;
  })();

  /** Where a graspable is now: position and whether it has been moved. */
  function graspableState(d) {
    return GRASPABLES.map(g => ({
      name: g.name,
      mass: g.mass,
      at: [r4(d.qpos[g.adr]), r4(d.qpos[g.adr + 1]), r4(d.qpos[g.adr + 2])],
    }));
  }

  function ballOf(d) {
    if (!BALL) return null;
    return [d.qpos[BALL.adr], d.qpos[BALL.adr + 1], d.qpos[BALL.adr + 2]].map(r4);
  }

  /** Put the ball down and stop it dead. */
  function placeBall(d, x, y, z = BALL_RADIUS) {
    if (!BALL) throw new Error('this world has no ball');
    d.qpos[BALL.adr] = x; d.qpos[BALL.adr + 1] = y; d.qpos[BALL.adr + 2] = z;
    d.qpos[BALL.adr + 3] = 1;
    for (let k = 4; k < 7; k++) d.qpos[BALL.adr + k] = 0;
    for (let k = 0; k < 6; k++) d.qvel[BALL.dof + k] = 0;
    mj.mj_forward(model, d);
  }
  /** Every .onnx this bench will run, by bare name — the whole allow-list. */
  function catalogue() {
    const out = new Map();
    // The shell walks its own storage and hands back NAMES; `readAsset` takes
    // one of those names back. A directory scan and a fetched manifest are the
    // same answer to the core, which is the whole point of the split.
    for (const name of env.listPolicies()) out.set(name, name);
    return out;
  }

  const sessions = new Map();

  /**
   * A policy, loaded once, WITH THE NEUTRAL POSE IT WAS TRAINED AGAINST.
   *
   * The reference is not decoration. A policy's action is an offset from its own
   * neutral, and its observation's joint block is a deviation from that same
   * pose; Pollen's ten files all declare one equal to HOME, which is why using
   * HOME everywhere went unnoticed, but the community `headspin.onnx` declares
   * neck_pitch 0.220 and head_pitch 0.680 where HOME has 0.349 and 0.349. Every
   * other runner here already honours it (record_intents.mjs, ac_check*.mjs) and
   * this one did not — which mattered the moment /policy let a caller swap to
   * that very file mid-run.
   */
  async function policy(name) {
    // An uploaded policy is already in `sessions` under its own filename and is
    // not in the catalogue, which only walks what shipped. Check there first.
    if (name.startsWith('uploaded-') && sessions.has(`${name}.onnx`)) {
      return sessions.get(`${name}.onnx`);
    }
    const known = catalogue();
    if (!known.has(name)) throw new Error(`unknown policy: ${name}`);
    const file = known.get(name);
    if (!sessions.has(file)) {
      // THE SHELL DECIDES WHAT A SESSION IS. onnxruntime here, the canonical
      // parameter bytes and a hand-written forward pass in a browser — and it
      // is the shell, holding the file, that reads the neutral pose the policy
      // declares. A session that declares none inherits HOME by identity, which
      // is what /policy reports on.
      const session = await env.makeSession(await env.readAsset(file), name);
      sessions.set(file, { name, net: session, reference: session.reference ?? HOME });
    }
    return sessions.get(file);
  }

  /**
   * ONE DUCK, PUT BACK WHERE IT STARTED, WITHOUT DISTURBING THE WORLD AROUND IT.
   *
   * `mj_resetData` cannot do this: it resets everything, which in a shared world
   * means teleporting the other ducks and every prop as well. Writing this duck's
   * own qpos and qvel is the only way to give one duck a fresh start while the
   * rest of the scene carries on, and it is what /reset with a `duck` name does.
   *
   * The spawn is the free joint's qpos0, so a duck lands back on the mark its
   * MJCF gave it rather than on the origin, which in a multi-duck scene is
   * somebody else's mark.
   */
  function placeDuck(d, duck, z = 0.1231) {
    const j = duck.joints, f = j.freeQpos;
    d.qpos[f] = duck.spawn[0]; d.qpos[f + 1] = duck.spawn[1]; d.qpos[f + 2] = z;
    d.qpos[f + 3] = 1; d.qpos[f + 4] = 0; d.qpos[f + 5] = 0; d.qpos[f + 6] = 0;
    for (let k = 0; k < 6; k++) d.qvel[j.freeDof + k] = 0;
    for (let i = 0; i < 14; i++) {
      d.qpos[j.qpos[i]] = HOME[i];
      d.qvel[j.dof[i]] = 0;
      d.ctrl[duck.ctrl[i]] = HOME[i];
    }
  }

  /**
   * The whole world back to its start, with every duck dropped from `z`.
   *
   * All of them, not just the one a caller is about to drive: a duck left in
   * whatever heap the last rollout ended in is still in the scene, still touching
   * the floor the driven duck walks on, and would make the run unrepeatable.
   */
  function reset(d, z = 0.1231) {
    mj.mj_resetData(model, d);
    for (const duck of DUCKS) placeDuck(d, duck, z);
    mj.mj_forward(model, d);
  }

  // The plant's own timestep, read out of the plant rather than assumed: one
  // mj_step of the settled duck advances the clock by exactly this. The control
  // loop below runs at C.tickHz and takes 1/tickHz worth of substeps per tick —
  // which is the 4 that used to sit in the loop as a bare literal, and which is
  // now checked against the model every boot instead of hoped for.
  reset(data);
  mj.mj_step(model, data);
  const TIMESTEP = data.time;
  const SUBSTEPS = Math.round(1 / C.tickHz / TIMESTEP);
  if (!(SUBSTEPS >= 1) || Math.abs(SUBSTEPS * TIMESTEP - 1 / C.tickHz) > 1e-9) {
    throw new Error(`${C.tickHz} Hz control does not divide a ${TIMESTEP} s timestep`);
  }

  /**
   * WHAT ONE CONTROL TICK OF PHYSICS COSTS ON THIS MACHINE, MEASURED.
   *
   * The whole question about a bench in a phone is whether the phone is fast
   * enough, and that is not a question anybody should answer from a spec sheet.
   * This times the SUBSTEPS mj_steps a control tick takes, on this plant, on this
   * hardware, over a hundred ticks after a warm-up — the physics only, with no
   * network in it, because the network is the part that differs between the two
   * shells. Real time is 20 ms a tick at 50 Hz; a number under that means this
   * machine can run the duck faster than the duck lives.
   *
   * It costs about a fifth of a second at boot and is reported in /health, where
   * a saved measurement can carry it.
   */
  const TICK_MILLIS = (() => {
    const probe = new mj.MjData(model);
    const now = () => (typeof performance === 'object' ? performance.now() : Date.now());
    for (let i = 0; i < 20 * SUBSTEPS; i++) mj.mj_step(model, probe);
    const started = now();
    const ticks = 100;
    for (let t = 0; t < ticks; t++) for (let s = 0; s < SUBSTEPS; s++) mj.mj_step(model, probe);
    const each = (now() - started) / ticks;
    probe.free?.();
    return Math.round(each * 1000) / 1000;
  })();

  /** The machine, as the shell describes it plus the one number measured here. */
  const HOST = {
    kind: env.host?.kind ?? 'desk',
    device: env.host?.device ?? 'unknown',
    engine: env.host?.engine ?? 'unknown',
    tickMillis: TICK_MILLIS,
  };

  /**
   * One control tick on `d`: observe, run the policy, clamp to servo travel, step.
   *
   * THE SAME TICK FOR EVERY CALLER. /record, /measure and /intent all come
   * through here, so a policy that measured 16/16 behaves identically when
   * quackd steers it live. Two loops would drift apart on the first fix.
   */
  /**
   * One control step's WORTH OF COMMANDS, for ONE duck, WITHOUT ADVANCING TIME.
   *
   * SPLIT OUT OF `tick` BECAUSE A WORLD HAS ONE CLOCK AND MAY HAVE SEVERAL DUCKS.
   * If each duck stepped physics after choosing its own action, then in a
   * two-duck world duck A would act, time would move, and duck B would then act
   * on a world half a control period older than the one A saw — a stagger nobody
   * asked for and which would be invisible in the answers. Every duck writes its
   * ctrl against the same instant, and then time moves once for all of them. That
   * is the difference between N ducks in a world and N worlds.
   *
   * `capture`, when given, collects the pairs a policy would have to reproduce to
   * perform this motion by itself: the 61-wide OBSERVATION the network was shown,
   * and the EFFECTIVE ACTION — what it would have had to output for the joints to
   * end up where the authored track put them. That second number is
   * `ctrl - reference`, because a pure policy's target is `reference + action`
   * while an authored run's is `reference + action + offset`. Cloning the sum is
   * cloning the motion.
   *
   * The observation is the untouched one, for the reason the comment below
   * already gives: a network told what it asked for rather than what it is
   * standing in learns something that only works in a recording.
   */
  async function actuate(d, duck, loaded, last, cmd, offsets = null, blend = 1, capture = null,
                         expert = null, teacherShare = 0, jitter = 0) {
    const { net, reference } = loaded;
    const J = duck.joints, f = J.freeQpos, g = duck.gyro;
    const q = [d.qpos[f + 3], d.qpos[f + 4], d.qpos[f + 5], d.qpos[f + 6]];
    const jp = [], jv = [];
    for (let k = 0; k < 14; k++) { jp.push(d.qpos[J.qpos[k]]); jv.push(d.qvel[J.dof[k]]); }
    const obs = buildObs([d.sensordata[g], d.sensordata[g + 1], d.sensordata[g + 2]],
                         projectedGravity(q), jp, jv, last, cmd, reference);
    const action = Array.from(await net.run(obs));
    for (let k = 0; k < 14; k++) {
      // AUTHORED OFFSETS RIDE ON TOP OF THE POLICY, exactly as record_intents.mjs
      // applies them: the policy keeps its balance and the track leans on it. The
      // OBSERVATION is untouched — the network is told what it is actually
      // standing in, not what the author asked for, because handing a policy its
      // own request back as state is how a motion looks fine in a recording and
      // falls over on a robot.
      const base = reference[k] + action[k];
      const target = offsets ? base + (offsets[k] - HOME[k]) * blend : base;
      // THIS DUCK'S OWN ACTUATORS, BY INDEX LOOKED UP AT BOOT. `ctrl[k]` was
      // right while there was one duck and every actuator in the model belonged
      // to it; with two, actuator 3 is the FIRST duck's left knee whichever duck
      // is being driven, and a second duck steered that way would have moved the
      // first one's legs.
      d.ctrl[duck.ctrl[k]] = Math.min(Math.max(target, LO[k]), HI[k]);
    }
    // AFTER THE CLAMP, NOT BEFORE. What the joints were actually commanded is the
    // thing a clone has to reproduce; a label taken before the clamp teaches the
    // network to ask for angles the hardware refuses.
    // What the next observation will be told the last action was. For a plain
    // rollout that is the acting network's own output; while CAPTURING it must be
    // the EFFECTIVE action instead — see the note where it is assigned.
    let fedBack = action;

    if (capture) {
      // WHAT A POLICY WOULD HAVE TO OUTPUT AT THIS STATE. A pure policy's target
      // is `reference + action`; an authored run's is that plus the offset. So
      // the label is `ctrl - reference`, taken AFTER the clamp, because a label
      // taken before it teaches the network to ask for angles the hardware
      // refuses.
      const effective = [];
      if (expert) {
        // THE TEACHER LABELS EVERY STATE, whoever drove the duck into it. That
        // is the pairing a clone actually needs — not "what happened on the good
        // run" but "what to do from here".
        const teach = Array.from(await expert.net.run(obs));
        const taught = [];
        for (let k = 0; k < 14; k++) {
          const base = expert.reference[k] + teach[k];
          const target = offsets ? base + (offsets[k] - HOME[k]) * blend : base;
          const clamped = Math.min(Math.max(target, LO[k]), HI[k]);
          taught.push(clamped);
          effective.push(clamped - expert.reference[k]);
        }
        // The teacher can also ACT, on a share of steps. A first clone is bad
        // enough to tear the integrator apart within a few steps — measured:
        // 2591 of 2720 unusable — so letting it drive alone yields nothing to
        // learn from.
        if (Math.random() < teacherShare) {
          for (let k = 0; k < 14; k++) d.ctrl[duck.ctrl[k]] = taught[k];
        }
      } else {
        for (let k = 0; k < 14; k++) effective.push(d.ctrl[duck.ctrl[k]] - reference[k]);
      }

      // A NON-FINITE STEP IS NOT A TRAINING PAIR, AND MUST NOT LEAVE AS ONE.
      // `JSON.stringify` writes NaN and Infinity as `null`, so a diverged step
      // travels as a hole a consumer reads back as NaN and trains on in silence.
      //
      // AND FINITE IS NOT PLAUSIBLE. A diverged MuJoCo state yields enormous
      // DOUBLES — measured at 6.8e37 — which `Number.isFinite` accepts happily;
      // train on those and the normaliser's deviation becomes 1e36, every real
      // observation flattens to zero, and the loss is NaN by the second epoch.
      // The bound is generous: past a thousand is not a duck in any pose.
      const row = Array.from(obs);
      if (row.every(sane) && effective.every(sane)) {
        capture.push({ obs: row, action: effective });
      } else {
        capture.rejected = (capture.rejected || 0) + 1;
      }

      // THE LABEL IS ALSO WHAT GETS FED BACK, and getting this wrong is what made
      // three clones in a row saturate a joint within half a second.
      //
      // `lastAction` is part of the observation. During capture the acting
      // network was the teacher, so the raw action fed back carried NO authored
      // offset — while a policy trained on these labels outputs the effective
      // action, offset included, and feeds THAT back. The clone therefore met an
      // observation block it had never seen on its very first step, and the error
      // grew every tick. The file was never wrong: checked offline, it answered
      // the training observations to 0.006 rad. The loop was.
      fedBack = effective;
    }

    // NOISE ON THE WAY OUT, NEVER ON THE LABEL. The label above is what the
    // teacher would do AT THIS STATE; the jitter then pushes the duck slightly
    // off the demonstrated path, so the NEXT step's label is a recovery. That is
    // the difference between a clone that memorises one trajectory and one with a
    // basin around it — measured: without it, a bow clone fell in all 32 of 32.
    if (jitter > 0) {
      for (let k = 0; k < 14; k++) {
        const c = duck.ctrl[k];
        d.ctrl[c] = Math.min(Math.max(d.ctrl[c] + (Math.random() * 2 - 1) * jitter,
                                      LO[k]), HI[k]);
      }
    }
    return fedBack;
  }

  /**
   * Time, moved once, for everything in the world at once.
   *
   * The plant's own substeps, so a control tick is a control tick whether one
   * duck is standing in this world or three are walking into each other.
   */
  function stepWorld(d) {
    for (let s = 0; s < SUBSTEPS; s++) mj.mj_step(model, d);
  }

  /**
   * The single-duck control tick: choose this duck's commands, then advance time.
   *
   * THE SAME TICK FOR EVERY CALLER that drives one duck — /record, /measure,
   * /perform and /capture all come through here, so a policy that measured 16/16
   * behaves identically wherever it is run. The live world does NOT use this one:
   * it actuates every duck first and calls `stepWorld` once, which is the same
   * sequence with the loop in the right place.
   */
  async function tick(d, duck, loaded, last, cmd, offsets = null, blend = 1, capture = null,
                      expert = null, teacherShare = 0, jitter = 0) {
    const fedBack = await actuate(d, duck, loaded, last, cmd, offsets, blend, capture,
                                  expert, teacherShare, jitter);
    stepWorld(d);
    return fedBack;
  }

  /**
   * Interpolate an authored keyframe track, the same smoothstep the phone draws
   * with and `record_intents.mjs` records with. TWENTY copies of this curve now
   * exist in this repo — `grep -rn "function poseAt"` over duck-sounds, counted
   * 2026-08-30: nineteen under sim/ and site/intent.mjs, which the browser
   * preview imports — and they all have to agree, or a motion previews as one
   * shape and runs as another. This comment said three, which is how a change
   * here comes to be costed as touching two other files when it touches
   * nineteen, one of them the recorder duckkit's shipped corpus came from and
   * one of them the preview a browser draws.
   */
  function poseAt(track, time) {
    if (!track || !track.length) return null;
    if (time <= 0) return HOME.slice();
    let pt = 0, pp = HOME;
    for (const f of track) {
      if (time <= f.at) {
        const u = (time - pt) / Math.max(f.at - pt, 1e-9), s = u * u * (3 - 2 * u);
        return f.pose.map((v, k) => pp[k] + (v - pp[k]) * s);
      }
      pt = f.at; pp = f.pose;
    }
    return track[track.length - 1].pose.slice();
  }

  /**
   * The canon loop, the same one every recorded clip came from: a settle under
   * the standing policy with a neutral command, then the training path — target
   * = HOME + action at scale 1.0, no filter — and the servo travel as the clamp.
   *
   * IT DRIVES ONE DUCK. In a multi-duck world the others are reset to HOME with
   * the rest of the scene and then left holding that pose on their position
   * servos for the length of the run — they are furniture, not participants, and
   * a rollout says which duck it drove rather than leaving a reader to assume.
   * Recording a swarm would mean a clip format that carries N sets of frames, and
   * duck-intent-clips/3 carries one; the live world is where several ducks are
   * actually steered at once.
   */
  async function rollout({ name, seconds, schedule, settle = SETTLE_TICKS, drop = 0.1231,
                           track = null, blend = 1, capture = null, expertName = null,
                           teacherShare = 0, jitter = 0, duck = DUCKS[0],
                           loaded = null, trace = null }) {
    // `loaded` IS FOR THE ONE CALLER THAT RUNS A NETWORK NO CATALOGUE HOLDS.
    // /tune folds a per-joint gain and trim into the last layer at request
    // time, so the network it scores exists for the length of one call and has
    // no name to look up. Every other caller passes a name and gets the cached
    // session, which is why this is an override and not a parameter.
    const net = loaded ?? await policy(name);
    const settling = await policy(STAND);
    // The teacher, when one is asked for: the policy the authored motion rides on.
    const expert = expertName ? await policy(expertName) : null;
    reset(data, drop);
    let last = new Array(14).fill(0);
    const frames = [], roots = [], commands = [];
    const ticks = Math.round(seconds * C.tickHz);
    const D = duck.joints;
    const f = D.freeQpos;
    for (let t = -settle; t < ticks; t++) {
      const cmd = command(t >= 0 ? commandAt(schedule, t / C.tickHz) : {});
      // NEUTRAL THROUGH THE SETTLE. The settle exists to let the drop-bounce die;
      // feeding the track through it starts the motion before recording does.
      const offsets = track && t >= 0 ? poseAt(track, t / C.tickHz) : null;
      // NOTHING IS CAPTURED DURING THE SETTLE. Those ticks are the drop bounce
      // dying under a different policy; teaching a clone from them would teach it
      // to recover from a fall it will never be dropped into.
      last = await tick(data, duck, t < 0 ? settling : net, last, cmd, offsets, blend,
                        t >= 0 ? capture : null, t >= 0 ? expert : null, teacherShare,
                        t >= 0 ? jitter : 0);
      if (t >= 0) {
        const after = [];
        for (let k = 0; k < 14; k++) {
          after.push(Math.min(Math.max(data.qpos[D.qpos[k]], LO[k]), HI[k]));
        }
        frames.push(after.map(r4));
        roots.push([data.qpos[f], data.qpos[f + 1], data.qpos[f + 2],
                    data.qpos[f + 3], data.qpos[f + 4], data.qpos[f + 5], data.qpos[f + 6]].map(r4));
        commands.push(cmd.slice(0, 3).map(r4));
        // THE UNROUNDED TRACK, FOR THE ONE CALLER THAT SCORES A REWARD FROM IT.
        // `frames`, `roots` and `commands` above are rounded to 1e-4 because
        // they are a CLIP — something a phone draws and a file stores — and
        // four decimals is what a clip has always carried. A reward term is
        // arithmetic, not a picture: `body_ang_vel` squares a rate and
        // `action_rate_l2` differences two consecutive actions, and quantising
        // either at 1e-4 first puts the quantum into the score. So /tune reads
        // this instead, and it holds what the clip cannot hold at all — the
        // trunk's twist and the network's own output.
        if (trace && !trace.diverged) {
          // A DIVERGED TICK ENDS THE TRACE AND IS NAMED, never summed. The
          // capture path above refuses such a tick by count; here the whole
          // episode is the unit — a term averaged over forty sane ticks and one
          // at 6.8e37 is not a term, and an answer with a null in `terms` used
          // to come back HTTP 200 and abort the caller's entire search.
          const root = [data.qpos[f], data.qpos[f + 1], data.qpos[f + 2],
                        data.qpos[f + 3], data.qpos[f + 4], data.qpos[f + 5], data.qpos[f + 6]];
          const joints = Array.from({ length: 14 }, (_, k) => data.qpos[D.qpos[k]]);
          const qv = [data.qvel[D.freeDof], data.qvel[D.freeDof + 1], data.qvel[D.freeDof + 2],
                      data.qvel[D.freeDof + 3], data.qvel[D.freeDof + 4], data.qvel[D.freeDof + 5]];
          if (!(root.every(sane) && joints.every(sane) && qv.every(sane)
                && Array.from(last).every(sane))) {
            trace.diverged = { tick: t, why: 'a non-finite or absurd state or action' };
          }
        }
        if (trace && !trace.diverged) {
          trace.push({
            root: [data.qpos[f], data.qpos[f + 1], data.qpos[f + 2],
                   data.qpos[f + 3], data.qpos[f + 4], data.qpos[f + 5], data.qpos[f + 6]],
            // MuJoCo's own convention for a free joint, said out loud because
            // getting it wrong is silent: the LINEAR half is in the world frame
            // and the ANGULAR half is in the body's. Neither is used raw below.
            qvel: [data.qvel[D.freeDof], data.qvel[D.freeDof + 1], data.qvel[D.freeDof + 2],
                   data.qvel[D.freeDof + 3], data.qvel[D.freeDof + 4], data.qvel[D.freeDof + 5]],
            joints: Array.from({ length: 14 }, (_, k) => data.qpos[D.qpos[k]]),
            // The network's own fourteen outputs at this tick — `tick` hands
            // them back as what the next observation will be told, and for a
            // plain rollout that is exactly the action.
            action: Array.from(last),
            command: [cmd[0], cmd[1], cmd[2]],
          });
        }
      }
    }
    return { frames, roots, commands };
  }

  /** A schedule is a list of [atSeconds, {vx, vy, vyaw}] — the last one that has begun wins. */
  function commandAt(schedule, secs) {
    let current = {};
    for (const [at, values] of schedule || []) if (secs >= at) current = values;
    return current;
  }

  const r4 = v => Math.round(v * 10000) / 10000;
  const upright = root => {
    const [, , , w, x, y] = root;
    return -(1 - 2 * (x * x + y * y)) < -0.5;
  };


  /**
   * ONE PHYSICS CALLER AT A TIME, PER WORLD.
   *
   * Node is single-threaded, but every tick awaits onnxruntime, so two
   * overlapping requests would interleave their steps into the same mjData and
   * produce a duck driven by two policies at once — a heisenbug that only shows
   * up under the load a steering loop actually generates. Each world gets its own
   * lane, and the lanes are separate so that a two-minute /measure cannot stall a
   * 100 ms /intent behind it.
   */
  function lane() {
    let tail = Promise.resolve();
    return job => {
      const run = tail.then(job, job);
      tail = run.then(() => {}, () => {});   // a rejection must not poison the lane
      return run;
    };
  }
  const liveLane = lane(), batchLane = lane(), climbLane = lane();

  /**
   * A /climb cell TAKES BOTH OTHER LANES, and this is the one place in the file
   * where that is true.
   *
   * Every other endpoint borrows an mjData. A climb cell borrows the MODEL: the
   * plant axis of the grid multiplies the foot geoms' friction, and laying a
   * flight out at all means zeroing the step blocks' conaffinity so fourteen
   * overlapping 200 kg boxes do not shove each other apart. Both are written
   * back in a `finally` inside the shared episode, so nothing survives the
   * request — but WHILE it runs, a /intent stepping the live world or a
   * /measure stepping the batch world would be stepping a duck with different
   * feet. Serialising against both is a second of latency for a steering loop
   * that happens to overlap a scoring run, and the alternative is a number
   * nobody can explain.
   */
  const climbJob = job => climbLane(() => liveLane(() => batchLane(job)));

  /**
   * THE CLIMB RIG, BUILT ON FIRST ASK AND NOT AT BOOT.
   *
   * It costs an mjData and a policy load, and a bench that is only ever asked
   * to /record should not pay for a staircase it never lays out. `CLIMB` is
   * `undefined` until something asks, then either the rig or `null` — and null
   * is an answer, not a failure: it means this plant cannot be asked, and
   * /climb says which of the three things is missing rather than throwing.
   *
   * ITS OWN mjData. The live world is the one a steering loop is keeping and
   * the batch world is the one /record resets; a climb episode resets its world
   * fourteen times per grid and must not be either of them.
   *
   * WHY THE ACTUATOR CHECK. The shared episode writes `data.ctrl[k]` for
   * k = 0..13, which is what rig3.mjs and robust.mjs have always written and
   * therefore what every audited number was measured through. In the canon
   * one-duck scene actuator k IS joint k; in a multi-duck scene it is not, and
   * a climb scored there would be driving the first duck's legs. So it is
   * checked, and a scene where it does not hold is refused by name.
   */
  let CLIMB, CLIMB_WHY = null;
  async function climbRig() {
    if (CLIMB !== undefined) return CLIMB;
    CLIMB = null;
    const duck = DUCKS.find(d => d.prefix === '');
    if (!duck) { CLIMB_WHY = `this scene's ducks are all prefixed (${DUCK_NAMES.join(', ')}); `
      + 'the climb episode drives the unprefixed duck the audits were measured on'; return CLIMB; }
    if (!duck.ctrl.every((a, k) => a === k)) {
      CLIMB_WHY = 'this scene\'s actuators are not in joint order, so the shared climb episode '
        + 'would drive the wrong joints'; return CLIMB;
    }
    const loaded = await policy(STAND);
    const rig = makeClimbRig({
      mj, model, data: new mj.MjData(model),
      D: duck.joints, HOME, LO, HI, buildObs, projectedGravity, command,
      tickHz: C.tickHz, reference: loaded.reference,
      // THE FORWARD PASS IS THE SHELL'S, exactly as it is for every other
      // endpoint: onnxruntime on the desk, policyforward.mjs in a browser. They
      // agree to 3.5e-6 per action and not exactly (sim/policy_parity.mjs), so
      // a phone's cell is the phone's own measurement — which is why the answer
      // carries the plant digest and /health carries the engine.
      run: obs => loaded.net.run(obs),
    });
    if (!rig) { CLIMB_WHY = `${PLANT} has no stair bank: its bodies step0..step13 are not in it`; return CLIMB; }
    CLIMB = rig;
    return CLIMB;
  }

  /**
   * THE LIVE WORLD: the ducks that keep standing between requests.
   *
   * Its own mjData, not the one /record and /measure reset on every call — that
   * is the whole difference between a bench and a transport. A steering loop
   * sends an intent, reads the state it caused, and sends the next one; sharing
   * a world with a rollout that resets to the drop height would teleport the duck
   * back to the origin in the middle of a walk.
   *
   * ONE WORLD, ONE CLOCK, A SLOT PER DUCK. `world` is singular on purpose: the
   * ducks are in the same mjData, so they collide, share a floor and disturb the
   * same props. What is per-duck is the STEERING — the policy it is running, the
   * command it is holding, and the last action its next observation will be told
   * about — because those are exactly the things two ducks in one world need to
   * differ in for the world to be worth having.
   */
  const live = {
    world: new mj.MjData(model),
    standing: false,
    slots: new Map(DUCKS.map(duck => [duck.name, {
      duck,
      policy: STAND,
      last: new Array(14).fill(0),
      cmd: { vx: 0, vy: 0, vyaw: 0 },
    }])),
  };

  /**
   * Put every duck on its feet, once, the first time anyone asks anything of it.
   *
   * ALL OF THEM, EVEN WHEN ONE WAS ASKED FOR. A duck left lying where the model
   * dropped it is still in the scene the addressed duck has to walk through, so
   * there is no such thing as settling one of them.
   *
   * Lazy rather than at boot because a bench that is only ever asked to /record
   * should not pay half a second of settle it will throw away.
   */
  async function ensureStanding() {
    if (live.standing) return;
    const settling = await policy(STAND);
    reset(live.world);
    const neutral = command({});
    for (const slot of live.slots.values()) slot.last = new Array(14).fill(0);
    for (let t = 0; t < SETTLE_TICKS; t++) {
      for (const slot of live.slots.values()) {
        slot.last = await actuate(live.world, slot.duck, settling, slot.last, neutral);
      }
      stepWorld(live.world);
    }
    live.standing = true;
  }

  /**
   * Advance the live world under the commands its ducks are currently holding.
   *
   * EVERY DUCK IS DRIVEN, WHOEVER ASKED. This is the sentence a caller most needs
   * to have read: /intent addressed to one duck advances the whole world, and the
   * other ducks keep walking under whatever they were last told, because there is
   * one integrator and it cannot advance half a scene. On hardware the equivalent
   * would be four robots each running their own loop and nothing keeping their
   * timelines together; here they are together by construction, and a caller who
   * wants a duck to stand still says so with /stop rather than by not mentioning
   * it.
   *
   * Each policy is loaded once for the whole span even when several ducks share
   * one — `policy()` caches by file, and this keeps the lookup out of the tick.
   */
  async function hold(seconds) {
    const driving = [];
    for (const slot of live.slots.values()) {
      driving.push({ slot, loaded: await policy(slot.policy), cmd: command(slot.cmd) });
    }
    const ticks = Math.max(1, Math.round(seconds * C.tickHz));
    for (let i = 0; i < ticks; i++) {
      for (const d of driving) {
        d.slot.last = await actuate(live.world, d.slot.duck, d.loaded, d.slot.last, d.cmd);
      }
      stepWorld(live.world);
    }
    return ticks;
  }

  /**
   * A one-line reading of every duck in the world, for the answers that address
   * only one of them.
   *
   * WHY IT RIDES ALONG ON EVERY ANSWER. In a shared world the other ducks are
   * part of what happened to this one — they are what it bumped into — so a
   * caller that reads a position without them is reading half a result. On the
   * canon one-duck scene it is a single entry and costs nothing.
   */
  function rollCall() {
    const d = live.world;
    return DUCKS.map(duck => {
      const slot = live.slots.get(duck.name), f = duck.joints.freeQpos;
      const root = [d.qpos[f], d.qpos[f + 1], d.qpos[f + 2],
                    d.qpos[f + 3], d.qpos[f + 4], d.qpos[f + 5], d.qpos[f + 6]];
      return {
        name: duck.name,
        position: root.slice(0, 3).map(r4),
        upright: upright(root),
        policy: slot.policy,
        command: slot.cmd,
      };
    });
  }

  /** What one duck is doing right now, in the live world. */
  function stateOf(slot) {
    const d = live.world, D = slot.duck.joints, f = D.freeQpos;
    const root = [d.qpos[f], d.qpos[f + 1], d.qpos[f + 2],
                  d.qpos[f + 3], d.qpos[f + 4], d.qpos[f + 5], d.qpos[f + 6]];
    const joints = [];
    for (let k = 0; k < 14; k++) joints.push(r4(d.qpos[D.qpos[k]]));
    return {
      // WHICH DUCK THIS IS ABOUT, ALWAYS SAID. A caller that never passes `duck`
      // gets the same name every time and can ignore it; a caller steering three
      // of them can check that the answer is about the one it addressed rather
      // than trusting that it asked correctly.
      duck: slot.duck.name,
      t: r4(d.time),
      position: root.slice(0, 3).map(r4),
      quaternion: root.slice(3).map(r4),
      height: r4(root[2]),
      upright: upright(root),
      joints,
      policy: slot.policy,
      command: slot.cmd,
      ducks: rollCall(),
      // NOT A NUMBER, ON PURPOSE. quackd's transport reports a battery because a
      // real duck has one; this world has no battery to read, and inventing a
      // percentage would be a number nobody measured that a verb might one day
      // decide to land on. null is the honest reading, and the field is here so
      // the shape matches.
      // GROUND TRUTH, AND IT IS NOT FOR STEERING. A vision loop must earn its
      // bearing from the camera; this is here so a run can be SCORED — the same
      // split measure_success.mjs already uses.
      ball: ballOf(d),
      ballRadius: BALL_RADIUS,
      battery: null,
      batteryWhy: 'simulated duck: there is nothing to discharge',
    };
  }

  /**
   * A command component. Finite or the request is refused.
   *
   * DELIBERATELY NOT CLAMPED. Training's own range is the only real limit and
   * this repo's schedules span it unevenly — record_intents drives vx and vy
   * around the full unit circle, while ac_headspin2 commands vyaw 3 to get a
   * spin rate. A tidy-looking clamp at ±1 would be a number nobody measured and
   * would quietly break the one policy that needs a big one.
   */
  function speed(v, field) {
    if (v === undefined || v === null) return 0;
    const n = Number(v);
    if (!Number.isFinite(n)) throw new Error(`${field} must be a finite number`);
    return n;
  }

  /**
   * WHICH DUCK A REQUEST IS ABOUT.
   *
   * `duck` in the body, or `?duck=` on the query string for the endpoints that
   * are GETs. ABSENT MEANS THE FIRST DUCK, which on the canon scene is the only
   * duck and is exactly what every existing caller has always been asking for;
   * in a scene with several, the first is the one the MJCF lists first and the
   * answer names it, so nobody has to guess which one moved.
   *
   * A name that is not in this world is refused with the names that are, rather
   * than silently steering the default — a typo that quietly drives the wrong
   * duck is the multi-duck version of the ball-instead-of-the-duck bug that
   * `findDuckJoints` was written to end.
   */
  function pickSlot(url, body) {
    const wanted = body?.duck ?? url.searchParams.get('duck');
    if (wanted === undefined || wanted === null || wanted === '') {
      return live.slots.get(DUCKS[0].name);
    }
    const slot = live.slots.get(String(wanted));
    if (!slot) throw new Error(`unknown duck: ${wanted} — this world holds ${DUCK_NAMES.join(', ')}`);
    return slot;
  }

  /** The same choice, for the batch endpoints, which address a duck but hold no live slot. */
  function pickDuck(url, body) {
    return pickSlot(url, body).duck;
  }

  async function handle(url, body) {
    if (url.pathname === '/health') {
      return {
    // BUMPED WITH THE PLANT FIELDS. /health, /perform and /record now carry
    // plantName and plantDigest, and a reader that sees duck-bench/2 knows it is
    // talking to a bench that CANNOT say which world it ran, as distinct from one
    // that did not. Additive fields alone leave those two indistinguishable.
    // BUMPED AGAIN FOR THE DUCKS. duck-bench/4 holds N of them in one world and
    // takes a `duck` name on the steering endpoints. The field is what lets a
    // reader tell "this bench has one duck" from "this bench cannot tell you how
    // many it has": a duck-bench/3 answer has no `ducks` key either way, and a
    // caller that guessed would be guessing about the world.
        // BUMPED AGAIN FOR THE HOST. duck-bench/5 says WHERE the physics ran:
        // a bench in a browser on an iPhone answers the same endpoints as the
        // one on the desk, and a measurement that does not say which machine
        // produced it is a number a reader has to guess about.
        bench: 'duck-bench/5',
        // WHOSE PHYSICS THIS IS. `kind` is the only field a caller should branch
        // on; `device` and `engine` are for a human reading a saved result, and
        // `tickMillis` is what one control tick actually cost here, measured at
        // boot rather than claimed — it is the number that decides whether a
        // phone can run this at all.
        host: HOST,
        plant: `${SCENE} — Pollen robot_allcollisions, training parameters`,
        plantName: PLANT,
        plantDigest: PLANT_DIGEST,
        // THE DUCKS PRESENT, walked out of the model at boot rather than
        // configured, so a scene answers with what is actually in it. `prefix` is
        // the MJCF name prefix each one's joints and actuators carry and is here
        // because it is what a caller would need to make sense of the scene XML
        // beside this answer; `spawn` is where /reset puts that duck back.
        ducks: DUCKS.map(d => ({
          name: d.name,
          prefix: d.prefix,
          spawn: d.spawn,
          policy: live.slots.get(d.name).policy,
        })),
        ducksWhy: 'One model, one integrator: a single step advances every duck, so they meet '
                + 'through contact and a shared floor rather than over a link. A hardware duck '
                + 'cannot perceive another duck at all — the observation is 61 values with no '
                + 'slot for one — so this is the only place several of them share a world.',
        // What is in this world that the duck could take hold of. NOT EMPTY ON
        // THE CANON SCENE ANY MORE: scene.mjb as served here on 2026-08-30
        // answers with five — block_a, block_b, block_c, cone_a, cone_b — where
        // this comment said it was bare. The list itself is walked out of the
        // model on every boot, so what is SERVED was right the whole time; it is
        // the comment that lied, and a caller who read it instead of the answer
        // is why it is corrected rather than deleted.
        // Name and mass only: the qpos addresses are this file's business.
        graspables: GRASPABLES.map(g => ({ name: g.name, mass: g.mass })),
        tickHz: C.tickHz,
        timestep: TIMESTEP,
        substepsPerTick: SUBSTEPS,
        cores: env.cores,
        policies: [...catalogue().keys()].sort(),
        // WHICH FILE IS DOING THE STANDING, RESOLVED BY ROLE. Every endpoint
        // that settles a duck runs this one, and it used to be pinned to the
        // literal `BEST_alpha_stand.onnx` — a training-run filename. The app
        // bundles the same network under its ROLE name, `alpha_stand.onnx`, so
        // a bench built from the app's assets had no file by the pinned name
        // and /reset, /intent, /stop, /record, /measure and /perform all died
        // on `unknown policy`. The role name is tried first and the training
        // name second, and the answer says which one was found rather than
        // leaving a caller to infer it from a failure.
        stand: STAND,
        standWhy: `the settling policy, resolved by role: ${STAND_TRIED.join(' then ')}`,
        records: true, measures: true,
        trains: false,
        // A SENTENCE ABOUT THE MACHINE THIS IS, NOT ABOUT A DESK IT INHERITED.
        // The desk text names the Hailo in the Pi and mjlab's GPU; served from
        // a phone it would be a claim about hardware the answer is not running
        // on, which is exactly the kind of inherited prose that turns into a
        // stated fact three files later.
        trainsWhy: HOST.kind === 'phone'
          ? 'A phone runs the policy and the physics and trains neither: there is no PPO on iOS, '
          + 'and the numbers here are measurements, never a training signal.'
          : 'The accelerator here is an inference ASIC, and mjlab wants a GPU. '
          + 'Recording and measuring are what this machine is for.',
        // The transport surface, named so a caller can tell at a glance which
        // half of quackd's DuckTransport this speaks and which half it does not.
        steers: true,
        transport: {
          // HOW A CALLER ACTUALLY REACHES THIS BENCH. On the desk it is an
          // HTTP server somebody dials over a tailnet; in a page there is no
          // socket, no port and nothing to dial — the app calls a function
          // inside its own WebView. Same paths, same bodies, same answers.
          protocol: HOST.kind === 'phone'
            ? 'in-page: globalThis.duckbench(pathAndQuery, bodyJSON) -> Promise<JSON string>'
            : 'quackd DuckTransport (rokbenko/quackd)',
          clock: 'sim',
          clockWhy: 'now() reads this world\'s own MuJoCo clock, so a verb steers at '
                  + 'sim speed here and at wall-clock speed on hardware, unchanged.',
          endpoints: ['GET /state', 'POST /intent', 'POST /stop', 'GET /now',
                    'POST /policy', 'POST /ball', 'POST /reset'],
          // Every steering endpoint except /now, which reads the world's clock
          // and so belongs to all of them at once.
          addressable: '`duck` in the body or ?duck= on a GET; absent means ' + DUCKS[0].name,
          frames: false,
          framesWhy: HOST.kind === 'phone'
            ? 'No camera and no renderer in this shell: it answers JSON and the app draws the duck.'
            : 'No camera and no renderer in this process: perception is duckvision.py, '
            + 'on a different MuJoCo build so that clips stay canon.',
          // The first duck's, kept under the name a duck-bench/3 reader already
          // knows; `ducks` above carries one of these per duck.
          activePolicy: live.slots.get(DUCKS[0].name).policy,
          standing: live.standing,
        },
      };
    }
    // GET /state — the duck as it is, in the world /intent has been advancing.
    if (url.pathname === '/state') {
      return liveLane(async () => {
        const slot = pickSlot(url, body);
        await ensureStanding();
        return stateOf(slot);
      });
    }
    // GET /now — the transport owns time, and this is the clock it owns. A
    // steering loop asks for it instead of Date.now() so the same verb code runs
    // here and on hardware; here it only moves when someone advances physics,
    // which is precisely what makes a sim loop reproducible.
    // POST /ball — put it somewhere, or {"bearing": deg, "range": m} to place it
    // relative to where the duck is looking, which is what a trial wants.
    if (url.pathname === '/ball') {
      return liveLane(async () => {
        // BEARING AND RANGE ARE RELATIVE TO A DUCK, so which duck is part of the
        // request even though the ball belongs to nobody. Absent, it is the first
        // one, the same as everywhere else.
        const slot = pickSlot(url, body);
        await ensureStanding();
        const d = live.world, f = slot.duck.joints.freeQpos;
        if (body.bearing !== undefined || body.range !== undefined) {
          const bearing = Number(body.bearing ?? 0), range = Number(body.range ?? 0.8);
          if (!Number.isFinite(bearing) || !Number.isFinite(range)) {
            throw new Error('bearing and range must be finite numbers');
          }
          const q = [d.qpos[f + 3], d.qpos[f + 4], d.qpos[f + 5], d.qpos[f + 6]];
          const yaw = Math.atan2(2 * (q[0] * q[3] + q[1] * q[2]),
                                 1 - 2 * (q[2] * q[2] + q[3] * q[3]));
          // Positive bearing is LEFT, the convention duckvision and the robot
          // both use, so a trial reads the same way the detector reports.
          const a = yaw + bearing * Math.PI / 180;
          placeBall(d, d.qpos[f] + range * Math.cos(a), d.qpos[f + 1] + range * Math.sin(a));
        } else {
          const x = Number(body.x), y = Number(body.y);
          if (!Number.isFinite(x) || !Number.isFinite(y)) {
            throw new Error('give {x, y} or {bearing, range}');
          }
          placeBall(d, x, y, Number.isFinite(+body.z) ? +body.z : BALL_RADIUS);
        }
        return stateOf(slot);
      });
    }
    // POST /reset — put the live world back to a known start.
    //
    // A TRIAL THAT BEGINS WHEREVER THE LAST ONE STOPPED IS NOT A TRIAL. Without
    // this, a steering run inherited the previous run's position, heading and
    // half-finished stride, and the second trial of a batch could open already
    // spinning — which looks exactly like a controller that cannot see.
    //
    // WITH A `duck` NAME IT RESETS THAT DUCK ONLY, and that is a different act,
    // not a smaller one. `mj_resetData` restarts everything in the world, which
    // in a shared scene means teleporting the other ducks and every prop; a
    // named reset writes one duck's qpos and qvel back to its spawn and leaves
    // the clock, the props and the other ducks exactly where they were. Use it
    // to give one duck another go at something the rest of the world is in the
    // middle of; use the plain form to start a trial.
    if (url.pathname === '/reset') {
      return liveLane(async () => {
        // Asked-for-ness, not truthiness: a duck could legitimately be named
        // `0`, and testing the name for truth would quietly reset the whole
        // world instead of that duck.
        const asked = body?.duck ?? url.searchParams.get('duck');
        const named = (asked === undefined || asked === null || asked === '')
          ? null : pickSlot(url, body);
        if (named) {
          await ensureStanding();
          placeDuck(live.world, named.duck);
          mj.mj_forward(model, live.world);
          named.cmd = { vx: 0, vy: 0, vyaw: 0 };
          named.last = new Array(14).fill(0);
          return { ...stateOf(named), reset: named.duck.name };
        }
        live.standing = false;
        for (const slot of live.slots.values()) slot.cmd = { vx: 0, vy: 0, vyaw: 0 };
        await ensureStanding();
        if (BALL) placeBall(live.world, 0.8, 0, BALL_RADIUS);
        return { ...stateOf(live.slots.get(DUCKS[0].name)), reset: 'the whole world' };
      });
    }
    if (url.pathname === '/now') {
      return liveLane(async () => {
        await ensureStanding();
        return { now: r4(live.world.time), clock: 'sim', tickHz: C.tickHz, timestep: TIMESTEP };
      });
    }
    /*
     * POST /intent — hold {vx, vy, vyaw} for a short window and advance physics.
     *
     * THE DEADMAN COSTS NOTHING HERE. quackd's verbs re-send a move every 100 ms
     * so a dropped link stops a real robot instead of leaving it walking into a
     * wall. This world only moves inside a request: miss the next intent and the
     * duck is not still walking, it is frozen mid-stride, so there is no timer to
     * arm and nothing to fail closed. That makes the repeat call the common case,
     * and it is cheap — no reset, no reload, one cached session and five ticks by
     * default, which is exactly the 100 ms the verbs re-send at.
     */
    if (url.pathname === '/intent') {
      const seconds = Math.min(Math.max(+body.hold || 0.1, 1 / C.tickHz), 2);
      return liveLane(async () => {
        const slot = pickSlot(url, body);
        await ensureStanding();
        slot.cmd = { vx: speed(body.vx, 'vx'), vy: speed(body.vy, 'vy'),
                     vyaw: speed(body.vyaw, 'vyaw') };
        // AND THE HOLD MOVES EVERY DUCK. Only this one's command changed; the
        // others keep the commands they were holding and keep walking under them
        // for the same window, because the world has one clock. Steering three
        // ducks therefore means three calls per window, and between them the
        // world does not wait.
        const ticks = await hold(seconds);
        return { ...stateOf(slot), held: r4(ticks / C.tickHz), ticks };
      });
    }
    // POST /stop — zero the command and let the duck settle under it. Not a
    // reset: stopping is a thing the policy does, and a duck that had to be
    // teleported upright to stop would be hiding the fall.
    // STOPPING ONE DUCK DOES NOT STOP THE WORLD, and that is the honest
    // behaviour rather than a limitation: the settle it asks for is time, and
    // time passing is something the other ducks are also in. `duck` with no name
    // stops the first one, which on the canon scene is the only one.
    if (url.pathname === '/stop') {
      const seconds = Math.min(Math.max(+body.settle || 0.5, 1 / C.tickHz), 5);
      return liveLane(async () => {
        const slot = pickSlot(url, body);
        await ensureStanding();
        slot.cmd = { vx: 0, vy: 0, vyaw: 0 };
        const ticks = await hold(seconds);
        return { ...stateOf(slot), settled: r4(ticks / C.tickHz), ticks };
      });
    }
    /*
     * POST /policy — swap the ACTIVE policy under the standing duck.
     *
     * SAME ALLOW-LIST, NO EXCEPTION. The name goes through `policy()`, which
     * looks it up in a Map built by scanning this directory: '../secrets.onnx',
     * '/etc/passwd' and 'https://example.com/p.onnx' are refused not because
     * anything inspects them for traversal but because they are not keys. A Map
     * is used rather than an object so that '__proto__' is a miss too. Nothing
     * is fetched, nothing is joined onto a path.
     *
     * The duck is NOT reset around the swap: hot means hot, and a policy that
     * cannot pick up another's pose mid-stance is telling you something true.
     */
    // A POLICY THAT WAS NOT ALREADY ON THIS DISK.
    //
    // /policy takes a NAME and loads from the bench's own directory, which is
    // right for the nine shipped networks and useless for a network somebody
    // just made. Duck Studio can now write an ONNX — it blends the ones it has —
    // and a blend that cannot be run is a file with nothing behind it.
    //
    // WRITTEN TO A SCRATCH FILE BECAUSE onnxruntime LOADS FROM A PATH. The name
    // is derived from the sha256 of the bytes, never from anything the caller
    // sent: a caller-chosen name is a path traversal waiting to happen, and the
    // digest is also exactly what identifies which network this is.
    //
    // This bench is OPEN on the network it is on, and this endpoint widens what
    // that means — it accepts bytes and runs them. It is a development tool on a
    // private network and was already running arbitrary policies from its own
    // directory, but that is a different sentence from "accepts arbitrary bytes",
    // and anybody putting this on a network they do not trust should know which
    // one they have.
    if (url.pathname === '/upload') {
      const b64 = typeof body.onnx === 'string' ? body.onnx : null;
      if (!b64) return { error: 'upload needs an `onnx` field: the file, base64' };
      let bytes;
      try { bytes = decodeBase64(b64); }
      catch { return { error: 'the `onnx` field is not base64' }; }
      if (!bytes.length) return { error: 'the uploaded policy is empty' };
      if (bytes.length > 8 * 1024 * 1024) {
        return { error: `the uploaded policy is ${bytes.length} bytes; the shipped ones are under 1 MB` };
      }
      const digest = await env.sha256(bytes);
      const name = `uploaded-${digest.slice(0, 12)}`;
      const file = `${name}.onnx`;
      if (!env.scratch.has(file)) env.scratch.set(file, bytes);
      // THE PARAMETERS BESIDE THE FILE, WHEN THE CALLER HAS THEM. A desk bench
      // runs an .onnx through onnxruntime and can score it, but /tune folds a
      // gain into the LAST LAYER, which means holding the parameters — and
      // nothing on a desk dumps them for an upload. The app has them (they
      // are what the fingerprint is taken over), so it sends both: the file
      // for the session and its declared neutral pose, the bytes for the fold.
      // On a phone the `onnx` field already IS the canonical bytes and this
      // field is simply absent.
      let parametersNote = '';
      if (typeof body.parameters === 'string') {
        let params;
        try { params = decodeBase64(body.parameters); }
        catch { return { error: 'the `parameters` field is not base64' }; }
        const seen = params.byteLength ?? params.length;
        if (seen !== FLOAT_COUNT * 4) {
          return { error: `the parameters are ${seen} bytes where this architecture's canonical `
                        + `parameters are ${FLOAT_COUNT * 4}` };
        }
        env.scratch.set(`${name}.params`, params);
        parametersNote = '; its canonical parameters are held too, so /tune can fold it';
      }
      try {
        // Load it now rather than at first use, so a file that onnxruntime
        // cannot open is refused HERE with the reason, not three calls later in
        // the middle of a rollout.
        const session = await env.makeSession(bytes, name);
        sessions.set(file, { name, net: session, reference: session.reference ?? HOME });
      } catch (e) {
        env.scratch.delete?.(file);
        return { error: `that file did not load as a policy: ${e.message}` };
      }
      return { policy: name, sha256: digest, bytes: bytes.length,
               note: 'loaded and ready — pass this name to /policy, /record, /measure or /perform'
                   + parametersNote };
    }

    if (url.pathname === '/policy') {
      const wanted = typeof body.policy === 'string' ? body.policy : String(body.policy ?? '');
      return liveLane(async () => {
        // ONE SLOT PER DUCK, which is what makes two networks in one world
        // possible: swap huey to a walker and leave dewey standing, and the
        // contact between them is between two different policies. That
        // experiment does not exist on hardware, where each duck is its own
        // process on its own robot and the only thing they share is a room.
        const slot = pickSlot(url, body);
        await ensureStanding();
        const loaded = await policy(wanted);   // throws `unknown policy: …` -> 400
        const was = slot.policy;
        slot.policy = wanted;
        return {
          ...stateOf(slot), was,
          // Whether this policy declared its own neutral pose or inherited HOME:
          // the one thing about a hot swap that is not visible in the duck's pose.
          reference: loaded.reference === HOME ? 'HOME' : 'declared by the policy',
        };
      });
    }
    if (url.pathname === '/record') {
      const seconds = Math.min(Math.max(+body.seconds || 3, 0.2), 30);
      const duck = pickDuck(url, body);
      const run = await batchLane(() =>
        rollout({ name: body.policy, seconds, schedule: body.schedule, duck }));
      const last = run.roots[run.roots.length - 1];
      return {
        format: 'duck-intent-clips/3',
        hz: C.tickHz,
        joints: C.jointNames.filter(n => n !== 'mouth'),
        policy: body.policy,
        // WHICH DUCK THE FRAMES ARE OF. One set of frames, one duck, said out
        // loud — a clip from a multi-duck world that did not name its duck would
        // be a recording of an unidentified robot.
        duck: duck.name,
        // The world this recording came out of, so a clip kept anywhere else
        // can still say where it was made. Same two keys /health reports.
        plantName: PLANT,
        plantDigest: PLANT_DIGEST,
        frames: run.frames, roots: run.roots, commands: run.commands,
        endsUpright: upright(last), endHeight: last[2],
      };
    }
    /*
     * POST /perform — run an AUTHORED motion in real physics.
     *
     * THE HOLE THIS FILLS. Every other endpoint runs a trained policy. An
     * authored motion — keyframes somebody wrote in Duck Studio — could be
     * previewed on a phone and published to the world without any physics engine
     * ever having seen it, because a phone has none. A preview is what you ASKED
     * for; this is what happens.
     *
     * IT RUNS THE MOTION MORE THAN ONCE, and that is the point. A single rollout
     * that stays up proves very little: the four authored stair motions in the
     * corpus get up their flight 0 times in 16. The answer to "does it work" is a
     * count, not a yes.
     *
     * The track rides on the standing policy as offsets from HOME, which is what
     * `record_intents.mjs` does and what the app's own preview draws.
     */
    if (url.pathname === '/perform') {
      const track = Array.isArray(body.track) ? body.track : null;
      if (!track || !track.length) return { error: 'perform needs a track of keyframes' };
      for (const key of track) {
        if (!Array.isArray(key.pose) || key.pose.length !== 14) {
          return { error: 'every keyframe needs a pose of 14 joint angles, mouth excluded' };
        }
        if (!key.pose.every(v => Number.isFinite(v))) {
          return { error: 'a keyframe pose holds something that is not a number' };
        }
        if (!Number.isFinite(+key.at)) return { error: 'every keyframe needs an `at` in seconds' };
      }
      const ordered = track.map(k => ({ at: +k.at, pose: k.pose.map(Number) }))
                           .sort((a, b) => a.at - b.at);
      const blend = Math.min(Math.max(+body.blend || 1, 0), 1);
      // NO TEACHER AND NO JITTER HERE, AND THE TWO LINES THAT SAID OTHERWISE
      // KILLED THIS ENDPOINT. `teacherShare` and `jitter` belong to /capture,
      // which has a `driver` and therefore something for a teacher to correct;
      // they were copied into /perform along with the block below them, where
      // `expertName` is not a binding at all. Reading a free variable is a
      // ReferenceError, so EVERY /perform request answered
      // `{"error":"expertName is not defined"}` — the endpoint whose whole job
      // is telling Duck Studio whether an authored motion survives real
      // physics, returning a 400 to every caller. Neither value was ever passed
      // to `rollout` from here, so removing them is the fix.
      const seconds = Math.min(Math.max(+body.seconds || (ordered[ordered.length - 1].at + 0.5),
                                        0.2), 30);
      const rollouts = Math.min(Math.max(+body.rollouts || 8, 1), 32);
      const name = body.policy || STAND;
      const duck = pickDuck(url, body);

      let first = null, ok = 0;
      const heights = [];
      for (let i = 0; i < rollouts; i++) {
        // Pollen's own randomisation, as measure_success.mjs uses it: the drop
        // height is what a bench can vary without touching the model.
        const drop = 0.12 + (0.01 * i) / Math.max(rollouts - 1, 1);
        const run = await batchLane(() =>
          rollout({ name, seconds, schedule: body.schedule, track: ordered, blend, drop, duck }));
        const last = run.roots[run.roots.length - 1];
        if (upright(last)) ok++;
        heights.push(last[2]);
        if (!first) first = run;
      }
      heights.sort((a, b) => a - b);
      const last = first.roots[first.roots.length - 1];

      // Peak joint rate over the first rollout: the fastest any joint was
      // actually moved, which is the number an authored track most often
      // overruns without noticing.
      let peak = 0;
      for (let t = 1; t < first.frames.length; t++) {
        for (let k = 0; k < 14; k++) {
          const rate = Math.abs(first.frames[t][k] - first.frames[t - 1][k]) * C.tickHz;
          if (rate > peak) peak = rate;
        }
      }

      return {
        format: 'duck-intent-clips/3',
        hz: C.tickHz,
        joints: C.jointNames.filter(n => n !== 'mouth'),
        policy: name,
        duck: duck.name,
        authored: true,
        // THE ANSWER A CALLER FILES AWAY. /perform is the one endpoint whose
        // result is stored and shown months later, so it is the one that most
        // has to name its own world rather than let the caller guess.
        plantName: PLANT,
        plantDigest: PLANT_DIGEST,
        blend,
        frames: first.frames, roots: first.roots, commands: first.commands,
        rollouts, achieves: ok,
        criterion: 'stayed upright to the end, over drop heights 0.120-0.130 m',
        medianHeight: r4(heights[Math.floor(heights.length / 2)]),
        endsUpright: upright(last), endHeight: r4(last[2]),
        peakJointRate: r4(peak),
      };
    }
    /*
     * POST /capture — the pairs a POLICY would have to reproduce to perform an
     * authored motion by itself.
     *
     * THE LAST MILE STARTS HERE, AND IT IS NOT A FILE FORMAT PROBLEM. A motion is
     * a function of time; a policy is a function of state, closed-loop at 50 Hz.
     * robotd runs the second kind and nothing else, so an authored motion reaches
     * a real duck only by being cloned into one. This endpoint produces the
     * training set for that: for every control step, the observation the network
     * was shown and the action it would have had to output.
     *
     * IT RUNS THE MOTION SEVERAL TIMES ON PURPOSE. One rollout teaches a clone a
     * single trajectory from a single drop height; the randomised drops are what
     * give it anything to generalise from. It is still a narrow dataset and the
     * clone is still a hypothesis — /measure is what settles whether it worked.
     */
    if (url.pathname === '/capture') {
      const track = Array.isArray(body.track) ? body.track : null;
      if (!track || !track.length) return { error: 'capture needs a track of keyframes' };
      for (const key of track) {
        if (!Array.isArray(key.pose) || key.pose.length !== 14) {
          return { error: 'every keyframe needs a pose of 14 joint angles, mouth excluded' };
        }
        if (!key.pose.every(v => Number.isFinite(v))) {
          return { error: 'a keyframe pose holds something that is not a number' };
        }
        if (!Number.isFinite(+key.at)) return { error: 'every keyframe needs an `at` in seconds' };
      }
      const ordered = track.map(k => ({ at: +k.at, pose: k.pose.map(Number) }))
                           .sort((a, b) => a.at - b.at);
      const rollouts = Math.min(Math.max(+body.rollouts || 8, 1), 32);
      const seconds = Math.min(Math.max(+body.seconds || (ordered[ordered.length - 1].at + 0.5),
                                        0.2), 30);
      // WHO DRIVES, AND WHO TEACHES. Without `driver` the authored motion runs on
      // the standing policy and labels itself — the first dataset. With one, the
      // named policy drives the duck into states of its OWN, and every label is
      // what the authored motion would have commanded from there. That second
      // dataset is the one a clone that fell over needs.
      const name = body.driver || body.policy || STAND;
      const expertName = body.driver ? (body.policy || STAND) : null;
      const blend = Math.min(Math.max(+body.blend || 1, 0), 1);
      // How often the teacher acts rather than the driver. Only meaningful with a
      // driver; 0 means the driver is on its own.
      const teacherShare = expertName
        ? Math.min(Math.max(body.teacherShare ?? 0.9, 0), 1) : 0;
      // Radians of noise added to the commanded joints AFTER labelling, so the
      // duck wanders off the demonstration and the labels teach the way back.
      const jitter = Math.min(Math.max(+body.jitter || 0, 0), 0.2);
      const duck = pickDuck(url, body);

      const pairs = [];
      // NOT `upright`: that is the module-level predicate, and a counter of the
      // same name shadows it inside this block — "upright is not a function".
      let stoodUp = 0;
      for (let i = 0; i < rollouts; i++) {
        const drop = 0.12 + (0.01 * i) / Math.max(rollouts - 1, 1);
        const run = await batchLane(() => rollout({
          name, seconds, schedule: body.schedule, track: ordered, blend, drop,
          capture: pairs, expertName, teacherShare, jitter, duck }));
        if (endedStanding(run)) stoodUp++;
      }
      function endedStanding(run) {
        const last = run.roots[run.roots.length - 1];
        return upright(last) && last[2] >= 0.100;
      }
      return {
        format: 'duck-clone-pairs/1',
        hz: C.tickHz,
        obsWidth: 61,
        actionWidth: 14,
        rollouts, seconds,
        duck: duck.name,
        ridingOn: expertName ?? name,
        drivenBy: name,
        corrective: expertName != null,
        teacherShare,
        jitter,
        // THE SOURCE MOTION'S OWN RECORD. A clone measured later has to be
        // comparable to the thing it was cloned from, and both are only
        // comparable within one world.
        plantName: PLANT,
        plantDigest: PLANT_DIGEST,
        uprightRollouts: stoodUp,
        criterion: 'the authored run itself ended standing, trunk at least 100 mm up',
        pairs: pairs.length,
        // Steps the physics diverged on, dropped rather than exported as nulls.
        rejected: pairs.rejected || 0,
        obs: pairs.map(p => p.obs.map(r4)),
        actions: pairs.map(p => p.action.map(r4)),
      };
    }

    /*
     * POST /tune — run ONE candidate residual and WEIGH IT, where the trace is.
     *
     * THE HOLE THIS FILLS, AND IT IS NOT A CONVENIENCE. Four of the six
     * evaluable terms of `microduck_velocity_env_cfg` need something no other
     * answer this bench gives carries. `track_linear_velocity`,
     * `track_angular_velocity` and `body_ang_vel` all read the trunk's TWIST;
     * `action_rate_l2` reads the NETWORK'S OWN OUTPUT. /state and /intent
     * answer with a position, a quaternion and fourteen joint angles, /record
     * adds the commands, and none of them carry a velocity or an action. A
     * client scoring a search from those would be scoring `upright` and `pose`
     * alone — and both of those are maximised by a duck that has stopped.
     * Measured on this bench at six seconds and vx 0.5: the standing policy
     * scores 2.9812 on those two where the walking policy scores 2.5287, having
     * travelled 1 mm against 1231. A search scored that way climbs by stopping.
     * The bench has the trace in front of it. Nothing else does.
     *
     * IT TAKES A RESIDUAL AND NOT A FILE. Twenty-eight numbers, against 791,584
     * bytes of base64 per candidate for a folded .onnx, and the bench already
     * holds the base network. It also keeps the fold in ONE implementation —
     * `foldParameters` is a transcription of duckkit's `DuckPolicyWriter.folding`
     * and is held to it by bytes, not by reading — so a gain that scores well
     * here behaves the same way once the app writes it into a file.
     *
     * WHICH FORWARD PASS RUNS, SAID OUT LOUD. /tune ALWAYS runs
     * `policyforward.mjs` over the canonical parameter bytes, on every shell,
     * even where the shell's own sessions are onnxruntime — because a fold is
     * arithmetic on parameters and onnxruntime has none to offer. The two agree
     * to 3.5e-6 per action (policy_parity.mjs), which is not nothing over a
     * closed loop, so a /tune number and a /record number about the same policy
     * are close relatives and not the same measurement. The alternative —
     * onnxruntime for the identity residual and this for everything else —
     * would measure a search's baseline on a different network from its
     * candidates, which is worse in the one place it would matter most.
     *
     * THE SETTLE IS THE SHELL'S, THOUGH, AND THAT IS SAID RATHER THAN HIDDEN.
     * The half second before the command starts runs the STANDING policy
     * through `rollout`, which loads it the way every other endpoint does — so
     * on the desk that settle is onnxruntime and the driven span is
     * policyforward. Within one bench that is consistent: /tune's settle is
     * bit-for-bit /record's and /perform's. Across two benches it is not, and
     * it is the reason a desk /tune and a phone /tune differ in the eighth
     * decimal of every term — measured here on 2026-09-02, `upright` 0.97838314
     * on the browser shell against 0.97838314 on the desk. A search is scored
     * against its own baseline on its own bench, so the difference does not
     * reach a verdict; a number copied from one bench to the other is a
     * different measurement.
     *
     * A TERM THIS BENCH CANNOT COMPUTE GOES IN `refused`, BY NAME. Two of the
     * six weights are negative, so a term silently left out of `terms` reads to
     * a client as the best possible value of the thing it punishes.
     */
    if (url.pathname === '/tune') {
      const name = body.policy;
      if (typeof name !== 'string' || !name) return { error: 'tune needs a policy by name' };
      const width = 14;
      const asVector = (value, field) => {
        if (!Array.isArray(value) || value.length !== width) {
          return `the ${field} is ${Array.isArray(value) ? value.length : 'not an array'} wide, `
               + `not ${width} — the mouth has no policy output, so a 15-joint array has to have `
               + 'index 9 dropped before it gets here';
        }
        // `typeof v === 'number'` AND NOT `Number.isFinite(+v)`. JSON has a
        // null, and `Number(null)` is 0 — so a trim array with a hole in it
        // would have been folded as a zero on that joint and scored as though
        // the caller had asked for one. A hole is a mistake, and a mistake in
        // twenty-eight numbers that become a network is refused by name.
        if (!value.every(v => typeof v === 'number' && Number.isFinite(v))) {
          return `the ${field} holds something that is not a number: a fold is arithmetic on `
               + 'every weight in the last layer, and one hole in it makes a network that loads '
               + 'and drives nothing';
        }
        return null;
      };
      const gain = body.gain, offset = body.offset;
      const badGain = asVector(gain, 'gain'), badOffset = asVector(offset, 'trim');
      if (badGain) return { error: badGain };
      if (badOffset) return { error: badOffset };
      // THE ENVELOPE, TRANSCRIBED FROM THE CLIENT RATHER THAN TRUSTED TO IT.
      // DuckTuner.TuningVector: a gain within 0.7–1.3 (the range the training
      // config randomises foot friction over, the one multiplicative range
      // anybody has evidence about) and a trim within ±0.05 rad or half the
      // joint's room between home and its nearer stop, whichever is smaller.
      // A bench whose only guard was a client's clamp was one caller away from
      // folding gain 0 (a dead network) or 1e9 (a diverged one) and scoring it.
      for (let k = 0; k < width; k++) {
        const joint = C.jointNames[k < 9 ? k : k + 1];   // the mouth, slot 9, has no policy output
        if (gain[k] < TUNE_ENVELOPE.gainLower || gain[k] > TUNE_ENVELOPE.gainUpper) {
          return { error: `the gain for ${joint} is ${gain[k]}, outside the ${TUNE_ENVELOPE.gainLower}–`
                        + `${TUNE_ENVELOPE.gainUpper} this search is allowed to try` };
        }
        const room = Math.max(0, Math.min(HI[k] - HOME[k], HOME[k] - LO[k]));
        const limit = Math.min(TUNE_ENVELOPE.offsetLimit, room / 2);
        if (Math.abs(offset[k]) > limit + 1e-12) {
          return { error: `the trim for ${joint} is ${offset[k]} rad, outside the ±${limit.toFixed(4)} `
                        + 'this search is allowed to try' };
        }
      }

      // The drops are the caller's, because the whole point of asking for
      // several is that the client wants a SPREAD it can read a noise floor
      // out of. Absent, it is the one height every other endpoint drops from.
      const drops = (Array.isArray(body.drops) && body.drops.length
                       ? body.drops : [0.1231]).map(Number);
      if (!drops.every(d => Number.isFinite(d) && d > 0 && d < 1)) {
        return { error: 'every drop height must be a finite number of metres between 0 and 1' };
      }
      if (drops.length > 32) return { error: 'at most 32 drop heights in one call' };
      // DUPLICATES ARE REFUSED, because the rollouts are deterministic: two
      // identical drops are one episode counted twice, and a noise floor read
      // off two identical rewards is a confident zero that waves anything through.
      if (new Set(drops.map(d => d.toFixed(9))).size !== drops.length) {
        return { error: 'the drop heights repeat; every drop is one deterministic episode, and '
                      + 'the same height twice is the same episode counted twice' };
      }
      const seconds = Math.min(Math.max(+body.seconds || 6, 0.2), 30);
      const duck = pickDuck(url, body);

      // WHAT WAS ASKED FOR, ANSWERED OR REFUSED — never quietly narrowed.
      const asked = Array.isArray(body.terms) && body.terms.length
                      ? body.terms.map(String) : TUNE_TERMS.slice();
      const wanted = [], refused = [];
      for (const term of asked) {
        if (TUNE_TERMS.includes(term)) { if (!wanted.includes(term)) wanted.push(term); continue; }
        refused.push({ name: term, why: TUNE_REFUSALS.get(term)
          ?? `this bench knows no reward term by that name; it computes ${TUNE_TERMS.join(', ')}` });
      }

      // THE PARAMETERS, WHICH ARE NOT THE SAME THING AS THE POLICY FILE. On a
      // phone the two are one — the shell serves canonical bytes under the
      // policy's own name — and on the desk the .onnx is what `readAsset`
      // hands back and the canonical bytes sit beside it. The shell is the half
      // that knows, so it is asked; where it does not say, a file that is
      // exactly the canonical length IS the canonical bytes.
      let params;
      try {
        // AN UPLOAD FIRST, under the name /upload handed back — which carries
        // no extension and is in no catalogue. Its parameters are either the
        // `.params` a desk client sent beside the file, or the `.onnx` itself
        // where a phone client sent canonical bytes under that name.
        let bytes = null;
        if (env.scratch.has(`${name}.params`)) bytes = env.scratch.get(`${name}.params`);
        else if (env.scratch.has(`${name}.onnx`)) {
          const held = env.scratch.get(`${name}.onnx`);
          if ((held.byteLength ?? held.length) === FLOAT_COUNT * 4) bytes = held;
          else {
            throw new Error(`${name} was uploaded as a file without its canonical parameters, `
                          + 'and /tune folds a gain into the last layer, which means holding '
                          + 'them: send `parameters` beside `onnx` to /upload');
          }
        }
        if (!bytes) {
          const known = catalogue();
          if (!known.has(name)) throw new Error(`unknown policy: ${name}`);
          bytes = env.readParameters
            ? await env.readParameters(known.get(name), name)
            : await env.readAsset(known.get(name));
        }
        const seen = bytes.byteLength ?? bytes.length;
        if (seen !== FLOAT_COUNT * 4) {
          throw new Error(`${name} is ${seen} bytes where this architecture's canonical `
                        + `parameters are ${FLOAT_COUNT * 4}: /tune folds a gain into the last `
                        + 'layer, which means holding the parameters, and this bench cannot '
                        + 'produce them for that file');
        }
        params = loadParameters(bytes);
      } catch (error) {
        return { error: String(error?.message || error) };
      }

      // The neutral pose the base policy declares, taken from the session the
      // rest of the bench already loaded — a fold changes the last layer and
      // changes nothing about what the observation is a deviation from.
      let reference = HOME;
      try { reference = (await policy(name)).reference; }
      catch { /* a shell that cannot make a session still folds: HOME by identity */ }

      let folded;
      try { folded = foldParameters(params, gain, offset); }
      catch (error) { return { error: String(error?.message || error) }; }
      const loaded = { name, reference, net: { name, run: obs => forward(folded, obs) } };

      const perDrop = [];
      const pooled = Object.fromEntries(TUNE_TERMS.map(t => [t, 0]));
      let pooledTicks = 0, pooledRateTicks = 0, standing = 0, diverged = 0;
      // THE TRACE ITSELF, WHEN A CALLER ASKS FOR IT, AND IT IS NOT A DEBUG
      // FLAG. The claim /tune makes is that this bench computes the SAME six
      // terms StudioKit's `RunMetrics` computes, and the only way to check a
      // claim like that is to hand both sides the same ticks and compare.
      // Without this the fixture behind that gate would have to be produced by
      // a private code path, and the gate would prove that the private path
      // agrees with Swift while saying nothing about the endpoint anybody
      // calls. It is off unless asked for, it is the FIRST drop only, and it
      // is capped — 300 ticks of 47 numbers is a quarter of a megabyte.
      const wantsTrace = body.trace === true || body.trace === 'true';
      const TRACE_CAP = 500;
      let shown = null;
      for (const drop of drops) {
        const trace = [];
        const run = await batchLane(() => rollout({
          name, seconds, schedule: body.schedule, drop, duck, loaded, trace }));
        if (wantsTrace && !shown) {
          shown = trace.slice(0, TRACE_CAP).map(f => ({
            root: f.root, qvel: f.qvel, twist: twistOf(f.root, f.qvel),
            joints: f.joints, action: f.action, command: f.command,
          }));
        }
        if (trace.diverged) {
          // NOT SCORED, NOT STANDING, AND SAID BY NAME: the candidate is the
          // failure, and the caller's search goes on without it.
          diverged++;
          perDrop.push({ drop, diverged: true, tick: trace.diverged.tick, why: trace.diverged.why,
                         travelled: 0, standing: false, terms: {}, netDisplacement: 0 });
          continue;
        }
        const last = run.roots[run.roots.length - 1];
        const stood = upright(last) && last[2] >= 0.100;
        if (stood) standing++;
        const { sums, ticks, rateTicks } = rewardSums(trace, DUCK_JOINTS, HOME);
        for (const term of TUNE_TERMS) pooled[term] += sums[term];
        pooledTicks += ticks; pooledRateTicks += rateTicks;
        const each = {};
        for (const term of wanted) {
          each[term] = sums[term] / (term === 'action_rate_l2' ? rateTicks : ticks);
        }
        perDrop.push({
          drop, travelled: travelledAlongCommand(trace), standing: stood, terms: each,
          // The other half of the pair, for a reader who wants to tell a duck
          // that walked in a circle from one that walked in a line.
          netDisplacement: netDisplacement(trace),
          endHeight: r4(last[2]),
        });
      }
      const terms = {};
      const scored = drops.length - diverged;
      if (scored > 0) {
        for (const term of wanted) {
          terms[term] = pooled[term] / (term === 'action_rate_l2' ? pooledRateTicks : pooledTicks);
        }
      } else {
        for (const term of wanted) {
          refused.push({ name: term, why: 'every episode diverged before it could be scored' });
        }
      }
      const travels = perDrop.map(d => d.travelled).sort((a, b) => a - b);
      return {
        policy: name,
        duck: duck.name,
        episodes: drops.length,
        scored,
        diverged,
        divergedWhy: diverged
          ? 'an episode whose state or action stopped being a duck — non-finite, or past a '
            + 'thousand in any coordinate — is named in perDrop with the tick it happened at, '
            + 'counted here, and left OUT of the pooled terms and the standing count. Score it '
            + 'as a failed candidate; do not score the numbers beside it.'
          : undefined,
        config: 'microduck_velocity_env_cfg',
        configWhy: 'the variances and the pose tolerances are the velocity config\'s, whatever '
                 + 'policy was named — a policy trained under another task is scored under this '
                 + 'one and a client that cares should refuse the mismatch',
        standing,
        criterion: 'ends standing: at the last tick the trunk\'s own up is still up — gravity '
                 + 'projects past −0.5 into the body\'s −z, the same test /measure and /perform '
                 + 'use — and the trunk is at least 100 mm above the floor. A duck that stood '
                 + 'still passes it perfectly, which is why `travelled` is beside it.',
        travelled: travels.length % 2
          ? travels[travels.length >> 1]
          : (travels[travels.length / 2 - 1] + travels[travels.length / 2]) / 2,
        travelledWhy: 'metres, the MEDIAN over the drops: the signed projection of the driven '
                    + 'span\'s net displacement (first driven tick to last) onto the direction '
                    + 'the schedule commanded, expressed in the frame the duck started the driven '
                    + 'span in. RunMetrics\'s "Forward" is the special case of this where that '
                    + 'starting yaw is zero and the command is +x. A schedule that commands no '
                    + 'linear velocity has no such direction, and the plain net displacement is '
                    + 'reported instead. `minTravelled` is the least of the drops, for a guard '
                    + 'that must not let one dead episode hide behind two live ones.',
        minTravelled: travels.length ? travels[0] : 0,
        terms,
        termsWhy: 'per-tick means, pooled over every episode — the sums divided by the total '
                + 'ticks, not a mean of per-episode means, which over episodes of unequal '
                + 'length is the mean of nothing. `action_rate_l2` differences consecutive '
                + 'decisions, so its denominator is one less per episode. The WEIGHTS are the '
                + 'client\'s: this bench does not say what a reward is worth.',
        perDrop,
        refused,
        engine: 'policyforward.mjs over duckkit\'s canonical parameter bytes, always — a fold '
              + 'is arithmetic on parameters, and onnxruntime runs a graph rather than offering '
              + 'one. Agrees with onnxruntime to 3.5e-6 per action, which is why a /tune number '
              + 'and a /record number about one policy are close relatives and not the same '
              + 'measurement.',
        seconds,
        plantName: PLANT,
        plantDigest: PLANT_DIGEST,
        ...(shown ? {
          trace: shown,
          traceWhy: `the first drop's ${shown.length} control ticks, unrounded, asked for with `
                  + '`"trace": true`: per tick the trunk\'s pose, MuJoCo\'s own free-joint '
                  + 'velocity (linear in the world, angular in the body), that velocity as the '
                  + 'trunk\'s own twist, the fourteen joint angles, the network\'s own fourteen '
                  + 'outputs and the command. It is here so that the claim "this bench scores '
                  + 'the terms StudioKit\'s RunMetrics scores" can be CHECKED against the '
                  + `endpoint rather than against a private code path. Capped at ${TRACE_CAP} `
                  + 'ticks.',
        } : {}),
      };
    }

    if (url.pathname === '/measure') {
      // The randomisation is Pollen's own, as measure_success.mjs uses it: the
      // drop height is what a bench can vary without touching the model.
      const rollouts = Math.min(Math.max(+body.rollouts || 8, 1), 32);
      const seconds = Math.min(Math.max(+body.seconds || 3, 0.2), 30);
      const duck = pickDuck(url, body);
      let ok = 0; const heights = [];
      for (let i = 0; i < rollouts; i++) {
        const drop = 0.12 + (0.01 * i) / Math.max(rollouts - 1, 1);
        const run = await batchLane(() =>
          rollout({ name: body.policy, seconds, schedule: body.schedule, drop, duck }));
        const last = run.roots[run.roots.length - 1];
        heights.push(last[2]);
        if (upright(last) && last[2] >= 0.100) ok++;
      }
      heights.sort((a, b) => a - b);
      return {
        policy: body.policy, rollouts, achieves: ok, duck: duck.name,
        criterion: 'ends standing, trunk at least 100 mm up',
        randomised: 'drop height 0.12-0.13 m (Pollen’s range)',
        medianHeight: r4(heights[heights.length >> 1]), worstHeight: r4(heights[0]),
      };
    }

    /**
     * GET /climb/grid — THE FOURTEEN CELLS, SO A CLIENT NEVER RETYPES THEM.
     *
     * The grid is not a preference. It is the axis set the round-4 judge scored
     * every published move on: three rises (h-10, h, h+10 mm) crossed with three
     * plants (nominal; a 10 mm higher fall on friction x0.7; a 5 mm higher fall
     * on friction x1.3) for the CORE nine, plus five more that round 3's grid
     * could not see — +/-5 mm on the nominal plant, and a slippery plant
     * (friction x0.5, drop 0.140) crossed with the three core rises. A client
     * that hard-codes fourteen numbers is a client that will one day be scoring
     * a different grid and reporting it under the same name, so the bench that
     * runs them is the one that lists them.
     */
    if (url.pathname === '/climb/grid') {
      const rig = await climbRig();
      return {
        cells: gridCells(),
        nCore: 9, nExt: 14,
        bar: 7,
        barWhy: 'the round-4 judge\'s bar: 7 of the 9 core cells cleared STABLY. '
              + 'The best move in the published corpus reaches 5.',
        uprightTailMin: UPRIGHT_TAIL_MIN,
        clearBonus: CLEAR_BONUS,
        riserX_m: CLIMB_RISER_X, lateral_m: CLIMB_LATERAL, ceilingAbove_m: CEILING_ABOVE,
        stairs: { count: 4, run_m: 0.28, start_m: 0.12 },
        declaredBounds: CLIMB_BOUNDS,
        criterion: CRITERION_SENTENCE,
        climbable: !!rig,
        ...(rig ? {} : { why: CLIMB_WHY }),
        plantName: PLANT, plantDigest: PLANT_DIGEST,
      };
    }

    /**
     * POST /climb — ONE CELL OF THE GRID, FOR ONE MOVE, IN THE REQUEST BODY.
     *
     * WHY ONE CELL AND NOT THE GRID. A cell is 0.5 s of settle, up to 4.1 s of
     * track and 1.0 s of tail — call it five and a half seconds of simulated
     * duck, measured at 0.57 s of wall clock on a Raspberry Pi 5 and 0.58 s in
     * that Pi's own browser shell. Fourteen of them is eight seconds here and
     * an unknown number on a phone that has never been timed, and a request
     * that runs that long is a request that times out somewhere between the app
     * and here with nothing at all to show for it. One cell answers in about a
     * second, the client asks fourteen times, and it can draw a progress row
     * and stop halfway.
     *
     * WHAT IT IS NOT. It is not a training signal and it is not a robot. It
     * scores a move against a staircase in this plant, and the answer says
     * which plant by digest, because a number from a phone and a number from
     * the desk are comparable only when the world and the criterion are.
     *
     *   body { intent: <the harness intent JSON>, rise: metres,
     *          cell: { dh, drop, fmul }, tail: "policy" }
     *
     * The intent is the file format climb/*.json uses — keyframes, blend, gap,
     * side, approach, and the optional event/servo/spawn blocks — and it is
     * scored EXACTLY as climb/rig3.mjs scoreSaved() would score the same JSON
     * saved to disk, because it is the same function. sim/climb_parity.mjs is
     * the acceptance test: every cell of the fourteen, five published files,
     * against climb/robust.mjs scoreRobust, at full float digits.
     */
    if (url.pathname === '/climb') {
      const started = (typeof performance === 'object' ? performance.now() : Date.now());
      const rig = await climbRig();
      if (!rig) {
        return { error: `no /climb here: ${CLIMB_WHY}`, climbable: false, why: CLIMB_WHY,
                 plantName: PLANT, plantDigest: PLANT_DIGEST };
      }
      let intent;
      try { intent = climbCheckIntent(body.intent, 'the request body'); }
      catch (e) { return { error: String(e.message || e) }; }
      const rise = +body.rise;
      if (!(rise > 0 && rise < 1)) return { error: 'climb needs a rise in METRES, 0 < rise < 1' };
      const cell = body.cell || {};
      const dh = cell.dh === undefined ? 0 : +cell.dh;
      const drop = cell.drop === undefined ? 0.120 : +cell.drop;
      const fmul = cell.fmul === undefined ? 1.0 : +cell.fmul;
      if (![dh, drop, fmul].every(Number.isFinite)) return { error: 'climb needs a cell of finite dh, drop and fmul' };
      const tail = body.tail === undefined ? 'policy' : String(body.tail);
      if (tail !== 'policy') {
        return { error: `this bench scores the grid, and the grid is tail "policy": ${tail} is not one of its cells` };
      }
      const hash = await env.sha256(new TextEncoder().encode(intentHashPayload(intent)));
      const answered = {
        hash, move: hash.slice(0, 12), rise, cell: { dh, drop, fmul }, tail,
        plantName: PLANT, plantDigest: PLANT_DIGEST,
        criterion: CRITERION_SENTENCE,
      };
      // ROUND 4, HOLE 4: bounds are enforced HERE, at scoring time, and not
      // only declared in a comment. A file outside the box it declared is not a
      // result, so it is not scored: it comes back invalid, with the parameter,
      // the value and the box named.
      const B = climbCheckBounds(intent);
      if (B.violations.length) {
        return { ...answered, invalid: true, bounds: B.bounds, boundViolations: B.violations,
          why: 'out of declared bounds: '
             + B.violations.map(v => `${v.param} = ${v.value} is outside [${v.lo}, ${v.hi}]`).join('; ')
             + '. A search that left its own declared box is not a result, so this cell is not scored.',
          seconds: ((typeof performance === 'object' ? performance.now() : Date.now()) - started) / 1000 };
      }
      const E = await climbJob(() => rig.runEpisode(
        intent.keyframes, climbOptsOf(intent), rise + dh, 'policy',
        { drop, fmul, isolate: intentIsolate(intent), stepCount: intentStepCount(intent) }));
      const s = E.afterTail;                        // the grid scores after the 50-tick policy tail
      const crit = climbCriteria(rise + dh, s);
      const honest = crit.honest;
      const stable = honest && E.uprightTailTicks >= UPRIGHT_TAIL_MIN;
      // FULL FLOAT DIGITS, in millimetres. Not rounded: sim/climb_parity.mjs
      // compares these against robust.mjs's own numbers with Object.is, and a
      // toFixed here would make that gate unable to tell a moved trajectory
      // from a rounded one.
      const mm = v => v * 1000;
      return {
        ...answered, invalid: false, why: null,
        honest, stable,
        uprightTailTicks: E.uprightTailTicks, tailTicks: E.tailTicks,
        above_mm: mm(s.above), x_mm: mm(s.x), dy_mm: mm(s.dy),
        feetOnTread: s.feetOnTread, feetOnTreadMax: E.feetOnTreadMax,
        peakAboveTread_mm: mm(E.maxZ - (rise + dh)),
        maxTq: E.maxTq,
        penetrationAtScore_mm: s.penetrationAtScore === null ? null : mm(s.penetrationAtScore),
        minPenetrationEpisode_mm: E.penetration.min === null ? null : mm(E.penetration.min),
        maxAbsDY_mm: mm(E.maxAbsDY),
        // ROUND 4, HOLE 2: a cell pays its upright credit only if the duck got
        // somewhere in it. Do-nothing never crosses x = 120 mm and so earns 0.
        reachedFlight: climbReachedFlight({ maxX: E.maxX, feetOnTreadMax: E.feetOnTreadMax }),
        seconds: ((typeof performance === 'object' ? performance.now() : Date.now()) - started) / 1000,
      };
    }
    return null;
  }
  // ── what a shell needs from a bench ───────────────────────────────────
  //
  // `plant` rather than a filename: the digest is the half that decides
  // whether two measurements are comparable, and handing back only the name is
  // how a shell comes to print a world it did not run.
  return {
    handle,
    plant: { name: PLANT, digest: PLANT_DIGEST, scene: SCENE },
    tickHz: C.tickHz,
    timestep: TIMESTEP,
    substeps: SUBSTEPS,
    ducks: DUCKS.map(d => ({ name: d.name, prefix: d.prefix, spawn: d.spawn })),
    stand: STAND,
  };
}

/**
 * base64 -> bytes, without node:buffer and without `atob`.
 *
 * DELIBERATELY LENIENT ABOUT WHITESPACE AND STRICT ABOUT NOTHING ELSE, which
 * is what `Buffer.from(s, 'base64')` was: a policy arrives as one long line in
 * a JSON body, and a client that wrapped it at 76 characters is not sending a
 * broken file. Characters outside the alphabet are dropped rather than thrown
 * on, so this decodes exactly what the Node shell used to.
 */
export function decodeBase64(text) {
  const A = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  const lookup = new Int8Array(256).fill(-1);
  for (let i = 0; i < A.length; i++) lookup[A.charCodeAt(i)] = i;
  const digits = [];
  for (let i = 0; i < text.length; i++) {
    const v = lookup[text.charCodeAt(i) & 0xff];
    if (v >= 0) digits.push(v);
  }
  const out = new Uint8Array((digits.length * 3) >> 2);
  let o = 0;
  for (let i = 0; i + 1 < digits.length; i += 4) {
    const a = digits[i], b = digits[i + 1], c = digits[i + 2], d = digits[i + 3];
    out[o++] = (a << 2) | (b >> 4);
    if (c !== undefined) out[o++] = ((b & 15) << 4) | (c >> 2);
    if (d !== undefined) out[o++] = ((c & 3) << 6) | d;
  }
  return out.subarray(0, o);
}
