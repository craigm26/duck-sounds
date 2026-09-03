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
         STAIR_RUN as CLIMB_STAIR_RUN, STAIR_START as CLIMB_STAIR_START,
         DECLARED_BOUNDS as CLIMB_BOUNDS } from './climb_score.mjs';
// THE BALL, AND THE ONE EPISODE THAT SCORES A CHASE WITH IT.
//
// The same arrangement, one challenge along: `chase_score.mjs` IS the episode
// `chase/chase_rig.mjs` runs and the one `chase/chase_robust.mjs`'s 14-cell
// grid is decided on, imported rather than transcribed, so POST /chase answers
// the number the package published instead of a number that looks like it. It
// carries Pollen's ball-kick reward — the nine terms this plant can compute and
// the three it refuses by name — and it reaches its interpolation curve and its
// tail bar through `./climb_score.mjs` and its quaternion arithmetic through
// `./reward_math.mjs`, both of which are real files in sim/ and flat files in
// the phone bundles.
import { makeChaseRig, gridCells as chaseGridCells, checkEntrant as chaseCheckEntrant,
         entrantHashPayload as chaseHashPayload, CHASE_REFUSALS, TERMS as CHASE_TERMS,
         CRITERION_SENTENCE as CHASE_CRITERION, TOUCH_MM, TRAVEL_MIN_MM,
         TAIL_TICKS as CHASE_TAIL_TICKS, UPRIGHT_TAIL_MIN as CHASE_UPRIGHT_TAIL_MIN,
         SETTLE_TICKS as CHASE_SETTLE_TICKS, N_CORE as CHASE_N_CORE, N_EXT as CHASE_N_EXT,
         BALL_CAVEAT, CHASE_CONFIG, CHASE_CONFIG_SOURCE, DEFAULT_SECONDS as CHASE_SECONDS,
         ACTION_RATE_SOURCE_WHY } from './chase_score.mjs';
// THE STEP BANK ITSELF, for the one endpoint that is about the world rather
// than about a duck in it. /world lays the fourteen compiled step blocks out
// where a caller asks for them, and it reaches them through the SAME file the
// climb episode reaches them through — the shim in sim/, the flat copy in the
// phone bundles — because two spellings of STEP_HALF_HEIGHT are two different
// tread heights and the one that goes stale is the one nobody reads.
import { findStairJoints, placeSteps, clearStairs, readStairs, STAIR_COUNT, STAIR_Y,
         STAIR_HALF_WIDTH, STEP_HALF_DEPTH, STEP_HALF_HEIGHT } from './stairs.js';

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

/**
 * THE QUATERNION ARITHMETIC IS NOT IN THIS FILE ANY MORE, AND IT IS STILL THIS
 * FILE'S EXPORT.
 *
 * `gravityXYSquared`, `rotate`, `unrotate` and `twistOf` moved to
 * `./reward_math.mjs` when the ball challenge arrived, because
 * `chase_score.mjs` transcribes a DIFFERENT config that shares three of this
 * one's terms and would otherwise have needed its own copy of the same two
 * identities. One copy, two configs. They are re-exported here under the names
 * they have always had, so `tune_parity.mjs` and every other importer is
 * untouched.
 */
export { gravityXYSquared, rotate, unrotate, twistOf } from './reward_math.mjs';
import { gravityXYSquared, rotate, unrotate, twistOf, yawOf } from './reward_math.mjs';

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
  // THE FRICTION BASELINE, READ ONCE, BEFORE ANY LANE RUNS. Both scoring rigs
  // scale the feet's friction for a cell and restore it after; a rig that read
  // its baseline at its own (lazy) construction could read it while the OTHER
  // rig's cell had the multiplier applied, and carry a poisoned baseline for
  // the life of the process. So the baseline is the plant's, captured here.
  const GEOM_FRICTION0 = Float64Array.from(model.geom_friction);
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

  /**
   * THE STEP BANK, AND THE FACT NOBODY HAD WRITTEN DOWN ABOUT IT.
   *
   * The plant compiles fourteen 340×340×200 mm blocks at (0, STAIR_Y, 0), each
   * on an x and a z slide and nothing else. The climb rig and the chase rig
   * each build their OWN mjData and park the bank in their `finally`; the LIVE
   * world has never parked it. So `live.world` boots with all fourteen stacked
   * inside each other at the origin's y-offset, colliding and shoving on every
   * live tick, and it has always done so — every /intent, every /state, every
   * number bench_parity and physics_parity pin was measured in that world.
   *
   * WHICH IS WHY NOTHING HERE PARKS THEM AT BOOT. Parking the bank would be an
   * improvement and it would also move the live trunk trajectory, i.e. every
   * leaf of the parity baselines. A caller that wants a flat room asks for one
   * (POST /world {clear:true}) and is told, in the answer, that asking is a
   * change to the world every published live number ran in.
   *
   * WALKED AT BOOT, BEFORE ANY LANE RUNS, for the same reason GEOM_FRICTION0 is
   * captured here: `stepLive` restores each step geom's conaffinity from this
   * array, and an array read lazily could be read while a climb cell had them
   * zeroed and would then restore zeros for the life of the process.
   */
  const STEPG = [], STEP_CONAFF0 = [];
  for (let g = 0; g < model.ngeom; g++) {
    if (/^step\d+_geom$/.test(model.geom(g).name || '')) {
      STEPG.push(g);
      STEP_CONAFF0.push(model.geom_conaffinity[g]);
    }
  }
  /**
   * The qpos and dof addresses of the bank's twenty-eight slides.
   *
   * `{ isolate: false }` ON PURPOSE. The default zeroes every step geom's
   * conaffinity as a side effect of looking the joints up, which is right for a
   * scoring rig that is about to lay a flight and wrong here: this runs at boot
   * in the shared model, and it would hand every lane — /record, /perform,
   * /climb, /chase — a world whose steps stopped colliding with each other
   * because the live lane once looked up an address. The isolation `stepLive`
   * needs is taken and given back inside one synchronous block instead.
   */
  const STAIR_ADDR = STEPG.length === STAIR_COUNT
    ? findStairJoints(model, { isolate: false }) : null;

  /**
   * The four static walls, read out of the plant by name.
   *
   * They have no joints, so this is a description and not a control: a world
   * cannot move them, and a request that asks for a wall is told so. What they
   * ARE is the limit on where a step may go — a 200 kg block half inside a
   * static wall is a solver problem, not a room — so the inner faces are
   * computed here and the refusal quotes them.
   */
  const ARENA = (() => {
    // `r4` is declared further down and this runs at load, so the rounding is
    // done longhand rather than reaching into the temporal dead zone — the same
    // reason GRASPABLES rounds its masses by hand.
    const q = v => Math.round(v * 10000) / 10000;
    const walls = [];
    for (let g = 0; g < model.ngeom; g++) {
      const name = model.geom(g).name || '';
      if (!/^wall_[nsew]$/.test(name)) continue;
      const s = g * 3;
      const size = [model.geom_size[s], model.geom_size[s + 1], model.geom_size[s + 2]];
      const pos = [model.geom_pos[s], model.geom_pos[s + 1], model.geom_pos[s + 2]];
      // The long axis is the one it runs along; the short one is its thickness.
      const along = size[0] >= size[1] ? 'x' : 'y';
      walls.push({
        name,
        x: q(pos[0]), y: q(pos[1]),
        along,
        halfLength: q(along === 'x' ? size[0] : size[1]),
        halfThickness: q(along === 'x' ? size[1] : size[0]),
        height: q(pos[2] + size[2]),
      });
    }
    walls.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
    const inner = axis => {
      const along = axis === 'x' ? 'y' : 'x';
      const faces = walls.filter(w => w.along === along)
        .map(w => Math.abs(axis === 'x' ? w.x : w.y) - w.halfThickness);
      return faces.length ? q(Math.min(...faces)) : null;
    };
    return {
      walls,
      innerX_m: inner('x'),
      innerY_m: inner('y'),
      why: 'static geoms with no joints: a world can put steps and props between them '
         + 'and cannot move them. The inner faces are where a step block has to stop.',
    };
  })();

  /**
   * THE WORLD THE LIVE LANE IS STANDING IN, or none.
   *
   * `set` false is the bench's own world — the one every published live number
   * was measured in, bank stacked and all — and while it is false the live tick
   * is byte-for-byte the tick it has always been. Once something has asked for a
   * world, this is what /reset re-lays after `mj_resetData` wipes it and what
   * GET /world reads back.
   */
  const WORLD = { set: false, name: null, steps: [], ball: null, props: [], unexpressed: [] };

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
   * One live tick, in whatever world the live lane has been asked to stand in.
   *
   * WITH NO WORLD SET THIS IS `stepWorld`, REACHED BY THE SAME CALL. That is
   * the first thing to check when reading it: a bench nobody has asked for a
   * world runs the tick it has always run, through one extra boolean test, so
   * `bench_parity` and `physics_parity` see the trajectory they pinned.
   *
   * WITH A WORLD SET IT BORROWS THE MODEL AND GIVES IT BACK. Two things have to
   * be true at once: the fourteen blocks must be where the world put them (they
   * are 200 kg bodies on frictionless slides — leave them and the solver walks
   * them out of the flight inside one tick, so they are re-written and re-pinned
   * every tick, which is what site/stairs.js means by "call after any qpos
   * write, and every tick"), and the step-step isolation that makes an
   * overlapping flight solid must be on. The isolation is a MODEL field, shared
   * with every other lane, so it is taken and handed back around the one step.
   *
   * THERE IS NO `await` IN HERE AND THERE MUST NEVER BE ONE. The isolation is
   * atomic only because this block is synchronous: JavaScript will not run
   * another lane's code between the zeroing and the `finally`. One `await`
   * inside it and a /record or a /climb that happens to overlap a steering loop
   * would step ITS duck through a world with the step blocks' conaffinity
   * zeroed — a different plant, silently, in a number somebody keeps. The
   * `finally` is what makes a throw safe; `world_parity.mjs` phase 4 is what
   * proves the leak does not happen, by scoring a /climb cell and a /record
   * with a world standing and requiring the same answers as without.
   *
   * (climb_score.mjs:431–443 takes the same loan the same way, for the same
   * reason, around a whole episode.)
   */
  function stepLive(d) {
    // A plant without a full bank has nothing to lay and nothing to isolate;
    // a world set on it (a bare-floor request, say) must not break driving.
    if (!WORLD.set || !STAIR_ADDR) return stepWorld(d);
    stepIsolated(d, WORLD.steps);
  }

  /**
   * ONE TICK INSIDE THE LOAN — the four statements above, once, for both lanes.
   *
   * THERE IS NO `await` IN HERE AND THERE MUST NEVER BE ONE, for the reason
   * `stepLive` gives above: `model.geom_conaffinity` is a MODEL field shared by
   * every lane, and the isolation is atomic only because this block is
   * synchronous. Both callbacks are synchronous qpos reads and neither may grow
   * an `await` either.
   *
   * `onLaid` IS THE ONLY MOMENT A LAID WORLD CAN BE READ EXACTLY. `placeSteps`
   * writes qpos and zeroes qvel, so between it and `stepWorld` the blocks are
   * precisely what was laid; four substeps later a 200 kg block on a
   * frictionless, undamped slide has fallen about two millimetres, which is
   * visible at the readback's 1e-4 rounding and cannot be compared with a POST
   * /world readback leaf for leaf. `onStepped` is the other end of that same
   * tick and is how far it fell — the number that says the pin is per tick and
   * not per run.
   */
  function stepIsolated(d, steps, onLaid = null, onStepped = null) {
    STEPG.forEach(g => { model.geom_conaffinity[g] = 0; });
    try {
      placeSteps(d, STAIR_ADDR, steps);
      if (onLaid) onLaid(d);          // the readback, exact: post-write, pre-step
      stepWorld(d);
      if (onStepped) onStepped(d);    // sag only
    } finally {
      STEPG.forEach((g, i) => { model.geom_conaffinity[g] = STEP_CONAFF0[i]; });
    }
  }

  /**
   * The single-duck control tick: choose this duck's commands, then advance time.
   *
   * THE SAME TICK FOR EVERY CALLER that drives one duck — /record, /measure,
   * /perform and /capture all come through here, so a policy that measured 16/16
   * behaves identically wherever it is run. The live world does NOT use this one:
   * it actuates every duck first and calls `stepWorld` once, which is the same
   * sequence with the loop in the right place.
   *
   * HOW TIME MOVES IS THE LAST PARAMETER AND DEFAULTS TO WHAT IT ALWAYS WAS.
   * `stepIn` is `stepWorld` for every caller that does not pass one, so
   * /record, /measure and /capture reach the same statement they always did
   * through a parameter with a default rather than through a name. A /perform
   * that was given a world passes a closure instead, which takes the step
   * geoms' conaffinity, re-lays the blocks, steps, and hands it back — the same
   * loan `stepLive` takes on the live lane, per tick, synchronously. The
   * `await` above it has already returned by then: the loan never spans one.
   */
  async function tick(d, duck, loaded, last, cmd, offsets = null, blend = 1, capture = null,
                      expert = null, teacherShare = 0, jitter = 0, stepIn = stepWorld) {
    const fedBack = await actuate(d, duck, loaded, last, cmd, offsets, blend, capture,
                                  expert, teacherShare, jitter);
    stepIn(d);
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
                           loaded = null, trace = null,
                           world = null, spawn = null, stood = null, lanes = null }) {
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
    // ─────────────────────────────────────────────────────────────────────────
    // THE DUCK MOVES TO THE BANK, BECAUSE THE BANK CANNOT MOVE TO THE DUCK.
    //
    // The fourteen step blocks are COMPILED at y = 1.3050000000000002 with an x
    // slide and a z slide and no y joint, so a flight can only ever stand on
    // that one row. A world laid without a spawn is a room the duck stands
    // BESIDE — a true picture of a useless run — so a caller that lays one can
    // say where the duck goes, and this is climb_score.mjs:450-458's rule
    // copied exactly: x and y written straight, and this rollout's drop added
    // to z as an offset from 0.120. With `spawn.z` defaulting to 0.120 that
    // reproduces /perform's existing 0.120 → 0.130 sweep byte for byte, so the
    // same motion starts at the same height on both routes.
    if (spawn) {
      data.qpos[f]     = spawn.x;
      data.qpos[f + 1] = spawn.y;
      data.qpos[f + 2] = (spawn.z === undefined ? 0.120 : spawn.z) + (drop - 0.120);
      data.qpos[f + 3] = 1; data.qpos[f + 4] = 0; data.qpos[f + 5] = 0; data.qpos[f + 6] = 0;
    }
    // ONCE PER ROLLOUT, NOT PER TICK. A prop and a ball are things the duck
    // MOVES; re-seating them every tick would make them walls. The blocks are
    // the other kind and are re-laid inside the loan at the top of every
    // stepped tick below.
    if (world) layWorld(data, world);
    if (world || spawn) mj.mj_forward(model, data);
    // READ OUT OF qpos, AT THE WRITE, NEVER ECHOED FROM THE REQUEST. This is
    // the one moment the duck is where it was put; by the last tick it has
    // walked somewhere else, which is the whole point of the run.
    if (stood) stood.placed(data, f);

    // THE TICK THIS ROLLOUT STEPS WITH, made ONCE. `tNow` mirrors the loop's
    // own `t` because the closure is built before the loop and the readback
    // fires on exactly one tick of it — rollout 0's last recorded one.
    let tNow = -settle;
    const lastRecorded = ticks - 1;
    const layAt = d => { if (stood && tNow === lastRecorded) stood.lay(d, tNow); };
    const sagAt = d => { if (stood && tNow === lastRecorded) stood.sag(d); };
    const bump = () => {
      if (!lanes) return;
      lanes.performTicks++;
      // `batchInflight > 1`: this rollout's own batch job is one of them.
      if (lanes.liveInflight || lanes.climbInflight || lanes.batchInflight > 1) {
        lanes.performTicksWithAnotherRequestInFlight++;
      }
    };
    // A rollout nobody asked anything new of — /record, /measure and
    // /capture, which pass no world, no spawn and no lanes — steps through
    // the identical statement it always did: `stepIn` IS `stepWorld`, not a
    // wrapper around it, so those routes cannot move by a floating-point
    // hair. Every /perform passes `lanes` and takes the counted path, whose
    // no-world case is pinned leaf for leaf by bench_parity instead.
    const stepIn = (world === null && stood === null && lanes === null)
      ? stepWorld
      : ((world && STAIR_ADDR)
          ? (d => { bump(); stepIsolated(d, world.steps, layAt, sagAt); })
          : (d => { bump(); layAt(d); stepWorld(d); sagAt(d); }));
    for (let t = -settle; t < ticks; t++) {
      tNow = t;
      const cmd = command(t >= 0 ? commandAt(schedule, t / C.tickHz) : {});
      // NEUTRAL THROUGH THE SETTLE. The settle exists to let the drop-bounce die;
      // feeding the track through it starts the motion before recording does.
      const offsets = track && t >= 0 ? poseAt(track, t / C.tickHz) : null;
      // NOTHING IS CAPTURED DURING THE SETTLE. Those ticks are the drop bounce
      // dying under a different policy; teaching a clone from them would teach it
      // to recover from a fall it will never be dropped into.
      last = await tick(data, duck, t < 0 ? settling : net, last, cmd, offsets, blend,
                        t >= 0 ? capture : null, t >= 0 ? expert : null, teacherShare,
                        t >= 0 ? jitter : 0, stepIn);
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
  const liveLane0 = lane(), batchLane0 = lane(), climbLane = lane();

  /**
   * WHO ELSE IS IN THE BUILDING — a diagnostic, and never part of a scored
   * answer.
   *
   * world_parity phase 4 awaits every call before it issues the next one, so a
   * loan held across an `await` would pass it: sequential is not a proof of
   * atomicity. Phase 5g issues an overlapping pair on purpose — but a race that
   * did not actually race proves nothing either, and a green phase that never
   * overlapped is worse than a red one. So the bench counts: every /perform
   * control tick, and how many of those ticks were stepped while some OTHER
   * request had been accepted and not yet answered. Phase 5g fails itself if
   * that second number does not move.
   *
   * IN FLIGHT MEANS ACCEPTED AND NOT YET ANSWERED — queued as much as running.
   * Node is single-threaded and each lane is serial, so a request that arrives
   * mid-/perform is queued rather than interleaved; "queued behind a /perform
   * that is holding the step geoms' conaffinity" is exactly the situation the
   * loan has to survive, and it is the one a counter can see.
   */
  const LANES = { performTicks: 0, performTicksWithAnotherRequestInFlight: 0,
                  liveInflight: 0, climbInflight: 0, batchInflight: 0 };
  /** A lane, with a gauge of how many jobs are on it. Ordering is untouched. */
  const gauged = (key, submit) => job => {
    LANES[key]++;
    const run = submit(job);
    const done = () => { LANES[key]--; };
    run.then(done, done);
    return run;
  };
  const liveLane = gauged('liveInflight', liveLane0);
  // THE BATCH LANE IS GAUGED TOO. /record, /measure, /capture and /tune share
  // /perform's mjData through it, and an overlap counter blind to them would
  // report a /perform that raced a /measure as a /perform that ran alone.
  const batchLane = gauged('batchInflight', batchLane0);

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
  // The gauge is taken on the OUTSIDE only: a climb job passes through the live
  // lane, and counting it there too would report one request as two and make
  // `liveInflight` mean something other than its name.
  const climbJob = gauged('climbInflight',
    job => climbLane(() => liveLane0(() => batchLane0(job))));
  /**
   * A /chase cell borrows the model too, for the same reason and on the same
   * lane. Its plant axis multiplies the foot geoms' friction exactly as a climb
   * cell's does — written back in a `finally` inside the shared episode — and
   * while it runs, a /intent stepping the live world or a /measure stepping the
   * batch world would be stepping a duck with different feet. The two
   * challenges share `climbLane` rather than taking one each, because a chase
   * cell and a climb cell touch the same friction array and two lanes would let
   * them interleave.
   */
  const chaseJob = climbJob;

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
  let CLIMB, CLIMB_WHY = null, CLIMB_BUILD;
  /**
   * The climb rig's own mjData, kept by name.
   *
   * The rig owns it and resets it at the top of every episode; this reference
   * exists so that the /climb route can read the ball and the graspables out of
   * the world the episode ran in when a caller asks for a clip. THE STEP BLOCKS
   * ARE NOT READ THROUGH IT — `runEpisode`'s `finally` parks all fourteen at
   * (i·1.5, −5), so a post-hoc read of the bank would report a parked flight as
   * the one the duck climbed. Those come out of the episode itself.
   */
  let CLIMB_DATA = null;
  /** ONE construction, however many callers arrive during it: the sentinel is
   *  the promise, not a null written before the first await — two concurrent
   *  first calls used to make the second answer "no /climb here: null". */
  async function climbRig() {
    if (CLIMB_BUILD === undefined) CLIMB_BUILD = buildClimbRig();
    return CLIMB_BUILD;
  }
  async function buildClimbRig() {
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
    CLIMB_DATA = new mj.MjData(model);
    const rig = makeClimbRig({
      geomFriction0: GEOM_FRICTION0,
      mj, model, data: CLIMB_DATA,
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
   * THE CHASE RIG, BUILT ON FIRST ASK AND NOT AT BOOT — climbRig's twin.
   *
   * It costs an mjData and a policy load, and a bench that is only ever asked
   * to /record should not pay for a ball challenge it never scores. `CHASE` is
   * `undefined` until something asks, then either the rig or `null` — and null
   * is an answer, not a failure: it means this plant cannot be asked, and
   * /chase says WHICH of the things is missing rather than throwing.
   *
   * ITS OWN mjData, for the reason the climb rig has one: a chase episode
   * resets its world fourteen times per grid, places a ball in it and drives a
   * duck across it, and it must not be the world a steering loop is keeping or
   * the one /record resets.
   *
   * WHY THE ACTUATOR CHECK. The shared episode writes `data.ctrl[k]` for
   * k = 0..13, which is what every scored chase number was measured through. In
   * the canon one-duck scene actuator k IS joint k; in a multi-duck scene it is
   * not, and a chase scored there would be driving the first duck's legs.
   */
  let CHASE, CHASE_WHY = null, CHASE_BUILD;
  async function chaseRig() {
    if (CHASE_BUILD === undefined) CHASE_BUILD = buildChaseRig();
    return CHASE_BUILD;
  }
  async function buildChaseRig() {
    if (CHASE !== undefined) return CHASE;
    CHASE = null;
    const duck = DUCKS.find(d => d.prefix === '');
    if (!duck) { CHASE_WHY = `this scene's ducks are all prefixed (${DUCK_NAMES.join(', ')}); `
      + 'the chase episode drives the unprefixed duck the challenge was measured on'; return CHASE; }
    if (!duck.ctrl.every((a, k) => a === k)) {
      CHASE_WHY = 'this scene\'s actuators are not in joint order, so the shared chase episode '
        + 'would drive the wrong joints'; return CHASE;
    }
    const loaded = await policy(STAND);
    let rig;
    try {
      rig = makeChaseRig({
        geomFriction0: GEOM_FRICTION0,
        mj, model, data: new mj.MjData(model),
        D: duck.joints, HOME, LO, HI, jointNames: C.jointNames,
        buildObs, projectedGravity, command, tickHz: C.tickHz,
        // THE FORWARD PASS IS THE SHELL'S, exactly as it is for /climb:
        // onnxruntime on the desk, policyforward.mjs in a browser. They agree
        // to 3.5e-6 per action and not exactly (sim/policy_parity.mjs), which
        // is why the answer carries the plant digest and /health the engine.
        stand: { run: obs => loaded.net.run(obs), reference: loaded.reference },
      });
    } catch (error) {
      CHASE_WHY = String(error?.message || error);
      return CHASE;
    }
    if (!rig) { CHASE_WHY = `${PLANT} has no ball: no body called "ball" on a free joint, `
      + 'and a chase without a ball is not a chase'; return CHASE; }
    CHASE = rig;
    return CHASE;
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
      stepLive(live.world);
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
      stepLive(live.world);
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
   * WHAT THIS WORLD CANNOT DO, SAID PER REQUEST RATHER THAN PER DOCUMENT.
   *
   * A caller describes a scene it drew somewhere else; this plant expresses some
   * of it exactly and some of it not at all, and the difference is a fact the
   * caller has to be able to draw. So every gap is an entry here — what was
   * asked, what it got instead, and why — and the answer that comes back is the
   * world that IS standing, never the world that was asked for.
   *
   * REFUSAL AND UNEXPRESSED ARE DIFFERENT ACTS. A step whose block would sit
   * half inside a static wall is refused, because laying it would put a 200 kg
   * body in a place the solver has to fight; a step at a y this bank has no
   * joint for is laid at the y it has and the difference is reported. Refusing
   * everything would make the endpoint unusable; expressing everything silently
   * would make its answers a lie.
   */
  function unexpressed(what, extra) { return { what, ...extra }; }

  /** Put a graspable body down at (x, y), upright and dead still. */
  function seatProp(d, p) {
    const g = GRASPABLES.find(q => q.name === p.name);
    if (!g) return false;
    d.qpos[g.adr] = p.x; d.qpos[g.adr + 1] = p.y; d.qpos[g.adr + 2] = p.z;
    d.qpos[g.adr + 3] = 1;
    for (let k = 4; k < 7; k++) d.qpos[g.adr + k] = 0;
    for (let k = 0; k < 6; k++) d.qvel[g.dof + k] = 0;
    return true;
  }

  /**
   * Put the standing world back after something wiped it.
   *
   * `mj_resetData` is the thing that wipes it — /reset runs it through
   * `ensureStanding`, and it takes every slide and every free body back to
   * qpos0, which for the step bank means fourteen blocks stacked at the origin
   * again. A world that survived a settle and not a reset would be a world that
   * quietly disappears the first time a trial starts, which is exactly when it
   * matters most. No-op with no world set, so the untouched bench keeps its
   * untouched code path.
   */
  function relayWorld() {
    if (!WORLD.set) return;
    if (STAIR_ADDR) placeSteps(live.world, STAIR_ADDR, WORLD.steps);
    for (const p of WORLD.props) seatProp(live.world, p);
    mj.mj_forward(model, live.world);
  }

  /**
   * A PER-REQUEST WORLD, LAID INTO SOMEBODY ELSE'S mjData.
   *
   * `relayWorld` above is the LIVE lane's and is left exactly as it is: it puts
   * the standing world back after `mj_resetData` wiped it, it takes the module
   * global as its subject, and its caller — not it — places the ball, after the
   * relay. That order is what world_parity phase 3 pins, so it is not folded
   * into anything.
   *
   * This one is /perform's: a plan, and the mjData a rollout is about to run
   * in. It writes nothing to the module global and reads nothing from it, so a
   * world sent with a run cannot become the world the live lane is standing in
   * — the bug that would make the next /intent step through a room nobody
   * asked for.
   *
   * NO `mj_forward` HERE. The caller does it once, after the spawn, so a
   * rollout pays for one forward and not three.
   */
  function layWorld(d, plan) {
    if (STAIR_ADDR) placeSteps(d, STAIR_ADDR, plan.steps);
    for (const p of plan.props) seatProp(d, p);
    if (plan.ball && BALL) placeBall(d, plan.ball.x, plan.ball.y, BALL_RADIUS);
  }

  /**
   * THE WORLD AS IT ACTUALLY STANDS, read out of the live mjData.
   *
   * READ, NOT ECHOED. The request is not the answer: a bank with no y joint
   * lays a step somewhere the caller did not name, a parked block is at (i·1.5,
   * −5) and not absent, and the ball is wherever the last kick left it. Reading
   * qpos back is the only version of this that stays true after a tick of
   * physics, and it is what the app draws — so what is on screen is the world
   * the bench built rather than the world the app asked for.
   */
  function worldReadback(d = live.world, plan = (WORLD.set ? WORLD : null), atSteps = null) {
    const steps = [];
    let parked = 0;
    // THE BANK, AS IT READ AT THE MOMENT THAT MATTERS. `atSteps` is a list of
    // `{x, z}` already read somewhere this function cannot reach — inside a
    // scoring episode, at the pin, before the episode's `finally` parked all
    // fourteen. With none it reads `d` here and now, which is what GET /world
    // has always done.
    const bank = atSteps ?? (STAIR_ADDR
      ? STAIR_ADDR.map(a => ({ x: d.qpos[a.x], z: d.qpos[a.z] })) : null);
    if (bank) {
      for (let i = 0; i < STAIR_COUNT; i++) {
        const z = bank[i].z;
        // PARKED IS A MARK SOMEBODY WROTE, NOT A DEPTH SOMETHING FELL TO.
        // `placeSteps` puts an unused block at (i·1.5, −5) and re-writes it
        // every tick, so its x is EXACTLY i·1.5 (nothing pushes along x and the
        // slide's velocity is zeroed) and its z is −5 or the two millimetres
        // below it that one tick of free fall adds after the write.
        //
        // AND ONLY A WORLD PARKS ANYTHING. With no world set nothing has ever
        // called `placeSteps` on this mjData, so nothing is parked and all
        // fourteen are listed wherever they are — which on a fresh bench is a
        // column from about −11.7 m to +1.6 m, because they boot stacked and
        // 200 kg bodies on frictionless slides do not stay stacked. That is the
        // fact this readback exists to stop being invisible, and calling those
        // blocks "parked" would be reporting an intention as a position.
        // THE FIRST `WORLD.steps.length` BLOCKS ARE LAID, whatever their qpos
        // reads after a tick — that is the contract placeSteps implements — and
        // only the ones after them are tested against the park mark. Inferring
        // "laid" from the mark alone could drop a laid block out of the readback.
        if (plan && i >= plan.steps.length && bank[i].x === i * 1.5 && z <= -5) { parked++; continue; }
        steps.push({
          x: r4(bank[i].x), y: r4(STAIR_Y), top: r4(z + STEP_HALF_HEIGHT),
          halfDepth: STEP_HALF_DEPTH, halfWidth: STAIR_HALF_WIDTH,
          halfHeight: STEP_HALF_HEIGHT,
        });
      }
    }
    const b = ballOf(d);
    return {
      world: { set: !!plan, name: plan ? plan.name : null },
      steps,
      ball: b ? { x: b[0], y: b[1], z: b[2] } : null,
      ballRadius: BALL ? BALL_RADIUS : null,
      props: graspableState(d),
      bank: {
        count: STAIR_COUNT,
        present: STAIR_ADDR ? STEPG.length : 0,
        y: r4(STAIR_Y),
        halfDepth: STEP_HALF_DEPTH,
        halfWidth: STAIR_HALF_WIDTH,
        halfHeight: STEP_HALF_HEIGHT,
        yWhy: `every block is COMPILED at y = ${r4(STAIR_Y)} with an x and a z slide and no y `
            + 'joint, so this is where the blocks are and not where a request puts them. '
            + 'Moving them means recompiling the scene (site/stairs.js STAIR_Y).',
        sizeWhy: `the bank is ${STAIR_COUNT} identical `
            + `${r4(STEP_HALF_DEPTH * 2)}×${r4(STAIR_HALF_WIDTH * 2)}×${r4(STEP_HALF_HEIGHT * 2)} m `
            + 'blocks; a step is placed by moving one, so a different size is not '
            + 'something this world can be asked for.',
        stackedAtBoot: 'nothing parks this bank in the live world: it boots stacked at '
            + `(0, ${r4(STAIR_Y)}, 0) and every live number ever published was measured `
            + 'with it there. A flat room is a change, not a default.',
        // NOT HIDDEN, COUNTED. `steps` lists every block this world is not
        // using as a park space — including one that has fallen through the
        // floor, because that is where it IS — and this is the rest. A reader
        // that saw four steps and no count would think the plant had four
        // blocks.
        parked,
        parkedWhy: `${parked} of the ${STAIR_COUNT} blocks are at the park mark, `
            + '(i·1.5, −5) — parked, not absent: this plant compiles the bank and a world '
            + `chooses how many of it to use. The other ${STAIR_COUNT - parked} are listed in `
            + '`steps`, wherever they actually are. Nothing is parked until a world parks it.',
      },
      arena: ARENA,
      unexpressed: plan ? plan.unexpressed : [],
      plantName: PLANT,
      plantDigest: PLANT_DIGEST,
    };
  }

  /**
   * Take a described world, lay what this plant can lay, and say what it could
   * not.
   *
   * ORDER MATTERS: everything is checked before anything is written, so a
   * refused request leaves the world it found. A half-applied world would be a
   * world nobody asked for standing under a duck somebody is steering.
   *
   * NOTHING IS WRITTEN HERE AT ALL. This is the whole of that checking, and it
   * ends by handing back a plan. `setWorld` commits it to the live lane;
   * /perform lays the same plan into its own rollout and never touches the
   * module global. ONE function, so the two routes accept byte-identical bodies
   * and refuse them in the same words in the same order — a /perform-only
   * refusal, or a /perform-only silence, would be a second world route wearing
   * the first one's name.
   *
   * `against` IS THE STANDING WORLD OR NOTHING, AND IT IS LOAD-BEARING. With a
   * world to merge against, a post that mentions only the ball keeps the steps
   * standing (and keeps the steps' notes). With `null` there is nothing to
   * inherit: a request that says nothing about the bank hits the same refusal a
   * FIRST /world gets, which is exactly what a /perform world must do — it can
   * never silently inherit the live lane's flight.
   *
   * `ballAt` READS THE BALL FOR THE ONE NOTE THAT NEEDS IT, out of whichever
   * mjData the caller is talking about, so `"ball": null` produces the same
   * `remove the ball` row on both routes instead of a refusal on one of them.
   *
   * `spawn` HAS THREE STATES AND ONLY /perform PASSES IT. Undefined is "not a
   * route that can move the duck", so no spawn row is ever emitted (the /world
   * route). `null` is "a route that could and was not asked to" — the §2.5 row.
   * A point is "it was asked", and nothing is said.
   */
  function planWorld(body, { against = null, ballAt = null, spawn } = {}) {
    const notes = [];
    const inX = ARENA.innerX_m, inY = ARENA.innerY_m;

    // --- steps -------------------------------------------------------------
    // A STANDING FLIGHT STAYS STANDING when a post says nothing about the
    // bank: moving the ball is not a request to park the stairs.
    let steps = against ? against.steps.slice() : [];
    const asked = body.steps !== undefined && body.steps !== null;
    if (body.clear === true && asked) {
      throw new Error('say one or the other: `clear: true` for a bare floor, or `steps` to lay them');
    }
    if (body.clear === true) {
      steps = [];
    } else if (asked) {
      if (!Array.isArray(body.steps)) throw new Error('steps must be an array of {x, top}');
      if (!STAIR_ADDR) {
        throw new Error(`this plant has no step bank: ${STEPG.length} step geoms, `
                      + `${STAIR_COUNT} wanted`);
      }
      if (body.steps.length > STAIR_COUNT) {
        throw new Error(`${body.steps.length} steps: this world has a bank of ${STAIR_COUNT} `
                      + 'blocks and cannot lay more than it has');
      }
      body.steps.forEach((s, i) => {
        const x = Number(s?.x), top = Number(s?.top);
        if (!Number.isFinite(x) || !Number.isFinite(top)) {
          throw new Error(`step ${i}: give {x, top} as finite numbers`);
        }
        // THE FOOTPRINT, NOT THE CENTRE. A block is 340 mm deep; a centre 100 mm
        // inside the wall still puts 70 mm of a 200 kg body inside a static
        // geom, which is a solver fight and not a room.
        if (inX !== null && Math.abs(x) + STEP_HALF_DEPTH > inX + 1e-12) {
          throw new Error(`step ${i} at x = ${x} reaches `
            + `${r4(Math.abs(x) + STEP_HALF_DEPTH)} m, past the arena's inner face at ${inX} m `
            + `(${ARENA.walls.filter(w => w.along === 'y').map(w => w.name).join(' and ')}); `
            + `a ${r4(STEP_HALF_DEPTH * 2)} m block has to fit between the walls`);
        }
        steps.push({ x, top });
        // Asked-for fields this bank cannot honour. Absent means not asked.
        if (s.y !== undefined && s.y !== null && Number(s.y) !== STAIR_Y) {
          notes.push(unexpressed('step y', { index: i, field: 'y', asked: Number(s.y),
            got: r4(STAIR_Y), why: 'the block has an x and a z slide and no y joint' }));
        }
        for (const [field, bank] of [['halfDepth', STEP_HALF_DEPTH],
                                     ['halfWidth', STAIR_HALF_WIDTH],
                                     ['halfHeight', STEP_HALF_HEIGHT]]) {
          if (s[field] !== undefined && s[field] !== null && Number(s[field]) !== bank) {
            notes.push(unexpressed('step size', { index: i, field, asked: Number(s[field]),
              got: bank, why: 'the bank is fourteen identical blocks; a step is one of them moved' }));
          }
        }
      });
      // A run longer than a block is deep leaves daylight between treads. It is
      // laid anyway — it is a legal world — but it is not the solid flight the
      // stairs challenge means by a staircase.
      for (let i = 1; i < steps.length; i++) {
        const gap = Math.abs(steps[i].x - steps[i - 1].x) - STEP_HALF_DEPTH * 2;
        if (gap > 1e-9) {
          // `asked` is the spacing that was laid, `got` the DAYLIGHT it leaves —
          // the blocks are where they were asked, so the number that is not
          // what was wanted is the gap, not a substituted run.
          notes.push(unexpressed('gap between steps', { index: i, field: 'x',
            asked: r4(Math.abs(steps[i].x - steps[i - 1].x)), got: r4(gap),
            why: `blocks ${STEP_HALF_DEPTH * 2} m deep laid ${r4(Math.abs(steps[i].x - steps[i - 1].x))} m `
               + `apart leave ${r4(gap)} m of daylight: this is not a solid flight, and a foot `
               + 'that lands in the gap falls to the floor' }));
        }
      }
    } else if (against) {
      steps = against.steps;                    // not mentioned: leave them where they are
    } else {
      // THE ONE REFUSAL THAT IS ABOUT A DEFAULT AND NOT ABOUT A NUMBER. Nothing
      // parks this bank in the live world: it boots with fourteen 200 kg blocks
      // stacked at (0, STAIR_Y, 0), colliding on every tick, and that is the
      // world every published live number was measured in. A first world that
      // said nothing about the bank would have to either park it (a silent
      // change to the plant under a duck somebody is steering) or freeze it
      // where it is (a silent pin). Both are decisions, so the caller makes
      // one.
      throw new Error('say what the step bank should do: `steps: [{x, top}]` to lay them, '
        + `\`steps: []\` or \`clear: true\` for a bare floor. This bench boots with all `
        + `${STAIR_COUNT} blocks stacked at (0, ${r4(STAIR_Y)}, 0) — nothing parks them — `
        + 'so a bare floor is a change to the world every live number here was measured in, '
        + 'not a default');
    }

    // --- the ball ----------------------------------------------------------
    let ball = against ? against.ball : null;
    if (body.ball === null) {
      const at = ballAt ? ballAt() : null;
      notes.push(unexpressed('remove the ball', { field: 'ball', asked: null,
        got: at ? { x: at[0], y: at[1], z: at[2] } : null,
        why: 'the ball is a permanent body in this plant, not something a world adds; '
           + 'it can be moved and it cannot be taken out' }));
    } else if (body.ball !== undefined) {
      if (!BALL) throw new Error('this world has no ball');
      const x = Number(body.ball.x), y = Number(body.ball.y);
      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        throw new Error('ball: give {x, y} as finite numbers');
      }
      if ((inX !== null && Math.abs(x) + BALL_RADIUS > inX + 1e-12)
       || (inY !== null && Math.abs(y) + BALL_RADIUS > inY + 1e-12)) {
        throw new Error(`a ball at (${x}, ${y}) is outside this arena: the inner faces are at `
                      + `x = ±${inX} and y = ±${inY}, and the ball is ${BALL_RADIUS} m in radius`);
      }
      ball = { x, y };
    }

    // --- props -------------------------------------------------------------
    let props = against ? against.props : [];
    if (body.props !== undefined && body.props !== null) {
      if (!Array.isArray(body.props)) throw new Error('props must be an array of {name, x, y}');
      props = [];
      body.props.forEach((p, i) => {
        const g = GRASPABLES.find(q => q.name === p?.name);
        if (!g) {
          notes.push(unexpressed('prop', { index: i, field: 'name', asked: p?.name ?? null,
            got: GRASPABLES.map(q => q.name),
            why: `${PLANT} has no body by that name; a prop is a body this plant compiled, `
               + 'and there is no way to add one at run time' }));
          return;
        }
        const x = Number(p.x), y = Number(p.y);
        if (!Number.isFinite(x) || !Number.isFinite(y)) {
          throw new Error(`prop ${i} (${g.name}): give {x, y} as finite numbers`);
        }
        // NOT BOUNDED BY THE ARENA, AND THAT IS NOT AN OVERSIGHT. A step block
        // is 200 kg and a static wall is immovable, so a block half inside one
        // is a fight the solver has to have every tick; a graspable is 18 to
        // 30 g on a 6 × 6 m floor, and one put down outside the walls simply
        // sits there. The refusals here are about the plant, not about tidiness.
        // Its own compiled drop height, so a cone lands the way the plant drops it.
        props.push({ name: g.name, x, y, z: model.qpos0[g.adr + 2] });
      });
    }

    // --- walls -------------------------------------------------------------
    if (body.walls !== undefined && body.walls !== null && Array.isArray(body.walls)) {
      body.walls.forEach((w, i) => {
        notes.push(unexpressed('wall', { index: i, field: 'walls', asked: w?.name ?? null,
          got: ARENA.walls.map(x => x.name),
          why: 'the four arena walls are static geoms with no joints; this world has exactly '
             + 'those four and cannot be given another' }));
      });
    }

    // --- the spawn ---------------------------------------------------------
    // THE ROW THAT EXISTS BECAUSE THE BANK CANNOT MOVE. A flight can only ever
    // stand on the one compiled row, so a world laid without a spawn puts the
    // steps 1.305 m to the duck's left: a TRUE picture of a USELESS run, which
    // is the original bug wearing a different hat. It is not a refusal — a
    // caller may legitimately want the room and not the walk — so it is said.
    if (spawn === null) {
      // AND THE ROW SAYS WHAT STOOD. A cleared world lays no flight, so "the
      // step bank is to its left" would name a flight that is parked below
      // the floor; the row still exists (nothing said where to stand) but its
      // reason is the true one.
      const why = steps.length
        ? 'nothing asked where the duck should stand, so it is on its compiled mark at '
          + `(0, 0) and the step bank is ${r4(STAIR_Y)} m to its left. A world laid without `
          + 'a spawn is a room the duck is beside, not one it is in.'
        : 'nothing asked where the duck should stand, so it is on its compiled mark at '
          + '(0, 0). No flight was laid for this run: the bank is parked below the floor, '
          + 'so there is nothing beside the duck to stand in.';
      notes.push(unexpressed('spawn', { field: 'spawn', asked: null, got: { x: 0, y: 0 }, why }));
    }

    // --- nothing has been written, here or above ---------------------------
    return {
      name: body.name === undefined ? (against ? against.name : null)
          : (body.name === null ? null : String(body.name)),
      steps, ball, props, notes,
      // Which sections this body actually restated, so the caller that MERGES
      // does not have to re-derive it from the body a second time and get a
      // different answer.
      restated: [
        ...((asked || body.clear === true) ? ['steps'] : []),
        ...(body.ball !== undefined ? ['ball'] : []),
        ...((body.props !== undefined && body.props !== null) ? ['props'] : []),
        ...(Array.isArray(body.walls) ? ['walls'] : []),
      ],
    };
  }

  /**
   * WHICH SECTION A NOTE BELONGS TO, so a restated clause drops its own old
   * notes and nobody else's.
   *
   * `spawn` is a STEPS note: it exists only because a flight was laid where the
   * duck is not, so a post that restates the bank restates it.
   */
  const sectionOf = n => ({ 'step y': 'steps', 'step size': 'steps', 'gap between steps': 'steps',
                            'remove the ball': 'ball', prop: 'props', wall: 'walls',
                            spawn: 'steps' })[n.what] ?? 'other';

  /**
   * THE LIVE LANE'S WORLD, COMMITTED.
   *
   * `planWorld` above did every check and every refusal; this adds the two
   * things that are only true of the live lane — the merge against what is
   * already standing, and the write.
   *
   * MERGE BY SECTION, NEVER REPLACE. A post that restates one part of a
   * standing world carries that part's honesty; the parts it did not mention
   * keep theirs, or a ball move would silently erase what the stairs could not
   * express.
   */
  function setWorld(body) {
    const plan = planWorld(body, { against: WORLD.set ? WORLD : null,
                                   ballAt: () => ballOf(live.world) });
    const restated = new Set(plan.restated);
    const kept = (WORLD.unexpressed || []).filter(n => !restated.has(sectionOf(n)));
    WORLD.steps = plan.steps;
    WORLD.ball = plan.ball;
    WORLD.props = plan.props;
    WORLD.name = plan.name;
    WORLD.unexpressed = kept.concat(plan.notes);
    WORLD.set = true;
    relayWorld();
    if (plan.ball && BALL) placeBall(live.world, plan.ball.x, plan.ball.y, BALL_RADIUS);
    else mj.mj_forward(model, live.world);
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
          // `yawOf` IS THE ONE THIS BENCH PLACES AND SCORES BY. It used to be
          // spelled out here; it now comes from `reward_math.mjs` because
          // `chase_score.mjs` reads the SAME yaw at the same instant as
          // Pollen's frozen `kick_dir` and as the axis `ballTravel_mm` is
          // projected onto. Two spellings of one atan2 would put the ball on
          // one line and score the travel along another. Same arithmetic, same
          // operand order — `bench_parity.mjs` is what says so.
          const yaw = yawOf([d.qpos[f + 3], d.qpos[f + 4], d.qpos[f + 5], d.qpos[f + 6]]);
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
    /*
     * GET /world — what is standing in the live world right now.
     * POST /world — put something else there.
     *
     * THE ONLY ENDPOINT THAT IS ABOUT THE ROOM AND NOT ABOUT A DUCK. Everything
     * else here steers, records or scores; this one moves the fourteen step
     * blocks the plant compiled, the ball and the graspables, so that a caller
     * driving a duck can drive it somewhere other than the room the plant was
     * compiled with.
     *
     * WHAT IT CANNOT DO IS PART OF THE ANSWER. This plant has a fixed bank of
     * blocks at a fixed y, four static walls and five graspables; nothing can be
     * added, nothing can be resized and nothing can be moved out of the room. A
     * request that asks for more is either refused (a step that would sit inside
     * a wall, more steps than there are blocks, a ball outside the arena) or
     * laid as close as the plant allows with the difference named in
     * `unexpressed` — never silently approximated.
     *
     * AND THE ANSWER IS THE READBACK, NOT THE REQUEST. `steps` comes back out of
     * qpos, so it is where the blocks are; an app that draws this draws the
     * world the bench built.
     *
     * IT ONLY TOUCHES THE LIVE WORLD. /record, /measure, /perform, /capture and
     * /tune reset their own mjData and /climb and /chase build theirs, so a
     * world standing here changes nothing any of them answer —
     * `sim/world_parity.mjs` phase 4 is the gate that says so.
     */
    if (url.pathname === '/world') {
      return liveLane(async () => {
        // A WORLD IS NOT ABOUT A DUCK, and `duck` is still refused if it names
        // one this scene does not hold. The addressing convention is the same
        // on every endpoint — a typo that quietly does the right thing on one
        // door and the wrong thing on another is worse than either — and
        // `pickSlot` is the one place that decides it.
        pickSlot(url, body);
        await ensureStanding();
        // A GET arrives here with an empty body in both shells, and reading is
        // the safe reading of "no fields": a caller that meant to change
        // something named the thing it wanted changed. `duck` is addressing and
        // not a field, so it alone is still a read.
        const FIELDS = ['name', 'steps', 'ball', 'props', 'walls', 'clear'];
        if (body && FIELDS.some(k => k in body)) setWorld(body);
        return worldReadback();
      });
    }
    /*
     * GET /lanes — WHO ELSE WAS IN THE BUILDING. DIAGNOSTIC ONLY.
     *
     * Nothing scored is in here and nothing in here is scored: these are four
     * cumulative counters about the SHAPE of the traffic, not about a duck.
     *
     * IT EXISTS SO A CONCURRENCY GATE CAN FAIL ITSELF. `world_parity` phase 4
     * awaits every call before issuing the next, so a collision loan held
     * across an `await` would pass it; phase 5g issues an overlapping pair on
     * purpose, and a race that did not actually race would pass just as
     * quietly. `performTicksWithAnotherRequestInFlight` is how 5g proves the
     * overlap happened: if it did not move, the phase reports that it proved
     * nothing and goes red.
     */
    if (url.pathname === '/lanes') {
      return {
        performTicks: LANES.performTicks,
        performTicksWithAnotherRequestInFlight: LANES.performTicksWithAnotherRequestInFlight,
        liveInflight: LANES.liveInflight,
        climbInflight: LANES.climbInflight,
        why: 'in flight means accepted and not yet answered — queued as much as running. Node '
           + 'is single-threaded and each lane is serial, so a request that arrives during a '
           + '/perform waits behind it; "waited while /perform held the step geoms\' '
           + 'conaffinity" is exactly the situation the loan has to survive.',
      };
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
        // AND THE WORLD IS RE-LAID, BECAUSE `mj_resetData` JUST WIPED IT.
        // `ensureStanding` above resets every slide and every free body to
        // qpos0, which puts the step bank back in the stack it boots in and the
        // props back on their compiled marks. A /reset that lost the world
        // would lose it at exactly the moment it is wanted most — the start of
        // a trial. With no world set this does nothing and the ball goes to the
        // literal (0.8, 0) it has always gone to.
        relayWorld();
        if (BALL) placeBall(live.world, WORLD.ball ? WORLD.ball.x : 0.8,
                                        WORLD.ball ? WORLD.ball.y : 0, BALL_RADIUS);
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

      // ── THE ROOM THE MOTION RUNS IN, AND WHERE THE DUCK STANDS IN IT ──────
      //
      // Both are OPTIONAL and both are ABSENT by default, and that absence is
      // a contract: a request that carries neither gets the answer it has
      // always got, leaf for leaf, with no `stood` key on it. `bench_parity`'s
      // entry #53 is that request.
      //
      // THE SPAWN IS CHECKED FIRST, because a world's spawn row depends on
      // whether one arrived. `spawn.z` is honoured rather than refused: with it
      // defaulting to 0.120 the write below reproduces this endpoint's existing
      // 0.120 → 0.130 sweep byte for byte, so honouring it costs nothing and
      // makes the same motion start at the same height here and on /climb.
      let spawnPoint = null;
      if (body.spawn !== undefined && body.spawn !== null) {
        const s = body.spawn;
        const bad = { error: 'spawn: give x and y as finite numbers, and z as a finite '
                           + 'number if you give one' };
        if (typeof s !== 'object' || Array.isArray(s)) return bad;
        const x = Number(s.x), y = Number(s.y);
        const hasZ = s.z !== undefined && s.z !== null;
        const z = hasZ ? Number(s.z) : 0.120;
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return bad;
        // THE POINT, NOT A FOOTPRINT. A step block is 340 mm deep and 200 kg and
        // is refused for reaching into a wall; a duck is 250 mm and walks, and
        // the only thing worth saying is that it was put down inside the room.
        if ((ARENA.innerX_m !== null && Math.abs(x) > ARENA.innerX_m + 1e-12)
         || (ARENA.innerY_m !== null && Math.abs(y) > ARENA.innerY_m + 1e-12)) {
          return { error: `a spawn at (${x}, ${y}) is outside this arena: the inner faces `
                        + `are at x = ±${ARENA.innerX_m} and y = ±${ARENA.innerY_m}` };
        }
        spawnPoint = { x, y, z };
      }
      // THE SAME VALIDATOR POST /world RUNS, WITH `against: null`. Nothing is
      // inherited: a /perform world that says nothing about the bank hits the
      // same refusal a first /world gets rather than silently borrowing the
      // live lane's flight. `ballAt` is the batch mjData, so `"ball": null`
      // produces the same `remove the ball` note it produces on /world and
      // there is no /perform-only ball refusal.
      let plan = null;
      if (body.world !== undefined && body.world !== null) {
        if (typeof body.world !== 'object' || Array.isArray(body.world)) {
          return { error: 'world must be an object of {name, steps, clear, ball, props, walls}'
                        + ' — the same shape POST /world takes' };
        }
        try {
          plan = planWorld(body.world, { against: null, ballAt: () => ballOf(data),
                                         spawn: spawnPoint });
        } catch (e) { return { error: String(e.message || e) }; }
      }
      // The plan, in the shape `layWorld` and `worldReadback` take. THE MODULE
      // GLOBAL `WORLD` IS NOT WRITTEN, READ OR CONSULTED anywhere on this
      // route: a room sent with a run is that run's room and nobody else's.
      const laid = plan ? { set: true, name: plan.name, steps: plan.steps, ball: plan.ball,
                            props: plan.props, unexpressed: plan.notes } : null;

      const wantStood = !!(laid || spawnPoint);
      let stoodBlock;
      const stood = wantStood ? {
        at: null,
        /** Where the duck was actually put, read out of qpos at the write. */
        placed(d, f) {
          this.at = { x: r4(d.qpos[f]), y: r4(d.qpos[f + 1]), z: r4(d.qpos[f + 2]) };
        },
        /** THE READ POINT: post-placeSteps, pre-stepWorld, inside the loan. */
        lay(d, t) {
          stoodBlock = {
            ...worldReadback(d, laid),
            spawn: this.at,
            spawnWhy: spawnPoint
              ? `the duck's free joint was written to (${spawnPoint.x.toFixed(3)}, `
                + `${spawnPoint.y.toFixed(3)}, ${spawnPoint.z.toFixed(3)}) before the settle, `
                + 'and this rollout\'s drop was added to z as an offset from 0.120 — the same '
                + 'rule climb_score.mjs uses, so the same motion starts at the same height on '
                + 'both routes.'
              : 'nothing moved the duck: it spawned at its compiled qpos0, (0, 0), and the '
                + `step bank is compiled at y = ${r4(STAIR_Y)}, so anything laid on that bank `
                + `stood ${r4(STAIR_Y)} m to the duck's left.`,
            ofRollout: 0,
            atTick: t,
            pinnedEveryTick: !!(laid && STAIR_ADDR),
            sag_mm: null,
            pinnedWhy: (laid && STAIR_ADDR)
              ? 'each step block is a 200 kg body on two frictionless, undamped slides with no '
                + 'gravity compensation, so it is re-written and re-pinned at the top of every '
                + 'control tick, inside the same synchronous collision loan the live lane '
                + 'takes. `steps` is read immediately after that write and before the step, so '
                + 'it is exactly what was laid; `sag_mm` is how far the lowest-sitting block '
                + 'had fallen by the end of that same tick.'
              : 'no world was laid for this run, so nothing was pinned and nothing sagged: '
                + 'the bank is wherever this plant left it, which on the batch lane is '
                + 'fourteen blocks that booted stacked and have been falling since. `steps` '
                + 'is still read before the step of the tick it names.',
          };
        },
        /** The other end of that same tick, and the only thing read after it. */
        sag(d) {
          if (!stoodBlock) return;
          if (!laid || !STAIR_ADDR || !laid.steps.length) return;
          let worst = -Infinity;
          for (let i = 0; i < laid.steps.length && i < STAIR_COUNT; i++) {
            const fell = (laid.steps[i].top
                        - (d.qpos[STAIR_ADDR[i].z] + STEP_HALF_HEIGHT)) * 1000;
            if (fell > worst) worst = fell;
          }
          stoodBlock.sag_mm = Number.isFinite(worst) ? r4(worst) : null;
        },
      } : null;

      let first = null, ok = 0;
      const heights = [];
      for (let i = 0; i < rollouts; i++) {
        // Pollen's own randomisation, as measure_success.mjs uses it: the drop
        // height is what a bench can vary without touching the model.
        const drop = 0.12 + (0.01 * i) / Math.max(rollouts - 1, 1);
        const run = await batchLane(() =>
          rollout({ name, seconds, schedule: body.schedule, track: ordered, blend, drop, duck,
                    // EVERY rollout runs in the world; only rollout 0 is read
                    // back, because `frames`, `roots` and `commands` are its.
                    world: laid, spawn: spawnPoint, stood: i === 0 ? stood : null,
                    lanes: LANES }));
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
        // WHAT WAS ACTUALLY SAMPLED. `drop = 0.12 + (0.01·i)/max(rollouts−1, 1)`,
        // so a single rollout only ever sees 0.120 — and this sentence claimed a
        // sweep across it. It is printed to a person on the challenge screen's
        // quick run, which is exactly where it was false. Nothing about `stood`
        // is appended here: a backtick-quoted JSON key is not English, and this
        // string is read out loud.
        criterion: rollouts === 1
          ? 'stayed upright to the end, dropped from 0.120 m'
          : 'stayed upright to the end, over drop heights 0.120-0.130 m',
        medianHeight: r4(heights[Math.floor(heights.length / 2)]),
        endsUpright: upright(last), endHeight: r4(last[2]),
        peakJointRate: r4(peak),
        // THE ONE NEW KEY, AND IT IS ABSENT UNLESS SOMETHING ASKED. A request
        // that carries neither a world nor a spawn answers exactly what it
        // always answered — which is what keeps parity/core-v5-performfix.json
        // frozen with no --allow.
        ...(wantStood ? { stood: stoodBlock } : {}),
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
      // THE PICTURE IS A REQUEST FIELD, NEVER PART OF THE INTENT. `intent` is
      // the object `intentHashPayload` is taken over, so a render flag inside
      // it would change the identity of every move that asked to be watched.
      // It rides in `runEpisode`'s FIFTH argument — the per-cell plant bag,
      // beside `isolate` and `stepCount` — and never through `optsOf`.
      const wantClip = body.clip === true;
      const E = await climbJob(() => rig.runEpisode(
        intent.keyframes, climbOptsOf(intent), rise + dh, 'policy',
        { drop, fmul, isolate: intentIsolate(intent), stepCount: intentStepCount(intent),
          clip: wantClip }));
      const s = E.afterTail;                        // the grid scores after the 50-tick policy tail
      const crit = climbCriteria(rise + dh, s);
      const honest = crit.honest;
      const stable = honest && E.uprightTailTicks >= UPRIGHT_TAIL_MIN;
      // FULL FLOAT DIGITS, in millimetres. Not rounded: sim/climb_parity.mjs
      // compares these against robust.mjs's own numbers with Object.is, and a
      // toFixed here would make that gate unable to tell a moved trajectory
      // from a rounded one.
      const mm = v => v * 1000;
      // ── THE CLIP AND THE FLIGHT IT RAN ON, both absent unless asked for ────
      //
      // NESTED, NOT MERGED FLAT: the twenty-six scored keys keep their names,
      // omitting a clip is trivial, and world_parity phase 4's /climb diff
      // stays at 28 readable leaves. `undefined` is dropped by JSON.stringify,
      // so a no-clip answer's key set is exactly today's.
      //
      // `stood.steps` COMES OUT OF THE EPISODE and not out of a later read:
      // `runEpisode`'s `finally` parks all fourteen blocks at (i·1.5, −5), so a
      // post-hoc readback would report a parked bank as the flight the duck
      // climbed. `E.stairsInEpisode` was read at the last pin, before that
      // tick's substeps — the same instant /perform's `stood` is read at, which
      // is what makes the two comparable leaf for leaf (world_parity 5d).
      let clipBlock, stoodBlock;
      if (wantClip && E.clip) {
        const lastRoot = E.clip.roots[E.clip.roots.length - 1];
        clipBlock = {
          format: 'duck-intent-clips/3',
          hz: C.tickHz,
          joints: C.jointNames.filter(n => n !== 'mouth'),
          policy: STAND,
          plantName: PLANT, plantDigest: PLANT_DIGEST,
          frames: E.clip.frames, roots: E.clip.roots, commands: E.clip.commands,
          ticks: E.clip.frames.length,
          endsUpright: lastRoot ? upright(lastRoot) : false,
          settleExcluded: true,
          settleWhy: 'the recorded ticks are the track\'s plus the fifty-tick policy tail; the '
                   + 'twenty-five settle ticks are a drop bounce dying under a different policy '
                   + 'and are not in here, which is /perform\'s own convention.',
        };
      }
      if (wantClip && E.stairsInEpisode) {
        const count = intentStepCount(intent);
        // The blocks the episode laid, said the way a world describes one. This
        // is `layoutStairs`' own arithmetic and world_parity phase 1 is the
        // gate that says `placeSteps` writes the identical slots — so the count
        // here is what decides which blocks the readback calls parked.
        const laidHere = Array.from({ length: Math.min(count, STAIR_COUNT) }, (_, i) => ({
          x: CLIMB_STAIR_START + i * CLIMB_STAIR_RUN + STEP_HALF_DEPTH,
          top: (i + 1) * (rise + dh),
        }));
        // NO NAME. Nothing named this room: the climb route lays the harness's
        // own flight from a rise and a cell, and inventing a sentence for it
        // here would be this bench composing prose about a world.
        stoodBlock = worldReadback(CLIMB_DATA,
          { set: true, name: null, steps: laidHere, unexpressed: [] },
          E.stairsInEpisode);
      }
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
        clip: clipBlock,
        stood: stoodBlock,
      };
    }

    /**
     * GET /chase/grid — THE FOURTEEN CELLS OF THE BALL CHALLENGE, AND ITS
     * CRITERION, SO A CLIENT NEVER RETYPES EITHER.
     *
     * The grid is not a preference: it is the axis set every published ball
     * result is decided on. Nine CORE cells — bearing {−20, 0, +20}° crossed
     * with range {0.45, 0.70, 0.95} m on the nominal plant — plus five EXTENDED
     * ones: the centre cell on the slippery and the grippy plants (the same two
     * plant pairs the stairs grid uses, so "the slippery plant" means one
     * thing), a ball well off the heading at ±40°, and a ball straight ahead but
     * far at 1.20 m. A client that hard-codes fourteen numbers is a client that
     * will one day be scoring a different grid and reporting it under the same
     * name, so the bench that runs them is the one that lists them.
     */
    if (url.pathname === '/chase/grid') {
      const rig = await chaseRig();
      return {
        cells: chaseGridCells(),
        // nExt is the FIVE extended cells, the count every leaderboard row's
        // "ext k/5" is over; nAll is every cell the grid runs. They used to
        // share one name with two meanings.
        nCore: CHASE_N_CORE, nExt: CHASE_N_EXT - CHASE_N_CORE, nAll: CHASE_N_EXT,
        criterion: CHASE_CRITERION,
        touchMm: TOUCH_MM, travelMinMm: TRAVEL_MIN_MM,
        tailTicks: CHASE_TAIL_TICKS, uprightTailMin: CHASE_UPRIGHT_TAIL_MIN,
        settleTicks: CHASE_SETTLE_TICKS,
        bearingWhy: 'degrees from the duck\'s settled heading to the ball. POSITIVE IS LEFT — '
                  + 'the convention POST /ball already uses, so a trial reads the same way '
                  + 'duckvision reports. Bearing is the axis that makes this a chase: a ball '
                  + 'dead ahead is reachable by walking forward, and a ball at ±40° is not.',
        rangeWhy: 'metres from the duck\'s root to the ball\'s centre, measured after the '
                + 'settle. Pollen\'s kick task spawns the ball 90 mm in front of the toe '
                + '(cfg 84), so the NEAREST cell here is five times the distance the bundled '
                + 'kick policies were trained at.',
        config: CHASE_CONFIG,
        configSource: CHASE_CONFIG_SOURCE,
        terms: CHASE_TERMS.map(t => ({ term: t.term, weight: t.weight,
                                       weightStage0: t.weightStage0, source: t.source,
                                       formula: t.formula })),
        refused: CHASE_REFUSALS,
        refusedWhy: 'the three terms of the ball-kick config this plant cannot answer, named '
                  + 'with their weights and the reason, in every answer. A shorter refusal list '
                  + 'is not a better one: each of these could be approximated by picking a '
                  + 'threshold or a softening fraction Pollen never wrote down, and each '
                  + 'approximation would be a different term wearing this one\'s name.',
        actionRateWhy: ACTION_RATE_SOURCE_WHY,
        caveat: BALL_CAVEAT,
        entrantKinds: ['move', 'policy'],
        entrantWhy: 'a move is {kind: "move", intent: {keyframes, blend}} and is run under the '
                  + 'bench\'s 25-tick settle exactly as a /climb cell is; a policy is '
                  + '{kind: "policy", policy: <name>, schedule: [[atSeconds, {vx, vy, vyaw}]]} '
                  + 'and is the format that makes "chase" a closed-loop question later — the '
                  + 'same field carries a fixed schedule today and one computed from the '
                  + 'ball\'s bearing tomorrow, with no change to the format or the hash.',
        chaseable: !!rig,
        ...(rig ? {} : { why: CHASE_WHY }),
        policies: [...catalogue().keys()].sort(),
        plantName: PLANT, plantDigest: PLANT_DIGEST,
      };
    }

    /**
     * POST /chase — ONE CELL OF THE GRID, FOR ONE ENTRANT, IN THE REQUEST BODY.
     *
     * WHY ONE CELL AND NOT THE GRID, for the reason /climb takes one: a cell is
     * half a second of settle, up to five seconds of entrant and a second of
     * tail, and fourteen of them is a request that times out somewhere between
     * the app and here with nothing at all to show for it. One cell answers in
     * about a second, the client asks fourteen times, and it can draw a progress
     * row and stop halfway.
     *
     *   body { entrant: { kind: "move",   intent: <harness JSON> }
     *              or   { kind: "policy", policy: <name>, schedule: [...] },
     *          seconds, cell: { bearing, range, drop, fmul }, tail: "policy" }
     *
     * WHAT IT ANSWERS. Eight plain facts, the nine computed terms of Pollen's
     * ball-kick config as {term, weight, value}, the three refused by name, the
     * verdict `chased` and the stricter `stable`, and the plant this was all
     * measured in. The nine terms are REPORTED and are NOT the verdict: a shaped
     * sum of nine weighted terms is not a thing a person can hold in their head,
     * and a leaderboard sorted on it would reward a duck that stands beautifully
     * still.
     *
     * `chase/chase_parity.mjs` is the acceptance test: every entrant file across
     * all fourteen cells, through here and through chase_robust, EXACT at full
     * float digits.
     */
    if (url.pathname === '/chase') {
      const started = (typeof performance === 'object' ? performance.now() : Date.now());
      const rig = await chaseRig();
      if (!rig) {
        return { error: `no /chase here: ${CHASE_WHY}`, chaseable: false, why: CHASE_WHY,
                 plantName: PLANT, plantDigest: PLANT_DIGEST };
      }
      let entrant;
      try { entrant = chaseCheckEntrant(body.entrant, 'the request body'); }
      catch (e) { return { error: String(e.message || e) }; }
      const cellIn = body.cell || {};
      const bearing = cellIn.bearing === undefined ? 0 : +cellIn.bearing;
      const range = cellIn.range === undefined ? 0.70 : +cellIn.range;
      const drop = cellIn.drop === undefined ? 0.120 : +cellIn.drop;
      const fmul = cellIn.fmul === undefined ? 1.0 : +cellIn.fmul;
      if (![bearing, range, drop, fmul].every(Number.isFinite)) {
        return { error: 'chase needs a cell of finite bearing, range, drop and fmul' };
      }
      if (!(range > 0 && range < 10)) return { error: 'chase needs a range in METRES, 0 < range < 10' };
      if (!(drop >= 0.05 && drop <= 0.30)) return { error: 'chase needs a drop in METRES within 0.05..0.30' };
      if (!(fmul > 0 && fmul <= 5)) return { error: 'chase needs a friction multiplier within 0 < fmul <= 5' };
      // THE DRIVEN SPAN IS PART OF THE HASHED ENTRANT. A body `seconds` that
      // disagrees with what the entrant declares would score one thing under
      // another thing's hash, so the disagreement is refused by name.
      if (body.seconds !== undefined && entrant.seconds !== undefined
          && +body.seconds !== +entrant.seconds) {
        return { error: `the body asks for ${+body.seconds} s but the entrant declares `
                      + `${+entrant.seconds} s; the entrant's seconds are part of its hash, so `
                      + 'send one or the other, not two that disagree' };
      }
      const seconds = body.seconds === undefined
        ? (entrant.seconds === undefined ? CHASE_SECONDS : +entrant.seconds)
        : +body.seconds;
      if (!(Number.isFinite(seconds) && seconds > 0 && seconds <= 30)) {
        return { error: 'chase needs a driven span in SECONDS, 0 < seconds <= 30' };
      }
      const tail = body.tail === undefined ? 'policy' : String(body.tail);
      if (tail !== 'policy') {
        return { error: `this bench scores the grid, and the grid is tail "policy": ${tail} is not one of its cells` };
      }
      // THE ENTRANT'S ACTOR. A move rides on the standing policy, which the rig
      // already holds; a policy entrant names a network, and a name this bench
      // has never heard of is refused BY NAME rather than quietly settled for.
      let actor = null;
      if (entrant.kind === 'policy') {
        try {
          const loaded = await policy(entrant.policy);
          actor = { run: obs => loaded.net.run(obs), reference: loaded.reference };
        } catch (e) { return { error: String(e.message || e) }; }
      }
      const hash = await env.sha256(new TextEncoder().encode(chaseHashPayload(entrant)));
      const answered = {
        hash, entrant: hash.slice(0, 12), kind: entrant.kind,
        policy: entrant.kind === 'policy' ? entrant.policy : null,
        cell: { bearing, range, drop, fmul }, tail,
        // SECONDS IS THE EPISODE, NOT THE STOPWATCH — and it differs from
        // /climb, where `seconds` is how long the request took. A chase answer
        // has to carry the driven span, because the span is part of what was
        // scored: the same entrant at 4 s and at 5 s is two measurements. The
        // stopwatch is `elapsedSeconds`.
        seconds,
        plantName: PLANT, plantDigest: PLANT_DIGEST,
        criterion: CHASE_CRITERION,
        config: CHASE_CONFIG,
      };
      let E;
      try {
        E = await chaseJob(() => rig.runEpisode(entrant, { bearing, range, drop, fmul },
                                                { seconds, tail, actor }));
      } catch (e) { return { ...answered, error: String(e?.message || e) }; }
      // FULL FLOAT DIGITS. Not rounded: chase/chase_parity.mjs compares these
      // against chase_robust's own numbers with Object.is, and a toFixed here
      // would make that gate unable to tell a moved trajectory from a rounded
      // one.
      return {
        ...answered,
        ...E.facts,
        chased: E.chased, stable: E.stable,
        uprightTailTicks: E.uprightTailTicks,
        tailTicks: E.tailTicks, drivenTicks: E.drivenTicks, rateTicks: E.rateTicks,
        terms: E.terms,
        termsWhy: 'per-tick MEANS over the DRIVEN SPAN — the entrant\'s own seconds. The 50-tick '
                + 'tail is this bench\'s standing test and not part of the entrant\'s episode, so '
                + 'averaging over it would put the tail length into Pollen\'s reward. The facts '
                + 'above span the whole episode, because the criterion asks about the end of it. '
                + 'These nine are REPORTED and are not the verdict.',
        refused: E.refused,
        actionRateSource: E.actionRateSource,
        actionRateWhy: ACTION_RATE_SOURCE_WHY,
        caveat: BALL_CAVEAT,
        yaw0: E.yaw0, kickDir: E.kickDir, ball0: E.ball0, ballEnd: E.ballEnd,
        kickDirWhy: 'the duck\'s heading at the FIRST DRIVEN TICK, frozen for the episode. It is '
                  + 'Pollen\'s own kick_dir (mdp.py 5700-5702, "Frozen for the episode so the '
                  + 'policy can\'t redefine "forward" by turning after the kick"), it is the '
                  + 'line the ball was placed against, and it is the axis ballTravel_mm projects '
                  + 'onto. One vector, read twice.',
        elapsedSeconds: ((typeof performance === 'object' ? performance.now() : Date.now()) - started) / 1000,
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
