// chase_score.mjs — THE ONE CHASE EPISODE, and the one criterion.
//
// WHY THIS FILE EXISTS AT ALL.
//
// It exists for the reason `climb_score.mjs` exists, written down in that
// file's header: an episode whose output is a trajectory cannot be copied and
// kept equal by being read carefully. A moved line that changes the order two
// numbers are added in changes the eleventh decimal, and by tick 250 the ball
// is somewhere else. So the ball challenge gets ONE episode and ONE criterion,
// here, and `chase/chase_rig.mjs`, `chase/chase_robust.mjs` and
// `duckbench-core.mjs`'s POST /chase all call it. `chase/chase_parity.mjs` is
// what says so: every entrant file across all fourteen cells, scored through
// the bench's own `handle` and through `chase_robust`, compared with
// `Object.is` at full float digits on every numeric field.
//
// WHAT IS PARAMETERISED, AND WHY EACH ONE IS A KNOB RATHER THAN A FORK:
//
//   bearing   degrees from the duck's settled heading to the ball. POSITIVE IS
//             LEFT — the convention `POST /ball` already uses, so a trial reads
//             the same way duckvision reports.
//   range     metres, duck root to ball centre, measured after the settle.
//   drop      spawn height. 0.120 on the nominal plant; the two extended plant
//             cells lift it, exactly as `climb_score.mjs`'s PLANTS do.
//   fmul      multiplier on foot friction[0], read from the model at boot. At
//             1.0 the multiplication is the identity and the model is written
//             back with the number it already had.
//   seconds   the entrant's own driven span. The 50-tick tail comes after it.
//   tail      'policy' — the only tail this grid has. The STANDING policy holds the
//             duck under a neutral command for 50 ticks, track finished — the standing
//             test — which is what makes "is the duck still a robot afterwards" answerable.
//
// THE REWARD IS TRANSCRIBED, NEVER INVENTED. Every weight and every formula in
// TERMS below is read out of Pollen's `microduck_ball_kick_env_cfg.py` at
// commit 1e79c29c97d8b38aee9eefde77a545860ba7658e, with the line reference on
// the term. Three of the config's twelve live terms cannot be computed on this
// plant, and they are REFUSED BY NAME with the reason — never dropped, never
// approximated. `chase/REWARD.md` is the full transcription and is the document
// this file implements.
//
// Contact is `mj_geomDistance`, never `data.contact.get(i)` — that returns an
// embind object that leaks the WASM heap to 2 GB in ~20 s even when `.delete()`
// is called (climb/rig2.mjs).
//
// THE PATHS BELOW ARE THE BUNDLE'S PATHS, exactly as climb_score.mjs's are.
// `./climb_score.mjs` and `./reward_math.mjs` are real files in sim/ and are
// copied flat into both phone bundles beside this one.
import { poseAt, UPRIGHT_TAIL_MIN as CLIMB_UPRIGHT_TAIL_MIN, PLANTS as CLIMB_PLANTS }
  from './climb_score.mjs';
import { gravityXYSquared, rotate, yawOf } from './reward_math.mjs';

// ---------------------------------------------------------------- the lines
//
// THESE FOUR CONSTANTS LIVE ONLY HERE. The kit pins a fallback copy and checks
// it against `GET /chase/grid`; nothing else is allowed a second opinion.

/** `touched` — any duck geometry within this many mm of the ball, at any tick. */
export const TOUCH_MM = 3.0;
/** The ball must finish at least this far along the duck's INITIAL heading. */
export const TRAVEL_MIN_MM = 100.0;
/** The standing tail, in control ticks, after the entrant's driven span. */
export const TAIL_TICKS = 50;
/**
 * `stable` needs the duck upright for this many of the 50 tail ticks.
 *
 * DELIBERATELY THE SAME BAR `climb_score.mjs` USES, and imported from it rather
 * than retyped, so that a person who has read one challenge already knows what
 * this one means and so that the two cannot drift apart.
 */
export const UPRIGHT_TAIL_MIN = CLIMB_UPRIGHT_TAIL_MIN;

/** The settle every entrant opens with, in ticks — the bench's own 25. */
export const SETTLE_TICKS = 25;
/** Pollen's ball in this plant: 100 mm across, 30 g. Its spawn height. */
export const BALL_RADIUS = 0.05;
/** The upright test, shared with the stairs rail: gravity's z in the trunk. */
export const UPRIGHT_GZ = -0.90;

/**
 * THE SENTENCE THE VERDICT IS, said the same way on every bench.
 *
 * It is exported as a string and answered by both `GET /chase/grid` and every
 * `POST /chase`, because a client that prints its own wording is a client that
 * will one day print a criterion this bench does not run.
 */
export const CRITERION_SENTENCE =
  'chased: the duck touched the ball — any duck geometry within 3 mm of it at any tick — '
  + 'and the ball finished at least 100 mm further along the duck\'s initial heading than it '
  + 'started, and the duck was still upright at the end of the episode. '
  + 'stable: chased, and upright for at least 45 of the 50 tail ticks.';

// ---------------------------------------------------------------- the grid

/** Positive is LEFT. Bearing is the axis that makes this a chase. */
export const BEARINGS = [-20, 0, 20];
/** Metres, duck root to ball centre. None of them is reachable without walking. */
export const RANGES = [0.45, 0.70, 0.95];
/** The nominal plant: `climb_score.mjs`'s PLANTS[0], the same numbers. */
export const NOMINAL = { drop: CLIMB_PLANTS[0].drop, fmul: CLIMB_PLANTS[0].fmul };

/**
 * THE FOURTEEN CELLS, IN THE ORDER chase_robust RUNS THEM.
 *
 * Core first — so a partial run is still the core grid — then the five
 * extended. The two extended PLANT pairs are lifted verbatim from
 * `climb_score.mjs`'s PLANTS[1] and PLANTS[2], so that "the slippery plant"
 * means the same thing in both challenges; the other three extended cells move
 * the ball rather than the world.
 *
 * WHY THESE AXES. A ball dead ahead can be reached by a policy that only walks
 * forward; a ball at ±20° cannot, and at ±40° certainly cannot. The grid is
 * built so that the bundled entrants pass some cells and fail others BY
 * CONSTRUCTION — the off-bearing cells are what the challenge is actually
 * about, and a leaderboard that only ever ran bearing 0 would look solved while
 * nothing could chase anything.
 *
 * WHY THESE RANGES. Pollen's kick task spawns the ball 90 mm in front of the
 * toe (cfg 84, `BALL_OFFSET_X = 0.09`), so the NEAREST cell here is five times
 * the distance the bundled kick policies were trained at. That gap is the
 * finding this challenge exists to expose, not a mistake in the grid.
 */
export function gridCells({ core = false } = {}) {
  const plan = [];
  for (const range of RANGES) {
    for (const bearing of BEARINGS) {
      plan.push({ bearing, range, drop: NOMINAL.drop, fmul: NOMINAL.fmul, tier: 'core' });
    }
  }
  if (!core) {
    // the centre cell on a slippery, higher-spawning plant, and on a grippy one
    plan.push({ bearing: 0, range: 0.70, drop: CLIMB_PLANTS[1].drop, fmul: CLIMB_PLANTS[1].fmul, tier: 'ext' });
    plan.push({ bearing: 0, range: 0.70, drop: CLIMB_PLANTS[2].drop, fmul: CLIMB_PLANTS[2].fmul, tier: 'ext' });
    // a ball well off the heading, to the right and to the left
    plan.push({ bearing: -40, range: 0.70, drop: NOMINAL.drop, fmul: NOMINAL.fmul, tier: 'ext' });
    plan.push({ bearing: 40, range: 0.70, drop: NOMINAL.drop, fmul: NOMINAL.fmul, tier: 'ext' });
    // straight ahead but far: a walk, not a lunge
    plan.push({ bearing: 0, range: 1.20, drop: NOMINAL.drop, fmul: NOMINAL.fmul, tier: 'ext' });
  }
  return plan;
}
export const N_CORE = 9;
export const N_EXT = 14;

// --------------------------------------------------------------- the reward
//
// TWELVE LIVE TERMS. Nine computable here, three refused. The config is built
// by inheriting mjlab v1.3.0's velocity template (velocity_env_cfg.py 275-373),
// DELETING eight terms (cfg 210-221) and ADDING seven (cfg 238-310). Of the six
// terms `duckbench-core.mjs`'s `rewardSums` computes for the VELOCITY config,
// only three survive this one: `track_linear_velocity`, `track_angular_velocity`
// and `pose` are deleted by cfg lines 211, 212 and 217 and MUST NOT appear in a
// /chase answer. Reporting them would be answering the wrong config under the
// right name.

/** cfg 95. The code's value, not the stale comment's 0.25 — see REWARD.md §2.3. */
export const BALL_TARGET_SPEED = 1.0;
/** mdp.py 5787-5806, the default `max_penalty`, not overridden by the config. */
export const OVERSHOOT_MAX_PENALTY = 5.0;
/** cfg 285-287 / mjlab velocity_env_cfg.py 286-293: std = sqrt(0.05). */
export const UPRIGHT_STD2 = 0.05;
/** cfg 100. Ten leg joints under this bench's 14-slot ordering. */
export const LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13];
export const LEG_STD = 0.5;                       // cfg 262-270
/** cfg 101. The four neck/head joints. */
export const NECK_JOINTS = [5, 6, 7, 8];
export const NECK_STD = 0.3;                      // cfg 274-282
/** cfg 98 / cfg 290-298: STAND_Z and the height tolerance. */
export const STAND_Z = 0.115;
export const HEIGHT_STD = 0.04;

/**
 * THE FOURTEEN JOINT SLOTS, ASSERTED RATHER THAN ASSUMED.
 *
 * duckkit's `jointNames` is fifteen long with `mouth` at index 9; every intent
 * pose and every policy action is fourteen with the mouth dropped. Under THAT
 * ordering — and only under it — `_LEG_JOINTS = [0,1,2,3,4,9,10,11,12,13]` is
 * exactly the ten leg joints and `_NECK_JOINTS = [5,6,7,8]` exactly the four
 * neck and head joints. Two index lists that agree today is precisely the kind
 * of agreement that stops holding in silence, so this is checked at boot and
 * throws by name if it ever stops.
 */
export const JOINT_ORDER = [
  'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle',
  'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll',
  'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle',
];

/** Throws with the mismatch named. `names` is duckkit's fifteen. */
export function assertJointOrder(names) {
  const driven = names.filter(n => n !== 'mouth');
  if (driven.length !== 14) {
    throw new Error(`chase_score: duckkit drives ${driven.length} joints, not 14`);
  }
  for (let k = 0; k < 14; k++) {
    if (driven[k] !== JOINT_ORDER[k]) {
      throw new Error('chase_score: joint slot ' + k + ' is ' + driven[k] + ', not ' + JOINT_ORDER[k]
        + ' — pose_stand_legs and pose_stand_neck read Pollen\'s _LEG_JOINTS and _NECK_JOINTS '
        + 'by index, and this ordering is what makes those index lists mean what the config '
        + 'means by them');
    }
  }
  return driven;
}

/**
 * The nine terms this plant CAN compute, in the config's own order, each with
 * the weight the trained policy lived under and the lines it was read from.
 *
 * `action_rate_l2` is scored at the ramp END, −1.0: the config starts it at
 * −0.1 (cfg 301) and the curriculum ramps it to −1.0 by iteration 1500 (cfg
 * 561-574), and the trained policy lived under the final value — the same
 * choice `RunMetrics.Task.actionRateWeight` already makes for `.ballKick`. The
 * stage-0 weight is published beside it so nobody has to guess which was used.
 */
export const TERMS = [
  { term: 'ball_forward_velocity', weight: 12.0,
    source: 'cfg 238-242 (weight, params); mdp.py 5761-5784 (function); cfg 95 (BALL_TARGET_SPEED = 1.0)',
    formula: 'clamp((ball world linear velocity xy · kick_dir), 0.0, 1.0) — one-sided: '
           + 'backward and lateral ball motion earn 0, not a penalty' },
  { term: 'ball_speed_overshoot', weight: -4.0,
    source: 'cfg 243-247; mdp.py 5787-5806',
    formula: 'clamp(fwd − 1.0, 0.0, 5.0), where fwd is the UNCLAMPED projection' },
  { term: 'upright', weight: 2.0,
    source: 'cfg 285-287 (re-weight); mjlab v1.3.0 velocity/mdp/rewards.py 67-110; velocity_env_cfg.py 286-293',
    formula: 'exp(−‖projected gravity xy‖² / 0.05) on trunk_base' },
  { term: 'pose_stand_legs', weight: 2.0,
    source: 'cfg 262-270, cfg 100 (_LEG_JOINTS); mdp.py 2396-2418 (pose_target_match)',
    formula: 'mean over the ten leg joints of exp(−((q − HOME)/0.5)²). NOT rewardSums\' '
           + '`pose`: that is mjlab\'s variable_posture, which this config deletes (cfg 217)' },
  { term: 'pose_stand_neck', weight: 1.0,
    source: 'cfg 274-282, cfg 101 (_NECK_JOINTS); mdp.py 2396-2418',
    formula: 'mean over the four neck/head joints of exp(−((q − HOME)/0.3)²)' },
  { term: 'height_stand', weight: 1.0,
    source: 'cfg 290-298; cfg 98 (STAND_Z = 0.115); mdp.py 2440-2451',
    formula: 'exp(−((z − 0.115)/0.04)²) on trunk_base, z above the terrain origin' },
  { term: 'body_ang_vel', weight: -0.05,
    source: 'cfg 302-303; mjlab v1.3.0 velocity/mdp/rewards.py 184-193',
    formula: 'ωx² + ωy² of trunk_base in the WORLD frame; z is deliberately unpenalised' },
  { term: 'action_rate_l2', weight: -1.0, weightStage0: -0.1,
    source: 'cfg 301 (stage-0 −0.1); cfg 561-574 (curriculum to −1.0 by iter 1500); '
          + 'mjlab v1.3.0 envs/mdp/rewards.py 58-65',
    formula: 'Σ(aₜ − aₜ₋₁)² over the fourteen raw actions, before scale and offset' },
  { term: 'angular_momentum', weight: -0.02,
    source: 'cfg 304; mjlab v1.3.0 velocity/mdp/rewards.py 196-206; velocity_env_cfg.py 312-315 '
          + '(sensor_name "robot/root_angmom")',
    formula: 'Σ(angmom²) — the squared magnitude of the subtree angular-momentum sensor' },
];

/**
 * WHAT action_rate_l2 MEANS FOR EACH KIND OF ENTRANT, said in the answer.
 *
 * For a POLICY, `aₜ` is the network's raw fourteen outputs — exactly mjlab's
 * `action_manager.action`. For a MOVE there is no network; the thing the move
 * emits each tick is the interpolated, clamped pose target, and that is its
 * `aₜ`. Same formula, comparable WITHIN a kind and not across it. Labelled
 * rather than refused, because the move genuinely has an action.
 */
export const ACTION_RATE_SOURCE = {
  policy: 'policy raw output',
  move: 'keyframe pose target',
};
export const ACTION_RATE_SOURCE_WHY =
  'action_rate_l2 differences consecutive ACTIONS. A policy entrant\'s action is the network\'s '
  + 'raw fourteen outputs (mjlab\'s action_manager.action); a move entrant has no network, and '
  + 'its action is the interpolated, clamped pose target the keyframes emit. The formula is the '
  + 'same and the number is comparable between moves and between policies, but not across the '
  + 'two kinds.';

/**
 * The three terms of the ball-kick config this PLANT cannot answer, by name,
 * with the weight and the reason — asked for, and REFUSED rather than dropped.
 *
 * A SHORTER LIST IS NOT A BETTER ONE. Each of these could be approximated by
 * picking a threshold or a softening fraction Pollen never wrote down, and each
 * approximation would be a different term wearing this one's name.
 */
export const CHASE_REFUSALS = [
  { term: 'support_foot_grounded', weight: 2.0,
    reason: 'reads a CONTACT SENSOR: cfg 160-171 builds support_foot_ground_contact, a '
          + 'ContactSensorCfg with primary geom left_foot_collision and secondary body terrain, '
          + 'reduced to a `found` flag, and single_foot_grounded_reward (mdp.py 5809-5824) '
          + 'returns clamp(found, 0, 1). This plant has six sensors and none is a contact '
          + 'sensor. mj_geomDistance could report how far a foot geom is from the floor, but '
          + 'that is a DISTANCE, and turning it into the config\'s binary `found` requires '
          + 'choosing a threshold Pollen never wrote — that would be inventing a reward' },
  { term: 'self_collisions', weight: -1.0,
    reason: 'reads the self_collision sensor (cfg 173-180), a subtree-vs-subtree ContactSensor '
          + 'on trunk_base; self_collision_cost (mjlab v1.3.0 velocity/mdp/rewards.py 162-181) '
          + 'counts its found slots or thresholds a force history at 10 N. No collision sensor '
          + 'in this plant, and no contact forces at all' },
  { term: 'dof_pos_limits', weight: -1.0,
    reason: 'mjlab velocity_env_cfg.py 317, NOT deleted by the kick config. joint_pos_limits '
          + '(mjlab envs/mdp/rewards.py 81-96) scores travel outside soft_joint_pos_limits — a '
          + 'configured fraction of the model\'s hard range. Neither duckkit nor this bench '
          + 'ships that fraction; scoring against the hard rangeLo/rangeHi would be a different '
          + 'term wearing this one\'s name, and picking a factor would be inventing it' },
];

/**
 * THE CAVEAT THAT SHADOWS EVERY ABSOLUTE NUMBER, carried in every grid answer.
 */
export const BALL_CAVEAT =
  'Pollen\'s ball is not this ball. The ball-kick config trains against a 70 mm-diameter, 15 g '
  + 'ball (cfg 76-77, BALL_RADIUS 0.035); this plant\'s ball is 100 mm across and 30 g — 1.43× '
  + 'the radius and 2× the mass. Every term is computed with the same formula and the same '
  + 'weight, but a speed target in m/s was tuned against a ball with half the inertia, so the '
  + 'two ball terms here are the config\'s function evaluated on a DIFFERENT ball, not a '
  + 'reproduction of Pollen\'s training signal. That is why every row carries plantName and '
  + 'plantDigest.';

/** The config this reward is, named the way /tune names its own. */
export const CHASE_CONFIG = 'microduck_ball_kick_env_cfg';
export const CHASE_CONFIG_SOURCE =
  'pollen-robotics/microduck_rl, src/mjlab_microduck/tasks/microduck_ball_kick_env_cfg.py, '
  + 'branch main, commit 1e79c29c97d8b38aee9eefde77a545860ba7658e, 661 lines, Apache-2.0. '
  + 'Five of its twelve live terms are defined upstream in mjlab v1.3.0 (pinned by that '
  + 'commit\'s pyproject.toml), so every mjlab line number cited is a v1.3.0 line number.';

// ------------------------------------------------------------- the entrant

/** A move rides on the standing policy; a policy entrant rides on itself. */
export const ENTRANT_KINDS = ['move', 'policy'];

/** Default driven span when neither the request nor the file says. */
export const DEFAULT_SECONDS = 5;

/**
 * The shape every scorer insists on. Throws with the path in the message.
 *
 * It checks the SHAPE and nothing else: an entrant that names a policy this
 * bench has never heard of is a valid entrant and an unknown policy, and the
 * two failures read differently to whoever sent it.
 */
export function checkEntrant(e, where = 'entrant') {
  if (!e || typeof e !== 'object' || Array.isArray(e)) throw new Error('no entrant in ' + where);
  if (!ENTRANT_KINDS.includes(e.kind)) {
    throw new Error(`entrant kind in ${where} must be "move" or "policy", not ${JSON.stringify(e.kind)}`);
  }
  if (e.seconds !== undefined && !(Number.isFinite(+e.seconds) && +e.seconds > 0 && +e.seconds <= 30)) {
    throw new Error(`seconds in ${where} must be a number of seconds, 0 < seconds <= 30`);
  }
  if (e.kind === 'move') {
    const j = e.intent;
    if (!j || typeof j !== 'object') throw new Error('a move entrant needs an intent in ' + where);
    if (!Array.isArray(j.keyframes) || !j.keyframes.length) throw new Error('no keyframes in ' + where);
    for (const f of j.keyframes) {
      if (!Array.isArray(f.pose) || f.pose.length !== 14) throw new Error('bad pose in ' + where);
      if (!Number.isFinite(+f.t)) throw new Error('a keyframe with no t in ' + where);
    }
    if (j.blend !== undefined && !Number.isFinite(+j.blend)) throw new Error('bad blend in ' + where);
  } else {
    if (typeof e.policy !== 'string' || !e.policy) {
      throw new Error('a policy entrant needs a policy name in ' + where);
    }
  }
  checkSchedule(e.schedule, where);
  return e;
}

/** `[[atSeconds, {vx, vy, vyaw}], ...]` — the core's own `commandAt` contract. */
export function checkSchedule(schedule, where = 'entrant') {
  if (schedule === undefined || schedule === null) return [];
  if (!Array.isArray(schedule)) throw new Error('schedule in ' + where + ' must be a list of [at, {vx, vy, vyaw}]');
  for (const row of schedule) {
    if (!Array.isArray(row) || row.length !== 2 || !Number.isFinite(+row[0])
        || !row[1] || typeof row[1] !== 'object') {
      throw new Error('every schedule entry in ' + where + ' is [atSeconds, {vx, vy, vyaw}]');
    }
    for (const k of ['vx', 'vy', 'vyaw']) {
      if (row[1][k] !== undefined && !Number.isFinite(+row[1][k])) {
        throw new Error(`${k} in the schedule of ${where} must be a finite number`);
      }
    }
  }
  return schedule;
}

/** The last entry that has begun wins — `duckbench-core.mjs`'s own contract. */
export function commandAt(schedule, secs) {
  let current = {};
  for (const [at, values] of schedule || []) if (secs >= at) current = values;
  return current;
}

/**
 * THE ENTRANT'S IDENTITY, as the STRING that gets hashed.
 *
 * sha256 over this string. It is a string and not a digest because the two
 * shells hash differently — node:crypto here, the asynchronous `crypto.subtle`
 * in a browser — and the one thing that must not differ between them is what
 * goes in.
 *
 * EVERY KEY IS HASHED EXCEPT `name` AND `note`. Unknown keys are preserved and
 * hashed rather than stripped, so an entrant file that also carries stairs
 * fields is a DIFFERENT entrant and not a silently equivalent one; `name` and
 * `note` are excluded because renaming a move or rewording its note is not a
 * different move, and the app's edit-score-keep loop would otherwise report a
 * new entrant every time somebody fixed a typo. Objects are serialised with
 * their keys sorted, at every depth, so that a file whose keys were written in
 * a different order hashes to the same value.
 */
export function entrantHashPayload(e) {
  const kept = {};
  for (const k of Object.keys(e)) {
    if (k === 'name' || k === 'note') continue;
    if (e[k] === undefined) continue;
    kept[k] = e[k];
  }
  return stableString(kept);
}

/** JSON with object keys sorted at every depth. Arrays keep their order. */
export function stableString(v) {
  if (Array.isArray(v)) return '[' + v.map(stableString).join(',') + ']';
  if (v && typeof v === 'object') {
    return '{' + Object.keys(v).sort()
      .filter(k => v[k] !== undefined)
      .map(k => JSON.stringify(k) + ':' + stableString(v[k])).join(',') + '}';
  }
  return JSON.stringify(v === undefined ? null : v);
}

// ------------------------------------------------------------ the criterion

/** THE VERDICT, out of the eight plain facts and nothing else. */
export function verdict(facts) {
  const chased = facts.touched
              && facts.ballTravel_mm >= TRAVEL_MIN_MM
              && facts.upright;
  return { chased, stable: chased && facts.uprightTailTicks >= UPRIGHT_TAIL_MIN };
}

// ------------------------------------------------------------------- the rig

/**
 * A chase rig over one MuJoCo model.
 *
 * ctx = {
 *   mj, model, data       the module, the plant, and a scratch mjData THIS RIG
 *                         OWNS. It is reset at the top of every episode, so it
 *                         must not be a world anything else is keeping. Because
 *                         it is its own mjData, the ball and everything else a
 *                         cell lays out are gone the moment the episode ends;
 *                         the ONLY thing an episode borrows from the shared
 *                         model is the feet's friction, and that is written
 *                         back in a `finally`.
 *   D                     duckloop's findDuckJoints(model)
 *   HOME, LO, HI          duckkit-constants, through duckloop's makeLoop
 *   jointNames            duckkit's fifteen, for the boot assertion
 *   buildObs, projectedGravity, command    the same, verbatim
 *   stand                 { run, reference } — the standing policy. It settles
 *                         every episode and it is also the actor a MOVE rides
 *                         on, exactly as a /climb cell rides on it.
 *   tickHz                C.tickHz
 * }
 *
 * Returns null when this plant has no ball, so a caller can say so rather than
 * score an empty floor and call it a chase.
 */
export function makeChaseRig(ctx) {
  const { mj, model, data, D, HOME, LO, HI, jointNames,
          buildObs, projectedGravity, command, stand, tickHz } = ctx;

  // THE JOINT ORDERING, ASSERTED AT BOOT. `_LEG_JOINTS` and `_NECK_JOINTS` are
  // index lists, and an index list is a claim about an ordering.
  assertJointOrder(jointNames);

  const DT = 1 / tickHz;

  // THE BALL, FOUND BY WALKING THE MODEL. A free joint on a body called `ball`
  // that is not the duck's own root. No ball, no chase, and the caller says so.
  const BALL = (() => {
    for (let j = 0; j < model.njnt; j++) {
      if (model.jnt_type[j] !== 0) continue;                 // mjJNT_FREE
      const adr = model.jnt_qposadr[j];
      if (adr === D.freeQpos) continue;                      // the duck
      if (model.body(model.jnt_bodyid[j]).name === 'ball') {
        return { adr, dof: model.jnt_dofadr[j], body: model.jnt_bodyid[j] };
      }
    }
    return null;
  })();
  if (!BALL) return null;
  let BALLG = -1;
  for (let g = 0; g < model.ngeom; g++) if (model.geom_bodyid[g] === BALL.body) { BALLG = g; break; }
  if (BALLG < 0) return null;

  // THE ANGULAR-MOMENTUM SENSOR, CHECKED RATHER THAN TRUSTED. mjlab's
  // `angular_momentum` names the sensor "robot/root_angmom" and means a SUBTREE
  // angular momentum rooted on trunk_base (mjSENS_SUBTREEANGMOM, type 37). If
  // the sensor at that name is anything else, summing three numbers off its
  // address would be reporting some other quantity under Pollen's term name, so
  // the term is refused by name instead.
  const ANGMOM = (() => {
    for (let i = 0; i < model.nsensor; i++) {
      if (model.sensor(i).name !== 'root_angmom') continue;
      const type = model.sensor_type[i], objtype = model.sensor_objtype[i];
      const objname = objtype === 1 ? model.body(model.sensor_objid[i]).name : null;
      if (type !== 37 || objtype !== 1 || objname !== 'trunk_base' || model.sensor_dim[i] !== 3) {
        return { adr: -1, why: `this plant's root_angmom sensor is type ${type} on `
          + `${objname ?? 'objtype ' + objtype} with dim ${model.sensor_dim[i]}, and mjlab's `
          + 'angular_momentum reads mjSENS_SUBTREEANGMOM (type 37) on body trunk_base with dim 3' };
      }
      return { adr: model.sensor_adr[i], why: null };
    }
    return { adr: -1, why: 'this plant has no root_angmom sensor, and mjlab\'s angular_momentum '
                         + 'reads one' };
  })();

  // THE GYRO THIS RIG READS, looked up once, here, because every caller found it
  // the same way and one of them finding it differently would feed the policy
  // some other sensor's three numbers as its angular velocity.
  let GYRO = -1;
  for (let i = 0; i < model.nsensor; i++) if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;
  if (GYRO < 0) throw new Error('this plant has no imu_ang_vel sensor: a policy cannot be run on it');

  // THE DUCK'S OWN COLLIDABLE GEOMS — the set `touched` is decided over.
  // Construction is climb_score.mjs's, verbatim, so "any duck geometry" means
  // the same set of geoms in both challenges.
  let DUCKROOT = -1;
  for (let j = 0; j < model.njnt; j++) {
    if (model.jnt_type[j] === 0 && model.jnt_qposadr[j] === D.freeQpos) { DUCKROOT = model.jnt_bodyid[j]; break; }
  }
  const underDuck = b => { let c = b; for (let i = 0; i < 64 && c > 0; i++) { if (c === DUCKROOT) return true; c = model.body_parentid[c]; } return c === DUCKROOT; };
  const DUCKG = [];
  for (let g = 0; g < model.ngeom; g++) {
    if (model.geom_contype[g] === 0 && model.geom_conaffinity[g] === 0) continue;
    if (DUCKROOT >= 0 && underDuck(model.geom_bodyid[g])) DUCKG.push(g);
  }
  if (!DUCKG.length) throw new Error('this plant has no collidable duck geoms: `touched` could never fire');

  /** The feet, for the friction knob — climb_score.mjs's own regex. */
  const FEET = [];
  for (let g = 0; g < model.ngeom; g++) if (/foot_collision|sole/.test(model.geom(g).name || '')) FEET.push(g);
  // The baseline is the PLANT'S, handed in by the shell that read it at boot
  // (duckbench-core.mjs GEOM_FRICTION0); a rig built lazily while the other
  // rig's cell had a multiplier applied would otherwise read that as normal.
  const FRICT0 = FEET.map(g => (ctx.geomFriction0 ? ctx.geomFriction0[g * 3] : model.geom_friction[g * 3]));

  const RBOUND = model.geom_rbound;
  // FIVE METRES, NOT climb's FIVE CENTIMETRES. mj_geomDistance returns `distmax`
  // when the true distance exceeds it, and `closest_mm` is a REPORTED FACT here
  // — the do-nothing control's answer is ~400 mm and a cap at 50 would flatten
  // every honest miss into one number.
  const DISTMAX = 5.0;
  const dist = (a, b) => mj.mj_geomDistance(model, data, a, b, DISTMAX, null);

  // THE PLANT'S OWN SUBSTEPS PER CONTROL TICK, read off the model rather than
  // written as a 4. Same derivation duckbench-core.mjs does at boot.
  const TIMESTEP = (() => {
    mj.mj_resetData(model, data);
    mj.mj_step(model, data);
    const t = data.time;
    mj.mj_resetData(model, data);
    return t;
  })();
  const SUBSTEPS = Math.round(1 / tickHz / TIMESTEP);
  if (!(SUBSTEPS >= 1) || Math.abs(SUBSTEPS * TIMESTEP - DT) > 1e-9) {
    throw new Error(`${tickHz} Hz control does not divide a ${TIMESTEP} s timestep`);
  }

  const quat = () => [data.qpos[D.freeQpos + 3], data.qpos[D.freeQpos + 4],
                      data.qpos[D.freeQpos + 5], data.qpos[D.freeQpos + 6]];
  const isUp = () => projectedGravity(quat())[2] < UPRIGHT_GZ;

  /**
   * The smallest distance from ANY duck geom to the ball, NOW.
   *
   * EXACT, NOT SAMPLED: the bounding-sphere separation is a LOWER BOUND on
   * mj_geomDistance, so a pair skipped because its lower bound already exceeds
   * the running best could not have improved the best. The same argument
   * climb_score.mjs's penetration tracker makes.
   */
  function ballDistanceNow(best) {
    const bx = data.geom_xpos[BALLG * 3], by = data.geom_xpos[BALLG * 3 + 1], bz = data.geom_xpos[BALLG * 3 + 2];
    let out = best;
    for (const g of DUCKG) {
      const dx = data.geom_xpos[g * 3] - bx, dy = data.geom_xpos[g * 3 + 1] - by,
            dz = data.geom_xpos[g * 3 + 2] - bz;
      const lb = Math.sqrt(dx * dx + dy * dy + dz * dz) - RBOUND[g] - RBOUND[BALLG];
      if (lb >= out) continue;
      const d = dist(g, BALLG);
      if (d < out) out = d;
    }
    return out;
  }

  /**
   * ONE EPISODE, ONE CELL, ONE ENTRANT.
   *
   * `actor` is the network that DRIVES the entrant: the named policy for a
   * policy entrant, and the standing policy for a move (a move rides on top of
   * it, exactly as a /climb cell does). The settle is always under the standing
   * policy, and `lastAction` carries across into the driven span, which is what
   * `duckbench-core.mjs`'s own `rollout` does.
   */
  async function runEpisode(entrant, cell, { seconds, tail = 'policy', actor = null } = {}) {
    const { drop = NOMINAL.drop, fmul = NOMINAL.fmul } = cell;
    FEET.forEach((g, i) => { model.geom_friction[g * 3] = FRICT0[i] * fmul; });
    try {
      return await episode(entrant, cell, seconds, tail, actor, drop, fmul);
    } finally {
      // THE WORLD IS HANDED BACK EXACTLY AS IT WAS LENT. The bench answers
      // /perform, /record and /intent out of the same model; a friction
      // multiplier left behind would move a number nobody asked about.
      FEET.forEach((g, i) => { model.geom_friction[g * 3] = FRICT0[i]; });
    }
  }

  async function episode(entrant, cell, seconds, tail, actor, drop, fmul) {
    const isMove = entrant.kind === 'move';
    const acting = isMove ? stand : actor;
    if (!acting || typeof acting.run !== 'function') {
      throw new Error('a policy entrant needs an actor: { run, reference }');
    }
    const actorRef = acting.reference ?? HOME;
    const schedule = entrant.schedule || [];
    const track = isMove ? entrant.intent.keyframes.map(f => ({ t: +f.t, pose: f.pose.slice() })) : null;
    const blend = isMove ? (entrant.intent.blend === undefined ? 1 : +entrant.intent.blend) : 1;

    // ---- the world, at the start
    mj.mj_resetData(model, data);
    const f = D.freeQpos;
    data.qpos[f] = model.qpos0[f];
    data.qpos[f + 1] = model.qpos0[f + 1];
    data.qpos[f + 2] = drop;
    data.qpos[f + 3] = 1; data.qpos[f + 4] = 0; data.qpos[f + 5] = 0; data.qpos[f + 6] = 0;
    for (let k = 0; k < 6; k++) data.qvel[D.freeDof + k] = 0;
    for (let k = 0; k < 14; k++) {
      data.qpos[D.qpos[k]] = HOME[k];
      data.qvel[D.dof[k]] = 0;
      data.ctrl[k] = HOME[k];
    }
    mj.mj_forward(model, data);

    let la = new Array(14).fill(0);
    const neutral = command({});

    /** One control tick: observe, run the network, ride the offsets, step. */
    const step = async (net, reference, offsets, cmd) => {
      const q = quat(); const jp = [], jv = [];
      for (let k = 0; k < 14; k++) { jp.push(data.qpos[D.qpos[k]]); jv.push(data.qvel[D.dof[k]]); }
      const obs = buildObs([data.sensordata[GYRO], data.sensordata[GYRO + 1], data.sensordata[GYRO + 2]],
                           projectedGravity(q), jp, jv, la, cmd, reference);
      la = Array.from(await net.run(obs));
      for (let k = 0; k < 14; k++) {
        // AUTHORED OFFSETS RIDE ON TOP OF THE POLICY, exactly as /climb applies
        // them: the policy keeps its balance and the track leans on it. The
        // OBSERVATION is untouched.
        const v = reference[k] + la[k] + (offsets ? (offsets[k] - HOME[k]) * blend : 0);
        const c = Math.min(Math.max(v, LO[k]), HI[k]);
        data.ctrl[k] = c;
      }
      for (let s = 0; s < SUBSTEPS; s++) mj.mj_step(model, data);
    };

    // ---- the settle, under the standing policy with a neutral command
    for (let t = 0; t < SETTLE_TICKS; t++) await step(stand, stand.reference ?? HOME, null, neutral);

    // ---- THE FROZEN HEADING, AND THE BALL PLACED AGAINST IT
    //
    // `kick_dir` is frozen at reset in Pollen's own code and never recomputed —
    // mdp.py 5700-5702: "Frozen for the episode so the policy can't redefine
    // 'forward' by turning after the kick." On this bench that instant is the
    // FIRST DRIVEN TICK: the settle is over, the entrant has not acted yet. The
    // same vector places the ball at its bearing and is the axis ballTravel_mm
    // projects onto. ONE VECTOR, READ TWICE.
    const yaw0 = yawOf(quat());
    const kickDir = [Math.cos(yaw0), Math.sin(yaw0)];
    const bearingRad = yaw0 + (cell.bearing * Math.PI) / 180;   // positive is LEFT
    const bx0 = data.qpos[f] + cell.range * Math.cos(bearingRad);
    const by0 = data.qpos[f + 1] + cell.range * Math.sin(bearingRad);
    data.qpos[BALL.adr] = bx0; data.qpos[BALL.adr + 1] = by0; data.qpos[BALL.adr + 2] = BALL_RADIUS;
    data.qpos[BALL.adr + 3] = 1;
    for (let k = 4; k < 7; k++) data.qpos[BALL.adr + k] = 0;
    for (let k = 0; k < 6; k++) data.qvel[BALL.dof + k] = 0;
    mj.mj_forward(model, data);
    const ball0 = [data.qpos[BALL.adr], data.qpos[BALL.adr + 1]];

    // ---- the running measurements
    const S = { ball_forward_velocity: 0, ball_speed_overshoot: 0, upright: 0,
                pose_stand_legs: 0, pose_stand_neck: 0, height_stand: 0,
                body_ang_vel: 0, action_rate_l2: 0, angular_momentum: 0 };
    let closest = Infinity, peakSpeed = 0, upTicks = 0, tailUpTicks = 0;
    let previousAction = null;
    let drivenTicks = 0, rateTicks = 0;

    /** Everything the reward reads, at the end of one DRIVEN tick. */
    const accumulate = (action) => {
      const q = quat();
      const vbx = data.qvel[BALL.dof], vby = data.qvel[BALL.dof + 1];
      // THE BALL'S OWN WORLD LINEAR VELOCITY, off the free joint — never a
      // finite difference of its position, which disagrees in the last digits.
      const fwd = vbx * kickDir[0] + vby * kickDir[1];
      S.ball_forward_velocity += Math.min(Math.max(fwd, 0), BALL_TARGET_SPEED);
      S.ball_speed_overshoot += Math.min(Math.max(fwd - BALL_TARGET_SPEED, 0), OVERSHOOT_MAX_PENALTY);
      S.upright += Math.exp(-gravityXYSquared(q) / UPRIGHT_STD2);
      let legs = 0;
      for (const k of LEG_JOINTS) {
        const d = (data.qpos[D.qpos[k]] - HOME[k]) / LEG_STD;
        legs += Math.exp(-(d * d));
      }
      S.pose_stand_legs += legs / LEG_JOINTS.length;
      let neck = 0;
      for (const k of NECK_JOINTS) {
        const d = (data.qpos[D.qpos[k]] - HOME[k]) / NECK_STD;
        neck += Math.exp(-(d * d));
      }
      S.pose_stand_neck += neck / NECK_JOINTS.length;
      const dz = (data.qpos[f + 2] - STAND_Z) / HEIGHT_STD;
      S.height_stand += Math.exp(-(dz * dz));
      // body_ang_vel wants the WORLD-frame angular velocity; MuJoCo keeps a free
      // joint's angular half in the BODY, so it is rotated out.
      const w = rotate(q, [data.qvel[D.freeDof + 3], data.qvel[D.freeDof + 4], data.qvel[D.freeDof + 5]]);
      S.body_ang_vel += w[0] * w[0] + w[1] * w[1];
      if (previousAction) {
        let r = 0;
        for (let k = 0; k < 14; k++) { const d = action[k] - previousAction[k]; r += d * d; }
        S.action_rate_l2 += r;
        rateTicks++;
      }
      previousAction = action;
      if (ANGMOM.adr >= 0) {
        let m = 0;
        for (let k = 0; k < 3; k++) { const v = data.sensordata[ANGMOM.adr + k]; m += v * v; }
        S.angular_momentum += m;
      }
      drivenTicks++;
    };

    /** The facts every tick contributes to, driven or tail. */
    const observe = (inTail) => {
      closest = ballDistanceNow(closest);
      const sp = Math.hypot(data.qvel[BALL.dof], data.qvel[BALL.dof + 1]);
      if (sp > peakSpeed) peakSpeed = sp;
      if (isUp()) { upTicks++; if (inTail) tailUpTicks++; }
    };

    // ---- the driven span
    const ticks = Math.max(1, Math.round(seconds * tickHz));
    for (let t = 0; t < ticks; t++) {
      const time = t * DT;
      const offsets = track ? poseAt(track, time, HOME) : null;
      const cmd = command(commandAt(schedule, time));
      await step(acting, actorRef, offsets, cmd);
      // WHAT aₜ IS, PER KIND. A policy's action is the network's own raw output;
      // a move's is the interpolated, clamped pose target it emitted. Same
      // formula, two quantities, and the answer says which.
      accumulate(isMove ? clampPose(offsets ?? HOME, LO, HI) : la.slice());
      observe(false);
    }

    // ---- the tail: THE STANDING TEST, and nothing else. The standing policy
    // holds the duck under a neutral command for 50 ticks, exactly as the
    // settle did and exactly as /climb's tail does. It used to keep the acting
    // network running under the entrant's own schedule, which made `stable` a
    // measurement of one more second of chasing rather than of standing, and
    // made every fact cover a span the entrant never declared. The round-3
    // review of this rail caught it; the criterion sentence promised a
    // standing test and now gets one.
    if (tail !== 'policy') throw new Error(`this grid is tail "policy": ${tail} is not one of its cells`);
    for (let t = 0; t < TAIL_TICKS; t++) {
      await step(stand, stand.reference ?? HOME, null, neutral);
      observe(true);
    }

    // ---- the eight plain facts
    const ballEnd = [data.qpos[BALL.adr], data.qpos[BALL.adr + 1]];
    const dx = ballEnd[0] - ball0[0], dy = ballEnd[1] - ball0[1];
    const rootEnd = [data.qpos[f], data.qpos[f + 1], data.qpos[f + 2]];
    const closest_mm = closest === Infinity ? DISTMAX * 1000 : closest * 1000;
    const facts = {
      // SIGNED, along the heading the duck STARTED the driven span on: a ball
      // pushed backwards scores negative, and a duck that turns to face wherever
      // the ball rolled cannot call that forward.
      ballTravel_mm: (dx * kickDir[0] + dy * kickDir[1]) * 1000,
      ballNet_mm: Math.hypot(dx, dy) * 1000,
      closest_mm,
      final_mm: Math.hypot(rootEnd[0] - ballEnd[0], rootEnd[1] - ballEnd[1]) * 1000,
      touched: closest_mm <= TOUCH_MM,
      // TRUE when no duck geom ever came within DISTMAX of the ball: closest_mm
      // is then the query's ceiling, not a measurement, and a reader must not
      // add it up as one.
      closestIsCeiling: closest === Infinity,
      ballPeakSpeed_mps: peakSpeed,
      upright: isUp(),
      uprightTailTicks: tailUpTicks,
    };

    // ---- the terms, as per-tick MEANS over the DRIVEN SPAN
    //
    // THE DRIVEN SPAN AND NOT THE TAIL. The tail is this bench's standing test,
    // not part of the entrant's episode; averaging Pollen's reward over ticks
    // the entrant did not ask for would put this bench's tail length into the
    // reward. The facts above span the whole episode because the criterion asks
    // about the END of it; the terms span what the entrant ran.
    const denom = t => (t === 'action_rate_l2' ? Math.max(rateTicks, 1) : Math.max(drivenTicks, 1));
    const terms = [], refused = CHASE_REFUSALS.map(r => ({ ...r }));
    for (const T of TERMS) {
      if (T.term === 'angular_momentum' && ANGMOM.adr < 0) {
        refused.push({ term: T.term, weight: T.weight, reason: ANGMOM.why });
        continue;
      }
      const row = { term: T.term, weight: T.weight, value: S[T.term] / denom(T.term),
                    source: T.source, formula: T.formula };
      if (T.term === 'action_rate_l2') {
        row.weightStage0 = T.weightStage0;
        row.action_rate_l2_source = ACTION_RATE_SOURCE[entrant.kind];
      }
      terms.push(row);
    }

    const V = verdict(facts);
    return {
      cell: { bearing: cell.bearing, range: cell.range, drop, fmul, tier: cell.tier },
      seconds, tail,
      yaw0, kickDir, ball0, ballEnd, rootEnd,
      facts, terms, refused,
      chased: V.chased, stable: V.stable,
      uprightTailTicks: facts.uprightTailTicks,
      uprightTicks: upTicks,
      drivenTicks, rateTicks, tailTicks: TAIL_TICKS,
      actionRateSource: ACTION_RATE_SOURCE[entrant.kind],
    };
  }

  return { runEpisode, BALL, BALLG, DUCKG, FEET, ANGMOM, GYRO, SUBSTEPS, TIMESTEP };
}

/** Clamp a fourteen-vector into the servo travel. */
export function clampPose(pose, LO, HI) {
  const out = new Array(14);
  for (let k = 0; k < 14; k++) out[k] = Math.min(Math.max(pose[k], LO[k]), HI[k]);
  return out;
}
