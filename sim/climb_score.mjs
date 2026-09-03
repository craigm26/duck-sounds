// climb_score.mjs — THE ONE CLIMB EPISODE, and the one criterion.
//
// WHY THIS FILE EXISTS AT ALL.
//
// The stair episode was written three times. climb/rig3.mjs holds the
// instrument the audits are quoted from; climb/robust.mjs holds the 14-cell
// grid every verdict is decided on and says so in its own header ("this is the
// ONE round-3 copy; every family imports scoreRobust from here rather than
// making a fourth"); and the bench was about to grow a fourth so that a phone
// could score a move without a desk. Three copies of a loop whose output is a
// trajectory do not stay equal by being read carefully — a moved line that
// changes the order two numbers are added in changes the eleventh decimal, and
// by tick 300 the duck is somewhere else. The audited numbers
// (climb/r6_judge-results.json phaseG) are quoted against robust.mjs's grid, so
// a phone that answers a DIFFERENT number under the same name is worse than a
// phone that answers nothing.
//
// So the loop is here, once, and rig3.mjs, robust.mjs and duckbench-core.mjs
// all call it. Each of them keeps its OWN record assembly — the exact field
// set it published before — because climb/audit_r6.mjs PHASE P1 compares 86
// rows of rig3's record by a deep recursive walk over every leaf, and a record
// that grew a field would fail that walk even though not one number moved.
// What is shared is the physics and the arithmetic. What is local is the shape
// of the answer.
//
// WHAT IS PARAMETERISED, AND WHY EACH ONE IS A KNOB RATHER THAN A FORK:
//
//   drop      spawn height. rig3 hard-codes 0.120; robust's plant axis moves it
//             to 0.125/0.130/0.140. At 0.120 the two are the same line.
//   fmul      multiplier on foot friction[0], read from the model at boot.
//             At 1.0 the multiplication is the identity and the model is
//             written back with the number it already had.
//   isolate   step-step collision isolation (site/stairs.js isolateSteps).
//             The SHIPPED affinity is captured at boot and restored after every
//             episode, so a bench that also answers /perform is handed its
//             world back exactly as it lent it.
//   stepCount how many of the fourteen blocks are laid out. 4 everywhere today.
//   tail      'policy' (climb_lib's own 50 stand-policy ticks) or 'hold'
//             (50 ticks with data.ctrl frozen at the clamped final pose).
//             robust only ever asks for 'policy'.
//
// EVERYTHING ELSE IS READ-ONLY MEASUREMENT and is computed on every episode
// whoever asked, because mj_geomDistance is a query: it touches neither qpos
// nor ctrl, so computing robust's wall-contact fractions inside a rig3 episode
// costs time and moves nothing. That is the only reason one loop can serve two
// callers that measured different things.
//
// Contact detection is mj_geomDistance, never data.contact.get(i) — that
// returns an embind object that leaks the WASM heap to 2 GB in ~20 s even when
// .delete() is called (climb/rig2.mjs).
//
// THE PATHS BELOW ARE THE BUNDLE'S PATHS. `./stairs.js`, `./climb_event.mjs`
// and `./climb_servo.mjs` are re-export shims in sim/ and are the real files in
// the phone bundle's flat assets/ directory, exactly as `./duckloop.mjs` is.
import { findStairJoints, layoutStairs, clearStairs, readStairs,
         STAIR_Y, STAIR_HALF_WIDTH } from './stairs.js';
import { normEvent, eventFires, eventError, buildDynTrack } from './climb_event.mjs';
import { normServo, servoBase, servoTick } from './climb_servo.mjs';

export { STAIR_Y, STAIR_HALF_WIDTH, findStairJoints, layoutStairs, clearStairs };

// ---------------------------------------------------------------- the lines

/** The flight is 340 mm wide. Anything outside it is not on the staircase. */
export const LATERAL = STAIR_HALF_WIDTH;          // 0.17 m
/** cfg.start — the first riser face. The line the trunk AND a foot must cross. */
export const RISER_X = 0.12;
/** The staircase every cell is laid out with. rig3's cfg, verbatim. */
export const STAIR_RUN = 0.28;
export const STAIR_START = 0.12;
export const DEFAULT_STEP_COUNT = 4;
/** `honest` needs the trunk this far above the tread. The bar, in metres. */
export const CEILING_ABOVE = 0.095;

/** The three plant settings. Cell 0 is the nominal plant rig3 itself uses. */
export const PLANTS = [
  { drop: 0.120, fmul: 1.0 },
  { drop: 0.130, fmul: 0.7 },
  { drop: 0.125, fmul: 1.3 },
];
/** The three rises, as offsets from the target. */
export const DHS = [-0.010, 0.000, 0.010];
/** Bonus added to the objective for each CORE cell cleared under 'honest'. */
export const CLEAR_BONUS = 4;
/** The round-4 extension: +/-5 mm on the nominal plant, and a slippery plant. */
export const EXT_DHS = [-0.005, 0.005];               // nominal plant only
export const EXT_PLANT = { drop: 0.140, fmul: 0.5 };  // crossed with DHS
export const EXT_CELL_COUNT = 14;
/** A cell's clear counts as STABLE only if the duck stood through the tail. */
export const UPRIGHT_TAIL_MIN = 45;   // of 50 ticks = 0.90 of the tail
export const UPRIGHT_BONUS = 4;
/** The DECLARED search bounds. A file outside them is not a result. */
export const DECLARED_BOUNDS = { blend: [0.7, 2.4], side: [-0.02, 0.09] };

/**
 * THE 14 CELLS, IN THE ORDER scoreRobust RUNS THEM.
 *
 * Core first (so a partial run is still the round-3 grid), then the five
 * extended ones, each tagged with which half it belongs to. This is the list
 * GET /climb/grid answers with, so a client never retypes it and a client that
 * pinned a fallback can be checked against it.
 */
export function gridCells({ core = false } = {}) {
  const plan = [];
  for (const dh of DHS) for (const p of PLANTS) plan.push({ dh, drop: p.drop, fmul: p.fmul, tier: 'core' });
  if (!core) {
    for (const dh of EXT_DHS) plan.push({ dh, drop: PLANTS[0].drop, fmul: PLANTS[0].fmul, tier: 'ext' });
    for (const dh of DHS) plan.push({ dh, drop: EXT_PLANT.drop, fmul: EXT_PLANT.fmul, tier: 'ext' });
  }
  return plan;
}

// ---------------------------------------------------------------- the intent

const isoOf = j => (j.isolate === undefined ? true : j.isolate !== false);
const scOf = j => j.stepCount || DEFAULT_STEP_COUNT;
export { isoOf as intentIsolate, scOf as intentStepCount };

/** Everything the episode actually reads, as options. rig3's and robust's, one copy. */
export function optsOf(j) {
  return {
    blend: j.blend, approach: j.approach || 0, gap: j.gap || 0, side: j.side || 0,
    spawn: j.spawn || null,
    // ROUND 4, FAMILY A: the optional event-triggered tail (climb_event.mjs).
    event: j.event || null,
    // ROUND 5: the optional servoed landing (climb_servo.mjs). Absent in every
    // file written before round 5 -> null -> not one line of the law runs.
    servo: j.servo || null,
    // ROUND 4, FAMILY B handoff fields; null for every older file
    spawnQuat: j.spawnQuat || null, spawnPose: j.spawnPose || null,
    spawnVel: j.spawnVel || null, spawnLastAction: j.spawnLastAction || null,
    settleTicks: j.settleTicks === undefined ? undefined : j.settleTicks,
  };
}

/** The shape every scorer insists on. Throws with the path in the message. */
export function checkIntent(j, where = 'intent') {
  if (!j || typeof j !== 'object') throw new Error('no intent in ' + where);
  if (!Array.isArray(j.keyframes) || !j.keyframes.length) throw new Error('no keyframes in ' + where);
  for (const f of j.keyframes) if (!Array.isArray(f.pose) || f.pose.length !== 14) throw new Error('bad pose in ' + where);
  return j;
}

const boundsOf = j => ({ ...DECLARED_BOUNDS, ...(j.bounds || {}) });
/** ROUND 4, HOLE 4: bounds are enforced at scoring time, not declared in a comment. */
export function checkBounds(j) {
  const B = boundsOf(j), bad = [];
  for (const [k, [lo, hi]] of Object.entries(B)) {
    const v = k === 'side' || k === 'gap' || k === 'approach' ? (j[k] || 0) : j[k];
    if (typeof v !== 'number' || !(v >= lo && v <= hi)) bad.push({ param: k, value: v, lo, hi });
  }
  return { bounds: B, violations: bad };
}

/**
 * THE MOVE'S IDENTITY, as the STRING that gets hashed.
 *
 * The hash itself is sha256 over this string. It is returned as a string and
 * not as a digest because the two shells hash differently — node:crypto here,
 * `crypto.subtle` (asynchronous) in a browser — and the ONE thing that must not
 * differ between them is what goes in. One vector published under three rise
 * labels produces one string and therefore one hash.
 */
export function intentHashPayload(j) {
  const h = {
    keyframes: j.keyframes, blend: j.blend, gap: j.gap || 0, side: j.side || 0,
    approach: j.approach || 0, spawn: j.spawn || null, isolate: isoOf(j), stepCount: scOf(j),
  };
  // PRESENT-ONLY, so a file written before a round has none of these keys and
  // hashes to exactly the value the published results carry.
  for (const k of ['event', 'servo', 'spawnQuat', 'spawnPose', 'spawnVel', 'spawnLastAction', 'settleTicks']) {
    if (j[k] !== undefined && j[k] !== null) h[k] = j[k];
  }
  return JSON.stringify(h);
}

// ---------------------------------------------------------- criterion + reward

/** climb_lib.mjs:80-86, verbatim. `home` is the pose a track departs from. */
export function poseAt(tr, time, home) {
  if (time <= 0) return home.slice();
  let pt = 0, pp = home;
  for (const f of tr) {
    if (time <= f.t) {
      const u = (time - pt) / Math.max(f.t - pt, 1e-9), s = u * u * (3 - 2 * u);
      return f.pose.map((v, k) => pp[k] + (v - pp[k]) * s);
    }
    pt = f.t; pp = f.pose;
  }
  return tr[tr.length - 1].pose.slice();
}

/**
 * Every criterion under consideration, evaluated on one snapshot.
 *
 *  orig    sim/climb_lib.mjs:150, unchanged.
 *  lat     orig + the lateral gate on trunk and feet. Still uses foot x > 0.05.
 *  honest  lat, plus a foot only counts as on the tread if it is past the riser
 *          face (x > 0.12) — the same line the trunk has to cross.
 *  honest60  honest with the height clause relaxed to 60 mm above the tread.
 *
 * ROUND 4, HOLE 1: the lateral gate is WHOLE-EPISODE here as well as in
 * reward(). The excursion arrives either on the snapshot or as the third
 * argument; with neither, the point gate is used and `lateralSource` says so.
 */
export function criteria(h, s, maxAbsDY) {
  const dyMax = (maxAbsDY !== undefined && maxAbsDY !== null) ? maxAbsDY
              : (s.maxAbsDY !== undefined && s.maxAbsDY !== null) ? s.maxAbsDY : null;
  const lateralAtScore = Math.abs(s.dy) <= LATERAL;
  const lateralEpisode = dyMax === null ? true : dyMax <= LATERAL;
  const lateral = lateralAtScore && lateralEpisode;
  const orig = s.up && s.x > RISER_X && s.above > 0.095 && s.feetUpRaw >= 2;
  const lat = s.up && lateral && s.x > RISER_X && s.above > 0.095 && s.feetUpLat >= 2;
  const honest = s.up && lateral && s.x > RISER_X && s.above > 0.095 && s.feetOnTread >= 2;
  const honest60 = s.up && lateral && s.x > RISER_X && s.above > 0.060 && s.feetOnTread >= 2;
  return { orig, lat, honest, honest60, lateral,
           lateralAtScore, lateralEpisode,
           lateralSource: dyMax === null ? 'point-only (no maxAbsDY supplied)' : 'whole-episode' };
}

/**
 * Shaped reward, WITH the lateral gate.
 *
 * The gate is hard and comes first: an episode that ever leaves the 340 mm of
 * flight scores 0 no matter what its x is.
 */
export function reward(rec) {
  const s = rec.scored;
  if (rec.maxAbsDY > LATERAL) return 0;            // left the flight at any point
  if (Math.abs(s.dy) > LATERAL) return 0;
  let r = 0;
  r += 3 * Math.max(0, Math.min(1, (s.x - (RISER_X - 0.20)) / 0.20));  // approach then cross the riser
  r += 2 * s.feetOnTread;                                              // the thing nobody has ever done
  r += 4 * Math.max(0, Math.min(1, s.above / 0.095));                  // stood up on it
  r += s.up ? 1 : 0;
  return r;
}

/**
 * ROUND 4, HOLE 2. A cell pays its upright credit only if the duck got
 * somewhere in that cell. Do-nothing never crosses x = 0.12 and so earns 0.
 */
export const reachedFlight = c => (c.maxX > RISER_X) || (c.feetOnTreadMax > 0);

/** THE SENTENCE THE VERDICT IS, said the same way on every bench. */
export const CRITERION_SENTENCE =
  'honest: at the scored instant the trunk is upright, past the riser at x > 120 mm, '
  + 'more than 95 mm above the tread, with at least two feet resting on a tread '
  + '(past the riser, within 5 mm below to 45 mm above it, and within 3 mm of a step), '
  + 'and the duck never left the 340 mm-wide flight at any tick of the episode. '
  + 'stable: honest, and upright for at least 45 of the 50 tail ticks.';

// ------------------------------------------------------------------- the rig

/**
 * A climb rig over one MuJoCo model.
 *
 * ctx = {
 *   mj, model, data       the module, the plant, and a scratch mjData THIS RIG
 *                         OWNS. It is reset at the top of every episode, so it
 *                         must not be a world anything else is keeping.
 *   D                     duckloop's findDuckJoints(model): { qpos[14], dof[14],
 *                         freeQpos, freeDof }
 *   HOME, LO, HI          duckkit-constants, through duckloop's makeLoop
 *   buildObs, projectedGravity, command    the same, verbatim
 *   reference             the neutral pose the acting policy declares. HOME for
 *                         alpha_stand, which is the only policy this scores
 *                         under; it is a parameter so that a bench which
 *                         resolved a policy that declares its own does not
 *                         quietly feed it a deviation from the wrong pose.
 *   run(Float32Array) -> ArrayLike(14)     the forward pass, awaited
 *   tickHz                C.tickHz
 * }
 *
 * Returns null when this plant has no stair bank, so a caller can say so rather
 * than score a flat room and call it a climb.
 */
export function makeClimbRig(ctx) {
  const { mj, model, data, D, HOME, LO, HI, buildObs, projectedGravity, command,
          run, tickHz, reference = HOME } = ctx;
  const DT = 1 / tickHz;

  // The shipped affinity is captured, and every episode sets and then RESTORES
  // it; see isolate above.
  const ADDR = findStairJoints(model, { isolate: false });
  if (!ADDR) return null;

  const bodyId = n => { for (let b = 0; b < model.nbody; b++) if (model.body(b).name === n) return b; return -1; };
  const JAWB = bodyId('jaw_soft');
  const JAW = [];
  for (let g = 0; g < model.ngeom; g++) {
    if (model.geom_bodyid[g] === JAWB && !(model.geom_contype[g] === 0 && model.geom_conaffinity[g] === 0)) JAW.push(g);
  }
  let STEP0 = -1, STEP1 = -1, FLOOR = -1, WALLN = -1, LFOOT = -1, RFOOT = -1;
  const STEPG = [];
  for (let g = 0; g < model.ngeom; g++) {
    const n = model.geom(g).name || '';
    if (n === 'step0_geom') STEP0 = g;
    if (n === 'step1_geom') STEP1 = g;
    if (n === 'floor') FLOOR = g;
    if (n === 'wall_n') WALLN = g;
    if (n === 'left_foot_collision') LFOOT = g;
    if (n === 'right_foot_collision') RFOOT = g;
    if (/^step\d+_geom$/.test(n)) STEPG.push(g);
  }
  // the exact geom set climb_lib line 141-145 walks
  const FEET = []; for (let g = 0; g < model.ngeom; g++) if (/foot_collision|sole/.test(model.geom(g).name || '')) FEET.push(g);
  // the duck's own legs and trunk, for the wall test (bodies hip_l/leg/ankle_*)
  const LEGG = [];
  for (let g = 0; g < model.ngeom; g++) {
    const b = model.body(model.geom_bodyid[g]).name || '';
    if (/^(hip_l|hip_l_2|leg|leg_2|ankle_left|ankle_right|trunk_base)$/.test(b) && model.geom_contype[g] === 5) LEGG.push(g);
  }
  const STEP_CONAFF0 = STEPG.map(g => model.geom_conaffinity[g]);
  // The baseline is the PLANT'S, handed in by the shell that read it at boot
  // (duckbench-core.mjs GEOM_FRICTION0); a rig built lazily while the other
  // rig's cell had a multiplier applied would otherwise read that as normal.
  const FRICT0 = FEET.map(g => (ctx.geomFriction0 ? ctx.geomFriction0[g * 3] : model.geom_friction[g * 3]));

  // ROUND 4, HOLE 3: every collidable geom that belongs to the DUCK, so that
  // penetration into a step block is a first-class field of a scored row.
  // Construction copied verbatim from climb/audit_r3.mjs (11 duck geoms,
  // 14 step geoms), which is the number those audits printed.
  let DUCKROOT = -1;
  for (let j = 0; j < model.njnt; j++) if (model.jnt_type[j] === 0 && model.jnt_qposadr[j] === D.freeQpos) { DUCKROOT = model.jnt_bodyid[j]; break; }
  const underDuck = b => { let c = b; for (let i = 0; i < 64 && c > 0; i++) { if (c === DUCKROOT) return true; c = model.body_parentid[c]; } return c === DUCKROOT; };
  const DUCKG = [];
  for (let g = 0; g < model.ngeom; g++) {
    if (model.geom_contype[g] === 0 && model.geom_conaffinity[g] === 0) continue;
    if (DUCKROOT >= 0 && underDuck(model.geom_bodyid[g])) DUCKG.push(g);
  }
  if (!DUCKG.length) { for (const g of JAW) DUCKG.push(g); for (const g of FEET) DUCKG.push(g); }

  const RBOUND = model.geom_rbound;
  const dist = (a, b) => mj.mj_geomDistance(model, data, a, b, 0.05, null);
  const quat = () => [data.qpos[D.freeQpos + 3], data.qpos[D.freeQpos + 4], data.qpos[D.freeQpos + 5], data.qpos[D.freeQpos + 6]];

  /**
   * A foot RESTING on the tread: past the riser line, inside the flight, within
   * 5 mm below to 45 mm above the tread's height, AND within 3 mm of a step geom.
   */
  function footResting(g, h) {
    const x = data.geom_xpos[g * 3], y = data.geom_xpos[g * 3 + 1], z = data.geom_xpos[g * 3 + 2];
    if (!(z > h - 0.005 && z < h + 0.045 && x > RISER_X && Math.abs(y - STAIR_Y) <= LATERAL)) return false;
    for (const sg of STEPG) if (dist(g, sg) < 0.003) return true;
    return false;
  }

  /** The most negative distance between ANY duck geom and ANY step geom, NOW. */
  function penetrationNow() {
    let pen = 1e9, pair = null;
    for (const g of DUCKG) for (const sg of STEPG) {
      const d = dist(g, sg);
      if (d < pen) { pen = d; pair = `${model.geom(g).name || 'g' + g}<->${model.geom(sg).name}`; }
    }
    return { pen: pen === 1e9 ? null : pen, pair };
  }

  /**
   * ROUND 5: the same query at EVERY control tick, kept as a running minimum,
   * so a move that passes THROUGH a block and arrives clean cannot score clean.
   * Exact, not sampled: the bounding-sphere test is a LOWER BOUND on
   * mj_geomDistance, so a skipped pair could not have improved the minimum.
   */
  function makePenTracker() {
    let best = 1e9, pair = null, ticks = 0, tickAt = -1;
    return {
      scan(tick) {
        ticks++;
        for (const g of DUCKG) {
          const gx = data.geom_xpos[g * 3], gy = data.geom_xpos[g * 3 + 1], gz = data.geom_xpos[g * 3 + 2];
          for (const sg of STEPG) {
            const dx = gx - data.geom_xpos[sg * 3], dy = gy - data.geom_xpos[sg * 3 + 1], dz = gz - data.geom_xpos[sg * 3 + 2];
            const lb = Math.sqrt(dx * dx + dy * dy + dz * dz) - RBOUND[g] - RBOUND[sg];
            if (lb >= best) continue;                     // cannot improve — exact
            const d = dist(g, sg);
            if (d < best) { best = d; pair = `${model.geom(g).name || 'g' + g}<->${model.geom(sg).name}`; tickAt = tick; }
          }
        }
      },
      get() { return { min: best === 1e9 ? null : best, pair, tick: tickAt, ticksScanned: ticks }; },
    };
  }

  /** Everything the criterion could possibly want, read off the live state. */
  function snapshot(h, maxAbsDY) {
    const x = data.qpos[D.freeQpos], y = data.qpos[D.freeQpos + 1], z = data.qpos[D.freeQpos + 2];
    const up = projectedGravity(quat())[2] < -0.90;
    // climb_lib.mjs:141-145 exactly as written — no y term, foot x > 0.05
    let feetUpRaw = 0, feetUpLat = 0, feetOnTread = 0;
    for (const g of FEET) {
      if (data.geom_xpos[g * 3 + 2] > h - 0.005 && data.geom_xpos[g * 3] > 0.05) feetUpRaw++;
      if (data.geom_xpos[g * 3 + 2] > h - 0.005 && data.geom_xpos[g * 3] > 0.05
        && Math.abs(data.geom_xpos[g * 3 + 1] - STAIR_Y) <= LATERAL) feetUpLat++;
      if (footResting(g, h)) feetOnTread++;
    }
    const foot = g => ({ x: data.geom_xpos[g * 3], y: data.geom_xpos[g * 3 + 1], z: data.geom_xpos[g * 3 + 2] });
    const P = penetrationNow();
    return { x, y, z, dy: y - STAIR_Y, above: z - h, up, feetUpRaw, feetUpLat, feetOnTread,
             lfoot: foot(LFOOT), rfoot: foot(RFOOT),
             maxAbsDY, penetrationAtScore: P.pen, penetrationPair: P.pair };
  }

  /** ROUND 4, FAMILY B: the complete state a beat-2 spawn needs. Read-only. */
  function handoffNow(h) {
    const jp = [], jv = [], free = [];
    for (let k = 0; k < 14; k++) { jp.push(data.qpos[D.qpos[k]]); jv.push(data.qvel[D.dof[k]]); }
    for (let k = 0; k < 6; k++) free.push(data.qvel[D.freeDof + k]);
    let head = false;
    for (const g of JAW) if (dist(g, STEP0) < 0.003) { head = true; break; }
    let footRiser = false, feetOnTread = 0;
    for (const g of [LFOOT, RFOOT]) {
      if (data.geom_xpos[g * 3 + 2] < h - 0.005 && dist(g, STEP0) < 0.003) footRiser = true;
    }
    for (const g of FEET) if (footResting(g, h)) feetOnTread++;
    const P = penetrationNow();
    return {
      penetration: P.pen, penetrationPair: P.pair,
      spawn: { x: data.qpos[D.freeQpos], y: data.qpos[D.freeQpos + 1], z: data.qpos[D.freeQpos + 2] },
      spawnQuat: [data.qpos[D.freeQpos + 3], data.qpos[D.freeQpos + 4], data.qpos[D.freeQpos + 5], data.qpos[D.freeQpos + 6]],
      spawnPose: jp, spawnVel: { free, joint: jv },
      head, footRiser, feetOnTread,
      up: projectedGravity(quat())[2] < -0.90,
    };
  }

  /**
   * ONE EPISODE. This is rig3.mjs runEpisodeRaw() and robust.mjs go(), which
   * were already proved equal on cell 0 (robust.mjs PHASE P), as one function.
   *
   * `clip` RIDES IN THE FIFTH ARGUMENT AND NOWHERE ELSE. That bag is the
   * per-cell PLANT — drop, friction, isolation, how many blocks — and it is the
   * one place a caller's flag cannot reach `intentHashPayload`: `optsOf` is the
   * intent-derived bag, and a render flag in it would change the identity of
   * every move that asked for a picture. It also keeps `clip` away from
   * `opts.pinEverySubstep`, the one inert-looking opt on this episode that
   * actually moves numbers.
   *
   * IT IS READ-ONLY AND IT IS PROVED TO BE. `sim/climb_parity.mjs` scores the
   * fourteen cells of five published files against climb/robust.mjs at full
   * float digits, with and without it.
   */
  async function runEpisode(track, opts, h, tail = 'policy',
                            { drop = 0.120, fmul = 1.0, isolate = true, stepCount,
                              clip = false } = {}) {
    const cfg = { count: stepCount || opts.stepCount || DEFAULT_STEP_COUNT, rise: h, run: STAIR_RUN, start: STAIR_START };
    STEPG.forEach((g, i) => { model.geom_conaffinity[g] = isolate ? 0 : STEP_CONAFF0[i]; });
    FEET.forEach((g, i) => { model.geom_friction[g * 3] = FRICT0[i] * fmul; });
    try {
      return await episode(track, opts, h, tail, cfg, drop, clip);
    } finally {
      // THE WORLD IS HANDED BACK EXACTLY AS IT WAS LENT. The bench answers
      // /perform out of the same model; a friction multiplier or a zeroed
      // conaffinity left behind would move a number nobody asked about.
      STEPG.forEach((g, i) => { model.geom_conaffinity[g] = STEP_CONAFF0[i]; });
      FEET.forEach((g, i) => { model.geom_friction[g * 3] = FRICT0[i]; });
      clearStairs(data, ADDR);
    }
  }

  async function episode(track, opts, h, tail, cfg, drop, clip = false) {
    mj.mj_resetData(model, data);
    layoutStairs(data, ADDR, cfg);
    if (opts.spawn) {
      data.qpos[D.freeQpos] = opts.spawn.x;
      data.qpos[D.freeQpos + 1] = opts.spawn.y;
      data.qpos[D.freeQpos + 2] = opts.spawn.z + (drop - 0.120);
    } else {
      data.qpos[D.freeQpos] = 0.12 - 0.07 - (opts.gap || 0);
      data.qpos[D.freeQpos + 1] = STAIR_Y + (opts.side || 0);
      data.qpos[D.freeQpos + 2] = drop;
    }
    data.qpos[D.freeQpos + 3] = 1;
    for (let i = 0; i < 14; i++) { data.qpos[D.qpos[i]] = HOME[i]; data.ctrl[i] = HOME[i]; }
    // ROUND 4, FAMILY B: the optional handoff state. Absent -> nothing runs.
    if (opts.spawnQuat) for (let k = 0; k < 4; k++) data.qpos[D.freeQpos + 3 + k] = opts.spawnQuat[k];
    if (opts.spawnPose) for (let i = 0; i < 14; i++) {
      data.qpos[D.qpos[i]] = opts.spawnPose[i];
      data.ctrl[i] = Math.min(Math.max(opts.spawnPose[i], LO[i]), HI[i]);
    }
    if (opts.spawnVel) {
      if (opts.spawnVel.free) for (let k = 0; k < 6; k++) data.qvel[D.freeDof + k] = opts.spawnVel.free[k];
      if (opts.spawnVel.joint) for (let i = 0; i < 14; i++) data.qvel[D.dof[i]] = opts.spawnVel.joint[i];
    }
    mj.mj_forward(model, data);
    const tr = track.map(f => ({ t: f.t, pose: f.pose.slice() }));
    let la = opts.spawnLastAction ? opts.spawnLastAction.slice() : new Array(14).fill(0);
    const cmd = command({ vx: opts.approach || 0 });

    const R = { ticks: 0, headTicks: 0, riserTicks: 0, wallTicks: 0, wallBearTicks: 0,
                upTicks: 0, sat: 0, ctrls: 0, maxX: -1e9, maxZ: -1e9, maxAbsDY: 0,
                feetOnTreadMax: 0, feetUpRawMax: 0, feetHighMax: 0, headOnlyTicks: 0,
                bothTicks: 0, sustainTicks: 0, liftIntegral: 0, maxGainBoth: -1e9,
                wallGain: -1e9, maxTq: 0, footNear: 1e9, bothNear: 1e9,
                // rig3 reads the tread drift over EVERY tick (its treadDrift()
                // runs in the settle too); robust reads it over the RECORDED
                // ticks only. Both are kept, because both were published.
                allSag_mm: 0, allDriftX_mm: 0, allGap_mm: 1e9,
                recDriftX_mm: 0, recGap_mm: 1e9,
                trace: [] };
    let Z0 = 0;                       // trunk z at the end of the settle
    let gtick = 0;
    const PEN = makePenTracker();
    // A LANDING SPOT on the first tread: mid-tread, one foot-thickness up.
    const TGT = [0.22, STAIR_Y, h + 0.015];
    const nearTo = g => {
      const dx = data.geom_xpos[g * 3] - TGT[0], dy = data.geom_xpos[g * 3 + 1] - TGT[1],
            dz = data.geom_xpos[g * 3 + 2] - TGT[2];
      return Math.sqrt(dx * dx + dy * dy + dz * dz);
    };

    /**
     * How far the tread has moved by the end of a control tick.
     *
     * site/stairs.js says of pin(): "Call after any qpos write, AND EVERY TICK."
     * climb_lib.mjs:132 calls layoutStairs once and then takes FOUR mj_steps. A
     * step is a 200 kg box on two frictionless, undamped slide joints with no
     * gravity compensation, so between pins it is in free fall.
     */
    const treadDrift = (rec) => {
      const topNow = data.geom_xpos[STEP0 * 3 + 2] + 0.10;
      const sag = (h - topNow) * 1000;
      if (sag > R.allSag_mm) R.allSag_mm = sag;
      const dx = Math.abs(data.geom_xpos[STEP0 * 3] - (0.12 + 0.17)) * 1000;
      if (dx > R.allDriftX_mm) R.allDriftX_mm = dx;
      if (rec && dx > R.recDriftX_mm) R.recDriftX_mm = dx;
      // Consecutive steps OVERLAP by design and share a collision bit, so below
      // a rise of 140 mm an un-isolated flight shoves itself apart horizontally.
      if (STEP1 >= 0 && cfg.count > 1) {
        const g = dist(STEP0, STEP1) * 1000;
        if (g < R.allGap_mm) R.allGap_mm = g;
        if (rec && g < R.recGap_mm) R.recGap_mm = g;
      }
      return sag;
    };
    const traceSample = (phase) => {
      if (!opts.trace) return;
      if (gtick % 10) return;
      R.trace.push({ tick: gtick, phase,
        x_mm: +(data.qpos[D.freeQpos] * 1000).toFixed(1),
        dy_mm: +((data.qpos[D.freeQpos + 1] - STAIR_Y) * 1000).toFixed(1),
        z_mm: +(data.qpos[D.freeQpos + 2] * 1000).toFixed(1),
        lfootZ_mm: +(data.geom_xpos[LFOOT * 3 + 2] * 1000).toFixed(1),
        rfootZ_mm: +(data.geom_xpos[RFOOT * 3 + 2] * 1000).toFixed(1),
        lfootX_mm: +(data.geom_xpos[LFOOT * 3] * 1000).toFixed(1),
        rfootX_mm: +(data.geom_xpos[RFOOT * 3] * 1000).toFixed(1),
        treadSag_mm: +treadDrift(false).toFixed(2),
        up: projectedGravity(quat())[2] < -0.90 });
    };

    const record = () => {
      R.ticks++;
      const x = data.qpos[D.freeQpos], y = data.qpos[D.freeQpos + 1], z = data.qpos[D.freeQpos + 2];
      if (x > R.maxX) R.maxX = x;
      if (z > R.maxZ) R.maxZ = z;
      const ady = Math.abs(y - STAIR_Y); if (ady > R.maxAbsDY) R.maxAbsDY = ady;
      if (projectedGravity(quat())[2] < -0.90) R.upTicks++;
      let head = false;
      for (const g of JAW) if (dist(g, STEP0) < 0.003) { head = true; break; }
      if (head) R.headTicks++;
      let footRiser = false, footAny = false;
      for (const g of [LFOOT, RFOOT]) {
        const onBlock = dist(g, STEP0) < 0.003, onFloor = FLOOR >= 0 && dist(g, FLOOR) < 0.003;
        if (onBlock || onFloor) footAny = true;
        if (onBlock && data.geom_xpos[g * 3 + 2] < h - 0.005) footRiser = true;
      }
      if (footRiser) R.riserTicks++;
      if (head && !footAny) R.headOnlyTicks++;
      // THE WALL as a third contact
      let wall = false;
      for (const g of LEGG) if (WALLN >= 0 && dist(g, WALLN) < 0.003) { wall = true; break; }
      if (wall) R.wallTicks++;
      let fot = 0, fur = 0, fhi = 0;
      for (const g of FEET) {
        const lat = Math.abs(data.geom_xpos[g * 3 + 1] - STAIR_Y) <= LATERAL;
        if (data.geom_xpos[g * 3 + 2] > h - 0.005 && data.geom_xpos[g * 3] > 0.05) fur++;
        if (footResting(g, h)) fot++;
        if (data.geom_xpos[g * 3 + 2] > h + 0.005 && lat) fhi++;
      }
      if (fot > R.feetOnTreadMax) R.feetOnTreadMax = fot;
      if (fur > R.feetUpRawMax) R.feetUpRawMax = fur;
      if (fhi > R.feetHighMax) R.feetHighMax = fhi;
      const nL = nearTo(LFOOT), nR = nearTo(RFOOT);
      if (Math.min(nL, nR) < R.footNear) R.footNear = Math.min(nL, nR);
      if (Math.max(nL, nR) < R.bothNear) R.bothNear = Math.max(nL, nR);
      // SUSTAINED LOAD TRANSFER: rising while the head and one foot both bear
      if (head && (footRiser || fot > 0)) {
        R.bothTicks++;
        const g = z - Z0;
        if (g > R.maxGainBoth) R.maxGainBoth = g;
        if (g > 0.02) R.sustainTicks++;
        if (g > 0) R.liftIntegral += g;
      }
      // was the wall ever LOAD-BEARING
      if (wall && (head || footRiser || fot > 0)) {
        R.wallBearTicks++;
        const g = z - Z0;
        if (g > R.wallGain) R.wallGain = g;
      }
      for (let a = 0; a < model.nu; a++) { const f = Math.abs(data.actuator_force[a]); if (f > R.maxTq) R.maxTq = f; }
    };

    /**
     * THE CELL'S PICTURE — a clip, in duck-intent-clips/3's own shape.
     *
     * A /climb answer has always been twenty-six numbers and no trajectory, so
     * a person who ran a cell could read that it cleared and could not watch it
     * do so. This is the trajectory, and it is opt-in because a clip is five
     * thousand leaves and a score is fourteen cells.
     *
     * READ IT AND BE SURE OF ONE THING, WHICH IS THE WHOLE REVIEW: it reads
     * `data.qpos` and pushes. It touches no `R.*`, it calls no `treadDrift` —
     * the trap `traceSample` sits one line away from, because treadDrift
     * updates the running maxima this rig PUBLISHES as `maxTreadSag_mm` — and
     * it runs after every instrument has already read the state. That is round
     * 6's `tailTrace` precedent, and it is why climb_parity is 70/70 with this
     * on and with it off.
     *
     * `stairs` IS READ AT A DIFFERENT MOMENT AND HAS TO BE. The blocks are
     * 200 kg bodies on frictionless, undamped slides and fall about two
     * millimetres between pins, so the only instant they are exactly what was
     * laid is immediately after `layoutStairs` and before the four substeps —
     * which is where it is taken, overwritten every tick so the last one wins.
     * A read after the episode would be worse still: `runEpisode`'s `finally`
     * parks all fourteen.
     */
    const CLIP = clip ? { frames: [], roots: [], commands: [], stairs: null } : null;
    const r4 = v => Math.round(v * 10000) / 10000;
    const clipSample = () => {
      if (!CLIP) return;
      const after = [];
      for (let k = 0; k < 14; k++) after.push(Math.min(Math.max(data.qpos[D.qpos[k]], LO[k]), HI[k]));
      CLIP.frames.push(after.map(r4));
      const f = D.freeQpos;
      CLIP.roots.push([data.qpos[f], data.qpos[f + 1], data.qpos[f + 2],
                       data.qpos[f + 3], data.qpos[f + 4], data.qpos[f + 5], data.qpos[f + 6]].map(r4));
      CLIP.commands.push([cmd[0], cmd[1], cmd[2]].map(r4));
    };

    // climb_lib.mjs:121-133, verbatim.
    // ROUND 5: `sv` is the servoed-landing target vector for this tick — a
    // number for every LEG slot the law owns, null everywhere else. It is
    // undefined on every tick of every file that carries no `servo` block, and
    // then the added condition is `undefined && ...` and not one number moves.
    const step = async (off, rec, sv) => {
      layoutStairs(data, ADDR, cfg);
      if (CLIP) CLIP.stairs = readStairs(data, ADDR);   // at the pin, before the substeps
      const q = quat(); const jp = [], jv = [];
      for (let k = 0; k < 14; k++) { jp.push(data.qpos[D.qpos[k]]); jv.push(data.qvel[D.dof[k]]); }
      const obs = buildObs([data.sensordata[GYRO], data.sensordata[GYRO + 1], data.sensordata[GYRO + 2]],
                           projectedGravity(q), jp, jv, la, cmd, reference);
      la = Array.from(await run(obs));
      for (let k = 0; k < 14; k++) {
        const v = (sv && sv[k] !== null) ? sv[k]
                : reference[k] + la[k] + (off ? (off[k] - HOME[k]) * opts.blend : 0);
        const c = Math.min(Math.max(v, LO[k]), HI[k]);
        data.ctrl[k] = c;
        if (rec) { R.ctrls++; if (c <= LO[k] + 1e-9 || c >= HI[k] - 1e-9) R.sat++; }
      }
      for (let s = 0; s < 4; s++) { if (opts.pinEverySubstep) layoutStairs(data, ADDR, cfg); mj.mj_step(model, data); }
      treadDrift(rec); traceSample(rec ? 'track' : 'settle'); PEN.scan(gtick); gtick++;
      if (rec) { record(); clipSample(); }
    };

    /** No policy at all: the servos hold the targets they are given. */
    const holdStep = (targets) => {
      layoutStairs(data, ADDR, cfg);
      if (CLIP) CLIP.stairs = readStairs(data, ADDR);   // at the pin, before the substeps
      for (let k = 0; k < 14; k++) data.ctrl[k] = targets[k];
      for (let s = 0; s < 4; s++) { if (opts.pinEverySubstep) layoutStairs(data, ADDR, cfg); mj.mj_step(model, data); }
      treadDrift(true); traceSample('tail'); PEN.scan(gtick); gtick++;
      record(); clipSample();
    };

    const SETTLE = (opts.settleTicks === undefined || opts.settleTicks === null) ? 25 : opts.settleTicks;
    for (let t = 0; t < SETTLE; t++) await step(null, false);
    const x0 = data.qpos[D.freeQpos];
    Z0 = data.qpos[D.freeQpos + 2];

    // ================================================== ROUND 4, FAMILY A
    // AN OPTIONAL EVENT-TRIGGERED TAIL. `opts.event` is absent in every file
    // written before round 4, normEvent(undefined) is null, TR === tr, `total`
    // never changes, and the loop below is the pre-round-4 loop verbatim.
    const EV = normEvent(opts.event);
    let TR = tr;
    let total = TR[TR.length - 1].t + 0.8;
    let evFired = false, evT = null, evE = null, evTrunkX = null;
    const beakDistNow = () => {
      let d = 1e9;
      for (const g of JAW) for (const sg of STEPG) { const v = dist(g, sg); if (v < d) d = v; }
      return d === 1e9 ? null : d;
    };
    // ================================================== ROUND 5, THE SERVO
    const SV = normServo(opts.servo);
    let svArmed = false, svT = null, svBase = null, svTicks = 0, svLastCtrl = null;
    const svLog = [];
    /** The five servo readings, off the live state. Read-only. */
    const svMeasure = () => ({
      above: data.qpos[D.freeQpos + 2] - h,
      pitch: projectedGravity(quat())[0],
      dxTrunk: data.qpos[D.freeQpos] - RISER_X,
      feet: [LFOOT, RFOOT].map(g => ({
        dx: data.geom_xpos[g * 3] - RISER_X,
        dz: data.geom_xpos[g * 3 + 2] - h,
      })),
    });
    for (let t = 0; t * DT < total; t++) {
      const time = t * DT;
      if (EV && !evFired && time >= EV.arm) {
        const fire = time >= EV.fallback || eventFires(EV, {
          beakDist: EV.type === 'beak' ? beakDistNow() : null,
          pitch: projectedGravity(quat())[0],
          above: data.qpos[D.freeQpos + 2] - h,
        });
        if (fire) {
          evFired = true; evT = time; evTrunkX = data.qpos[D.freeQpos];
          evE = eventError(EV, evTrunkX);
          TR = buildDynTrack(tr, EV, time, poseAt(TR, time, HOME), evE);
          total = TR[TR.length - 1].t + 0.8;
        }
      }
      let svTargets;
      if (SV) {
        if (!svArmed && ((SV.at !== null && time >= SV.at) || (SV.onEvent && evFired))) {
          svArmed = true; svT = time; svBase = servoBase(SV, poseAt(TR, time, HOME));
        }
        if (svArmed) {
          const prev = []; for (let k = 0; k < 14; k++) prev.push(data.ctrl[k]);
          const m = svMeasure();
          svTargets = servoTick(SV, svBase, m, prev, LO, HI);
          svLastCtrl = svTargets;
          svTicks++;
          if (opts.servoTrace && svTicks % 5 === 1) svLog.push({
            t: +time.toFixed(3),
            above_mm: +((m.above) * 1000).toFixed(1), pitch: +m.pitch.toFixed(4),
            trunkX_mm: +((m.dxTrunk + RISER_X) * 1000).toFixed(1),
            lfoot: { dx_mm: +(m.feet[0].dx * 1000).toFixed(1), dz_mm: +(m.feet[0].dz * 1000).toFixed(1) },
            rfoot: { dx_mm: +(m.feet[1].dx * 1000).toFixed(1), dz_mm: +(m.feet[1].dz * 1000).toFixed(1) },
            cmd: svTargets.map(v => v === null ? null : +v.toFixed(4)),
          });
        }
      }
      await step(poseAt(TR, time, HOME), true, svTargets);
    }

    const atTrackEnd = snapshot(h, R.maxAbsDY);
    // ROUND 4, FAMILY B: the handoff point. This instant IS where a beat-2 track
    // is concatenated (beat 1's last keyframe + 0.8 s).
    const terminal = handoffNow(h); terminal.spawnLastAction = la.slice();
    // what the servos were last told, and what 'hold' will freeze them at
    const ctrlAtHandoff = []; for (let k = 0; k < 14; k++) ctrlAtHandoff.push(data.ctrl[k]);
    const finalPose = poseAt(TR, total, HOME);   // TR === tr when the file has no event
    const held = finalPose.map((v, k) => Math.min(Math.max(v, LO[k]), HI[k]));
    // ROUND 5: if the servo took the legs, the 'hold' tail freezes at what the
    // SERVO last commanded, not at a keyframe the legs stopped following.
    if (svArmed && svLastCtrl) for (let k = 0; k < 14; k++) if (svLastCtrl[k] !== null) held[k] = svLastCtrl[k];
    let ctrlJump = 0; for (let k = 0; k < 14; k++) ctrlJump = Math.max(ctrlJump, Math.abs(held[k] - ctrlAtHandoff[k]));

    // ROUND 4, HOLE 2: how much of the 50-tick tail was the duck upright?
    const upBeforeTail = R.upTicks, ticksBeforeTail = R.ticks;
    // ROUND 6, THE TAIL. Two additive things, both inert unless asked for:
    //  (a) servo.tailTicks (default 0) keeps the leg slots on the law for the
    //      first n tail ticks. With 0 the loop is `await step(null, true,
    //      undefined)`, the pre-round-6 call verbatim.
    //  (b) opts.tailTrace (default false), a read-only per-tail-tick record.
    const tailLog = [];
    let svTailRun = 0;
    const tailSample = (t, servoed) => {
      const pg = projectedGravity(quat());
      const cmdv = [], qv = [];
      let sat = 0;
      for (let k = 0; k < 14; k++) {
        const c = data.ctrl[k];
        cmdv.push(c); qv.push(data.qpos[D.qpos[k]]);
        if (c <= LO[k] + 1e-9 || c >= HI[k] - 1e-9) sat++;
      }
      const fg = g => [data.geom_xpos[g * 3], data.geom_xpos[g * 3 + 1], data.geom_xpos[g * 3 + 2]];
      tailLog.push({
        t, servoed, gz: pg[2], up: pg[2] < -0.90, pitch: pg[0], roll: pg[1],
        x: data.qpos[D.freeQpos], y: data.qpos[D.freeQpos + 1], z: data.qpos[D.freeQpos + 2],
        above: data.qpos[D.freeQpos + 2] - h,
        lfoot: fg(LFOOT), rfoot: fg(RFOOT),
        cmd: cmdv, qpos: qv, sat,
      });
    };
    // 'policy' RUNS THE POLICY TAIL AND EVERYTHING ELSE HOLDS — including
    // 'none', which is the SAME simulation as 'hold' and differs only in which
    // snapshot the caller scores. rig3 has always read it that way and one run
    // yields both; a `tail === 'hold'` test here instead would silently give
    // 'none' the policy tail.
    let afterTail;
    if (tail !== 'policy') {
      for (let t = 0; t < 50; t++) holdStep(held);
      afterTail = snapshot(h, R.maxAbsDY);
    } else {
      for (let t = 0; t < 50; t++) {
        let svTail;
        if (SV && svArmed && t < SV.tailTicks) {
          const prev = []; for (let k = 0; k < 14; k++) prev.push(data.ctrl[k]);
          svTail = servoTick(SV, svBase, svMeasure(), prev, LO, HI);
          svTicks++; svTailRun++;
        }
        await step(null, true, svTail);
        if (opts.tailTrace) tailSample(t, svTail !== undefined);
      }
      afterTail = snapshot(h, R.maxAbsDY);
    }
    const uprightTailTicks = R.upTicks - upBeforeTail;
    const tailTicks = R.ticks - ticksBeforeTail;
    const P = PEN.get();

    return {
      rise: h, x0, z0Settle: Z0, ctrlJump, ctrlAtHandoff, held, finalPose,
      atTrackEnd, afterTail, terminal,
      event: EV ? { type: EV.type, fired: evFired, tFire: evT, trunkXAtFire: evTrunkX,
                    e_mm: evE === null ? null : +(evE * 1000).toFixed(2) } : null,
      servo: SV ? { armed: svArmed, tArm: svT, ticks: svTicks, at: SV.at, onEvent: SV.onEvent,
                    tailAuthority: SV.tailTicks, tailTicksRun: svTailRun, base: svBase,
                    trace: opts.servoTrace ? svLog : undefined } : null,
      tailTrace: opts.tailTrace ? tailLog : undefined,
      uprightTailTicks, tailTicks,
      penetration: { min: P.min, pair: P.pair, tick: P.tick, ticksScanned: P.ticksScanned },
      maxX: R.maxX, maxZ: R.maxZ, maxAbsDY: R.maxAbsDY, maxTq: R.maxTq,
      feetOnTreadMax: R.feetOnTreadMax, feetUpRawMax: R.feetUpRawMax, feetHighMax: R.feetHighMax,
      ticks: R.ticks, headTicks: R.headTicks, riserTicks: R.riserTicks, upTicks: R.upTicks,
      wallTicks: R.wallTicks, wallBearTicks: R.wallBearTicks, headOnlyTicks: R.headOnlyTicks,
      bothTicks: R.bothTicks, sustainTicks: R.sustainTicks, liftIntegral: R.liftIntegral,
      maxGainBoth: R.maxGainBoth, wallGain: R.wallGain,
      sat: R.sat, ctrls: R.ctrls,
      // BOTH `undefined` UNLESS ASKED FOR, and therefore structurally invisible:
      // climb/rig3.mjs and climb/robust.mjs each assemble their own record by
      // picking named fields off this object and never by spreading it, and
      // climb_parity compares a fixed list of fourteen named fields.
      clip: CLIP ? { frames: CLIP.frames, roots: CLIP.roots, commands: CLIP.commands } : undefined,
      stairsInEpisode: CLIP ? CLIP.stairs : undefined,
      footNear: R.footNear, bothNear: R.bothNear,
      // rig3's tread drift (every tick) and robust's (recorded ticks only)
      allSag_mm: R.allSag_mm, allDriftX_mm: R.allDriftX_mm, allGap_mm: R.allGap_mm,
      recDriftX_mm: R.recDriftX_mm, recGap_mm: R.recGap_mm,
      trace: R.trace,
    };
  }

  // The gyro this rig reads. Looked up once, here rather than in ctx, because
  // every caller found it the same way and one of them finding it differently
  // would feed the policy some other sensor's three numbers as its angular
  // velocity — a lie that produces a plausible-looking rollout.
  let GYRO = 0;
  for (let i = 0; i < model.nsensor; i++) if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;

  return { runEpisode, ADDR, DUCKG, STEPG, FEET, JAW, LEGG, STEP0, STEP1, FLOOR, WALLN, LFOOT, RFOOT, GYRO,
           snapshotFields: ['x', 'y', 'z', 'dy', 'above', 'up', 'feetUpRaw', 'feetUpLat', 'feetOnTread'] };
}
