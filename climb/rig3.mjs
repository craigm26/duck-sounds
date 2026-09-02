// rig3 — the instrument, fixed.
//
// This is a parity-checked copy of sim/climb_lib.mjs's attempt()/replay()
// (that file's lines 99-152). sim/ is off-limits to edit and importing it runs
// its top level (a second MjModel + a second ONNX session), so this is a COPY.
// Every line of the episode staging below — cfg, spawn, the 25-tick settle, the
// +0.8 s track tail, the 50-tick hold, the ctrl blend, the clamp — is
// reproduced verbatim from climb_lib. Phase P below PROVES that: same track,
// same opts, same rise, through both harnesses, compared at full float digits
// (climb/audit_cross.mjs's method).
//
// What is DIFFERENT, and why:
//
//  (1) A LATERAL GATE. climb_lib's criterion (line 150) has no y term at all,
//      and neither does its feetUp loop (line 144: z and x only). The flight is
//      only 340 mm wide (site/stairs.js STAIR_HALF_WIDTH = 0.17), so a duck
//      that walks 426 mm off the side of it and stands on the floor scores
//      "trunk past the riser" — the podium confound at flight scale. Here
//      |y - STAIR_Y| <= 0.17 gates the criterion, the feetUp test, AND the
//      exported reward.
//
//  (2) THE FOOT-x BUG. climb_lib line 144 accepts a foot as "at/above the
//      tread" when x > 0.05. The first riser is at x = 0.12. A foot at
//      x = 0.06, z = h is 60 mm IN FRONT of the riser face, in mid-air, and it
//      counts. feetOnTread here requires x > 0.12 — the same line the trunk
//      has to cross.
//
//  (3) SCORING READS A FILE. scoreSaved() takes a PATH and JSON.parse()s it.
//      There is no entry point that scores an in-memory candidate, because the
//      exported keyframes are rounded (search_0.mjs:300 pose->4 dp, blend/gap/
//      side->4 dp; search_2.mjs:307 pose->5 dp) while the numbers quoted in the
//      notes came from the unrounded candidate. Phase R measures how far apart
//      those two are.
//
//  (4) THREE TAILS. climb_lib scores 50 ticks of BEST_alpha_stand.onnx AFTER
//      the track ends. Terminal trunk z is 115.3-118.1 mm in 41 of 41 upright
//      replays and a do-nothing control lands in the same band, which is what
//      you would expect if that tail simply walks the duck back to a floor
//      stance. --tail selects:
//        policy : 50 ticks of the stand policy   (climb_lib's own tail)
//        hold   : 50 ticks with data.ctrl frozen at clamp(final track pose),
//                 no ONNX call at all — the servos hold position
//        none   : score at the track's last tick; then run the 50 hold ticks
//                 anyway and report whether it is still upright after them
//      'none' and 'hold' are the same simulation (they differ only in which
//      snapshot is the answer), so one run yields both.
//
//  (4b) ROUND 5, THE SERVOED LANDING. An optional `servo` field (see
//      climb/servo.mjs) that, once armed by an authored track time or by the
//      round-4 event, takes the ten LEG slots off the keyframe track and
//      commands them EVERY CONTROL TICK from measured trunk height above the
//      tread, trunk pitch, trunk x relative to the riser line, and each foot's
//      x and z relative to the tread edge and top. Head and neck keep
//      following the keyframes. Targets are rate-limited and clamped to
//      [LO, HI]; the actuator ceiling is untouched. A file with no `servo`
//      replays byte-identically — climb/_r5_parity.mjs proves that on 318 rows
//      against climb/rig3_pre_r5.mjs (a byte copy of this file at commit
//      e00e1e4) and reproduces the round-4 judge's own PHASE P counts against
//      climb/rig3_pre_r4.mjs.
//
//  (4c) ROUND 5, THE WHOLE-EPISODE PENETRATION FIELD. penetrationAtScore is
//      one instant, so a move that passes THROUGH a block on the way up and
//      arrives clean scored clean (round 3 measured -9.4 mm transiently).
//      `minPenetrationEpisode` is the most negative mj_geomDistance between
//      any duck geom and any step geom at ANY control tick of the episode —
//      settle, track and tail — with the pair and the tick named. Exact, not
//      sampled: the bounding-sphere skip is a lower bound on the distance.
//
//  (5) A SPAWN OVERRIDE. An intent JSON may carry `spawn: {x,y,z}`, which
//      replaces the gap/side spawn. That is the only way to author the row a
//      criterion MUST pass: a duck already standing on the first tread.
//
// Contact detection is mj_geomDistance, never data.contact.get(i) — that
// returns an embind object that leaks the WASM heap to 2 GB in ~20 s even when
// .delete() is called (see climb/rig2.mjs).
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/rig3.mjs
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
import { findStairJoints, layoutStairs, STAIR_Y, STAIR_HALF_WIDTH } from '../site/stairs.js';
import { normEvent, eventFires, eventError, buildDynTrack } from '../climb/event.mjs';
import { normServo, servoBase, servoTick } from '../climb/servo.mjs';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
export const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const model = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model), ADDR = findStairJoints(model);
let GYRO = 0;
for (let i = 0; i < model.nsensor; i++) if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;
const stand = await ort.InferenceSession.create('./BEST_alpha_stand.onnx');
const DT = 1 / C.tickHz;

/** The flight is 340 mm wide. Anything outside it is not on the staircase. */
export const LATERAL = STAIR_HALF_WIDTH;          // 0.17 m
export const RISER_X = 0.12;                      // cfg.start — the first riser face

const bodyId = n => { for (let b = 0; b < model.nbody; b++) if (model.body(b).name === n) return b; return -1; };
const JAWB = bodyId('jaw_soft');
const JAW = []; for (let g = 0; g < model.ngeom; g++) if (model.geom_bodyid[g] === JAWB && !(model.geom_contype[g] === 0 && model.geom_conaffinity[g] === 0)) JAW.push(g);
let STEP0 = -1, STEP1 = -1;
for (let g = 0; g < model.ngeom; g++) {
  if (model.geom(g).name === 'step0_geom') STEP0 = g;
  if (model.geom(g).name === 'step1_geom') STEP1 = g;
}
let LFOOT = -1, RFOOT = -1;
for (let g = 0; g < model.ngeom; g++) {
  const n = model.geom(g).name || '';
  if (n === 'left_foot_collision') LFOOT = g;
  if (n === 'right_foot_collision') RFOOT = g;
}
// the exact geom set climb_lib line 141-145 walks
const FEET = []; for (let g = 0; g < model.ngeom; g++) if (/foot_collision|sole/.test(model.geom(g).name || '')) FEET.push(g);
// Every step geom, for the contact half of the foot clause below.
const STEPG = []; for (let g = 0; g < model.ngeom; g++) if (/^step\d+_geom$/.test(model.geom(g).name || '')) STEPG.push(g);

// ROUND 4, HOLE 3: every collidable geom that belongs to the DUCK, so that
// penetration into a step block is a first-class field of a scored row rather
// than something only the audit ever looked at. Construction copied verbatim
// from climb/audit_r3.mjs (which reported 11 duck geoms, 14 step geoms) so the
// number printed here is the number that audit printed.
let DUCKROOT = -1;
for (let j = 0; j < model.njnt; j++) if (model.jnt_type[j] === 0 && model.jnt_qposadr[j] === D.freeQpos) { DUCKROOT = model.jnt_bodyid[j]; break; }
const underDuck = b => { let c = b; for (let i = 0; i < 64 && c > 0; i++) { if (c === DUCKROOT) return true; c = model.body_parentid[c]; } return c === DUCKROOT; };
export const DUCKG = [];
for (let g = 0; g < model.ngeom; g++) {
  if (model.geom_contype[g] === 0 && model.geom_conaffinity[g] === 0) continue;
  if (DUCKROOT >= 0 && underDuck(model.geom_bodyid[g])) DUCKG.push(g);
}
if (!DUCKG.length) { for (const g of JAW) DUCKG.push(g); for (const g of FEET) DUCKG.push(g); }

/**
 * A foot RESTING on the tread: past the riser line, inside the flight, within
 * 5 mm below to 45 mm above the tread's height, AND within 3 mm of a step geom.
 *
 * The clause used to have no ceiling and no contact test, so a foot in mid-air
 * above the step counted (audit_r2 found it; across its 64 rows the tightened
 * clause never disagreed with the loose one, so no verdict moves — it is
 * closed before a future search learns to exploit it).
 */
function footResting(g, h) {
  const x = data.geom_xpos[g * 3], y = data.geom_xpos[g * 3 + 1], z = data.geom_xpos[g * 3 + 2];
  if (!(z > h - 0.005 && z < h + 0.045 && x > RISER_X && Math.abs(y - STAIR_Y) <= LATERAL)) return false;
  for (const sg of STEPG) if (mj.mj_geomDistance(model, data, g, sg, 0.05, null) < 0.003) return true;
  return false;
}

const quat = () => [data.qpos[D.freeQpos + 3], data.qpos[D.freeQpos + 4], data.qpos[D.freeQpos + 5], data.qpos[D.freeQpos + 6]];

/** climb_lib.mjs:80-86, verbatim. */
export function poseAt(tr, time) {
  if (time <= 0) return HOME.slice();
  let pt = 0, pp = HOME;
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
 * ROUND 4, FAMILY B — THE HANDOFF FIELDS (additive, and off unless asked for).
 *
 * Family B splits the 80-120 mm band into two beats and has to start beat 2
 * from the state beat 1 ended in. `spawn:{x,y,z}` carries only the trunk's
 * position, so four optional fields carry the rest:
 *
 *   spawnQuat        [4]   trunk orientation (default: the identity rig3 sets)
 *   spawnPose        [14]  joint qpos AND the initial ctrl (default: HOME)
 *   spawnVel  {free:[6], joint:[14]}  the velocities (default: zero)
 *   spawnLastAction  [14]  the policy's own last-action term in the
 *                          observation (default: zeros, as before)
 *   settleTicks      int   ticks of the stand policy run BEFORE the track
 *                          (default 25 — climb_lib's own settle). Beat 2 uses
 *                          0, because a settle would erase the handoff.
 *
 * EVERY ONE of them is absent from every file written before this round, and
 * when a field is absent not one line of it executes, so the episode is the
 * episode rig3 already ran. climb/famB_parity.log proves that at full float
 * digits against climb/rig3_prefamB.mjs (a byte copy of this file as it stood,
 * differing only in its isMain guard string).
 *
 * A spawn handoff is NOT a climb. It reproduces qpos, qvel and the last-action
 * term; it does not reproduce the fact that the duck GOT there. Any result
 * that only works from a handoff spawn is reported as a beat-2 result.
 */

/** The complete state a beat-2 spawn needs, read off the live sim. Read-only. */
function handoffNow(h) {
  const jp = [], jv = [], free = [];
  for (let k = 0; k < 14; k++) { jp.push(data.qpos[D.qpos[k]]); jv.push(data.qvel[D.dof[k]]); }
  for (let k = 0; k < 6; k++) free.push(data.qvel[D.freeDof + k]);
  let head = false;
  for (const g of JAW) if (mj.mj_geomDistance(model, data, g, STEP0, 0.05, null) < 0.003) { head = true; break; }
  let footRiser = false, feetOnTread = 0;
  for (const g of [LFOOT, RFOOT]) {
    if (data.geom_xpos[g * 3 + 2] < h - 0.005 && mj.mj_geomDistance(model, data, g, STEP0, 0.05, null) < 0.003) footRiser = true;
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

// ---------------------------------------------------------------- snapshots

/**
 * ROUND 4, HOLE 3. The most negative mj_geomDistance between ANY collidable
 * duck geom and ANY step geom, AT THIS INSTANT, with the pair named.
 * A duck standing 9 mm inside a block is not standing on it. This is read-only:
 * mj_geomDistance is a query, it touches neither qpos nor ctrl.
 */
function penetrationNow() {
  let pen = 1e9, pair = null;
  for (const g of DUCKG) for (const sg of STEPG) {
    const d = mj.mj_geomDistance(model, data, g, sg, 0.05, null);
    if (d < pen) { pen = d; pair = `${model.geom(g).name || 'g' + g}<->${model.geom(sg).name}`; }
  }
  return { pen: pen === 1e9 ? null : pen, pair };
}

/**
 * ROUND 5, THE NEW HOLE. penetrationNow() is read AT THE SCORED INSTANT ONLY.
 * Round 3 measured -9.4 mm TRANSIENTLY, so a move that passes THROUGH a block
 * on the way up and arrives clean scores clean. This is the same query run at
 * EVERY control tick of the episode — settle, track and tail — and kept as a
 * running minimum, so `minPenetrationEpisode` is the deepest the duck was ever
 * inside the flight, not the deepest it was inside it when someone looked.
 *
 * It is exact, not sampled. The bounding-sphere test below is a LOWER BOUND on
 * mj_geomDistance (centre separation minus the two rbounds), so skipping a
 * pair whose lower bound already exceeds the running minimum cannot skip a
 * pair that would have improved it. Every reported number is a real
 * mj_geomDistance. Like penetrationNow() it is a query: it touches neither
 * qpos nor ctrl, which is why the 86-row and 225-row parity sets are exact.
 */
const RBOUND = model.geom_rbound;
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
          const d = mj.mj_geomDistance(model, data, g, sg, 0.05, null);
          if (d < best) { best = d; pair = `${model.geom(g).name || 'g' + g}<->${model.geom(sg).name}`; tickAt = tick; }
        }
      }
    },
    get() { return { min: best === 1e9 ? null : best, pair, tick: tickAt, ticksScanned: ticks }; },
  };
}

/**
 * Everything the criterion could possibly want, read off the live state.
 *
 * ROUND 4: `maxAbsDY` (the WHOLE-EPISODE worst lateral excursion so far) and
 * `penetrationAtScore` are threaded in here so criteria() and every consumer
 * see them as ordinary snapshot fields. maxAbsDY is undefined only when a
 * caller predates round 4; criteria() then falls back to the point gate and
 * says so in `lateralSource`.
 */
function snapshot(h, maxAbsDY) {
  const x = data.qpos[D.freeQpos], y = data.qpos[D.freeQpos + 1], z = data.qpos[D.freeQpos + 2];
  const up = projectedGravity(quat())[2] < -0.90;
  // climb_lib.mjs:141-145 exactly as written — no y term, foot x > 0.05
  let feetUpRaw = 0;
  for (const g of FEET) if (data.geom_xpos[g * 3 + 2] > h - 0.005 && data.geom_xpos[g * 3] > 0.05) feetUpRaw++;
  // the same test with the lateral gate added
  let feetUpLat = 0;
  for (const g of FEET) if (data.geom_xpos[g * 3 + 2] > h - 0.005 && data.geom_xpos[g * 3] > 0.05
    && Math.abs(data.geom_xpos[g * 3 + 1] - STAIR_Y) <= LATERAL) feetUpLat++;
  // and with the foot-x bug fixed: a foot on the tread is PAST the riser
  let feetOnTread = 0;
  for (const g of FEET) if (footResting(g, h)) feetOnTread++;
  const foot = g => ({ x: data.geom_xpos[g * 3], y: data.geom_xpos[g * 3 + 1], z: data.geom_xpos[g * 3 + 2] });
  const P = penetrationNow();
  return {
    x, y, z, dy: y - STAIR_Y, above: z - h, up,
    feetUpRaw, feetUpLat, feetOnTread,
    lfoot: foot(LFOOT), rfoot: foot(RFOOT),
    maxAbsDY,                                   // whole-episode, threaded in
    penetrationAtScore: P.pen, penetrationPair: P.pair,
  };
}

/**
 * Every criterion under consideration, evaluated on one snapshot.
 *
 *  orig    sim/climb_lib.mjs:150, unchanged.
 *  lat     orig + the lateral gate on trunk and feet. Still uses foot x > 0.05.
 *  honest  lat, plus a foot only counts as on the tread if it is past the riser
 *          face (x > 0.12) — the same line the trunk has to cross.
 *  honest60  honest with the height clause relaxed to 60 mm above the tread
 *          (a deep crouch on the tread rather than a full stand).
 */
export function criteria(h, s, maxAbsDY) {
  // ROUND 4, HOLE 1. reward() has always applied the lateral gate over the
  // WHOLE episode; criteria() applied it only at the scored instant, so a move
  // could swing 426 mm off the side of the 340 mm flight, come back, and score
  // `honest`. The gate is now whole-episode HERE TOO. The excursion arrives
  // either on the snapshot (snapshot(h, maxAbsDY)) or as the third argument;
  // when neither is present the point gate is used and `lateralSource` says so,
  // which is the only way a pre-round-4 caller can silently differ.
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
 * flight scores 0 no matter what its x is. That is the fix for the single best
 * "trunk past the riser" number in round 1, which was bought by walking
 * 426 mm off the side.
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

// ---------------------------------------------------------------- the episode

/**
 * One episode. Identical to sim/climb_lib.mjs attempt() up to the tail.
 * INTERNAL — nothing outside this file may score an in-memory track; see
 * scoreSaved().
 */
async function runEpisodeRaw(track, opts, h, tail) {
  const cfg = { count: opts.stepCount || 4, rise: h, run: 0.28, start: 0.12 };
  mj.mj_resetData(model, data);
  layoutStairs(data, ADDR, cfg);
  if (opts.spawn) {
    data.qpos[D.freeQpos] = opts.spawn.x;
    data.qpos[D.freeQpos + 1] = opts.spawn.y;
    data.qpos[D.freeQpos + 2] = opts.spawn.z;
  } else {
    data.qpos[D.freeQpos] = 0.12 - 0.07 - opts.gap;
    data.qpos[D.freeQpos + 1] = STAIR_Y + (opts.side || 0);
    data.qpos[D.freeQpos + 2] = 0.12;
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
  const cmd = command({ vx: opts.approach });

  const R = { ticks: 0, headTicks: 0, riserTicks: 0, upTicks: 0, sat: 0, ctrls: 0,
              maxX: -1e9, maxZ: -1e9, maxAbsDY: 0, feetOnTreadMax: 0, feetUpRawMax: 0,
              maxTreadSag_mm: 0, maxTreadDriftX_mm: 0, minStepGap_mm: 1e9, trace: [],
              // --- ADDITIVE INSTRUMENTATION (round 2, family C). Read-only:
              // it records, it never touches ctrl, qpos, the criterion or any
              // pre-existing field, so parity and every published number are
              // unchanged. It exists because "sustained LOAD TRANSFER" — trunk
              // z gain while the head AND a foot are both bearing — is the
              // reward round 2 needs, and contact fractions cannot express it.
              bothTicks: 0, maxGainBoth: -1e9, sustainTicks: 0, liftIntegral: 0 };
  let gtick = 0;
  const PEN = makePenTracker();          // ROUND 5: whole-episode penetration
  let Z0 = 0;                       // trunk z at the end of the 25-tick settle
  /**
   * How far the tread has moved by the end of a control tick.
   *
   * site/stairs.js says of pin(): "Call after any qpos write, AND EVERY TICK."
   * climb_lib.mjs:132 calls layoutStairs once and then takes FOUR mj_steps.
   * A step is a 200 kg box (sim/scene_physics.xml:91) on two frictionless,
   * undamped slide joints with no gravity compensation, so between pins it is
   * in free fall: 0.02 s of it is 1.96 mm of drop and 0.196 m/s of downward
   * surface velocity, re-teleported to nominal at the next control tick.
   */
  const treadDrift = () => {
    const topNow = data.geom_xpos[STEP0 * 3 + 2] + 0.10;
    const sag = (h - topNow) * 1000;
    if (sag > R.maxTreadSag_mm) R.maxTreadSag_mm = sag;
    const dx = Math.abs(data.geom_xpos[STEP0 * 3] - (0.12 + 0.17)) * 1000;
    if (dx > R.maxTreadDriftX_mm) R.maxTreadDriftX_mm = dx;
    // Consecutive steps OVERLAP by design (STEP_HALF_DEPTH 0.17 > run/2 = 0.14,
    // "so steps overlap into one solid flight" — site/stairs.js:38) and they
    // share contype/conaffinity 4 (sim/scene_physics.xml:89,96), so they
    // COLLIDE WITH EACH OTHER. Box-box normals point along the axis of least
    // penetration: x-overlap is 60 mm, z-overlap is (0.2 - rise), so below a
    // rise of 140 mm the flight shoves itself apart HORIZONTALLY.
    if (STEP1 >= 0 && cfg.count > 1) {
      const g = mj.mj_geomDistance(model, data, STEP0, STEP1, 0.4, null) * 1000;
      if (g < R.minStepGap_mm) R.minStepGap_mm = g;
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
      treadSag_mm: +treadDrift().toFixed(2),
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
    for (const g of JAW) if (mj.mj_geomDistance(model, data, g, STEP0, 0.05, null) < 0.003) { head = true; break; }
    if (head) R.headTicks++;
    let footRiser = false;
    for (const g of [LFOOT, RFOOT]) {
      if (data.geom_xpos[g * 3 + 2] >= h - 0.005) continue;
      if (mj.mj_geomDistance(model, data, g, STEP0, 0.05, null) < 0.003) { footRiser = true; break; }
    }
    if (footRiser) R.riserTicks++;
    let fot = 0, fur = 0;
    for (const g of FEET) {
      if (data.geom_xpos[g * 3 + 2] > h - 0.005 && data.geom_xpos[g * 3] > 0.05) fur++;
      if (footResting(g, h)) fot++;
    }
    if (fot > R.feetOnTreadMax) R.feetOnTreadMax = fot;
    if (fur > R.feetUpRawMax) R.feetUpRawMax = fur;
    // load transfer: is the duck rising while the head and a foot both bear?
    if (head && (footRiser || fot > 0)) {
      R.bothTicks++;
      const g = z - Z0;
      if (g > R.maxGainBoth) R.maxGainBoth = g;
      if (g > 0.02) R.sustainTicks++;
      if (g > 0) R.liftIntegral += g;
    }
  };

  // climb_lib.mjs:121-133, verbatim
  // ROUND 5: `sv` is the servoed-landing target vector for this tick — a
  // number for every LEG slot the law owns, null everywhere else. It is
  // undefined on every tick of every file that carries no `servo` block, and
  // then the added condition is `undefined && ...` and not one number moves.
  const step = async (off, rec, sv) => {
    layoutStairs(data, ADDR, cfg);
    const q = quat(); const jp = [], jv = [];
    for (let k = 0; k < 14; k++) { jp.push(data.qpos[D.qpos[k]]); jv.push(data.qvel[D.dof[k]]); }
    const obs = buildObs([data.sensordata[GYRO], data.sensordata[GYRO + 1], data.sensordata[GYRO + 2]], projectedGravity(q), jp, jv, la, cmd);
    const r = await stand.run({ obs: new ort.Tensor('float32', obs, [1, 61]) });
    la = Array.from(r.actions.data);
    for (let k = 0; k < 14; k++) {
      const v = (sv && sv[k] !== null) ? sv[k]
              : HOME[k] + la[k] + (off ? (off[k] - HOME[k]) * opts.blend : 0);
      const c = Math.min(Math.max(v, LO[k]), HI[k]);
      data.ctrl[k] = c;
      if (rec) { R.ctrls++; if (c <= LO[k] + 1e-9 || c >= HI[k] - 1e-9) R.sat++; }
    }
    for (let s = 0; s < 4; s++) { if (opts.pinEverySubstep) layoutStairs(data, ADDR, cfg); mj.mj_step(model, data); }
    treadDrift(); traceSample(rec ? 'track' : 'settle'); PEN.scan(gtick); gtick++;
    if (rec) record();
  };

  /** No policy at all: the servos hold the targets they are given. */
  const holdStep = (targets) => {
    layoutStairs(data, ADDR, cfg);
    for (let k = 0; k < 14; k++) data.ctrl[k] = targets[k];
    for (let s = 0; s < 4; s++) { if (opts.pinEverySubstep) layoutStairs(data, ADDR, cfg); mj.mj_step(model, data); }
    treadDrift(); traceSample('tail'); PEN.scan(gtick); gtick++;
    record();
  };

  const SETTLE = (opts.settleTicks === undefined || opts.settleTicks === null) ? 25 : opts.settleTicks;
  for (let t = 0; t < SETTLE; t++) await step(null, false);
  const x0 = data.qpos[D.freeQpos];
  Z0 = data.qpos[D.freeQpos + 2];
  // ================================================== ROUND 4, FAMILY A
  // AN OPTIONAL EVENT-TRIGGERED TAIL. `opts.event` is absent in every file
  // written before round 4, and normEvent(undefined) is null, in which case
  // TR === tr, `total` never changes, and the two lines below are the
  // pre-round-4 loop verbatim: same tick count, same poses, same ONNX calls.
  // climb/famA_r4.mjs PHASE P proves that on every existing best_* file.
  const EV = normEvent(opts.event);
  let TR = tr;
  let total = TR[TR.length - 1].t + 0.8;
  let evFired = false, evT = null, evE = null, evTrunkX = null;
  const beakDistNow = () => {
    let d = 1e9;
    for (const g of JAW) for (const sg of STEPG) { const v = mj.mj_geomDistance(model, data, g, sg, 0.05, null); if (v < d) d = v; }
    return d === 1e9 ? null : d;
  };
  // ================================================== ROUND 5, THE SERVO
  // An OPTIONAL per-tick feedback law for the LEG slots (climb/servo.mjs).
  // `opts.servo` is absent in every file written before round 5, normServo()
  // returns null for it, and then SV is null: svArmed never becomes true,
  // svTargets stays undefined, and `await step(poseAt(TR, time), true,
  // svTargets)` is the pre-round-5 call with a trailing undefined.
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
        TR = buildDynTrack(tr, EV, time, poseAt(TR, time), evE);
        total = TR[TR.length - 1].t + 0.8;
      }
    }
    let svTargets;
    if (SV) {
      if (!svArmed && ((SV.at !== null && time >= SV.at) || (SV.onEvent && evFired))) {
        svArmed = true; svT = time; svBase = servoBase(SV, poseAt(TR, time));
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
    await step(poseAt(TR, time), true, svTargets);
  }

  const atTrackEnd = snapshot(h, R.maxAbsDY);
  // ROUND 4, FAMILY B: the handoff point. This instant IS where a beat-2 track
  // is concatenated (beat 1's last keyframe + 0.8 s), so terminal is exactly the
  // state a two-beat concatenation would hand over.
  const terminal = handoffNow(h); terminal.spawnLastAction = la.slice();
  // what the servos were last told, and what 'hold' will freeze them at
  const ctrlAtHandoff = []; for (let k = 0; k < 14; k++) ctrlAtHandoff.push(data.ctrl[k]);
  const finalPose = poseAt(TR, total);   // TR === tr when the file has no event
  const held = finalPose.map((v, k) => Math.min(Math.max(v, LO[k]), HI[k]));
  // ROUND 5: if the servo took the legs, the 'hold' tail freezes at what the
  // SERVO last commanded, not at a keyframe the legs stopped following. A file
  // with no servo never enters this line.
  if (svArmed && svLastCtrl) for (let k = 0; k < 14; k++) if (svLastCtrl[k] !== null) held[k] = svLastCtrl[k];
  let ctrlJump = 0; for (let k = 0; k < 14; k++) ctrlJump = Math.max(ctrlJump, Math.abs(held[k] - ctrlAtHandoff[k]));

  // ROUND 4, HOLE 2. Count how many of the 50 TAIL ticks the duck is upright.
  // The objective had no term for "still standing at the end", so a candidate
  // that reaches the tread and topples through the tail could outrank one that
  // stands. upTicks is cumulative, so the tail's share is the difference.
  const upBeforeTail = R.upTicks, ticksBeforeTail = R.ticks;
  let afterTail;
  // ROUND 6, THE TAIL. Two additive things, both inert unless asked for:
  //
  //  (a) servo.tailTicks (default 0). For the first n ticks of the POLICY tail
  //      the same law keeps commanding the leg slots from the same live
  //      readings. With n = 0 the loop below is `await step(null, true,
  //      undefined)`, which is the pre-round-6 call verbatim.
  //  (b) opts.tailTrace (default false). A read-only per-tail-tick record of
  //      what the duck was doing: projected gravity, trunk, both feet, the 14
  //      commands, the 14 measured joint angles, and the saturation count. It
  //      is sampled AFTER the tick's four substeps and touches nothing.
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
  if (tail === 'policy') {
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
  } else {
    for (let t = 0; t < 50; t++) holdStep(held);
    afterTail = snapshot(h, R.maxAbsDY);
  }
  const uprightTailTicks = R.upTicks - upBeforeTail;
  const tailTicks = R.ticks - ticksBeforeTail;

  const scored = (tail === 'none') ? atTrackEnd : afterTail;
  const rec = {
    tail, rise: h, x0, ctrlJump,
    event: EV ? { type: EV.type, fired: evFired, tFire: evT, trunkXAtFire: evTrunkX, e_mm: evE === null ? null : +(evE * 1000).toFixed(2) } : null,
    scored, atTrackEnd, afterTail,
    // ROUND 4 first-class fields of every scored row
    penetrationAtScore: scored.penetrationAtScore,
    penetrationPair: scored.penetrationPair,
    // ROUND 5, THE NEW HOLE: the deepest the duck was EVER inside a step geom,
    // over every control tick of the episode (settle + track + tail), with the
    // pair and the tick named. penetrationAtScore is one instant; this is the
    // whole episode, so a move that passes THROUGH a block and arrives clean
    // can no longer score clean.
    minPenetrationEpisode: PEN.get().min,
    minPenetrationPair: PEN.get().pair,
    minPenetrationTick: PEN.get().tick,
    penetrationTicksScanned: PEN.get().ticksScanned,
    // ROUND 5, THE SERVOED LANDING. null for a file with no `servo` block.
    servo: SV ? { armed: svArmed, tArm: svT, ticks: svTicks,
                  at: SV.at, onEvent: SV.onEvent,
                  // ROUND 6: authority asked for, and ticks of it actually run.
                  tailAuthority: SV.tailTicks, tailTicksRun: svTailRun,
                  base: svBase, trace: opts.servoTrace ? svLog : undefined } : null,
    // ROUND 6: read-only per-tail-tick record; undefined unless asked for.
    tailTrace: opts.tailTrace ? tailLog : undefined,
    uprightTailTicks, tailTicks,
    uprightTailFrac: uprightTailTicks / Math.max(tailTicks, 1),
    terminal,          // ROUND 4, FAMILY B: the beat-1 -> beat-2 handoff state
    crit: criteria(h, scored),
    critAtTrackEnd: criteria(h, atTrackEnd),
    critAfterTail: criteria(h, afterTail),
    maxX: R.maxX, maxZ: R.maxZ, maxAbsDY: R.maxAbsDY,
    feetOnTreadMax: R.feetOnTreadMax, feetUpRawMax: R.feetUpRawMax,
    maxTreadSag_mm: R.maxTreadSag_mm, maxTreadDriftX_mm: R.maxTreadDriftX_mm,
    minStepGap_mm: R.minStepGap_mm === 1e9 ? null : R.minStepGap_mm, trace: R.trace,
    headFrac: R.headTicks / Math.max(R.ticks, 1),
    riserFrac: R.riserTicks / Math.max(R.ticks, 1),
    upFrac: R.upTicks / Math.max(R.ticks, 1),
    satFrac: R.sat / Math.max(R.ctrls, 1),
    // additive instrumentation, see R above
    z0Settle: Z0,
    bothFrac: R.bothTicks / Math.max(R.ticks, 1),
    maxGainBoth: R.maxGainBoth === -1e9 ? null : R.maxGainBoth,
    sustainFrac: R.sustainTicks / Math.max(R.ticks, 1),
    liftIntegral: R.liftIntegral,
  };
  // climb_lib's own return, for the parity check
  rec.legacy = { onTop: rec.crit.orig, x: scored.x, z: scored.z, above: scored.above,
                 feetUp: scored.feetUpRaw, up: scored.up };
  rec.reward = reward(rec);
  return rec;
}

// ---------------------------------------------------------------- public API

/**
 * THE ONLY SCORER. It takes a PATH, not a track.
 *
 * The intent shape on disk is { keyframes:[{t,pose[14]}], blend, gap, side,
 * approach, spawn? }. Everything scored in this file goes through here, so
 * every number reported is a number the saved file actually produces —
 * candidate-vs-export drift cannot hide.
 */
export async function scoreSaved(path, { rise, tail = 'policy', gapOffset = 0, overrides = {} } = {}) {
  const j = JSON.parse(fs.readFileSync(path, 'utf8'));
  if (!Array.isArray(j.keyframes) || !j.keyframes.length) throw new Error('no keyframes in ' + path);
  for (const f of j.keyframes) if (!Array.isArray(f.pose) || f.pose.length !== 14) throw new Error('bad pose in ' + path);
  const opts = {
    blend: j.blend, approach: j.approach || 0,
    gap: (j.gap || 0) + gapOffset, side: j.side || 0,
    spawn: j.spawn || null,
    // ROUND 4, FAMILY A: the optional event-triggered tail (climb/event.mjs).
    event: j.event || null,
    // ROUND 5: the optional servoed landing (climb/servo.mjs). Absent in every
    // file written before round 5 -> null -> not one line of the law runs.
    servo: j.servo || null,
    // ROUND 4, FAMILY B handoff fields — null/undefined for every older file
    spawnQuat: j.spawnQuat || null, spawnPose: j.spawnPose || null,
    spawnVel: j.spawnVel || null, spawnLastAction: j.spawnLastAction || null,
    settleTicks: j.settleTicks === undefined ? undefined : j.settleTicks,
    ...overrides,
  };
  const rec = await runEpisodeRaw(j.keyframes, opts, rise, tail);
  rec.source = path; rec.opts = { ...opts, gapOffset };
  return rec;
}

/** Author a track, round-trip it through JSON on disk, hand back the path. */
export function saveTrack(obj, path) {
  fs.writeFileSync(path, JSON.stringify(obj, null, 2));
  return path;
}

/** Parity only. Not a scorer — it takes an in-memory track on purpose. */
export async function __parityEpisode(track, opts, h, tail) { return runEpisodeRaw(track, opts, h, tail); }

// ================================================================== EXPERIMENT

const isMain = process.argv[1] && process.argv[1].endsWith('rig3.mjs');
if (!isMain) { /* imported as a library */ } else {

const OUT = '../climb/rig3-results.json';
const RISES = [0.020, 0.040, 0.060, 0.090, 0.120, 0.180];
const TAILS = ['policy', 'hold', 'none'];
const mm = v => (v * 1000).toFixed(1);
const results = { generated: new Date().toISOString(), plant: 'scene.mjb', policy: 'BEST_alpha_stand.onnx',
                  lateralGate_m: LATERAL, riserX_m: RISER_X };
const t00 = Date.now();

// ---------------------------------------------------------------- PHASE P
// Parity: is this the same episode as sim/climb_lib.mjs attempt()?
// audit_cross.mjs's method — same track, same opts, same rise, full digits.
console.log('=== PHASE P — parity against sim/climb_lib.mjs (tail=policy, criterion=orig) ===');
const { replay: libReplay } = await import('../sim/climb_lib.mjs');
const parityCases = [];
{
  const a = HOME.slice(); a[5] = -1.3; a[6] = 0.7; a[7] = 1.4;
  parityCases.push({ label: 'synthetic (climb/parity.mjs track)', h: 0.04,
    track: [{ t: 0.5, pose: a }, { t: 1.6, pose: HOME.slice() }],
    opts: { blend: 1.6, approach: 0, gap: 0.03, side: 0.06 } });
}
for (const f of ['best_0_40mm.json', 'best_1_90mm.json', 'best_2_180mm.json']) {
  const j = JSON.parse(fs.readFileSync('../climb/' + f, 'utf8'));
  parityCases.push({ label: f, h: parseInt(f.match(/_(\d+)mm/)[1], 10) / 1000,
    track: j.keyframes, opts: { blend: j.blend, approach: j.approach || 0, gap: j.gap, side: j.side } });
}
results.parity = [];
let parityAll = true;
for (const c of parityCases) {
  const A = await libReplay(c.track, c.opts, c.h);
  const B = await __parityEpisode(c.track, c.opts, c.h, 'policy');
  const L = B.legacy;
  const match = A.onTop === L.onTop && A.x === L.x && A.z === L.z && A.feetUp === L.feetUp && A.up === L.up;
  parityAll = parityAll && match;
  results.parity.push({ case: c.label, rise_mm: c.h * 1000,
    climb_lib: { onTop: A.onTop, x: A.x, z: A.z, feetUp: A.feetUp, up: A.up },
    rig3: { onTop: L.onTop, x: L.x, z: L.z, feetUp: L.feetUp, up: L.up }, exactMatch: match });
  console.log(`  ${c.label.padEnd(34)} rise ${(c.h*1000).toString().padStart(3)}  ` +
    `climb_lib x=${A.x} z=${A.z} feetUp=${A.feetUp}\n  ${' '.repeat(34)}      ` +
    `rig3      x=${L.x} z=${L.z} feetUp=${L.feetUp}   EXACT=${match}`);
}
results.parityAll = parityAll;
console.log(`  PARITY ${parityAll ? 'PASS — rig3 is climb_lib' : 'FAIL'} (${results.parity.length} cases, full float digits)\n`);

// ---------------------------------------------------------------- PHASE R
// Round-trip: what does the export's rounding cost? (search_0 pose->4 dp,
// search_2 pose->5 dp; the notes quote the unrounded candidate.)
console.log('=== PHASE R — export rounding sensitivity (score the file, never the candidate) ===');
results.roundTrip = [];
for (const f of ['best_0_40mm.json', 'best_2_40mm.json', 'best_2_180mm.json']) {
  const h = parseInt(f.match(/_(\d+)mm/)[1], 10) / 1000;
  const asSaved = await scoreSaved('../climb/' + f, { rise: h, tail: 'policy' });
  const j = JSON.parse(fs.readFileSync('../climb/' + f, 'utf8'));
  j.keyframes = j.keyframes.map(k => ({ t: k.t, pose: k.pose.map(v => +v.toFixed(3)) }));
  const p = saveTrack(j, '/tmp/rig3_rt.json');
  const rounded = await scoreSaved(p, { rise: h, tail: 'policy' });
  const row = { file: f, dx_mm: +mm(rounded.scored.x - asSaved.scored.x), dz_mm: +mm(rounded.scored.z - asSaved.scored.z),
    saved: { x_mm: +mm(asSaved.scored.x), z_mm: +mm(asSaved.scored.z) },
    rounded3dp: { x_mm: +mm(rounded.scored.x), z_mm: +mm(rounded.scored.z) } };
  results.roundTrip.push(row);
  console.log(`  ${f.padEnd(20)} saved x=${mm(asSaved.scored.x).padStart(8)} z=${mm(asSaved.scored.z).padStart(7)}  ` +
    `| pose re-rounded to 3 dp x=${mm(rounded.scored.x).padStart(8)} z=${mm(rounded.scored.z).padStart(7)}  ` +
    `| moved ${Math.abs(row.dx_mm).toFixed(1)} mm in x, ${Math.abs(row.dz_mm).toFixed(1)} mm in z`);
}
console.log();

// ---------------------------------------------------------------- controls
const HOLD_TRACK = (pose) => [{ t: 1.0, pose: pose.slice() }, { t: 2.9, pose: pose.slice() }];

// (a) do-nothing: HOME held, no forward command. gap 0.05 -> spawn x = 0.000.
saveTrack({ name: 'control_do_nothing', keyframes: HOLD_TRACK(HOME), blend: 1, gap: 0.05, side: 0, approach: 0,
  note: 'DO-NOTHING CONTROL. HOME pose for the whole track, vx=0. A criterion that this row PASSES is not a climbing test.' },
  '../climb/ctrl_do_nothing.json');
// (a2) walk-only: HOME held, but the policy is commanded forward. The stronger
// control: if walking alone clears a rise, that rise is not a climbing test.
saveTrack({ name: 'control_walk_only', keyframes: HOLD_TRACK(HOME), blend: 1, gap: 0.05, side: 0, approach: 0.30,
  note: 'WALK-ONLY CONTROL. HOME pose, vx=0.30 — the stand policy walks at the flight with no authored motion at all.' },
  '../climb/ctrl_walk_only.json');
// short episode, for a length-matched pair of controls
const SHORT_TRACK = (pose) => [{ t: 0.2, pose: pose.slice() }, { t: 0.5, pose: pose.slice() }];
saveTrack({ name: 'control_do_nothing_short', keyframes: SHORT_TRACK(HOME), blend: 1, gap: 0.05, side: 0, approach: 0,
  note: 'DO-NOTHING CONTROL, short episode (1.3 s track). Length-matched partner of the short on-tread control.' },
  '../climb/ctrl_do_nothing_short.json');
// (c) the row a criterion MUST pass: already standing on the first tread.
// tread 0 spans x in [0.12, 0.46]; step 1 covers from x=0.40, so the EXPOSED
// tread of step 0 is x in [0.12, 0.40] — 280 mm of it. Spawn at three places
// along it and at two episode lengths, because the first attempt (one place,
// one length) confounds "the criterion cannot express standing on a tread"
// with "this particular placement walks off the front edge".
const ON_TREAD_X = [0.20, 0.26, 0.32];
const ON_TREAD_LEN = { short: SHORT_TRACK, med: HOLD_TRACK };
const onTreadFiles = [];
for (const rmm of [20, 40, 60, 90, 120, 180]) {
  for (const sx of ON_TREAD_X) for (const [ln, mk] of Object.entries(ON_TREAD_LEN)) {
    const p = `../climb/ctrl_on_tread_${rmm}mm_x${Math.round(sx*100)}_${ln}.json`;
    saveTrack({ name: `control_on_tread_${rmm}mm_x${Math.round(sx*100)}_${ln}`, keyframes: mk(HOME),
      blend: 1, gap: 0.05, side: 0, approach: 0,
      spawn: { x: sx, y: STAIR_Y, z: rmm / 1000 + 0.12 },
      note: `ON-TREAD CONTROL. Duck spawned standing on the first tread of a ${rmm} mm flight at x=${sx} `
          + `(exposed tread is x in [0.12, 0.40]); trunk 0.12 m above the tread, the same spawn height `
          + `attempt() uses above the floor. HOME pose, vx=0, ${ln} episode. `
          + `A criterion that NO placement of this row passes cannot express success at all.` }, p);
    onTreadFiles.push({ rmm, sx, ln, path: p });
  }
}

/** Run one saved file at one rise under all three tails (2 simulations). */
async function allTails(path, rise, gapOffset = 0) {
  const pol = await scoreSaved(path, { rise, tail: 'policy', gapOffset });
  const hld = await scoreSaved(path, { rise, tail: 'hold', gapOffset });
  const non = { ...hld, tail: 'none', scored: hld.atTrackEnd, crit: hld.critAtTrackEnd,
                penetrationAtScore: hld.atTrackEnd.penetrationAtScore,
                penetrationPair: hld.atTrackEnd.penetrationPair,
                stillUpAfter50Hold: hld.afterTail.up, stillPassesAfter50Hold: hld.critAfterTail };
  non.reward = reward(non);
  return { policy: pol, hold: hld, none: non };
}

const hdr = 'row                         rise  tail    x_mm     y-Y_mm    z_mm  above_mm  feetRaw feetLat feetTread up  ORIG  LAT  HONEST  H60';
const fmtRow = (label, rise, tail, r) => {
  const s = r.scored, c = r.crit;
  return `${label.padEnd(26)} ${(rise*1000).toString().padStart(4)}  ${tail.padEnd(6)} ` +
    `${mm(s.x).padStart(7)} ${mm(s.dy).padStart(8)} ${mm(s.z).padStart(7)} ${mm(s.above).padStart(8)}  ` +
    `${String(s.feetUpRaw).padStart(6)} ${String(s.feetUpLat).padStart(6)} ${String(s.feetOnTread).padStart(8)}  ` +
    `${s.up ? 'Y' : 'n'}  ${c.orig ? 'PASS' : ' .  '}  ${c.lat ? 'PASS' : ' . '}  ${c.honest ? ' PASS ' : '  .   '}  ${c.honest60 ? 'PASS' : ' . '}`;
};
const slim = r => ({ tail: r.tail, x_mm: +mm(r.scored.x), dy_mm: +mm(r.scored.dy), z_mm: +mm(r.scored.z),
  above_mm: +mm(r.scored.above), up: r.scored.up, feetUpRaw: r.scored.feetUpRaw, feetUpLat: r.scored.feetUpLat,
  feetOnTread: r.scored.feetOnTread, crit: r.crit, reward: +r.reward.toFixed(3),
  peakX_mm: +mm(r.maxX), peakZ_mm: +mm(r.maxZ), maxAbsDY_mm: +mm(r.maxAbsDY),
  feetOnTreadMax: r.feetOnTreadMax, feetUpRawMax: r.feetUpRawMax,
  headFrac: +r.headFrac.toFixed(3), riserFrac: +r.riserFrac.toFixed(3), ctrlJump: +r.ctrlJump.toFixed(4),
  penetrationAtScore_mm: r.penetrationAtScore === null ? null : +mm(r.penetrationAtScore),
  penetrationPair: r.penetrationPair,
  minPenetrationEpisode_mm: r.minPenetrationEpisode === null ? null : +mm(r.minPenetrationEpisode),
  minPenetrationPair: r.minPenetrationPair, minPenetrationTick: r.minPenetrationTick,
  uprightTailTicks: r.uprightTailTicks, tailTicks: r.tailTicks,
  stillUpAfter50Hold: r.stillUpAfter50Hold, stillPassesAfter50Hold: r.stillPassesAfter50Hold });

// ---------------------------------------------------------------- PHASE A
console.log('=== PHASE A — the do-nothing control. A criterion this row passes is not a climbing test. ===');
console.log(hdr);
results.controls = { doNothing: [], walkOnly: [] };
for (const h of RISES) {
  const t = await allTails('../climb/ctrl_do_nothing.json', h);
  for (const k of TAILS) { console.log(fmtRow('do-nothing (HOME, vx=0)', h, k, t[k])); results.controls.doNothing.push({ rise_mm: h*1000, ...slim(t[k]) }); }
}
console.log();
console.log('--- walk-only control: same HOME pose, but vx=0.30 (no authored motion at all) ---');
console.log(hdr);
for (const h of RISES) {
  const t = await allTails('../climb/ctrl_walk_only.json', h);
  for (const k of TAILS) { console.log(fmtRow('walk-only (HOME, vx=0.30)', h, k, t[k])); results.controls.walkOnly.push({ rise_mm: h*1000, ...slim(t[k]) }); }
}
console.log();

// ---------------------------------------------------------------- PHASE C
console.log('=== PHASE C — spawned ON the first tread, HOME pose. A criterion NO placement passes cannot express success. ===');
console.log('--- length-matched do-nothing (short episode) ---');
console.log(hdr);
results.controls.doNothingShort = [];
for (const h of RISES) {
  const t = await allTails('../climb/ctrl_do_nothing_short.json', h);
  for (const k of TAILS) { console.log(fmtRow('do-nothing short', h, k, t[k])); results.controls.doNothingShort.push({ rise_mm: h*1000, ...slim(t[k]) }); }
}
console.log();
console.log(hdr);
results.controls.onTread = [];
for (const c of onTreadFiles) {
  const t = await allTails(c.path, c.rmm / 1000);
  for (const k of TAILS) {
    console.log(fmtRow(`on-tread x=${c.sx.toFixed(2)} ${c.ln}`, c.rmm / 1000, k, t[k]));
    results.controls.onTread.push({ rise_mm: c.rmm, spawnX: c.sx, len: c.ln, ...slim(t[k]) });
  }
}
console.log('\n  best on-tread placement per rise (does ANY placement read as success?)');
console.log('  rise  tail    passes HONEST   best row');
for (const h of RISES) {
  for (const k of TAILS) {
    const rows = results.controls.onTread.filter(r => r.rise_mm === h * 1000 && r.tail === k);
    const win = rows.filter(r => r.crit.honest);
    const b = win[0] || rows.slice().sort((a, c) => c.above_mm - a.above_mm)[0];
    console.log(`  ${(h*1000).toString().padStart(4)}  ${k.padEnd(6)}  ${String(win.length).padStart(2)}/${rows.length}          ` +
      `x=${b.spawnX.toFixed(2)} ${b.len.padEnd(5)} x_mm=${b.x_mm.toFixed(1).padStart(7)} z_mm=${b.z_mm.toFixed(1).padStart(6)} ` +
      `above=${b.above_mm.toFixed(1).padStart(7)} feetTread=${b.feetOnTread} up=${b.up?'Y':'n'}`);
  }
}
console.log();

// ---------------------------------------------------------------- PHASE B
console.log('=== PHASE B — the 18 published bests, re-scored from the SAVED file, under each tail, at -10/0/+10 mm ===');
const bests = fs.readdirSync('../climb').filter(f => /^best_[012]_\d+mm\.json$/.test(f)).sort();
results.bests = [];
let anyClear = { orig: 0, lat: 0, honest: 0, honest60: 0 };
console.log('file                     rise off  tail    x_mm     y-Y_mm    z_mm  above_mm  feetRaw feetTread  up  ORIG  LAT  HONEST  H60   peakX  maxDY');
for (const f of bests) {
  const h = parseInt(f.match(/_(\d+)mm/)[1], 10) / 1000;
  for (const d of [-0.010, 0, 0.010]) {
    const t = await allTails('../climb/' + f, h, d);
    for (const k of TAILS) {
      const r = t[k], s = r.scored, c = r.crit;
      anyClear.orig += c.orig ? 1 : 0; anyClear.lat += c.lat ? 1 : 0;
      anyClear.honest += c.honest ? 1 : 0; anyClear.honest60 += c.honest60 ? 1 : 0;
      results.bests.push({ file: f, rise_mm: h * 1000, startOffset_mm: +(d * 1000).toFixed(0), ...slim(r) });
      console.log(`${f.padEnd(22)} ${(h*1000).toString().padStart(4)} ${(d*1000).toFixed(0).padStart(3)}  ${k.padEnd(6)} ` +
        `${mm(s.x).padStart(7)} ${mm(s.dy).padStart(8)} ${mm(s.z).padStart(7)} ${mm(s.above).padStart(8)}  ` +
        `${String(s.feetUpRaw).padStart(6)} ${String(s.feetOnTread).padStart(8)}   ${s.up?'Y':'n'}  ` +
        `${c.orig?'PASS':' .  '}  ${c.lat?'PASS':' . '}  ${c.honest?' PASS ':'  .   '}  ${c.honest60?'PASS':' . '}  ` +
        `${mm(r.maxX).padStart(7)} ${mm(r.maxAbsDY).padStart(6)}`);
    }
  }
}
results.bestsCleared = anyClear;
results.bestsRows = results.bests.length;
console.log(`\n  18 bests x 3 offsets x 3 tails = ${results.bests.length} scored rows.`);
console.log(`  cleared: ORIG ${anyClear.orig}   LAT ${anyClear.lat}   HONEST ${anyClear.honest}   HONEST60 ${anyClear.honest60}`);

// ---------------------------------------------------------------- PHASE D
// WHY does the on-tread control get thrown off at 20-120 mm but stand still at
// 180 mm? Two candidates: (i) the tread is in free fall between pins, so a
// duck standing on it rides a 50 Hz vibrating platform; (ii) something about
// the low-rise geometry. Measure the tread's motion, then re-run with the
// stairs pinned before EVERY mj_step — site/stairs.js's own stated contract
// ("call after any qpos write, and every tick"), which climb_lib.mjs:132 does
// not honour. The pinned variant is a DIAGNOSTIC, not the plant: it changes no
// physics constant, only how often the harness re-writes the stair joints.
console.log('=== PHASE D — is the tread standing still? (stairs are 200 kg boxes on frictionless slides) ===');
results.treadMotion = [];
console.log('rise  variant             tail    maxTreadSag_mm maxTreadDriftX_mm   x_mm    dy_mm   z_mm  feetTread  up  HONEST');
for (const h of [0.020, 0.040, 0.090, 0.120, 0.180]) {
  const rmm = Math.round(h * 1000);
  const path = `../climb/ctrl_on_tread_${rmm}mm_x26_med.json`;
  for (const [vlabel, ov] of [['plant (pin per ctrl tick)', {}], ['pinned per mj_step', { pinEverySubstep: true }]]) {
    for (const k of ['policy', 'none']) {
      const r = await scoreSaved(path, { rise: h, tail: k === 'none' ? 'hold' : 'policy',
        overrides: { ...ov, trace: true } });
      const rr = (k === 'none') ? { ...r, scored: r.atTrackEnd, crit: r.critAtTrackEnd } : r;
      const s = rr.scored, c = rr.crit;
      results.treadMotion.push({ rise_mm: rmm, variant: vlabel, tail: k,
        maxTreadSag_mm: +r.maxTreadSag_mm.toFixed(2), maxTreadDriftX_mm: +r.maxTreadDriftX_mm.toFixed(2),
        x_mm: +mm(s.x), dy_mm: +mm(s.dy), z_mm: +mm(s.z), above_mm: +mm(s.above),
        feetOnTread: s.feetOnTread, up: s.up, crit: c,
        trace: (vlabel.startsWith('plant') && k === 'none') ? r.trace.slice(0, 30) : undefined });
      console.log(`${rmm.toString().padStart(4)}  ${vlabel.padEnd(26)} ${k.padEnd(6)} ` +
        `${r.maxTreadSag_mm.toFixed(2).padStart(10)} ${r.maxTreadDriftX_mm.toFixed(2).padStart(14)}   ` +
        `${mm(s.x).padStart(7)} ${mm(s.dy).padStart(7)} ${mm(s.z).padStart(6)} ${String(s.feetOnTread).padStart(8)}   ` +
        `${s.up ? 'Y' : 'n'}  ${c.honest ? 'PASS' : ' . '}`);
    }
  }
}
console.log();

// ---------------------------------------------------------------- PHASE E
// The flight collides with itself. Consecutive step blocks overlap by 60 mm in
// x (STEP_HALF_DEPTH 0.17 vs run/2 = 0.14) and by (200 - rise) mm in z, and
// they share collision bits. A box-box normal points along the axis of LEAST
// penetration, so for rise < 140 mm the z-overlap is the larger one and the
// blocks push each other APART ALONG X — the tread lurches, is teleported back
// by the next layoutStairs, and lurches again, 50 times a second.
// Test: lay out ONE step instead of four (nothing to collide with) and see
// whether the on-tread control then stands. Diagnostic only — a 1-step flight
// is not the 4-step flight attempt() scores.
console.log('=== PHASE E — does the flight throw the duck off because the steps collide with each other? ===');
results.stepSelfCollision = [];
console.log('rise  steps  minStepGap_mm  maxTreadDriftX_mm  maxTreadSag_mm    x_mm    dy_mm   z_mm  feetTread  up  HONEST');
for (const h of RISES) {
  const rmm = Math.round(h * 1000);
  const path = `../climb/ctrl_on_tread_${rmm}mm_x26_med.json`;
  for (const n of [4, 1]) {
    const r = await scoreSaved(path, { rise: h, tail: 'policy', overrides: { stepCount: n } });
    const s = r.scored, c = r.crit;
    results.stepSelfCollision.push({ rise_mm: rmm, stepCount: n,
      minStepGap_mm: r.minStepGap_mm === null ? null : +r.minStepGap_mm.toFixed(2),
      maxTreadDriftX_mm: +r.maxTreadDriftX_mm.toFixed(2), maxTreadSag_mm: +r.maxTreadSag_mm.toFixed(2),
      x_mm: +mm(s.x), dy_mm: +mm(s.dy), z_mm: +mm(s.z), above_mm: +mm(s.above),
      feetOnTread: s.feetOnTread, up: s.up, crit: c });
    console.log(`${rmm.toString().padStart(4)}  ${String(n).padStart(5)}  ` +
      `${(r.minStepGap_mm === null ? 'n/a' : r.minStepGap_mm.toFixed(2)).padStart(13)}  ` +
      `${r.maxTreadDriftX_mm.toFixed(2).padStart(17)}  ${r.maxTreadSag_mm.toFixed(2).padStart(14)}   ` +
      `${mm(s.x).padStart(7)} ${mm(s.dy).padStart(7)} ${mm(s.z).padStart(6)} ${String(s.feetOnTread).padStart(8)}   ` +
      `${s.up ? 'Y' : 'n'}  ${c.honest ? 'PASS' : ' . '}`);
  }
}
// Where is the threshold? Predicted at rise = 200 - 60 = 140 mm, the rise at
// which the z-overlap between consecutive blocks stops being the deeper one.
console.log('\n  threshold sweep — the rise at which a duck can stand on its own tread (4 steps, plant as-is)');
console.log('  rise steps z-ovl  minStepGap  driftX   sag     x_mm   z_mm  above  lfootZ-h  rfootZ-h  feetTread up HONEST');
results.thresholdSweep = [];
for (const rmm of [20, 40, 60, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]) {
  const h = rmm / 1000;
  const p = saveTrack({ name: `control_on_tread_${rmm}mm_sweep`, keyframes: HOLD_TRACK(HOME),
    blend: 1, gap: 0.05, side: 0, approach: 0, spawn: { x: 0.26, y: STAIR_Y, z: h + 0.12 },
    note: `ON-TREAD CONTROL for the step-self-collision threshold sweep, rise ${rmm} mm.` },
    `/tmp/rig3_sweep_${rmm}.json`);
  for (const n of [4, 1]) {
    const r = await scoreSaved(p, { rise: h, tail: 'policy', overrides: { stepCount: n } });
    const s = r.scored, c = r.crit;
    const lz = (s.lfoot.z - h) * 1000, rz = (s.rfoot.z - h) * 1000;
    results.thresholdSweep.push({ rise_mm: rmm, stepCount: n, zOverlap_mm: 200 - rmm,
      minStepGap_mm: r.minStepGap_mm === null ? null : +r.minStepGap_mm.toFixed(2),
      maxTreadDriftX_mm: +r.maxTreadDriftX_mm.toFixed(2), maxTreadSag_mm: +r.maxTreadSag_mm.toFixed(2),
      x_mm: +mm(s.x), z_mm: +mm(s.z), above_mm: +mm(s.above),
      lfootZminusRise_mm: +lz.toFixed(1), rfootZminusRise_mm: +rz.toFixed(1),
      feetOnTread: s.feetOnTread, up: s.up, crit: c });
    console.log(`  ${rmm.toString().padStart(4)} ${String(n).padStart(5)} ${(200 - rmm).toString().padStart(5)}  ` +
      `${(r.minStepGap_mm === null ? 'n/a' : r.minStepGap_mm.toFixed(2)).padStart(10)}  ` +
      `${r.maxTreadDriftX_mm.toFixed(2).padStart(6)} ${r.maxTreadSag_mm.toFixed(2).padStart(5)} ` +
      `${mm(s.x).padStart(8)} ${mm(s.z).padStart(6)} ${mm(s.above).padStart(6)} ` +
      `${lz.toFixed(1).padStart(9)} ${rz.toFixed(1).padStart(9)} ${String(s.feetOnTread).padStart(9)}  ${s.up?'Y':'n'}  ${c.honest?'PASS':' . '}`);
  }
}
console.log();

// ---------------------------------------------------------------- summary
const tailVerdict = {};
for (const k of TAILS) {
  const on = results.controls.onTread.filter(r => r.tail === k);
  const dn = results.controls.doNothing.filter(r => r.tail === k).concat(results.controls.doNothingShort.filter(r => r.tail === k));
  const wo = results.controls.walkOnly.filter(r => r.tail === k);
  const risesExpressed = RISES.filter(h => on.some(r => r.rise_mm === h * 1000 && r.crit.honest));
  tailVerdict[k] = {
    onTread_pass_honest: on.filter(r => r.crit.honest).length + '/' + on.length,
    onTread_pass_orig: on.filter(r => r.crit.orig).length + '/' + on.length,
    onTread_rises_expressed_mm: risesExpressed.map(h => h * 1000),
    onTread_upright_frac: on.filter(r => r.up).length + '/' + on.length,
    doNothing_pass_honest: dn.filter(r => r.crit.honest).length + '/' + dn.length,
    doNothing_pass_orig: dn.filter(r => r.crit.orig).length + '/' + dn.length,
    doNothing_upright: dn.filter(r => r.up).length + '/' + dn.length,
    walkOnly_pass_honest: wo.filter(r => r.crit.honest).length + '/' + wo.length,
    walkOnly_pass_orig: wo.filter(r => r.crit.orig).length + '/' + wo.length,
    canExpressSuccess: risesExpressed.length > 0 && dn.filter(r => r.crit.honest).length === 0,
  };
}
results.tailVerdict = tailVerdict;
console.log('\n=== TAIL VERDICT (does the tail let a real climb read as success?) ===');
for (const k of TAILS) {
  const v = tailVerdict[k];
  console.log(`  tail=${k.padEnd(7)} on-tread HONEST ${v.onTread_pass_honest.padStart(6)}  upright ${v.onTread_upright_frac.padStart(6)}  ` +
    `| do-nothing HONEST ${v.doNothing_pass_honest} upright ${v.doNothing_upright}  ` +
    `| walk-only HONEST ${v.walkOnly_pass_honest}  -> CAN EXPRESS SUCCESS: ${v.canExpressSuccess}`);
  console.log(`             rises where a real on-tread stance reads as PASS (mm): [${v.onTread_rises_expressed_mm.join(', ')}]`);
}
// the podium confound, counted
const offFlight = results.bests.filter(r => r.maxAbsDY_mm > LATERAL * 1000);
results.lateralConfound = {
  rows_off_flight: offFlight.length, rows_total: results.bests.length,
  files_off_flight: [...new Set(offFlight.map(r => r.file))].sort(),
  worst_mm: Math.max(...results.bests.map(r => r.maxAbsDY_mm)),
};
console.log(`\n=== LATERAL CONFOUND ===\n  ${offFlight.length}/${results.bests.length} scored rows leave the 340 mm flight ` +
  `(|y-STAIR_Y| > ${LATERAL*1000} mm at some tick); worst excursion ${results.lateralConfound.worst_mm.toFixed(1)} mm.`);
console.log(`  files that ever leave the flight: ${results.lateralConfound.files_off_flight.join(', ') || 'none'}`);

results.elapsed_s = +((Date.now() - t00) / 1000).toFixed(1);
fs.writeFileSync(OUT, JSON.stringify(results, null, 1));
console.log(`\nWROTE ${OUT}  (${results.elapsed_s}s)`);
}
