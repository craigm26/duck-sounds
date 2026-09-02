// robust.mjs — THE ROUND-3 SCORER. One copy, imported by every family.
//
// Round 2 reported single-cell results and then discovered, in audit_r2, that
// a 40 mm clear survived 1 of 7 perturbations and a 60 mm clear 3 of 7. A move
// that passes once is a coincidence. So round 3 scores a SAVED FILE over a
// grid and reports "cleared k of N". ROUND 4 extended that grid to 14 cells
// and added an upright-through-the-tail term; the core 9 below are unchanged:
//
//   rise  in { h-10 mm, h, h+10 mm }
//   plant in { (drop 0.120, friction x1.0),      -- nominal
//              (drop 0.130, friction x0.7),      -- higher fall, slippery
//              (drop 0.125, friction x1.3) }     -- higher fall, grippy
//
// THE EPISODE LOOP BELOW IS audit_r2.mjs's go(), which is itself a verbatim
// copy of rig3.mjs runEpisodeRaw() (7/7 EXACT parity, climb/audit_r2-results
// .json). sim/ is off-limits to edit and importing rig3 for the loop is not
// possible (its loop is not exported), so this is the ONE round-3 copy; every
// family imports scoreRobust from here rather than making a fourth.
//
// Differences from rig3's loop, all of them knobs, none of them physics:
//   - drop      : spawn height (rig3 hard-codes 0.12)
//   - fmul      : multiplier on foot geom friction[0] (nominal read from model)
//   - isolate   : step-step collision isolation. The shipped flight is now
//                 isolated by site/stairs.js findStairJoints(); this file
//                 captures the SHIPPED affinity with {isolate:false} and sets
//                 it per run, so isolate:true reproduces the shipped plant
//                 exactly. Phase P proves it against rig3.scoreSaved.
//
// criteria() and reward() are IMPORTED from rig3.mjs — not copied — so the
// verdict and the shaped reward cannot drift from the instrument.
//
// Contacts via mj_geomDistance only; data.contact.get(i) leaks the WASM heap.
//
// Run from sim/ for the parity phase:
//   cd ~/projects/duck-sounds/sim && node ../climb/robust.mjs
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { makeLoop } from '../site/duckloop.mjs';
import { findStairJoints, layoutStairs, STAIR_Y, STAIR_HALF_WIDTH } from '../site/stairs.js';
import { normEvent, eventFires, eventError, buildDynTrack } from '../climb/event.mjs';
import { criteria as rig3Criteria, reward as rig3Reward } from '../climb/rig3_pre_r5.mjs';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
export const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/r.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const model = mj.MjModel.mj_loadBinary('/r.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model);
// capture the SHIPPED conaffinity; this file toggles isolation itself
const ADDR = findStairJoints(model, { isolate: false });
let GYRO = 0;
for (let i = 0; i < model.nsensor; i++) if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;
const stand = await ort.InferenceSession.create('./BEST_alpha_stand.onnx');
const DT = 1 / C.tickHz;

export const LATERAL = STAIR_HALF_WIDTH;   // 0.17 m
export const RISER_X = 0.12;
export { STAIR_Y };

// ---------------------------------------------------------------- geom ids
const bodyId = n => { for (let b = 0; b < model.nbody; b++) if (model.body(b).name === n) return b; return -1; };
const JAWB = bodyId('jaw_soft');
const JAW = []; for (let g = 0; g < model.ngeom; g++) if (model.geom_bodyid[g] === JAWB && !(model.geom_contype[g] === 0 && model.geom_conaffinity[g] === 0)) JAW.push(g);
let STEP0 = -1, STEP1 = -1, FLOOR = -1, LFOOT = -1, RFOOT = -1, WALLN = -1;
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
const FEET = []; for (let g = 0; g < model.ngeom; g++) if (/foot_collision|sole/.test(model.geom(g).name || '')) FEET.push(g);
// the duck's own legs and trunk, for the wall test (bodies hip_l/leg/ankle_*)
const LEGG = [];
for (let g = 0; g < model.ngeom; g++) {
  const b = model.body(model.geom_bodyid[g]).name || '';
  if (/^(hip_l|hip_l_2|leg|leg_2|ankle_left|ankle_right|trunk_base)$/.test(b) && model.geom_contype[g] === 5) LEGG.push(g);
}
const STEP_CONAFF0 = STEPG.map(g => model.geom_conaffinity[g]);
const FRICT0 = FEET.map(g => model.geom_friction[g * 3]);

// ROUND 4, HOLE 3: every collidable duck geom, for penetrationAtScore.
// Same construction as rig3.mjs / audit_r3.mjs.
let DUCKROOT = -1;
for (let j = 0; j < model.njnt; j++) if (model.jnt_type[j] === 0 && model.jnt_qposadr[j] === D.freeQpos) { DUCKROOT = model.jnt_bodyid[j]; break; }
const underDuck = b => { let c = b; for (let i = 0; i < 64 && c > 0; i++) { if (c === DUCKROOT) return true; c = model.body_parentid[c]; } return c === DUCKROOT; };
export const DUCKG = [];
for (let g = 0; g < model.ngeom; g++) {
  if (model.geom_contype[g] === 0 && model.geom_conaffinity[g] === 0) continue;
  if (DUCKROOT >= 0 && underDuck(model.geom_bodyid[g])) DUCKG.push(g);
}
if (!DUCKG.length) { for (const g of JAW) DUCKG.push(g); for (const g of FEET) DUCKG.push(g); }

const quat = () => [data.qpos[D.freeQpos + 3], data.qpos[D.freeQpos + 4], data.qpos[D.freeQpos + 5], data.qpos[D.freeQpos + 6]];
const dist = (a, b) => mj.mj_geomDistance(model, data, a, b, 0.05, null);

/** rig3.mjs poseAt / climb_lib.mjs:80-86, verbatim. */
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

/** rig3.mjs footResting(), verbatim: ceiling AND a real 3 mm contact. */
function footResting(g, h) {
  const x = data.geom_xpos[g * 3], y = data.geom_xpos[g * 3 + 1], z = data.geom_xpos[g * 3 + 2];
  if (!(z > h - 0.005 && z < h + 0.045 && x > RISER_X && Math.abs(y - STAIR_Y) <= LATERAL)) return false;
  for (const sg of STEPG) if (dist(g, sg) < 0.003) return true;
  return false;
}

/** rig3.mjs penetrationNow(), verbatim. Read-only: mj_geomDistance is a query. */
function penetrationNow() {
  let pen = 1e9, pair = null;
  for (const g of DUCKG) for (const sg of STEPG) {
    const d = dist(g, sg);
    if (d < pen) { pen = d; pair = `${model.geom(g).name || 'g' + g}<->${model.geom(sg).name}`; }
  }
  return { pen: pen === 1e9 ? null : pen, pair };
}

/** rig3.mjs snapshot() fields that criteria() and reward() read. */
function snapshot(h, maxAbsDY) {
  const x = data.qpos[D.freeQpos], y = data.qpos[D.freeQpos + 1], z = data.qpos[D.freeQpos + 2];
  const up = projectedGravity(quat())[2] < -0.90;
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

/** rig3.mjs handoffNow(), verbatim: the state a beat-2 spawn needs. Read-only. */
function handoffNow(h) {
  const jp = [], jv = [], free = [];
  for (let k = 0; k < 14; k++) { jp.push(data.qpos[D.qpos[k]]); jv.push(data.qvel[D.dof[k]]); }
  for (let k = 0; k < 6; k++) free.push(data.qvel[D.freeDof + k]);
  let head = false;
  for (const g of JAW) if (dist(g, STEP0) < 0.003) { head = true; break; }
  let footRiser = false, feetOnTread = 0;
  for (const g of [LFOOT, RFOOT]) if (data.geom_xpos[g * 3 + 2] < h - 0.005 && dist(g, STEP0) < 0.003) footRiser = true;
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

// ---------------------------------------------------------------- the episode
/**
 * One episode. audit_r2.mjs go(), with the round-3 instrumentation added.
 * INTERNAL: nothing outside this file scores an in-memory track.
 */
async function go(track, o, h, { drop = 0.120, fmul = 1.0, isolate = true, stepCount = 4 } = {}) {
  const cfg = { count: stepCount, rise: h, run: 0.28, start: 0.12 };
  STEPG.forEach((g, i) => { model.geom_conaffinity[g] = isolate ? 0 : STEP_CONAFF0[i]; });
  FEET.forEach((g, i) => { model.geom_friction[g * 3] = FRICT0[i] * fmul; });
  mj.mj_resetData(model, data);
  layoutStairs(data, ADDR, cfg);
  if (o.spawn) {
    data.qpos[D.freeQpos] = o.spawn.x;
    data.qpos[D.freeQpos + 1] = o.spawn.y;
    data.qpos[D.freeQpos + 2] = o.spawn.z + (drop - 0.120);
  } else {
    data.qpos[D.freeQpos] = 0.12 - 0.07 - (o.gap || 0);
    data.qpos[D.freeQpos + 1] = STAIR_Y + (o.side || 0);
    data.qpos[D.freeQpos + 2] = drop;
  }
  data.qpos[D.freeQpos + 3] = 1;
  for (let i = 0; i < 14; i++) { data.qpos[D.qpos[i]] = HOME[i]; data.ctrl[i] = HOME[i]; }
  // ROUND 4, FAMILY B: the optional handoff state (see rig3.mjs for the
  // documentation). Absent -> not one line of it runs, so every pre-round-4
  // file scores exactly as it did; climb/famB_parity.log proves that against
  // climb/robust_prefamB.mjs at full float digits.
  if (o.spawnQuat) for (let k = 0; k < 4; k++) data.qpos[D.freeQpos + 3 + k] = o.spawnQuat[k];
  if (o.spawnPose) for (let i = 0; i < 14; i++) {
    data.qpos[D.qpos[i]] = o.spawnPose[i];
    data.ctrl[i] = Math.min(Math.max(o.spawnPose[i], LO[i]), HI[i]);
  }
  if (o.spawnVel) {
    if (o.spawnVel.free) for (let k = 0; k < 6; k++) data.qvel[D.freeDof + k] = o.spawnVel.free[k];
    if (o.spawnVel.joint) for (let i = 0; i < 14; i++) data.qvel[D.dof[i]] = o.spawnVel.joint[i];
  }
  mj.mj_forward(model, data);
  const tr = track.map(f => ({ t: f.t, pose: f.pose.slice() }));
  let la = o.spawnLastAction ? o.spawnLastAction.slice() : new Array(14).fill(0);
  const cmd = command({ vx: o.approach || 0 });

  const R = { ticks: 0, headTicks: 0, riserTicks: 0, wallTicks: 0, wallBearTicks: 0,
              upTicks: 0, sat: 0, ctrls: 0, maxX: -1e9, maxZ: -1e9, maxAbsDY: 0,
              feetOnTreadMax: 0, feetHighMax: 0, headOnlyTicks: 0,
              bothTicks: 0, sustainTicks: 0, liftIntegral: 0, maxGainBoth: -1e9,
              wallGain: -1e9, maxTreadDriftX_mm: 0, minStepGap_mm: 1e9, maxTq: 0,
              footNear: 1e9, bothNear: 1e9 };
  let Z0 = 0;
  // A LANDING SPOT on the first tread: mid-tread, one foot-thickness up.
  // footNear / bothNear are how close a foot (and the worse of the two feet)
  // ever came to it. Graded, so a search has a gradient toward landing instead
  // of the all-or-nothing 3 mm contact test. Read-only: they touch no verdict.
  const TGT = [0.22, STAIR_Y, h + 0.015];
  const nearTo = g => {
    const dx = data.geom_xpos[g * 3] - TGT[0], dy = data.geom_xpos[g * 3 + 1] - TGT[1],
          dz = data.geom_xpos[g * 3 + 2] - TGT[2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
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
      const onBlock = dist(g, STEP0) < 0.003, onFloor = dist(g, FLOOR) < 0.003;
      if (onBlock || onFloor) footAny = true;
      if (onBlock && data.geom_xpos[g * 3 + 2] < h - 0.005) footRiser = true;
    }
    if (footRiser) R.riserTicks++;
    if (head && !footAny) R.headOnlyTicks++;
    // THE WALL as a third contact
    let wall = false;
    for (const g of LEGG) if (dist(g, WALLN) < 0.003) { wall = true; break; }
    if (wall) R.wallTicks++;
    let fot = 0, fhi = 0;
    for (const g of FEET) {
      const lat = Math.abs(data.geom_xpos[g * 3 + 1] - STAIR_Y) <= LATERAL;
      if (footResting(g, h)) fot++;
      if (data.geom_xpos[g * 3 + 2] > h + 0.005 && lat) fhi++;
    }
    if (fot > R.feetOnTreadMax) R.feetOnTreadMax = fot;
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
    // was the wall ever LOAD-BEARING: in contact while the duck is above its
    // settled height and at least one other contact is carrying it
    if (wall && (head || footRiser || fot > 0)) {
      R.wallBearTicks++;
      const g = z - Z0;
      if (g > R.wallGain) R.wallGain = g;
    }
  };

  const step = async (off, rec) => {
    layoutStairs(data, ADDR, cfg);
    const q = quat(); const jp = [], jv = [];
    for (let k = 0; k < 14; k++) { jp.push(data.qpos[D.qpos[k]]); jv.push(data.qvel[D.dof[k]]); }
    const obs = buildObs([data.sensordata[GYRO], data.sensordata[GYRO + 1], data.sensordata[GYRO + 2]], projectedGravity(q), jp, jv, la, cmd);
    const r = await stand.run({ obs: new ort.Tensor('float32', obs, [1, 61]) });
    la = Array.from(r.actions.data);
    for (let k = 0; k < 14; k++) {
      const v = HOME[k] + la[k] + (off ? (off[k] - HOME[k]) * o.blend : 0);
      const c = Math.min(Math.max(v, LO[k]), HI[k]);
      data.ctrl[k] = c;
      if (rec) { R.ctrls++; if (c <= LO[k] + 1e-9 || c >= HI[k] - 1e-9) R.sat++; }
    }
    for (let s = 0; s < 4; s++) mj.mj_step(model, data);
    if (rec) {
      for (let a = 0; a < model.nu; a++) { const f = Math.abs(data.actuator_force[a]); if (f > R.maxTq) R.maxTq = f; }
      const dx = Math.abs(data.geom_xpos[STEP0 * 3] - (0.12 + 0.17)) * 1000;
      if (dx > R.maxTreadDriftX_mm) R.maxTreadDriftX_mm = dx;
      if (STEP1 >= 0 && cfg.count > 1) {
        const gp = dist(STEP0, STEP1) * 1000;
        if (gp < R.minStepGap_mm) R.minStepGap_mm = gp;
      }
      record();
    }
  };

  const SETTLE = (o.settleTicks === undefined || o.settleTicks === null) ? 25 : o.settleTicks;
  for (let t = 0; t < SETTLE; t++) await step(null, false);
  const x0 = data.qpos[D.freeQpos];
  Z0 = data.qpos[D.freeQpos + 2];
  // ================================================== ROUND 4, FAMILY A
  // AN OPTIONAL EVENT-TRIGGERED TAIL. `opts.event` is absent in every file
  // written before round 4, and normEvent(undefined) is null, in which case
  // TR === tr, `total` never changes, and the two lines below are the
  // pre-round-4 loop verbatim: same tick count, same poses, same ONNX calls.
  // climb/famA_r4.mjs PHASE P proves that on every existing best_* file.
  const EV = normEvent(o.event);          // (famB: go()'s options object is `o`, not `opts`)
  let TR = tr;
  let total = TR[TR.length - 1].t + 0.8;
  let evFired = false, evT = null, evE = null, evTrunkX = null;
  const beakDistNow = () => {
    let d = 1e9;
    for (const g of JAW) for (const sg of STEPG) { const v = dist(g, sg); if (v < d) d = v; }
    return d === 1e9 ? null : d;
  };
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
    await step(poseAt(TR, time), true);
  }
  const atTrackEnd = snapshot(h, R.maxAbsDY);
  // ROUND 4, FAMILY B: the beat-1 -> beat-2 handoff state, at the instant a
  // concatenated beat 2 would begin (beat 1's last keyframe + 0.8 s).
  const terminal = handoffNow(h); terminal.spawnLastAction = la.slice();
  // ROUND 4, HOLE 2: how much of the 50-tick tail was the duck upright?
  const upBeforeTail = R.upTicks, ticksBeforeTail = R.ticks;
  for (let t = 0; t < 50; t++) await step(null, true);   // tail 'policy' — climb_lib's own
  const scored = snapshot(h, R.maxAbsDY);
  const uprightTailTicks = R.upTicks - upBeforeTail;
  const tailTicks = R.ticks - ticksBeforeTail;

  const rec = {
    rise: h, x0, scored, atTrackEnd,
    event: EV ? { type: EV.type, fired: evFired, tFire: evT, trunkXAtFire: evTrunkX, e_mm: evE === null ? null : +(evE * 1000).toFixed(2) } : null,
    penetrationAtScore: scored.penetrationAtScore, penetrationPair: scored.penetrationPair,
    uprightTailTicks, tailTicks, uprightTailFrac: uprightTailTicks / Math.max(tailTicks, 1),
    terminal,          // ROUND 4, FAMILY B: the beat-1 -> beat-2 handoff state
    crit: rig3Criteria(h, scored), critAtTrackEnd: rig3Criteria(h, atTrackEnd),
    maxX: R.maxX, maxZ: R.maxZ, maxAbsDY: R.maxAbsDY, maxTq: R.maxTq,
    feetOnTreadMax: R.feetOnTreadMax, feetHighMax: R.feetHighMax,
    headFrac: R.headTicks / Math.max(R.ticks, 1),
    riserFrac: R.riserTicks / Math.max(R.ticks, 1),
    wallFrac: R.wallTicks / Math.max(R.ticks, 1),
    wallBearFrac: R.wallBearTicks / Math.max(R.ticks, 1),
    wallGain: R.wallGain === -1e9 ? null : R.wallGain,
    headOnlyFrac: R.headOnlyTicks / Math.max(R.ticks, 1),
    bothFrac: R.bothTicks / Math.max(R.ticks, 1),
    sustainFrac: R.sustainTicks / Math.max(R.ticks, 1),
    liftIntegral: R.liftIntegral,
    maxGainBoth: R.maxGainBoth === -1e9 ? null : R.maxGainBoth,
    upFrac: R.upTicks / Math.max(R.ticks, 1),
    satFrac: R.sat / Math.max(R.ctrls, 1),
    z0Settle: Z0,
    footNear: R.footNear, bothNear: R.bothNear,
    maxTreadDriftX_mm: R.maxTreadDriftX_mm,
    minStepGap_mm: R.minStepGap_mm === 1e9 ? null : R.minStepGap_mm,
  };
  rec.reward = rig3Reward(rec);      // rig3's own reward(), imported
  return rec;
}

// ---------------------------------------------------------------- the grid
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

// ------------------------------------------------------------ ROUND 4 grid
// The core 9 cells above are unchanged, so `kCore` is directly comparable to
// round 3's "k of 9". The EXTENDED grid adds five cells that round 3's grid
// could not see:
//   * rise h-5 mm and h+5 mm at the NOMINAL plant. Round 3's rise axis was
//     +/-10 mm only, and the beak-strut vault's failure is a landing that is
//     fixed at authoring time and lands ~10 mm short: a 5 mm step exposes how
//     narrow the surviving window actually is.
//   * the SLIPPERY plant, friction x0.5 with drop 0.140, crossed with the three
//     core rises. Round 3's worst plant was x0.7 / drop 0.130 and it already
//     cleared only 1 of 21; x0.5 at a 20 mm higher fall is the axis the audit
//     ran as an extra and no family ever optimised against.
// N = 9 + 2 + 3 = 14.
export const EXT_DHS = [-0.005, 0.005];               // nominal plant only
export const EXT_PLANT = { drop: 0.140, fmul: 0.5 };  // crossed with DHS
export const EXT_CELL_COUNT = 14;

/**
 * ROUND 4, HOLE 2. A cell's +CLEAR_BONUS is earned only if the duck was upright
 * for at least UPRIGHT_TAIL_MIN of the 50 tail ticks; a candidate that reaches
 * the tread and topples inside the tail forfeits it. On top of that the mean
 * upright-tail fraction is worth up to UPRIGHT_BONUS, so standing beats
 * toppling even between two candidates that clear nothing.
 */
export const UPRIGHT_TAIL_MIN = 45;   // of 50 ticks = 0.90 of the tail
export const UPRIGHT_BONUS = 4;
/**
 * THE UPRIGHT TERM IS EARNED, NOT GIVEN. Measured on this plant: an ungated
 * upright-tail term pays the DO-NOTHING control the full +4.00 for standing
 * still on the floor (objective 5.451 -> 9.452 at 40 mm), which puts doing
 * nothing above best_r3_landvault_80mm (4.154) and best_r3_landvault_90mm
 * (8.370). A term that rewards not trying is not a fix for hole 2. So a cell
 * pays its upright credit only if the duck got somewhere in that cell: the
 * trunk crossed the riser line at some tick, or a foot rested on a tread at
 * some tick. Do-nothing never crosses x = 0.12 (peak trunk x 2-3 mm) and so
 * earns 0.
 */
const reachedFlight = c => (c.maxX > RISER_X) || (c.feetOnTreadMax > 0);

/**
 * ROUND 4, HOLE 4 + the family-C breach. The DECLARED search bounds. A saved
 * file whose parameters fall outside them is not a result: it is a search that
 * left its own declared box, and round 3 only ever noticed after the fact.
 * A file may narrow (or widen) these with its own `bounds` object; whatever it
 * declares is what it is held to, and the declaration travels in every row.
 */
export const DECLARED_BOUNDS = { blend: [0.7, 2.4], side: [-0.02, 0.09] };

const boundsOf = j => ({ ...DECLARED_BOUNDS, ...(j.bounds || {}) });
export function checkBounds(j) {
  const B = boundsOf(j), bad = [];
  for (const [k, [lo, hi]] of Object.entries(B)) {
    const v = k === 'side' || k === 'gap' || k === 'approach' ? (j[k] || 0) : j[k];
    if (typeof v !== 'number' || !(v >= lo && v <= hi)) bad.push({ param: k, value: v, lo, hi });
  }
  return { bounds: B, violations: bad };
}

const readIntent = path => {
  const j = JSON.parse(fs.readFileSync(path, 'utf8'));
  if (!Array.isArray(j.keyframes) || !j.keyframes.length) throw new Error('no keyframes in ' + path);
  for (const f of j.keyframes) if (!Array.isArray(f.pose) || f.pose.length !== 14) throw new Error('bad pose in ' + path);
  return j;
};
const optsOf = j => ({ blend: j.blend, approach: j.approach || 0, gap: j.gap || 0, side: j.side || 0,
                       spawn: j.spawn || null,
                       // ROUND 4, FAMILY A: the optional event-triggered tail.
                       event: j.event || null,
                       // ROUND 4, FAMILY B handoff fields; null for every older file
                       spawnQuat: j.spawnQuat || null, spawnPose: j.spawnPose || null,
                       spawnVel: j.spawnVel || null, spawnLastAction: j.spawnLastAction || null,
                       settleTicks: j.settleTicks === undefined ? undefined : j.settleTicks });
const isoOf = j => (j.isolate === undefined ? true : j.isolate !== false);
const scOf = j => j.stepCount || 4;

/**
 * ROUND 4, HOLE 4. THE MOVE'S IDENTITY: sha256 over everything the episode
 * actually reads. One vector published under three rise labels hashes to one
 * value; it is one move, and the hash travels in every result row so a table
 * cannot quietly count it three times.
 */
export function intentHash(j) {
  const h = {
    keyframes: j.keyframes, blend: j.blend, gap: j.gap || 0, side: j.side || 0,
    approach: j.approach || 0, spawn: j.spawn || null, isolate: isoOf(j), stepCount: scOf(j),
  };
  // ROUND 4, FAMILY B. The hash must cover everything the episode reads, so the
  // handoff fields go in — but ONLY when the file actually has them. A file
  // written before this round has none of them and therefore hashes to exactly
  // the value climb/r4_audit-results.json published (checked in famB_parity).
  // ROUND 4, FAMILY A adds 'event' to the same present-only list, for the same
  // reason: an older file has none of these keys and hashes to exactly what
  // climb/r4_audit-results.json published (PHASE H in famA_r4.mjs re-checks it).
  for (const k of ['event', 'spawnQuat', 'spawnPose', 'spawnVel', 'spawnLastAction', 'settleTicks']) {
    if (j[k] !== undefined && j[k] !== null) h[k] = j[k];
  }
  return crypto.createHash('sha256').update(JSON.stringify(h)).digest('hex');
}
export function intentHashOfFile(path) { return intentHash(readIntent(path)); }

/** The loud refusal. Printed to stderr AND stdout so no log can miss it. */
function shoutInvalid(path, viol, B) {
  const bar = '!'.repeat(78);
  const lines = [bar,
    `INVALID INTENT — OUT OF DECLARED BOUNDS: ${path}`,
    ...viol.map(v => `   ${v.param} = ${v.value}  is outside the declared [${v.lo}, ${v.hi}]`),
    `   declared bounds: ${JSON.stringify(B)}`,
    '   This file scores as INVALID. objective = -Infinity, k = 0. Bounds are',
    '   enforced HERE, at scoring time, not only declared in a comment.', bar];
  for (const l of lines) { console.log(l); console.error(l); }
}

/**
 * Score ONE cell of the grid, from a SAVED file. Used for cheap screening;
 * a reported result must always come from scoreRobust().
 */
export async function scoreCell(path, { rise, dh = 0, drop = 0.120, fmul = 1.0, isolate, stepCount, skipBounds = false } = {}) {
  const j = readIntent(path);
  const B = checkBounds(j);
  if (B.violations.length && !skipBounds) {
    shoutInvalid(path, B.violations, B.bounds);
    return { invalid: true, boundViolations: B.violations, bounds: B.bounds,
             sha256: intentHash(j), reward: -Infinity, crit: { honest: false, orig: false } };
  }
  const r = await go(j.keyframes, optsOf(j), rise + dh,
    { drop, fmul, isolate: isolate === undefined ? isoOf(j) : isolate,
      stepCount: stepCount || scOf(j) });
  r.sha256 = intentHash(j);
  return r;
}

/**
 * THE SCORER. Scores a SAVED file over the EXTENDED 14-cell grid (pass
 * {core:true} for the round-3 9 cells only).
 *
 *   kCore / nCore  cells cleared under rig3 `honest` on the round-3 9 cells,
 *                  reported unchanged so round 3's numbers stay comparable
 *   kExt  / nExt   the same count over all 14
 *   *Stable        the same cells, minus any that toppled inside the tail
 *
 *   objective     = meanReward(14) + 4*kExtStable  + 4*meanUprightTailFrac(14)
 *   objectiveCore = meanReward(9)  + 4*kCoreStable + 4*meanUprightTailFrac(9)
 *   objectiveR3   = meanReward(9)  + 4*kCore          <- exactly round 3's
 *
 * A file outside its declared bounds does not get scored: it returns
 * invalid:true with objective -Infinity, after shouting.
 */
export async function scoreRobust(path, { rise, isolate, stepCount, onCell, skipBounds = false, core = false } = {}) {
  const j = readIntent(path);
  const sha = intentHash(j);
  const B = checkBounds(j);
  if (B.violations.length && !skipBounds) {
    shoutInvalid(path, B.violations, B.bounds);
    return { source: path, rise, sha256: sha, move: sha.slice(0, 12), invalid: true,
             boundViolations: B.violations, bounds: B.bounds,
             k: 0, kCore: 0, kExt: 0, kCoreStable: 0, kExtStable: 0,
             meanReward: -Infinity, objective: -Infinity, objectiveCore: -Infinity, objectiveR3: -Infinity,
             cells: [], verdicts: [] };
  }
  const o = optsOf(j);
  const iso = isolate === undefined ? isoOf(j) : isolate;
  const sc = stepCount || scOf(j);

  // THE GRID. Core first (so a partial run is still the round-3 grid), then the
  // five extended cells. Every cell is tagged with which half it belongs to.
  const plan = [];
  for (const dh of DHS) for (const p of PLANTS) plan.push({ dh, p, tier: 'core' });
  if (!core) {
    for (const dh of EXT_DHS) plan.push({ dh, p: PLANTS[0], tier: 'ext' });
    for (const dh of DHS) plan.push({ dh, p: EXT_PLANT, tier: 'ext' });
  }

  const cells = [];
  for (const { dh, p, tier } of plan) {
    const r = await go(j.keyframes, o, rise + dh, { drop: p.drop, fmul: p.fmul, isolate: iso, stepCount: sc });
    r.cell = { rise_mm: Math.round((rise + dh) * 1000), drop: p.drop, fmul: p.fmul, tier };
    r.sha256 = sha;
    // ROUND 4, HOLE 2: a clear only counts toward the BONUS if it also stood up
    // through the tail. `honest` itself is untouched, so kCore stays comparable.
    r.stableClear = r.crit.honest && r.uprightTailTicks >= UPRIGHT_TAIL_MIN;
    cells.push(r);
    if (onCell) onCell(r);
  }
  const coreCells = cells.filter(c => c.cell.tier === 'core');
  const kCore = coreCells.filter(c => c.crit.honest).length;
  const kExt = cells.filter(c => c.crit.honest).length;
  const kCoreStable = coreCells.filter(c => c.stableClear).length;
  const kExtStable = cells.filter(c => c.stableClear).length;
  const meanOf = (arr, f) => arr.reduce((a, c) => a + f(c), 0) / Math.max(arr.length, 1);
  const mean = f => meanOf(cells, f);
  const meanReward = mean(c => c.reward);
  const meanRewardCore = meanOf(coreCells, c => c.reward);
  const credit = c => (reachedFlight(c) ? c.uprightTailFrac : 0);
  const upFracExt = mean(credit);
  const upFracCore = meanOf(coreCells, credit);
  const upFracRawExt = mean(c => c.uprightTailFrac);
  const reachedCells = cells.filter(reachedFlight).length;
  return {
    source: path, rise, sha256: sha, move: sha.slice(0, 12), invalid: false,
    bounds: B.bounds, boundViolations: [],
    // round-3-comparable
    k: kCore, kCore, nCore: coreCells.length, kCoreStable,
    // the round-4 extended grid
    kExt, nExt: cells.length, kExtStable,
    meanReward, meanRewardCore,
    uprightTailFrac: upFracExt, uprightTailFracCore: upFracCore,
    uprightTailFracRaw: upFracRawExt, reachedFlightCells: reachedCells,
    // what the objective would be if the upright term were NOT earned
    objectiveUngatedUpright: meanReward + CLEAR_BONUS * kExtStable + UPRIGHT_BONUS * upFracRawExt,
    // THE OBJECTIVE every family optimises, over the extended grid
    objective: meanReward + CLEAR_BONUS * kExtStable + UPRIGHT_BONUS * upFracExt,
    objectiveCore: meanRewardCore + CLEAR_BONUS * kCoreStable + UPRIGHT_BONUS * upFracCore,
    // exactly what round 3 printed, so the re-baseline can show the delta
    objectiveR3: meanRewardCore + CLEAR_BONUS * kCore,
    cells,
    agg: {
      minPenetrationAtScore_mm: Math.min(...cells.map(c => c.penetrationAtScore === null ? 0 : c.penetrationAtScore)) * 1000,
      meanUprightTailTicks: mean(c => c.uprightTailTicks),
      minUprightTailTicks: Math.min(...cells.map(c => c.uprightTailTicks)),
      lateralEscapeCells: cells.filter(c => c.maxAbsDY > LATERAL).length,
      maxZ: Math.max(...cells.map(c => c.maxZ)),
      meanMaxZ: mean(c => c.maxZ),
      meanPeakGain: mean(c => c.maxZ - c.z0Settle),
      headFrac: mean(c => c.headFrac),
      riserFrac: mean(c => c.riserFrac),
      wallFrac: mean(c => c.wallFrac),
      wallBearFrac: mean(c => c.wallBearFrac),
      bothFrac: mean(c => c.bothFrac),
      sustainFrac: mean(c => c.sustainFrac),
      liftIntegral: mean(c => c.liftIntegral),
      footNear_mm: mean(c => c.footNear) * 1000,
      bothNear_mm: mean(c => c.bothNear) * 1000,
      feetOnTreadMax: Math.max(...cells.map(c => c.feetOnTreadMax)),
      meanFeetOnTreadMax: mean(c => c.feetOnTreadMax),
      meanFeetOnTreadFinal: mean(c => c.scored.feetOnTread),
      meanAbove_mm: mean(c => c.scored.above) * 1000,
      meanX_mm: mean(c => c.scored.x) * 1000,
      upFinal: cells.filter(c => c.scored.up).length,
      satFrac: mean(c => c.satFrac),
      maxTq: Math.max(...cells.map(c => c.maxTq)),
      maxAbsDY_mm: Math.max(...cells.map(c => c.maxAbsDY)) * 1000,
      maxTreadDriftX_mm: Math.max(...cells.map(c => c.maxTreadDriftX_mm)),
    },
    verdicts: cells.map(c => ({ rise_mm: c.cell.rise_mm, drop: c.cell.drop, fmul: c.cell.fmul,
      tier: c.cell.tier, sha256: sha, move: sha.slice(0, 12),
      honest: c.crit.honest, stableClear: c.stableClear,
      eventFired: c.event ? c.event.fired : null, eventT: c.event ? c.event.tFire : null,
      eventE_mm: c.event ? c.event.e_mm : null,
      uprightTailTicks: c.uprightTailTicks, tailTicks: c.tailTicks,
      penetrationAtScore_mm: c.penetrationAtScore === null ? null : +(c.penetrationAtScore * 1000).toFixed(2),
      penetrationPair: c.penetrationPair,
      maxAbsDY_mm: +(c.maxAbsDY * 1000).toFixed(1), lateralEpisode: c.crit.lateralEpisode,
      reward: +c.reward.toFixed(3),
      x_mm: +(c.scored.x * 1000).toFixed(1), z_mm: +(c.scored.z * 1000).toFixed(1),
      above_mm: +(c.scored.above * 1000).toFixed(1), dy_mm: +(c.scored.dy * 1000).toFixed(1),
      up: c.scored.up, feetOnTread: c.scored.feetOnTread, feetOnTreadMax: c.feetOnTreadMax,
      peakZ_mm: +(c.maxZ * 1000).toFixed(1), headFrac: +c.headFrac.toFixed(3),
      riserFrac: +c.riserFrac.toFixed(3), wallFrac: +c.wallFrac.toFixed(3) })),
  };
}

/** Write an intent object to disk and hand back the path. Nothing scores memory. */
export function saveIntent(obj, path) { fs.writeFileSync(path, JSON.stringify(obj, null, 2)); return path; }

// ================================================================== PHASE P
const isMain = process.argv[1] && process.argv[1].endsWith('robust_pre_r5.mjs');
if (isMain) {
  const { scoreSaved } = await import('../climb/rig3.mjs');
  console.log('=== robust.mjs PHASE P — is cell 0 (drop 0.120, x1.0, isolate on) rig3? ===');
  console.log(`duck leg/trunk geoms for the wall test: ${LEGG.length}  wall_n geom ${WALLN}  jaw geoms ${JAW.length}`);
  const cases = [['best_r2_vault_40mm.json', 0.040], ['best_r2_vault_60mm.json', 0.060],
                 ['best_r2_vault_90mm.json', 0.090], ['ctrl_on_tread_90mm.json', 0.090],
                 ['ctrl_do_nothing.json', 0.090]];
  let all = true;
  for (const [f, h] of cases) {
    const A = await scoreSaved('../climb/' + f, { rise: h, tail: 'policy' });
    const B = await scoreCell('../climb/' + f, { rise: h, isolate: true });
    const ok = A.scored.x === B.scored.x && A.scored.z === B.scored.z
      && A.scored.feetOnTread === B.scored.feetOnTread && A.crit.honest === B.crit.honest
      && A.reward === B.reward;
    if (!ok) all = false;
    console.log(`  ${f.padEnd(30)} rig3 x=${A.scored.x} z=${A.scored.z} rew=${A.reward.toFixed(4)} | robust x=${B.scored.x} z=${B.scored.z} rew=${B.reward.toFixed(4)} EXACT=${ok}`);
  }
  console.log(`  parityAll = ${all}`);
  const t0 = Date.now();
  const g = await scoreRobust('../climb/best_r2_vault_60mm.json', { rise: 0.060 });
  const dt = (Date.now() - t0) / 1000;
  console.log(`=== extended grid on best_r2_vault_60mm @60mm: kCore=${g.kCore}/${g.nCore} kExt=${g.kExt}/${g.nExt} kExtStable=${g.kExtStable}/${g.nExt} meanReward=${g.meanReward.toFixed(3)} objective=${g.objective.toFixed(3)} objectiveR3=${g.objectiveR3.toFixed(3)}  (${dt.toFixed(1)} s, ${(dt / g.nExt).toFixed(2)} s/cell) ===`);
  for (const v of g.verdicts) console.log(`   [${v.tier}] rise=${v.rise_mm} drop=${v.drop} f=${v.fmul} honest=${v.honest} stable=${v.stableClear} upTail=${v.uprightTailTicks}/${v.tailTicks} pen=${v.penetrationAtScore_mm}mm maxDY=${v.maxAbsDY_mm}mm rew=${v.reward} x=${v.x_mm} above=${v.above_mm} fot=${v.feetOnTread}`);
  const g9 = await scoreRobust('../climb/best_r2_vault_90mm.json', { rise: 0.090 });
  console.log(`=== extended grid on best_r2_vault_90mm @90mm: kCore=${g9.kCore}/${g9.nCore} kExt=${g9.kExt}/${g9.nExt} meanReward=${g9.meanReward.toFixed(3)} objective=${g9.objective.toFixed(3)} minUpTail=${g9.agg.minUprightTailTicks}/50 minPen=${g9.agg.minPenetrationAtScore_mm.toFixed(2)}mm ===`);
}
