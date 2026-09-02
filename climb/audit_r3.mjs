// ROUND 3 AUDIT — the adversarial re-score.
//
// Round 3 shipped three families. Only family A claimed any cell cleared under
// rig3's `honest`. This file re-derives every one of those claims from the
// SAVED JSON, through a loop it first PROVES is rig3, and then tries to break
// each claimed clear on axes the round-3 grid never touched.
//
// The loop below is a verbatim copy of rig3.mjs runEpisodeRaw() with four knobs
// (drop, foot-friction multiplier, isolate, stepCount) plus read-only forensic
// instrumentation. Phase P proves it is rig3 at FULL FLOAT DIGITS on x, y, z,
// above, up, feetUpRaw, feetOnTread, honest, orig and reward, on every file
// under audit and both controls. Nothing downstream is believed until that
// prints EXACT=true everywhere.
//
// WHAT IS CHECKED, beyond re-running the 9-cell grid:
//   * the extended plant: foot friction x0.5 and x1.5, spawn drop 0.11 and
//     0.14, each crossed with the same three rises (12 more cells per file).
//   * THE LATERAL GATE, whole-episode. rig3.criteria() tests |dy| only at the
//     scored instant; rig3.reward() applies the whole-episode gate. So a cell
//     can be `honest` while the duck left the 340 mm flight mid-episode. Every
//     pass is checked against maxAbsDY.
//   * SERVO FORCE against the plant's own actuator_forcerange ceiling, read
//     from the model, not assumed.
//   * INTERPENETRATION: min mj_geomDistance between every collidable duck geom
//     and every step geom over the episode. A pass standing inside a block is
//     not a pass.
//   * SPAWN ON THE FLOOR: no spawn override, spawn z == the plant's drop, and
//     after the 25-tick settle the trunk is still behind the riser with no foot
//     resting on a tread.
//   * PARAMETER RANGES: blend in [0.7, 2.4], side in [-0.02, 0.09].
//   * CONTROLS on the same plant: do-nothing must fail 9 of 9, a duck placed on
//     the tread must pass 9 of 9, at every rise where a clear is claimed.
//
// Contacts via mj_geomDistance only; data.contact.get(i) leaks the WASM heap.
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/audit_r3.mjs
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
import { findStairJoints, layoutStairs, STAIR_Y, STAIR_HALF_WIDTH } from '../site/stairs.js';
import { scoreSaved as rig3Score, criteria as rig3Criteria, reward as rig3Reward } from '../climb/rig3.mjs';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/a3.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const model = mj.MjModel.mj_loadBinary('/a3.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model);
// capture the SHIPPED conaffinity; this file toggles isolation itself
const ADDR = findStairJoints(model, { isolate: false });
let GYRO = 0; for (let i = 0; i < model.nsensor; i++) if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;
const stand = await ort.InferenceSession.create('./BEST_alpha_stand.onnx');
const DT = 1 / C.tickHz;
const LATERAL = STAIR_HALF_WIDTH, RISER_X = 0.12;

// ---------------------------------------------------------------- geom ids
const bodyId = n => { for (let b = 0; b < model.nbody; b++) if (model.body(b).name === n) return b; return -1; };
const JAWB = bodyId('jaw_soft');
const JAW = []; for (let g = 0; g < model.ngeom; g++) if (model.geom_bodyid[g] === JAWB && !(model.geom_contype[g] === 0 && model.geom_conaffinity[g] === 0)) JAW.push(g);
let STEP0 = -1, STEP1 = -1, FLOOR = -1, LFOOT = -1, RFOOT = -1;
const STEPG = [];
for (let g = 0; g < model.ngeom; g++) {
  const n = model.geom(g).name || '';
  if (n === 'step0_geom') STEP0 = g;
  if (n === 'step1_geom') STEP1 = g;
  if (n === 'floor') FLOOR = g;
  if (n === 'left_foot_collision') LFOOT = g;
  if (n === 'right_foot_collision') RFOOT = g;
  if (/^step\d+_geom$/.test(n)) STEPG.push(g);
}
const FEET = []; for (let g = 0; g < model.ngeom; g++) if (/foot_collision|sole/.test(model.geom(g).name || '')) FEET.push(g);
const STEP_CONAFF0 = STEPG.map(g => model.geom_conaffinity[g]);
const FRICT0 = FEET.map(g => model.geom_friction[g * 3]);
// the plant's own ceiling, read from the model
let FR = 0; for (let a = 0; a < model.nu; a++) FR = Math.max(FR, model.actuator_forcerange[a * 2 + 1]);

// every collidable geom belonging to the duck (the body tree under the free joint)
let DUCKROOT = -1;
// the model has more than one free joint (there is a ball prop). The DUCK's is
// the one at D.freeQpos, which findDuckJoints resolved.
for (let j = 0; j < model.njnt; j++) if (model.jnt_type[j] === 0 && model.jnt_qposadr[j] === D.freeQpos) { DUCKROOT = model.jnt_bodyid[j]; break; }
const underDuck = b => { let c = b; for (let i = 0; i < 64 && c > 0; i++) { if (c === DUCKROOT) return true; c = model.body_parentid[c]; } return c === DUCKROOT; };
const DUCKG = [];
for (let g = 0; g < model.ngeom; g++) {
  if (model.geom_contype[g] === 0 && model.geom_conaffinity[g] === 0) continue;
  if (DUCKROOT >= 0 && underDuck(model.geom_bodyid[g])) DUCKG.push(g);
}
if (!DUCKG.length) { for (const g of JAW) DUCKG.push(g); for (const g of FEET) DUCKG.push(g); }

const dist = (a, b) => mj.mj_geomDistance(model, data, a, b, 0.05, null);
const quat = () => [data.qpos[D.freeQpos + 3], data.qpos[D.freeQpos + 4], data.qpos[D.freeQpos + 5], data.qpos[D.freeQpos + 6]];

/** rig3.mjs poseAt / climb_lib.mjs:80-86, verbatim. */
function poseAt(tr, time) {
  if (time <= 0) return HOME.slice();
  let pt = 0, pp = HOME;
  for (const f of tr) {
    if (time <= f.t) { const u = (time - pt) / Math.max(f.t - pt, 1e-9), s = u * u * (3 - 2 * u);
      return f.pose.map((v, k) => pp[k] + (v - pp[k]) * s); }
    pt = f.t; pp = f.pose;
  }
  return tr[tr.length - 1].pose.slice();
}

/** rig3.mjs footResting(), verbatim. */
function footResting(g, h) {
  const x = data.geom_xpos[g * 3], y = data.geom_xpos[g * 3 + 1], z = data.geom_xpos[g * 3 + 2];
  if (!(z > h - 0.005 && z < h + 0.045 && x > RISER_X && Math.abs(y - STAIR_Y) <= LATERAL)) return false;
  for (const sg of STEPG) if (dist(g, sg) < 0.003) return true;
  return false;
}

/** rig3.mjs snapshot(), the fields criteria() and reward() read. */
function snapshot(h) {
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
  return { x, y, z, dy: y - STAIR_Y, above: z - h, up, feetUpRaw, feetUpLat, feetOnTread,
           lfoot: foot(LFOOT), rfoot: foot(RFOOT) };
}

// ---------------------------------------------------------------- the episode
async function go(track, o, h, { drop = 0.120, fmul = 1.0, isolate = true, stepCount = 4 } = {}) {
  const cfg = { count: stepCount, rise: h, run: 0.28, start: 0.12 };
  STEPG.forEach((g, i) => { model.geom_conaffinity[g] = isolate ? 0 : STEP_CONAFF0[i]; });
  FEET.forEach((g, i) => { model.geom_friction[g * 3] = FRICT0[i] * fmul; });
  mj.mj_resetData(model, data);
  layoutStairs(data, ADDR, cfg);
  let spawn;
  if (o.spawn) {
    data.qpos[D.freeQpos] = o.spawn.x;
    data.qpos[D.freeQpos + 1] = o.spawn.y;
    data.qpos[D.freeQpos + 2] = o.spawn.z + (drop - 0.120);
  } else {
    data.qpos[D.freeQpos] = 0.12 - 0.07 - (o.gap || 0);
    data.qpos[D.freeQpos + 1] = STAIR_Y + (o.side || 0);
    data.qpos[D.freeQpos + 2] = drop;
  }
  spawn = { x: data.qpos[D.freeQpos], y: data.qpos[D.freeQpos + 1], z: data.qpos[D.freeQpos + 2],
            override: !!o.spawn };
  data.qpos[D.freeQpos + 3] = 1;
  for (let i = 0; i < 14; i++) { data.qpos[D.qpos[i]] = HOME[i]; data.ctrl[i] = HOME[i]; }
  mj.mj_forward(model, data);
  const tr = track.map(f => ({ t: f.t, pose: f.pose.slice() }));
  let la = new Array(14).fill(0);
  const cmd = command({ vx: o.approach || 0 });

  let maxTq = 0, maxAbsDY = 0, maxX = -1e9, maxZ = -1e9, minStepGap = 1e9, maxDriftX = 0;
  let minPenStep = 1e9, minPenFloor = 1e9, feetOnTreadMax = 0, sat = 0, ctrls = 0, ticks = 0, upTicks = 0;
  let dyAtPass = null;

  const forensics = () => {
    ticks++;
    for (let a = 0; a < model.nu; a++) { const f = Math.abs(data.actuator_force[a]); if (f > maxTq) maxTq = f; }
    // interpenetration: every collidable duck geom vs every step geom, and the feet vs the floor
    for (const g of DUCKG) for (const sg of STEPG) { const d = dist(g, sg); if (d < minPenStep) minPenStep = d; }
    if (FLOOR >= 0) for (const g of DUCKG) { const d = dist(g, FLOOR); if (d < minPenFloor) minPenFloor = d; }
    const dy = Math.abs(data.qpos[D.freeQpos + 1] - STAIR_Y); if (dy > maxAbsDY) maxAbsDY = dy;
    if (data.qpos[D.freeQpos] > maxX) maxX = data.qpos[D.freeQpos];
    if (data.qpos[D.freeQpos + 2] > maxZ) maxZ = data.qpos[D.freeQpos + 2];
    if (projectedGravity(quat())[2] < -0.90) upTicks++;
    if (STEP1 >= 0 && cfg.count > 1) { const g = dist(STEP0, STEP1); if (g < minStepGap) minStepGap = g; }
    const dx = Math.abs(data.geom_xpos[STEP0 * 3] - (0.12 + 0.17)); if (dx > maxDriftX) maxDriftX = dx;
    let fot = 0; for (const g of FEET) if (footResting(g, h)) fot++;
    if (fot > feetOnTreadMax) feetOnTreadMax = fot;
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
      if (rec) { ctrls++; if (c <= LO[k] + 1e-9 || c >= HI[k] - 1e-9) sat++; }
    }
    for (let s = 0; s < 4; s++) mj.mj_step(model, data);
    if (rec) forensics();
  };

  for (let t = 0; t < 25; t++) await step(null, false);
  // SPAWN CHECK: where is the duck after the settle, before the track runs?
  const settle = snapshot(h);
  let minFootStepSettle = 1e9;
  for (const g of FEET) for (const sg of STEPG) { const d = dist(g, sg); if (d < minFootStepSettle) minFootStepSettle = d; }
  const total = tr[tr.length - 1].t + 0.8;
  for (let t = 0; t * DT < total; t++) await step(poseAt(tr, t * DT), true);
  const atTrackEnd = snapshot(h);
  for (let t = 0; t < 50; t++) await step(null, true);
  const scored = snapshot(h);
  // penetration AT THE SCORED INSTANT (the whole-episode minimum above can be a
  // transient beak strike; a pass that is standing INSIDE a block is a different
  // thing), with the geom pair named.
  let penNow = 1e9, penPair = null;
  for (const g of DUCKG) for (const sg of STEPG) {
    const d = dist(g, sg);
    if (d < penNow) { penNow = d; penPair = `${model.geom(g).name || 'g' + g}<->${model.geom(sg).name}`; }
  }
  // DOES IT HOLD? 100 more ticks of the same stand policy (2.0 s at 50 Hz),
  // then the criterion again. A stance that dissolves is not a climb.
  for (let t = 0; t < 100; t++) await step(null, false);
  const held = snapshot(h);
  const critHeld = rig3Criteria(h, held);

  const rec = { rise: h, spawn, scored, atTrackEnd, held, critHeld, penNow, penPair,
    settle: { x: settle.x, z: settle.z, dy: settle.dy, feetOnTread: settle.feetOnTread,
              minFootStep: minFootStepSettle, up: settle.up },
    crit: rig3Criteria(h, scored), critAtTrackEnd: rig3Criteria(h, atTrackEnd),
    maxX, maxZ, maxAbsDY, maxTq, minPenStep, minPenFloor, minStepGap: minStepGap === 1e9 ? null : minStepGap,
    maxDriftX, feetOnTreadMax, satFrac: sat / Math.max(ctrls, 1), upFrac: upTicks / Math.max(ticks, 1),
    z0Settle: settle.z };
  rec.reward = rig3Reward(rec);
  return rec;
}

// ---------------------------------------------------------------- helpers
const P = '../climb/';
const readIntent = f => {
  const j = JSON.parse(fs.readFileSync(P + f, 'utf8'));
  if (!Array.isArray(j.keyframes) || !j.keyframes.length) throw new Error('no keyframes in ' + f);
  for (const k of j.keyframes) if (!Array.isArray(k.pose) || k.pose.length !== 14) throw new Error('bad pose in ' + f);
  return j;
};
const optsOf = j => ({ blend: j.blend, approach: j.approach || 0, gap: j.gap || 0, side: j.side || 0, spawn: j.spawn || null });
const isoOf = j => (j.isolate === undefined ? true : j.isolate !== false);
const scOf = j => j.stepCount || 4;
const mm = v => (v * 1000).toFixed(1);
const log = s => { console.log(s); LOG.push(s); };
const LOG = [];
const OUT = { generated: new Date().toISOString(), plant: 'scene.mjb', forceCeiling: FR,
              lateralGate_m: LATERAL, riserX_m: RISER_X, nominalFootFriction: FRICT0[0],
              duckGeoms: DUCKG.length, stepGeoms: STEPG.length };

// the 9-cell grid, verbatim from robust.mjs
const PLANTS = [{ drop: 0.120, fmul: 1.0 }, { drop: 0.130, fmul: 0.7 }, { drop: 0.125, fmul: 1.3 }];
const DHS = [-0.010, 0.000, 0.010];
// the axes round 3 never tried
const XPLANTS = [{ drop: 0.120, fmul: 0.5 }, { drop: 0.120, fmul: 1.5 },
                 { drop: 0.110, fmul: 1.0 }, { drop: 0.140, fmul: 1.0 }];

const BLEND_LO = 0.7, BLEND_HI = 2.4, SIDE_LO = -0.02, SIDE_HI = 0.09;

import crypto from 'node:crypto';
/** The move's identity: everything the episode actually reads. */
const moveHash = j => crypto.createHash('sha256').update(JSON.stringify({
  keyframes: j.keyframes, blend: j.blend, gap: j.gap || 0, side: j.side || 0,
  approach: j.approach || 0, spawn: j.spawn || null, isolate: isoOf(j), stepCount: scOf(j),
})).digest('hex').slice(0, 12);

async function grid(file, rise, plants, label) {
  const j = readIntent(file), o = optsOf(j), iso = isoOf(j), sc = scOf(j);
  const cells = [];
  for (const dh of DHS) for (const p of plants) {
    const r = await go(j.keyframes, o, rise + dh, { drop: p.drop, fmul: p.fmul, isolate: iso, stepCount: sc });
    cells.push({ rise_mm: Math.round((rise + dh) * 1000), drop: p.drop, fmul: p.fmul,
      honest: r.crit.honest, orig: r.crit.orig, reward: +r.reward.toFixed(4),
      x_mm: +mm(r.scored.x), z_mm: +mm(r.scored.z), above_mm: +mm(r.scored.above), dy_mm: +mm(r.scored.dy),
      up: r.scored.up, feetOnTread: r.scored.feetOnTread, feetOnTreadMax: r.feetOnTreadMax,
      honestAtTrackEnd: r.critAtTrackEnd.honest, honestHeld100: r.critHeld.honest,
      heldAbove_mm: +mm(r.held.above), heldFot: r.held.feetOnTread, heldUp: r.held.up,
      penNow_mm: +mm(r.penNow), penPair: r.penPair,
      peakZ_mm: +mm(r.maxZ), maxAbsDY_mm: +mm(r.maxAbsDY), maxTq: +r.maxTq.toFixed(4),
      minPenStep_mm: +mm(r.minPenStep), minPenFloor_mm: +mm(r.minPenFloor),
      driftX_mm: +mm(r.maxDriftX), stepGap_mm: r.minStepGap === null ? null : +mm(r.minStepGap),
      settleX_mm: +mm(r.settle.x), settleZ_mm: +mm(r.settle.z), settleFot: r.settle.feetOnTread,
      settleMinFootStep_mm: +mm(r.settle.minFootStep), spawn: r.spawn, satFrac: +r.satFrac.toFixed(3),
      lfoot_mm: [+mm(r.scored.lfoot.x), +mm(r.scored.lfoot.y), +mm(r.scored.lfoot.z)],
      rfoot_mm: [+mm(r.scored.rfoot.x), +mm(r.scored.rfoot.y), +mm(r.scored.rfoot.z)] });
  }
  const k = cells.filter(c => c.honest).length;
  const meanReward = cells.reduce((a, c) => a + c.reward, 0) / cells.length;
  return { file, label, move: moveHash(j), rise_mm: Math.round(rise * 1000), k, n: cells.length,
           meanReward: +meanReward.toFixed(4), objective: +(meanReward + 4 * k).toFixed(4), cells };
}

// ================================================================== PHASE P
log(`plant scene.mjb — actuator forcerange ceiling read from the model: ${FR.toFixed(4)} N.m`);
log(`flight half-width ${mm(LATERAL)} mm | riser face x ${mm(RISER_X)} mm | nominal foot friction ${FRICT0[0]}`);
log(`duck collidable geoms ${DUCKG.length} (root body ${DUCKROOT}) | step geoms ${STEPG.length}`);
log('');
log('=== PHASE P — is this loop rig3? same file, same rise, nominal cell, FULL FLOAT DIGITS ===');
const PARITY = [
  ['best_r3_vault_40mm.json', 0.040], ['best_r3_vault_50mm.json', 0.050],
  ['best_r3_vault_60mm.json', 0.060], ['best_r3_vault_70mm.json', 0.070],
  ['best_r3_vault_80mm.json', 0.080], ['best_r2_vault_40mm.json', 0.040],
  ['best_r2_vault_60mm.json', 0.060], ['best_r3_landvault_90mm.json', 0.090],
  ['best_r3_cornerclimb_120mm.json', 0.120],
  ['ctrl_do_nothing.json', 0.060], ['ctrl_on_tread_60mm.json', 0.060],
];
OUT.parity = [];
let parityAll = true;
for (const [f, h] of PARITY) {
  const j = readIntent(f);
  const A = await rig3Score(P + f, { rise: h, tail: 'policy' });
  const B = await go(j.keyframes, optsOf(j), h, { isolate: isoOf(j), stepCount: scOf(j) });
  const fields = ['x', 'y', 'z', 'above', 'dy'];
  const same = fields.every(k => A.scored[k] === B.scored[k])
    && A.scored.up === B.scored.up && A.scored.feetUpRaw === B.scored.feetUpRaw
    && A.scored.feetOnTread === B.scored.feetOnTread
    && A.crit.honest === B.crit.honest && A.crit.orig === B.crit.orig
    && A.reward === B.reward;
  if (!same) parityAll = false;
  OUT.parity.push({ file: f, rise_mm: Math.round(h * 1000), exact: same,
    rig3: { x: A.scored.x, y: A.scored.y, z: A.scored.z, fot: A.scored.feetOnTread, honest: A.crit.honest, reward: A.reward },
    mine: { x: B.scored.x, y: B.scored.y, z: B.scored.z, fot: B.scored.feetOnTread, honest: B.crit.honest, reward: B.reward } });
  log(`  ${f.padEnd(32)} h=${String(Math.round(h * 1000)).padStart(3)}  rig3 x=${A.scored.x} z=${A.scored.z} rew=${A.reward}`);
  log(`  ${''.padEnd(32)}        mine x=${B.scored.x} z=${B.scored.z} rew=${B.reward}   EXACT=${same}`);
}
log(`  parityAll = ${parityAll}`);
OUT.parityAll = parityAll;
log('');
if (!parityAll) { log('PARITY FAILED — every number below is unusable.'); }

// ================================================================== PHASE F
// static file checks: parameter ranges and spawn declarations
log('=== PHASE F — declared parameters (blend in [0.7,2.4], side in [-0.02,0.09]), spawn overrides ===');
const FILES_ALL = ['best_r3_vault_40mm.json', 'best_r3_vault_50mm.json', 'best_r3_vault_60mm.json',
  'best_r3_vault_70mm.json', 'best_r3_vault_80mm.json', 'best_r2_vault_40mm.json', 'best_r2_vault_60mm.json',
  'best_r3_landvault_80mm.json', 'best_r3_landvault_90mm.json',
  'best_r3_cornerclimb_120mm.json', 'best_r3_cornerclimb2_120mm.json', 'best_r3_cornerclimb_180mm.json'];
OUT.paramCheck = [];
for (const f of FILES_ALL) {
  const j = readIntent(f);
  const blendOk = j.blend >= BLEND_LO && j.blend <= BLEND_HI;
  const sideOk = (j.side || 0) >= SIDE_LO && (j.side || 0) <= SIDE_HI;
  const rec = { file: f, blend: j.blend, blendOk, side: j.side || 0, sideOk, gap: j.gap || 0,
    approach: j.approach || 0, spawnOverride: !!j.spawn, spawn: j.spawn || null,
    isolate: isoOf(j), stepCount: scOf(j), keyframes: j.keyframes.length };
  OUT.paramCheck.push(rec);
  log(`  ${f.padEnd(34)} blend=${String(j.blend).padStart(7)} ${blendOk ? 'ok ' : 'OUT'}  side=${String(j.side || 0).padStart(8)} ${sideOk ? 'ok ' : 'OUT'}  gap=${String(j.gap || 0).padStart(8)}  approach=${String(j.approach || 0).padStart(7)}  spawnOverride=${!!j.spawn}  isolate=${isoOf(j)}  steps=${scOf(j)}`);
}
log('');

// ================================================================== PHASE G
// the 9-cell grid, re-derived for every claimed clear
log('=== PHASE G — the 9-cell grid re-derived from the SAVED file, for every k>=1 claim ===');
const CLAIMS = [
  { file: 'best_r3_vault_40mm.json', rise: 0.040, claimedK: 2, claimedObj: 15.4718 },
  { file: 'best_r3_vault_50mm.json', rise: 0.050, claimedK: 2, claimedObj: 17.0209 },
  { file: 'best_r3_vault_60mm.json', rise: 0.060, claimedK: 4, claimedObj: 23.7882 },
  { file: 'best_r3_vault_70mm.json', rise: 0.070, claimedK: 2, claimedObj: 13.4779 },
  { file: 'best_r3_vault_80mm.json', rise: 0.080, claimedK: 1, claimedObj: 8.9192 },
  { file: 'best_r2_vault_40mm.json', rise: 0.040, claimedK: 1, claimedObj: 9.6465 },
  { file: 'best_r2_vault_60mm.json', rise: 0.060, claimedK: 2, claimedObj: 11.4366 },
];
OUT.grids = [];
for (const c of CLAIMS) {
  const g = await grid(c.file, c.rise, PLANTS, '9-cell');
  g.claimedK = c.claimedK; g.claimedObjective = c.claimedObj;
  g.kMatch = g.k === c.claimedK;
  g.objDelta = +(g.objective - c.claimedObj).toFixed(4);
  OUT.grids.push(g);
  log(`  ${c.file.padEnd(30)} @${g.rise_mm}mm  claimed k=${c.claimedK}/9 obj=${c.claimedObj}   REPRODUCED k=${g.k}/9 obj=${g.objective}  match=${g.kMatch}  dObj=${g.objDelta}`);
  for (const v of g.cells.filter(x => x.honest))
    log(`      PASS  rise=${v.rise_mm} drop=${v.drop} fric=x${v.fmul}  x=${v.x_mm}mm above=${v.above_mm}mm dy=${v.dy_mm}mm fot=${v.feetOnTread} maxDY=${v.maxAbsDY_mm}mm maxTq=${v.maxTq} penStep=${v.minPenStep_mm}mm rew=${v.reward}`);
}
log('');

// ================================================================== PHASE Z
// families B and C claimed zero. Confirm zero.
log('=== PHASE Z — the two families that claimed k=0, confirmed from file ===');
OUT.zeroClaims = [];
for (const [f, h] of [['best_r3_landvault_80mm.json', 0.080], ['best_r3_landvault_90mm.json', 0.090],
                      ['best_r3_cornerclimb_120mm.json', 0.120], ['best_r3_cornerclimb2_120mm.json', 0.120],
                      ['best_r3_cornerclimb_180mm.json', 0.180]]) {
  const g = await grid(f, h, PLANTS, '9-cell');
  OUT.zeroClaims.push(g);
  log(`  ${f.padEnd(34)} @${g.rise_mm}mm  k=${g.k}/9 obj=${g.objective} meanRew=${g.meanReward}`);
}
log('');

// ================================================================== PHASE X
// the axes round 3 never tried
log('=== PHASE X — the extended plant (friction x0.5 / x1.5, drop 0.11 / 0.14) x the same three rises ===');
OUT.extended = [];
for (const c of CLAIMS) {
  const g = await grid(c.file, c.rise, XPLANTS, '12-cell extended');
  OUT.extended.push(g);
  const base = OUT.grids.find(x => x.file === c.file);
  const byPlant = {};
  for (const v of g.cells) { const key = `d${v.drop}/f${v.fmul}`; byPlant[key] = (byPlant[key] || 0) + (v.honest ? 1 : 0); }
  g.byPlant = byPlant;
  log(`  ${c.file.padEnd(30)} @${g.rise_mm}mm  extended k=${g.k}/12  ${Object.entries(byPlant).map(([k2, v]) => `${k2}:${v}/3`).join('  ')}   ==> combined ${base.k + g.k} of 21`);
  for (const v of g.cells.filter(x => x.honest))
    log(`      PASS  rise=${v.rise_mm} drop=${v.drop} fric=x${v.fmul}  x=${v.x_mm}mm above=${v.above_mm}mm fot=${v.feetOnTread} maxDY=${v.maxAbsDY_mm}mm`);
}
log('');

// ================================================================== PHASE C
// controls, on exactly the plant the clears used
log('=== PHASE C — controls on the same plant: do-nothing must fail 9/9, a placed duck must pass 9/9 ===');
// the placed-duck recipe from climb/ctrl_on_tread_*.json: spawn x 0.25, y STAIR_Y, z rise + 0.12
const placedFor = (rm) => {
  const have = `ctrl_on_tread_${rm}mm.json`;
  if (fs.existsSync(P + have)) return have;
  const src = JSON.parse(fs.readFileSync(P + 'ctrl_on_tread_60mm.json', 'utf8'));
  const out = { ...src, name: `placed on tread, rise ${rm} mm (audit_r3)`,
    spawn: { x: 0.25, y: STAIR_Y, z: rm / 1000 + 0.12 },
    note: 'generated by climb/audit_r3.mjs from the ctrl_on_tread_*.json recipe (x 0.25, y STAIR_Y, z rise+0.12)' };
  const name = `audit_r3_ctrl_on_tread_${rm}mm.json`;
  fs.writeFileSync(P + name, JSON.stringify(out, null, 2));
  return name;
};
OUT.controls = [];
for (const rm of [40, 50, 60, 70, 80]) {
  const rise = rm / 1000;
  const dn = await grid('ctrl_do_nothing.json', rise, PLANTS, `do-nothing @${rm}`);
  const pf = placedFor(rm);
  const pd = await grid(pf, rise, PLANTS, `placed-duck @${rm}`);
  OUT.controls.push({ rise_mm: rm, doNothing: dn, placed: pd, placedFile: pf });
  log(`  rise ${String(rm).padStart(3)} mm   do-nothing k=${dn.k}/9 (must be 0)   placed-duck (${pf}) k=${pd.k}/9 (must be 9)   ` +
      `placed above=${(pd.cells.reduce((a, c) => a + c.above_mm, 0) / 9).toFixed(1)}mm fot=${(pd.cells.reduce((a, c) => a + c.feetOnTread, 0) / 9).toFixed(2)}`);
  for (const v of pd.cells.filter(x => !x.honest))
    log(`      placed-duck FAIL  rise=${v.rise_mm} drop=${v.drop} fric=x${v.fmul}  up=${v.up} x=${v.x_mm} above=${v.above_mm} fot=${v.feetOnTread}`);
}
log('');

// ================================================================== PHASE H
// hygiene over every cell this audit ran
log('=== PHASE H — hygiene over every cell scored above ===');
const allCells = [];
for (const g of [...OUT.grids, ...OUT.zeroClaims, ...OUT.extended]) for (const c of g.cells) allCells.push({ g: g.file, ...c });
for (const c of OUT.controls) for (const g of [c.doNothing, c.placed]) for (const cell of g.cells) allCells.push({ g: g.label, ...cell });
const maxOf = f => allCells.reduce((a, c) => Math.max(a, f(c)), -1e9);
const minOf = f => allCells.reduce((a, c) => Math.min(a, f(c)), 1e9);
const tqMax = maxOf(c => c.maxTq), dyMax = maxOf(c => c.maxAbsDY_mm);
const penMin = minOf(c => c.minPenStep_mm), driftMax = maxOf(c => c.driftX_mm);
const gapMin = minOf(c => c.stepGap_mm === null ? 1e9 : c.stepGap_mm);
log(`  cells scored: ${allCells.length}`);
log(`  max |actuator_force| over every cell : ${tqMax.toFixed(4)} N.m  (plant ceiling ${FR.toFixed(4)}) -> ${tqMax <= FR + 1e-9 ? 'WITHIN' : 'OVER'}`);
log(`  max |trunk y - STAIR_Y| over every cell : ${dyMax.toFixed(1)} mm  (gate ${mm(LATERAL)} mm) -> ${dyMax <= LATERAL * 1000 ? 'inside' : 'SOME CELL LEFT THE FLIGHT'}`);
log(`  most-negative duck<->step mj_geomDistance : ${penMin.toFixed(2)} mm`);
log(`  max tread drift in x : ${driftMax.toFixed(3)} mm ; min step0<->step1 gap : ${gapMin === 1e9 ? 'n/a' : gapMin.toFixed(1)} mm`);

// the sharp question: does any HONEST pass anywhere depend on leaving the flight,
// over-torque, interpenetration, or a spawn that was not on the floor?
const passes = allCells.filter(c => c.honest);
OUT.passAudit = passes.map(c => ({
  file: c.g, rise_mm: c.rise_mm, drop: c.drop, fmul: c.fmul,
  x_mm: c.x_mm, above_mm: c.above_mm, dy_mm: c.dy_mm, feetOnTread: c.feetOnTread,
  reward: c.reward,
  leftFlight: c.maxAbsDY_mm > LATERAL * 1000,
  overTorque: c.maxTq > FR + 1e-9,
  interpenetration_mm: c.minPenStep_mm, penAtScoredInstant_mm: c.penNow_mm, penPair: c.penPair,
  honestAtTrackEnd: c.honestAtTrackEnd, stillHonestAfter100MoreTicks: c.honestHeld100,
  heldAbove_mm: c.heldAbove_mm, heldFot: c.heldFot, heldUp: c.heldUp,
  spawnOverride: c.spawn.override,
  spawnZ_mm: +(c.spawn.z * 1000).toFixed(1),
  spawnX_mm: +(c.spawn.x * 1000).toFixed(1),
  settleX_mm: c.settleX_mm, settleFot: c.settleFot, settleMinFootStep_mm: c.settleMinFootStep_mm,
  rewardZeroedByLateralGate: c.reward === 0,
}));
const badLat = OUT.passAudit.filter(p => p.leftFlight);
const badTq = OUT.passAudit.filter(p => p.overTorque);
const badPen = OUT.passAudit.filter(p => p.interpenetration_mm < -2);
const badSpawn = OUT.passAudit.filter(p => !/placed|on_tread/.test(p.file) && (p.spawnOverride || p.settleFot > 0 || p.settleX_mm > 120));
log(`  HONEST passes anywhere in this audit: ${passes.length}`);
log(`    that left the 340 mm flight mid-episode : ${badLat.length}`);
log(`    that exceeded the actuator ceiling      : ${badTq.length}`);
log(`    with >2 mm of duck-into-step penetration: ${badPen.length}`);
log(`    that did not start on the floor         : ${badSpawn.length}`);
for (const b of badLat) log(`      LATERAL-GATE ESCAPE: ${b.file} rise=${b.rise_mm} drop=${b.drop} f=${b.fmul} dy_at_score=${b.dy_mm}mm reward=${b.reward}`);
// the widest lateral excursion anywhere, named
const widest = allCells.reduce((a, c) => (c.maxAbsDY_mm > a.maxAbsDY_mm ? c : a));
log(`  widest lateral excursion anywhere: ${widest.maxAbsDY_mm} mm  (${widest.g} rise=${widest.rise_mm} drop=${widest.drop} f=x${widest.fmul}, honest=${widest.honest})`);
log(`  NOTE: rig3.criteria() tests |dy| ONLY at the scored instant; rig3.reward() applies the whole-episode gate.`);
log(`        So an 'honest' pass can in principle have left the flight mid-episode. In this audit ${badLat.length} did.`);
// penetration: vault passes vs a duck simply placed on the tread
const placedCells = [].concat(...OUT.controls.map(c => c.placed.cells));
const placedPenNow = Math.min(...placedCells.map(c => c.penNow_mm));
const placedPenEp = Math.min(...placedCells.map(c => c.minPenStep_mm));
const vaultPasses = OUT.passAudit.filter(p => /vault/.test(p.file));
log(`  soft-contact reference — a duck PLACED on the tread: min duck<->step distance ${placedPenEp.toFixed(2)} mm over the episode, ${placedPenNow.toFixed(2)} mm at the scored instant`);
if (vaultPasses.length) {
  log(`  the vault's clears: whole-episode min ${Math.min(...vaultPasses.map(p => p.interpenetration_mm)).toFixed(2)} mm, at the scored instant ${Math.min(...vaultPasses.map(p => p.penAtScoredInstant_mm)).toFixed(2)} mm`);
}
// does the stance hold?
log(`  of the ${passes.length} honest passes, still honest after 100 MORE ticks of the stand policy: ${passes.filter(c => c.honestHeld100).length}`);
const vp = allCells.filter(c => c.honest && /vault/.test(c.g));
log(`  of the ${vp.length} vault clears, still honest after 100 more ticks: ${vp.filter(c => c.honestHeld100).length}` +
    `  | honest already at the track's end (before the 50-tick tail): ${vp.filter(c => c.honestAtTrackEnd).length}`);
// distinct moves: 60/70/80 mm were suspected to be one vector
const byMove = {};
for (const g of [...OUT.grids, ...OUT.extended]) (byMove[g.move] = byMove[g.move] || []).push(`${g.file}@${g.rise_mm}`);
log(`  distinct MOVES among the ${OUT.grids.length} claimed candidates: ${new Set(OUT.grids.map(g => g.move)).size}`);
for (const [hsh, files] of Object.entries(byMove)) if (new Set(files.map(f => f.split('@')[0])).size > 1)
  log(`      identical vector ${hsh}: ${[...new Set(files.map(f => f.split('@')[0]))].join(', ')}`);
// tallest rise cleared anywhere in this audit, and by which plant
const clearedRises = [...new Set(allCells.filter(c => c.honest && !/placed|on_tread/.test(c.g)).map(c => c.rise_mm))].sort((a, b) => a - b);
log(`  rises cleared by a SEARCHED move anywhere in this audit: ${clearedRises.join(', ')} mm`);
const tallest = Math.max(...clearedRises);
const tallestCells = allCells.filter(c => c.honest && !/placed|on_tread/.test(c.g) && c.rise_mm === tallest);
log(`  tallest: ${tallest} mm, cleared only under plant(s) ${[...new Set(tallestCells.map(c => `drop ${c.drop}/fric x${c.fmul}`))].join(' ; ')}`);
OUT.clearedRises_mm = clearedRises;
OUT.tallestClearedRise_mm = tallest;
OUT.distinctMoves = new Set(OUT.grids.map(g => g.move)).size;
OUT.holdSurvival = { passes: passes.length, stillHonestAfter100: passes.filter(c => c.honestHeld100).length,
  vaultPasses: vp.length, vaultStillHonestAfter100: vp.filter(c => c.honestHeld100).length,
  vaultHonestAtTrackEnd: vp.filter(c => c.honestAtTrackEnd).length };
OUT.placedPenetration = { episodeMin_mm: +placedPenEp.toFixed(2), atScored_mm: +placedPenNow.toFixed(2) };
OUT.hygiene = { cells: allCells.length, maxTq: +tqMax.toFixed(4), forceCeiling: FR, torqueWithin: tqMax <= FR + 1e-9,
  maxAbsDY_mm: +dyMax.toFixed(1), lateralGate_mm: +(LATERAL * 1000).toFixed(1),
  minPenStep_mm: +penMin.toFixed(2), maxTreadDriftX_mm: +driftMax.toFixed(3),
  minStepGap_mm: gapMin === 1e9 ? null : +gapMin.toFixed(1),
  honestPasses: passes.length, lateralEscapes: badLat.length, overTorque: badTq.length,
  deepPenetration: badPen.length, offFloorSpawn: badSpawn.length };
log('');

// ================================================================== VERDICT
log('=== VERDICT ===');
for (const g of OUT.grids) {
  const x = OUT.extended.find(e => e.file === g.file);
  log(`  ${g.file.padEnd(30)} @${String(g.rise_mm).padStart(3)}mm  claimed ${g.claimedK}/9 -> reproduced ${g.k}/9 (${g.kMatch ? 'MATCH' : 'MISMATCH'})  extended ${x.k}/12  combined ${g.k + x.k}/21`);
}
const bestG = OUT.grids.reduce((a, b) => (b.k > a.k ? b : a));
log(`  best single move on the 9-cell grid: ${bestG.file} at ${bestG.rise_mm} mm, k=${bestG.k}/9`);
log(`  moves reaching k>=7 of 9 (the round-3 bar for "cleared"): ${OUT.grids.filter(g => g.k >= 7).length}`);
OUT.verdict = {
  parityAll,
  reproduced: OUT.grids.map(g => ({ file: g.file, rise_mm: g.rise_mm, claimedK: g.claimedK, k: g.k,
    match: g.kMatch, extendedK: OUT.extended.find(e => e.file === g.file).k,
    combined: g.k + OUT.extended.find(e => e.file === g.file).k })),
  bestK: bestG.k, bestFile: bestG.file, bestRise_mm: bestG.rise_mm,
  anyAtOrAbove7: OUT.grids.filter(g => g.k >= 7).length,
};
fs.writeFileSync(P + 'audit_r3-results.json', JSON.stringify(OUT, null, 2));
fs.writeFileSync(P + 'audit_r3.log', LOG.join('\n') + '\n');
log(`wrote ${P}audit_r3-results.json and ${P}audit_r3.log`);
