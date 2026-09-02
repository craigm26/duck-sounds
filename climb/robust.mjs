// robust.mjs — THE ROUND-3 SCORER. One copy, imported by every family.
//
// Round 2 reported single-cell results and then discovered, in audit_r2, that
// a 40 mm clear survived 1 of 7 perturbations and a 60 mm clear 3 of 7. A move
// that passes once is a coincidence. So round 3 scores a SAVED FILE over a
// fixed 9-cell grid and reports "cleared k of 9":
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
import { makeLoop } from '../site/duckloop.mjs';
import { findStairJoints, layoutStairs, STAIR_Y, STAIR_HALF_WIDTH } from '../site/stairs.js';
import { criteria as rig3Criteria, reward as rig3Reward } from '../climb/rig3.mjs';

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

/** rig3.mjs snapshot() fields that criteria() and reward() read. */
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
  mj.mj_forward(model, data);
  const tr = track.map(f => ({ t: f.t, pose: f.pose.slice() }));
  let la = new Array(14).fill(0);
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

  for (let t = 0; t < 25; t++) await step(null, false);
  const x0 = data.qpos[D.freeQpos];
  Z0 = data.qpos[D.freeQpos + 2];
  const total = tr[tr.length - 1].t + 0.8;
  for (let t = 0; t * DT < total; t++) await step(poseAt(tr, t * DT), true);
  const atTrackEnd = snapshot(h);
  for (let t = 0; t < 50; t++) await step(null, true);   // tail 'policy' — climb_lib's own
  const scored = snapshot(h);

  const rec = {
    rise: h, x0, scored, atTrackEnd,
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
/** Bonus added to the objective for each of the 9 cells cleared under 'honest'. */
export const CLEAR_BONUS = 4;

const readIntent = path => {
  const j = JSON.parse(fs.readFileSync(path, 'utf8'));
  if (!Array.isArray(j.keyframes) || !j.keyframes.length) throw new Error('no keyframes in ' + path);
  for (const f of j.keyframes) if (!Array.isArray(f.pose) || f.pose.length !== 14) throw new Error('bad pose in ' + path);
  return j;
};
const optsOf = j => ({ blend: j.blend, approach: j.approach || 0, gap: j.gap || 0, side: j.side || 0,
                       spawn: j.spawn || null });

/**
 * Score ONE cell of the grid, from a SAVED file. Used for cheap screening;
 * a reported result must always come from scoreRobust().
 */
export async function scoreCell(path, { rise, dh = 0, drop = 0.120, fmul = 1.0, isolate, stepCount } = {}) {
  const j = readIntent(path);
  return go(j.keyframes, optsOf(j), rise + dh,
    { drop, fmul, isolate: isolate === undefined ? (j.isolate !== false) : isolate,
      stepCount: stepCount || j.stepCount || 4 });
}

/**
 * THE ROUND-3 SCORER. Scores a SAVED file over all 9 cells.
 *
 * Returns k (cells cleared under rig3's `honest`), the mean of rig3's reward
 * across the 9 cells, and objective = meanReward + CLEAR_BONUS * k, plus the
 * per-cell records so a family can shape on load transfer without a second
 * scorer.
 */
export async function scoreRobust(path, { rise, isolate, stepCount, onCell } = {}) {
  const j = readIntent(path);
  const o = optsOf(j);
  const iso = isolate === undefined ? (j.isolate !== false) : isolate;
  const sc = stepCount || j.stepCount || 4;
  const cells = [];
  for (const dh of DHS) for (const p of PLANTS) {
    const r = await go(j.keyframes, o, rise + dh, { drop: p.drop, fmul: p.fmul, isolate: iso, stepCount: sc });
    r.cell = { rise_mm: Math.round((rise + dh) * 1000), drop: p.drop, fmul: p.fmul };
    cells.push(r);
    if (onCell) onCell(r);
  }
  const k = cells.filter(c => c.crit.honest).length;
  const mean = f => cells.reduce((a, c) => a + f(c), 0) / cells.length;
  const meanReward = mean(c => c.reward);
  return {
    source: path, rise, k, meanReward,
    objective: meanReward + CLEAR_BONUS * k,
    cells,
    agg: {
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
      honest: c.crit.honest, reward: +c.reward.toFixed(3),
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
const isMain = process.argv[1] && process.argv[1].endsWith('robust.mjs');
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
  console.log(`=== 9-cell grid on best_r2_vault_60mm @60mm: k=${g.k}/9 meanReward=${g.meanReward.toFixed(3)} objective=${g.objective.toFixed(3)}  (${dt.toFixed(1)} s, ${(dt / 9).toFixed(2)} s/cell) ===`);
  for (const v of g.verdicts) console.log(`   rise=${v.rise_mm} drop=${v.drop} f=${v.fmul} honest=${v.honest} rew=${v.reward} x=${v.x_mm} above=${v.above_mm} fot=${v.feetOnTread} peakZ=${v.peakZ_mm} head=${v.headFrac} wall=${v.wallFrac}`);
  const g9 = await scoreRobust('../climb/best_r2_vault_90mm.json', { rise: 0.090 });
  console.log(`=== 9-cell grid on best_r2_vault_90mm @90mm: k=${g9.k}/9 meanReward=${g9.meanReward.toFixed(3)} objective=${g9.objective.toFixed(3)} peakGain=${(g9.agg.meanPeakGain * 1000).toFixed(1)}mm headFrac=${g9.agg.headFrac.toFixed(3)} ===`);
}
