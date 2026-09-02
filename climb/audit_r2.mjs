// ROUND 2 AUDIT. Extends audit_cheats.mjs.
//
// Every claimed success re-scored FROM ITS SAVED JSON, at -10/0/+10 mm rise,
// under drop 0.12/0.125/0.13 and foot friction x0.7/x1.3, with and without the
// step-collision-isolation fix that family C's vault run introduced.
//
// The loop below is a verbatim copy of rig3.mjs runEpisodeRaw() with four knobs
// bolted on (drop, foot-friction multiplier, isolate, stepCount). Phase P proves
// it is rig3 at full float digits, so every perturbed number is a rig3 number.
//
// Run from sim/:  node ../climb/audit_r2.mjs
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
import { findStairJoints, layoutStairs, STAIR_Y, STAIR_HALF_WIDTH } from '../site/stairs.js';
import { scoreSaved as rig3Score } from '../climb/rig3.mjs';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/a.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const model = mj.MjModel.mj_loadBinary('/a.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model), ADDR = findStairJoints(model, { isolate: false }); // this script toggles the repair itself; capture the SHIPPED affinity
let GYRO = 0; for (let i = 0; i < model.nsensor; i++) if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;
const stand = await ort.InferenceSession.create('./BEST_alpha_stand.onnx');
const DT = 1 / C.tickHz;
const LATERAL = STAIR_HALF_WIDTH, RISER_X = 0.12;

const bodyId = n => { for (let b = 0; b < model.nbody; b++) if (model.body(b).name === n) return b; return -1; };
const JAWB = bodyId('jaw_soft');
const JAW = []; for (let g = 0; g < model.ngeom; g++) if (model.geom_bodyid[g] === JAWB && !(model.geom_contype[g] === 0 && model.geom_conaffinity[g] === 0)) JAW.push(g);
let STEP0 = -1, STEP1 = -1, LFOOT = -1, RFOOT = -1;
const STEPG = [];
for (let g = 0; g < model.ngeom; g++) {
  const n = model.geom(g).name || '';
  if (n === 'step0_geom') STEP0 = g;
  if (n === 'step1_geom') STEP1 = g;
  if (n === 'left_foot_collision') LFOOT = g;
  if (n === 'right_foot_collision') RFOOT = g;
  if (/^step\d+_geom$/.test(n)) STEPG.push(g);
}
const FEET = []; for (let g = 0; g < model.ngeom; g++) if (/foot_collision|sole/.test(model.geom(g).name || '')) FEET.push(g);
const STEP_CONAFF0 = STEPG.map(g => model.geom_conaffinity[g]);
const FRICT0 = FEET.map(g => model.geom_friction[g * 3]);
// the plant's own ceiling, read not assumed
let FR = 0; for (let a = 0; a < model.nu; a++) FR = Math.max(FR, model.actuator_forcerange[a * 2 + 1]);

const quat = () => [data.qpos[D.freeQpos + 3], data.qpos[D.freeQpos + 4], data.qpos[D.freeQpos + 5], data.qpos[D.freeQpos + 6]];
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

function snapshot(h) {
  const x = data.qpos[D.freeQpos], y = data.qpos[D.freeQpos + 1], z = data.qpos[D.freeQpos + 2];
  const up = projectedGravity(quat())[2] < -0.90;
  let feetUpRaw = 0, feetOnTread = 0, feetResting = 0, feetTouching = 0;
  for (const g of FEET) {
    const fz = data.geom_xpos[g * 3 + 2], fx = data.geom_xpos[g * 3], fy = data.geom_xpos[g * 3 + 1];
    if (fz > h - 0.005 && fx > 0.05) feetUpRaw++;
    const lat = Math.abs(fy - STAIR_Y) <= LATERAL;
    if (fz > h - 0.005 && fx > RISER_X && lat) feetOnTread++;
    // family D's finding: the feet clause has no CEILING. A foot in mid-air
    // above the step passes it. One foot thickness is ~45 mm.
    if (fz > h - 0.005 && fz < h + 0.045 && fx > RISER_X && lat) feetResting++;
    // and the physical test: is the foot actually TOUCHING a step?
    let near = false;
    for (const s of STEPG) if (mj.mj_geomDistance(model, data, g, s, 0.05, null) < 0.003) { near = true; break; }
    if (near && fx > RISER_X && lat && fz > h - 0.005) feetTouching++;
  }
  const foot = g => ({ x: data.geom_xpos[g * 3], y: data.geom_xpos[g * 3 + 1], z: data.geom_xpos[g * 3 + 2] });
  return { x, y, z, dy: y - STAIR_Y, above: z - h, up, feetUpRaw, feetOnTread, feetResting, feetTouching,
           lfoot: foot(LFOOT), rfoot: foot(RFOOT) };
}
const criteria = (h, s) => {
  const lateral = Math.abs(s.dy) <= LATERAL;
  return {
    orig: s.up && s.x > RISER_X && s.above > 0.095 && s.feetUpRaw >= 2,
    honest: s.up && lateral && s.x > RISER_X && s.above > 0.095 && s.feetOnTread >= 2,
    // honest + the missing ceiling on the foot clause
    honestC: s.up && lateral && s.x > RISER_X && s.above > 0.095 && s.feetResting >= 2,
    // honest + the foot must actually be in contact with a step
    honestT: s.up && lateral && s.x > RISER_X && s.above > 0.095 && s.feetTouching >= 2,
    lateral,
  };
};

async function go(track, o, h, { drop = 0.12, fmul = 1.0, isolate = false, stepCount = 4 } = {}) {
  const cfg = { count: stepCount, rise: h, run: 0.28, start: 0.12 };
  STEPG.forEach((g, i) => { model.geom_conaffinity[g] = isolate ? 0 : STEP_CONAFF0[i]; });
  FEET.forEach((g, i) => { model.geom_friction[g * 3] = FRICT0[i] * fmul; });
  mj.mj_resetData(model, data);
  layoutStairs(data, ADDR, cfg);
  if (o.spawn) {
    data.qpos[D.freeQpos] = o.spawn.x; data.qpos[D.freeQpos + 1] = o.spawn.y;
    data.qpos[D.freeQpos + 2] = o.spawn.z + (drop - 0.12);
  } else {
    data.qpos[D.freeQpos] = 0.12 - 0.07 - o.gap;
    data.qpos[D.freeQpos + 1] = STAIR_Y + (o.side || 0);
    data.qpos[D.freeQpos + 2] = drop;
  }
  data.qpos[D.freeQpos + 3] = 1;
  for (let i = 0; i < 14; i++) { data.qpos[D.qpos[i]] = HOME[i]; data.ctrl[i] = HOME[i]; }
  mj.mj_forward(model, data);
  const tr = track.map(f => ({ t: f.t, pose: f.pose.slice() }));
  let la = new Array(14).fill(0);
  const cmd = command({ vx: o.approach || 0 });
  let maxTq = 0, minPen = 1e9, maxAbsDY = 0, maxX = -1e9, maxZ = -1e9, minStepGap = 1e9, maxDriftX = 0;
  let feetOnTreadMax = 0, feetTouchMax = 0;
  const step = async (off) => {
    layoutStairs(data, ADDR, cfg);
    const q = quat(); const jp = [], jv = [];
    for (let k = 0; k < 14; k++) { jp.push(data.qpos[D.qpos[k]]); jv.push(data.qvel[D.dof[k]]); }
    const obs = buildObs([data.sensordata[GYRO], data.sensordata[GYRO + 1], data.sensordata[GYRO + 2]], projectedGravity(q), jp, jv, la, cmd);
    const r = await stand.run({ obs: new ort.Tensor('float32', obs, [1, 61]) });
    la = Array.from(r.actions.data);
    for (let k = 0; k < 14; k++) {
      const v = HOME[k] + la[k] + (off ? (off[k] - HOME[k]) * o.blend : 0);
      data.ctrl[k] = Math.min(Math.max(v, LO[k]), HI[k]);
    }
    for (let s = 0; s < 4; s++) mj.mj_step(model, data);
    for (let a = 0; a < model.nu; a++) maxTq = Math.max(maxTq, Math.abs(data.actuator_force[a]));
    for (const g of JAW.concat(FEET)) for (const s of [STEP0, STEP1]) if (s >= 0)
      minPen = Math.min(minPen, mj.mj_geomDistance(model, data, g, s, 0.05, null));
    const dy = Math.abs(data.qpos[D.freeQpos + 1] - STAIR_Y);
    if (dy > maxAbsDY) maxAbsDY = dy;
    if (data.qpos[D.freeQpos] > maxX) maxX = data.qpos[D.freeQpos];
    if (data.qpos[D.freeQpos + 2] > maxZ) maxZ = data.qpos[D.freeQpos + 2];
    if (STEP1 >= 0 && cfg.count > 1) minStepGap = Math.min(minStepGap, mj.mj_geomDistance(model, data, STEP0, STEP1, 0.4, null));
    maxDriftX = Math.max(maxDriftX, Math.abs(data.geom_xpos[STEP0 * 3] - (0.12 + 0.17)));
    const s = snapshot(h);
    if (s.feetOnTread > feetOnTreadMax) feetOnTreadMax = s.feetOnTread;
    if (s.feetTouching > feetTouchMax) feetTouchMax = s.feetTouching;
  };
  for (let t = 0; t < 25; t++) await step(null);
  const total = tr[tr.length - 1].t + 0.8;
  for (let t = 0; t * DT < total; t++) await step(poseAt(tr, t * DT));
  for (let t = 0; t < 50; t++) await step(null);
  const s = snapshot(h);
  return { scored: s, crit: criteria(h, s), maxTq, minPen, maxAbsDY, maxX, maxZ,
           minStepGap: minStepGap === 1e9 ? null : minStepGap, maxDriftX, feetOnTreadMax, feetTouchMax };
}

const P = '../climb/';
const load1 = f => JSON.parse(fs.readFileSync(P + f, 'utf8'));
const optsOf = j => ({ blend: j.blend, approach: j.approach || 0, gap: j.gap || 0, side: j.side || 0, spawn: j.spawn || null });
const mm = v => (v * 1000).toFixed(1);
const out = { generated: new Date().toISOString(), forceCeiling: FR, lateral: LATERAL, rows: [], parity: [] };
const log = s => { console.log(s); };

log(`plant: scene.mjb  actuator forcerange ceiling read from the model = +/-${FR.toFixed(4)} N.m`);
log(`flight half-width ${mm(LATERAL)} mm ; riser face x = ${mm(RISER_X)} mm ; nominal foot friction ${FRICT0[0]}`);
log('');

// ------------------------------------------------------------------ PHASE P
log('=== PHASE P: is this loop rig3? (same file, same rise, isolate off, nominal) ===');
const PAR = ['best_r2_vault_40mm.json', 'best_r2_vault_60mm.json', 'best_r2_vault_90mm.json',
             'best_r2_cornerstem_180mm.json', 'best_r2_cornerstem_120mm.json',
             'best_r2_blockclimb_90mm.json', 'best_r2_blockclimb_180mm.json'];
let parityAll = true;
for (const f of PAR) {
  const h = parseInt(f.match(/_(\d+)mm/)[1], 10) / 1000;
  const j = load1(f);
  const a = await go(j.keyframes, optsOf(j), h, {});
  const b = await rig3Score(P + f, { rise: h, tail: 'policy' });
  const dx = Math.abs(a.scored.x - b.scored.x), dz = Math.abs(a.scored.z - b.scored.z);
  const ok = dx === 0 && dz === 0 && a.scored.feetOnTread === b.scored.feetOnTread && a.crit.honest === b.crit.honest;
  if (!ok) parityAll = false;
  out.parity.push({ file: f, mine_x: a.scored.x, rig3_x: b.scored.x, mine_z: a.scored.z, rig3_z: b.scored.z, exact: ok });
  log(`  ${f.padEnd(32)} mine x=${a.scored.x} z=${a.scored.z}  rig3 x=${b.scored.x} z=${b.scored.z}  EXACT=${ok}`);
}
log(`  parityAll = ${parityAll}`);
log('');

const hdr = 'file                            rise iso drop  fric  | HONEST honestC honestT orig |    x_mm   dy_mm    z_mm above_mm fT fC fR | maxTq   pen_mm  maxDY_mm stepGap_mm drift_mm';
const row = (label, rise, o, r) => {
  const s = r.scored;
  const line = `${label.padEnd(31)} ${String(Math.round(rise * 1000)).padStart(4)} ${o.isolate ? ' Y ' : ' n '} ${(o.drop ?? 0.12).toFixed(3)} ${(o.fmul ?? 1).toFixed(2)}  | ` +
    `${String(r.crit.honest).padStart(6)} ${String(r.crit.honestC).padStart(7)} ${String(r.crit.honestT).padStart(7)} ${String(r.crit.orig).padStart(5)} | ` +
    `${mm(s.x).padStart(8)} ${mm(s.dy).padStart(7)} ${mm(s.z).padStart(7)} ${mm(s.above).padStart(8)} ${s.feetTouching}  ${s.feetResting}  ${s.feetOnTread} | ` +
    `${r.maxTq.toFixed(4)} ${mm(r.minPen).padStart(7)} ${mm(r.maxAbsDY).padStart(8)} ${(r.minStepGap === null ? '     n/a' : mm(r.minStepGap)).padStart(10)} ${mm(r.maxDriftX).padStart(8)}`;
  log(line);
  out.rows.push({ label, rise_mm: Math.round(rise * 1000), isolate: !!o.isolate, drop: o.drop ?? 0.12, fmul: o.fmul ?? 1,
    honest: r.crit.honest, honestCeil: r.crit.honestC, honestTouch: r.crit.honestT, orig: r.crit.orig,
    x_mm: +mm(s.x), dy_mm: +mm(s.dy), z_mm: +mm(s.z), above_mm: +mm(s.above),
    feetOnTread: s.feetOnTread, feetResting: s.feetResting, feetTouching: s.feetTouching,
    lfootZ_mm: +mm(s.lfoot.z), rfootZ_mm: +mm(s.rfoot.z), lfootX_mm: +mm(s.lfoot.x), rfootX_mm: +mm(s.rfoot.x),
    up: s.up, maxTq: +r.maxTq.toFixed(4), minPen_mm: +mm(r.minPen), maxAbsDY_mm: +mm(r.maxAbsDY),
    minStepGap_mm: r.minStepGap === null ? null : +mm(r.minStepGap), maxDriftX_mm: +mm(r.maxDriftX),
    feetOnTreadMax: r.feetOnTreadMax, feetTouchMax: r.feetTouchMax });
  return r;
};

// ------------------------------------------------------------------ PHASE A
log('=== PHASE A: the three claimed vault results, -10/0/+10 mm, isolate off then on ===');
log(hdr);
for (const f of ['best_r2_vault_40mm.json', 'best_r2_vault_60mm.json', 'best_r2_vault_90mm.json']) {
  const h0 = parseInt(f.match(/_(\d+)mm/)[1], 10) / 1000;
  const j = load1(f), o = optsOf(j);
  for (const iso of [false, true]) for (const dh of [-0.010, 0, 0.010]) {
    row(f.replace('best_r2_', '').replace('.json', ''), h0 + dh, { isolate: iso }, await go(j.keyframes, o, h0 + dh, { isolate: iso }));
  }
}
log('');

// ------------------------------------------------------------------ PHASE B
log('=== PHASE B: perturbations at the trained rise (drop 0.125/0.13, foot friction x0.7/x1.3) ===');
log(hdr);
for (const f of ['best_r2_vault_40mm.json', 'best_r2_vault_60mm.json']) {
  const h0 = parseInt(f.match(/_(\d+)mm/)[1], 10) / 1000;
  const j = load1(f), o = optsOf(j);
  for (const iso of [false, true]) for (const v of [{ drop: 0.125 }, { drop: 0.130 }, { fmul: 0.7 }, { fmul: 1.3 }]) {
    row(f.replace('best_r2_', '').replace('.json', ''), h0, { isolate: iso, ...v }, await go(j.keyframes, o, h0, { isolate: iso, ...v }));
  }
}
log('');

// ------------------------------------------------------------------ PHASE C
log('=== PHASE C: controls on the SAME plant the clears used (isolate on) and off ===');
log(hdr);
for (const rise of [0.040, 0.060, 0.090]) {
  const rm = Math.round(rise * 1000);
  for (const iso of [false, true]) {
    for (const [nm, file] of [['do-nothing', 'ctrl_do_nothing.json'], ['walk-only', 'ctrl_walk_only.json'],
                              ['on-tread-x26', `ctrl_on_tread_${rm}mm_x26_med.json`]]) {
      const j = load1(file);
      row(`${nm}@${rm}`, rise, { isolate: iso }, await go(j.keyframes, optsOf(j), rise, { isolate: iso }));
    }
  }
}
log('');

// ------------------------------------------------------------------ PHASE D
log('=== PHASE D: the two zero-clear families, re-scored from file at -10/0/+10 (isolate off) ===');
log(hdr);
for (const f of ['best_r2_cornerstem_120mm.json', 'best_r2_cornerstem_180mm.json',
                 'best_r2_blockclimb_90mm.json', 'best_r2_blockclimb_180mm.json']) {
  const h0 = parseInt(f.match(/_(\d+)mm/)[1], 10) / 1000;
  const j = load1(f), o = optsOf(j);
  for (const dh of [-0.010, 0, 0.010])
    row(f.replace('best_r2_', '').replace('.json', ''), h0 + dh, {}, await go(j.keyframes, o, h0 + dh, {}));
}
log('');

// ------------------------------------------------------------------ summary
const claimed = out.rows.filter(r => /vault_(40|60)mm/.test(r.label));
const cleared = out.rows.filter(r => r.honest);
log('=== SUMMARY ===');
log(`rows: ${out.rows.length}   HONEST passes: ${cleared.length}`);
for (const r of cleared) log(`  PASS  ${r.label} rise=${r.rise_mm} iso=${r.isolate} drop=${r.drop} fric=${r.fmul}  x=${r.x_mm} above=${r.above_mm} feetOnTread=${r.feetOnTread} feetResting=${r.feetResting} feetTouching=${r.feetTouching}`);
const tqMax = Math.max(...out.rows.map(r => r.maxTq));
const penMin = Math.min(...out.rows.map(r => r.minPen_mm));
const dyMax = Math.max(...out.rows.map(r => r.maxAbsDY_mm));
log(`max |actuator_force| across every row: ${tqMax.toFixed(4)} N.m  (plant ceiling ${FR.toFixed(4)})`);
log(`most-negative geom distance duck<->step across every row: ${penMin.toFixed(2)} mm`);
log(`largest |y - STAIR_Y| across every row: ${dyMax.toFixed(1)} mm (gate ${mm(LATERAL)} mm)`);
out.summary = { rows: out.rows.length, honestPasses: cleared.length, maxTq: tqMax, minPen_mm: penMin, maxAbsDY_mm: dyMax, parityAll };
fs.writeFileSync(P + 'audit_r2-results.json', JSON.stringify(out, null, 2));
log(`wrote ${P}audit_r2-results.json`);
