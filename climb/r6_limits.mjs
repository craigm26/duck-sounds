// r6_limits.mjs — ROUND 6, PHASE 3: WHAT PHYSICALLY LIMITS THE CEILING.
//
// The ceiling search answers "how high does the trunk get". This answers WHY it
// stops there. It is a DIAGNOSTIC, never a scorer: no verdict, no k, no
// objective is produced here, and nothing in it feeds a published result. Every
// number that decides anything still comes from climb/robust.mjs.
//
// It needs per-actuator torque per tick, which robust.mjs does not expose
// (it keeps only maxTq over all 14 actuators). load() from 'mujoco' returns a
// NEW instance per call — checked — so robust.mjs's mj_step cannot be wrapped
// from outside. So the episode loop below is a copy of robust.mjs go(), minus
// the parts no file here uses (event, servo, handoff spawn), and it is GATED:
//
//   PARITY. For every core cell of every file it looks at, this loop's peak
//   trunk height and peak |actuator force| must equal what robust.mjs
//   scoreRobust reports for the same cell, to 0.01 mm and 1e-6 N.m. If a single
//   cell disagrees the run stops and prints the disagreement. A diagnostic that
//   is not running the same episode is worth nothing.
//
// Run from sim/:  node ../climb/r6_limits.mjs [file ...]
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
import { findStairJoints, layoutStairs, STAIR_Y } from '../site/stairs.js';
// poseAt is IMPORTED from the scorer, not copied: the first version of this
// file re-implemented it as a linear interpolation from the first keyframe and
// the parity gate caught it on all 9 cells (up to 70 mm of peak height). The
// real one smoothsteps from HOME. Nothing that shapes the trajectory is copied.
import { scoreRobust, PLANTS, DHS, poseAt } from './robust.mjs';

const P = '../climb/';
const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/l.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const model = mj.MjModel.mj_loadBinary('/l.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model);
const ADDR = findStairJoints(model, { isolate: false });
let GYRO = 0;
for (let i = 0; i < model.nsensor; i++) if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;
const stand = await ort.InferenceSession.create('./BEST_alpha_stand.onnx');
const DT = 1 / C.tickHz;
const RISER_X = 0.12;

const STEPG = [], FEET = [];
let STEP0 = -1;
for (let g = 0; g < model.ngeom; g++) {
  const n = model.geom(g).name || '';
  if (/^step\d+_geom$/.test(n)) STEPG.push(g);
  if (n === 'step0_geom') STEP0 = g;
  if (/foot_collision|sole/.test(n)) FEET.push(g);
}
const STEP_CONAFF0 = STEPG.map(g => model.geom_conaffinity[g]);
const FRICT0 = FEET.map(g => model.geom_friction[g * 3]);
const bodyId = n => { for (let b = 0; b < model.nbody; b++) if (model.body(b).name === n) return b; return -1; };
const jntId = n => { for (let j = 0; j < model.njnt; j++) if (model.jnt(j).name === n) return j; return -1; };
const JAWB = bodyId('jaw_soft');
const NECKJ = jntId('neck_pitch'), HEADJ = jntId('head_pitch');
const quat = () => [data.qpos[D.freeQpos + 3], data.qpos[D.freeQpos + 4], data.qpos[D.freeQpos + 5], data.qpos[D.freeQpos + 6]];

// duck slot -> actuator. data.ctrl[k] is indexed by duck slot in every loop in
// this repo, so actuator k IS duck slot k; printed once so it is checkable.
const ANAME = [], FHI = [];
for (let a = 0; a < model.nu; a++) { ANAME.push(model.actuator(a).name || `#${a}`); FHI.push(model.actuator_forcerange[a * 2 + 1]); }
const SLOT = { lhy: 0, lhr: 1, lhp: 2, lk: 3, la: 4, np: 5, hp: 6, hy: 7, hr: 8, rhy: 9, rhr: 10, rhp: 11, rk: 12, ra: 13 };
const GROUPS = { hipPitch: [SLOT.lhp, SLOT.rhp], knee: [SLOT.lk, SLOT.rk], ankle: [SLOT.la, SLOT.ra],
                 hipRoll: [SLOT.lhr, SLOT.rhr], hipYaw: [SLOT.lhy, SLOT.rhy],
                 neckPitch: [SLOT.np], headPitch: [SLOT.hp] };
const SATEPS = 1e-4;                       // N.m below the forcerange edge

/** ONE INSTRUMENTED EPISODE. Copy of robust.mjs go() for a file with no event,
 *  no servo and no handoff spawn. Returns a per-tick trace and nothing else. */
async function trace(j, h, { drop = 0.120, fmul = 1.0, isolate = true, stepCount = 4 } = {}) {
  const o = { blend: j.blend, approach: j.approach || 0, gap: j.gap || 0, side: j.side || 0 };
  const cfg = { count: stepCount, rise: h, run: 0.28, start: 0.12 };
  STEPG.forEach((g, i) => { model.geom_conaffinity[g] = isolate ? 0 : STEP_CONAFF0[i]; });
  FEET.forEach((g, i) => { model.geom_friction[g * 3] = FRICT0[i] * fmul; });
  mj.mj_resetData(model, data);
  layoutStairs(data, ADDR, cfg);
  data.qpos[D.freeQpos] = 0.12 - 0.07 - o.gap;
  data.qpos[D.freeQpos + 1] = STAIR_Y + o.side;
  data.qpos[D.freeQpos + 2] = drop;
  data.qpos[D.freeQpos + 3] = 1;
  for (let i = 0; i < 14; i++) { data.qpos[D.qpos[i]] = HOME[i]; data.ctrl[i] = HOME[i]; }
  mj.mj_forward(model, data);
  const tr = j.keyframes.map(f => ({ t: f.t, pose: f.pose.slice() }));
  let la = new Array(14).fill(0);
  const cmd = command({ vx: o.approach });
  const T = [];
  let maxTq = 0, maxZ = -1e9;
  const step = async (off, rec, phase, tTrack) => {
    layoutStairs(data, ADDR, cfg);
    const q = quat(); const jp = [], jv = [];
    for (let k = 0; k < 14; k++) { jp.push(data.qpos[D.qpos[k]]); jv.push(data.qvel[D.dof[k]]); }
    const obs = buildObs([data.sensordata[GYRO], data.sensordata[GYRO + 1], data.sensordata[GYRO + 2]], projectedGravity(q), jp, jv, la, cmd);
    const r = await stand.run({ obs: new ort.Tensor('float32', obs, [1, 61]) });
    la = Array.from(r.actions.data);
    const clipped = [];
    for (let k = 0; k < 14; k++) {
      const v = HOME[k] + la[k] + (off ? (off[k] - HOME[k]) * o.blend : 0);
      const c = Math.min(Math.max(v, LO[k]), HI[k]);
      data.ctrl[k] = c;
      clipped.push(c <= LO[k] + 1e-9 || c >= HI[k] - 1e-9);
    }
    for (let s = 0; s < 4; s++) mj.mj_step(model, data);
    if (rec) {
      const f = [], sat = [];
      for (let a = 0; a < model.nu; a++) {
        const v = Math.abs(data.actuator_force[a]);
        f.push(v); sat.push(v >= FHI[a] - SATEPS);
        if (v > maxTq) maxTq = v;
      }
      const z = data.qpos[D.freeQpos + 2];
      if (z > maxZ) maxZ = z;
      // NECK STRUT GEOMETRY: horizontal distance from the neck_pitch anchor to
      // the jaw body, which is the lever the neck torque acts through when it
      // holds the trunk up over a planted beak.
      const na = [data.xanchor[NECKJ * 3], data.xanchor[NECKJ * 3 + 1], data.xanchor[NECKJ * 3 + 2]];
      const jw = [data.xpos[JAWB * 3], data.xpos[JAWB * 3 + 1], data.xpos[JAWB * 3 + 2]];
      const armH = Math.hypot(jw[0] - na[0], jw[1] - na[1]);
      T.push({ phase, tTrack, z, x: data.qpos[D.freeQpos], f, sat, clipped, armH,
               pg2: projectedGravity(quat())[2] });
    }
  };
  for (let t = 0; t < 25; t++) await step(null, false, 'settle', null);
  const total = tr[tr.length - 1].t + 0.8;
  for (let t = 0; t * DT < total; t++) await step(poseAt(tr, t * DT), true, 'track', t * DT);
  for (let t = 0; t < 50; t++) await step(null, true, 'tail', null);
  return { T, maxTq, maxZ };
}

// ------------------------------------------------------------------ windows
const frac = (T, idx, slots) => {
  if (!idx.length) return null;
  let n = 0;
  for (const i of idx) if (slots.some(s => T[i].sat[s])) n++;
  return +(n / idx.length).toFixed(4);
};
function windows(T, kf) {
  const trackIdx = T.map((r, i) => [r, i]).filter(([r]) => r.phase === 'track').map(([, i]) => i);
  // THE PUSH-OFF: the rise itself — from the lowest trunk z during the track to
  // the tick the trunk peaks. That is when the legs are making height.
  let lo = trackIdx[0], hi = trackIdx[0];
  for (const i of trackIdx) if (T[i].z < T[lo].z) lo = i;
  for (const i of trackIdx) if (i > lo && T[i].z > T[hi].z) hi = i;
  if (hi <= lo) hi = trackIdx[trackIdx.length - 1];
  const push = []; for (let i = lo; i <= hi; i++) push.push(i);
  // the vault SEGMENT, keyframe B -> keyframe D (preload through tuck)
  const segIdx = (t0, t1) => trackIdx.filter(i => T[i].tTrack >= t0 && T[i].tTrack <= t1);
  const seg = {};
  const names = ['reach', 'pre', 'vault', 'tuck', 'land', 'return'];
  for (let s = 0; s < kf.length; s++) seg[names[s] || `k${s}`] = segIdx(s === 0 ? 0 : kf[s - 1].t, kf[s].t);
  return { trackIdx, push, lo, hi, seg };
}

// ------------------------------------------------------------------ run
const FILES = process.argv.slice(2).length ? process.argv.slice(2)
  : ['best_r3_vault_60mm.json'];
const RISE = 0.060;
const OUT = { generated: new Date().toISOString(),
  what: 'ROUND 6 — what physically limits the ceiling',
  role: 'DIAGNOSTIC ONLY. No verdict, k, or objective is produced here.',
  parity: 'every core cell is checked against climb/robust.mjs scoreRobust: peak trunk height to 0.01 mm, peak |actuator force| to 1e-6 N.m',
  forcerange_Nm: FHI[0], actuatorsInSlotOrder: ANAME,
  neckStall_N: 7.66, bodyWeight_N: 7.23,
  rise_mm: 60, files: [] };

console.log(`actuators in duck-slot order: ${ANAME.join(' ')}`);
console.log(`forcerange upper: ${[...new Set(FHI.map(v => v.toFixed(4)))].join(' / ')} N.m\n`);

for (const file of FILES) {
  const j = JSON.parse(fs.readFileSync(P + file, 'utf8'));
  if (j.event || j.servo || j.spawn || j.spawnPose) { console.log(`SKIP ${file} — has event/servo/spawn, outside this diagnostic's copy`); continue; }
  const ref = await scoreRobust(P + file, { rise: RISE, core: true });
  const rows = [];
  let parityOK = true;
  for (let ci = 0; ci < 9; ci++) {
    const dh = DHS[Math.floor(ci / 3)], pl = PLANTS[ci % 3];
    const h = RISE + dh;
    const tr = await trace(j, h, { drop: pl.drop, fmul: pl.fmul, isolate: j.isolate !== false, stepCount: j.stepCount || 4 });
    const v = ref.verdicts.filter(x => x.tier === 'core')[ci];
    const cRef = ref.cells.filter(x => x.cell.tier === 'core')[ci];
    const dz = Math.abs(tr.maxZ - cRef.maxZ) * 1000, dtq = Math.abs(tr.maxTq - cRef.maxTq);
    if (dz > 1e-5 || dtq > 1e-6) {
      parityOK = false;
      console.log(`  PARITY FAIL cell ${ci}: peakZ mine ${(tr.maxZ * 1000).toFixed(6)} mm vs robust ${(cRef.maxZ * 1000).toFixed(6)} mm (d ${dz.toFixed(6)} mm); maxTq mine ${tr.maxTq.toFixed(9)} vs ${cRef.maxTq.toFixed(9)}`);
    }
    const W = windows(tr.T, j.keyframes);
    const peakAbove_mm = +(tr.maxZ * 1000 - v.rise_mm).toFixed(1);
    const g = idx => Object.fromEntries(Object.entries(GROUPS).map(([k, s]) => [k, frac(tr.T, idx, s)]));
    const anySat = idx => frac(tr.T, idx, [...Array(14).keys()]);
    // NECK STRUT: the vertical force the neck can make at the beak, through the
    // horizontal lever it actually had, during the push-off.
    const arms = W.push.map(i => tr.T[i].armH).filter(a => a > 1e-4);
    const armMin = Math.min(...arms), armMax = Math.max(...arms);
    const neckF = a => FHI[SLOT.np] / a;
    rows.push({ cell: ci, rise_mm: v.rise_mm, drop: pl.drop, fmul: pl.fmul,
      peakAboveTread_mm: peakAbove_mm, over95: peakAbove_mm > 95,
      parity: { myPeakZ_mm: +(tr.maxZ * 1000).toFixed(6), robustPeakZ_mm: +(cRef.maxZ * 1000).toFixed(6),
                dz_mm: +dz.toFixed(8), myMaxTq: +tr.maxTq.toFixed(9), robustMaxTq: +cRef.maxTq.toFixed(9),
                dTq_Nm: +dtq.toFixed(9), exact: dz <= 1e-5 && dtq <= 1e-6 },
      pushOffTicks: W.push.length, pushOffMs: W.push.length * 20,
      peakAtTick: W.hi, peakAtTrackTime_s: tr.T[W.hi].tTrack,
      peakDuringTrack: tr.T[W.hi].phase === 'track',
      trackTicks: W.trackIdx.length,
      torqueCeilingFrac_pushOff: g(W.push),
      torqueCeilingFrac_track: g(W.trackIdx),
      anyActuatorAtCeiling_pushOff: anySat(W.push),
      commandClipFrac_pushOff: +(W.push.filter(i => tr.T[i].clipped.some(Boolean)).length / Math.max(W.push.length, 1)).toFixed(4),
      neckLeverH_mm: { min: +(armMin * 1000).toFixed(1), max: +(armMax * 1000).toFixed(1) },
      neckMaxForceAtBeak_N: { atMinLever: +neckF(armMin).toFixed(3), atMaxLever: +neckF(armMax).toFixed(3) },
      segmentTorqueCeilingFrac: Object.fromEntries(Object.entries(W.seg).map(([k, idx]) => [k, g(idx)])),
    });
    console.log(`  ${file}  cell ${ci} rise ${v.rise_mm} d${pl.drop} f${pl.fmul}: peak ${peakAbove_mm} mm  push-off ${W.push.length} ticks  hip@ceiling ${(rows[ci].torqueCeilingFrac_pushOff.hipPitch * 100).toFixed(0)}%  knee ${(rows[ci].torqueCeilingFrac_pushOff.knee * 100).toFixed(0)}%  ankle ${(rows[ci].torqueCeilingFrac_pushOff.ankle * 100).toFixed(0)}%  neck ${(rows[ci].torqueCeilingFrac_pushOff.neckPitch * 100).toFixed(0)}%  parity dz ${dz.toFixed(6)} mm dTq ${dtq.toFixed(9)}`);
  }
  const mean = f => +(rows.reduce((a, r) => a + f(r), 0) / rows.length).toFixed(4);
  const summary = {
    meanPeakAboveTread_mm: mean(r => r.peakAboveTread_mm),
    ceilingCore: rows.filter(r => r.over95).length,
    meanPushOffTicks: mean(r => r.pushOffTicks),
    meanTorqueCeilingFrac_pushOff: Object.fromEntries(Object.keys(GROUPS).map(k =>
      [k, mean(r => r.torqueCeilingFrac_pushOff[k])])),
    meanAnyActuatorAtCeiling_pushOff: mean(r => r.anyActuatorAtCeiling_pushOff),
    meanCommandClipFrac_pushOff: mean(r => r.commandClipFrac_pushOff),
    neckLeverH_mm_min: Math.min(...rows.map(r => r.neckLeverH_mm.min)),
    neckLeverH_mm_max: Math.max(...rows.map(r => r.neckLeverH_mm.max)),
  };
  OUT.files.push({ file, sha256: ref.sha256, move: ref.move, parityOK, parityCells: `${rows.filter(r => r.parity.exact).length}/9 EXACT`, rows, summary });
  console.log(`  ${file} PARITY vs robust.mjs: ${rows.filter(r => r.parity.exact).length}/9 cells EXACT`);
  console.log(`  ${file} SUMMARY: ceilingCore ${summary.ceilingCore}/9, mean peak ${summary.meanPeakAboveTread_mm} mm, push-off ${summary.meanPushOffTicks} ticks`);
  console.log(`    fraction of push-off ticks with the actuator AT its 0.6405 N.m forcerange:`);
  for (const [k, v] of Object.entries(summary.meanTorqueCeilingFrac_pushOff)) console.log(`      ${k.padEnd(10)} ${(v * 100).toFixed(1)}%`);
  console.log(`    any of the 14 at the ceiling: ${(summary.meanAnyActuatorAtCeiling_pushOff * 100).toFixed(1)}% of push-off ticks`);
  console.log(`    position command clipped to a joint limit: ${(summary.meanCommandClipFrac_pushOff * 100).toFixed(1)}% of push-off ticks`);
  console.log(`    neck_pitch horizontal lever during push-off: ${summary.neckLeverH_mm_min}..${summary.neckLeverH_mm_max} mm`);
  console.log(`    -> max vertical force the neck can hold at the beak: ${(FHI[SLOT.np] / (summary.neckLeverH_mm_max / 1000)).toFixed(2)}..${(FHI[SLOT.np] / (summary.neckLeverH_mm_min / 1000)).toFixed(2)} N  (body weight 7.23 N, neck stall 7.66 N)\n`);
}
fs.writeFileSync(`${P}r6_limits-results.json`, JSON.stringify(OUT, null, 1) + '\n');
console.log('wrote climb/r6_limits-results.json');
