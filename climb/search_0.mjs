// Strategy A — BEAK HOOK + WALL WALK. A keyframe optimiser.
//
// The four authored moves fail the same way (climb/why.mjs): the duck stands
// upright on the floor short of the riser. This searches the six-frame skeleton
// that plants the head ON the tread first, then walks the soles UP the riser
// face while the neck hauls the trunk over.
//
// It does NOT edit sim/. It copies sim/climb_lib.mjs's attempt() loop verbatim
// (lines 118-152 there) and adds per-tick sampling, because attempt() returns
// only the terminal five numbers (sim/climb_lib.mjs:151) and the shaped reward
// needs peaks, contact fractions and saturation.
//
// Run from sim/:
//   cd ~/projects/duck-sounds/sim && node ../climb/search_0.mjs --rise 40 --seconds 300 --seed 1
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
import { findStairJoints, layoutStairs, STAIR_Y } from '../site/stairs.js';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const model = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model), ADDR = findStairJoints(model);
let GYRO = 0;
for (let i = 0; i < model.nsensor; i++) if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;
const stand = await ort.InferenceSession.create('./BEST_alpha_stand.onnx');
const DT = 1 / C.tickHz;

// sim/climb_lib.mjs:99-105 poseAt, copied so this file does not import that
// module: importing it loads a SECOND mujoco model + onnx session and the WASM
// heap hits its 2 GB cap.
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

// site/intent.mjs:14-18 order.
const J = { lhy: 0, lhr: 1, lhp: 2, lk: 3, la: 4, np: 5, hp: 6, hy: 7, hr: 8, rhy: 9, rhr: 10, rhp: 11, rk: 12, ra: 13 };

// geoms, by name / body, never by index.
const geomId = n => { for (let g = 0; g < model.ngeom; g++) if (model.geom(g).name === n) return g; return -1; };
const bodyId = n => { for (let b = 0; b < model.nbody; b++) if (model.body(b).name === n) return b; return -1; };
const STEP0 = geomId('step0_geom');
const LFOOT = geomId('left_foot_collision'), RFOOT = geomId('right_foot_collision');
const HEADG = new Set();
{
  const jaw = bodyId('jaw_soft'), head = bodyId('neck_pitch');
  for (let g = 0; g < model.ngeom; g++) {
    const b = model.geom_bodyid[g];
    if ((b === jaw || b === head) && !(model.geom_contype[g] === 0 && model.geom_conaffinity[g] === 0)) HEADG.add(g);
  }
}

// ---------------------------------------------------------------- parameters
// 35 searched numbers. Left/right are exact negations on this robot
// (site/intent.mjs:31), so the symmetric frames use one number for the pair.
// hip_roll is bounded to its REAL travel: DuckModel.swift jointRanges hip_roll
// is +-0.3840 with HOME at -+0.0873, so absolute targets outside
// [-0.297, +0.471] are a clamped duplicate of the endpoint.
const B = {
  // F1 PRELOAD (head still home, trunk pitches nose-down on hips/ankles)
  t1: [0.25, 0.55], f1hip: [0.70, 1.40], f1knee: [-0.30, 0.60], f1ank: [0.10, 0.90],
  f1np: [0.00, 0.35], f1hp: [0.20, 0.60],
  // F2 HOOK
  t2d: [0.15, 0.40], f2np: [-1.5708, -1.15], f2hp: [0.30, 1.05],
  // F3 LOAD
  t3d: [0.10, 0.30], f3np: [-1.5708, -1.25], f3hp: [0.55, 1.25], f3knee: [0.20, 0.90],
  // F4 HAUL1 + RUNG L
  t4d: [0.15, 0.45], f4np: [-0.60, 0.75], f4hp: [-1.20, -0.20],
  f4lhp: [-1.5708, -1.10], f4lk: [0.55, 1.30], f4la: [0.85, 1.45], f4lhr: [-0.297, 0.384],
  // F5 RUNG R
  t5d: [0.15, 0.45], f5rhp: [1.10, 1.5708], f5rk: [-1.30, -0.55], f5ra: [-1.45, -0.85],
  f5rhr: [-0.384, 0.297], f5np: [-0.20, 0.90], f5hp: [-1.30, -0.30],
  // F6 HIP-OVER
  t6d: [0.25, 0.70], f6np: [0.40, 1.0472], f6hp: [-1.5708, -0.50],
  f6hip: [-0.20, 0.80], f6knee: [-0.60, 0.40], f6ank: [-0.60, 0.60],
  // opts
  blend: [0.8, 2.4], gap: [0.020, 0.045], side: [0.0, 0.085],
};
const KEYS = Object.keys(B);

function trackOf(p) {
  const f1 = HOME.slice();
  f1[J.lhp] = -p.f1hip; f1[J.rhp] = +p.f1hip;
  f1[J.lk] = +p.f1knee; f1[J.rk] = -p.f1knee;
  f1[J.la] = +p.f1ank; f1[J.ra] = -p.f1ank;
  f1[J.np] = p.f1np; f1[J.hp] = p.f1hp;
  const f2 = f1.slice(); f2[J.np] = p.f2np; f2[J.hp] = p.f2hp;
  const f3 = f2.slice(); f3[J.np] = p.f3np; f3[J.hp] = p.f3hp;
  f3[J.lk] = +p.f3knee; f3[J.rk] = -p.f3knee;
  const f4 = f3.slice(); f4[J.np] = p.f4np; f4[J.hp] = p.f4hp;
  f4[J.lhp] = p.f4lhp; f4[J.lk] = p.f4lk; f4[J.la] = p.f4la; f4[J.lhr] = p.f4lhr;
  const f5 = f4.slice(); f5[J.rhp] = p.f5rhp; f5[J.rk] = p.f5rk; f5[J.ra] = p.f5ra;
  f5[J.rhr] = p.f5rhr; f5[J.np] = p.f5np; f5[J.hp] = p.f5hp;
  const f6 = f5.slice(); f6[J.np] = p.f6np; f6[J.hp] = p.f6hp;
  f6[J.lhp] = +p.f6hip; f6[J.rhp] = -p.f6hip;
  f6[J.lk] = +p.f6knee; f6[J.rk] = -p.f6knee;
  f6[J.la] = +p.f6ank; f6[J.ra] = -p.f6ank;
  const t1 = p.t1, t2 = t1 + p.t2d, t3 = t2 + p.t3d, t4 = t3 + p.t4d, t5 = t4 + p.t5d, t6 = t5 + p.t6d;
  return [{ t: t1, pose: f1 }, { t: t2, pose: f2 }, { t: t3, pose: f3 },
          { t: t4, pose: f4 }, { t: t5, pose: f5 }, { t: t6, pose: f6 },
          { t: t6 + 0.7, pose: HOME.slice() }];
}

const quat = () => [data.qpos[D.freeQpos + 3], data.qpos[D.freeQpos + 4], data.qpos[D.freeQpos + 5], data.qpos[D.freeQpos + 6]];

/**
 * sim/climb_lib.mjs attempt(), copied, plus per-tick sampling.
 * `dx` shifts the start position (the +-10 mm offset check).
 */
async function run(p, h, track, dx = 0) {
  const cfg = { count: 4, rise: h, run: 0.28, start: 0.12 };
  mj.mj_resetData(model, data);
  layoutStairs(data, ADDR, cfg);
  data.qpos[D.freeQpos] = 0.12 - 0.07 - p.gap + dx;
  data.qpos[D.freeQpos + 1] = STAIR_Y + (p.side || 0);
  data.qpos[D.freeQpos + 2] = 0.12; data.qpos[D.freeQpos + 3] = 1;
  for (let i = 0; i < 14; i++) { data.qpos[D.qpos[i]] = HOME[i]; data.ctrl[i] = HOME[i]; }
  mj.mj_forward(model, data);
  const tr = track || trackOf(p);
  let la = new Array(14).fill(0);
  const cmd = command({ vx: 0 });
  let sat = 0, satN = 0;
  const step = async (off) => {
    layoutStairs(data, ADDR, cfg);
    const q = quat(); const jp = [], jv = [];
    for (let k = 0; k < 14; k++) { jp.push(data.qpos[D.qpos[k]]); jv.push(data.qvel[D.dof[k]]); }
    const obs = buildObs([data.sensordata[GYRO], data.sensordata[GYRO + 1], data.sensordata[GYRO + 2]], projectedGravity(q), jp, jv, la, cmd);
    const r = await stand.run({ obs: new ort.Tensor('float32', obs, [1, 61]) });
    la = Array.from(r.actions.data);
    for (let k = 0; k < 14; k++) {
      const v = HOME[k] + la[k] + (off ? (off[k] - HOME[k]) * p.blend : 0);
      const c = Math.min(Math.max(v, LO[k]), HI[k]);
      data.ctrl[k] = c;
      if (off) { satN++; if (c <= LO[k] + 1e-9 || c >= HI[k] - 1e-9) sat++; }
    }
    for (let s = 0; s < 4; s++) mj.mj_step(model, data);
  };
  for (let t = 0; t < 25; t++) await step(null);
  const x0 = data.qpos[D.freeQpos];

  const total = tr[tr.length - 1].t + 0.8;
  let maxX = -9, maxZ = -9, minX = 9, headTicks = 0, nTicks = 0, upTicks = 0;
  let lz = -9, rz = -9, headEver = false, footRiser = 0;
  const sample = () => {
    nTicks++;
    const x = data.qpos[D.freeQpos], z = data.qpos[D.freeQpos + 2];
    if (x > maxX) maxX = x; if (x < minX) minX = x; if (z > maxZ) maxZ = z;
    if (projectedGravity(quat())[2] < -0.90) upTicks++;
    let hit = false;
    // contact.get() hands back an embind object. NOT calling delete() on every
    // one of them grows the WASM heap past its 2 GB cap in about 20 s of
    // search: "Cannot enlarge memory, requested 2147487744 bytes". Measured.
    const cv = data.contact;
    for (let c = 0; c < data.ncon; c++) {
      const ct = cv.get(c);
      const a = ct.geom1, b = ct.geom2;
      if ((HEADG.has(a) && b === STEP0) || (HEADG.has(b) && a === STEP0)) hit = true;
      if (((a === LFOOT || a === RFOOT) && b === STEP0) || ((b === LFOOT || b === RFOOT) && a === STEP0)) footRiser++;
      ct.delete();
    }
    if (typeof cv.delete === 'function') cv.delete();
    if (hit) { headTicks++; headEver = true; }
    if (data.geom_xpos[LFOOT * 3 + 2] > lz) lz = data.geom_xpos[LFOOT * 3 + 2];
    if (data.geom_xpos[RFOOT * 3 + 2] > rz) rz = data.geom_xpos[RFOOT * 3 + 2];
  };
  for (let t = 0; t * DT < total; t++) { await step(poseAt(tr, t * DT)); sample(); }
  for (let t = 0; t < 50; t++) { await step(null); sample(); }

  const x = data.qpos[D.freeQpos], z = data.qpos[D.freeQpos + 2];
  const up = projectedGravity(quat())[2] < -0.90;
  let feetUp = 0;
  for (let g = 0; g < model.ngeom; g++) {
    const n = model.geom(g).name || '';
    if (!/foot_collision|sole/.test(n)) continue;
    if (data.geom_xpos[g * 3 + 2] > h - 0.005 && data.geom_xpos[g * 3] > 0.05) feetUp++;
  }
  const onTop = up && x > 0.12 && (z - h) > 0.095 && feetUp >= 2;
  return {
    onTop, x, z, above: z - h, feetUp, up,
    x0, maxX, maxZ, minX, headFrac: headTicks / Math.max(nTicks, 1), headEver,
    footRiserTicks: footRiser, lz, rz, upFrac: upTicks / Math.max(nTicks, 1),
    satFrac: sat / Math.max(satN, 1),
  };
}

/** The shaped objective from the strategy brief. */
function score(r, h) {
  const cl = (v, a, b) => Math.min(b, Math.max(a, v));
  return 3.0 * cl(r.maxX - r.x0, 0, 0.20) / 0.20
       + 3.0 * cl(r.maxZ - h, 0, 0.12) / 0.12
       + 2.0 * r.headFrac
       + 1.5 * ((cl(r.lz, 0, h) + cl(r.rz, 0, h)) / 2) / Math.max(h, 1e-6)
       + 1.0 * r.upFrac
       - 5.0 * Math.max(0, r.x0 - r.minX) / 0.10
       - 0.5 * r.satFrac
       + 100 * (r.onTop ? 1 : 0);
}

// ------------------------------------------------------------------- search
function mulberry32(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const args = Object.fromEntries(process.argv.slice(2).join(' ').split('--').filter(Boolean).map(s => s.trim().split(/\s+/)).map(([k, v]) => [k, v]));
const RISE = (+args.rise) / 1000;
const SECONDS = +(args.seconds || 240);
const SEED = +(args.seed || 1);
const OUT = args.out || `../climb/search_0-${args.rise}mm-seed${SEED}.json`;
const rnd = mulberry32(SEED * 7919 + 13);
const pick = ([a, b]) => a + rnd() * (b - a);
const randP = () => Object.fromEntries(KEYS.map(k => [k, pick(B[k])]));
const jitter = (p, s) => Object.fromEntries(KEYS.map(k => {
  const [a, b] = B[k];
  return [k, Math.min(b, Math.max(a, p[k] + (rnd() * 2 - 1) * (b - a) * s))];
}));

// A seed drawn from the brief's own measurements: hook deep, haul hard.
const SEED_P = {
  t1: 0.40, f1hip: 1.05, f1knee: 0.15, f1ank: 0.50, f1np: 0.20, f1hp: 0.40,
  t2d: 0.25, f2np: -1.45, f2hp: 0.70,
  t3d: 0.20, f3np: -1.50, f3hp: 0.95, f3knee: 0.55,
  t4d: 0.30, f4np: 0.10, f4hp: -0.80, f4lhp: -1.35, f4lk: 0.95, f4la: 1.15, f4lhr: 0.10,
  t5d: 0.30, f5rhp: 1.35, f5rk: -0.95, f5ra: -1.15, f5rhr: -0.10, f5np: 0.35, f5hp: -0.80,
  t6d: 0.45, f6np: 0.90, f6hp: -1.00, f6hip: 0.30, f6knee: -0.10, f6ank: 0.00,
  blend: 1.6, gap: 0.030, side: 0.040,
};

const t0 = Date.now();
const log = [];
let best = null, evals = 0;
const evalP = async (p) => { const r = await run(p, RISE); evals++; return { p, r, s: score(r, RISE) }; };

// random-restart hill climbing with a shrinking step
let cand = await evalP(SEED_P);
best = cand;
log.push({ eval: evals, s: +cand.s.toFixed(4), note: 'seed', ...brief(cand.r) });
let cur = cand, step = 0.28, stall = 0;
while ((Date.now() - t0) / 1000 < SECONDS) {
  let p;
  if (stall > 26) { p = randP(); stall = 0; step = 0.28; }
  else p = jitter(cur.p, step);
  const c = await evalP(p);
  if (c.s > cur.s + 1e-9) {
    cur = c; stall = 0; step = Math.max(0.05, step * 0.92);
    if (c.s > best.s) {
      best = c;
      log.push({ eval: evals, t: +((Date.now() - t0) / 1000).toFixed(1), s: +c.s.toFixed(4), ...brief(c.r), p: round(c.p) });
      console.log(`[${args.rise}mm] eval ${evals} t=${((Date.now() - t0) / 1000).toFixed(0)}s  S=${c.s.toFixed(3)}  ${JSON.stringify(brief(c.r))}`);
    }
  } else { stall++; if (stall % 9 === 0) step = Math.max(0.05, step * 0.85); }
  if (stall > 26 && cur.s < best.s) cur = best;
}

function brief(r) {
  return {
    onTop: r.onTop, x: mm(r.x), z: mm(r.z), above: mm(r.above), feetUp: r.feetUp, up: r.up,
    peakZ: mm(r.maxZ), peakX: mm(r.maxX), backMM: mm(r.x0 - r.minX),
    headFrac: +r.headFrac.toFixed(3), footRiserTicks: r.footRiserTicks,
    lzMM: mm(r.lz), rzMM: mm(r.rz), satFrac: +r.satFrac.toFixed(3),
  };
}
function mm(v) { return +(v * 1000).toFixed(1); }
function round(p) { return Object.fromEntries(Object.entries(p).map(([k, v]) => [k, +v.toFixed(4)])); }

// offset robustness on the best
const offsets = [-0.010, 0, 0.010];
const offRuns = [];
for (const dx of offsets) {
  const r = await run(best.p, RISE, trackOf(best.p), dx);
  offRuns.push({ dxMM: dx * 1000, ...brief(r) });
}
const cleared = offRuns.filter(o => o.onTop).length;

const failure = (() => {
  const r = best.r;
  if (!r.up) return 'toppled';
  if (r.x <= 0.12) return r.headEver ? 'head reached the tread, trunk never got past the riser face' : 'never reached: upright, short of the riser, head never touched';
  if (r.feetUp < 2) return `past the face but ${r.feetUp}/2 feet at tread height`;
  if (r.above <= 0.095) return 'feet up but crouched below standing height';
  return 'on the step';
})();

const out = {
  strategy: 'A - beak hook + wall walk', riseMM: +args.rise, seed: SEED, seconds: SECONDS,
  evals, best: { score: +best.s.toFixed(4), params: round(best.p), ...brief(best.r) },
  offsets: offRuns, cleared, of: offsets.length, failure, improvements: log,
  command: `cd ~/projects/duck-sounds/sim && node ../climb/search_0.mjs --rise ${args.rise} --seconds ${SECONDS} --seed ${SEED}`,
};
fs.writeFileSync(OUT, JSON.stringify(out, null, 2));

// intent JSON shape the site loads: keyframes + opts + note
const kf = trackOf(best.p).map(f => ({ t: +f.t.toFixed(4), pose: f.pose.map(v => +v.toFixed(4)) }));
fs.writeFileSync(`../climb/best_0_${args.rise}mm.json`, JSON.stringify({
  name: `beak_hook_wall_walk_${args.rise}mm`,
  keyframes: kf, blend: +best.p.blend.toFixed(4), gap: +best.p.gap.toFixed(4),
  side: +best.p.side.toFixed(4), approach: 0,
  note: `strategy A. objective ${best.s.toFixed(3)}, onTop=${best.r.onTop}, cleared ${cleared}/3 start offsets. `
      + `reproduce: ${out.command}`,
}, null, 2));
console.log(`[${args.rise}mm] DONE evals=${evals} bestS=${best.s.toFixed(3)} onTop=${best.r.onTop} cleared=${cleared}/3 -> ${OUT}`);
