// vault.mjs — FAMILY C, the beak-strut vault, on a flight that is not fighting itself.
//
// Round 1 never ran this family (a picker bug). Round 2's instrument work
// (climb/rig3.mjs) then found that the 4-step flight self-collides below a
// ~145 mm rise and throws any duck off the first tread within 0.2 s, so the
// 40/60/90 mm band this family is aimed at was untestable.
//
// THE EPISODE BELOW IS A VERBATIM COPY of climb/rig3.mjs's runEpisodeRaw
// (which is itself parity-proved bit-identical to sim/climb_lib.mjs attempt()).
// sim/ is off-limits and importing it runs its top level, so this is a copy.
// Phase P re-proves the copy against rig3.scoreSaved on the same saved file.
//
// TWO THINGS ARE ADDED, neither a physics constant:
//
//  (1) STEP COLLISION ISOLATION (opts.isolate). sim/scene_physics.xml:89-96
//      gives every step geom contype=4 conaffinity=4, and site/stairs.js:38
//      sets STEP_HALF_DEPTH = 0.17 against a 0.28 run, so consecutive 200 kg
//      blocks interpenetrate by 60 mm in x and (200 - rise) mm in z and shove
//      each other apart horizontally. Setting model.geom_conaffinity = 0 on the
//      step geoms only makes steps invisible TO EACH OTHER:
//        step-step : (4 & 0) | (4 & 0) = 0   -> gone
//        step-duck : (4 & 5) = 4             -> unchanged (duck is contype/conaff 5)
//        step-floor: (4 & 1) | (1 & 0) = 0   -> unchanged (was already 0)
//      The half-depth cannot be changed instead: 0.17 is compiled into the geom
//      (scene_physics.xml:89 size="0.17 0.17 0.10"); shrinking the JS constant
//      would move the riser face off x = 0.12 and leave a 60 mm hole between
//      blocks. Friction, gravity, timestep, gains and mass are untouched.
//
//  (2) VAULT INSTRUMENTS. headOnly ticks (jaw on the tread and NEITHER foot
//      touching anything), the trunk-z gain across those ticks, and how high
//      the feet ever got. That is the thing this family is supposed to do.
//
// Criterion is rig3's `honest`, tail is 'policy'. Nothing else is scored.
// Every candidate is written to JSON and re-read before it is scored, so the
// number reported is the number the saved file produces.
//
// Contacts via mj_geomDistance only — data.contact.get(i) leaks the WASM heap.
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/vault.mjs <phase>
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
import { findStairJoints, layoutStairs, STAIR_Y, STAIR_HALF_WIDTH } from '../site/stairs.js';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const model = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model), ADDR = findStairJoints(model, { isolate: false }); // this script toggles the repair itself; capture the SHIPPED affinity
let GYRO = 0;
for (let i = 0; i < model.nsensor; i++) if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;
const stand = await ort.InferenceSession.create('./BEST_alpha_stand.onnx');
const DT = 1 / C.tickHz;

const LATERAL = STAIR_HALF_WIDTH;          // 0.17 m
const RISER_X = 0.12;

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

/** (1) above: make step blocks invisible to each other, and to nothing else. */
function setIsolate(on) {
  STEPG.forEach((g, i) => { model.geom_conaffinity[g] = on ? 0 : STEP_CONAFF0[i]; });
}

const quat = () => [data.qpos[D.freeQpos + 3], data.qpos[D.freeQpos + 4], data.qpos[D.freeQpos + 5], data.qpos[D.freeQpos + 6]];
const dist = (a, b) => mj.mj_geomDistance(model, data, a, b, 0.05, null);

/** climb_lib.mjs:80-86 / rig3.mjs, verbatim. */
function poseAt(tr, time) {
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

function snapshot(h) {
  const x = data.qpos[D.freeQpos], y = data.qpos[D.freeQpos + 1], z = data.qpos[D.freeQpos + 2];
  const up = projectedGravity(quat())[2] < -0.90;
  let feetUpRaw = 0, feetOnTread = 0;
  for (const g of FEET) {
    if (data.geom_xpos[g * 3 + 2] > h - 0.005 && data.geom_xpos[g * 3] > 0.05) feetUpRaw++;
    if (data.geom_xpos[g * 3 + 2] > h - 0.005 && data.geom_xpos[g * 3] > RISER_X
      && Math.abs(data.geom_xpos[g * 3 + 1] - STAIR_Y) <= LATERAL) feetOnTread++;
  }
  return { x, y, z, dy: y - STAIR_Y, above: z - h, up, feetUpRaw, feetOnTread };
}

/** rig3.mjs criteria(). `honest` is the one and only criterion used here. */
function criteria(h, s) {
  const lateral = Math.abs(s.dy) <= LATERAL;
  return {
    orig: s.up && s.x > RISER_X && s.above > 0.095 && s.feetUpRaw >= 2,
    honest: s.up && lateral && s.x > RISER_X && s.above > 0.095 && s.feetOnTread >= 2,
    honest60: s.up && lateral && s.x > RISER_X && s.above > 0.060 && s.feetOnTread >= 2,
    lateral,
  };
}

const c01 = v => Math.max(0, Math.min(1, v));

/** rig3.mjs reward() — the lateral gate is hard and comes first. */
function baseReward(rec) {
  const s = rec.scored;
  if (rec.maxAbsDY > LATERAL || Math.abs(s.dy) > LATERAL) return 0;
  return 3 * c01((s.x - (RISER_X - 0.20)) / 0.20) + 2 * s.feetOnTread
       + 4 * c01(s.above / 0.095) + (s.up ? 1 : 0);
}

/** The vault objective: base, plus the pivot, plus feet-above-tread. */
function objective(rec) {
  const s = rec.scored;
  if (rec.maxAbsDY > LATERAL || Math.abs(s.dy) > LATERAL) return 0;
  return baseReward(rec)
       + 3 * c01(rec.pivotGain / 0.06)            // trunk z gained while the head is the ONLY contact
       + 2 * c01(rec.headOnlyTicks / 15)          // there was a strut phase at all
       + 2 * c01(rec.feetOnTreadMax / 2)          // feet ever past the riser at tread height
       + 1 * c01(rec.feetHighMax / 2);            // feet ever left the floor above the tread
}

// ---------------------------------------------------------------- the episode
async function runEpisodeRaw(track, opts, h, tail = 'policy') {
  setIsolate(!!opts.isolate);
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
  mj.mj_forward(model, data);
  const tr = track.map(f => ({ t: f.t, pose: f.pose.slice() }));
  let la = new Array(14).fill(0);
  const cmd = command({ vx: opts.approach });

  const R = { ticks: 0, headTicks: 0, riserTicks: 0, upTicks: 0, sat: 0, ctrls: 0,
              maxX: -1e9, maxZ: -1e9, maxAbsDY: 0, feetOnTreadMax: 0, feetHighMax: 0,
              headOnlyTicks: 0, pivotZ0: null, pivotZmax: -1e9,
              minStepGap_mm: 1e9, maxTreadDriftX_mm: 0, maxTreadSag_mm: 0 };
  const treadDrift = () => {
    const sag = (h - (data.geom_xpos[STEP0 * 3 + 2] + 0.10)) * 1000;
    if (sag > R.maxTreadSag_mm) R.maxTreadSag_mm = sag;
    const dx = Math.abs(data.geom_xpos[STEP0 * 3] - (0.12 + 0.17)) * 1000;
    if (dx > R.maxTreadDriftX_mm) R.maxTreadDriftX_mm = dx;
    if (STEP1 >= 0 && cfg.count > 1) {
      const g = mj.mj_geomDistance(model, data, STEP0, STEP1, 0.4, null) * 1000;
      if (g < R.minStepGap_mm) R.minStepGap_mm = g;
    }
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
    // a foot pressing the RISER face: below the tread, touching the block
    let footRiser = false, footAny = false;
    for (const g of [LFOOT, RFOOT]) {
      const onBlock = dist(g, STEP0) < 0.003, onFloor = dist(g, FLOOR) < 0.003;
      if (onBlock || onFloor) footAny = true;
      if (onBlock && data.geom_xpos[g * 3 + 2] < h - 0.005) footRiser = true;
    }
    if (footRiser) R.riserTicks++;
    // THE PIVOT: the head is the only thing touching the world.
    if (head && !footAny) {
      R.headOnlyTicks++;
      if (R.pivotZ0 === null) R.pivotZ0 = z;
      if (z > R.pivotZmax) R.pivotZmax = z;
    }
    let fot = 0, fhi = 0;
    for (const g of FEET) {
      const lat = Math.abs(data.geom_xpos[g * 3 + 1] - STAIR_Y) <= LATERAL;
      if (data.geom_xpos[g * 3 + 2] > h - 0.005 && data.geom_xpos[g * 3] > RISER_X && lat) fot++;
      if (data.geom_xpos[g * 3 + 2] > h + 0.005 && lat) fhi++;
    }
    if (fot > R.feetOnTreadMax) R.feetOnTreadMax = fot;
    if (fhi > R.feetHighMax) R.feetHighMax = fhi;
  };

  const step = async (off, rec) => {
    layoutStairs(data, ADDR, cfg);
    const q = quat(); const jp = [], jv = [];
    for (let k = 0; k < 14; k++) { jp.push(data.qpos[D.qpos[k]]); jv.push(data.qvel[D.dof[k]]); }
    const obs = buildObs([data.sensordata[GYRO], data.sensordata[GYRO + 1], data.sensordata[GYRO + 2]], projectedGravity(q), jp, jv, la, cmd);
    const r = await stand.run({ obs: new ort.Tensor('float32', obs, [1, 61]) });
    la = Array.from(r.actions.data);
    for (let k = 0; k < 14; k++) {
      const v = HOME[k] + la[k] + (off ? (off[k] - HOME[k]) * opts.blend : 0);
      const c = Math.min(Math.max(v, LO[k]), HI[k]);
      data.ctrl[k] = c;
      if (rec) { R.ctrls++; if (c <= LO[k] + 1e-9 || c >= HI[k] - 1e-9) R.sat++; }
    }
    for (let s = 0; s < 4; s++) mj.mj_step(model, data);
    treadDrift();
    if (rec) record();
  };

  for (let t = 0; t < 25; t++) await step(null, false);
  const x0 = data.qpos[D.freeQpos];
  const total = tr[tr.length - 1].t + 0.8;
  for (let t = 0; t * DT < total; t++) await step(poseAt(tr, t * DT), true);
  const atTrackEnd = snapshot(h);
  for (let t = 0; t < 50; t++) await step(null, true);   // tail = 'policy', climb_lib's own
  const afterTail = snapshot(h);

  const scored = (tail === 'none') ? atTrackEnd : afterTail;
  const rec = {
    tail, rise: h, x0, scored, atTrackEnd, afterTail,
    crit: criteria(h, scored), critAtTrackEnd: criteria(h, atTrackEnd),
    maxX: R.maxX, maxZ: R.maxZ, maxAbsDY: R.maxAbsDY,
    feetOnTreadMax: R.feetOnTreadMax, feetHighMax: R.feetHighMax,
    headOnlyTicks: R.headOnlyTicks,
    pivotGain: R.pivotZ0 === null ? 0 : Math.max(0, R.pivotZmax - R.pivotZ0),
    minStepGap_mm: R.minStepGap_mm === 1e9 ? null : R.minStepGap_mm,
    maxTreadDriftX_mm: R.maxTreadDriftX_mm, maxTreadSag_mm: R.maxTreadSag_mm,
    headFrac: R.headTicks / Math.max(R.ticks, 1),
    riserFrac: R.riserTicks / Math.max(R.ticks, 1),
    upFrac: R.upTicks / Math.max(R.ticks, 1),
    satFrac: R.sat / Math.max(R.ctrls, 1),
  };
  rec.legacy = { onTop: rec.crit.orig, x: scored.x, z: scored.z, above: scored.above, feetUp: scored.feetUpRaw, up: scored.up };
  rec.base = baseReward(rec);
  rec.objective = objective(rec);
  return rec;
}

/** THE ONLY SCORER. Takes a PATH. */
async function scoreSaved(path, { rise, tail = 'policy', overrides = {} } = {}) {
  const j = JSON.parse(fs.readFileSync(path, 'utf8'));
  if (!Array.isArray(j.keyframes) || !j.keyframes.length) throw new Error('no keyframes in ' + path);
  for (const f of j.keyframes) if (!Array.isArray(f.pose) || f.pose.length !== 14) throw new Error('bad pose in ' + path);
  const opts = { blend: j.blend, approach: j.approach || 0, gap: j.gap || 0, side: j.side || 0,
                 spawn: j.spawn || null, isolate: !!j.isolate, stepCount: j.stepCount || 4, ...overrides };
  const rec = await runEpisodeRaw(j.keyframes, opts, rise, tail);
  rec.source = path; rec.opts = opts;
  return rec;
}

// ================================================================== TRACK
const J = { lhy: 0, lhr: 1, lhp: 2, lk: 3, la: 4, np: 5, hp: 6, hy: 7, hr: 8,
            rhy: 9, rhr: 10, rhp: 11, rk: 12, ra: 13 };

/**
 * FAMILY C — the beak-strut vault.
 *
 * A. reach   : crouch and drive the neck down and forward so the beak lands on
 *              the tread. The neck angles set here are the STRUT and they do
 *              not move again until the feet are down.
 * B. preload : legs coil under the body, strut held.
 * C. vault   : hips and knees EXTEND against the planted head — the trunk
 *              rotates up and over the beak contact. Strut held.
 * D. tuck    : legs fold up and forward so the feet clear the tread edge.
 *              Strut held. This is the top of the vault.
 * E. land    : legs reach down onto the tread and the neck is released, so the
 *              trunk comes upright over the feet instead of over the beak.
 * then HOME.
 */
function trackOf(p) {
  const put = (q, hip, knee, ank, roll) => {
    q[J.lhp] = HOME[J.lhp] + hip;  q[J.rhp] = HOME[J.rhp] - hip;
    q[J.lk] = HOME[J.lk] + knee;   q[J.rk] = HOME[J.rk] - knee;
    q[J.la] = HOME[J.la] + ank;    q[J.ra] = HOME[J.ra] - ank;
    q[J.lhr] = HOME[J.lhr] + roll; q[J.rhr] = HOME[J.rhr] + roll;
    return q;
  };
  const strut = q => { q[J.np] = p.strutNeck; q[J.hp] = p.strutHead; return q; };

  const A = strut(put(HOME.slice(), p.crouchHip, p.crouchKnee, p.crouchAnk, p.roll));
  const B = strut(put(HOME.slice(), p.preHip, p.preKnee, p.preAnk, p.roll));
  const Cc = strut(put(HOME.slice(), p.vaultHip, p.vaultKnee, p.vaultAnk, p.roll));
  const Dd = strut(put(HOME.slice(), p.tuckHip, p.tuckKnee, p.tuckAnk, p.roll));
  const E = put(HOME.slice(), p.landHip, p.landKnee, p.landAnk, 0);
  E[J.np] = p.landNeck; E[J.hp] = p.landHead;
  const t1 = p.tReach, t2 = t1 + p.tPre, t3 = t2 + p.tVault, t4 = t3 + p.tTuck, t5 = t4 + p.tLand;
  return [{ t: t1, pose: A }, { t: t2, pose: B }, { t: t3, pose: Cc },
          { t: t4, pose: Dd }, { t: t5, pose: E }, { t: t5 + 0.7, pose: HOME.slice() }];
}

const BOUNDS = {
  gap: [0.01, 0.10], side: [-0.02, 0.09], approach: [0.0, 0.45], blend: [0.8, 2.4],
  tReach: [0.30, 0.90], tPre: [0.10, 0.40], tVault: [0.10, 0.45], tTuck: [0.10, 0.45], tLand: [0.15, 0.60],
  // the strut. neck_pitch home 0.349, range [-1.571, 1.047]; head_pitch home 0.349, range [-1.571, 1.571]
  strutNeck: [-1.55, 1.04], strutHead: [-1.55, 1.55],
  crouchHip: [-1.2, 1.2], crouchKnee: [-1.2, 1.2], crouchAnk: [-1.2, 1.2],
  preHip: [-1.2, 1.2], preKnee: [-1.2, 1.2], preAnk: [-1.2, 1.2],
  vaultHip: [-1.4, 1.4], vaultKnee: [-1.4, 1.4], vaultAnk: [-1.4, 1.4],
  tuckHip: [-1.4, 1.4], tuckKnee: [-1.4, 1.4], tuckAnk: [-1.4, 1.4],
  landHip: [-1.4, 1.4], landKnee: [-1.4, 1.4], landAnk: [-1.4, 1.4],
  landNeck: [-1.0, 1.04], landHead: [-1.0, 1.55],
  roll: [-0.30, 0.30],
};
const KEYS = Object.keys(BOUNDS);

/**
 * A hand-designed vault, from the static kinematics measured in phase `probe`
 * (climb/vault-probe.json), used as the CEM's initial mean. Every sign here is
 * a measured one, not a guess:
 *   neck_pitch NEGATIVE drives the beak forward and down
 *     (np 0.349 -> jaw at x 36.4 / z 223.5 mm;  np -1.5 -> x 124.7 / z 138.3)
 *   head_pitch POSITIVE adds a little more forward reach (hp 1.5 -> x 57.7)
 *   hip delta POSITIVE swings the foot BACKWARD  (d +1.2 -> foot x -80.3 mm)
 *     — that is the stroke that rotates the trunk FORWARD over a planted head
 *   hip delta NEGATIVE swings the foot FORWARD and UP (d -1.2 -> x 90.3 / z 69.4)
 *     — that is the tuck that puts the feet over the tread
 *   knee delta POSITIVE lifts the foot forward and up (d +1.2 -> x 42.9 / z 67.6)
 * The search is not confined to this: a quarter of every population is drawn
 * uniformly from the full bounds, so a vault with the opposite stroke can win.
 */
const PRIOR = {
  gap: 0.05, side: 0.03, approach: 0.15, blend: 1.6,
  tReach: 0.55, tPre: 0.20, tVault: 0.25, tTuck: 0.25, tLand: 0.35,
  strutNeck: -1.20, strutHead: 0.90,
  crouchHip: -0.60, crouchKnee: 0.50, crouchAnk: 0.30,
  preHip: -0.30, preKnee: 0.60, preAnk: 0.20,
  vaultHip: 0.80, vaultKnee: -0.60, vaultAnk: -0.30,
  tuckHip: -1.00, tuckKnee: 0.90, tuckAnk: 0.20,
  landHip: -0.50, landKnee: 0.10, landAnk: -0.20,
  landNeck: 0.35, landHead: 0.35, roll: 0.05,
};

// mulberry32
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
let RND = mulberry32(20260901);
const clampB = (k, v) => Math.min(BOUNDS[k][1], Math.max(BOUNDS[k][0], v));
const randP = () => Object.fromEntries(KEYS.map(k => [k, BOUNDS[k][0] + RND() * (BOUNDS[k][1] - BOUNDS[k][0])]));
function gauss() { let u = 0, v = 0; while (!u) u = RND(); while (!v) v = RND(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
const sampleN = (mu, sg) => Object.fromEntries(KEYS.map(k => [k, clampB(k, mu[k] + sg[k] * gauss())]));

const r5 = v => +v.toFixed(5);
function intentOf(p, note, rise) {
  return { name: `beak_strut_vault_${Math.round(rise * 1000)}mm`, family: 'C beak-strut vault',
           keyframes: trackOf(p).map(f => ({ t: +f.t.toFixed(4), pose: f.pose.map(r5) })),
           blend: +p.blend.toFixed(4), gap: +p.gap.toFixed(4), side: +p.side.toFixed(4),
           approach: +p.approach.toFixed(4), isolate: true, stepCount: 4, params: p, note };
}

// ================================================================== MAIN
const PHASE = process.argv[2] || 'probe';
const DEADLINE_S = +(process.argv[3] || 0);
const SCRATCH = '/tmp/claude-1000/-home-craigm26/43f07dce-e62f-464d-ae1f-2fe020620950/scratchpad';
fs.mkdirSync(SCRATCH, { recursive: true });
const mm = v => +(v * 1000).toFixed(1);
const t0 = Date.now();
const el = () => ((Date.now() - t0) / 1000).toFixed(1);

if (PHASE === 'probe') {
  const out = { generated: new Date().toISOString() };
  // ---- P: parity of THIS copy against climb/rig3.mjs on the same saved file
  const rig3 = await import('./rig3.mjs');
  out.parity = [];
  for (const f of ['best_0_40mm.json', 'best_1_90mm.json', 'best_2_180mm.json']) {
    const rise = parseInt(f.match(/_(\d+)mm/)[1], 10) / 1000;
    const A = await rig3.scoreSaved('../climb/' + f, { rise, tail: 'policy' });
    const B = await scoreSaved('../climb/' + f, { rise, tail: 'policy' });   // isolate absent -> false
    const ok = A.scored.x === B.scored.x && A.scored.z === B.scored.z
      && A.scored.feetOnTread === B.scored.feetOnTread && A.crit.honest === B.crit.honest;
    out.parity.push({ file: f, rig3_x: A.scored.x, vault_x: B.scored.x, rig3_z: A.scored.z, vault_z: B.scored.z, match: ok });
    console.log(`parity ${f}: rig3 x=${A.scored.x} z=${A.scored.z} | vault x=${B.scored.x} z=${B.scored.z} | ${ok ? 'MATCH' : 'DIFF'}`);
  }
  out.parityAll = out.parity.every(r => r.match);

  // ---- G: does the collision fix stop the flight fighting itself, and does an
  //         on-tread duck then stay on the tread?
  out.gate = [];
  for (const rise of [0.040, 0.060, 0.090]) {
    for (const iso of [false, true]) {
      const path = SCRATCH + '/ctrl_ontread.json';
      fs.writeFileSync(path, JSON.stringify({
        keyframes: [{ t: 1.0, pose: HOME.map(r5) }, { t: 2.9, pose: HOME.map(r5) }],
        blend: 1, gap: 0.05, side: 0, approach: 0, isolate: iso, stepCount: 4,
        spawn: { x: 0.26, y: STAIR_Y, z: rise + 0.12 },
      }, null, 2));
      const r = await scoreSaved(path, { rise });
      out.gate.push({ rise_mm: rise * 1000, isolate: iso, minStepGap_mm: r.minStepGap_mm,
        treadDriftX_mm: r.maxTreadDriftX_mm, x_mm: mm(r.scored.x), z_mm: mm(r.scored.z),
        above_mm: mm(r.scored.above), feetOnTread: r.scored.feetOnTread, honest: r.crit.honest });
      console.log(`gate rise=${rise * 1000}mm isolate=${iso} stepGap=${r.minStepGap_mm} driftX=${r.maxTreadDriftX_mm.toFixed(2)}mm -> x=${mm(r.scored.x)} z=${mm(r.scored.z)} feetOnTread=${r.scored.feetOnTread} HONEST=${r.crit.honest}`);
    }
  }
  // ---- W: walk-only control on the FIXED flight (must not pass)
  for (const rise of [0.040, 0.060, 0.090]) {
    const path = SCRATCH + '/ctrl_walk.json';
    fs.writeFileSync(path, JSON.stringify({ keyframes: [{ t: 1.0, pose: HOME.map(r5) }, { t: 2.6, pose: HOME.map(r5) }],
      blend: 1, gap: 0.05, side: 0, approach: 0.35, isolate: true, stepCount: 4 }, null, 2));
    const r = await scoreSaved(path, { rise });
    out.gate.push({ rise_mm: rise * 1000, isolate: true, control: 'walk-only', x_mm: mm(r.scored.x),
      z_mm: mm(r.scored.z), feetOnTread: r.scored.feetOnTread, honest: r.crit.honest, objective: +r.objective.toFixed(3) });
    console.log(`walk-only rise=${rise * 1000}mm -> x=${mm(r.scored.x)} feetOnTread=${r.scored.feetOnTread} HONEST=${r.crit.honest} obj=${r.objective.toFixed(2)}`);
  }

  // ---- K: kinematics. Which way does the beak go, which way does a hip go?
  const jawTip = () => { let best = null; for (const g of JAW) { const x = data.geom_xpos[g * 3]; if (!best || x > best.x) best = { x, y: data.geom_xpos[g * 3 + 1], z: data.geom_xpos[g * 3 + 2] }; } return best; };
  const staticPose = (mut) => {
    mj.mj_resetData(model, data);
    layoutStairs(data, ADDR, { count: 4, rise: 0.06, run: 0.28, start: 0.12 });
    data.qpos[D.freeQpos] = 0; data.qpos[D.freeQpos + 1] = STAIR_Y; data.qpos[D.freeQpos + 2] = 0.12; data.qpos[D.freeQpos + 3] = 1;
    const q = HOME.slice(); mut(q);
    for (let i = 0; i < 14; i++) data.qpos[D.qpos[i]] = q[i];
    mj.mj_forward(model, data);
    const t = jawTip();
    return { jawX: mm(t.x), jawZ: mm(t.z), lfX: mm(data.geom_xpos[LFOOT * 3]), lfZ: mm(data.geom_xpos[LFOOT * 3 + 2]) };
  };
  out.kin = { neck: [], head: [], hip: [], knee: [], ank: [] };
  for (const v of [-1.5, -1.0, -0.5, 0, 0.349, 0.7, 1.04]) out.kin.neck.push({ np: v, ...staticPose(q => { q[J.np] = v; }) });
  for (const v of [-1.5, -1.0, -0.5, 0, 0.349, 0.7, 1.5]) out.kin.head.push({ hp: v, ...staticPose(q => { q[J.hp] = v; }) });
  for (const d of [-1.2, -0.6, 0, 0.6, 1.2]) out.kin.hip.push({ dhip: d, ...staticPose(q => { q[J.lhp] = HOME[J.lhp] + d; q[J.rhp] = HOME[J.rhp] - d; }) });
  for (const d of [-1.2, -0.6, 0, 0.6, 1.2]) out.kin.knee.push({ dknee: d, ...staticPose(q => { q[J.lk] = HOME[J.lk] + d; q[J.rk] = HOME[J.rk] - d; }) });
  for (const d of [-1.2, -0.6, 0, 0.6, 1.2]) out.kin.ank.push({ dank: d, ...staticPose(q => { q[J.la] = HOME[J.la] + d; q[J.ra] = HOME[J.ra] - d; }) });
  console.log('neck sweep (np -> jawX_mm, jawZ_mm):', out.kin.neck.map(r => `${r.np}:${r.jawX}/${r.jawZ}`).join('  '));
  console.log('head sweep (hp -> jawX_mm, jawZ_mm):', out.kin.head.map(r => `${r.hp}:${r.jawX}/${r.jawZ}`).join('  '));
  console.log('hip  sweep (d  -> lfX_mm, lfZ_mm):  ', out.kin.hip.map(r => `${r.dhip}:${r.lfX}/${r.lfZ}`).join('  '));
  console.log('knee sweep (d  -> lfX_mm, lfZ_mm):  ', out.kin.knee.map(r => `${r.dknee}:${r.lfX}/${r.lfZ}`).join('  '));
  console.log('ank  sweep (d  -> lfX_mm, lfZ_mm):  ', out.kin.ank.map(r => `${r.dank}:${r.lfX}/${r.lfZ}`).join('  '));

  // ---- C: episode cost
  const tc = Date.now();
  const p = randP(); const path = SCRATCH + '/cost.json';
  fs.writeFileSync(path, JSON.stringify(intentOf(p, 'cost probe', 0.06), null, 2));
  for (let i = 0; i < 3; i++) await scoreSaved(path, { rise: 0.06 });
  out.episode_s = (Date.now() - tc) / 3000;
  console.log('episode cost:', out.episode_s.toFixed(3), 's   total probe', el(), 's');
  fs.writeFileSync('../climb/vault-probe.json', JSON.stringify(out, null, 2));
}

/**
 * PHASE refine90 — one targeted re-run at 90 mm.
 *
 * The 90 mm winner from PHASE search has vaultHip +1.08 AND tuckHip +1.13:
 * both strokes drive the feet BACKWARD (hip delta positive = foot back, from
 * the probe sweep), so the tuck never swings the feet forward over the tread
 * and feetOnTread is 0 in all 1401 episodes. The 40 mm winner has the correct
 * shape (vaultHip +1.23 to rotate the trunk over the beak, then tuckHip ~0 with
 * tuckKnee +0.55 to bring the feet forward and up). This restarts the 90 mm CEM
 * from that shape instead of from the generic prior.
 */
if (PHASE === 'refine90') {
  const rise = 0.090, tag = 90, end = t0 + DEADLINE_S * 1000;
  RND = mulberry32(777090);
  const seedP = JSON.parse(fs.readFileSync('../climb/best_r2_vault_40mm.json', 'utf8')).params;
  let mu = { ...seedP }, sg = Object.fromEntries(KEYS.map(k => [k, (BOUNDS[k][1] - BOUNDS[k][0]) / 6]));
  const tmp = SCRATCH + '/cand_r90.json';
  const POP = 24, ELITE = 6;
  let best = null, n = 0, clears = 0, gen = 0; const log = [];
  const prev = JSON.parse(fs.readFileSync('../climb/vault-results.json', 'utf8'));
  const prevObj = prev.rises['90'].verified.objective;
  while (Date.now() < end) {
    const pop = [];
    for (let i = 0; i < POP && Date.now() < end; i++) {
      const p = (i % 6 === 5) ? randP() : sampleN(mu, sg);
      fs.writeFileSync(tmp, JSON.stringify(intentOf(p, `refine90 ${n}`, rise), null, 2));
      const r = await scoreSaved(tmp, { rise }); n++;
      if (r.crit.honest) clears++;
      pop.push({ p, r });
      if (!best || r.objective > best.r.objective) {
        best = { p, r };
        log.push({ n, t: +el(), objective: +r.objective.toFixed(4), honest: r.crit.honest,
          x_mm: mm(r.scored.x), above_mm: mm(r.scored.above), trunkPeakZ_mm: mm(r.maxZ),
          feetOnTread: r.scored.feetOnTread, feetOnTreadMax: r.feetOnTreadMax,
          headOnlyTicks: r.headOnlyTicks, pivotGain_mm: mm(r.pivotGain),
          headFrac: +r.headFrac.toFixed(3), riserFrac: +r.riserFrac.toFixed(3) });
        console.log(`[refine90 n=${n} ${el()}s] obj=${r.objective.toFixed(3)} honest=${r.crit.honest} x=${mm(r.scored.x)} above=${mm(r.scored.above)} maxZ=${mm(r.maxZ)} fot=${r.scored.feetOnTread}/${r.feetOnTreadMax} headOnly=${r.headOnlyTicks} pivot=${mm(r.pivotGain)}mm head=${r.headFrac.toFixed(3)} riser=${r.riserFrac.toFixed(3)}`);
        if (r.objective > prevObj) fs.writeFileSync('../climb/best_r2_vault_90mm.json',
          JSON.stringify(intentOf(p, `best beak-strut vault at 90 mm (refine90, warm-started from the 40 mm winner); objective ${r.objective.toFixed(3)}; honest=${r.crit.honest}`, rise), null, 2));
      }
    }
    if (pop.length < ELITE) break;
    pop.sort((a, b) => b.r.objective - a.r.objective);
    const e = pop.slice(0, ELITE);
    for (const k of KEYS) {
      const v = e.map(x => x.p[k]), m = v.reduce((a, b) => a + b, 0) / v.length;
      mu[k] = m; sg[k] = Math.max(Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length), (BOUNDS[k][1] - BOUNDS[k][0]) * 0.04);
    }
    gen++;
  }
  const verify = await scoreSaved('../climb/best_r2_vault_90mm.json', { rise });
  prev.refine90 = { seededFrom: 'best_r2_vault_40mm.json params', seed: 777090, episodes: n,
    generations: gen, clears, priorObjective: prevObj, improvements: log,
    verified: { objective: +verify.objective.toFixed(4), honest: verify.crit.honest,
      honest60: verify.crit.honest60, x_mm: mm(verify.scored.x), dy_mm: mm(verify.scored.dy),
      above_mm: mm(verify.scored.above), trunkPeakZ_mm: mm(verify.maxZ),
      feetOnTread: verify.scored.feetOnTread, feetOnTreadMax: verify.feetOnTreadMax,
      headOnlyTicks: verify.headOnlyTicks, pivotGain_mm: mm(verify.pivotGain),
      headFrac: +verify.headFrac.toFixed(3), riserFrac: +verify.riserFrac.toFixed(3),
      upFrac: +verify.upFrac.toFixed(3), satFrac: +verify.satFrac.toFixed(3) } };
  prev.refine90.neighbourhood = [];
  for (const d of [-10, 0, 10]) {
    const r = await scoreSaved('../climb/best_r2_vault_90mm.json', { rise: (90 + d) / 1000 });
    prev.refine90.neighbourhood.push({ testedAt_mm: 90 + d, objective: +r.objective.toFixed(4),
      honest: r.crit.honest, x_mm: mm(r.scored.x), above_mm: mm(r.scored.above),
      feetOnTread: r.scored.feetOnTread, trunkPeakZ_mm: mm(r.maxZ) });
    console.log(`neigh refine90 @ ${90 + d}mm: obj=${r.objective.toFixed(3)} honest=${r.crit.honest} x=${mm(r.scored.x)} above=${mm(r.scored.above)} fot=${r.scored.feetOnTread}`);
  }
  fs.writeFileSync('../climb/vault-results.json', JSON.stringify(prev, null, 2));
  console.log(`== refine90 done: ${n} episodes, ${gen} gens, ${clears} honest clears, best obj ${verify.objective.toFixed(3)} honest=${verify.crit.honest} (was ${prevObj})`);
}

if (PHASE === 'search') {
  const RISES = [0.040, 0.060, 0.090];
  const results = { generated: new Date().toISOString(), family: 'C beak-strut vault',
    plant: 'scene.mjb', policy: 'BEST_alpha_stand.onnx', criterion: 'rig3 honest', tail: 'policy',
    flightFix: 'step geoms conaffinity 0 (blocks invisible to each other); 4 steps, run 0.28, riser at x=0.12',
    seed: 20260901, deadline_s: DEADLINE_S, rises: {} };
  const slice = DEADLINE_S / RISES.length;
  let warm = null;
  for (let ri = 0; ri < RISES.length; ri++) {
    const rise = RISES[ri];
    const tag = Math.round(rise * 1000);
    const end = t0 + (slice * (ri + 1)) * 1000;
    RND = mulberry32(20260901 + tag);
    const tmp = SCRATCH + `/cand_${tag}.json`;
    const log = [];
    let best = null, n = 0, clears = 0;

    // CEM from the measured-kinematics prior, with a quarter of every
    // population drawn uniformly from the full bounds so the opposite stroke
    // can still win. Warm-started from the previous rise's elite mean.
    let mu = warm ? { ...warm.mu } : { ...PRIOR };
    let sg = Object.fromEntries(KEYS.map(k => [k, (BOUNDS[k][1] - BOUNDS[k][0]) / 4]));
    const POP = 24, ELITE = 6;
    let gen = 0;

    while (Date.now() < end) {
      const pop = [];
      for (let i = 0; i < POP && Date.now() < end; i++) {
        const p = (i % 4 === 3) ? randP() : sampleN(mu, sg);
        fs.writeFileSync(tmp, JSON.stringify(intentOf(p, `cand ${n}`, rise), null, 2));
        const r = await scoreSaved(tmp, { rise });    // <- read back from disk
        n++;
        if (r.crit.honest) clears++;
        pop.push({ p, r });
        if (!best || r.objective > best.r.objective) {
          best = { p, r };
          const line = { n, t: +el(), objective: +r.objective.toFixed(4), base: +r.base.toFixed(4),
            honest: r.crit.honest, x_mm: mm(r.scored.x), dy_mm: mm(r.scored.dy), z_mm: mm(r.scored.z),
            above_mm: mm(r.scored.above), maxZ_mm: mm(r.maxZ), feetOnTread: r.scored.feetOnTread,
            feetOnTreadMax: r.feetOnTreadMax, feetHighMax: r.feetHighMax,
            headOnlyTicks: r.headOnlyTicks, pivotGain_mm: mm(r.pivotGain),
            headFrac: +r.headFrac.toFixed(3), riserFrac: +r.riserFrac.toFixed(3), up: r.scored.up };
          log.push(line);
          console.log(`[${tag}mm n=${n} ${el()}s] obj=${line.objective} honest=${line.honest} x=${line.x_mm} z=${line.z_mm} above=${line.above_mm} maxZ=${line.maxZ_mm} fot=${line.feetOnTread}/${line.feetOnTreadMax} hiFeet=${line.feetHighMax} headOnly=${line.headOnlyTicks} pivot=${line.pivotGain_mm}mm head=${line.headFrac} riser=${line.riserFrac}`);
          fs.writeFileSync(`../climb/best_r2_vault_${tag}mm.json`, JSON.stringify(intentOf(p, `best beak-strut vault at ${tag} mm; objective ${r.objective.toFixed(3)}; honest=${r.crit.honest}`, rise), null, 2));
        }
      }
      if (pop.length < ELITE) break;
      pop.sort((a, b) => b.r.objective - a.r.objective);
      const el_ = pop.slice(0, ELITE);
      for (const k of KEYS) {
        const vals = el_.map(e => e.p[k]);
        const m = vals.reduce((a, b) => a + b, 0) / vals.length;
        const v = Math.sqrt(vals.reduce((a, b) => a + (b - m) * (b - m), 0) / vals.length);
        mu[k] = m; sg[k] = Math.max(v, (BOUNDS[k][1] - BOUNDS[k][0]) * 0.04);
      }
      gen++;
    }
    warm = { mu: { ...mu } };
    // score the SAVED best file one more time — the number that gets reported
    const savedPath = `../climb/best_r2_vault_${tag}mm.json`;
    const verify = await scoreSaved(savedPath, { rise });
    results.rises[tag] = { rise_mm: tag, episodes: n, generations: gen, clears,
      improvements: log, bestFile: savedPath,
      verified: { objective: +verify.objective.toFixed(4), base: +verify.base.toFixed(4),
        honest: verify.crit.honest, honest60: verify.crit.honest60, orig: verify.crit.orig,
        x_mm: mm(verify.scored.x), dy_mm: mm(verify.scored.dy), z_mm: mm(verify.scored.z),
        above_mm: mm(verify.scored.above), trunkPeakZ_mm: mm(verify.maxZ),
        feetOnTread: verify.scored.feetOnTread, feetOnTreadMax: verify.feetOnTreadMax,
        feetHighMax: verify.feetHighMax, headOnlyTicks: verify.headOnlyTicks,
        pivotGain_mm: mm(verify.pivotGain), headFrac: +verify.headFrac.toFixed(3),
        riserFrac: +verify.riserFrac.toFixed(3), upFrac: +verify.upFrac.toFixed(3),
        satFrac: +verify.satFrac.toFixed(3), up: verify.scored.up,
        minStepGap_mm: verify.minStepGap_mm, treadDriftX_mm: +verify.maxTreadDriftX_mm.toFixed(2) } };
    console.log(`== ${tag}mm done: ${n} episodes, ${gen} gens, ${clears} honest clears, best obj ${verify.objective.toFixed(3)} honest=${verify.crit.honest}`);
    fs.writeFileSync('../climb/vault-results.json', JSON.stringify(results, null, 2));
  }

  // ---- neighbourhood: re-run every saved best at -10 / 0 / +10 mm
  results.neighbourhood = [];
  for (const tag of [40, 60, 90]) {
    for (const d of [-10, 0, 10]) {
      const rise = (tag + d) / 1000;
      const r = await scoreSaved(`../climb/best_r2_vault_${tag}mm.json`, { rise });
      results.neighbourhood.push({ trainedAt_mm: tag, testedAt_mm: tag + d,
        objective: +r.objective.toFixed(4), honest: r.crit.honest, honest60: r.crit.honest60,
        x_mm: mm(r.scored.x), dy_mm: mm(r.scored.dy), above_mm: mm(r.scored.above),
        trunkPeakZ_mm: mm(r.maxZ), feetOnTread: r.scored.feetOnTread, feetOnTreadMax: r.feetOnTreadMax,
        headFrac: +r.headFrac.toFixed(3), riserFrac: +r.riserFrac.toFixed(3), up: r.scored.up });
      console.log(`neigh ${tag}mm @ ${tag + d}mm: obj=${r.objective.toFixed(3)} honest=${r.crit.honest} x=${mm(r.scored.x)} above=${mm(r.scored.above)} fot=${r.scored.feetOnTread}`);
    }
  }
  results.elapsed_s = +el();
  fs.writeFileSync('../climb/vault-results.json', JSON.stringify(results, null, 2));
  console.log('done', el(), 's');
}
