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
let STEP0 = -1; for (let g = 0; g < model.ngeom; g++) if (model.geom(g).name === 'step0_geom') STEP0 = g;
let LFOOT = -1, RFOOT = -1;
for (let g = 0; g < model.ngeom; g++) {
  const n = model.geom(g).name || '';
  if (n === 'left_foot_collision') LFOOT = g;
  if (n === 'right_foot_collision') RFOOT = g;
}
// the exact geom set climb_lib line 141-145 walks
const FEET = []; for (let g = 0; g < model.ngeom; g++) if (/foot_collision|sole/.test(model.geom(g).name || '')) FEET.push(g);

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

// ---------------------------------------------------------------- snapshots

/** Everything the criterion could possibly want, read off the live state. */
function snapshot(h) {
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
  for (const g of FEET) if (data.geom_xpos[g * 3 + 2] > h - 0.005 && data.geom_xpos[g * 3] > RISER_X
    && Math.abs(data.geom_xpos[g * 3 + 1] - STAIR_Y) <= LATERAL) feetOnTread++;
  const foot = g => ({ x: data.geom_xpos[g * 3], y: data.geom_xpos[g * 3 + 1], z: data.geom_xpos[g * 3 + 2] });
  return {
    x, y, z, dy: y - STAIR_Y, above: z - h, up,
    feetUpRaw, feetUpLat, feetOnTread,
    lfoot: foot(LFOOT), rfoot: foot(RFOOT),
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
export function criteria(h, s) {
  const lateral = Math.abs(s.dy) <= LATERAL;
  const orig = s.up && s.x > RISER_X && s.above > 0.095 && s.feetUpRaw >= 2;
  const lat = s.up && lateral && s.x > RISER_X && s.above > 0.095 && s.feetUpLat >= 2;
  const honest = s.up && lateral && s.x > RISER_X && s.above > 0.095 && s.feetOnTread >= 2;
  const honest60 = s.up && lateral && s.x > RISER_X && s.above > 0.060 && s.feetOnTread >= 2;
  return { orig, lat, honest, honest60, lateral };
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
  const cfg = { count: 4, rise: h, run: 0.28, start: 0.12 };
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
              maxX: -1e9, maxZ: -1e9, maxAbsDY: 0, feetOnTreadMax: 0, feetUpRawMax: 0 };

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
      if (data.geom_xpos[g * 3 + 2] > h - 0.005 && data.geom_xpos[g * 3] > RISER_X
        && Math.abs(data.geom_xpos[g * 3 + 1] - STAIR_Y) <= LATERAL) fot++;
    }
    if (fot > R.feetOnTreadMax) R.feetOnTreadMax = fot;
    if (fur > R.feetUpRawMax) R.feetUpRawMax = fur;
  };

  // climb_lib.mjs:121-133, verbatim
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
    if (rec) record();
  };

  /** No policy at all: the servos hold the targets they are given. */
  const holdStep = (targets) => {
    layoutStairs(data, ADDR, cfg);
    for (let k = 0; k < 14; k++) data.ctrl[k] = targets[k];
    for (let s = 0; s < 4; s++) mj.mj_step(model, data);
    record();
  };

  for (let t = 0; t < 25; t++) await step(null, false);
  const x0 = data.qpos[D.freeQpos];
  const total = tr[tr.length - 1].t + 0.8;
  for (let t = 0; t * DT < total; t++) await step(poseAt(tr, t * DT), true);

  const atTrackEnd = snapshot(h);
  // what the servos were last told, and what 'hold' will freeze them at
  const ctrlAtHandoff = []; for (let k = 0; k < 14; k++) ctrlAtHandoff.push(data.ctrl[k]);
  const finalPose = poseAt(tr, total);
  const held = finalPose.map((v, k) => Math.min(Math.max(v, LO[k]), HI[k]));
  let ctrlJump = 0; for (let k = 0; k < 14; k++) ctrlJump = Math.max(ctrlJump, Math.abs(held[k] - ctrlAtHandoff[k]));

  let afterTail;
  if (tail === 'policy') {
    for (let t = 0; t < 50; t++) await step(null, true);
    afterTail = snapshot(h);
  } else {
    for (let t = 0; t < 50; t++) holdStep(held);
    afterTail = snapshot(h);
  }

  const scored = (tail === 'none') ? atTrackEnd : afterTail;
  const rec = {
    tail, rise: h, x0, ctrlJump,
    scored, atTrackEnd, afterTail,
    crit: criteria(h, scored),
    critAtTrackEnd: criteria(h, atTrackEnd),
    critAfterTail: criteria(h, afterTail),
    maxX: R.maxX, maxZ: R.maxZ, maxAbsDY: R.maxAbsDY,
    feetOnTreadMax: R.feetOnTreadMax, feetUpRawMax: R.feetUpRawMax,
    headFrac: R.headTicks / Math.max(R.ticks, 1),
    riserFrac: R.riserTicks / Math.max(R.ticks, 1),
    upFrac: R.upTicks / Math.max(R.ticks, 1),
    satFrac: R.sat / Math.max(R.ctrls, 1),
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
    spawn: j.spawn || null, ...overrides,
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
