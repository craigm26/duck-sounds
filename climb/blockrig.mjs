// blockrig.mjs — family D: DRAG A BLOCK, THEN CLIMB.
//
// Round 2, one family: the duck first shoves a loose 60 mm block flush against
// the riser, then climbs the flight using that block as an intermediate tread.
// At a 90 mm rise the remaining step from the block top is 30 mm; at 180 mm it
// is 120 mm.
//
// THE INSTRUMENT. Everything below the "episode" banner is sim/climb_lib.mjs
// attempt() (lines 99-152) as reproduced by climb/rig3.mjs — same cfg, same
// spawn, same 25-tick settle, same +0.8 s track tail, same 50-tick policy tail,
// same ctrl blend and clamp. The criterion is rig3's `honest` and NOTHING else
// (rig3.mjs criteria()): upright, |y - STAIR_Y| <= 0.17, trunk x > 0.12,
// trunk z - h > 0.095, and at least two /foot_collision|sole/ geoms with
// z > h - 0.005 AND x > 0.12 AND |y - STAIR_Y| <= 0.17. Tail = 'policy'.
//
// TWO THINGS ARE NEW, and both are stated in the header of the scene they
// live in (climb/compile_block.mjs):
//
//  (1) THE FLIGHT NO LONGER COLLIDES WITH ITSELF. Every step geom's
//      conaffinity goes 4 -> 0 in climb/scene_block.xml. A step still collides
//      with the duck and with the props; two steps no longer see each other.
//      Without this, site/stairs.js:24's STEP_HALF_DEPTH = 0.17 against a
//      0.28 run overlaps consecutive 200 kg blocks by 60 mm in x, and below a
//      ~145 mm rise the staircase shoves itself apart under the duck. Phase P
//      measures the difference against rig3 on the untouched plant.
//      No physics constant moves: friction, gravity, timestep, masses,
//      actuator gains and solref are byte-identical to sim/scene_physics.xml.
//
//  (2) A FREE 60 mm CUBE, 100 g, condim 4, friction 0.9 0.02 0.002 — the
//      existing block_a's geom line with a bigger size and mass. Parked at
//      x = 3 m when a run is blockless, so a blockless row here is the same
//      episode as rig3's.
//
// Contact detection is mj_geomDistance, never data.contact.get(i) (embind leak).
// Scoring reads a FILE: every number reported comes from a JSON round-trip,
// pose rounded to 5 dp and scalars to 4 dp exactly as search_2.mjs:307 saves.
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/blockrig.mjs
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
import { findStairJoints, layoutStairs, STAIR_Y, STAIR_HALF_WIDTH } from '../site/stairs.js';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/sb.mjb', new Uint8Array(fs.readFileSync('../climb/scene_block.mjb')));
const model = mj.MjModel.mj_loadBinary('/sb.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model), ADDR = findStairJoints(model);
let GYRO = 0;
for (let i = 0; i < model.nsensor; i++) if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;
const stand = await ort.InferenceSession.create('./BEST_alpha_stand.onnx');
const DT = 1 / C.tickHz;

const LATERAL = STAIR_HALF_WIDTH;   // 0.17
const RISER_X = 0.12;
const BLOCK_HALF = 0.03;

const J = { lhy:0, lhr:1, lhp:2, lk:3, la:4, np:5, hp:6, rhy:9, rhr:10, rhp:11, rk:12, ra:13 };

const geomId = n => { for (let g = 0; g < model.ngeom; g++) if (model.geom(g).name === n) return g; return -1; };
const bodyId = n => { for (let b = 0; b < model.nbody; b++) if (model.body(b).name === n) return b; return -1; };
const STEP0 = geomId('step0_geom'), STEP1 = geomId('step1_geom');
const BLOCKG = geomId('prop_block_geom');
const LFOOT = geomId('left_foot_collision'), RFOOT = geomId('right_foot_collision');
const JAWB = bodyId('jaw_soft');
const JAW = []; for (let g = 0; g < model.ngeom; g++) if (model.geom_bodyid[g] === JAWB && !(model.geom_contype[g] === 0 && model.geom_conaffinity[g] === 0)) JAW.push(g);
const FEET = []; for (let g = 0; g < model.ngeom; g++) if (/foot_collision|sole/.test(model.geom(g).name || '')) FEET.push(g);
let BLOCKQ = -1, BLOCKD = -1;
for (let j = 0; j < model.njnt; j++) if (model.jnt(j).name === 'prop_block_free') { BLOCKQ = model.jnt_qposadr[j]; BLOCKD = model.jnt_dofadr[j]; }
if (BLOCKQ < 0) throw new Error('prop_block_free missing');

const quat = () => [data.qpos[D.freeQpos+3], data.qpos[D.freeQpos+4], data.qpos[D.freeQpos+5], data.qpos[D.freeQpos+6]];

/** climb_lib.mjs:80-86, verbatim. */
function poseAt(tr, time) {
  if (time <= 0) return HOME.slice();
  let pt = 0, pp = HOME;
  for (const f of tr) {
    if (time <= f.t) { const u = (time - pt) / Math.max(f.t - pt, 1e-9), s = u*u*(3-2*u);
      return f.pose.map((v, k) => pp[k] + (v - pp[k]) * s); }
    pt = f.t; pp = f.pose;
  }
  return tr[tr.length-1].pose.slice();
}

function snapshot(h) {
  const x = data.qpos[D.freeQpos], y = data.qpos[D.freeQpos+1], z = data.qpos[D.freeQpos+2];
  const up = projectedGravity(quat())[2] < -0.90;
  let feetOnTread = 0, feetUpRaw = 0;
  for (const g of FEET) {
    if (data.geom_xpos[g*3+2] > h - 0.005 && data.geom_xpos[g*3] > 0.05) feetUpRaw++;
    if (data.geom_xpos[g*3+2] > h - 0.005 && data.geom_xpos[g*3] > RISER_X
        && Math.abs(data.geom_xpos[g*3+1] - STAIR_Y) <= LATERAL) feetOnTread++;
  }
  return { x, y, z, dy: y - STAIR_Y, above: z - h, up, feetOnTread, feetUpRaw };
}

/** rig3.mjs criteria() — `honest` is the only one that decides anything here. */
function criteria(h, s) {
  const lateral = Math.abs(s.dy) <= LATERAL;
  return {
    honest: s.up && lateral && s.x > RISER_X && s.above > 0.095 && s.feetOnTread >= 2,
    honest60: s.up && lateral && s.x > RISER_X && s.above > 0.060 && s.feetOnTread >= 2,
    orig: s.up && s.x > RISER_X && s.above > 0.095 && s.feetUpRaw >= 2,
  };
}

/** rig3.mjs reward(), unchanged. */
function reward(rec) {
  const s = rec.scored;
  if (rec.maxAbsDY > LATERAL) return 0;
  if (Math.abs(s.dy) > LATERAL) return 0;
  let r = 0;
  r += 3 * Math.max(0, Math.min(1, (s.x - (RISER_X - 0.20)) / 0.20));
  r += 2 * s.feetOnTread;
  r += 4 * Math.max(0, Math.min(1, s.above / 0.095));
  r += s.up ? 1 : 0;
  return r;
}

// ==================================================================== episode

async function runEpisodeRaw(intent, h, { tail = 'policy', stepCount = 4, trace = false } = {}) {
  const cfg = { count: stepCount, rise: h, run: 0.28, start: 0.12 };
  mj.mj_resetData(model, data);
  layoutStairs(data, ADDR, cfg);

  // the block: on the flight a set distance in front of the riser, or parked
  const B = intent.block || { on: false };
  const bx = B.on ? B.x : 3.0;
  const by = B.on ? STAIR_Y + (B.dy || 0) : STAIR_Y;
  data.qpos[BLOCKQ] = bx; data.qpos[BLOCKQ+1] = by; data.qpos[BLOCKQ+2] = BLOCK_HALF;
  data.qpos[BLOCKQ+3] = 1; data.qpos[BLOCKQ+4] = 0; data.qpos[BLOCKQ+5] = 0; data.qpos[BLOCKQ+6] = 0;
  for (let k = 0; k < 6; k++) data.qvel[BLOCKD+k] = 0;

  // climb_lib.mjs:110-116, verbatim (plus rig3's spawn override)
  if (intent.spawn) {
    data.qpos[D.freeQpos] = intent.spawn.x; data.qpos[D.freeQpos+1] = intent.spawn.y;
    data.qpos[D.freeQpos+2] = intent.spawn.z;
  } else {
    data.qpos[D.freeQpos] = 0.12 - 0.07 - intent.gap;
    data.qpos[D.freeQpos+1] = STAIR_Y + (intent.side || 0);
    data.qpos[D.freeQpos+2] = 0.12;
  }
  data.qpos[D.freeQpos+3] = 1;
  for (let i = 0; i < 14; i++) { data.qpos[D.qpos[i]] = HOME[i]; data.ctrl[i] = HOME[i]; }
  mj.mj_forward(model, data);

  const tr = intent.keyframes.map(f => ({ t: f.t, pose: f.pose.slice() }));
  let la = new Array(14).fill(0);
  let cmd = command({ vx: intent.approach || 0 });

  const R = { ticks:0, headTicks:0, riserTicks:0, upTicks:0, blockFootTicks:0,
              maxX:-1e9, maxZ:-1e9, maxAbsDY:0, feetOnTreadMax:0,
              minStepGap_mm:1e9, maxTreadSag_mm:0, maxTreadDriftX_mm:0, trace:[],
              // STANDING on the block is not the same event as TOUCHING it: the
              // shove itself is 100% foot-block contact. A foot counts as on the
              // block only when it is at the block's top face and over the
              // block's footprint.
              footOnBlockTopTicks:0, bothFeetOnBlockTicks:0, maxZonBlock:-1e9,
              feetRestingMax:0, uprightBothRestingTicks:0 };
  let gtick = 0;

  const treadDrift = () => {
    const sag = (h - (data.geom_xpos[STEP0*3+2] + 0.10)) * 1000;
    if (sag > R.maxTreadSag_mm) R.maxTreadSag_mm = sag;
    const dx = Math.abs(data.geom_xpos[STEP0*3] - (0.12 + 0.17)) * 1000;
    if (dx > R.maxTreadDriftX_mm) R.maxTreadDriftX_mm = dx;
    if (STEP1 >= 0 && cfg.count > 1) {
      const g = mj.mj_geomDistance(model, data, STEP0, STEP1, 0.4, null) * 1000;
      if (g < R.minStepGap_mm) R.minStepGap_mm = g;
    }
  };

  const record = () => {
    R.ticks++;
    const x = data.qpos[D.freeQpos], y = data.qpos[D.freeQpos+1], z = data.qpos[D.freeQpos+2];
    if (x > R.maxX) R.maxX = x;
    if (z > R.maxZ) R.maxZ = z;
    const ady = Math.abs(y - STAIR_Y); if (ady > R.maxAbsDY) R.maxAbsDY = ady;
    if (projectedGravity(quat())[2] < -0.90) R.upTicks++;
    let head = false;
    for (const g of JAW) if (mj.mj_geomDistance(model, data, g, STEP0, 0.05, null) < 0.003) { head = true; break; }
    if (head) R.headTicks++;
    let footRiser = false, footBlock = false;
    for (const g of [LFOOT, RFOOT]) {
      if (data.geom_xpos[g*3+2] < h - 0.005 && mj.mj_geomDistance(model, data, g, STEP0, 0.05, null) < 0.003) footRiser = true;
      if (B.on && mj.mj_geomDistance(model, data, g, BLOCKG, 0.05, null) < 0.003) footBlock = true;
    }
    if (footRiser) R.riserTicks++;
    if (footBlock) R.blockFootTicks++;
    if (B.on) {
      const btop = data.qpos[BLOCKQ+2] + BLOCK_HALF;
      let onTop = 0;
      for (const g of [LFOOT, RFOOT]) {
        const fx = data.geom_xpos[g*3], fy = data.geom_xpos[g*3+1], fz = data.geom_xpos[g*3+2];
        if (fz > btop - 0.010 && Math.abs(fx - data.qpos[BLOCKQ]) < 0.05
            && Math.abs(fy - data.qpos[BLOCKQ+1]) < 0.05) onTop++;
      }
      if (onTop >= 1) { R.footOnBlockTopTicks++; if (z > R.maxZonBlock) R.maxZonBlock = z; }
      if (onTop >= 2) R.bothFeetOnBlockTicks++;
    }
    let fot = 0, fresting = 0;
    for (const g of FEET) {
      const fx = data.geom_xpos[g*3], fy = data.geom_xpos[g*3+1], fz = data.geom_xpos[g*3+2];
      const lat = Math.abs(fy - STAIR_Y) <= LATERAL;
      if (fz > h - 0.005 && fx > RISER_X && lat) fot++;
      // rig3's foot test (climb_lib.mjs:144 with the lateral gate and the
      // x > 0.12 fix) has NO CEILING on z: a foot 120 mm ABOVE the tread, in
      // mid-air over the step because the duck has fallen past it, counts as
      // "on the tread". This is the same class of bug as the x > 0.05 one.
      // A foot RESTING on the tread is within a foot's thickness of it.
      if (fz > h - 0.005 && fz < h + 0.045 && fx > RISER_X && lat) fresting++;
    }
    if (fot > R.feetOnTreadMax) R.feetOnTreadMax = fot;
    if (fresting > R.feetRestingMax) R.feetRestingMax = fresting;
    if (fresting >= 2 && projectedGravity(quat())[2] < -0.90) R.uprightBothRestingTicks++;
  };

  const sampleTrace = (phase) => {
    if (!trace || gtick % 10) return;
    R.trace.push({ tick: gtick, phase,
      x_mm: +(data.qpos[D.freeQpos]*1000).toFixed(1),
      z_mm: +(data.qpos[D.freeQpos+2]*1000).toFixed(1),
      dy_mm: +((data.qpos[D.freeQpos+1]-STAIR_Y)*1000).toFixed(1),
      blockX_mm: +(data.qpos[BLOCKQ]*1000).toFixed(1),
      blockZ_mm: +(data.qpos[BLOCKQ+2]*1000).toFixed(1),
      lfootX_mm: +(data.geom_xpos[LFOOT*3]*1000).toFixed(1),
      lfootZ_mm: +(data.geom_xpos[LFOOT*3+2]*1000).toFixed(1),
      rfootX_mm: +(data.geom_xpos[RFOOT*3]*1000).toFixed(1),
      rfootZ_mm: +(data.geom_xpos[RFOOT*3+2]*1000).toFixed(1),
      up: projectedGravity(quat())[2] < -0.90 });
  };

  // climb_lib.mjs:121-133, verbatim apart from the recorder hooks
  const step = async (off, blend, rec) => {
    layoutStairs(data, ADDR, cfg);
    const q = quat(); const jp = [], jv = [];
    for (let k = 0; k < 14; k++) { jp.push(data.qpos[D.qpos[k]]); jv.push(data.qvel[D.dof[k]]); }
    const obs = buildObs([data.sensordata[GYRO], data.sensordata[GYRO+1], data.sensordata[GYRO+2]], projectedGravity(q), jp, jv, la, cmd);
    const r = await stand.run({ obs: new ort.Tensor('float32', obs, [1, 61]) });
    la = Array.from(r.actions.data);
    for (let k = 0; k < 14; k++) {
      const v = HOME[k] + la[k] + (off ? (off[k] - HOME[k]) * blend : 0);
      data.ctrl[k] = Math.min(Math.max(v, LO[k]), HI[k]);
    }
    for (let s = 0; s < 4; s++) mj.mj_step(model, data);
    treadDrift(); sampleTrace(rec ? 'run' : 'settle'); gtick++;
    if (rec) record();
  };

  // --- settle (climb_lib's 25 ticks)
  for (let t = 0; t < 25; t++) await step(null, 0, false);

  // --- STAGE 1: shove the block. A walk command plus, optionally, a held
  //     "push pose" (neck/head down and out) blended in the same way the climb
  //     track is. Skipped entirely when push.t1 is 0, which makes a blockless
  //     run identical to rig3's episode.
  const P = intent.push || { t1: 0 };
  let stage1 = null;
  if (P.t1 > 0) {
    cmd = command({ vx: P.vx });
    const pushPose = HOME.slice();
    pushPose[J.np] = P.neck; pushPose[J.hp] = P.head;
    const n1 = Math.round(P.t1 / DT);
    for (let t = 0; t < n1; t++) await step(pushPose, P.blend, true);
    stage1 = {
      blockX: data.qpos[BLOCKQ], blockY: data.qpos[BLOCKQ+1], blockZ: data.qpos[BLOCKQ+2],
      // gap between the block's front face and the riser face
      blockGap_mm: (RISER_X - (data.qpos[BLOCKQ] + BLOCK_HALF)) * 1000,
      blockDY_mm: (data.qpos[BLOCKQ+1] - STAIR_Y) * 1000,
      duckX: data.qpos[D.freeQpos], duckZ: data.qpos[D.freeQpos+2],
      up: projectedGravity(quat())[2] < -0.90,
      footOnBlockFrac: R.blockFootTicks / Math.max(R.ticks, 1),
    };
    cmd = command({ vx: intent.approach || 0 });
  }

  // --- STAGE 2: climb_lib's track playback, verbatim
  const total = tr[tr.length-1].t + 0.8;
  for (let t = 0; t*DT < total; t++) await step(poseAt(tr, t*DT), intent.blend, true);
  const atTrackEnd = snapshot(h);

  // --- the tail climb_lib scores at
  if (tail === 'policy') { for (let t = 0; t < 50; t++) await step(null, 0, true); }
  const afterTail = snapshot(h);
  const scored = tail === 'none' ? atTrackEnd : afterTail;

  const rec = {
    rise: h, tail, stepCount, scored, atTrackEnd, afterTail,
    crit: criteria(h, scored), critAtTrackEnd: criteria(h, atTrackEnd),
    stage1,
    blockEnd: { x: data.qpos[BLOCKQ], z: data.qpos[BLOCKQ+2],
                dy: data.qpos[BLOCKQ+1] - STAIR_Y,
                gap_mm: (RISER_X - (data.qpos[BLOCKQ] + BLOCK_HALF)) * 1000,
                onTread: B.on && data.qpos[BLOCKQ] > RISER_X && data.qpos[BLOCKQ+2] > h },
    maxX: R.maxX, maxZ: R.maxZ, maxAbsDY: R.maxAbsDY, feetOnTreadMax: R.feetOnTreadMax,
    feetRestingMax: R.feetRestingMax, uprightBothRestingTicks: R.uprightBothRestingTicks,
    headFrac: R.headTicks / Math.max(R.ticks,1),
    riserFrac: R.riserTicks / Math.max(R.ticks,1),
    blockFootFrac: R.blockFootTicks / Math.max(R.ticks,1),
    footOnBlockTopFrac: R.footOnBlockTopTicks / Math.max(R.ticks,1),
    bothFeetOnBlockFrac: R.bothFeetOnBlockTicks / Math.max(R.ticks,1),
    maxZonBlock: R.maxZonBlock === -1e9 ? null : R.maxZonBlock,
    upFrac: R.upTicks / Math.max(R.ticks,1),
    minStepGap_mm: R.minStepGap_mm === 1e9 ? null : R.minStepGap_mm,
    maxTreadSag_mm: R.maxTreadSag_mm, maxTreadDriftX_mm: R.maxTreadDriftX_mm,
    trace: R.trace,
  };
  rec.reward = reward(rec);
  return rec;
}

// ------------------------------------------------------------- the ONLY scorer
const r5 = v => +(+v).toFixed(5), r4 = v => +(+v).toFixed(4);
function exportIntent(intent) {
  return {
    family: intent.family || 'block',
    gap: r4(intent.gap || 0), side: r4(intent.side || 0),
    approach: r4(intent.approach || 0), blend: r4(intent.blend || 1),
    spawn: intent.spawn || null,
    block: intent.block && intent.block.on
      ? { on: true, x: r4(intent.block.x), dy: r4(intent.block.dy || 0) } : { on: false },
    push: intent.push && intent.push.t1 > 0
      ? { t1: r4(intent.push.t1), vx: r4(intent.push.vx), blend: r4(intent.push.blend),
          neck: r4(intent.push.neck), head: r4(intent.push.head) } : { t1: 0 },
    keyframes: intent.keyframes.map(f => ({ t: r4(f.t), pose: f.pose.map(r5) })),
  };
}
/** Write, read back, score. Nothing here scores an in-memory candidate. */
async function scoreSaved(path, h, opts) {
  const j = JSON.parse(fs.readFileSync(path, 'utf8'));
  if (!Array.isArray(j.keyframes) || !j.keyframes.length) throw new Error('no keyframes: ' + path);
  for (const f of j.keyframes) if (!Array.isArray(f.pose) || f.pose.length !== 14) throw new Error('bad pose: ' + path);
  const rec = await runEpisodeRaw(j, h, opts);
  rec.source = path;
  return rec;
}
function saveAndScore(intent, h, path, opts) {
  fs.writeFileSync(path, JSON.stringify(exportIntent(intent), null, 2));
  return scoreSaved(path, h, opts);
}

export { runEpisodeRaw, scoreSaved, saveAndScore, exportIntent, criteria, reward, poseAt,
         HOME, LO, HI, J, DT, STAIR_Y, LATERAL, RISER_X, BLOCK_HALF };
