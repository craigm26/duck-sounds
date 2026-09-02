// search_block.mjs — round 2, family D (drag a block, then climb).
// ONE node process, seeded (mulberry32), every improvement logged, every
// reported number produced by re-reading a JSON file off disk.
//   cd ~/projects/duck-sounds/sim && node ../climb/search_block.mjs
import fs from 'node:fs';
import { runEpisodeRaw, saveAndScore, scoreSaved, exportIntent, criteria,
         HOME, LO, HI, J, STAIR_Y, LATERAL, RISER_X, BLOCK_HALF } from '../climb/blockrig.mjs';

const OUT = '../climb/search_block-results.json';
const SCRATCH = '/tmp/claude-1000/-home-craigm26/43f07dce-e62f-464d-ae1f-2fe020620950/scratchpad';
try { fs.mkdirSync(SCRATCH, { recursive: true }); } catch {}
const CAND = SCRATCH + '/cand.json';
const mm = v => (v * 1000).toFixed(1);
const T0 = Date.now();
const el = () => ((Date.now() - T0) / 1000).toFixed(0) + 's';
const R = { generated: new Date().toISOString(), plant: '../climb/scene_block.mjb',
            policy: 'BEST_alpha_stand.onnx', criterion: 'rig3 honest, tail=policy',
            phases: {} };
const say = (...a) => console.log(...a);

/** mulberry32 */
function rng(seed) { return function () {
  seed |= 0; seed = seed + 0x6D2B79F5 | 0;
  let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
  t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
  return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

// ============================================================ the search space
// stage 2 is climb_lib.mjs's own B (lines 30-49), unchanged.
const B = {
  gap:[0.02,0.12], approach:[0.0,0.4], side:[0.0,0.085],
  tShelf:[0.25,0.9], tA:[0.10,0.45], tB:[0.10,0.45], tC:[0.10,0.45], tOver:[0.2,0.9],
  shelfNeck:[0.3,1.5], shelfHead:[-0.6,1.5],
  rollA:[-0.9,0.9], rollB:[-0.9,0.9], rollC:[-0.9,0.9],
  hipA:[0.0,1.5], kneeA:[-1.5,0.6], ankA:[-1.4,0.9],
  hipB:[0.0,1.5], kneeB:[-1.5,0.6], ankB:[-1.4,0.9],
  hipC:[0.0,1.6], kneeC:[-1.5,0.6], ankC:[-1.4,0.9],
  overHip:[-1.4,1.4], overKnee:[-1.4,1.4], overNeck:[-1.3,1.0], overHead:[-1.3,1.0],
  blend:[0.7,2.4],
};
// stage 1 (the shove) is new.
const PB = {
  pt1:[0.6,3.0], pvx:[0.05,0.40], pblend:[0.3,2.0], pneck:[-0.4,1.5], phead:[-1.0,1.5],
  bx:[0.00,0.075], bdy:[-0.06,0.06],
};
const ALL = { ...B, ...PB };
const pick = (r, sp) => Object.fromEntries(Object.keys(sp).map(k => [k, sp[k][0] + r()*(sp[k][1]-sp[k][0])]));
const jit = (r, p, s, sp) => Object.fromEntries(Object.keys(sp).map(k => {
  const [a,b] = sp[k]; return [k, Math.min(b, Math.max(a, p[k] + (r()*2-1)*(b-a)*s))]; }));

/** climb_lib.mjs trackOf(), verbatim. */
function trackOf(p) {
  const shelf = HOME.slice();
  shelf[J.np] = p.shelfNeck; shelf[J.hp] = p.shelfHead;
  const a = shelf.slice();
  a[J.lhp]=HOME[J.lhp]+p.hipA; a[J.lk]=HOME[J.lk]+p.kneeA; a[J.la]=HOME[J.la]+p.ankA;
  a[J.lhr]=HOME[J.lhr]+p.rollA; a[J.rhr]=HOME[J.rhr]+p.rollA;
  const b = a.slice();
  b[J.rhp]=HOME[J.rhp]-p.hipB; b[J.rk]=HOME[J.rk]-p.kneeB; b[J.ra]=HOME[J.ra]-p.ankB;
  b[J.lhr]=HOME[J.lhr]+p.rollB; b[J.rhr]=HOME[J.rhr]+p.rollB;
  const c = b.slice();
  c[J.lhp]=HOME[J.lhp]+p.hipC; c[J.lk]=HOME[J.lk]+p.kneeC; c[J.la]=HOME[J.la]+p.ankC;
  c[J.lhr]=HOME[J.lhr]+p.rollC; c[J.rhr]=HOME[J.rhr]+p.rollC;
  const d = c.slice();
  d[J.lhp]=HOME[J.lhp]+p.overHip; d[J.rhp]=HOME[J.rhp]-p.overHip;
  d[J.lk]=HOME[J.lk]+p.overKnee;  d[J.rk]=HOME[J.rk]-p.overKnee;
  d[J.np]=p.overNeck; d[J.hp]=p.overHead;
  const t1=p.tShelf, t2=t1+p.tA, t3=t2+p.tB, t4=t3+p.tC, t5=t4+p.tOver;
  return [{t:t1,pose:shelf},{t:t2,pose:a},{t:t3,pose:b},{t:t4,pose:c},{t:t5,pose:d},
          {t:t5+0.7,pose:HOME.slice()}];
}
const intentOf = (p, withBlock) => ({
  family: 'block', gap: p.gap, side: p.side, approach: p.approach, blend: p.blend,
  block: withBlock ? { on: true, x: p.bx, dy: p.bdy } : { on: false },
  push: withBlock ? { t1: p.pt1, vx: p.pvx, blend: p.pblend, neck: p.pneck, head: p.phead } : { t1: 0 },
  keyframes: trackOf(p),
});

// ================================================================== PHASE P
say('=== PHASE P — what the collision-bit fix changes, against rig3 on the untouched plant ===');
{
  const { __parityEpisode } = await import('../climb/rig3.mjs');
  const a = HOME.slice(); a[5] = -1.3; a[6] = 0.7; a[7] = 1.4;
  const track = [{ t: 0.5, pose: a }, { t: 1.6, pose: HOME.slice() }];
  const opts = { blend: 1.6, approach: 0, gap: 0.03, side: 0.06 };
  const rows = [];
  for (const [h, sc] of [[0.180,1],[0.180,4],[0.090,4],[0.040,4]]) {
    const lib = await __parityEpisode(track, { ...opts, stepCount: sc }, h, 'policy');
    const mine = await runEpisodeRaw({ ...opts, block:{on:false}, push:{t1:0}, keyframes: track }, h, { stepCount: sc });
    const d = Math.abs(lib.scored.x - mine.scored.x) + Math.abs(lib.scored.z - mine.scored.z);
    rows.push({ rise_mm: h*1000, steps: sc,
      rig3: { x: lib.scored.x, z: lib.scored.z, minStepGap_mm: lib.minStepGap_mm, maxTreadDriftX_mm: lib.maxTreadDriftX_mm },
      fixed: { x: mine.scored.x, z: mine.scored.z, minStepGap_mm: mine.minStepGap_mm, maxTreadDriftX_mm: mine.maxTreadDriftX_mm },
      diff: d });
    say(`  rise ${(h*1000).toFixed(0)}mm steps=${sc}  rig3 x=${lib.scored.x.toFixed(9)} z=${lib.scored.z.toFixed(9)} gap=${lib.minStepGap_mm===null?'-':lib.minStepGap_mm.toFixed(2)} driftX=${lib.maxTreadDriftX_mm.toFixed(2)}`);
    say(`                        fixed x=${mine.scored.x.toFixed(9)} z=${mine.scored.z.toFixed(9)} gap=${mine.minStepGap_mm===null?'-':mine.minStepGap_mm.toFixed(2)} driftX=${mine.maxTreadDriftX_mm.toFixed(2)}   |diff|=${d.toExponential(3)}`);
  }
  R.phases.P = rows;
}

// ================================================================== PHASE A
say(`=== PHASE A [${el()}] — controls on the fixed flight (blockless), criterion=honest ===`);
{
  const rows = [];
  const doNothing = { gap: 0.06, side: 0, approach: 0, blend: 1.0, block:{on:false}, push:{t1:0},
                      keyframes: [{ t: 2.0, pose: HOME.slice() }] };
  for (const h of [0.090, 0.180]) {
    const r = await saveAndScore(doNothing, h, CAND, {});
    rows.push({ control: 'do-nothing', rise_mm: h*1000, honest: r.crit.honest, x_mm:+mm(r.scored.x), z_mm:+mm(r.scored.z), feetOnTread: r.scored.feetOnTread });
    say(`  do-nothing  ${(h*1000).toFixed(0)}mm  honest=${r.crit.honest}  x=${mm(r.scored.x)} z=${mm(r.scored.z)} feet=${r.scored.feetOnTread}`);
  }
  const walk = { gap: 0.06, side: 0, approach: 0.35, blend: 1.0, block:{on:false}, push:{t1:0},
                 keyframes: [{ t: 3.0, pose: HOME.slice() }] };
  for (const h of [0.090, 0.180]) {
    const r = await saveAndScore(walk, h, CAND, {});
    rows.push({ control: 'walk-only', rise_mm: h*1000, honest: r.crit.honest, x_mm:+mm(r.scored.x), z_mm:+mm(r.scored.z), feetOnTread: r.scored.feetOnTread });
    say(`  walk-only   ${(h*1000).toFixed(0)}mm  honest=${r.crit.honest}  x=${mm(r.scored.x)} z=${mm(r.scored.z)} feet=${r.scored.feetOnTread}`);
  }
  for (const h of [0.090, 0.180]) {
    const onTread = { gap: 0, side: 0, approach: 0, blend: 1.0, block:{on:false}, push:{t1:0},
                      spawn: { x: 0.12 + 0.14, y: STAIR_Y, z: h + 0.12 },
                      keyframes: [{ t: 2.0, pose: HOME.slice() }] };
    const r = await saveAndScore(onTread, h, CAND, {});
    rows.push({ control: 'on-tread', rise_mm: h*1000, honest: r.crit.honest, x_mm:+mm(r.scored.x), z_mm:+mm(r.scored.z), feetOnTread: r.scored.feetOnTread });
    say(`  on-tread    ${(h*1000).toFixed(0)}mm  honest=${r.crit.honest}  x=${mm(r.scored.x)} z=${mm(r.scored.z)} feet=${r.scored.feetOnTread}`);
  }
  R.phases.A = rows;
}

// ================================================================== PHASE B
// Stage 1 alone: can the duck put the block flush against the riser, and does
// it end up standing on it? Objective: minimise |gap| (front face to riser),
// hard-gated on staying on the flight and upright.
const BUDGET_B_MS = Number(process.env.BUDGET_B_MIN || 8) * 60 * 1000;
say(`=== PHASE B [${el()}] — stage 1 alone: shove the block flush (budget 8 min) ===`);
let bestPush = null;
{
  const r = rng(0xB10C);
  const stub = [{ t: 0.2, pose: HOME.slice() }];   // no climb: stage 1 is the whole episode
  const scoreP = async (p) => {
    const intent = { family:'block-push', gap: p.gap, side: p.side, approach: 0, blend: 1.0,
                     block: { on: true, x: p.bx, dy: p.bdy },
                     push: { t1: p.pt1, vx: p.pvx, blend: p.pblend, neck: p.pneck, head: p.phead },
                     keyframes: stub };
    const rec = await saveAndScore(intent, 0.090, CAND, {});
    const s1 = rec.stage1;
    const off = Math.abs(s1.blockDY_mm) > LATERAL*1000 || Math.abs(rec.maxAbsDY) > LATERAL;
    // Flush = the block's FRONT FACE at the riser face; negative gap means it is
    // jammed into it. Three other things matter and are priced in, because a
    // block that arrives flush but 60 mm off the duck's line, or on its corner,
    // is not a step: keep it near the centre of the flight, keep it FLAT
    // (a 60 mm cube resting square has its centre at z = 30 mm; a tilted one
    // rides higher), and keep the duck on its feet.
    const tilt = Math.abs(s1.blockZ - BLOCK_HALF) * 1000;
    const obj = off ? -1000
      : -Math.abs(s1.blockGap_mm) - 0.25 * Math.abs(s1.blockDY_mm) - 0.5 * tilt + (s1.up ? 5 : 0);
    return { rec, obj, intent };
  };
  let best = null, n = 0;
  const t1 = Date.now();
  while (Date.now() - t1 < BUDGET_B_MS) {
    const p = (best && r() < 0.65) ? jit(r, best.p, r() < 0.5 ? 0.08 : 0.25, ALL) : pick(r, ALL);
    const { rec, obj, intent } = await scoreP(p); n++;
    if (!best || obj > best.obj) {
      best = { p, obj, rec, intent };
      say(`  [${el()}] n=${n} obj=${obj.toFixed(2)}  blockGap=${rec.stage1.blockGap_mm.toFixed(1)}mm dy=${rec.stage1.blockDY_mm.toFixed(1)}mm blockZ=${mm(rec.stage1.blockZ)}mm duckX=${mm(rec.stage1.duckX)}mm up=${rec.stage1.up} footOnBlock=${(rec.blockFootFrac*100).toFixed(0)}%  vx=${p.pvx.toFixed(3)} t1=${p.pt1.toFixed(2)} bx0=${mm(p.bx)}mm`);
    }
  }
  bestPush = best;
  fs.writeFileSync('../climb/best_r2_blockpush.json', JSON.stringify(exportIntent(best.intent), null, 2));
  const v = await scoreSaved('../climb/best_r2_blockpush.json', 0.090, { trace: true });
  R.phases.B = { episodes: n, best: { params: best.p, obj: best.obj },
    verified: { blockGap_mm: v.stage1.blockGap_mm, blockDY_mm: v.stage1.blockDY_mm,
      blockZ_mm: +mm(v.stage1.blockZ), blockX_mm: +mm(v.stage1.blockX),
      duckX_mm: +mm(v.stage1.duckX), up: v.stage1.up,
      footOnBlockFrac: v.blockFootFrac, flushWithin20mm: Math.abs(v.stage1.blockGap_mm) <= 20 },
    trace: v.trace };
  say(`  PHASE B done: ${n} episodes. verified-from-file blockGap=${v.stage1.blockGap_mm.toFixed(1)}mm (flush<=20mm: ${Math.abs(v.stage1.blockGap_mm)<=20})`);
}

// ================================================================== PHASE C
const BUDGET_C_MS = Number(process.env.BUDGET_C_MIN || 24) * 60 * 1000;
say(`=== PHASE C [${el()}] — two-stage: push then climb, at 90 and 180 mm ===`);
const bests = {};
{
  const RISES = [0.090, 0.180];
  const per = BUDGET_C_MS / RISES.length;
  for (let ri = 0; ri < RISES.length; ri++) {
    const h = RISES[ri];
    const r = rng(0xC0DE + ri * 977);
    let best = null, n = 0, cleared = 0;
    const tS = Date.now();
    // seed the push at phase B's answer
    const seedPush = bestPush.p;
    while (Date.now() - tS < per) {
      let p;
      if (!best) { p = { ...pick(r, ALL), pt1: seedPush.pt1, pvx: seedPush.pvx, pblend: seedPush.pblend,
                          pneck: seedPush.pneck, phead: seedPush.phead, bx: seedPush.bx, bdy: seedPush.bdy,
                          gap: seedPush.gap, side: seedPush.side }; }
      else if (r() < 0.7) p = jit(r, best.p, r() < 0.5 ? 0.08 : 0.25, ALL);
      else p = { ...pick(r, ALL), pt1: seedPush.pt1, pvx: seedPush.pvx, pblend: seedPush.pblend,
                 pneck: seedPush.pneck, phead: seedPush.phead, bx: seedPush.bx, bdy: seedPush.bdy };
      const intent = intentOf(p, true);
      const rec = await saveAndScore(intent, h, CAND, {}); n++;
      if (rec.crit.honest) cleared++;
      // SEARCH-ONLY SHAPING. rig3's reward() is scored at one instant after the
      // tail, so an episode that gets a foot onto the tread mid-flight and then
      // loses it is worth exactly as much as one that never left the floor.
      // These two extra terms are gradient, not criterion: nothing here relaxes
      // `honest`, and every reported number is rig3's reward and rig3's crit.
      const obj = rec.reward + (rec.crit.honest ? 100 : 0)
        + 2 * rec.feetOnTreadMax
        + 3 * Math.max(0, Math.min(1, (rec.maxZ - 0.116) / 0.100));
      if (!best || obj > best.obj) {
        best = { p, obj, intent, rec };
        say(`  [${el()}] ${(h*1000).toFixed(0)}mm n=${n} obj=${obj.toFixed(3)}(shaped) reward=${rec.reward.toFixed(3)} honest=${rec.crit.honest} x=${mm(rec.scored.x)} z=${mm(rec.scored.z)} above=${mm(rec.scored.above)} feetOnTread=${rec.scored.feetOnTread}/max${rec.feetOnTreadMax} peakZ=${mm(rec.maxZ)} head=${(rec.headFrac*100).toFixed(0)}% riser=${(rec.riserFrac*100).toFixed(0)}% block=${(rec.blockFootFrac*100).toFixed(0)}% blockGap=${rec.blockEnd.gap_mm.toFixed(0)}mm`);
      }
    }
    const path = `../climb/best_r2_blockclimb_${(h*1000).toFixed(0)}mm.json`;
    fs.writeFileSync(path, JSON.stringify(exportIntent(best.intent), null, 2));
    const v = await scoreSaved(path, h, { trace: true });
    bests[h] = { path, params: best.p, verified: v };
    say(`  ${(h*1000).toFixed(0)}mm: ${n} episodes, ${cleared} cleared honest. re-scored from ${path}: honest=${v.crit.honest} reward=${v.reward.toFixed(3)} x=${mm(v.scored.x)} z=${mm(v.scored.z)}`);
    R.phases[`C_${(h*1000).toFixed(0)}mm`] = {
      episodes: n, clearedHonest: cleared, params: best.p, path,
      verified: {
        honest: v.crit.honest, honest60: v.crit.honest60, orig: v.crit.orig, reward: v.reward,
        x_mm:+mm(v.scored.x), y_dy_mm:+mm(v.scored.dy), z_mm:+mm(v.scored.z), above_mm:+mm(v.scored.above),
        up: v.scored.up, feetOnTread: v.scored.feetOnTread, feetOnTreadMax: v.feetOnTreadMax,
        peakZ_mm:+mm(v.maxZ), maxX_mm:+mm(v.maxX), maxAbsDY_mm:+mm(v.maxAbsDY),
        headFrac: v.headFrac, riserFrac: v.riserFrac, blockFootFrac: v.blockFootFrac, upFrac: v.upFrac,
        minStepGap_mm: v.minStepGap_mm, maxTreadSag_mm: v.maxTreadSag_mm,
        blockEnd_x_mm:+mm(v.blockEnd.x), blockEnd_z_mm:+mm(v.blockEnd.z), blockEnd_gap_mm: v.blockEnd.gap_mm,
        blockOnTread: v.blockEnd.onTread,
        stage1: v.stage1,
      },
      trace: v.trace,
    };
  }
}

// ================================================================== PHASE D
say(`=== PHASE D [${el()}] — re-run each best at -10/0/+10 mm, and blockless ablation ===`);
{
  const rows = [];
  for (const key of Object.keys(bests)) {
    const h0 = Number(key), b = bests[key];
    for (const dh of [-0.010, 0, 0.010]) {
      const h = h0 + dh;
      const v = await scoreSaved(b.path, h, {});
      rows.push({ from: b.path, rise_mm: +(h*1000).toFixed(0), dh_mm: dh*1000, honest: v.crit.honest,
        reward: v.reward, x_mm:+mm(v.scored.x), z_mm:+mm(v.scored.z), above_mm:+mm(v.scored.above),
        feetOnTread: v.scored.feetOnTread, peakZ_mm:+mm(v.maxZ), blockGap_mm: v.blockEnd.gap_mm });
      say(`  ${b.path.split('/').pop()} @ ${(h*1000).toFixed(0)}mm  honest=${v.crit.honest} reward=${v.reward.toFixed(3)} x=${mm(v.scored.x)} z=${mm(v.scored.z)} feet=${v.scored.feetOnTread}`);
    }
    // ablation: same track, block parked 3 m away
    const j = JSON.parse(fs.readFileSync(b.path, 'utf8'));
    j.block = { on: false }; j.push = { t1: 0 };
    const ab = '../climb/ablate_noblock_' + b.path.split('_').pop();
    fs.writeFileSync(ab, JSON.stringify(j, null, 2));
    const v2 = await scoreSaved(ab, h0, {});
    rows.push({ from: ab, ablation: 'no block, no stage 1', rise_mm: h0*1000, honest: v2.crit.honest,
      reward: v2.reward, x_mm:+mm(v2.scored.x), z_mm:+mm(v2.scored.z), feetOnTread: v2.scored.feetOnTread,
      peakZ_mm:+mm(v2.maxZ) });
    say(`  ABLATION no-block @ ${(h0*1000).toFixed(0)}mm  honest=${v2.crit.honest} reward=${v2.reward.toFixed(3)} x=${mm(v2.scored.x)} z=${mm(v2.scored.z)} feet=${v2.scored.feetOnTread}`);
  }
  R.phases.D = rows;
}

R.elapsed_s = (Date.now() - T0) / 1000;
fs.writeFileSync(OUT, JSON.stringify(R, null, 2));
say(`WROTE ${OUT} after ${el()}`);
