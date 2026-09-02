// family_b.mjs — ROUND 3, FAMILY B: LAND THE VAULT AT 90 mm.
//
// Round 2's beak-strut vault clears 40 and 60 mm and, at 90 mm, lifts the trunk
// on the beak and then FAILS TO LAND (head contact 0.726 of the episode,
// actuator saturation 0.233, feet never reach the tread). Round 2's corner-stem
// arm separately found that CONTACT ORDER inverts with height: foot-on-riser
// first wins at 180 mm, beak first at 120 mm, and wall-first is the worst
// opening at both.
//
// So this family searches the vault with three things round 2 did not have:
//
//  (1) A DISCRETE CONTACT-ORDER GENE. beak-first / foot-on-riser-first / both.
//      The stem is split into two keyframes S1 (first contact) and S2 (stem
//      fully loaded) and the gene decides which component S1 carries.
//
//  (2) THE SIDE WALL AS AN EXPLICIT THIRD CONTACT, arriving LATE. wall_n's
//      inner face is at y = 1.45 (sim/scene_physics.xml:184, half-thickness
//      0.05), i.e. 145 mm to the duck's left of STAIR_Y = 1.305. A spawn side
//      offset plus a common hip roll and a left-hip splay that switch on at the
//      VAULT frame — never at the stem — put the wall-side leg on the wall
//      after the stem is loaded. Round 2 measured wall-first as the worst
//      opening at both heights, so wall-first is not in the gene space.
//
//  (3) A SEPARATELY PARAMETRISED LANDING. An extra keyframe U between vault and
//      tuck lifts the TRAILING (wall-side) leg on its own at time
//      tVault + fTrail * tTuck — that is "when the second foot leaves the
//      floor" as a gene. tuck{Hip,Knee,Ank} is how far the hips fold before the
//      feet reach forward, and land{Hip,Knee,Ank} is where the feet aim on the
//      tread.
//
// SCORING. climb/robust.mjs only, which is rig3's loop (5/5 EXACT parity) over
// the 9-cell grid. Every candidate is WRITTEN TO JSON and the FILE is scored.
// The reported objective is the canonical one: mean of rig3's reward over the
// 9 cells + 4 per cell cleared under 'honest'. The SEARCH additionally shapes on
// sustained load transfer and feet-reaching-the-tread (below), because at 90 mm
// every candidate starts at k = 0 and the canonical objective alone is flat.
// Shaping never enters a reported k or objective.
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/family_b.mjs <seconds>
import fs from 'node:fs';
import { scoreCell, scoreRobust, saveIntent, HOME, STAIR_Y } from '../climb/robust.mjs';

const BUDGET_S = +(process.argv[2] || 2280);
const t0 = Date.now();
const el = () => (Date.now() - t0) / 1000;
const left = () => BUDGET_S - el();
const OUT = '../climb/';
const SCRATCH = '/tmp/claude-1000/-home-craigm26/43f07dce-e62f-464d-ae1f-2fe020620950/scratchpad';
fs.mkdirSync(SCRATCH, { recursive: true });

// ---------------------------------------------------------------- rng
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const SEED_BASE = 8919;
let RND = mulberry32(SEED_BASE);
function gauss() { let u = 0, v = 0; while (!u) u = RND(); while (!v) v = RND(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

// ---------------------------------------------------------------- the track
const J = { lhy: 0, lhr: 1, lhp: 2, lk: 3, la: 4, np: 5, hp: 6, hy: 7, hr: 8,
            rhy: 9, rhr: 10, rhp: 11, rk: 12, ra: 13 };
// The duck faces +x, so +y is its LEFT. wall_n is at +y: the LEFT leg is the
// wall-side (trailing) leg and the RIGHT leg is the riser-pressing (lead) leg.
/**
 * Asymmetric put(). With hipL===hipR etc. and splay 0 this is byte-identical to
 * climb/vault.mjs's symmetric put(), so the round-2 winner is inside this space.
 */
function put(q, L, R, roll, splay) {
  q[J.lhp] = HOME[J.lhp] + L.hip;  q[J.rhp] = HOME[J.rhp] - R.hip;
  q[J.lk] = HOME[J.lk] + L.knee;   q[J.rk] = HOME[J.rk] - R.knee;
  q[J.la] = HOME[J.la] + L.ank;    q[J.ra] = HOME[J.ra] - R.ank;
  q[J.lhr] = HOME[J.lhr] + roll + splay;
  q[J.rhr] = HOME[J.rhr] + roll;
  return q;
}
const leg = (hip, knee, ank) => ({ hip, knee, ank });
const strutOn = (q, p) => { q[J.np] = p.strutNeck; q[J.hp] = p.strutHead; return q; };

function trackOf(p) {
  const order = p.order | 0;                 // 0 beak-first, 1 riser-first, 2 both
  const crouchL = leg(p.crouchHip, p.crouchKnee, p.crouchAnk);
  const crouchR = leg(p.crouchHip, p.crouchKnee, p.crouchAnk);
  const riserR = leg(p.crouchHip + p.riserDHip, p.crouchKnee + p.riserDKnee, p.crouchAnk + p.riserDAnk);

  // S1: the FIRST contact only.
  let S1;
  if (order === 0)      S1 = strutOn(put(HOME.slice(), crouchL, crouchR, p.roll0, 0), p);   // beak alone
  else if (order === 1) S1 = put(HOME.slice(), crouchL, riserR, p.roll0, 0);                // riser foot alone
  else                  S1 = strutOn(put(HOME.slice(), crouchL, riserR, p.roll0, 0), p);    // both at once
  // S2: the stem fully loaded — beak planted AND the lead foot on the riser.
  const S2 = strutOn(put(HOME.slice(), crouchL, riserR, p.roll0, 0), p);
  // P: preload, stem held, no wall yet.
  const P = strutOn(put(HOME.slice(), leg(p.preHip, p.preKnee, p.preAnk),
                        leg(p.preHip + p.riserDHip, p.preKnee + p.riserDKnee, p.preAnk + p.riserDAnk),
                        p.roll0, 0), p);
  // V: the vault. Hips and knees extend against the stem. THE WALL ARRIVES HERE.
  const V = strutOn(put(HOME.slice(), leg(p.vaultHip, p.vaultKnee, p.vaultAnk),
                        leg(p.vaultHip, p.vaultKnee, p.vaultAnk), p.wallRoll, p.wallSplay), p);
  // U: the TRAILING (wall-side, left) leg unweights on its own. This keyframe's
  //    time is the "second foot leaves the floor" gene.
  const U = strutOn(put(HOME.slice(), leg(p.vaultHip + p.trailDHip, p.vaultKnee + p.trailDKnee, p.vaultAnk),
                        leg(p.vaultHip, p.vaultKnee, p.vaultAnk), p.wallRoll, p.wallSplay), p);
  // T: the tuck — how far the hips fold before the feet reach forward.
  const T = strutOn(put(HOME.slice(), leg(p.tuckHip, p.tuckKnee, p.tuckAnk),
                        leg(p.tuckHip, p.tuckKnee, p.tuckAnk), p.wallRoll, p.wallSplay), p);
  // L: the landing — where the feet aim on the tread, strut released.
  const L = put(HOME.slice(), leg(p.landHip, p.landKnee, p.landAnk),
                leg(p.landHip, p.landKnee, p.landAnk), p.landRoll, 0);
  L[J.np] = p.landNeck; L[J.hp] = p.landHead;

  const t1 = p.tS1, t2 = t1 + p.tS2, t3 = t2 + p.tPre, t4 = t3 + p.tVault;
  const tU = t4 + Math.max(0.02, p.fTrail * p.tTuck);
  const t5 = t4 + p.tTuck, t6 = t5 + p.tLand;
  const kf = [{ t: t1, pose: S1 }, { t: t2, pose: S2 }, { t: t3, pose: P }, { t: t4, pose: V }];
  if (tU < t5 - 0.02) kf.push({ t: tU, pose: U });
  kf.push({ t: t5, pose: T }, { t: t6, pose: L }, { t: t6 + 0.7, pose: HOME.slice() });
  return kf;
}

// ---------------------------------------------------------------- genes
const BOUNDS = {
  gap: [0.005, 0.09], side: [-0.02, 0.10], approach: [0.0, 0.35], blend: [0.8, 2.4],
  tS1: [0.25, 0.80], tS2: [0.06, 0.30], tPre: [0.08, 0.35], tVault: [0.10, 0.45],
  tTuck: [0.10, 0.45], tLand: [0.15, 0.60], fTrail: [0.0, 0.95],
  strutNeck: [-1.55, 1.04], strutHead: [-1.55, 1.55],
  crouchHip: [-1.2, 1.2], crouchKnee: [-1.2, 1.2], crouchAnk: [-1.2, 1.2],
  riserDHip: [-1.2, 1.2], riserDKnee: [-1.2, 1.2], riserDAnk: [-1.0, 1.0],
  preHip: [-1.2, 1.2], preKnee: [-1.2, 1.2], preAnk: [-1.2, 1.2],
  vaultHip: [-1.4, 1.4], vaultKnee: [-1.4, 1.4], vaultAnk: [-1.4, 1.4],
  trailDHip: [-1.2, 1.2], trailDKnee: [-1.2, 1.2],
  tuckHip: [-1.4, 1.4], tuckKnee: [-1.4, 1.4], tuckAnk: [-1.4, 1.4],
  landHip: [-1.4, 1.4], landKnee: [-1.4, 1.4], landAnk: [-1.4, 1.4],
  landNeck: [-1.0, 1.04], landHead: [-1.0, 1.55],
  roll0: [-0.30, 0.30], wallRoll: [-0.30, 0.38], wallSplay: [-0.10, 0.38], landRoll: [-0.30, 0.30],
};
const KEYS = Object.keys(BOUNDS);
const clampB = (k, v) => Math.min(BOUNDS[k][1], Math.max(BOUNDS[k][0], v));
const randP = () => {
  const p = Object.fromEntries(KEYS.map(k => [k, BOUNDS[k][0] + RND() * (BOUNDS[k][1] - BOUNDS[k][0])]));
  p.order = Math.floor(RND() * 3);
  return p;
};

/**
 * The round-2 60 mm winner, lifted into this gene space. Every shared name is
 * its measured value; the new genes start neutral (no wall, trailing leg tucks
 * with the rest). `order` starts at 2 (both) because that is what round 2's
 * five-keyframe track actually did.
 */
function liftR2(file, order) {
  const R2 = JSON.parse(fs.readFileSync(OUT + file, 'utf8')).params;
  const p = {
    gap: R2.gap, side: R2.side, approach: R2.approach, blend: R2.blend,
    tS1: R2.tReach, tS2: 0.12, tPre: R2.tPre, tVault: R2.tVault, tTuck: R2.tTuck, tLand: R2.tLand,
    fTrail: 0.5,
    strutNeck: R2.strutNeck, strutHead: R2.strutHead,
    crouchHip: R2.crouchHip, crouchKnee: R2.crouchKnee, crouchAnk: R2.crouchAnk,
    riserDHip: -0.35, riserDKnee: 0.35, riserDAnk: 0.0,
    preHip: R2.preHip, preKnee: R2.preKnee, preAnk: R2.preAnk,
    vaultHip: R2.vaultHip, vaultKnee: R2.vaultKnee, vaultAnk: R2.vaultAnk,
    trailDHip: -0.30, trailDKnee: 0.30,
    tuckHip: R2.tuckHip, tuckKnee: R2.tuckKnee, tuckAnk: R2.tuckAnk,
    landHip: R2.landHip, landKnee: R2.landKnee, landAnk: R2.landAnk,
    landNeck: R2.landNeck, landHead: R2.landHead,
    roll0: R2.roll, wallRoll: R2.roll, wallSplay: 0.0, landRoll: 0.0,
  };
  for (const k of KEYS) p[k] = clampB(k, p[k]);
  p.order = order;
  return p;
}
// `order` starts at 2 (both) because that is what round 2's five-keyframe
// track actually did — one frame carrying beak and legs together.
const PRIOR = liftR2('best_r2_vault_60mm.json', 2);
const SEEDS = [PRIOR, liftR2('best_r2_vault_40mm.json', 2), liftR2('best_r2_vault_90mm.json', 2),
               { ...PRIOR, order: 0 }, { ...PRIOR, order: 1 }];

// ---------------------------------------------------------------- fitness
const c01 = v => Math.max(0, Math.min(1, v));
/**
 * The SEARCH's shaping, on top of rig3's reward. It is never reported as an
 * objective. It pays for exactly the two things the brief names: sustained load
 * transfer (trunk z gain while the head AND a foot bear) and feet reaching the
 * tread. Head contact on its own is deliberately NOT paid — round 2's 90 mm
 * failure had head contact 0.726 and no landing.
 */
function shape(r) {
  const gain = r.maxZ - r.z0Settle;
  return 3.0 * c01(gain / 0.09)                       // the trunk actually rose
       + 3.0 * c01(r.feetOnTreadMax / 2)              // a foot reached the tread
       + 2.5 * c01(r.sustainFrac / 0.08)              // SUSTAINED load transfer
       + 1.0 * c01(r.bothFrac / 0.20)                 // there was a two-contact phase
       + 1.0 * c01(r.liftIntegral / 1.5)
       + 0.5 * c01(r.feetHighMax / 2)
       // graded landing gradient: how close a foot, and then the WORSE foot,
       // ever came to a landing spot on the tread. Without this the landing is
       // all-or-nothing and the search is flat everywhere short of a clear.
       + 2.0 * c01((0.32 - r.footNear) / 0.28)
       + 3.5 * c01((0.32 - r.bothNear) / 0.28);
}

let evals = 0, fullGrids = 0, bestScreen = -1e9;
const r5 = v => +v.toFixed(5);
function intentOf(p, rise, note) {
  return { name: `b_vault_land_${Math.round(rise * 1000)}mm`, family: 'B land-the-vault',
           order: ['beak-first', 'riser-foot-first', 'both'][p.order | 0],
           keyframes: trackOf(p).map(f => ({ t: +f.t.toFixed(4), pose: f.pose.map(r5) })),
           blend: +p.blend.toFixed(4), gap: +p.gap.toFixed(4), side: +p.side.toFixed(4),
           approach: +p.approach.toFixed(4), isolate: true, stepCount: 4, params: p, note };
}

const TMP = SCRATCH + '/fb_cand.json';
/**
 * SCREEN: one cell (nominal plant, target rise). Cheap, 0.37 s.
 *
 * The screen score is deliberately ON THE SAME SCALE as the full-grid fitness:
 * for a candidate that clears nothing, grid objective = mean rig3 reward and
 * grid fitness = that + mean shaping, which is what one cell estimates. Run 1
 * of this family scaled ungraded candidates by 0.30 instead, and the CEM elite
 * filled with ungraded junk (eliteFit=[7.5 7.0 2.4 2.2 2.2 2.2],
 * climb/family_b_run1.log) — the mean was pulled off the promoted candidates
 * every generation and the search stalled with 3 improvements in 1291 s.
 */
async function screen(p, rise) {
  saveIntent(intentOf(p, rise, 'candidate'), TMP);
  const c0 = await scoreCell(TMP, { rise, isolate: true });
  evals++;
  const s0 = c0.reward + shape(c0);
  if (s0 > bestScreen) bestScreen = s0;
  return { screen: s0, c0 };
}
/** GRADE: the full 9 cells, from the saved file. */
async function grade(p, rise) {
  saveIntent(intentOf(p, rise, 'candidate'), TMP);
  const g = await scoreRobust(TMP, { rise, isolate: true });
  fullGrids++;
  const shaped = g.cells.reduce((a, c) => a + shape(c), 0) / g.cells.length;
  return { fit: g.objective + shaped, grid: g };
}

// ---------------------------------------------------------------- CEM
const SIG0 = Object.fromEntries(KEYS.map(k => [k, 0.30 * (BOUNDS[k][1] - BOUNDS[k][0])]));
const results = { family: 'B land-the-vault', seedBase: SEED_BASE, generated: new Date().toISOString(),
                  scorer: 'climb/robust.mjs scoreRobust (9 cells) — rig3 parity 5/5 EXACT',
                  objective: 'mean(rig3 reward over 9 cells) + 4 * (cells cleared under honest)',
                  searchShaping: 'load transfer + feet-on-tread, search only, never reported',
                  baselines: [], runs: [], improvements: [] };

async function baseline(file, rise, label) {
  const g = await scoreRobust(OUT + file, { rise, isolate: true });
  const row = { label, file, rise_mm: Math.round(rise * 1000), k: g.k,
                meanReward: +g.meanReward.toFixed(3), objective: +g.objective.toFixed(3),
                peakGain_mm: +(g.agg.meanPeakGain * 1000).toFixed(1),
                headFrac: +g.agg.headFrac.toFixed(3), riserFrac: +g.agg.riserFrac.toFixed(3),
                wallFrac: +g.agg.wallFrac.toFixed(3), feetOnTreadMax: g.agg.feetOnTreadMax,
                meanFeetOnTreadFinal: +g.agg.meanFeetOnTreadFinal.toFixed(2),
                bothNear_mm: +g.agg.bothNear_mm.toFixed(1), footNear_mm: +g.agg.footNear_mm.toFixed(1),
                peakZ_mm: +(g.agg.meanMaxZ * 1000).toFixed(1), verdicts: g.verdicts };
  results.baselines.push(row);
  console.log(`[${el().toFixed(0)}s] BASELINE ${label} @${row.rise_mm}mm: k=${g.k}/9 obj=${row.objective} peakGain=${row.peakGain_mm}mm head=${row.headFrac} riser=${row.riserFrac} wall=${row.wallFrac} fotMax=${row.feetOnTreadMax}`);
  return g;
}

/**
 * One CEM run at one rise. `seedPop` is an optional list of parameter sets to
 * inject into generation 0 (the stepping stone at 80 mm feeds 90 mm).
 */
async function run(rise, deadlineS, tag, seedPop = []) {
  const rm = Math.round(rise * 1000);
  console.log(`\n[${el().toFixed(0)}s] === CEM at ${rm} mm, deadline ${deadlineS.toFixed(0)}s ===`);
  let mu = { ...PRIOR }, sg = { ...SIG0 };
  let pOrder = [1 / 3, 1 / 3, 1 / 3];
  const POP = 24, ELITE = 6, PROMOTE = 4;
  let best = null, gen = 0, stale = 0, restarts = 0;
  const seen = [];
  const rec = { rise_mm: rm, tag, generations: [], evalsStart: evals };

  while (el() < deadlineS) {
    const gen0 = gen === 0 ? seedPop.concat(SEEDS) : [];
    const pop = [];
    for (let i = 0; i < POP; i++) {
      if (i < gen0.length) { pop.push({ ...gen0[i] }); continue; }
      if (RND() < 0.20) { pop.push(randP()); continue; }              // keep exploring
      const p = Object.fromEntries(KEYS.map(k => [k, clampB(k, mu[k] + sg[k] * gauss())]));
      const u = RND(); p.order = u < pOrder[0] ? 0 : (u < pOrder[0] + pOrder[1] ? 1 : 2);
      pop.push(p);
    }
    // PASS 1 — screen the whole population on one cell.
    const scored = [];
    for (const p of pop) {
      if (el() > deadlineS) break;
      const s = await screen(p, rise);
      scored.push({ p, ...s, fit: s.screen, grid: null });
      seen.push(s.screen);
    }
    if (!scored.length) break;
    // PASS 2 — grade the top PROMOTE of them on the full 9 cells. The elite is
    // then always anchored on grid-graded candidates.
    scored.sort((a, b) => b.screen - a.screen);
    for (const c of scored.slice(0, PROMOTE)) {
      if (el() > deadlineS) break;
      const f = await grade(c.p, rise);
      c.fit = f.fit; c.grid = f.grid;
    }
    for (const c of scored) {
      const p = c.p, f = c;
      if (f.grid && (!best || f.fit > best.fit)) {
        best = { p, ...f }; stale = -1;
        const g = f.grid;
        const line = { t_s: +el().toFixed(0), gen, k: g.k, objective: +g.objective.toFixed(3),
          meanReward: +g.meanReward.toFixed(3), searchFit: +f.fit.toFixed(3), order: p.order,
          peakZ_mm: +(g.agg.meanMaxZ * 1000).toFixed(1), peakGain_mm: +(g.agg.meanPeakGain * 1000).toFixed(1),
          headFrac: +g.agg.headFrac.toFixed(3), riserFrac: +g.agg.riserFrac.toFixed(3),
          wallFrac: +g.agg.wallFrac.toFixed(3), wallBearFrac: +g.agg.wallBearFrac.toFixed(3),
          sustainFrac: +g.agg.sustainFrac.toFixed(3), feetOnTreadMax: g.agg.feetOnTreadMax,
          meanFeetOnTreadFinal: +g.agg.meanFeetOnTreadFinal.toFixed(2),
          bothNear_mm: +g.agg.bothNear_mm.toFixed(1), footNear_mm: +g.agg.footNear_mm.toFixed(1),
          meanAbove_mm: +g.agg.meanAbove_mm.toFixed(1), upFinal: g.agg.upFinal };
        results.improvements.push({ rise_mm: rm, ...line });
        console.log(`[${el().toFixed(0)}s] ${rm}mm gen${gen} IMPROVED cleared ${g.k} of 9  obj=${line.objective} fit=${line.searchFit} order=${['beak','riser','both'][p.order]} peakZ=${line.peakZ_mm}mm gain=${line.peakGain_mm}mm head=${line.headFrac} riser=${line.riserFrac} wall=${line.wallFrac} sustain=${line.sustainFrac} fotMax=${line.feetOnTreadMax} fotFinal=${line.meanFeetOnTreadFinal} bothNear=${line.bothNear_mm}mm up=${line.upFinal}/9`);
        saveIntent(intentOf(p, rise, `family B best; cleared ${g.k} of 9; objective ${g.objective.toFixed(3)}`),
          `${OUT}best_r3_landvault_${rm}mm.json`);
      }
    }
    scored.sort((a, b) => b.fit - a.fit);
    const elite = scored.slice(0, Math.min(ELITE, scored.length));
    const nmu = {}, nsg = {};
    for (const k of KEYS) {
      const v = elite.map(e => e.p[k]);
      const m = v.reduce((a, b) => a + b, 0) / v.length;
      const s = Math.sqrt(v.reduce((a, b) => a + (b - m) * (b - m), 0) / v.length);
      nmu[k] = m;
      nsg[k] = Math.max(0.05 * (BOUNDS[k][1] - BOUNDS[k][0]), 0.6 * sg[k] + 0.4 * (s + 0.02 * (BOUNDS[k][1] - BOUNDS[k][0])));
    }
    mu = nmu; sg = nsg;
    // stagnation restart: three generations with no new best and the search
    // re-inflates and jumps halfway back toward the round-2 prior.
    stale++;
    if (stale >= 3) {
      for (const k of KEYS) { mu[k] = clampB(k, 0.5 * mu[k] + 0.5 * PRIOR[k]); sg[k] = 0.55 * SIG0[k]; }
      stale = 0; restarts++;
      console.log(`[${el().toFixed(0)}s] ${rm}mm RESTART #${restarts} (3 stale generations)`);
    }
    const cnt = [0, 0, 0]; for (const e of elite) cnt[e.p.order | 0]++;
    pOrder = cnt.map(c => 0.10 + 0.70 * (c / elite.length));
    const z = pOrder.reduce((a, b) => a + b, 0); pOrder = pOrder.map(v => v / z);
    rec.generations.push({ gen, t_s: +el().toFixed(0), evals, fullGrids,
      eliteFit: elite.map(e => +e.fit.toFixed(2)), pOrder: pOrder.map(v => +v.toFixed(2)),
      bestK: best ? best.grid.k : 0 });
    console.log(`[${el().toFixed(0)}s] ${rm}mm gen${gen} done: evals=${evals} grids=${fullGrids} bestScreen=${bestScreen.toFixed(2)} eliteFit=[${elite.map(e => e.fit.toFixed(1)).join(' ')}] pOrder=[${pOrder.map(v => v.toFixed(2)).join(' ')}] bestK=${best ? best.grid.k : 0}`);
    gen++;
  }
  rec.gens = gen; rec.evals = evals - rec.evalsStart; rec.restarts = restarts;
  if (best) {
    rec.best = { k: best.grid.k, objective: +best.grid.objective.toFixed(3),
      meanReward: +best.grid.meanReward.toFixed(3), order: ['beak-first', 'riser-foot-first', 'both'][best.p.order | 0],
      agg: Object.fromEntries(Object.entries(best.grid.agg).map(([k, v]) => [k, typeof v === 'number' ? +v.toFixed(4) : v])),
      verdicts: best.grid.verdicts, file: `climb/best_r3_landvault_${rm}mm.json` };
  }
  results.runs.push(rec);
  return best;
}

// ---------------------------------------------------------------- main
console.log(`family B — land the vault. budget ${BUDGET_S}s, seed base ${SEED_BASE}`);
await baseline('best_r2_vault_60mm.json', 0.080, 'r2 vault 60mm move, at 80mm');
await baseline('best_r2_vault_90mm.json', 0.090, 'r2 vault 90mm move, at 90mm');
await baseline('ctrl_on_tread_90mm.json', 0.090, 'placed-on-tread control');
await baseline('ctrl_do_nothing.json', 0.090, 'do-nothing control');

// Run 1 of this family (climb/family_b_run1.log) stalled on a ranking bug in
// the promotion gate; its two saved bests are carried into run 2's generation 0
// so none of that compute is thrown away.
const CARRY = [];
for (const f of ['run1_80.json', 'run1_90.json']) {
  const q = SCRATCH + '/' + f;
  if (fs.existsSync(q)) { const j = JSON.parse(fs.readFileSync(q, 'utf8')); if (j.params) CARRY.push(j.params); }
}
console.log(`carrying ${CARRY.length} run-1 bests into generation 0`);

const T_END = BUDGET_S;
const T_80 = el() + (T_END - el()) * 0.30;
const best80 = await run(0.080, T_80, '80mm stepping stone', CARRY);
// carry the 80 mm elite forward: its own params plus the prior, then let 90 mm
// re-search from there.
const seed90 = CARRY.slice();
if (best80) seed90.unshift(best80.p);
const best90 = await run(0.090, T_END - 30, '90mm target', seed90);

// cross-scores: what does each best do at the other rise?
results.cross = [];
for (const [file, rise, label] of [['best_r3_landvault_80mm.json', 0.090, '80mm best scored at 90mm'],
                                   ['best_r3_landvault_90mm.json', 0.080, '90mm best scored at 80mm']]) {
  if (!fs.existsSync(OUT + file)) continue;
  const g = await scoreRobust(OUT + file, { rise, isolate: true });
  results.cross.push({ label, file, rise_mm: Math.round(rise * 1000), k: g.k,
    objective: +g.objective.toFixed(3), meanReward: +g.meanReward.toFixed(3) });
  console.log(`[${el().toFixed(0)}s] CROSS ${label}: k=${g.k}/9 obj=${g.objective.toFixed(3)}`);
}

results.totals = { evals, fullGrids, wall_s: +el().toFixed(0) };
fs.writeFileSync(OUT + 'r3_landvault-results.json', JSON.stringify(results, null, 2));
console.log(`\n[${el().toFixed(0)}s] wrote ${OUT}r3_landvault-results.json  evals=${evals} fullGrids=${fullGrids}`);
if (best80) console.log(`80mm best: cleared ${best80.grid.k} of 9, objective ${best80.grid.objective.toFixed(3)}`);
if (best90) console.log(`90mm best: cleared ${best90.grid.k} of 9, objective ${best90.grid.objective.toFixed(3)}`);
