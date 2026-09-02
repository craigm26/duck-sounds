// FAMILY A (round 3) — harden the beak-strut vault at 40 and 60 mm, then climb
// the ladder (50 -> 70 -> 80 mm).
//
// Warm-started CEM (cross-entropy method) over the family-C parametrisation
// (climb/vault.mjs trackOf/BOUNDS, copied verbatim below — vault.mjs cannot be
// imported without standing up a second MjModel and a second ONNX session).
// Population 24, elite 5 (0.21), Gaussian over every keyframe offset, the
// blend, gap, side, approach and the five segment durations, with a hard floor
// on the per-key variance so the distribution cannot collapse onto the seed.
//
// THE OBJECTIVE IS climb/robust.mjs scoreRobust() — the shared 9-cell grid,
// rise {h-10, h, h+10} mm x plant {(drop .120, friction x1.0), (drop .130,
// x0.7), (drop .125, x1.3)}, objective = mean(rig3 reward) + CLEAR_BONUS * k.
// Nothing here re-implements a scorer.
//
// SCREENING. A full 9-cell evaluation costs 3.4 s; one cell costs 0.42 s. Every
// candidate is first run on cell 0 (the NOMINAL cell, which robust.mjs proves
// is rig3.scoreSaved exactly). If its reward is below SCREEN_CUT the other
// eight cells are skipped. Screened candidates are then ranked among THEMSELVES
// by tier2() (see below) and only ever fill the elite set BELOW every fully
// evaluated candidate, so the real objective is always authoritative and the
// two scores are never compared with each other. No reported k of 9 ever comes
// from a screened evaluation: `best` only accepts a full 9-cell record.
//
// Every candidate is WRITTEN TO JSON and re-read before it is scored.
// Seed base 1000, mulberry32.
//
// Run from sim/:  node ../climb/famA_vault.mjs [totalSeconds]
import fs from 'node:fs';
import { scoreRobust, scoreCell, CLEAR_BONUS, PLANTS, DHS, HOME } from './robust.mjs';

const P = '../climb/';
const SCRATCH = '/tmp/claude-1000/-home-craigm26/43f07dce-e62f-464d-ae1f-2fe020620950/scratchpad';
fs.mkdirSync(SCRATCH, { recursive: true });
const T0 = Date.now();
const el = () => ((Date.now() - T0) / 1000);
const log = s => { console.log(`[${el().toFixed(0).padStart(4)}s] ${s}`); };

// ------------------------------------------------------------------ track
// climb/vault.mjs trackOf() and BOUNDS, verbatim.
const J = { lhy: 0, lhr: 1, lhp: 2, lk: 3, la: 4, np: 5, hp: 6, hy: 7, hr: 8,
            rhy: 9, rhr: 10, rhp: 11, rk: 12, ra: 13 };
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
const RANGE = Object.fromEntries(KEYS.map(k => [k, BOUNDS[k][1] - BOUNDS[k][0]]));

// ------------------------------------------------------------------ rng
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const SEED_BASE = 1000;
let RND = mulberry32(SEED_BASE);
function gauss() { let u = 0, v = 0; while (!u) u = RND(); while (!v) v = RND(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
const clampB = (k, v) => Math.min(BOUNDS[k][1], Math.max(BOUNDS[k][0], v));
const randP = () => Object.fromEntries(KEYS.map(k => [k, BOUNDS[k][0] + RND() * RANGE[k]]));
const sampleN = (mu, sg) => Object.fromEntries(KEYS.map(k => [k, clampB(k, mu[k] + sg[k] * gauss())]));

// ------------------------------------------------------------------ eval
const r5 = v => +v.toFixed(5);
function intentOf(p, rise, note) {
  return { name: `beak_strut_vault_r3_${Math.round(rise * 1000)}mm`, family: 'A beak-strut vault (round 3)',
           keyframes: trackOf(p).map(f => ({ t: +f.t.toFixed(4), pose: f.pose.map(r5) })),
           blend: +p.blend.toFixed(4), gap: +p.gap.toFixed(4), side: +p.side.toFixed(4),
           approach: +p.approach.toFixed(4), isolate: true, stepCount: 4, params: p, note };
}
// SCREEN_CUT is not a magic number: rig3's reward is
//   3*x_term(<=3) + 2*feetOnTread + 4*height_term(<=4) + up(<=1),
// so the most a duck that has NO foot resting on the tread can score is 8.
// A cut at 9.5 therefore means exactly "at least one foot is resting on the
// tread at the end of the nominal episode", and nothing else gets the other
// eight cells spent on it.
const SCREEN_CUT = 9.5;
const c01 = v => Math.max(0, Math.min(1, v));
/**
 * TIER-2 RANK, for candidates the screen stopped. It is NOT an objective and is
 * never compared against one (see the two-tier elite selection below): it only
 * orders the candidates we could not afford to score on all 9 cells.
 *
 * It exists because rig3's reward alone ranks A DUCK THAT DOES NOTHING (stands
 * on the floor: x-term ~2, height 4, up 1 = ~7) ABOVE a vault that plants the
 * beak, gets the trunk over the riser and misses the landing. Ranking a
 * screened population by reward alone therefore walks the CEM mean out of the
 * vault basin and into standing still. These four terms are the vault:
 * a foot ever on the tread, the beak ever on the block, the trunk ever past
 * the riser line, on top of the reward itself.
 */
function tier2(c0) {
  return c0.reward
       + 3 * c0.feetOnTreadMax
       + 2 * c01(c0.headFrac / 0.30)
       + 4 * c01((c0.maxX - 0.12) / 0.15);
}
let EVALS = 0, FULL = 0;
async function evalP(p, rise, { screen = true } = {}) {
  const path = `${SCRATCH}/famA_cand.json`;
  fs.writeFileSync(path, JSON.stringify(intentOf(p, rise, 'candidate'), null, 2));
  EVALS++;
  if (screen) {
    const c0 = await scoreCell(path, { rise, dh: 0, drop: PLANTS[0].drop, fmul: PLANTS[0].fmul, isolate: true });
    if (c0.reward < SCREEN_CUT) {
      return { screened: true, k: c0.crit.honest ? 1 : 0, objective: c0.reward / 9,
               rank2: tier2(c0), screenReward: c0.reward, feetMax: c0.feetOnTreadMax,
               headFrac: c0.headFrac, maxX: c0.maxX, meanReward: c0.reward / 9, verdicts: null };
    }
  }
  const r = await scoreRobust(path, { rise, isolate: true });
  FULL++;
  return { screened: false, k: r.k, objective: r.objective, meanReward: r.meanReward,
           verdicts: r.verdicts, agg: r.agg };
}

/** Which axis kills the failing cells. */
function blame(verdicts, rise) {
  if (!verdicts) return 'screened';
  const byRise = new Map(), byPlant = new Map();
  for (const v of verdicts) {
    const dh = v.rise_mm - Math.round(rise * 1000);
    const rk = `${dh >= 0 ? '+' : ''}${dh}`;
    const pk = `d${v.drop.toFixed(3)}/f${v.fmul.toFixed(1)}`;
    (byRise.get(rk) || byRise.set(rk, []).get(rk)).push(v.honest);
    (byPlant.get(pk) || byPlant.set(pk, []).get(pk)).push(v.honest);
  }
  const fmt = m => [...m.entries()].map(([k, v]) => `${k}:${v.filter(Boolean).length}/3`).join(' ');
  return `rise[${fmt(byRise)}] plant[${fmt(byPlant)}]`;
}

// ------------------------------------------------------------------ CEM
const POP = 24, ELITE = 5;
// Fractions of each key's bound range. This is a HARDENING search warm-started
// on a move that already works once, so the proposal is deliberately tight; the
// floor stops the elite covariance collapsing onto the seed, and one uniformly
// drawn explorer per generation keeps a path out of the basin.
const SIG0 = 0.045, SIGMIN = 0.018;
// rank weights 5,4,3,2,1 over the elite set: the incumbent is always rank 0, so
// a single surviving full evaluation still owns a third of the new mean.
const RANKW = [5, 4, 3, 2, 1];

function saveBest(best, rise) {
  const mmr = Math.round(rise * 1000);
  const obj = intentOf(best.p, rise,
    `round-3 family A beak-strut vault at ${mmr} mm; cleared ${best.r.k} of 9; objective ${best.r.objective.toFixed(4)}; meanReward ${best.r.meanReward.toFixed(4)}`);
  obj.robust = { k: best.r.k, of: 9, objective: +best.r.objective.toFixed(4),
                 meanReward: +best.r.meanReward.toFixed(4), clearBonus: CLEAR_BONUS,
                 verdicts: best.r.verdicts, agg: best.r.agg };
  fs.writeFileSync(`${P}best_r3_vault_${mmr}mm.json`, JSON.stringify(obj, null, 2));
}

async function cem(rise, mu0, deadline, tag, sigScale = 1) {
  let mu = { ...mu0 };
  let sg = Object.fromEntries(KEYS.map(k => [k, SIG0 * sigScale * RANGE[k]]));
  const sgFloor = Object.fromEntries(KEYS.map(k => [k, SIGMIN * RANGE[k]]));
  let gen = 0;
  const history = [];
  const seed = await evalP(mu, rise, { screen: false });
  let best = { p: { ...mu }, r: seed };
  log(`${tag} warm-start: k=${seed.k}/9 obj=${seed.objective.toFixed(3)} meanR=${seed.meanReward.toFixed(3)} | ${blame(seed.verdicts, rise)}`);
  history.push({ gen: 0, kind: 'warm-start', k: seed.k, objective: +seed.objective.toFixed(4),
                 meanReward: +seed.meanReward.toFixed(4), verdicts: seed.verdicts });
  saveBest(best, rise);
  if (best.r.k >= 7) { log(`${tag} warm start already clears ${best.r.k}/9`); return { best, gen, history, hit7: true }; }
  while (Date.now() < deadline) {
    gen++;
    const scored = [];
    for (let i = 0; i < POP; i++) {
      if (Date.now() >= deadline) break;
      const p = (i === POP - 1) ? randP() : sampleN(mu, sg);     // one uniform explorer per generation
      const r = await evalP(p, rise);
      scored.push({ p, r });
      if (!r.screened && r.objective > best.r.objective) {
        best = { p, r };
        log(`${tag} gen${gen} IMPROVE k=${r.k}/9 obj=${r.objective.toFixed(3)} meanR=${r.meanReward.toFixed(3)} | ${blame(r.verdicts, rise)}`);
        history.push({ gen, kind: 'improve', k: r.k, objective: +r.objective.toFixed(4),
                       meanReward: +r.meanReward.toFixed(4), verdicts: r.verdicts });
        saveBest(best, rise);
        if (r.k >= 7) { log(`${tag} k>=7 at gen${gen} — stage complete`); return { best, gen, history, hit7: true }; }
      }
    }
    if (!scored.length) break;
    // TWO-TIER ELITE SELECTION. Fully-evaluated candidates are ranked by the
    // real 9-cell objective and always come first; screened candidates fill the
    // rest of the elite set, ordered by tier2(). The two scores are never
    // compared with each other.
    const fullE = scored.filter(s => !s.r.screened).sort((a, b) => b.r.objective - a.r.objective);
    const scrE = scored.filter(s => s.r.screened).sort((a, b) => b.r.rank2 - a.r.rank2);
    let eli = [...fullE, ...scrE].slice(0, ELITE);
    // the incumbent is always rank 0 so the mean cannot drift off the best clear
    eli = [best, ...eli.filter(e => e.r !== best.r)].slice(0, ELITE);
    const wsum = eli.reduce((a, _, i) => a + RANKW[i], 0);
    const nmu = {}, nsg = {};
    for (const k of KEYS) {
      let m = 0; for (let i = 0; i < eli.length; i++) m += RANKW[i] * eli[i].p[k]; m /= wsum;
      let v = 0; for (let i = 0; i < eli.length; i++) v += RANKW[i] * (eli[i].p[k] - m) ** 2; v = Math.sqrt(v / wsum);
      nmu[k] = m; nsg[k] = Math.max(v, sgFloor[k]);
    }
    mu = nmu; sg = nsg;
    log(`${tag} gen${gen}: full=${fullE.length} eliteObj ${eli.map(e => e.r.screened ? `s${e.r.rank2.toFixed(1)}` : e.r.objective.toFixed(2)).join(' ')} | best k=${best.r.k}/9 obj=${best.r.objective.toFixed(3)} | evals=${EVALS} fullEvals=${FULL}`);
  }
  return { best, gen, history, hit7: best.r.k >= 7 };
}

// ------------------------------------------------------------------ main
const TOTAL = +(process.argv[2] || 2400);
log(`family A start. clear bonus ${CLEAR_BONUS}, pop ${POP}, elite ${ELITE}, sig0 ${SIG0}, sigMin ${SIGMIN}, seed ${SEED_BASE}, budget ${TOTAL}s`);
log(`grid: rise offsets ${DHS.map(d => d * 1000).join('/')} mm x plants ${PLANTS.map(p => `${p.drop}/x${p.fmul}`).join(' ')}`);

const RESULTS = { generated: new Date().toISOString(), family: 'A beak-strut vault (round 3)',
                  scorer: 'climb/robust.mjs scoreRobust (9 cells)', seedBase: SEED_BASE,
                  pop: POP, elite: ELITE, sig0: SIG0, sigMin: SIGMIN, screenCut: SCREEN_CUT,
                  clearBonus: CLEAR_BONUS, plants: PLANTS, dhs: DHS, budget_s: TOTAL,
                  baseline: [], stages: [] };
const flush = () => fs.writeFileSync(`${P}r3_vault-results.json`, JSON.stringify(RESULTS, null, 2));

for (const [f, h] of [['best_r2_vault_40mm.json', 0.040], ['best_r2_vault_60mm.json', 0.060]]) {
  const r = await scoreRobust(P + f, { rise: h, isolate: true });
  log(`baseline ${f}: k=${r.k}/9 obj=${r.objective.toFixed(3)} meanR=${r.meanReward.toFixed(3)} | ${blame(r.verdicts, h)}`);
  RESULTS.baseline.push({ file: f, rise_mm: h * 1000, k: r.k, objective: +r.objective.toFixed(4),
                          meanReward: +r.meanReward.toFixed(4), verdicts: r.verdicts });
}
flush();

const p40 = JSON.parse(fs.readFileSync(P + 'best_r2_vault_40mm.json', 'utf8')).params;
const p60 = JSON.parse(fs.readFileSync(P + 'best_r2_vault_60mm.json', 'utf8')).params;

const rec = (rise, from, s) => {
  RESULTS.stages.push({ rise_mm: Math.round(rise * 1000), from, gens: s.gen, k: s.best.r.k,
    objective: +s.best.r.objective.toFixed(4), meanReward: +s.best.r.meanReward.toFixed(4),
    hit7: s.hit7, blame: blame(s.best.r.verdicts, rise), verdicts: s.best.r.verdicts,
    agg: s.best.r.agg, history: s.history, params: s.best.p,
    file: `climb/best_r3_vault_${Math.round(rise * 1000)}mm.json` });
  flush();
};

const rem0 = TOTAL - el();
const s40 = await cem(0.040, p40, T0 + (el() + rem0 * 0.24) * 1000, '40mm');
rec(0.040, 'best_r2_vault_40mm.json', s40);

const rem1 = TOTAL - el();
const s60 = await cem(0.060, p60, T0 + (el() + rem1 * 0.32) * 1000, '60mm');
rec(0.060, 'best_r2_vault_60mm.json', s60);

// ---- the ladder
let s70 = null;
const ladder = [
  { rise: 0.050, from: () => s40.best.p, tag: '50mm', src: 'best 40 mm vector' },
  { rise: 0.070, from: () => s60.best.p, tag: '70mm', src: 'best 60 mm vector' },
  { rise: 0.080, from: () => (s70 ? s70.best.p : s60.best.p), tag: '80mm', src: 'best 70 mm vector' },
];
for (let i = 0; i < ladder.length; i++) {
  const st = ladder[i];
  const left = TOTAL - el();
  if (left < 40) { log(`${st.tag} skipped — ${left.toFixed(0)}s left`); continue; }
  const share = left / (ladder.length - i);
  const s = await cem(st.rise, st.from(), T0 + (el() + share) * 1000, st.tag, 0.8);
  if (st.tag === '70mm') s70 = s;
  rec(st.rise, st.src, s);
}

RESULTS.evals = EVALS; RESULTS.fullEvals = FULL; RESULTS.wall_s = +el().toFixed(1);
flush();
log(`DONE. evals=${EVALS} (full 9-cell ${FULL}) wall=${el().toFixed(0)}s`);
for (const s of RESULTS.stages) log(`  ${s.rise_mm} mm: cleared ${s.k} of 9, objective ${s.objective}, ${s.blame}`);
