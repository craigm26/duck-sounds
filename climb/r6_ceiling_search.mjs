// r6_ceiling_search.mjs — ROUND 6, PHASE 2: SEARCH THE CEILING AND NOTHING ELSE.
//
// Round 5's finding: rig3 `honest` needs the trunk more than 95 mm above the
// tread at the scored instant, so
//
//   ceilingCore = # of the 9 core cells whose trunk EVER exceeds 95 mm above
//                 that cell's tread  ( = max(trunk z) - cellRise > 0.095 )
//
// is a HARD UPPER BOUND on kCore under ANY landing law. Both round-5 bests had
// kCore == ceilingCore, so a perfect landing added nothing: the binding
// constraint is HEIGHT, not the landing. This search therefore optimises the
// ceiling ALONE — no landing term, no upright term, no servo, no event —
// over the beak-strut vault's launch parameters at 60 mm.
//
//   objective = ceilingCore + mean over the 9 core cells of
//               min(peak trunk height above that cell's tread, 0.12 m)
//
// The continuous term is bounded by 0.12, so it can never trade one ceiling
// cell for height elsewhere; it only orders moves that tie on ceilingCore.
//
// PARAMETRISATION: climb/vault.mjs trackOf()/BOUNDS as copied verbatim into
// climb/famA_vault.mjs — the crouch / preload / vault / tuck / land keyframes,
// their five durations, blend, gap, side, approach, the strut angles and roll.
//
// SCREENING. A 9-cell evaluation costs ~4.8 s on this box; one cell ~0.53 s.
// Every candidate is first run on the NOMINAL cell (rise 60, drop 0.120,
// friction x1.0). A duck standing still on the floor puts its trunk about
// 60 mm above a 60 mm tread, so a nominal peak below 80 mm means the trunk
// never got 20 mm above standing height and the move is not a vault.
// No reported ceilingCore ever comes from a screened evaluation.
//
// THE FIRST ATTEMPT AT THIS SEARCH FAILED, AND WHY. Run 1 (sig0 0.060, cut
// 0.085) put screened candidates into the elite set below the fully evaluated
// ones, exactly as climb/famA_vault.mjs does. At this sigma NOTHING passed the
// screen: seven generations ran with 0 full evaluations, so the elite was four
// junk vectors plus the incumbent and the CEM mean walked straight out of the
// vault basin. It was killed at 88 s. Two changes, both stated here so the
// difference from famA is visible:
//   1. THE MEAN MOVES ONLY ON FULL EVALUATIONS. Screened candidates never
//      enter the elite set at all. They carry no ceilingCore, so they have
//      nothing to say about the objective; all they report is that the
//      proposal is too wide. If a generation produces no full evaluation the
//      mean does not move.
//   2. THE STEP SIZE ADAPTS to the screen. Fewer than 3 full evaluations in a
//      generation shrinks sigma by 0.75 (down to the floor); 8 or more grows
//      it by 1.15 (up to 2x sig0). That is the standard CEM/CMA step-size
//      control, and here it is what finds the width of the basin instead of
//      guessing it.
//
// Every candidate is WRITTEN TO JSON and re-read before it is scored, through
// climb/robust.mjs scoreRobust — the one shared scorer. mulberry32, seed 6000.
//
// Run from sim/:  node ../climb/r6_ceiling_search.mjs [totalSeconds]
import fs from 'node:fs';
import { scoreRobust, scoreCell, PLANTS, DHS, HOME } from './robust.mjs';

const P = '../climb/';
const SCRATCH = '/tmp/claude-1000/-home-craigm26/43f07dce-e62f-464d-ae1f-2fe020620950/scratchpad';
fs.mkdirSync(SCRATCH, { recursive: true });
const RISE = 0.060;
const BAR = 0.095;          // the criterion's own height bar; NEVER lowered
const CAP = 0.12;           // the continuous term's cap
const T0 = Date.now();
const el = () => ((Date.now() - T0) / 1000);
const log = s => { console.log(`[${el().toFixed(0).padStart(4)}s] ${s}`); };

// ------------------------------------------------------------------ track
// climb/vault.mjs trackOf() and BOUNDS, verbatim (as in climb/famA_vault.mjs).
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
const SEED_BASE = 6000;
let RND = mulberry32(SEED_BASE);
function gauss() { let u = 0, v = 0; while (!u) u = RND(); while (!v) v = RND(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
const clampB = (k, v) => Math.min(BOUNDS[k][1], Math.max(BOUNDS[k][0], v));
const randP = () => Object.fromEntries(KEYS.map(k => [k, BOUNDS[k][0] + RND() * RANGE[k]]));
const sampleN = (mu, sg) => Object.fromEntries(KEYS.map(k => [k, clampB(k, mu[k] + sg[k] * gauss())]));

// ------------------------------------------------------------------ eval
const r5 = v => +v.toFixed(5);
function intentOf(p, note) {
  return { name: `beak_strut_vault_r6_ceiling_60mm`, family: 'R6 ceiling — beak-strut vault launch',
           keyframes: trackOf(p).map(f => ({ t: +f.t.toFixed(4), pose: f.pose.map(r5) })),
           blend: +p.blend.toFixed(4), gap: +p.gap.toFixed(4), side: +p.side.toFixed(4),
           approach: +p.approach.toFixed(4), isolate: true, stepCount: 4, params: p, note };
}
/** THE CEILING, off the scorer's own verdict rows. Nothing is recomputed. */
function ceilOf(verdicts) {
  const core = verdicts.filter(v => v.tier === 'core');
  const peaks = core.map(v => +((v.peakZ_mm - v.rise_mm) / 1000).toFixed(4));   // m above that cell's tread
  const ceilingCore = peaks.filter(z => z > BAR).length;
  const cont = peaks.reduce((a, z) => a + Math.min(z, CAP), 0) / peaks.length;
  return { ceilingCore, peaks, cont, objective: ceilingCore + cont };
}
const SCREEN_CUT = 0.080;   // m of peak trunk above the tread, nominal cell
let EVALS = 0, FULL = 0;
async function evalP(p, { screen = true } = {}) {
  const path = `${SCRATCH}/r6_cand.json`;
  fs.writeFileSync(path, JSON.stringify(intentOf(p, 'candidate'), null, 2));
  EVALS++;
  if (screen) {
    const c0 = await scoreCell(path, { rise: RISE, dh: 0, drop: PLANTS[0].drop, fmul: PLANTS[0].fmul, isolate: true });
    const peak0 = c0.maxZ - RISE;
    if (!(peak0 >= SCREEN_CUT)) {
      return { screened: true, ceilingCore: null, objective: -1, rank2: peak0,
               peak0_mm: +(peak0 * 1000).toFixed(1) };
    }
  }
  const r = await scoreRobust(path, { rise: RISE, core: true });
  FULL++;
  if (r.invalid) return { screened: true, ceilingCore: null, objective: -1, rank2: -1, invalid: true };
  const c = ceilOf(r.verdicts);
  return { screened: false, ...c, kCore: r.kCore, kCoreStable: r.kCoreStable,
           sha256: r.sha256, move: r.move, verdicts: r.verdicts, agg: r.agg,
           maxTq: r.agg.maxTq, minPen_mm: +r.agg.minPenetrationEpisode_mm.toFixed(2),
           satFrac: r.agg.satFrac };
}

// ------------------------------------------------------------------ CEM
const POP = 24, ELITE = 5;
const SIG0 = 0.030, SIGMIN = 0.008;      // fractions of each key's bound range
const SIGMAX = 0.075;                    // adaptive ceiling, fraction of range
const GROW = 1.15, SHRINK = 0.75, WANT_FULL_LO = 3, WANT_FULL_HI = 8;
const RANKW = [5, 4, 3, 2, 1];

const SEEN = new Map();                  // sha256 -> row, every DISTINCT full eval
const note = r => `ceil=${r.ceilingCore}/9 obj=${r.objective.toFixed(4)} kCore=${r.kCore}/9 kStab=${r.kCoreStable}/9 tq=${r.maxTq.toFixed(4)} pen=${r.minPen_mm}mm`;
function remember(p, r) {
  if (r.screened || SEEN.has(r.sha256)) return;
  SEEN.set(r.sha256, { sha256: r.sha256, move: r.move, ceilingCore: r.ceilingCore,
    objective: +r.objective.toFixed(4), continuousTerm: +r.cont.toFixed(4),
    peakAboveTread_mm: r.peaks.map(z => +(z * 1000).toFixed(1)),
    kCore: r.kCore, kCoreStable: r.kCoreStable, maxTq: r.maxTq,
    minPenetrationEpisode_mm: r.minPen_mm, satFrac: +r.satFrac.toFixed(4),
    foundAt_s: +el().toFixed(1), params: p,
    verdicts: r.verdicts.filter(v => v.tier === 'core') });
}

const RESULTS = { generated: new Date().toISOString(),
  phase: 'ROUND 6 — THE CEILING', rise_mm: 60,
  objective: 'ceilingCore + mean over the 9 core cells of min(peak trunk height above that cell\'s tread, 0.12 m)',
  bar_m: BAR, cap_m: CAP,
  landingTerm: 'NONE', servo: 'NONE', event: 'NONE',
  scorer: 'climb/robust.mjs scoreRobust({rise:0.060, core:true}) — the one shared scorer',
  parametrisation: 'climb/vault.mjs trackOf/BOUNDS (beak-strut vault launch), as in climb/famA_vault.mjs',
  seedBase: SEED_BASE, rng: 'mulberry32', pop: POP, elite: ELITE, sig0: SIG0, sigMin: SIGMIN,
  screenCut_m: SCREEN_CUT, plants: PLANTS, dhs: DHS,
  warmStarts: [], generations: [], distinctMoves: [], best: null };
// The distinct-move table is rebuilt on EVERY flush, not only at the end: the
// first version of this file assembled it once after the loops and a scope bug
// in the finaliser would have thrown 35 minutes of search away.
const flush = () => {
  RESULTS.distinctMoves = [...SEEN.values()].sort((a, b) => b.ceilingCore - a.ceilingCore || b.objective - a.objective);
  RESULTS.distinct = SEEN.size; RESULTS.evals = EVALS; RESULTS.fullEvals = FULL;
  RESULTS.bestCeilingCore = RESULTS.distinctMoves.length ? RESULTS.distinctMoves[0].ceilingCore : null;
  RESULTS.wall_s = +el().toFixed(1);
  fs.writeFileSync(`${P}r6_ceiling-results.json`, JSON.stringify(RESULTS, null, 1) + '\n');
};

function publish(p, r, file, tag) {
  const obj = intentOf(p, `${tag}: ceilingCore ${r.ceilingCore} of 9 at 60 mm; objective ${r.objective.toFixed(4)}; kCore ${r.kCore} of 9 (no landing law, no servo)`);
  obj.ceiling = { rise_mm: 60, bar_mm: 95, ceilingCore: r.ceilingCore, of: 9,
    objective: +r.objective.toFixed(4), continuousTerm: +r.cont.toFixed(4),
    peakAboveTread_mm: r.peaks.map(z => +(z * 1000).toFixed(1)),
    kCore: r.kCore, kCoreStable: r.kCoreStable, maxTq: r.maxTq,
    minPenetrationEpisode_mm: r.minPen_mm, verdicts: r.verdicts };
  fs.writeFileSync(P + file, JSON.stringify(obj, null, 2));
  return file;
}

const TOTAL = +(process.argv[2] || 2100);
log(`R6 CEILING. rise 60 mm only. bar ${BAR * 1000} mm (NOT moved). pop ${POP}, elite ${ELITE}, sig0 ${SIG0}, sigMin ${SIGMIN}, sigMax ${SIGMAX}, seed ${SEED_BASE}, budget ${TOTAL}s`);
log(`grid: rise ${DHS.map(d => 60 + d * 1000).join('/')} mm x plants ${PLANTS.map(p => `${p.drop}/x${p.fmul}`).join(' ')}`);

const HIST = [];
let BEST = null;                 // { p, r } over BOTH chains
let hit7 = false;
let GENS = 0;                    // generations run over BOTH chains

async function cem(mu0, deadline, tag, seedFile) {
  let mu = { ...mu0 };
  let sgf = SIG0;
  let sg = Object.fromEntries(KEYS.map(k => [k, sgf * RANGE[k]]));
  const seed = await evalP(mu, { screen: false });
  remember(mu, seed);
  if (seed.screened) { log(`${tag} warm start INVALID — skipped`); return null; }
  let best = { p: { ...mu }, r: seed };
  log(`${tag} warm-start (${seedFile}, ${seed.move}): ${note(seed)} peaks[${seed.peaks.map(z => (z * 1000).toFixed(1)).join(' ')}]`);
  RESULTS.warmStarts.push({ chain: tag, file: seedFile, sha256: seed.sha256, ...SEEN.get(seed.sha256) });
  if (!BEST || seed.objective > BEST.r.objective) BEST = best;
  flush();
  let gen = 0;
  while (Date.now() < deadline && !hit7) {
    gen++; GENS++;
    const scored = [];
    for (let i = 0; i < POP; i++) {
      if (Date.now() >= deadline) break;
      // one local ESCAPE per generation at 2.5x sigma, and a uniform draw over
      // the whole box every fifth generation, so a path out of the basin stays
      // open without four junk vectors ever touching the mean.
      const p = (i === POP - 1)
        ? (gen % 5 === 0 ? randP() : sampleN(mu, Object.fromEntries(KEYS.map(k => [k, sg[k] * 2.5]))))
        : sampleN(mu, sg);
      const r = await evalP(p);
      remember(p, r);
      scored.push({ p, r });
      if (!r.screened && r.objective > best.r.objective) {
        best = { p, r };
        log(`${tag} gen${gen} IMPROVE ${note(r)} ${r.move} peaks[${r.peaks.map(z => (z * 1000).toFixed(1)).join(' ')}]`);
        HIST.push({ chain: tag, gen, kind: 'improve', at_s: +el().toFixed(1), sha256: r.sha256,
                    ceilingCore: r.ceilingCore, objective: +r.objective.toFixed(4),
                    kCore: r.kCore, peakAboveTread_mm: r.peaks.map(z => +(z * 1000).toFixed(1)) });
        if (!BEST || r.objective > BEST.r.objective) { BEST = best; publish(p, r, 'best_r6_ceilvault_60mm.json', 'round-6 ceiling search, running best'); }
        flush();
        if (r.ceilingCore >= 7) { log(`${tag} ceilingCore >= 7 at gen${gen} — stopping`); hit7 = true; break; }
      }
    }
    if (!scored.length) break;
    // THE MEAN MOVES ONLY ON FULL EVALUATIONS (see the header). Screened
    // candidates carry no ceilingCore and never enter the elite set.
    const fullE = scored.filter(s => !s.r.screened).sort((a, b) => b.r.objective - a.r.objective);
    if (fullE.length) {
      let eli = [best, ...fullE.filter(e => e.r !== best.r)].slice(0, ELITE);
      const wsum = eli.reduce((a, _, i) => a + RANKW[i], 0);
      const nmu = {}, nsg = {};
      for (const k of KEYS) {
        let m = 0; for (let i = 0; i < eli.length; i++) m += RANKW[i] * eli[i].p[k]; m /= wsum;
        let v = 0; for (let i = 0; i < eli.length; i++) v += RANKW[i] * (eli[i].p[k] - m) ** 2; v = Math.sqrt(v / wsum);
        nmu[k] = m; nsg[k] = v;
      }
      mu = nmu; sg = nsg;
    }
    // adaptive step size against the screen pass rate
    if (fullE.length < WANT_FULL_LO) sgf = Math.max(SIGMIN, sgf * SHRINK);
    else if (fullE.length >= WANT_FULL_HI) sgf = Math.min(SIGMAX, sgf * GROW);
    for (const k of KEYS) sg[k] = Math.min(Math.max(sg[k] || 0, sgf * RANGE[k]), SIGMAX * RANGE[k]);
    const maxCeilSoFar = Math.max(...[...SEEN.values()].map(r => r.ceilingCore));
    log(`${tag} gen${gen}: full=${fullE.length}/${scored.length} sig=${sgf.toFixed(4)} elite [${(fullE.length ? [best, ...fullE].slice(0, ELITE) : [best]).map(e => e.r.objective.toFixed(3)).join(' ')}] | best ceil=${best.r.ceilingCore}/9 obj=${best.r.objective.toFixed(4)} | maxCeilSeen=${maxCeilSoFar} | evals=${EVALS} full=${FULL} distinct=${SEEN.size}`);
    RESULTS.generations.push({ chain: tag, gen, at_s: +el().toFixed(1), full: fullE.length, of: scored.length,
      sigmaFraction: +sgf.toFixed(4),
      bestCeilingCore: best.r.ceilingCore, bestObjective: +best.r.objective.toFixed(4),
      maxCeilingCoreSeen: maxCeilSoFar, evals: EVALS, fullEvals: FULL, distinct: SEEN.size });
    flush();
  }
  return { best, gen };
}

// CHAIN A: the record (4b9110c448ec), ceilingCore 5 — the highest in the corpus.
// CHAIN B: best_r5_servo_60mm (e0434c2c90da), ceilingCore 4 — the same
// parametrisation, a different basin, and the only other launch in the corpus
// above ceilingCore 3.
const muA = JSON.parse(fs.readFileSync(P + 'best_r3_vault_60mm.json', 'utf8')).params;
const muB = JSON.parse(fs.readFileSync(P + 'best_r5_servo_60mm.json', 'utf8')).params;
const A = await cem(muA, T0 + TOTAL * 0.60 * 1000, 'A', 'best_r3_vault_60mm.json');
if (!hit7) await cem(muB, T0 + TOTAL * 1000, 'B', 'best_r5_servo_60mm.json');

// ------------------------------------------------------------------ publish
const table = [...SEEN.values()].sort((a, b) => b.ceilingCore - a.ceilingCore || b.objective - a.objective);
RESULTS.distinctMoves = table;   // (flush() also rebuilds this every generation)
RESULTS.history = HIST;
RESULTS.chains = 2;
RESULTS.evals = EVALS; RESULTS.fullEvals = FULL; RESULTS.distinct = SEEN.size;
RESULTS.wall_s = +el().toFixed(1); RESULTS.generationsRun = GENS;
RESULTS.bestCeilingCore = table.length ? table[0].ceilingCore : null;
RESULTS.best = table.length ? table[0] : null;
RESULTS.killCondition = {
  rule: 'if no launch reaches ceilingCore >= 7 of 9 at 60 mm, the search is FINISHED AT THIS SCALE',
  bestCeilingCore: RESULTS.bestCeilingCore,
  result: (RESULTS.bestCeilingCore >= 7) ? 'PASSED' : 'FAILED',
};

// One published file per DISTINCT vector, for the top three that are not the record.
const WARM_SHAS = new Set(RESULTS.warmStarts.map(w => w.sha256));
const files = [];
let n = 0;
for (const row of table) {
  if (WARM_SHAS.has(row.sha256)) continue;
  if (n >= 3) break;
  const nm = ['best_r6_ceilvault_60mm.json', 'best_r6_ceilvaultB_60mm.json', 'best_r6_ceilvaultC_60mm.json'][n];
  publish(row.params, { ceilingCore: row.ceilingCore, objective: row.objective,
    cont: row.continuousTerm, peaks: row.peakAboveTread_mm.map(v => v / 1000),
    kCore: row.kCore, kCoreStable: row.kCoreStable, maxTq: row.maxTq,
    minPen_mm: row.minPenetrationEpisode_mm, verdicts: row.verdicts },
    nm, `round-6 ceiling search, rank ${n + 1}`);
  files.push({ file: `climb/${nm}`, sha256: row.sha256, ceilingCore: row.ceilingCore, objective: row.objective });
  n++;
}
RESULTS.published = files;
flush();

log(`DONE. gens=${GENS} evals=${EVALS} full9=${FULL} distinct=${SEEN.size} wall=${el().toFixed(0)}s`);
console.log('\n=== EVERY DISTINCT MOVE, sorted by ceilingCore ===');
console.log('  ceil  objective  kCore kStab  move          maxTq   minPen(mm)  satFrac  peak trunk above tread, core 9 (mm)');
for (const r of table.slice(0, 60))
  console.log(`  ${String(r.ceilingCore).padStart(2)}/9  ${r.objective.toFixed(4).padStart(9)}  ${String(r.kCore).padStart(2)}/9 ${String(r.kCoreStable).padStart(4)}/9  ${r.move}  ${r.maxTq.toFixed(4)}  ${String(r.minPenetrationEpisode_mm).padStart(9)}  ${r.satFrac.toFixed(4)}  [${r.peakAboveTread_mm.map(x => String(x).padStart(6)).join(' ')}]`);
console.log(`\n  BEST ceilingCore over ${SEEN.size} distinct moves: ${RESULTS.bestCeilingCore} of 9.`);
console.log(`  KILL CONDITION (needs ceilingCore >= 7 of 9 at 60 mm): ${RESULTS.killCondition.result}`);
console.log(`  published: ${files.map(f => f.file).join(', ') || '(none — nothing beat the record)'}`);
console.log(`  wrote climb/r6_ceiling-results.json`);
