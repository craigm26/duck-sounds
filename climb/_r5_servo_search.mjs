// ROUND 5 — a SCREENING search over the servoed landing, on the round-3 vault.
//
// This is not a 336-evaluation CEM run; round 4 already spent one of those on
// this vector and improved it zero times. This is the smallest honest question
// the new lever can be asked: with the vault's own approach, blend, gap and
// side held fixed, and the neck still doing the strut, does ANY setting of the
// per-tick landing law beat the timed landing on the 9-cell core grid?
//
// Genes (16): the arm time, five set-points, nine gains, the rate limit.
// Everything else — the six keyframes, blend 2.1572, gap 0.0187, side 0.0078,
// approach 0.1663 — is the published round-3 vector, unchanged.
//
// Two stages, because a full 9-cell evaluation is ~5 s and a single nominal
// cell is ~0.55 s:
//   STAGE 1  one cell (60 mm, drop 0.120, friction x1.0), rank by rig3 reward
//   STAGE 2  the survivors on the full 9-cell core grid, ranked by kCoreStable
//            then objectiveCore
//
// Every candidate is written to disk and scored FROM DISK. mulberry32 seeds.
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/_r5_servo_search.mjs
import fs from 'node:fs';
import { scoreCell, scoreRobust, intentHash, saveIntent } from '../climb/robust.mjs';

const P = '../climb/';
const LOG = [];
const log = s => { console.log(s); LOG.push(String(s)); };
const T0 = Date.now();
const el = () => ((Date.now() - T0) / 1000).toFixed(0) + 's';
const MIN = 60000;
const STAGE1_UNTIL = 6.0 * MIN;
const STAGE2_UNTIL = 12.5 * MIN;

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
const SEED = 0x5EED0005;
const rnd = mulberry32(SEED);
const U = (lo, hi) => lo + (hi - lo) * rnd();

const base = JSON.parse(fs.readFileSync(P + 'best_r3_vault_60mm.json', 'utf8'));
const KT = base.keyframes.map(k => k.t);
const RISE = 0.060;

// The gene box. Arm anywhere from the beak-plant keyframe (0.7343 s) to the
// last keyframe before the settle-out (1.7311 s) — i.e. anywhere across the
// pivot and the landing.
const GENES = [
  ['at', KT[2], KT[4]],
  ['zTarget', 0.06, 0.16], ['xTrunk', 0.05, 0.30], ['xFoot', 0.02, 0.24],
  ['fz', -0.01, 0.06], ['pitchRef', -0.6, 0.6],
  ['kHipZ', -4, 4], ['kHipPitch', -1.5, 1.5], ['kHipX', -4, 4], ['kHipTrunkX', -3, 3],
  ['kKneeZ', -5, 5], ['kKneeFz', -5, 5], ['kKneeX', -4, 4],
  ['kAnkPitch', -1.5, 1.5], ['kAnkFz', -5, 5],
  ['rate', 0.03, 0.35],
];
const draw = () => Object.fromEntries(GENES.map(([n, lo, hi]) => [n, +U(lo, hi).toFixed(5)]));
const jitter = (g, s) => Object.fromEntries(GENES.map(([n, lo, hi]) => {
  const w = (hi - lo) * s;
  return [n, +Math.max(lo, Math.min(hi, g[n] + (rnd() * 2 - 1) * w)).toFixed(5)];
}));

const CAND = P + '_r5_cand.json';
const mk = (g, name, note) => ({ ...base, name, family: 'r5_servo',
  servo: { ...g, yawRoll: 'hold', span: 1.2, sign: [1, -1] }, note });

log('================================================================');
log(`ROUND 5 SERVO SCREEN — base best_r3_vault_60mm.json (sha ${intentHash(base).slice(0, 12)}), rise 60 mm, seed 0x${SEED.toString(16)}`);
log(`   16 genes; blend ${base.blend} gap ${base.gap} side ${base.side} approach ${base.approach} and all six keyframes are the published round-3 vector, unchanged.`);

// ------------------------------------------------------------------ BASELINE
const baseGrid = await scoreRobust(P + 'best_r3_vault_60mm.json', { rise: RISE, core: true });
log(`   BASELINE (timed landing): kCore=${baseGrid.kCore}/9 kCoreStable=${baseGrid.kCoreStable}/9 objectiveCore=${baseGrid.objectiveCore.toFixed(4)} meanRewardCore=${baseGrid.meanRewardCore.toFixed(4)}  [${el()}]`);

// ------------------------------------------------------------------- STAGE 1
log('');
log('STAGE 1 — one cell (60 mm, drop 0.120, friction x1.0), rank by rig3 reward');
const pool = [];
let n1 = 0, bestR = -1;
while (Date.now() - T0 < STAGE1_UNTIL) {
  // 70% fresh draws, 30% jitters around the best so far
  const g = (pool.length && rnd() < 0.30) ? jitter(pool[0].g, 0.18) : draw();
  saveIntent(mk(g, 'r5_servo_cand', 'STAGE-1 candidate'), CAND);
  const r = await scoreCell(CAND, { rise: RISE });
  n1++;
  const row = { g, reward: r.reward, honest: r.crit.honest, x: r.scored.x, above: r.scored.above,
                feetOnTread: r.scored.feetOnTread, up: r.scored.up, armed: r.servo ? r.servo.armed : null,
                minPenEpisode_mm: r.minPenetrationEpisode === null ? null : +(r.minPenetrationEpisode * 1000).toFixed(2) };
  pool.push(row);
  pool.sort((a, b) => b.reward - a.reward || (b.honest ? 1 : 0) - (a.honest ? 1 : 0));
  if (pool.length > 24) pool.length = 24;
  if (row.reward > bestR) {
    bestR = row.reward;
    log(`   #${String(n1).padStart(4)} reward ${row.reward.toFixed(3)} honest=${row.honest} feetOnTread=${row.feetOnTread} x=${(row.x * 1000).toFixed(1)}mm above=${(row.above * 1000).toFixed(1)}mm minPenEp=${row.minPenEpisode_mm}mm at=${row.g.at}  [${el()}]`);
  }
}
log(`   ${n1} candidates screened; best single-cell reward ${bestR.toFixed(3)} (baseline cell reward would be the vault's own)  [${el()}]`);

// ------------------------------------------------------------------- STAGE 2
log('');
log('STAGE 2 — survivors on the full 9-cell core grid. THE GATE: kCoreStable >= 7 of 9.');
const results = [];
let bestK = -1, bestObj = -Infinity, bestFile = null, bestSha = null;
let i = 0;
for (const p of pool) {
  if (Date.now() - T0 > STAGE2_UNTIL) { log(`   (stopped at ${el()}, ${i}/${pool.length} survivors evaluated)`); break; }
  i++;
  const obj = mk(p.g, 'r5_servo_s2', 'STAGE-2 candidate, 9-cell core grid');
  saveIntent(obj, CAND);
  const g = await scoreRobust(CAND, { rise: RISE, core: true });
  const row = { rank: i, genes: p.g, sha256: intentHash(obj), kCore: g.kCore, kCoreStable: g.kCoreStable,
    objectiveCore: g.objectiveCore, meanRewardCore: g.meanRewardCore,
    minPenetrationEpisode_mm: +g.agg.minPenetrationEpisode_mm.toFixed(2),
    minPenetrationAtScore_mm: +g.agg.minPenetrationAtScore_mm.toFixed(2),
    maxTq: g.agg.maxTq, servoArmedCells: g.verdicts.filter(v => v.servoArmed).length };
  results.push(row);
  const better = row.kCoreStable > bestK || (row.kCoreStable === bestK && row.objectiveCore > bestObj);
  log(`   s2 #${i} kCore=${row.kCore}/9 kCoreStable=${row.kCoreStable}/9 objCore=${row.objectiveCore.toFixed(4)} minPenEp=${row.minPenetrationEpisode_mm}mm maxTq=${row.maxTq.toFixed(4)} armedCells=${row.servoArmedCells}/9${better ? '   <-- IMPROVEMENT' : ''}  [${el()}]`);
  if (better) {
    bestK = row.kCoreStable; bestObj = row.objectiveCore;
    bestFile = P + 'best_r5_servo_60mm.json';
    saveIntent({ ...obj, name: 'best_r5_servo_60mm',
      note: `ROUND 5, servo family. best_r3_vault_60mm.json's six keyframes / blend / gap / side / approach, unchanged, with a SERVOED LANDING (climb/servo.mjs) armed at t=${p.g.at}s. Screened over ${n1} single-cell candidates from mulberry32 seed 0x${SEED.toString(16)}; this is the best of ${i} survivors on the 9-cell core grid. kCore=${row.kCore}/9, kCoreStable=${row.kCoreStable}/9, objectiveCore=${row.objectiveCore.toFixed(4)}. THE TIMED LANDING SCORES kCore=${baseGrid.kCore}/9, kCoreStable=${baseGrid.kCoreStable}/9, objectiveCore=${baseGrid.objectiveCore.toFixed(4)}.`,
      robust: { kCore: row.kCore, kCoreStable: row.kCoreStable, objectiveCore: row.objectiveCore, sha256: row.sha256 } }, bestFile);
    bestSha = intentHash(JSON.parse(fs.readFileSync(bestFile, 'utf8')));
  }
}

results.sort((a, b) => b.kCoreStable - a.kCoreStable || b.objectiveCore - a.objectiveCore);
const gatePassed = bestK >= 7;
log('');
log(`BEST SERVOED: kCoreStable=${bestK}/9 objectiveCore=${bestObj === -Infinity ? 'n/a' : bestObj.toFixed(4)}   file ${bestFile}   sha ${bestSha}`);
log(`BASELINE TIMED: kCoreStable=${baseGrid.kCoreStable}/9 objectiveCore=${baseGrid.objectiveCore.toFixed(4)}  (sha ${baseGrid.sha256.slice(0, 12)})`);
log(`KILL GATE (kCoreStable >= 7 of 9 at 60 mm): ${gatePassed ? 'PASSED' : 'NOT REACHED — on this evidence the 40-80 mm band is CLOSED and the negative is the result'}`);
fs.writeFileSync(P + 'r5_servo_search-results.json', JSON.stringify({
  generated: new Date().toISOString(), script: '_r5_servo_search.mjs', seed: '0x' + SEED.toString(16),
  base: 'best_r3_vault_60mm.json', baseSha: intentHash(base), rise_mm: 60,
  genes: GENES.map(([n, lo, hi]) => ({ gene: n, lo, hi })),
  baseline: { kCore: baseGrid.kCore, kCoreStable: baseGrid.kCoreStable, objectiveCore: baseGrid.objectiveCore,
              meanRewardCore: baseGrid.meanRewardCore, sha256: baseGrid.sha256 },
  stage1Candidates: n1, stage1BestReward: bestR, stage2Evaluated: results.length,
  best: { kCoreStable: bestK, objectiveCore: bestObj === -Infinity ? null : bestObj, file: bestFile, sha256: bestSha },
  killGate: `if no move reaches kCoreStable >= 7 of 9 at 60 mm, the 40-80 mm band is CLOSED and the negative is the result. ${gatePassed ? 'REACHED.' : 'NOT REACHED.'}`,
  results,
}, null, 2));
fs.writeFileSync(P + '_r5_servo_search.log', LOG.join('\n') + '\n');
log(`wrote climb/r5_servo_search-results.json  [${el()}]`);
