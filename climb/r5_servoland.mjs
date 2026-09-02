// r5_servoland.mjs — ROUND 5, THE ONE UNTRIED LEVER.
//
// THE FAMILY. The round-3 beak-strut vault (best_r3_vault_60mm.json, move
// 4b9110c448ec) keeps its LAUNCH — the four keyframes that plant the beak,
// lock the neck as a strut, extend the hips and pitch the trunk over the head.
// Its LANDING is deleted and replaced by climb/servo.mjs's per-tick feedback
// law: from the arm tick on, hip pitch / knee / ankle are commanded every
// 5.66 ms control tick from the MEASURED trunk height above the tread, trunk
// pitch, and each foot's own position relative to the tread's front edge and
// top. The head and neck slots (5-8) stay on the track: the neck is the strut.
//
// WHAT IS SEARCHED. 21 numbers, together, by CEM under climb/robust.mjs's
// 14-cell grid at 60 mm:
//   launch  blend, gap, side, and the four keyframe DURATIONS
//   law     arm time, the three set-points (trunk height, foot x, foot z),
//           the pitch reference, nine gains, and the rate limit
//
// RANKING is robust.mjs's own `objective` over all 14 cells. Every candidate
// is WRITTEN TO DISK and scored from the file; nothing in memory is scored.
//
// SCREENING. A 14-cell evaluation costs 6.6 s. Before spending it, two single
// cells (robust.scoreCell, which robust.mjs itself designates for screening)
// are run at 60 mm: the nominal plant and the slippery x0.5 plant. A candidate
// that is honest in neither AND whose better reward is below 5.5 is not scored
// on the grid; it is ranked below every scored candidate by -100 + its screen
// mean, so CEM still has a gradient among the rejects. The warm start screens
// (12.000, 5.011). Only scoreRobust numbers are ever reported.
import fs from 'node:fs';
import { scoreRobust, scoreCell, intentHash, saveIntent, HOME } from '../climb/robust.mjs';

const RISE = 0.060;
const OUT = '../climb';
const CAND = OUT + '/_r5_cand.json';
const BUDGET_S = +(process.env.R5_BUDGET_S || 1950);   // CEM wall-clock budget
const T_START = Date.now();
const el = () => (Date.now() - T_START) / 1000;

// ---------------------------------------------------------------- mulberry32
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rnd = mulberry32(0x5EED0005);
let spare = null;
function gauss() {                       // Box-Muller on mulberry32
  if (spare !== null) { const v = spare; spare = null; return v; }
  let u = 0, v = 0;
  while (u <= 1e-12) u = rnd();
  v = rnd();
  const r = Math.sqrt(-2 * Math.log(u));
  spare = r * Math.sin(2 * Math.PI * v);
  return r * Math.cos(2 * Math.PI * v);
}

// ------------------------------------------------- the round-3 vault's poses
const F1 = [0,-0.21255,-0.36327,0.4539,0.48228,-0.32872,-0.7469,0,0,0,-0.03795,0.36327,-0.4539,-0.48228];
const F2 = [0,-0.21255,-0.24628,-0.50213,0.12444,-0.32872,-0.7469,0,0,0,-0.03795,0.24628,0.50213,-0.12444];
const F3 = [0,-0.21255,-0.03892,0.67747,0.50507,-0.32872,-0.7469,0,0,0,-0.03795,0.03892,-0.67747,-0.50507];
const F4 = [0,-0.21255,0.03891,-1.05397,0.83188,-0.32872,-0.7469,0,0,0,-0.03795,-0.03891,1.05397,-0.83188];
const F5 = [0,-0.0873,-0.26283,-0.25249,0.8677,0.58128,-0.28134,0,0,0,0.0873,0.26283,0.25249,-0.8677];
const F6 = [0,-0.0873,-0.4579,-0.0049,0.453,0.3491,0.3491,0,0,0,0.0873,0.4579,0.0049,-0.453];
const D_LAND = 0.5827, D_SETTLE = 0.7;

// ------------------------------------------------------------- the parameters
const r4 = v => +v.toFixed(4), r6 = v => +v.toFixed(6);
const SPEC = [
  // name          mu        sigma    lo      hi     round
  ['blend',      2.1572,   0.120,   0.70,   2.40,  r4],
  ['gap',        0.0187,   0.0050,  0.000,  0.045, r4],
  ['side',       0.0078,   0.0080, -0.020,  0.090, r4],
  ['dReach',     0.3119,   0.025,   0.150,  0.500, r4],
  ['dPre',       0.2341,   0.025,   0.100,  0.450, r4],
  ['dVault',     0.1883,   0.025,   0.080,  0.400, r4],
  ['dTuck',      0.4141,   0.040,   0.150,  0.800, r4],
  ['atFrac',     1.0000,   0.600,  -0.100,  4.000, r4],   // arm time: a fraction of the tuck leg, measured from the vault frame. 1 = the tuck keyframe, ~2.4 = the landing keyframe, above that = the servo as a late stabiliser after the throw has landed
  ['baseMix',    1.0000,   0.250,  -0.200,  1.600, r6],   // 0 = the tuck pose, 1 = the throw's landing pose
  ['zTarget',    0.115,    0.025,   0.040,  0.200, r6],
  ['xFoot',      0.100,    0.035,   0.020,  0.220, r6],
  ['fz',         0.015,    0.018,  -0.020,  0.080, r6],
  ['kHipZ',      0.000,    2.500,  -8.000,  8.000, r6],
  ['kHipPitch',  0.000,    0.400,  -2.000,  2.000, r6],
  ['kHipX',     -2.000,    2.500,  -8.000,  8.000, r6],
  ['kKneeZ',     0.000,    2.500,  -8.000,  8.000, r6],
  ['kKneeFz',   -4.000,    3.000, -12.000, 12.000, r6],
  ['kKneeX',     0.000,    2.500,  -8.000,  8.000, r6],
  ['kAnkPitch',  0.000,    0.400,  -2.000,  2.000, r6],
  ['kAnkFz',     1.000,    2.500,  -8.000,  8.000, r6],
  ['rate',       0.120,    0.060,   0.020,  0.500, r6],
  ['span',       1.200,    0.400,   0.300,  2.600, r6],
];
const NAMES = SPEC.map(s => s[0]);
const MU0 = SPEC.map(s => s[1]), SIG0 = SPEC.map(s => s[2]);
const LOB = SPEC.map(s => s[3]), HIB = SPEC.map(s => s[4]), RND = SPEC.map(s => s[5]);
const SIGFLOOR = SIG0.map(s => s * 0.30);   // the variance floor
const clampv = v => v.map((x, i) => RND[i](Math.min(Math.max(x, LOB[i]), HIB[i])));
const asObj = v => Object.fromEntries(NAMES.map((n, i) => [n, v[i]]));

/**
 * THE BLEND IS PART OF THE COMMAND, SO IT MUST BE PART OF THE BASE.
 * rig3/robust command an unservoed slot as HOME[k] + action[k] + (pose[k] -
 * HOME[k]) * blend: at this move's blend of 2.157 the keyframe landing is
 * amplified 2.157x. servo.mjs's default base is the RAW track pose, which at
 * the same instant is a 2.157x smaller excursion — a servo armed with zero
 * gains and no authored base is therefore not "the throw held", it is a
 * different and much weaker command. So this family AUTHORS servo.base:
 *
 *   base[k] = HOME[k] + (mix(tuckPose, landPose, baseMix)[k] - HOME[k]) * blend
 *
 * baseMix 1.0 is the throw's own landing pose put through the same blend, so
 * the law is a CORRECTION around the round-3 landing rather than a
 * replacement of it; baseMix 0 freezes the legs at the tuck.
 */
const blendedBase = (pose, blend) => pose.map((v, i) => HOME[i] + (v - HOME[i]) * blend);
const mixPose = (a, b, m) => a.map((v, i) => v + (b[i] - v) * m);

/** A parameter vector -> the saved intent object. */

function intentOf(v, extra = {}) {
  const P = asObj(v);
  const t1 = r4(P.dReach), t2 = r4(t1 + P.dPre), t3 = r4(t2 + P.dVault), t4 = r4(t3 + P.dTuck);
  const t5 = r4(t4 + D_LAND), t6 = r4(t5 + D_SETTLE);
  // arm time: a fraction of the tuck leg measured from the vault frame, so it
  // tracks the launch when the launch timing moves. 1.0 == the tuck keyframe.
  const at = r4(Math.min(Math.max(t3 + P.atFrac * (t4 - t3), t3), t6));
  const B = blendedBase(mixPose(F4, F5, P.baseMix), P.blend).map(r6);
  return {
    name: 'servoed_landing_r5_60mm',
    family: 'Round 5: the round-3 beak-strut LAUNCH + a per-tick servoed landing (climb/servo.mjs)',
    keyframes: [{ t: t1, pose: F1 }, { t: t2, pose: F2 }, { t: t3, pose: F3 },
                { t: t4, pose: F4 }, { t: t5, pose: F5 }, { t: t6, pose: F6 }],
    blend: P.blend, gap: P.gap, side: P.side, approach: 0.1663,
    isolate: true, stepCount: 4,
    servo: {
      at,
      base: { hip: [B[2], B[11]], knee: [B[3], B[12]], ankle: [B[4], B[13]] },
      yawRoll: [B[0], B[1], B[9], B[10]],
      zTarget: P.zTarget, xFoot: P.xFoot, fz: P.fz, pitchRef: 0,
      kHipZ: P.kHipZ, kHipPitch: P.kHipPitch, kHipX: P.kHipX,
      kKneeZ: P.kKneeZ, kKneeFz: P.kKneeFz, kKneeX: P.kKneeX,
      kAnkPitch: P.kAnkPitch, kAnkFz: P.kAnkFz,
      rate: P.rate, span: P.span,
    },
    params: P,
  };
}
/** The same builder with the servo REMOVED — the launch alone. */
function intentNoServo(v) { const j = intentOf(v); delete j.servo; j.name = 'launch_only_r5'; return j; }

// ---------------------------------------------------------------- the ledger
const SEEN = new Map();     // sha256 -> full scoreRobust result
let nFull = 0, nScreen = 0, nRejected = 0;
const IMPROVEMENTS = [];
let BEST = null, BESTV = null;

async function evaluate(j, { label = null, force = false, vec = null } = {}) {
  const sha = intentHash(j);
  if (SEEN.has(sha)) { const r = SEEN.get(sha); return { ...r, cached: true }; }
  saveIntent(j, CAND);
  if (!force) {
    // SCREEN: 60 mm nominal + 60 mm slippery (robust.mjs's designated cheap path)
    nScreen++;
    const a = await scoreCell(CAND, { rise: RISE, drop: 0.120, fmul: 1.0 });
    const b = await scoreCell(CAND, { rise: RISE, drop: 0.140, fmul: 0.5 });
    const best2 = Math.max(a.reward, b.reward);
    if (!(a.crit.honest || b.crit.honest || best2 >= 5.5)) {
      nRejected++;
      return { screened: true, sha256: sha, move: sha.slice(0, 12),
               objective: -100 + (a.reward + b.reward) / 2,
               kCore: 0, kCoreStable: 0, kExt: 0, kExtStable: 0 };
    }
  }
  nFull++;
  const g = await scoreRobust(CAND, { rise: RISE });
  g.label = label;
  g.intent = JSON.parse(fs.readFileSync(CAND, 'utf8'));
  SEEN.set(sha, g);
  if (!BEST || g.objective > BEST.objective) {
    BEST = g; BESTV = vec;
    const line = `[${el().toFixed(0).padStart(4)}s] IMPROVEMENT ${g.move} obj=${g.objective.toFixed(4)} objCore=${g.objectiveCore.toFixed(4)} kCore=${g.kCore}/9 kCoreStable=${g.kCoreStable}/9 kExt=${g.kExt}/14 kExtStable=${g.kExtStable}/14 meanRew=${g.meanReward.toFixed(3)} minPenEp=${g.agg.minPenetrationEpisode_mm.toFixed(2)}mm${label ? '  <' + label + '>' : ''}`;
    console.log(line);
    IMPROVEMENTS.push({ t_s: +el().toFixed(1), sha256: g.sha256, move: g.move, label,
      objective: g.objective, objectiveCore: g.objectiveCore, kCore: g.kCore,
      kCoreStable: g.kCoreStable, kExt: g.kExt, kExtStable: g.kExtStable,
      meanReward: g.meanReward, minPenEpisode_mm: g.agg.minPenetrationEpisode_mm,
      params: JSON.parse(fs.readFileSync(CAND, 'utf8')).params || null });
    fs.writeFileSync(OUT + '/_r5_bestcand.json', fs.readFileSync(CAND));
  }
  return g;
}

// ============================================================ PHASE 0: checks
console.log('=== r5_servoland PHASE 0 — the builder, the warm start, the probes ===');
const warmHash = intentHash(JSON.parse(fs.readFileSync(OUT + '/best_r3_vault_60mm.json', 'utf8')));
const rebuilt = intentNoServo(clampv(MU0));
const rebuiltHash = intentHash(rebuilt);
console.log(`  warm start best_r3_vault_60mm.json sha256 ${warmHash.slice(0, 12)}`);
console.log(`  this file's builder at mu0, servo REMOVED, sha256 ${rebuiltHash.slice(0, 12)}  REPRODUCES=${rebuiltHash === warmHash}`);
const WARM = await scoreRobust(OUT + '/best_r3_vault_60mm.json', { rise: RISE });
SEEN.set(WARM.sha256, WARM); WARM.label = 'WARM START (round-3 vault, no servo)';
console.log(`  WARM ${WARM.move} obj=${WARM.objective.toFixed(4)} objCore=${WARM.objectiveCore.toFixed(4)} kCore=${WARM.kCore}/9 kCoreStable=${WARM.kCoreStable}/9 kExt=${WARM.kExt}/14 minPenEpisode=${WARM.agg.minPenetrationEpisode_mm.toFixed(2)}mm`);
BEST = WARM;
IMPROVEMENTS.push({ t_s: +el().toFixed(1), sha256: WARM.sha256, move: WARM.move,
  label: WARM.label, objective: WARM.objective, objectiveCore: WARM.objectiveCore,
  kCore: WARM.kCore, kCoreStable: WARM.kCoreStable, kExt: WARM.kExt, kExtStable: WARM.kExtStable,
  meanReward: WARM.meanReward, minPenEpisode_mm: WARM.agg.minPenetrationEpisode_mm, params: null });

// Named probes, all forced onto the full grid so the family has references.
const probe = (over, label) => {
  const v = clampv(MU0.slice());
  for (const [k, val] of Object.entries(over)) v[NAMES.indexOf(k)] = RND[NAMES.indexOf(k)](val);
  return { v: clampv(v), label };
};
const ZERO = { kHipZ: 0, kHipPitch: 0, kHipX: 0, kKneeZ: 0, kKneeFz: 0, kKneeX: 0, kAnkPitch: 0, kAnkFz: 0 };
const PROBES = [
  probe({ ...ZERO, baseMix: 1 },
        'P1 the throw held: base = the round-3 landing pose through the same blend, ZERO gains, armed at the tuck'),
  probe({ ...ZERO, baseMix: 0 },
        'P2 no landing at all: base = the TUCK pose, ZERO gains, armed at the tuck'),
  probe({},
        'P3 the reasoned law around the landing pose (kHipX -2, kKneeFz -4, kAnkFz +1)'),
  probe({ atFrac: 0.0 },
        'P4 the reasoned law armed at the VAULT frame'),
  probe({ kHipX: -4, kKneeFz: -8, kAnkFz: 2 },
        'P5 the reasoned law, doubled'),
  probe({ ...ZERO, kHipZ: 4, kKneeZ: -4 },
        'P6 trunk-height feedback only'),
  probe({ ...ZERO, baseMix: 1, rate: 0.5, span: 2.6 },
        'P7 the throw held, with the rate limit and span opened right up'),
  probe({ atFrac: 1.6 },
        'P8 the reasoned law armed midway between the tuck and the landing keyframe'),
  probe({ atFrac: 2.6 },
        'P9 the servo as a LATE STABILISER: the whole throw runs, the law arms just after the landing keyframe'),
  probe({ ...ZERO, atFrac: 2.6, baseMix: 1 },
        'P10 late arming, ZERO gains: what the mechanism alone costs once the throw has already landed'),
  probe({ atFrac: 2.6, kHipZ: 4, kKneeZ: -4, kAnkPitch: 1 },
        'P11 late stabiliser with trunk-height and pitch feedback'),
];
for (const p of PROBES) {
  const g = await evaluate(intentOf(p.v, {}), { label: p.label, force: true, vec: p.v });
  console.log(`  ${p.label}\n     ${g.move} obj=${g.objective.toFixed(4)} objCore=${g.objectiveCore.toFixed(4)} kCore=${g.kCore}/9 kCoreStable=${g.kCoreStable}/9 kExt=${g.kExt}/14 meanRew=${g.meanReward.toFixed(3)} armed=${g.cells[3].servo.armed} svTicks=${g.cells[3].servo.ticks} minPenEp=${g.agg.minPenetrationEpisode_mm.toFixed(2)}mm`);
}
console.log(`  phase 0 done at ${el().toFixed(0)} s`);

// ================================================================= PHASE 1 CEM
const NPOP = 22, NELITE = 6;
let mu = clampv(MU0.slice()), sig = SIG0.slice();
// gen-0 mean is the reasoned law; the probes above are its neighbours.
let gen = 0;
const GENLOG = [];
while (el() < BUDGET_S) {
  gen++;
  const pool = [];
  for (let n = 0; n < NPOP && el() < BUDGET_S; n++) {
    const v = clampv(mu.map((m, i) => m + sig[i] * gauss()));
    const g = await evaluate(intentOf(v), { label: `g${gen}n${n}`, vec: v });
    pool.push({ v, obj: g.objective, kCore: g.kCore, kCoreStable: g.kCoreStable, kExt: g.kExt, move: g.move });
  }
  if (!pool.length) break;
  // elitism: the best-so-far vector always competes (already scored, free)
  if (BESTV) pool.push({ v: BESTV, obj: BEST.objective, kCore: BEST.kCore, kCoreStable: BEST.kCoreStable, kExt: BEST.kExt, move: BEST.move });
  pool.sort((a, b) => b.obj - a.obj);
  const elite = pool.slice(0, Math.min(NELITE, pool.length));
  const newMu = mu.map((_, i) => elite.reduce((a, e) => a + e.v[i], 0) / elite.length);
  const newSig = mu.map((_, i) => {
    const m = newMu[i];
    const s = Math.sqrt(elite.reduce((a, e) => a + (e.v[i] - m) ** 2, 0) / elite.length);
    return Math.max(s, SIGFLOOR[i]);
  });
  mu = clampv(newMu); sig = newSig;
  const row = { gen, t_s: +el().toFixed(1), eliteBestObj: elite[0].obj, eliteBestMove: elite[0].move,
    eliteBestKCore: elite[0].kCore, eliteBestKCoreStable: elite[0].kCoreStable,
    eliteMeanObj: elite.reduce((a, e) => a + e.obj, 0) / elite.length,
    bestSoFar: { move: BEST.move, objective: BEST.objective, kCore: BEST.kCore, kCoreStable: BEST.kCoreStable },
    mu: asObj(mu) };
  GENLOG.push(row);
  console.log(`[${el().toFixed(0).padStart(4)}s] gen ${String(gen).padStart(2)}  eliteBest obj=${elite[0].obj.toFixed(3)} (${elite[0].move}, kCore ${elite[0].kCore}, kCoreStable ${elite[0].kCoreStable})  eliteMean=${row.eliteMeanObj.toFixed(3)}  best=${BEST.objective.toFixed(3)} kCoreStable=${BEST.kCoreStable}  full=${nFull} screened=${nScreen} rejected=${nRejected}`);
}

// ================================================================== PHASE 2
// THE TABLE. Every DISTINCT hash that reached the 14-cell grid, with its k's,
// its objective, and — for the bests — the per-cell verdicts and the
// whole-episode minimum penetration of every clear.
console.log(`\n=== r5_servoland PHASE 2 — the table (${SEEN.size} distinct hashes on the full grid) ===`);
const rows = [...SEEN.values()].sort((a, b) => b.objective - a.objective);
console.log('  move          kCore kCoreStable kExt kExtStable objective objectiveCore meanReward minPenEp_mm label');
for (const g of rows) {
  console.log(`  ${g.move}  ${String(g.kCore).padStart(2)}/9  ${String(g.kCoreStable).padStart(6)}/9  ${String(g.kExt).padStart(2)}/14 ${String(g.kExtStable).padStart(7)}/14 ${g.objective.toFixed(4).padStart(9)} ${g.objectiveCore.toFixed(4).padStart(13)} ${g.meanReward.toFixed(4).padStart(10)} ${g.agg.minPenetrationEpisode_mm.toFixed(2).padStart(11)}  ${g.label || ''}`);
}

// The BESTS OF THIS FAMILY are round-5 candidates (they carry an intent this
// file built); the warm start is a round-3 file and is reported beside them.
const r5rows = rows.filter(g => g.intent);
const bestObj = r5rows.length ? r5rows[0] : rows[0];
const bestK = (r5rows.length ? [...r5rows] : [...rows]).sort((a, b) => (b.kCoreStable - a.kCoreStable) || (b.objective - a.objective))[0];
const overallK = [...rows].sort((a, b) => (b.kCoreStable - a.kCoreStable) || (b.objective - a.objective))[0];
const KILL = overallK.kCoreStable >= 7 ? 'PASSED' : 'FAILED';
console.log(`\n  BEST BY OBJECTIVE  ${bestObj.move} obj=${bestObj.objective.toFixed(4)} kCore=${bestObj.kCore}/9 kCoreStable=${bestObj.kCoreStable}/9`);
console.log(`  BEST BY kCoreStable ${bestK.move} kCoreStable=${bestK.kCoreStable}/9 obj=${bestK.objective.toFixed(4)}`);
console.log(`  BEST kCoreStable OVERALL (family + warm start) ${overallK.move} = ${overallK.kCoreStable}/9  <${overallK.label || ''}>`);
console.log(`  KILL GATE (kCoreStable >= 7 of 9 at 60 mm): ${KILL}`);

const cellTable = g => {
  console.log(`  --- per-cell verdicts, ${g.move} (${g.label || ''}) ---`);
  for (const v of g.verdicts)
    console.log(`   [${v.tier}] rise=${String(v.rise_mm).padStart(2)} drop=${v.drop} f=${v.fmul} honest=${String(v.honest).padEnd(5)} stable=${String(v.stableClear).padEnd(5)} upTail=${String(v.uprightTailTicks).padStart(2)}/50 servoArmed=${v.servoArmed} svTicks=${v.servoTicks} penScore=${v.penetrationAtScore_mm}mm minPenEp=${v.minPenetrationEpisode_mm}mm@t${v.minPenetrationTick} rew=${v.reward} x=${v.x_mm} above=${v.above_mm} fot=${v.feetOnTread}/${v.feetOnTreadMax} peakZ=${v.peakZ_mm}`);
};
cellTable(bestObj);
if (bestK.sha256 !== bestObj.sha256) cellTable(bestK);
cellTable(WARM);

// ------------------------------------------------------------- save the bests
const saved = [];
const stamp = (g, file, note) => {
  if (!g.intent) return;
  const j = { ...g.intent };
  j.note = note;
  j.bounds = { blend: [0.7, 2.4], side: [-0.02, 0.09] };
  j.robust = {
    kCore: g.kCore, nCore: 9, kCoreStable: g.kCoreStable, kExt: g.kExt, nExt: 14,
    kExtStable: g.kExtStable, objective: g.objective, objectiveCore: g.objectiveCore,
    objectiveR3: g.objectiveR3, meanReward: g.meanReward, sha256: g.sha256,
    minPenetrationEpisode_mm: g.agg.minPenetrationEpisode_mm,
    minPenetrationAtScore_mm: g.agg.minPenetrationAtScore_mm,
    maxTq: g.agg.maxTq, verdicts: g.verdicts, agg: g.agg,
  };
  const path = OUT + '/' + file;
  if (fs.existsSync(path)) { console.log(`  REFUSING to overwrite existing ${file}`); return; }
  const check = intentHash(j);
  saveIntent(j, path);
  saved.push({ file, sha256: g.sha256, hashAfterStamp: check, unchanged: check === g.sha256 });
  console.log(`  saved ${file}  sha256 ${g.sha256}  (hash unchanged by the stamp: ${check === g.sha256})`);
};
stamp(bestObj, 'best_r5_servoland_60mm.json',
  `ROUND 5, the servoed landing at 60 mm: the round-3 beak-strut LAUNCH with its keyframe landing deleted and replaced by climb/servo.mjs's per-tick feedback law. Best of ${nFull} full 14-cell evaluations (${nScreen} screened, ${nRejected} rejected on the 2-cell screen) over ${gen} CEM generations. kCore ${bestObj.kCore}/9, kCoreStable ${bestObj.kCoreStable}/9, kExt ${bestObj.kExt}/14, objective ${bestObj.objective.toFixed(4)}. WARM START for comparison: kCore ${WARM.kCore}/9, kCoreStable ${WARM.kCoreStable}/9, objective ${WARM.objective.toFixed(4)}.`);
if (bestK.sha256 !== bestObj.sha256)
  stamp(bestK, 'best_r5_servoland_kstable_60mm.json',
    `ROUND 5, the servoed landing at 60 mm: the vector with the highest kCoreStable (${bestK.kCoreStable}/9), which is not the vector with the highest objective. objective ${bestK.objective.toFixed(4)}.`);

// ------------------------------------------------------------- results record
const results = {
  round: 5, family: 'servoland', rise_mm: 60, generatedAt: new Date().toISOString(),
  instrument: { scorer: 'climb/robust.mjs scoreRobust, 14-cell grid', law: 'climb/servo.mjs',
    objective: 'meanReward(14) + 4*kExtStable + 4*meanUprightCredit(14)' },
  budget: { wallClock_s: +el().toFixed(1), cemBudget_s: BUDGET_S, generations: gen,
    fullGridEvaluations: nFull, screened: nScreen, rejectedByScreen: nRejected,
    distinctHashesOnFullGrid: SEEN.size,
    screenRule: 'scoreCell at 60 mm on (drop 0.120, x1.0) and (drop 0.140, x0.5); pass if honest in either or max reward >= 5.5' },
  builderReproducesWarmStart: rebuiltHash === warmHash,
  warmStart: { file: 'climb/best_r3_vault_60mm.json', sha256: WARM.sha256, kCore: WARM.kCore,
    kCoreStable: WARM.kCoreStable, kExt: WARM.kExt, kExtStable: WARM.kExtStable,
    objective: WARM.objective, objectiveCore: WARM.objectiveCore, meanReward: WARM.meanReward,
    minPenetrationEpisode_mm: WARM.agg.minPenetrationEpisode_mm, verdicts: WARM.verdicts },
  killGate: { rule: 'kCoreStable >= 7 of 9 at 60 mm', bestKCoreStable: overallK.kCoreStable,
    bestKCoreStableMove: overallK.move, bestKCoreStableLabel: overallK.label,
    bestFamilyKCoreStable: bestK.kCoreStable,
    result: KILL,
    consequence: KILL === 'PASSED' ? 'the 40-80 mm band stays open'
      : 'THE 40-80 mm BAND IS CLOSED. The negative is the result.' },
  improvements: IMPROVEMENTS,
  generations: GENLOG,
  saved,
  table: rows.map(g => ({
    sha256: g.sha256, move: g.move, label: g.label,
    kCore: g.kCore, kCoreStable: g.kCoreStable, kExt: g.kExt, kExtStable: g.kExtStable,
    objective: g.objective, objectiveCore: g.objectiveCore, objectiveR3: g.objectiveR3,
    meanReward: g.meanReward, uprightTailFrac: g.uprightTailFrac,
    minPenetrationEpisode_mm: g.agg.minPenetrationEpisode_mm,
    minPenetrationAtScore_mm: g.agg.minPenetrationAtScore_mm,
    meanPeakGain_mm: g.agg.meanPeakGain * 1000, maxTq: g.agg.maxTq,
    servoArmedCells: g.cells.filter(c => c.servo && c.servo.armed).length,
    servoTicksMean: g.cells.reduce((a, c) => a + (c.servo ? c.servo.ticks : 0), 0) / g.cells.length,
    params: g.intent ? g.intent.params : null,
    servo: g.intent ? (g.intent.servo || null) : null,
    verdicts: g.verdicts,
  })),
};
fs.writeFileSync(OUT + '/r5_servoland-results.json', JSON.stringify(results, null, 2));
console.log(`\n  wrote climb/r5_servoland-results.json  (${el().toFixed(0)} s total)`);
