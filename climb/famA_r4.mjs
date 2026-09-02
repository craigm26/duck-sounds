// FAMILY A (round 4) — THE BEAK-STRUT VAULT, LANDING ON A MEASURED EVENT.
//
// THE MEASURED FAILURE. Round 3's vault is one ballistic pivot whose tuck/land
// segment fires at an AUTHORED TIME. Re-baselined this round it is 4 of 9 core
// at 60 mm (climb/best_r3_vault_60mm.json, sha 4b9110c4), 2 of 9 at 40 and 50,
// and its landing misses in the SAME DIRECTION every time: the rise axis alone
// costs 13 of 21 cells and the -10 mm rise clears 7 of 21 while +10 mm clears
// 1 of 21. That is not a lift problem (the trunk reaches 176-190 mm on the
// cells it clears). It is a PLACEMENT problem: a clock cannot know that this
// plant threw the body 10 mm shorter than the last one.
//
// THE CHANGE. The tuck/land tail now fires on a MEASURED EVENT, and the landing
// keyframe's leg targets are shifted by how far the trunk actually is from
// where the author expected it at that instant. The mechanism is climb/event.mjs
// and it is an OPTIONAL field of the intent, so a file without it replays as
// before. Genes: the event TYPE (discrete: beak contact / trunk pitch / trunk z),
// its threshold, an arming time, a fallback time, a post-event delay, the
// reference trunk x, and three per-joint feedback gains.
//
// THE LADDER IS FROZEN at 60 mm. No other rise is searched until a move reaches
// k >= 7 of 9 on the CORE grid at 60 mm.
//
// THE OBJECTIVE is climb/robust.mjs scoreRobust(..., {core:true}).objectiveCore
// = meanReward(9) + 4*kCoreStable + 4*meanUprightCredit(9). Nothing here
// re-implements a scorer, and nothing is scored that was not first written to
// disk and read back.
//
// PHASES, all in this ONE Node process:
//   P  parity: the event code is inert on every file that has no event. Both
//      harnesses (rig3.scoreSaved, robust.scoreCell) against climb/rig3_noevA
//      .mjs / climb/robust_noevA.mjs, which climb/mk_noevA_ref.py produced by
//      mechanically reverting EXACTLY the event code from the files as they
//      stand. (A byte copy would not do: these files were being edited by more
//      than one family in the same window, so a copy is not a control for MY
//      change alone.)
//   H  the move ids in climb/r4_audit-results.json are unchanged by adding
//      `event` to intentHash().
//   E  the quantisation cost of routing the round-3 move through the event
//      path with a degenerate event (fires at the round-3 tuck time, zero gain).
//   S  the CEM search at 60 mm.
//   F  the winner re-scored on the full 14-cell extended grid.
//
// Seed base 4000, mulberry32.
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/famA_r4.mjs [searchSeconds]
import fs from 'node:fs';
import {
  scoreRobust, scoreCell, intentHashOfFile, checkBounds, saveIntent,
  CLEAR_BONUS, UPRIGHT_BONUS, UPRIGHT_TAIL_MIN, PLANTS, DHS, DECLARED_BOUNDS, HOME,
} from './robust.mjs';
import { scoreSaved as newSaved } from './rig3.mjs';
import { scoreSaved as refSaved } from './rig3_noevA.mjs';
import { scoreCell as refCell } from './robust_noevA.mjs';

const P = '../climb/';
const SCRATCH = '/tmp/claude-1000/-home-craigm26/43f07dce-e62f-464d-ae1f-2fe020620950/scratchpad';
fs.mkdirSync(SCRATCH, { recursive: true });
const T0 = Date.now();
const el = () => (Date.now() - T0) / 1000;
const LOGLINES = [];
const log = s => { const l = `[${el().toFixed(0).padStart(5)}s] ${s}`; console.log(l); LOGLINES.push(l); };
const SEARCH_SECONDS = +(process.argv[2] || 2280);          // 38 min of search
const OUT = {
  generated: new Date().toISOString(), family: 'A', round: 4,
  what: 'beak-strut vault whose tuck/land tail fires on a MEASURED EVENT',
  seedBase: 4000, searchSeconds: SEARCH_SECONDS,
  objective: 'robust.mjs scoreRobust(core:true).objectiveCore = meanReward(9) + 4*kCoreStable + 4*meanUprightCredit(9)',
  ladder: 'frozen at 60 mm until k>=7 of 9 core',
};

// ------------------------------------------------------------------ track
const J = { lhy: 0, lhr: 1, lhp: 2, lk: 3, la: 4, np: 5, hp: 6, hy: 7, hr: 8,
            rhy: 9, rhr: 10, rhp: 11, rk: 12, ra: 13 };
const put = (q, hip, knee, ank, roll) => {
  q[J.lhp] = HOME[J.lhp] + hip;  q[J.rhp] = HOME[J.rhp] - hip;
  q[J.lk] = HOME[J.lk] + knee;   q[J.rk] = HOME[J.rk] - knee;
  q[J.la] = HOME[J.la] + ank;    q[J.ra] = HOME[J.ra] - ank;
  q[J.lhr] = HOME[J.lhr] + roll; q[J.rhr] = HOME[J.rhr] + roll;
  return q;
};
/** climb/vault.mjs trackOf(), verbatim — the round-3 six-keyframe vault. */
function baseTrack(p) {
  const strut = q => { q[J.np] = p.strutNeck; q[J.hp] = p.strutHead; return q; };
  const A = strut(put(HOME.slice(), p.crouchHip, p.crouchKnee, p.crouchAnk, p.roll));
  const B = strut(put(HOME.slice(), p.preHip, p.preKnee, p.preAnk, p.roll));
  const Cc = strut(put(HOME.slice(), p.vaultHip, p.vaultKnee, p.vaultAnk, p.roll));
  const Dd = strut(put(HOME.slice(), p.tuckHip, p.tuckKnee, p.tuckAnk, p.roll));
  const E = put(HOME.slice(), p.landHip, p.landKnee, p.landAnk, 0);
  E[J.np] = p.landNeck; E[J.hp] = p.landHead;
  const t1 = p.tReach, t2 = t1 + p.tPre, t3 = t2 + p.tVault, t4 = t3 + p.tTuck, t5 = t4 + p.tLand;
  return { t1, t2, t3, t4, t5, A, B, C: Cc, D: Dd, E,
           kf: [{ t: t1, pose: A }, { t: t2, pose: B }, { t: t3, pose: Cc },
                { t: t4, pose: Dd }, { t: t5, pose: E }, { t: t5 + 0.7, pose: HOME.slice() }] };
}
/**
 * THE EVENT. post = [tuck, land, home] measured from (fire + delay), so a move
 * whose event fires exactly at t3 with delay 0 and zero gains replays the
 * round-3 track. `adapt` is non-zero only on the LANDING keyframe's six leg
 * joints, mirrored the way put() mirrors them: that is "the landing keyframe's
 * foot targets are relative to the trunk at the event".
 */
const EV_TYPES = ['beak', 'pitch', 'trunkZ'];
function eventOf(p, T) {
  const type = EV_TYPES[Math.min(2, Math.floor(p.evType))];
  const threshold = type === 'beak' ? p.evThreshBeak : type === 'pitch' ? p.evThreshPitch : p.evThreshZ;
  const adapt = new Array(14).fill(0);
  adapt[J.lhp] = p.kHip;  adapt[J.rhp] = -p.kHip;
  adapt[J.lk] = p.kKnee;  adapt[J.rk] = -p.kKnee;
  adapt[J.la] = p.kAnk;   adapt[J.ra] = -p.kAnk;
  return {
    type, threshold: +threshold.toFixed(6),
    arm: +(T.t1 + p.evArmFrac * (T.t3 - T.t1)).toFixed(4),
    fallback: +(T.t3 + p.evFallExtra).toFixed(4),
    delay: +p.evDelay.toFixed(4),
    refX: +p.evRefX.toFixed(4),
    clamp: 0.12,
    post: [
      { dt: +p.tTuck.toFixed(4), pose: T.D.map(r5) },
      { dt: +(p.tTuck + p.tLand).toFixed(4), pose: T.E.map(r5), adapt: adapt.map(v => +v.toFixed(5)) },
      { dt: +(p.tTuck + p.tLand + 0.7).toFixed(4), pose: HOME.map(r5) },
    ],
  };
}
const r5 = v => +v.toFixed(5);

const BOUNDS = {
  // round-3 family A / climb/vault.mjs, unchanged
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
  // ROUND 4: the event
  evType: [0, 2.999], evThreshBeak: [0.0005, 0.030], evThreshPitch: [-0.95, 0.95], evThreshZ: [-0.03, 0.10],
  evArmFrac: [0, 1], evFallExtra: [0, 0.35], evDelay: [0, 0.30], evRefX: [0.08, 0.32],
  kHip: [-3.0, 3.0], kKnee: [-3.0, 3.0], kAnk: [-3.0, 3.0],
};
// gap/side/approach/blend are also held to robust.mjs DECLARED_BOUNDS at scoring
// time; assert here that this search box cannot leave it (the family-C breach).
for (const [k, [lo, hi]] of Object.entries(DECLARED_BOUNDS)) {
  if (!BOUNDS[k]) throw new Error('no search bound for declared ' + k);
  if (BOUNDS[k][0] < lo || BOUNDS[k][1] > hi)
    throw new Error(`search box for ${k} [${BOUNDS[k]}] leaves declared [${lo}, ${hi}]`);
}
const KEYS = Object.keys(BOUNDS);
const RANGE = Object.fromEntries(KEYS.map(k => [k, BOUNDS[k][1] - BOUNDS[k][0]]));

// ------------------------------------------------------------------ rng
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const SEED_BASE = 4000;
let RND = mulberry32(SEED_BASE);
function gauss() { let u = 0, v = 0; while (!u) u = RND(); while (!v) v = RND(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
const clampB = (k, v) => Math.min(BOUNDS[k][1], Math.max(BOUNDS[k][0], v));
const randP = () => Object.fromEntries(KEYS.map(k => [k, BOUNDS[k][0] + RND() * RANGE[k]]));
const sampleN = (mu, sg) => Object.fromEntries(KEYS.map(k => [k, clampB(k, mu[k] + sg[k] * gauss())]));

// ------------------------------------------------------------------ intent
function intentOf(p, rise, note) {
  const T = baseTrack(p);
  return {
    name: `beak_strut_vault_event_r4_${Math.round(rise * 1000)}mm`,
    family: 'A beak-strut vault, event-triggered landing (round 4)',
    keyframes: T.kf.map(f => ({ t: +f.t.toFixed(4), pose: f.pose.map(r5) })),
    blend: +p.blend.toFixed(4), gap: +p.gap.toFixed(4), side: +p.side.toFixed(4),
    approach: +p.approach.toFixed(4), isolate: true, stepCount: 4,
    event: eventOf(p, T),
    bounds: { blend: BOUNDS.blend.slice(), side: BOUNDS.side.slice() },
    params: p, note,
  };
}

// ------------------------------------------------------------------ eval
let EVALS = 0, FULL = 0, SCREENED = 0;
const RISE = 0.060;
const SLIP = PLANTS[1];                                   // drop 0.130, friction x0.7
const c01 = v => Math.max(0, Math.min(1, v));
/**
 * TIER-2 RANK for candidates the screen stopped. NOT an objective, never
 * compared against one: screened candidates only ever fill the elite set BELOW
 * every fully evaluated one. It is the round-3 tier2, kept because the reason
 * still holds — rig3's reward alone ranks a duck standing still on the floor
 * (~7) above a vault that plants the beak and misses the landing.
 */
const tier2 = c => c.reward + 3 * c.feetOnTreadMax + 2 * c01(c.headFrac / 0.30) + 4 * c01((c.maxX - 0.12) / 0.15);
/**
 * THE SCREEN is run ON THE SLIPPERY CELL (drop 0.130, friction x0.7) at the
 * nominal rise — the cell round 3 cleared 1 of 21 times and the cell this
 * family is asked to tune on. A candidate is stopped only when it got NOWHERE
 * on it: the trunk never crossed the riser line AND no foot ever rested on a
 * tread. The round-3 incumbent passes that (its slippery cell reaches 2 feet on
 * the tread before toppling), so the screen cannot cut the basin we start in;
 * it cuts the uniform explorers that fall over in front of the step.
 */
async function evalP(p, rise, { screen = true } = {}) {
  const path = `${SCRATCH}/famA_r4_cand.json`;
  const intent = intentOf(p, rise, 'candidate');
  fs.writeFileSync(path, JSON.stringify(intent, null, 2));
  EVALS++;
  if (screen) {
    const c = await scoreCell(path, { rise, dh: 0, drop: SLIP.drop, fmul: SLIP.fmul, isolate: true });
    if (c.invalid) throw new Error('candidate out of declared bounds — the search left its own box');
    if (!(c.maxX > 0.12) && c.feetOnTreadMax === 0) {
      SCREENED++;
      return { screened: true, rank2: tier2(c), objective: -1e6 + tier2(c), k: 0,
               screenReward: c.reward, maxX: c.maxX, feetMax: c.feetOnTreadMax };
    }
  }
  const r = await scoreRobust(path, { rise, isolate: true, core: true });
  if (r.invalid) throw new Error('candidate out of declared bounds — the search left its own box');
  FULL++;
  return { screened: false, k: r.kCore, kStable: r.kCoreStable, objective: r.objectiveCore,
           meanReward: r.meanRewardCore, upCredit: r.uprightTailFracCore,
           verdicts: r.verdicts, agg: r.agg, sha256: r.sha256 };
}

/** Which axis is killing the failing cells. */
function blame(verdicts, rise) {
  if (!verdicts) return 'screened';
  const byRise = new Map(), byPlant = new Map();
  for (const v of verdicts) {
    const dh = v.rise_mm - Math.round(rise * 1000);
    const rk = `${dh >= 0 ? '+' : ''}${dh}`;
    const pk = `d${v.drop.toFixed(3)}/f${v.fmul.toFixed(1)}`;
    if (!byRise.has(rk)) byRise.set(rk, []); byRise.get(rk).push(v.honest);
    if (!byPlant.has(pk)) byPlant.set(pk, []); byPlant.get(pk).push(v.honest);
  }
  const fmt = m => [...m.entries()].map(([k, v]) => `${k}:${v.filter(Boolean).length}/${v.length}`).join(' ');
  const fired = verdicts.filter(v => v.eventFired).length;
  return `rise[${fmt(byRise)}] plant[${fmt(byPlant)}] evFired ${fired}/${verdicts.length}`;
}

// ================================================================== PHASE P
const PARITY_FILES = [
  'best_r3_vault_40mm.json', 'best_r3_vault_50mm.json', 'best_r3_vault_60mm.json',
  'best_r3_vault_70mm.json', 'best_r2_vault_40mm.json', 'best_r2_vault_60mm.json',
  'best_r3_landvault_80mm.json', 'best_r3_landvault_90mm.json',
  'ctrl_do_nothing.json', 'r4_ctrl_on_tread_60mm.json',
];
const riseOfFile = f => { const m = f.match(/_(\d+)mm/); return m ? +m[1] / 1000 : 0.060; };
log('=== PHASE P — the event code is INERT on every file that carries no event ===');
log('    reference: climb/rig3_noevA.mjs + climb/robust_noevA.mjs (mk_noevA_ref.py:');
log('    the current shared files with EXACTLY the event code reverted)');
OUT.parity = [];
let parityAll = true;
for (const f of PARITY_FILES) {
  const h = riseOfFile(f);
  for (const tail of ['policy', 'hold', 'none']) {
    const A = await newSaved(P + f, { rise: h, tail });
    const B = await refSaved(P + f, { rise: h, tail });
    const same = A.scored.x === B.scored.x && A.scored.z === B.scored.z && A.scored.y === B.scored.y
      && A.scored.feetOnTread === B.scored.feetOnTread && A.reward === B.reward
      && A.crit.honest === B.crit.honest && A.maxAbsDY === B.maxAbsDY
      && A.uprightTailTicks === B.uprightTailTicks
      && A.penetrationAtScore === B.penetrationAtScore;
    parityAll = parityAll && same;
    OUT.parity.push({ file: f, rise_mm: Math.round(h * 1000), tail, harness: 'rig3.scoreSaved',
      exact: same, x: A.scored.x, xRef: B.scored.x, z: A.scored.z, zRef: B.scored.z,
      reward: A.reward, rewardRef: B.reward, honest: A.crit.honest, event: A.event });
    if (!same) log(`  MISMATCH ${f} tail=${tail}  x ${A.scored.x} vs ${B.scored.x}  z ${A.scored.z} vs ${B.scored.z}`);
  }
  const c1 = await scoreCell(P + f, { rise: h, dh: 0, drop: 0.120, fmul: 1.0, isolate: true });
  const c0 = await refCell(P + f, { rise: h, dh: 0, drop: 0.120, fmul: 1.0, isolate: true });
  const same = c1.scored.x === c0.scored.x && c1.scored.z === c0.scored.z && c1.reward === c0.reward
    && c1.crit.honest === c0.crit.honest && c1.uprightTailTicks === c0.uprightTailTicks;
  parityAll = parityAll && same;
  OUT.parity.push({ file: f, rise_mm: Math.round(h * 1000), tail: 'policy', harness: 'robust.scoreCell(cell0)',
    exact: same, x: c1.scored.x, xRef: c0.scored.x, z: c1.scored.z, zRef: c0.scored.z,
    reward: c1.reward, rewardRef: c0.reward, honest: c1.crit.honest, event: c1.event });
  log(`  ${f.padEnd(32)} 3 tails + cell0 : ${same ? 'EXACT' : 'MISMATCH'}  (x=${c1.scored.x.toFixed(6)} rew=${c1.reward.toFixed(4)})`);
}
OUT.parityAll = parityAll;
OUT.parityRows = OUT.parity.length;
log(`  PARITY ${parityAll ? 'PASS' : 'FAIL'} — ${OUT.parity.length} rows, full float digits, ${PARITY_FILES.length} files x (3 tails + cell 0)`);

// ================================================================== PHASE H
log('=== PHASE H — adding `event` to intentHash() changes NO existing move id ===');
const published = JSON.parse(fs.readFileSync(P + 'r4_audit-results.json', 'utf8')).hashes || [];
OUT.hashes = [];
let hashAll = true;
for (const row of published) {
  let now = null;
  try { now = intentHashOfFile(P + row.file); } catch (e) { now = 'ERR:' + e.message; }
  const same = now === row.sha256;
  hashAll = hashAll && same;
  OUT.hashes.push({ file: row.file, published: row.sha256, now, same });
}
OUT.hashesUnchanged = hashAll;
log(`  ${published.length} published move ids re-hashed: ${hashAll ? 'ALL UNCHANGED' : 'CHANGED — STOP'}`);
if (!hashAll) for (const r of OUT.hashes.filter(r => !r.same)) log(`   CHANGED ${r.file} ${r.published} -> ${r.now}`);

// ================================================================== PHASE E
// Route the round-3 60 mm move through the EVENT path with a DEGENERATE event
// (trunkZ, unreachable threshold, so it can only fire at the fallback; zero
// gains). Two versions, because the difference between them is the single most
// useful number this family measured:
//   E1 ALIGNED. The event fires on the first 20 ms tick at or after the
//      round-3 vault time, and the post-event segment durations are measured
//      FROM THAT TICK (dt = t4 - tFire, t5 - tFire, ...). The rebuilt track is
//      then the round-3 track keyframe for keyframe, and the move must score
//      exactly what it scores through the timed path.
//   E2 UNALIGNED. The same event, but the durations are the round-3 segment
//      durations measured from the round-3 vault time (dt = tTuck, ...), so
//      every post keyframe lands ONE TICK (5.7 ms) late.
log('=== PHASE E — the round-3 move re-expressed as an event (degenerate) ===');
{
  const j = JSON.parse(fs.readFileSync(P + 'best_r3_vault_60mm.json', 'utf8'));
  const pp = j.params, kf = j.keyframes;
  const t3 = pp.tReach + pp.tPre + pp.tVault;
  const DT = 0.02;
  const tFire = Math.ceil(t3 / DT - 1e-9) * DT;             // the tick the fallback lands on
  const mk = (dts, tag) => {
    const d = JSON.parse(JSON.stringify(j));
    d.event = { type: 'trunkZ', threshold: 9.99, arm: 0, fallback: +t3.toFixed(4), delay: 0,
                refX: 0.2, clamp: 0.12,
                post: [{ dt: +dts[0].toFixed(4), pose: kf[3].pose },
                       { dt: +dts[1].toFixed(4), pose: kf[4].pose },
                       { dt: +dts[2].toFixed(4), pose: kf[5].pose }] };
    const p = `${SCRATCH}/famA_r4_degen_${tag}.json`;
    fs.writeFileSync(p, JSON.stringify(d, null, 2));
    return p;
  };
  const A = await scoreRobust(P + 'best_r3_vault_60mm.json', { rise: RISE, isolate: true, core: true });
  const E1 = await scoreRobust(mk([kf[3].t - tFire, kf[4].t - tFire, kf[5].t - tFire], 'aligned'),
                               { rise: RISE, isolate: true, core: true });
  const E2 = await scoreRobust(mk([pp.tTuck, pp.tTuck + pp.tLand, pp.tTuck + pp.tLand + 0.7], 'unaligned'),
                               { rise: RISE, isolate: true, core: true });
  const row = (r) => ({ kCore: r.kCore, kCoreStable: r.kCoreStable,
                        objectiveCore: +r.objectiveCore.toFixed(4), meanReward: +r.meanRewardCore.toFixed(4),
                        honestPattern: r.verdicts.map(v => v.honest ? 1 : 0).join(''), sha256: r.sha256 });
  const dmax = (B) => ({
    dx_mm: Math.max(...A.verdicts.map((v, i) => Math.abs(B.verdicts[i].x_mm - v.x_mm))),
    dz_mm: Math.max(...A.verdicts.map((v, i) => Math.abs(B.verdicts[i].z_mm - v.z_mm))),
  });
  OUT.degenerate = {
    tFire_s: +tFire.toFixed(4), t3_s: +t3.toFixed(4), tickShift_ms: +((tFire - t3) * 1000).toFixed(2),
    timed: row(A), alignedEvent: row(E1), unalignedEvent: row(E2),
    alignedDelta: dmax(E1), unalignedDelta: dmax(E2),
    alignedExact: row(E1).honestPattern === row(A).honestPattern && dmax(E1).dx_mm === 0 && dmax(E1).dz_mm === 0,
    finding: 'the round-3 vault\'s 4-of-9 is knife-edge in TIME to within one 20 ms control tick',
  };
  log(`  timed            kCore=${A.kCore}/9 objCore=${A.objectiveCore.toFixed(3)} honest=${row(A).honestPattern}`);
  log(`  event ALIGNED    kCore=${E1.kCore}/9 objCore=${E1.objectiveCore.toFixed(3)} honest=${row(E1).honestPattern} ` +
      `max|dx|=${dmax(E1).dx_mm.toFixed(2)}mm max|dz|=${dmax(E1).dz_mm.toFixed(2)}mm  fires at ${E1.verdicts[0].eventT}s`);
  log(`  event UNALIGNED  kCore=${E2.kCore}/9 objCore=${E2.objectiveCore.toFixed(3)} honest=${row(E2).honestPattern} ` +
      `max|dx|=${dmax(E2).dx_mm.toFixed(2)}mm max|dz|=${dmax(E2).dz_mm.toFixed(2)}mm  (every post keyframe ${((tFire - t3) * 1000).toFixed(1)} ms late)`);
  OUT.degenerate.alignedTTuck = +(kf[3].t - tFire).toFixed(4);
}

// ================================================================== PHASE S
const POP = 24, ELITE = 5;
const SIG0 = 0.055, SIGMIN = 0.020;
// Two gene groups need a wider proposal than the pose genes. evType is a
// DISCRETE gene carried as a continuous one (floor -> beak/pitch/trunkZ): at
// 0.055 of its range the proposal can never leave the branch it starts in, so
// no type but the seed's would ever be tried. The three feedback gains start at
// exactly 0 (the seed is the round-3 move, which has no feedback), and a
// proposal of 0.055*6 = 0.33 rad/m moves the landing by 0.04 rad over the full
// +/-0.12 m of shortfall — below the resolution of the thing being fixed.
const SIG_MUL = { evType: 6.4, kHip: 2.7, kKnee: 2.7, kAnk: 2.7 };
const SIGMIN_MUL = { evType: 10.0, kHip: 2.0, kKnee: 2.0, kAnk: 2.0 };
const RANKW = [5, 4, 3, 2, 1];
const HIST = [];

function saveBest(best, rise, tag) {
  const mmr = Math.round(rise * 1000);
  const obj = intentOf(best.p, rise,
    `round-4 family A beak-strut vault with an EVENT-TRIGGERED landing at ${mmr} mm; ` +
    `cleared ${best.r.k} of 9 core (${best.r.kStable} stable); objectiveCore ${best.r.objective.toFixed(4)}; ` +
    `meanReward ${best.r.meanReward.toFixed(4)}`);
  obj.robust = { kCore: best.r.k, nCore: 9, kCoreStable: best.r.kStable,
                 objectiveCore: +best.r.objective.toFixed(4), meanRewardCore: +best.r.meanReward.toFixed(4),
                 uprightCreditCore: +best.r.upCredit.toFixed(4),
                 clearBonus: CLEAR_BONUS, uprightBonus: UPRIGHT_BONUS, uprightTailMin: UPRIGHT_TAIL_MIN,
                 verdicts: best.r.verdicts, agg: best.r.agg };
  const path = `${P}best_r4_famA_${mmr}mm.json`;
  fs.writeFileSync(path, JSON.stringify(obj, null, 2));
  return path;
}

async function cem(rise, mu0, deadline, tag, sigScale = 1) {
  let mu = { ...mu0 };
  let sg = Object.fromEntries(KEYS.map(k => [k, SIG0 * (SIG_MUL[k] || 1) * sigScale * RANGE[k]]));
  const sgFloor = Object.fromEntries(KEYS.map(k => [k, SIGMIN * (SIGMIN_MUL[k] || 1) * RANGE[k]]));
  const seed = await evalP(mu, rise, { screen: false });
  let best = { p: { ...mu }, r: seed };
  log(`${tag} warm-start: k=${seed.k}/9 (stable ${seed.kStable}) objCore=${seed.objective.toFixed(3)} meanR=${seed.meanReward.toFixed(3)} | ${blame(seed.verdicts, rise)}`);
  HIST.push({ tag, gen: 0, kind: 'warm-start', k: seed.k, kStable: seed.kStable,
              objectiveCore: +seed.objective.toFixed(4), meanReward: +seed.meanReward.toFixed(4),
              blame: blame(seed.verdicts, rise) });
  saveBest(best, rise, tag);
  let gen = 0;
  while (Date.now() < deadline) {
    gen++;
    const scored = [];
    for (let i = 0; i < POP; i++) {
      if (Date.now() >= deadline) break;
      const p = (i === POP - 1) ? randP() : sampleN(mu, sg);
      const r = await evalP(p, rise);
      scored.push({ p, r });
      if (!r.screened && r.objective > best.r.objective) {
        best = { p, r };
        const bl = blame(r.verdicts, rise);
        log(`${tag} gen${gen} IMPROVE k=${r.k}/9 (stable ${r.kStable}) objCore=${r.objective.toFixed(3)} meanR=${r.meanReward.toFixed(3)} | ${bl}`);
        HIST.push({ tag, gen, kind: 'improve', k: r.k, kStable: r.kStable,
                    objectiveCore: +r.objective.toFixed(4), meanReward: +r.meanReward.toFixed(4),
                    upCredit: +r.upCredit.toFixed(4), blame: bl, elapsed_s: +el().toFixed(0),
                    event: { type: EV_TYPES[Math.min(2, Math.floor(p.evType))],
                             fired: r.verdicts.filter(v => v.eventFired).length } });
        saveBest(best, rise, tag);
        if (best.r.k >= 7) { log(`${tag} REACHED k=${best.r.k}/9 core`); return { best, gen, hit7: true }; }
      }
    }
    if (!scored.length) break;
    // full evaluations always outrank screened ones
    scored.sort((a, b) => (a.r.screened === b.r.screened)
      ? (a.r.screened ? b.r.rank2 - a.r.rank2 : b.r.objective - a.r.objective)
      : (a.r.screened ? 1 : -1));
    const elite = [{ p: best.p, r: best.r }, ...scored.slice(0, ELITE)];
    const w = [RANKW[0] + 1, ...RANKW.slice(0, elite.length - 1)];
    const wsum = w.reduce((a, b) => a + b, 0);
    const newMu = {}, newSg = {};
    for (const k of KEYS) {
      newMu[k] = elite.reduce((a, e, i) => a + w[i] * e.p[k], 0) / wsum;
      const varr = elite.reduce((a, e, i) => a + w[i] * (e.p[k] - newMu[k]) ** 2, 0) / wsum;
      newSg[k] = Math.max(sgFloor[k], Math.sqrt(varr));
    }
    mu = newMu; sg = newSg;
    const fullN = scored.filter(s => !s.r.screened).length;
    log(`${tag} gen${gen} done: ${scored.length} cand (${fullN} full, ${scored.length - fullN} screened) best k=${best.r.k}/9 objCore=${best.r.objective.toFixed(3)}  [${EVALS} evals, ${FULL} full]`);
  }
  return { best, gen, hit7: best.r.k >= 7 };
}

// ------------------------------------------------------ the warm-start ensemble
// PHASE E measured that the round-3 move's 4 of 9 survives a 5.7 ms shift of its
// tuck as 1 of 9. A single warm start is therefore a coin toss: it lands wherever
// the tick grid puts it. So the CEM is seeded from the BEST of an ensemble of
// event parametrisations of the same round-3 vector, each fully scored on the
// core 9 before the search starts.
//
// evArmFrac = 1 makes the event a CLOCK: arm == fallback == t3, so nothing can
// fire before the fallback tick whatever the type or threshold says. That is the
// member that reproduces round 3. The rest arm early and fire on a real
// measurement. tTuck is ALIGNED per member (t4 - the tick the fallback lands on)
// so the tuck ends where round 3 ended it.
const seedJ = JSON.parse(fs.readFileSync(P + 'best_r3_vault_60mm.json', 'utf8'));
const SP = seedJ.params;
const T3 = SP.tReach + SP.tPre + SP.tVault, T4 = T3 + SP.tTuck;
const tickAt = t => Math.ceil(t / 0.02 - 1e-9) * 0.02;
const member = (name, over) => {
  const fallExtra = over.evFallExtra || 0;
  const aligned = over.align === false ? SP.tTuck : T4 - tickAt(T3 + fallExtra);
  const p = { ...SP, tTuck: aligned,
    evType: 2.5, evThreshBeak: 0.003, evThreshPitch: 0.30, evThreshZ: 0.05,
    evArmFrac: 0.35, evFallExtra: fallExtra, evDelay: 0, evRefX: 0.185,
    kHip: 0, kKnee: 0, kAnk: 0, ...over };
  delete p.align;
  for (const k of KEYS) p[k] = clampB(k, p[k]);
  return { name, p };
};
const ENSEMBLE = [
  member('clock, tuck aligned to the fallback tick', { evArmFrac: 1.0 }),
  member('clock, round-3 tuck (one tick late)', { evArmFrac: 1.0, align: false }),
  member('clock, fallback +1 tick', { evArmFrac: 1.0, evFallExtra: 0.02 }),
  member('clock, fallback +2 ticks', { evArmFrac: 1.0, evFallExtra: 0.04 }),
  member('beak contact < 3 mm', { evType: 0.5, evThreshBeak: 0.003 }),
  member('beak contact < 10 mm', { evType: 0.5, evThreshBeak: 0.010 }),
  member('beak contact < 20 mm', { evType: 0.5, evThreshBeak: 0.020 }),
  member('trunk pitch >= 0.30', { evType: 1.5, evThreshPitch: 0.30 }),
  member('trunk pitch >= 0.60', { evType: 1.5, evThreshPitch: 0.60 }),
  member('trunk z >= h + 50 mm', { evType: 2.5, evThreshZ: 0.05 }),
  member('trunk z >= h + 80 mm', { evType: 2.5, evThreshZ: 0.08 }),
];
log(`=== PHASE W — warm-start ensemble, ${ENSEMBLE.length} event parametrisations of the round-3 60 mm vector ===`);
OUT.ensemble = [];
let MU0 = null, bestSeed = null;
for (const m of ENSEMBLE) {
  const r = await evalP(m.p, RISE, { screen: false });
  const bl = blame(r.verdicts, RISE);
  OUT.ensemble.push({ name: m.name, kCore: r.k, kCoreStable: r.kStable,
                      objectiveCore: +r.objective.toFixed(4), meanReward: +r.meanReward.toFixed(4),
                      firesAt_s: r.verdicts[0].eventT, eventFiredCells: r.verdicts.filter(v => v.eventFired).length,
                      honestPattern: r.verdicts.map(v => v.honest ? 1 : 0).join(''), sha256: r.sha256, blame: bl });
  log(`  ${m.name.padEnd(42)} k=${r.k}/9 (stable ${r.kStable}) objCore=${r.objective.toFixed(3)} fires@${r.verdicts[0].eventT}s honest=${r.verdicts.map(v => v.honest ? 1 : 0).join('')}`);
  if (!bestSeed || r.objective > bestSeed.objective) { bestSeed = r; MU0 = { ...m.p }; OUT.ensembleBest = m.name; }
}
log(`  seeding the CEM from: ${OUT.ensembleBest} (objCore ${bestSeed.objective.toFixed(3)}, k=${bestSeed.k}/9)`);

log(`=== PHASE S — CEM at 60 mm, pop ${POP}, elite ${ELITE}, sig0 ${SIG0}, floor ${SIGMIN}, budget ${SEARCH_SECONDS}s ===`);
const deadline = Date.now() + SEARCH_SECONDS * 1000;
const run = await cem(RISE, MU0, deadline, '60mm');
OUT.search = { rise_mm: 60, generations: run.gen, evals: EVALS, fullEvals: FULL, screened: SCREENED,
               pop: POP, elite: ELITE, sig0: SIG0, sigMin: SIGMIN, history: HIST, reached7: run.hit7 };

// ================================================================== PHASE F
const bestPath = saveBest(run.best, RISE, 'final');
log(`=== PHASE F — the winner on the FULL 14-cell extended grid ===`);
const F = await scoreRobust(bestPath, { rise: RISE, isolate: true });
OUT.final = {
  file: bestPath, sha256: F.sha256, move: F.move, rise_mm: 60,
  kCore: F.kCore, nCore: F.nCore, kCoreStable: F.kCoreStable,
  kExt: F.kExt, nExt: F.nExt, kExtStable: F.kExtStable,
  meanReward: +F.meanReward.toFixed(4), meanRewardCore: +F.meanRewardCore.toFixed(4),
  objective: +F.objective.toFixed(4), objectiveCore: +F.objectiveCore.toFixed(4),
  objectiveR3: +F.objectiveR3.toFixed(4),
  uprightTailFrac: +F.uprightTailFrac.toFixed(4), reachedFlightCells: F.reachedFlightCells,
  boundViolations: F.boundViolations, bounds: F.bounds,
  agg: F.agg, verdicts: F.verdicts,
};
// also re-score the round-3 incumbent on the same 14 cells, in the same process
const R3 = await scoreRobust(P + 'best_r3_vault_60mm.json', { rise: RISE, isolate: true });
OUT.incumbent = { file: 'best_r3_vault_60mm.json', sha256: R3.sha256, move: R3.move,
  kCore: R3.kCore, kCoreStable: R3.kCoreStable, kExt: R3.kExt, kExtStable: R3.kExtStable,
  objective: +R3.objective.toFixed(4), objectiveCore: +R3.objectiveCore.toFixed(4),
  objectiveR3: +R3.objectiveR3.toFixed(4), meanReward: +R3.meanReward.toFixed(4) };
log(`  best_r4_famA_60mm: kCore ${F.kCore}/${F.nCore} (stable ${F.kCoreStable}) kExt ${F.kExt}/${F.nExt} (stable ${F.kExtStable})`);
log(`     objective ${F.objective.toFixed(3)}  objectiveCore ${F.objectiveCore.toFixed(3)}  objectiveR3 ${F.objectiveR3.toFixed(3)}  move ${F.move}`);
log(`  incumbent r3 : kCore ${R3.kCore}/9 (stable ${R3.kCoreStable}) kExt ${R3.kExt}/14  objective ${R3.objective.toFixed(3)}  move ${R3.move}`);
for (const v of F.verdicts) {
  log(`   [${v.tier}] rise=${v.rise_mm} drop=${v.drop} f=${v.fmul} honest=${v.honest} stable=${v.stableClear} ` +
      `upTail=${v.uprightTailTicks}/${v.tailTicks} evFired=${v.eventFired}@${v.eventT} e=${v.eventE_mm}mm ` +
      `pen=${v.penetrationAtScore_mm}mm maxDY=${v.maxAbsDY_mm}mm rew=${v.reward} x=${v.x_mm} above=${v.above_mm} fot=${v.feetOnTread}`);
}
// which axis still kills it
const byAxis = {};
for (const v of F.verdicts) {
  const dh = v.rise_mm - 60;
  byAxis[`rise${dh >= 0 ? '+' : ''}${dh}`] = byAxis[`rise${dh >= 0 ? '+' : ''}${dh}`] || [0, 0];
  byAxis[`rise${dh >= 0 ? '+' : ''}${dh}`][1]++; if (v.honest) byAxis[`rise${dh >= 0 ? '+' : ''}${dh}`][0]++;
  const pk = `plant d${v.drop}/f${v.fmul}`;
  byAxis[pk] = byAxis[pk] || [0, 0]; byAxis[pk][1]++; if (v.honest) byAxis[pk][0]++;
}
OUT.final.byAxis = Object.fromEntries(Object.entries(byAxis).map(([k, [a, b]]) => [k, `${a}/${b}`]));
log('  axis breakdown: ' + Object.entries(OUT.final.byAxis).map(([k, v]) => `${k} ${v}`).join('  '));

OUT.elapsed_s = +el().toFixed(1);
fs.writeFileSync(P + 'r4_famA-results.json', JSON.stringify(OUT, null, 2));
fs.writeFileSync(P + 'famA_r4.log', LOGLINES.join('\n') + '\n');
log(`wrote ${P}r4_famA-results.json and ${P}famA_r4.log`);
