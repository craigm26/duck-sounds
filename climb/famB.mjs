// famB.mjs — ROUND 4, FAMILY B: the 80-120 mm band as TWO BEATS.
//
// Round 3's verdict for this band: at 90 mm the 'honest' criterion needs the
// trunk at 185 mm and the beak-strut buys 13-21 mm of lift; at 120 mm a
// foot-on-riser move rests a foot on the tread in 6 of 9 cells but the trunk
// never crosses x = 0.12. Every one of those was ONE ballistic move authored
// from the floor. This family stops searching from the floor and splits the
// climb:
//
//   BEAT 1  from the floor: plant the beak on the tread AND get a foot bearing
//           on the riser face / tread edge, with the trunk as high as it will
//           go. Scored by the TERMINAL state (trunk z, head contact, foot
//           bearing) under the whole-episode lateral gate, NOT by 'honest'.
//   BEAT 2  from beat 1's terminal state (rig3's spawn override extended with
//           spawnPose / spawnQuat / spawnVel / spawnLastAction / settleTicks —
//           see rig3.mjs, parity proved in PHASE P below): hips fold, the
//           trailing leg pushes, the feet come to rest on the tread. Scored
//           under 'honest' on the shared 9-cell core grid.
//   BEAT 3  the two beats CONCATENATED into ONE file, spawned on the floor
//           with no handoff at all, scored on the shared 14-cell grid. This is
//           the only one of the three that is a climb. A beat-2 clear that
//           needs the handoff spawn is reported as a beat-2 result.
//
// The concatenation is exact by construction: beat 1's episode runs its track
// and then holds the last keyframe for 0.8 s, and beat 2's first keyframe IS
// beat 1's last keyframe placed at that instant, so poseAt() is constant across
// the seam and the handoff state is read at exactly that tick.
//
// One Node process. mulberry32 seeds from base 11919.
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/famB.mjs
import fs from 'node:fs';
import * as R3 from '../climb/rig3.mjs';
import * as RB from '../climb/robust.mjs';

const ARG = Object.fromEntries(process.argv.slice(2).map(a => {
  const m = a.match(/^--([^=]+)(?:=(.*))?$/); return m ? [m[1], m[2] === undefined ? '1' : m[2]] : [a, '1'];
}));
const SMOKE = ARG.smoke === '1';
const OUTDIR = '../climb/';
const LOG = fs.createWriteStream(OUTDIR + (SMOKE ? 'famB_smoke.log' : 'famB.log'), { flags: 'a' });
const say = (...a) => { const s = a.join(' '); console.log(s); LOG.write(s + '\n'); };

// ---------------------------------------------------------------- rng
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const SEED_BASE = 11919;

const HOME = R3.HOME, LO = R3.LO, HI = R3.HI;
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const mm = v => (v * 1000).toFixed(1);

// ---------------------------------------------------------------- pose builder
// 14 policy joints: 0 lHipYaw 1 lHipRoll 2 lHipPitch 3 lKnee 4 lAnkle
//                   5 neckPitch 6 headPitch 7 headYaw 8 headRoll
//                   9 rHipYaw 10 rHipRoll 11 rHipPitch 12 rKnee 13 rAnkle
// The right leg mirrors the left with the opposite sign (HOME: -0.4579/+0.4579,
// -0.0049/+0.0049, +0.453/-0.453), which is the convention every round-3 file
// uses, so `legs(L, Rg)` writes Rg negated into 11/12/13.
function pose({ L = [HOME[2], HOME[3], HOME[4]], Rg = [-HOME[2], -HOME[3], -HOME[4]],
                neck = HOME[5], head = HOME[6], roll = 0 } = {}) {
  const p = HOME.slice();
  p[1] = clamp(HOME[1] + roll, LO[1], HI[1]);
  p[10] = clamp(HOME[10] + roll, LO[10], HI[10]);
  p[2] = clamp(L[0], LO[2], HI[2]); p[3] = clamp(L[1], LO[3], HI[3]); p[4] = clamp(L[2], LO[4], HI[4]);
  p[11] = clamp(-Rg[0], LO[11], HI[11]); p[12] = clamp(-Rg[1], LO[12], HI[12]); p[13] = clamp(-Rg[2], LO[13], HI[13]);
  p[5] = clamp(neck, LO[5], HI[5]); p[6] = clamp(head, LO[6], HI[6]);
  return p;
}

// ---------------------------------------------------------------- genes
const lerp = (u, lo, hi) => lo + u * (hi - lo);
// Every gene is a u in [0,1]; SPEC maps it into its range. Bounds that
// robust.mjs ENFORCES (blend, side) are inside the declared box by construction,
// so nothing this family writes can leave it.
const B1SPEC = [
  ['gap', 0.005, 0.055], ['side', -0.02, 0.09], ['blend', 0.70, 2.40], ['approach', 0.0, 0.50],
  ['t1', 0.10, 0.40], ['t2', 0.10, 0.45], ['t3', 0.10, 0.45], ['t4', 0.15, 0.60],
  ['cHip', -1.20, 0.60], ['cKnee', -1.30, 1.30], ['cAnk', -1.30, 1.30],
  ['strutNeck', -1.55, 1.00], ['strutHead', -1.55, 1.55],
  ['rHip', -1.20, 1.20], ['rKnee', -1.45, 1.45], ['rAnk', -1.45, 1.45],
  ['lHip', -1.20, 1.45], ['lKnee', -1.50, 1.50], ['lAnk', -1.50, 1.50],
  ['tHip', -1.20, 1.45], ['tKnee', -1.50, 1.50], ['tAnk', -1.50, 1.50],
  ['roll0', -0.33, 0.33],
];
const B2SPEC = [
  ['u1', 0.08, 0.40], ['u2', 0.08, 0.45], ['u3', 0.10, 0.55],
  ['fHip', -1.30, 1.45], ['fKnee', -1.50, 1.50], ['fAnk', -1.50, 1.50],
  ['pHip', -1.30, 1.45], ['pKnee', -1.50, 1.50], ['pAnk', -1.50, 1.50],
  ['sHip', -1.30, 1.45], ['sKnee', -1.50, 1.50], ['sAnk', -1.50, 1.50],
  ['aHip', -1.30, 1.45], ['aKnee', -1.50, 1.50], ['aAnk', -1.50, 1.50],
  ['neck2', -1.55, 1.00], ['head2', -1.55, 1.55], ['roll2', -0.33, 0.33],
];
const decode = (spec, u) => Object.fromEntries(spec.map(([n, lo, hi], i) => [n, lerp(clamp(u[i], 0, 1), lo, hi)]));

/** BEAT 1: crouch -> beak down onto the tread -> lead foot onto the riser -> rise on the strut. */
function beat1Track(g) {
  const t1 = g.t1, t2 = t1 + g.t2, t3 = t2 + g.t3, t4 = t3 + g.t4;
  const strut = { neck: g.strutNeck, head: g.strutHead };
  const K = [
    { t: +t1.toFixed(4), pose: pose({ L: [g.cHip, g.cKnee, g.cAnk], Rg: [g.cHip, g.cKnee, g.cAnk], roll: g.roll0 }) },
    { t: +t2.toFixed(4), pose: pose({ L: [g.rHip, g.rKnee, g.rAnk], Rg: [g.rHip, g.rKnee, g.rAnk], ...strut, roll: g.roll0 }) },
    { t: +t3.toFixed(4), pose: pose({ L: [g.lHip, g.lKnee, g.lAnk], Rg: [g.rHip, g.rKnee, g.rAnk], ...strut, roll: g.roll0 }) },
    { t: +t4.toFixed(4), pose: pose({ L: [g.lHip, g.lKnee, g.lAnk], Rg: [g.tHip, g.tKnee, g.tAnk], ...strut, roll: g.roll0 }) },
  ];
  return K;
}
const beat1Intent = (g, h, name) => ({
  name, family: 'B two-beat (round 4) — beat 1, from the floor',
  beat: 1, rise_mm: Math.round(h * 1000),
  keyframes: beat1Track(g), blend: g.blend, gap: g.gap, side: g.side, approach: g.approach,
  isolate: true, stepCount: 4, params: g,
});

/** BEAT 2: from the handoff. K0 is beat 1's LAST KEYFRAME, so the seam is flat. */
function beat2Track(g2, k0pose) {
  const u1 = 0.02 + g2.u1, u2 = u1 + g2.u2, u3 = u2 + g2.u3;
  const st = { neck: g2.neck2, head: g2.head2 };
  return [
    { t: 0.02, pose: k0pose.slice() },
    { t: +u1.toFixed(4), pose: pose({ L: [g2.fHip, g2.fKnee, g2.fAnk], Rg: [g2.pHip, g2.pKnee, g2.pAnk], ...st, roll: g2.roll2 }) },
    { t: +u2.toFixed(4), pose: pose({ L: [g2.sHip, g2.sKnee, g2.sAnk], Rg: [g2.sHip, g2.sKnee, g2.sAnk], ...st, roll: g2.roll2 }) },
    { t: +u3.toFixed(4), pose: pose({ L: [g2.aHip, g2.aKnee, g2.aAnk], Rg: [g2.aHip, g2.aKnee, g2.aAnk], ...st, roll: g2.roll2 }) },
  ];
}
const beat2Intent = (g2, b1, T, h, name) => ({
  name, family: 'B two-beat (round 4) — beat 2, FROM A HANDOFF SPAWN (not a climb)',
  beat: 2, rise_mm: Math.round(h * 1000),
  handoffFrom: b1.file, handoffNote: 'spawned in beat 1\'s terminal qpos+qvel+lastAction; settleTicks 0. NOT a climb.',
  keyframes: beat2Track(g2, b1.intent.keyframes[b1.intent.keyframes.length - 1].pose),
  blend: b1.intent.blend, gap: 0, side: 0, approach: 0,
  spawn: T.spawn, spawnQuat: T.spawnQuat, spawnPose: T.spawnPose,
  spawnVel: T.spawnVel, spawnLastAction: T.spawnLastAction, settleTicks: 0,
  isolate: true, stepCount: 4, params: g2,
});

/** BEAT 3: the two beats as ONE track from the floor. No handoff of any kind. */
function concatIntent(b1, b2i, h, name) {
  const T0 = b1.intent.keyframes[b1.intent.keyframes.length - 1].t;
  const shift = T0 + 0.8 - 0.02;      // beat 2's K0 lands exactly at the handoff instant
  const kf = b1.intent.keyframes.map(f => ({ t: f.t, pose: f.pose.slice() }))
    .concat(b2i.keyframes.map(f => ({ t: +(f.t + shift).toFixed(4), pose: f.pose.slice() })));
  return {
    name, family: 'B two-beat (round 4) — CONCATENATED, spawned on the floor',
    beat: '1+2', rise_mm: Math.round(h * 1000),
    keyframes: kf, blend: b1.intent.blend, gap: b1.intent.gap, side: b1.intent.side,
    approach: b1.intent.approach, isolate: true, stepCount: 4,
    params: { beat1: b1.intent.params, beat2: b2i.params },
  };
}

// ---------------------------------------------------------------- scoring
const TMP1 = OUTDIR + '_famB_tmp_b1.json';
const TMP2 = OUTDIR + '_famB_tmp_b2.json';
const TMP3 = OUTDIR + '_famB_tmp_b3.json';
const write = (o, p) => { fs.writeFileSync(p, JSON.stringify(o, null, 2)); return p; };

/** One nominal cell (drop 0.120, x1.0, isolate on) — the cheap screen. */
const cell0 = (p, h) => RB.scoreCell(p, { rise: h, dh: 0, drop: 0.120, fmul: 1.0, isolate: true });

/**
 * BEAT-1 OBJECTIVE. Not 'honest' — beat 1 is not supposed to finish the climb.
 * It is the handoff: how high is the trunk at the seam, with the beak on the
 * tread and a foot bearing, without ever leaving the 340 mm flight.
 */
function beat1Score(r, h) {
  if (r.invalid) return { J: -1e9, why: 'bounds' };
  if (r.maxAbsDY > RB.LATERAL) return { J: -1e9, why: 'left the flight' };
  const T = r.terminal;
  const pen_mm = T.penetration === null ? 0 : 1000 * T.penetration;
  // THE SEAM'S DISTANCE TO THE CRITERION, in millimetres. 'honest' needs the
  // trunk past x = 120 mm and more than 95 mm above the tread; dx and dz are
  // exactly how far short the handoff instant falls on each. Measuring the
  // deficit rather than rewarding x and z separately is what stops the search
  // buying trunk x by toppling forward: a fall gains x and loses far more z.
  const dx = Math.max(0, 120 - 1000 * T.spawn.x);
  const dz = Math.max(0, (1000 * h + 95) - 1000 * T.spawn.z);
  const dzPeak = Math.max(0, (1000 * h + 95) - 1000 * r.maxZ);
  const J = -(dx + dz) - 0.5 * dzPeak
    + 40 * (T.head ? 1 : 0) + 40 * (T.footRiser ? 1 : 0) + 45 * T.feetOnTread
    + 25 * (T.up ? 1 : 0)
    + 12 * r.headFrac + 12 * r.riserFrac
    - 3 * Math.max(0, -pen_mm - 5);           // support bought by sinking into the block is not support
  return { J, why: null, dx, dz };
}
/** BEAT-2 / BEAT-3 SCREEN: the shared objective's shape, on one cell. */
function honestScore(r) {
  if (r.invalid) return -1e9;
  const reached = (r.maxX > RB.RISER_X) || (r.feetOnTreadMax > 0);
  return r.reward + 4 * (r.crit.honest && r.uprightTailTicks >= 45 ? 1 : 0)
    + 4 * (reached ? r.uprightTailFrac : 0);
}

// ---------------------------------------------------------------- ES driver
/**
 * (mu, lambda) with a shrinking sigma, mulberry32-seeded. Random for the first
 * `explore` evaluations, then Gaussian around the elite set. Stops on a wall
 * clock so the family fits its compute budget.
 */
async function search({ spec, seed, deadline, explore, evalOne, label, sigma0 = 0.30, seedElite = [] }) {
  const rnd = mulberry32(seed);
  const n = spec.length;
  // A warm start carries the PARAMETER VECTORS of a previous stage's elite, not
  // its scores: every one of them is re-evaluated here, at this rise, on this
  // plant, before it can be a parent.
  const elite = seedElite.slice(0, 5).map(u => ({ u: u.slice(), J: -1e9, extra: null }));
  let best = null, evals = 0;
  const gauss = () => { let s = 0; for (let i = 0; i < 3; i++) s += rnd(); return (s - 1.5) / 0.5 * 0.4082; };
  while (Date.now() < deadline) {
    let u;
    if (evals < seedElite.length) u = seedElite[evals].slice();          // re-score the warm start first
    else if (evals < explore || !elite.length) u = Array.from({ length: n }, () => rnd());
    else {
      const par = elite[Math.floor(rnd() * elite.length)];
      const sg = sigma0 * Math.max(0.25, 1 - evals / (explore * 6));
      u = par.u.map(v => clamp(v + gauss() * sg, 0, 1));
    }
    const out = await evalOne(decode(spec, u), evals);
    evals++;
    const rec = { u, J: out.J, extra: out.extra };
    elite.push(rec);
    for (let i = elite.length - 1; i >= 0; i--) if (elite[i].J === -1e9 && elite[i].extra === null) elite.splice(i, 1);
    elite.sort((a, b) => b.J - a.J); if (elite.length > 5) elite.length = 5;
    if (!best || out.J > best.J) {
      best = rec;
      say(`  [${label}] eval ${evals} NEW BEST J=${out.J.toFixed(3)}  ${out.line}`);
    }
  }
  return { best, evals, elite };
}

// ================================================================== PHASE P
say(`\n================ famB PHASE P — parity of the handoff extension ================`);
say(`${new Date().toISOString()}  smoke=${SMOKE}`);
const R3P = await import('../climb/rig3_prefamB.mjs');
const RBP = await import('../climb/robust_prefamB.mjs');
const PFILES = ['best_r3_vault_60mm.json', 'best_r2_vault_40mm.json', 'best_r3_landvault_90mm.json',
                'ctrl_do_nothing.json', 'best_0_40mm.json', 'best_1_90mm.json'];
const PRISE = { 'best_r3_vault_60mm.json': 0.060, 'best_r2_vault_40mm.json': 0.040,
                'best_r3_landvault_90mm.json': 0.090, 'ctrl_do_nothing.json': 0.090,
                'best_0_40mm.json': 0.040, 'best_1_90mm.json': 0.090 };
let parityAll = true;
const parityRows = [];
for (const f of PFILES) {
  const p = OUTDIR + f, h = PRISE[f];
  const A = await R3P.scoreSaved(p, { rise: h, tail: 'policy' });
  const B = await R3.scoreSaved(p, { rise: h, tail: 'policy' });
  const C = await RBP.scoreCell(p, { rise: h, isolate: true });
  const D = await RB.scoreCell(p, { rise: h, isolate: true });
  const hA = RBP.intentHash(JSON.parse(fs.readFileSync(p, 'utf8')));
  const hB = RB.intentHash(JSON.parse(fs.readFileSync(p, 'utf8')));
  const eq = (a, b) => a === b;
  const ok3 = eq(A.scored.x, B.scored.x) && eq(A.scored.z, B.scored.z) && eq(A.reward, B.reward)
    && eq(A.crit.honest, B.crit.honest) && eq(A.uprightTailTicks, B.uprightTailTicks)
    && eq(A.penetrationAtScore, B.penetrationAtScore) && eq(A.maxAbsDY, B.maxAbsDY);
  const okR = eq(C.scored.x, D.scored.x) && eq(C.scored.z, D.scored.z) && eq(C.reward, D.reward)
    && eq(C.crit.honest, D.crit.honest) && eq(C.uprightTailTicks, D.uprightTailTicks)
    && eq(C.maxAbsDY, D.maxAbsDY);
  const okH = hA === hB;
  if (!(ok3 && okR && okH)) parityAll = false;
  parityRows.push({ file: f, rise_mm: Math.round(h * 1000), rig3EXACT: ok3, robustEXACT: okR,
    hashEXACT: okH, sha256_8: hB.slice(0, 8),
    x: B.scored.x, z: B.scored.z, reward: B.reward, honest: B.crit.honest,
    upTail: B.uprightTailTicks,
    terminal_z_mm: +mm(B.terminal.spawn.z), terminal_head: B.terminal.head,
    terminal_footRiser: B.terminal.footRiser });
  say(`  ${f.padEnd(30)} rig3 EXACT=${ok3}  robust EXACT=${okR}  hash EXACT=${okH} (${hB.slice(0, 8)})  x=${B.scored.x} z=${B.scored.z} rew=${B.reward}`);
}
say(`  parityAll = ${parityAll}`);
if (!parityAll) { say('  PARITY FAILED — refusing to search on a changed instrument.'); process.exit(2); }

// ---------------------------------------------------------------- timing
const tT = Date.now();
await cell0(OUTDIR + 'best_r3_vault_60mm.json', 0.060);
const SEC_PER_CELL = (Date.now() - tT) / 1000;
say(`  one nominal cell = ${SEC_PER_CELL.toFixed(2)} s`);

// ================================================================== the run
// 90 mm first and at full length; 80 and 120 are refinements WARM-STARTED from
// the 90 mm elite (the parameter vectors, re-scored at their own rise). That
// spends the budget where the band's question is sharpest instead of running
// three independent under-sampled searches.
const RISES = SMOKE ? [0.090] : [0.090, 0.080, 0.120];
const B1_SEC = SMOKE ? 25 : { 90: 330, 80: 130, 120: 130 };
const B2_SEC = SMOKE ? 25 : 200;
const B3_SEC = SMOKE ? 20 : 200;
const secs = (tbl, hm) => (typeof tbl === 'number' ? tbl : tbl[hm]);
const results = {
  generated: new Date().toISOString(), family: 'B two-beat (80-120 mm band)',
  seedBase: SEED_BASE, secPerCell: SEC_PER_CELL,
  instrument: {
    changed: ['climb/rig3.mjs', 'climb/robust.mjs'],
    added: ['climb/rig3_prefamB.mjs', 'climb/robust_prefamB.mjs', 'climb/famB.mjs'],
    extension: 'optional spawnQuat / spawnPose / spawnVel / spawnLastAction / settleTicks + a read-only `terminal` handoff record; absent -> not one line runs',
    alsoFixed: "robust.mjs go(): family A's event block referenced `opts.event` where go()'s options object is named `o` — a ReferenceError that made the shared scorer unusable for every family. Changed to o.event, nothing else.",
    parity: parityRows, parityAll,
  },
  beat1: [], beat2: [], concat: [], grids: [],
};
let elite90 = [];

for (const h of RISES) {
  const hm = Math.round(h * 1000);
  say(`\n================ RISE ${hm} mm ================`);

  // ------------------------------------------------------------ BEAT 1
  const b1sec = secs(B1_SEC, hm);
  const warm = (hm === 90 || SMOKE) ? [] : elite90;
  const b1 = await search({
    spec: B1SPEC, seed: SEED_BASE + hm, deadline: Date.now() + b1sec * 1000,
    explore: SMOKE ? 6 : Math.max(15, Math.round(0.40 * b1sec / SEC_PER_CELL)),
    label: `b1 ${hm}`, seedElite: warm,
    evalOne: async (g) => {
      const p = write(beat1Intent(g, h, `famB_b1_${hm}`), TMP1);
      const r = await cell0(p, h);
      const S = beat1Score(r, h);
      const T = r.terminal || { spawn: { x: 0, z: 0 }, head: false, footRiser: false, feetOnTread: 0, up: false, penetration: null };
      return { J: S.J, extra: { g },
        line: `zEnd=${mm(T.spawn.z)} xEnd=${mm(T.spawn.x)} dx=${S.dx === undefined ? 'na' : S.dx.toFixed(1)} dz=${S.dz === undefined ? 'na' : S.dz.toFixed(1)} head=${T.head} riser=${T.footRiser} fot=${T.feetOnTread} up=${T.up} peakZ=${mm(r.maxZ)} maxDY=${mm(r.maxAbsDY)} pen=${T.penetration === null ? 'na' : mm(T.penetration)} (mm)` };
    },
  });
  if (hm === 90 && !SMOKE) elite90 = b1.elite.map(e => e.u.slice());
  say(`  beat 1 @${hm}mm: ${b1.evals} evaluations in ${b1sec}s, best J = ${b1.best.J.toFixed(3)}`);
  const g1 = b1.best.extra.g;
  const b1intent = beat1Intent(g1, h, `famB_b1_${hm}`);
  const b1file = OUTDIR + `best_r4_famB_beat1_${hm}mm.json`;
  write(b1intent, b1file);
  const b1r = await cell0(b1file, h);            // rescored FROM THE SAVED FILE
  const T = b1r.terminal;
  say(`  beat-1 TERMINAL @${hm}mm: trunk z=${mm(T.spawn.z)}mm x=${mm(T.spawn.x)}mm dy=${mm(T.spawn.y - RB.STAIR_Y)}mm` +
      ` | beak on tread=${T.head} foot on riser=${T.footRiser} feet resting on tread=${T.feetOnTread} upright=${T.up}` +
      ` | penetration=${T.penetration === null ? 'na' : mm(T.penetration)}mm peakZ=${mm(b1r.maxZ)}mm maxDY=${mm(b1r.maxAbsDY)}mm maxTq=${b1r.maxTq.toFixed(3)}Nm`);
  results.beat1.push({ rise_mm: hm, file: b1file, sha256: RB.intentHashOfFile(b1file), evals: b1.evals,
    searchSeconds: b1sec, warmStartedFrom90mm: warm.length > 0, J: +b1.best.J.toFixed(3),
    terminal: { trunk_x_mm: +mm(T.spawn.x), trunk_z_mm: +mm(T.spawn.z), trunk_dy_mm: +mm(T.spawn.y - RB.STAIR_Y),
      beakOnTread: T.head, footOnRiser: T.footRiser, feetRestingOnTread: T.feetOnTread, upright: T.up,
      penetration_mm: T.penetration === null ? null : +mm(T.penetration) },
    peakZ_mm: +mm(b1r.maxZ), maxAbsDY_mm: +mm(b1r.maxAbsDY), maxTq_Nm: +b1r.maxTq.toFixed(4),
    headFrac: +b1r.headFrac.toFixed(3), riserFrac: +b1r.riserFrac.toFixed(3),
    honestOnItsOwn: b1r.crit.honest, params: g1 });

  // ------------------------------------------------------------ BEAT 2 (handoff)
  const b1ctx = { file: b1file, intent: b1intent };
  const b2 = await search({
    spec: B2SPEC, seed: SEED_BASE + hm + 7, deadline: Date.now() + B2_SEC * 1000,
    explore: SMOKE ? 6 : Math.max(15, Math.round(0.40 * B2_SEC / SEC_PER_CELL)), label: `b2 ${hm}`,
    evalOne: async (g2) => {
      const p = write(beat2Intent(g2, b1ctx, T, h, `famB_b2_${hm}`), TMP2);
      const r = await cell0(p, h);
      return { J: honestScore(r), extra: { g: g2 },
        line: `honest=${r.crit.honest} rew=${r.reward.toFixed(2)} x=${mm(r.scored.x)} above=${mm(r.scored.above)} fot=${r.scored.feetOnTread} fotMax=${r.feetOnTreadMax} upTail=${r.uprightTailTicks}/50` };
    },
  });
  const g2 = b2.best.extra.g;
  const b2file = OUTDIR + `best_r4_famB_beat2_${hm}mm.json`;
  write(beat2Intent(g2, b1ctx, T, h, `famB_b2_${hm}`), b2file);
  say(`  beat 2 @${hm}mm: ${b2.evals} evaluations, best single-cell J = ${b2.best.J.toFixed(3)}`);
  const G2 = await RB.scoreRobust(b2file, { rise: h, core: true });
  say(`  BEAT-2-FROM-HANDOFF @${hm}mm (NOT a climb): kCore=${G2.kCore}/9 kCoreStable=${G2.kCoreStable}/9 objectiveR3=${G2.objectiveR3.toFixed(3)} objectiveCore=${G2.objectiveCore.toFixed(3)} meanReward=${G2.meanReward.toFixed(3)} minUpTail=${G2.agg.minUprightTailTicks}/50 minPen=${G2.agg.minPenetrationAtScore_mm.toFixed(2)}mm maxDY=${G2.agg.maxAbsDY_mm.toFixed(1)}mm meanAbove=${G2.agg.meanAbove_mm.toFixed(1)}mm meanX=${G2.agg.meanX_mm.toFixed(1)}mm`);
  results.beat2.push({ rise_mm: hm, file: b2file, sha256: G2.sha256, evals: b2.evals,
    kCore: G2.kCore, kCoreStable: G2.kCoreStable, objectiveR3: +G2.objectiveR3.toFixed(3),
    objectiveCore: +G2.objectiveCore.toFixed(3), meanReward: +G2.meanReward.toFixed(3),
    isAClimb: false, note: 'spawned in beat 1 terminal qpos+qvel+lastAction, settleTicks 0 — a beat-2 result, not a climb',
    verdicts: G2.verdicts, agg: G2.agg, params: g2 });

  // ------------------------------------------------------------ BEAT 3 (the climb)
  // Re-tune the beat-2 genes AGAINST THE CONCATENATED FILE — the real move,
  // spawned on the floor, no handoff — warm-started from beat 2's elite.
  const b3 = await search({
    spec: B2SPEC, seed: SEED_BASE + hm + 13, deadline: Date.now() + B3_SEC * 1000,
    explore: SMOKE ? 6 : Math.max(15, Math.round(0.30 * B3_SEC / SEC_PER_CELL)), label: `b3 ${hm}`,
    seedElite: b2.elite.map(e => e.u.slice()),
    evalOne: async (g3) => {
      const bi = beat2Intent(g3, b1ctx, T, h, `famB_b2_${hm}`);
      const p = write(concatIntent(b1ctx, bi, h, `famB_concat_${hm}`), TMP3);
      // TWO cells, not one: the nominal plant and the plant that killed round 3
      // (drop 0.130, friction x0.7). The concatenated file is the only artifact
      // of this family that is a climb, so it is the one screened for robustness.
      const r = await cell0(p, h);
      const r2 = await RB.scoreCell(p, { rise: h, dh: 0, drop: 0.130, fmul: 0.7, isolate: true });
      return { J: 0.5 * (honestScore(r) + honestScore(r2)), extra: { g: g3 },
        line: `nom honest=${r.crit.honest} rew=${r.reward.toFixed(2)} x=${mm(r.scored.x)} above=${mm(r.scored.above)} fot=${r.scored.feetOnTread} fotMax=${r.feetOnTreadMax} peakZ=${mm(r.maxZ)} upTail=${r.uprightTailTicks}/50 | slip honest=${r2.crit.honest} rew=${r2.reward.toFixed(2)}` };
    },
  });
  const g3 = b3.best.extra.g;
  const b3file = OUTDIR + `best_r4_famB_concat_${hm}mm.json`;
  write(concatIntent(b1ctx, beat2Intent(g3, b1ctx, T, h, `famB_b2_${hm}`), h, `famB_concat_${hm}`), b3file);
  say(`  beat 3 @${hm}mm: ${b3.evals} evaluations re-tuning beat 2 on the concatenated file, best single-cell J = ${b3.best.J.toFixed(3)}`);
  const G3 = await RB.scoreRobust(b3file, { rise: h });
  say(`  CONCATENATED (ONE file, from the floor — this is the climb) @${hm}mm: kCore=${G3.kCore}/9 kCoreStable=${G3.kCoreStable}/9 kExt=${G3.kExt}/14 kExtStable=${G3.kExtStable}/14 objective=${G3.objective.toFixed(3)} objectiveCore=${G3.objectiveCore.toFixed(3)} objectiveR3=${G3.objectiveR3.toFixed(3)} meanReward=${G3.meanReward.toFixed(3)} maxZ=${mm(G3.agg.maxZ)}mm meanAbove=${G3.agg.meanAbove_mm.toFixed(1)}mm meanX=${G3.agg.meanX_mm.toFixed(1)}mm feetOnTreadMax=${G3.agg.feetOnTreadMax} minUpTail=${G3.agg.minUprightTailTicks}/50 maxDY=${G3.agg.maxAbsDY_mm.toFixed(1)}mm minPen=${G3.agg.minPenetrationAtScore_mm.toFixed(2)}mm`);
  results.concat.push({ rise_mm: hm, file: b3file, sha256: G3.sha256, evals: b3.evals,
    kCore: G3.kCore, kCoreStable: G3.kCoreStable, kExt: G3.kExt, kExtStable: G3.kExtStable,
    objective: +G3.objective.toFixed(3), objectiveCore: +G3.objectiveCore.toFixed(3),
    objectiveR3: +G3.objectiveR3.toFixed(3), meanReward: +G3.meanReward.toFixed(3),
    isAClimb: true, verdicts: G3.verdicts, agg: G3.agg, params: g3 });

  // beat 1 alone, on the shared grid — it is a saved file like any other
  const G1 = await RB.scoreRobust(b1file, { rise: h });
  say(`  beat 1 alone @${hm}mm on the shared grid: kCore=${G1.kCore}/9 kExt=${G1.kExt}/14 objective=${G1.objective.toFixed(3)} objectiveR3=${G1.objectiveR3.toFixed(3)}`);
  results.grids.push({ what: 'beat1-alone', rise_mm: hm, file: b1file, sha256: G1.sha256,
    kCore: G1.kCore, kExt: G1.kExt, kCoreStable: G1.kCoreStable, kExtStable: G1.kExtStable,
    objective: +G1.objective.toFixed(3), objectiveR3: +G1.objectiveR3.toFixed(3),
    meanReward: +G1.meanReward.toFixed(3), verdicts: G1.verdicts, agg: G1.agg });

  fs.writeFileSync(OUTDIR + (SMOKE ? 'r4_famB-results.smoke.json' : 'r4_famB-results.json'),
    JSON.stringify(results, null, 2));
}

function slim(r) {
  return { x: r.scored.x, z: r.scored.z, honest: r.crit ? r.crit.honest : false };
}

fs.writeFileSync(OUTDIR + (SMOKE ? 'r4_famB-results.smoke.json' : 'r4_famB-results.json'),
  JSON.stringify(results, null, 2));
say(`\nwrote ${OUTDIR}${SMOKE ? 'r4_famB-results.smoke.json' : 'r4_famB-results.json'}`);
for (const f of [TMP1, TMP2, TMP3]) { try { fs.unlinkSync(f); } catch { } }
say(`done ${new Date().toISOString()}`);
