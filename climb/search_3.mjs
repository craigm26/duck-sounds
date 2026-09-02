// FAMILY C — THE CORNER STEM, SEARCHING CONTACT ORDER.  120 mm and 180 mm.
//
// Round 1 searched three whole-body strategies and got 0 clears in ~6000
// episodes. The judge's last instrument finding was that all three searched
// TIMINGS INSIDE A FIXED SEQUENCE: head first, then left foot, then right,
// then over. Nobody ever asked whether that is the right order. This does.
//
// What is parametrised here that was not before:
//   order      which contact is established FIRST — the head on the tread, a
//              foot on the riser face, or the outboard leg braced on the side
//              wall — as a discrete gene over 6 sequences, not a fixed script.
//   lead       which foot goes to the riser (L or R). The legs are mirrored
//              (right_hip_pitch HOME +0.4579 vs left -0.4579), so this is not
//              a symmetry.
//   liftAt     WHEN the second foot leaves the floor: after the 1st, 2nd or
//              3rd contact. That is the moment the duck commits its weight.
//   retractAt  WHEN the neck retracts. Retracting the neck with the beak
//              planted is the actual haul — it is the only actuator that can
//              pull the trunk toward a contact it is already making.
//   endHome    whether the track finally returns the legs to HOME (a stand) or
//              holds the climbed pose.
// plus 30 continuous keyframe/timing genes including the side offset, which
// only means anything because the flight is flush to a wall at y = 1.475.
//
// SCORING. climb/rig3.mjs, its `honest` criterion, tail = 'policy'. Nothing
// here re-implements an episode: every number comes from rig3.scoreSaved(PATH),
// which JSON.parse()s a file off disk. A candidate is written to disk (rounded
// exactly as it will be published) and then scored FROM THAT FILE, so the
// number in the log is a number the saved file produces. The saved best is
// re-scored from its published path at the end.
//
// REWARD. rig3's shaped reward is kept whole (approach, feet on tread, height,
// upright, hard lateral gate) and three LOAD-TRANSFER terms are added on top,
// because the brief is to reward lift-while-bearing rather than contact time:
// trunk z gain while the head and a foot are both in contact, the fraction of
// the episode spent more than 20 mm up in that state, and peak trunk z.
//
// FLIGHT. rig3 Phase E: with four steps the blocks interpenetrate and shove
// each other along x below a ~140 mm rise, and a duck placed standing on the
// first tread is thrown off at every rise from 20 to 120 mm. So the 120 mm arm
// searches a ONE-BLOCK flight (opts.stepCount = 1 — the same knob rig3 used to
// demonstrate the bug; it changes no physics constant) and the 180 mm arm
// searches the canonical four-step flight, where the on-tread control passes.
// Both bests are re-scored on both flights at the end.
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/search_3.mjs
import fs from 'node:fs';
import { scoreSaved, saveTrack, HOME, LO, HI, criteria } from '../climb/rig3.mjs';

const OUT = '../climb/search_3-results.json';
const TMP = '/tmp/claude-1000/-home-craigm26/43f07dce-e62f-464d-ae1f-2fe020620950/scratchpad';
const BUDGET_S = +(process.env.BUDGET_S || 2280);          // wall-clock for the two search arms
const T00 = Date.now();
const el = () => (Date.now() - T00) / 1000;
const mm = v => (v * 1000).toFixed(1);

// ---------------------------------------------------------------- rng
/** mulberry32 — seeded, so every run in this file reproduces. */
function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------- the gene
const J = { lhy: 0, lhr: 1, lhp: 2, lk: 3, la: 4, np: 5, hp: 6, hy: 7, hr: 8,
            rhy: 9, rhr: 10, rhp: 11, rk: 12, ra: 13 };

const CONT = {
  gap:      [0.00, 0.12],
  approach: [0.00, 0.45],
  side:     [0.00, 0.14],   // toward the wall at +y; the flight is 0.17 half-wide
  blend:    [0.70, 2.40],
  d1: [0.12, 0.70], d2: [0.12, 0.70], d3: [0.12, 0.70],
  d4: [0.12, 0.70], d5: [0.12, 0.70], d6: [0.12, 0.70],
  // H — plant the beak: the trunk has to pitch nose-down for the head to reach
  //     forward at all (climb/jaw-reach.mjs: the 190 mm of reach exists only at
  //     25-55 deg of trunk pitch, and trunk pitch is not a servo).
  hipLean: [0.00, 1.40], kneeFold: [-1.00, 1.00], ankLean: [-1.40, 0.60],
  neck: [-1.57, 1.05], headP: [-1.40, 1.50],
  // F — lead foot onto the riser FACE
  hipF: [-0.20, 1.60], kneeF: [-1.50, 0.80], ankF: [-1.40, 0.90], rollF: [-0.38, 0.38],
  // T — trail foot leaves the floor
  hipT: [-0.20, 1.60], kneeT: [-1.50, 0.80], ankT: [-1.40, 0.90], rollT: [-0.38, 0.38],
  // W — brace on the side wall
  rollW: [-0.38, 0.38], yawW: [-0.43, 0.52],
  // R — retract the neck and haul the body over
  npEnd: [-1.57, 1.05], hpEnd: [-1.40, 1.50],
  hipEnd: [-1.40, 1.40], kneeEnd: [-1.40, 1.40], ankEnd: [-1.40, 0.90],
};
const CKEYS = Object.keys(CONT);

/** The six contact orders. 'H' head, 'F' lead foot on riser, 'W' wall brace. */
const ORDERS = [
  ['H', 'F', 'W'], ['H', 'W', 'F'], ['F', 'H', 'W'],
  ['F', 'W', 'H'], ['W', 'H', 'F'], ['W', 'F', 'H'],
];
const DISC = { order: ORDERS.length, lead: 2, liftAt: 3, retractAt: 3, endHome: 2 };
const DKEYS = Object.keys(DISC);

function randGene(r) {
  const g = {};
  for (const k of CKEYS) { const [a, b] = CONT[k]; g[k] = a + r() * (b - a); }
  for (const k of DKEYS) g[k] = Math.floor(r() * DISC[k]);
  return g;
}
/** Gaussian-ish jitter on the continuous genes; discrete genes flip rarely. */
function jitter(g, s, r, freezeDisc) {
  const o = {};
  for (const k of CKEYS) {
    const [a, b] = CONT[k];
    o[k] = Math.min(b, Math.max(a, g[k] + (r() * 2 - 1) * (b - a) * s));
  }
  for (const k of DKEYS) {
    o[k] = (!freezeDisc && k !== 'order' && r() < 0.15) ? Math.floor(r() * DISC[k]) : g[k];
  }
  return o;
}

// ---------------------------------------------------------------- the track
const legIdx = lead => lead === 0
  ? { hip: J.lhp, knee: J.lk, ank: J.la, roll: J.lhr, s: +1 }
  : { hip: J.rhp, knee: J.rk, ank: J.ra, roll: J.rhr, s: -1 };

/**
 * Build the keyframe track from a gene.
 *
 * Stages are CUMULATIVE — each keyframe starts from the previous one and adds
 * its own contact — so the order gene really does change which contact is made
 * while which others are already held, not just the order of independent poses.
 */
function trackOf(g) {
  const seq = ORDERS[g.order].slice();
  const lift = { i: Math.min(g.liftAt, seq.length), tok: 'T' };
  const retr = { i: Math.min(g.retractAt, seq.length), tok: 'R' };
  // insert T then R (R never before T: the haul needs the second foot committed)
  seq.splice(lift.i + 1, 0, 'T');
  seq.splice(Math.max(retr.i + 1, lift.i + 2), 0, 'R');

  const L = legIdx(g.lead), T = legIdx(1 - g.lead);
  let p = HOME.slice();
  const frames = [];
  const dur = [g.d1, g.d2, g.d3, g.d4, g.d5, g.d6];
  let t = 0;
  seq.forEach((tok, i) => {
    p = p.slice();
    if (tok === 'H') {
      p[J.lhp] = HOME[J.lhp] + g.hipLean;  p[J.rhp] = HOME[J.rhp] - g.hipLean;
      p[J.lk]  = HOME[J.lk] + g.kneeFold;  p[J.rk]  = HOME[J.rk] - g.kneeFold;
      p[J.la]  = HOME[J.la] + g.ankLean;   p[J.ra]  = HOME[J.ra] - g.ankLean;
      p[J.np]  = g.neck;                   p[J.hp]  = g.headP;
    } else if (tok === 'F') {
      p[L.hip] = HOME[L.hip] + L.s * g.hipF;
      p[L.knee] = HOME[L.knee] + L.s * g.kneeF;
      p[L.ank] = HOME[L.ank] + L.s * g.ankF;
      p[L.roll] = HOME[L.roll] + g.rollF;
    } else if (tok === 'W') {
      p[J.lhr] = HOME[J.lhr] + g.rollW;  p[J.rhr] = HOME[J.rhr] + g.rollW;
      p[J.lhy] = HOME[J.lhy] + g.yawW;   p[J.rhy] = HOME[J.rhy] + g.yawW;
    } else if (tok === 'T') {
      p[T.hip] = HOME[T.hip] + T.s * g.hipT;
      p[T.knee] = HOME[T.knee] + T.s * g.kneeT;
      p[T.ank] = HOME[T.ank] + T.s * g.ankT;
      p[T.roll] = HOME[T.roll] + g.rollT;
    } else if (tok === 'R') {
      p[J.np] = g.npEnd; p[J.hp] = g.hpEnd;
      p[J.lhp] = HOME[J.lhp] + g.hipEnd;  p[J.rhp] = HOME[J.rhp] - g.hipEnd;
      p[J.lk]  = HOME[J.lk] + g.kneeEnd;  p[J.rk]  = HOME[J.rk] - g.kneeEnd;
      p[J.la]  = HOME[J.la] + g.ankEnd;   p[J.ra]  = HOME[J.ra] - g.ankEnd;
    }
    t += dur[i % dur.length];
    frames.push({ t, pose: p });
  });
  if (g.endHome) {
    const e = HOME.slice(); e[J.np] = g.npEnd; e[J.hp] = g.hpEnd;
    t += 0.45; frames.push({ t, pose: e });
  }
  return { frames, seq };
}

/** The intent object as it will be PUBLISHED — rounded here, once. */
function intentOf(g, note) {
  const { frames, seq } = trackOf(g);
  return {
    name: 'cornerstem',
    family: 'C_corner_stem_contact_order',
    order: ORDERS[g.order].join(''), sequence: seq.join(''),
    lead: g.lead ? 'R' : 'L', liftAt: g.liftAt, retractAt: g.retractAt, endHome: !!g.endHome,
    keyframes: frames.map(f => ({ t: +f.t.toFixed(4), pose: f.pose.map(v => +v.toFixed(5)) })),
    blend: +g.blend.toFixed(4), gap: +g.gap.toFixed(4),
    side: +g.side.toFixed(4), approach: +g.approach.toFixed(4),
    gene: Object.fromEntries(Object.entries(g).map(([k, v]) => [k, +(+v).toFixed(6)])),
    note: note || '',
  };
}

// ---------------------------------------------------------------- reward
/**
 * rig3's shaped reward (approach, feet on tread, height, upright, hard lateral
 * gate) plus the load-transfer terms this round is about. The criterion is NOT
 * this — the criterion is rig3's `honest`, unmodified.
 */
function score(rec, rise) {
  if (!(rec.maxAbsDY <= 0.17)) return { r: 0, why: 'off the flight' };
  const s = rec.scored;
  if (Math.abs(s.dy) > 0.17) return { r: 0, why: 'off the flight' };
  const clamp = v => Math.max(0, Math.min(1, v));
  const gain = rec.maxGainBoth == null ? 0 : rec.maxGainBoth;
  // Everything that rewards touching the tread or gaining height is multiplied
  // by how much of the episode the duck was UPRIGHT. The first smoke run found
  // the hole: a duck that falls face-first onto the step has a foot on the
  // tread and scored 8.4 while an upright approach scored 4.4. Falling over is
  // not load transfer.
  const uf = rec.upFrac;
  let r = 0;
  r += 3 * clamp((s.x - (0.12 - 0.20)) / 0.20);              // where it ended up
  r += 1.5 * clamp((rec.maxX - (0.12 - 0.20)) / 0.20);       // how far it ever got
  r += s.up ? 2 * s.feetOnTread : 0;
  r += 4 * clamp(s.above / 0.095);
  r += s.up ? 1 : 0;
  r += 8 * uf * clamp(gain / rise);                          // LOAD TRANSFER
  r += 3 * uf * rec.sustainFrac;                             // sustained
  r += 2 * uf * clamp((rec.maxZ - rec.z0Settle) / rise);     // peak lift
  if (uf > 0.5) r += (rec.feetOnTreadMax >= 1 ? 4 : 0) + (rec.feetOnTreadMax >= 2 ? 6 : 0);
  if (rec.crit.honest) r += 50;
  return { r, why: mode(rec) };
}
function mode(rec) {
  const s = rec.scored;
  if (rec.maxAbsDY > 0.17) return 'off the flight';
  if (!s.up) return 'fell';
  if (rec.crit.honest) return 'CLEARED';
  if (rec.maxX <= 0.12) return 'never reached the riser';
  if (rec.feetOnTreadMax === 0) return 'past the riser, no foot ever on the tread';
  if (s.feetOnTread < 2) return 'foot touched the tread, not both at the end';
  return 'both feet on the tread, not standing';
}

// ---------------------------------------------------------------- eval
let EVALS = 0;
const HIST = [];
/** Write the candidate, then score it FROM THE FILE. No in-memory scoring. */
async function evalGene(g, rise, stepCount, tag) {
  const path = `${TMP}/cand_${tag}.json`;
  saveTrack(intentOf(g), path);
  let rec;
  try {
    rec = await scoreSaved(path, { rise, tail: 'policy', overrides: { stepCount } });
  } catch (e) { return { r: -1, why: 'error: ' + e.message, rec: null }; }
  EVALS++;
  const sc = score(rec, rise);
  return { r: sc.r, why: sc.why, rec };
}

const brief = (rec, r) => ({
  reward: +r.toFixed(3),
  honest: rec.crit.honest, lat: rec.crit.lat, orig: rec.crit.orig,
  x_mm: +mm(rec.scored.x), dy_mm: +mm(rec.scored.dy), z_mm: +mm(rec.scored.z),
  above_mm: +mm(rec.scored.above), up: rec.scored.up,
  feetOnTread: rec.scored.feetOnTread, feetOnTreadMax: rec.feetOnTreadMax,
  maxX_mm: +mm(rec.maxX), peakZ_mm: +mm(rec.maxZ), maxAbsDY_mm: +mm(rec.maxAbsDY),
  headFrac: +rec.headFrac.toFixed(3), riserFrac: +rec.riserFrac.toFixed(3),
  bothFrac: +rec.bothFrac.toFixed(3), sustainFrac: +rec.sustainFrac.toFixed(3),
  maxGainBoth_mm: rec.maxGainBoth == null ? null : +mm(rec.maxGainBoth),
  liftIntegral: +rec.liftIntegral.toFixed(4),
  satFrac: +rec.satFrac.toFixed(3), upFrac: +rec.upFrac.toFixed(3),
  minStepGap_mm: rec.minStepGap_mm == null ? null : +rec.minStepGap_mm.toFixed(2),
  maxTreadDriftX_mm: +rec.maxTreadDriftX_mm.toFixed(2),
});

// ================================================================== run
const results = { generated: new Date().toISOString(), family: 'C — corner stem, contact order',
                  criterion: "rig3 honest, tail=policy", budget_s: BUDGET_S, arms: {}, guard: null };

// ---- GUARD: the instrumented rig3 still reproduces a published number.
// rig3.log Phase E, 180 mm, 4 steps, on-tread control: z = 288.7 mm, HONEST PASS.
{
  const g = await scoreSaved('../climb/ctrl_on_tread_180mm_x26_med.json',
    { rise: 0.180, tail: 'policy', overrides: { stepCount: 4 } });
  const d = await scoreSaved('../climb/ctrl_do_nothing.json',
    { rise: 0.180, tail: 'policy', overrides: { stepCount: 4 } });
  results.guard = { onTread180_z_mm: +mm(g.scored.z), onTread180_honest: g.crit.honest,
                    expected_z_mm: 288.7, doNothing180_honest: d.crit.honest };
  console.log(`GUARD  on-tread 180 mm control: z = ${mm(g.scored.z)} mm (rig3.log says 288.7), honest = ${g.crit.honest}` +
              `  |  do-nothing 180 mm honest = ${d.crit.honest}`);
  if (!g.crit.honest || Math.abs(g.scored.z * 1000 - 288.7) > 0.2 || d.crit.honest) {
    console.log('GUARD FAILED — the instrument moved. Stopping.');
    fs.writeFileSync(OUT, JSON.stringify(results, null, 2)); process.exit(1);
  }
}

const ARMS = [
  { rise: 0.180, stepCount: 4, seed: 0x51CB180, share: 0.5,
    note: 'canonical four-step flight; rig3 Phase E says the on-tread control passes here' },
  { rise: 0.120, stepCount: 1, seed: 0x51CB120, share: 0.5,
    note: 'ONE-BLOCK flight: with four steps the blocks shove each other and throw a standing duck off at 120 mm' },
];

for (const arm of ARMS) {
  const rmm = Math.round(arm.rise * 1000);
  const tag = `${rmm}_${arm.stepCount}`;
  const deadline = el() + BUDGET_S * arm.share;
  const r = mulberry32(arm.seed);
  console.log(`\n=== ARM ${rmm} mm, ${arm.stepCount}-step flight — ${arm.note}`);
  console.log(`    budget to t=${deadline.toFixed(0)}s`);

  const perOrder = ORDERS.map(() => null);       // elite gene+score per order
  const improvements = [];
  let best = null;

  const consider = (g, out, phase) => {
    if (out.r < 0 || !out.rec) return false;
    const o = g.order;
    const isOrderBest = !perOrder[o] || out.r > perOrder[o].r;
    if (isOrderBest) perOrder[o] = { g, r: out.r, rec: out.rec };
    if (!best || out.r > best.r) {
      best = { g, r: out.r, rec: out.rec, why: out.why, phase };
      const b = brief(out.rec, out.r);
      improvements.push({ t_s: +el().toFixed(1), evals: EVALS, phase,
        order: ORDERS[g.order].join(''), seq: trackOf(g).seq.join(''),
        lead: g.lead ? 'R' : 'L', liftAt: g.liftAt, retractAt: g.retractAt, ...b, why: out.why });
      console.log(`  + [${el().toFixed(0)}s ev${EVALS}] ${phase} order=${ORDERS[g.order].join('')} ` +
        `seq=${trackOf(g).seq.join('')} lead=${g.lead ? 'R' : 'L'} lift@${g.liftAt} retr@${g.retractAt} ` +
        `r=${out.r.toFixed(3)}  x=${b.x_mm} z=${b.z_mm} above=${b.above_mm} feetTread=${b.feetOnTread}/${b.feetOnTreadMax} ` +
        `head=${b.headFrac} riser=${b.riserFrac} gainBoth=${b.maxGainBoth_mm} :: ${out.why}`);
      return true;
    }
    return false;
  };

  // ---- PHASE A: every order, random. The point of the arm.
  const PER_ORDER = 9;
  for (let i = 0; i < PER_ORDER; i++) {
    for (let o = 0; o < ORDERS.length; o++) {
      if (el() > deadline) break;
      const g = randGene(r); g.order = o;
      consider(g, await evalGene(g, arm.rise, arm.stepCount, tag), 'A');
    }
  }
  const ranked = perOrder.map((e, o) => ({ o, r: e ? e.r : -1 })).sort((a, b) => b.r - a.r);
  console.log(`  -- phase A done (t=${el().toFixed(0)}s, ${EVALS} evals). order ranking: ` +
    ranked.map(x => `${ORDERS[x.o].join('')}=${x.r.toFixed(2)}`).join('  '));

  // ---- PHASE B: hill-climb the three best orders, keeping the order fixed.
  const KEEP = ranked.slice(0, 3).map(x => x.o).filter(o => perOrder[o]);
  const sigmas = [0.30, 0.18, 0.10, 0.05];
  let round = 0;
  while (el() < deadline) {
    const s = sigmas[round % sigmas.length];
    for (const o of KEEP) {
      if (el() > deadline) break;
      for (let k = 0; k < 4 && el() < deadline; k++) {
        const g = jitter(perOrder[o].g, s, r, false); g.order = o;
        const out = await evalGene(g, arm.rise, arm.stepCount, tag);
        consider(g, out, `B/s=${s}`);
      }
    }
    round++;
    // every 6 rounds, one fresh random restart into each kept order
    if (round % 6 === 0) for (const o of KEEP) {
      if (el() > deadline) break;
      const g = randGene(r); g.order = o;
      consider(g, await evalGene(g, arm.rise, arm.stepCount, tag), 'B/restart');
    }
  }
  console.log(`  -- arm done t=${el().toFixed(0)}s, ${EVALS} evals total. best r=${best ? best.r.toFixed(3) : 'n/a'}`);

  // ---- publish the best, then RE-SCORE IT FROM ITS PUBLISHED PATH.
  const bestPath = `../climb/best_r2_cornerstem_${rmm}mm.json`;
  saveTrack(intentOf(best.g, `FAMILY C corner stem, contact order ${ORDERS[best.g.order].join('')} ` +
    `(full sequence ${trackOf(best.g).seq.join('')}), lead ${best.g.lead ? 'R' : 'L'}, ` +
    `rise ${rmm} mm, ${arm.stepCount}-step flight. Scored by climb/rig3.mjs honest criterion, tail=policy.`),
    bestPath);
  const re = await scoreSaved(bestPath, { rise: arm.rise, tail: 'policy', overrides: { stepCount: arm.stepCount } });
  const reR = score(re, arm.rise);
  console.log(`  RE-SCORED FROM ${bestPath}: r=${reR.r.toFixed(3)} (search said ${best.r.toFixed(3)}), honest=${re.crit.honest}`);

  results.arms[`${rmm}mm`] = {
    rise_mm: rmm, stepCount: arm.stepCount, seed: arm.seed, note: arm.note,
    evals: EVALS, orderRanking: ranked.map(x => ({ order: ORDERS[x.o].join(''), best_r: +x.r.toFixed(3) })),
    perOrderBest: perOrder.map((e, o) => e ? { order: ORDERS[o].join(''), ...brief(e.rec, e.r), why: mode(e.rec) } : null),
    improvements, bestPath,
    best: { ...brief(best.rec, best.r), why: best.why, order: ORDERS[best.g.order].join(''),
            sequence: trackOf(best.g).seq.join(''), lead: best.g.lead ? 'R' : 'L' },
    reScoredFromFile: { ...brief(re, reR.r), why: reR.why },
    reproduces: Math.abs(reR.r - best.r) < 1e-9,
  };
  fs.writeFileSync(OUT, JSON.stringify(results, null, 2));
}

// ---------------------------------------------------------------- +/-10 mm
// Every arm's published best, re-run at -10/0/+10 mm on BOTH flights. The brief
// asks for this on a success; running it unconditionally is what says whether a
// near-miss is a knife-edge or a plateau.
console.log('\n=== +/-10 mm AND BOTH FLIGHTS, scored from the published files ===');
console.log('file                                  rise  steps      r   x_mm   z_mm above_mm feetTread head riser gainBoth  HONEST');
results.robustness = [];
for (const arm of ARMS) {
  const rmm = Math.round(arm.rise * 1000);
  const p = `../climb/best_r2_cornerstem_${rmm}mm.json`;
  for (const dr of [-10, 0, 10]) {
    const h = (rmm + dr) / 1000;
    for (const n of [arm.stepCount, arm.stepCount === 4 ? 1 : 4]) {
      const rec = await scoreSaved(p, { rise: h, tail: 'policy', overrides: { stepCount: n } });
      const sc = score(rec, h);
      const b = brief(rec, sc.r);
      results.robustness.push({ file: p, rise_mm: rmm + dr, stepCount: n, ...b, why: sc.why });
      console.log(`${p.replace('../climb/', '').padEnd(38)}${String(rmm + dr).padStart(4)}${String(n).padStart(7)}` +
        `${sc.r.toFixed(2).padStart(7)}${b.x_mm.toFixed(1).padStart(7)}${b.z_mm.toFixed(1).padStart(7)}` +
        `${b.above_mm.toFixed(1).padStart(9)}${String(b.feetOnTread).padStart(10)}${b.headFrac.toFixed(2).padStart(5)}` +
        `${b.riserFrac.toFixed(2).padStart(6)}${String(b.maxGainBoth_mm).padStart(9)}   ${rec.crit.honest ? 'PASS' : ' . '}`);
    }
  }
}

results.totalEvals = EVALS;
results.elapsed_s = +el().toFixed(1);
fs.writeFileSync(OUT, JSON.stringify(results, null, 2));
console.log(`\nWROTE ${OUT}  (${EVALS} episodes, ${el().toFixed(0)}s)`);
