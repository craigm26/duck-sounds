// FAMILY C — ROUND 3. THE 120 / 180 mm BAND, AS A WHOLE-BODY CLIMB.
//
// Round 2 (climb/search_3.mjs, climb/best_r2_cornerstem_{120,180}mm.json)
// searched a three-contact "corner stem" and got a stem that LIFTS BUT DOES
// NOT TRANSLATE: at 180 mm the trunk peaked at 153 mm with head contact 0.72
// and foot-on-riser 0.52, and trunk max x reached 90.5 mm against a riser face
// at 120 mm. Re-scored today on the REPAIRED flight over the 9-cell grid, both
// round-2 bests clear 0 of 9 (120 mm objective 2.774, 180 mm 1.133) and the
// worse of the two feet never comes within 206 mm of the landing spot.
//
// So the missing mechanism is not another contact. It is the TRANSFER: a push
// off the trailing leg while the head and the lead foot bear, and then the two
// feet actually arriving on the tread. Round 2 had no stage that put a foot on
// the tread at all; `honest` needs BOTH feet resting there.
//
// WHAT IS PARAMETRISED HERE THAT ROUND 2 DID NOT HAVE
//   opening order  which of {F lead foot on the riser face, H beak on the
//                  tread, W outboard leg braced on the side wall} is
//                  established first — 6 permutations, cumulative, plus a
//                  useW gene that drops the wall brace entirely.
//   P  THE PUSH    the trailing leg extends against the FLOOR while F and H
//                  bear. This is the load transfer. Round 2's 'T' only lifted
//                  the trail foot; it never drove.
//   K  THE ROLL-UP the lead foot leaves the riser FACE and arrives on the
//                  TREAD. Round 2 never had this stage, which is why its
//                  feetOnTread was 0 by construction.
//   S  THE SWING   the trail foot follows onto the tread — the second half of
//                  `honest`.
//   pr             whether the neck haul R comes before or after the push P.
//   E              a stance frame on the tread so the 50-tick stand-policy
//                  tail has something to stabilise.
// The neck stall is a hard fact and is used as one: 7.66 N through the
// 0.0836 m lever against 7.23 N of body weight means the head cannot lift the
// duck, so R is scored only through what it does to trunk z WHILE A FOOT ALSO
// BEARS (robust.mjs maxGainBoth / sustainFrac), never on head contact time.
//
// SCORING. climb/robust.mjs — the one round-3 scorer, imported, not copied.
// Every number comes from a SAVED FILE: a candidate is written to disk with
// the exact rounding it will be published with and then scored from that path.
// The published objective is robust.mjs's own: meanReward over the 9 cells
// plus CLEAR_BONUS per cell cleared under rig3's `honest`. The SEARCH also
// carries load-transfer and landing-proximity shaping on top (shaped()), because
// meanReward alone is flat across every candidate that never lands.
//
// Flight: the canonical 4-step flight, repaired (site/stairs.js isolates the
// step geoms). approach is FIXED AT 0 — BEST_alpha_stand.onnx advances 4 mm in
// 8 s, so an approach gene is a wasted dimension.
//
// One process, mulberry32 seeded from 16838.
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/family_c.mjs
import fs from 'node:fs';
import { scoreCell, scoreRobust, saveIntent, HOME, LO, HI, PLANTS, DHS, CLEAR_BONUS }
  from '../climb/robust.mjs';

const SEED_BASE = 16838;
const BUDGET_S = +(process.env.BUDGET_S || 2280);
const TMP = process.env.TMPDIR
  || '/tmp/claude-1000/-home-craigm26/43f07dce-e62f-464d-ae1f-2fe020620950/scratchpad';
const OUT = '../climb/r3_cornerclimb-results.json';
const T00 = Date.now();
const el = () => (Date.now() - T00) / 1000;
const mm = v => (v * 1000).toFixed(1);
const clip = v => Math.max(0, Math.min(1, v));

export function mulberry32(a) {
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

export const CONT = {
  // spawn: within one duck-length of the riser at x = 0.12. approach is not a gene.
  gap:  [-0.02, 0.10], side: [-0.06, 0.14], blend: [0.70, 2.40],
  // F — lead foot onto the riser FACE
  hipF: [-0.20, 1.60], kneeF: [-1.50, 0.80], ankF: [-1.40, 0.90],
  rollF: [-0.38, 0.38], yawF: [-0.43, 0.52],
  // H — plant the beak on the tread (trunk pitch is not a servo; the hips are)
  hipH: [0.00, 1.40], kneeH: [-1.00, 1.00], ankH: [-1.40, 0.60],
  neckH: [-1.57, 1.04], headH: [-1.40, 1.50],
  // W — brace the outboard leg on the side wall
  rollW: [-0.38, 0.38], yawW: [-0.43, 0.52],
  // P — THE PUSH: the trailing leg drives against the floor
  hipP: [-1.40, 1.00], kneeP: [-1.40, 1.00], ankP: [-1.40, 1.00],
  // R — the neck haul
  npR: [-1.57, 1.04], hpR: [-1.40, 1.50], hipR: [-1.40, 1.40], kneeR: [-1.40, 1.40],
  // K — the lead foot leaves the face and arrives on the TREAD
  hipK: [-0.20, 1.60], kneeK: [-1.50, 1.00], ankK: [-1.40, 1.00],
  // S — the trail foot follows onto the tread
  hipS: [-0.20, 1.60], kneeS: [-1.50, 1.00], ankS: [-1.40, 1.00], rollS: [-0.38, 0.38],
  // E — stance on the tread for the policy tail to catch
  hipE: [-1.00, 1.00], kneeE: [-1.00, 1.00], ankE: [-1.20, 0.90], npE: [-1.57, 1.04],
  d1: [0.10, 0.45], d2: [0.10, 0.45], d3: [0.10, 0.45], d4: [0.10, 0.45], d5: [0.10, 0.45],
  d6: [0.10, 0.45], d7: [0.10, 0.45], d8: [0.10, 0.45], d9: [0.10, 0.45],
};
const CKEYS = Object.keys(CONT);
export const ORDERS = [['F','H','W'], ['F','W','H'], ['H','F','W'],
                ['H','W','F'], ['W','F','H'], ['W','H','F']];
export const DISC = { order: ORDERS.length, lead: 2, pr: 2, useW: 2, endHome: 2 };
const DKEYS = Object.keys(DISC);

export function randGene(r) {
  const g = {};
  for (const k of CKEYS) { const [a, b] = CONT[k]; g[k] = a + r() * (b - a); }
  for (const k of DKEYS) g[k] = Math.floor(r() * DISC[k]);
  return g;
}
export function clampGene(g) {
  const o = {};
  for (const k of CKEYS) { const [a, b] = CONT[k]; o[k] = Math.min(b, Math.max(a, g[k] ?? (a + b) / 2)); }
  for (const k of DKEYS) o[k] = Math.max(0, Math.min(DISC[k] - 1, Math.round(g[k] ?? 0)));
  return o;
}
export function jitter(g, s, r) {
  const o = {};
  for (const k of CKEYS) {
    const [a, b] = CONT[k];
    o[k] = Math.min(b, Math.max(a, g[k] + (r() * 2 - 1) * (b - a) * s));
  }
  for (const k of DKEYS) o[k] = (r() < (k === 'order' ? 0.05 : 0.12)) ? Math.floor(r() * DISC[k]) : g[k];
  return o;
}
export function cross(a, b, r) {
  const o = {};
  for (const k of CKEYS) o[k] = r() < 0.5 ? a[k] : b[k];
  for (const k of DKEYS) o[k] = r() < 0.5 ? a[k] : b[k];
  return o;
}

// ---------------------------------------------------------------- the track
const legIdx = lead => lead === 0
  ? { hip: J.lhp, knee: J.lk, ank: J.la, roll: J.lhr, yaw: J.lhy, s: +1 }
  : { hip: J.rhp, knee: J.rk, ank: J.ra, roll: J.rhr, yaw: J.rhy, s: -1 };
const cl = (v, k) => Math.min(HI[k], Math.max(LO[k], v));

/**
 * Compile a gene into a cumulative keyframe track.
 *
 * Stages are CUMULATIVE: each frame starts from the previous pose and adds its
 * own contact, so the order gene changes which contact is made while which
 * others are already held. A stage never silently un-does a contact an earlier
 * stage established: H and W touch the LEAD leg only if F has not run yet.
 */
export function trackOf(g) {
  const L = legIdx(g.lead), T = legIdx(1 - g.lead);
  let seq = ORDERS[g.order].filter(t => t !== 'W' || g.useW);
  seq = seq.concat(g.pr ? ['R', 'P'] : ['P', 'R'], ['K', 'S', 'E']);
  const dur = [g.d1, g.d2, g.d3, g.d4, g.d5, g.d6, g.d7, g.d8, g.d9];
  let p = HOME.slice(), t = 0, placedF = false;
  const frames = [];
  seq.forEach((tok, i) => {
    p = p.slice();
    if (tok === 'F') {
      p[L.hip]  = HOME[L.hip]  + L.s * g.hipF;
      p[L.knee] = HOME[L.knee] + L.s * g.kneeF;
      p[L.ank]  = HOME[L.ank]  + L.s * g.ankF;
      p[L.roll] = HOME[L.roll] + g.rollF;
      p[L.yaw]  = HOME[L.yaw]  + L.s * g.yawF;
      placedF = true;
    } else if (tok === 'H') {
      // the trunk pitches nose-down over the trailing leg; the lead leg only
      // joins in if it is not already on the riser face
      p[T.hip]  = HOME[T.hip]  + T.s * g.hipH;
      p[T.knee] = HOME[T.knee] + T.s * g.kneeH;
      p[T.ank]  = HOME[T.ank]  + T.s * g.ankH;
      if (!placedF) {
        p[L.hip]  = HOME[L.hip]  + L.s * g.hipH;
        p[L.knee] = HOME[L.knee] + L.s * g.kneeH;
        p[L.ank]  = HOME[L.ank]  + L.s * g.ankH;
      }
      p[J.np] = g.neckH; p[J.hp] = g.headH;
    } else if (tok === 'W') {
      p[T.roll] = HOME[T.roll] + g.rollW;
      p[T.yaw]  = HOME[T.yaw]  + T.s * g.yawW;
      p[L.yaw]  = HOME[L.yaw]  + L.s * g.yawW;
      if (!placedF) p[L.roll] = HOME[L.roll] + g.rollW;
    } else if (tok === 'P') {
      p[T.hip]  = HOME[T.hip]  + T.s * g.hipP;
      p[T.knee] = HOME[T.knee] + T.s * g.kneeP;
      p[T.ank]  = HOME[T.ank]  + T.s * g.ankP;
    } else if (tok === 'R') {
      p[J.np] = g.npR; p[J.hp] = g.hpR;
      p[L.hip]  = HOME[L.hip]  + L.s * g.hipR;
      p[L.knee] = HOME[L.knee] + L.s * g.kneeR;
    } else if (tok === 'K') {
      p[L.hip]  = HOME[L.hip]  + L.s * g.hipK;
      p[L.knee] = HOME[L.knee] + L.s * g.kneeK;
      p[L.ank]  = HOME[L.ank]  + L.s * g.ankK;
    } else if (tok === 'S') {
      p[T.hip]  = HOME[T.hip]  + T.s * g.hipS;
      p[T.knee] = HOME[T.knee] + T.s * g.kneeS;
      p[T.ank]  = HOME[T.ank]  + T.s * g.ankS;
      p[T.roll] = HOME[T.roll] + g.rollS;
    } else if (tok === 'E') {
      p[J.lhp] = HOME[J.lhp] + g.hipE;  p[J.rhp] = HOME[J.rhp] - g.hipE;
      p[J.lk]  = HOME[J.lk]  + g.kneeE; p[J.rk]  = HOME[J.rk]  - g.kneeE;
      p[J.la]  = HOME[J.la]  + g.ankE;  p[J.ra]  = HOME[J.ra]  - g.ankE;
      p[J.lhr] = HOME[J.lhr]; p[J.rhr] = HOME[J.rhr];
      p[J.lhy] = HOME[J.lhy]; p[J.rhy] = HOME[J.rhy];
      p[J.np] = g.npE; p[J.hp] = HOME[J.hp];
    }
    t += dur[i % dur.length];
    frames.push({ t, pose: p.map((v, k) => cl(v, k)) });
  });
  if (g.endHome) {
    const e = HOME.slice(); e[J.np] = g.npE;
    t += 0.35; frames.push({ t, pose: e.map((v, k) => cl(v, k)) });
  }
  return { frames, seq };
}

/** The intent object AS IT WILL BE PUBLISHED — rounded here, once. */
export function intentOf(g, note) {
  const { frames, seq } = trackOf(g);
  return {
    name: 'cornerclimb', family: 'C_whole_body_corner_climb_r3',
    order: ORDERS[g.order].join(''), sequence: seq.join(''),
    lead: g.lead ? 'R' : 'L', pr: g.pr ? 'R_before_P' : 'P_before_R',
    useW: !!g.useW, endHome: !!g.endHome,
    keyframes: frames.map(f => ({ t: +f.t.toFixed(4), pose: f.pose.map(v => +v.toFixed(5)) })),
    blend: +g.blend.toFixed(4), gap: +g.gap.toFixed(4), side: +g.side.toFixed(4),
    approach: 0,
    gene: Object.fromEntries(Object.entries(g).map(([k, v]) => [k, +(+v).toFixed(6)])),
    note: note || '',
  };
}

// ---------------------------------------------------------------- reward
/**
 * SEARCH shaping for ONE cell. rig3's reward is kept whole and the brief's
 * load-transfer terms are added on top of it:
 *   maxGainBoth   trunk z gain while the head AND a foot both bear  (THE term)
 *   sustainFrac   fraction of the episode >20 mm up in that state
 *   footNear/bothNear  how close each foot ever came to a landing spot on the
 *                 tread — the gradient round 2 did not have, so a candidate
 *                 that reaches toward the tread outscores one that only lifts
 * The lateral gate is hard and comes first, exactly as in rig3.
 * PUBLISHED objective is robust.mjs's own (meanReward + CLEAR_BONUS*k); this
 * function only ranks the search.
 */
function shaped(c) {
  if (c.maxAbsDY > 0.17) return 0;
  let s = c.reward;                                              // 0..12, rig3's
  s += 3.0 * clip((c.maxZ - c.z0Settle) / 0.12);                 // peak lift
  s += 4.0 * clip((c.maxGainBoth || 0) / 0.10);                  // LOAD TRANSFER
  s += 2.0 * c.sustainFrac;
  s += 1.0 * clip(c.liftIntegral / 6);
  s += 4.0 * clip(1 - c.footNear / 0.28);                        // one foot reaching the tread
  s += 4.0 * clip(1 - c.bothNear / 0.32);                        // both feet reaching it
  s += 1.5 * c.feetOnTreadMax + 0.8 * c.feetHighMax;
  s += 2.0 * clip((c.maxX - 0.02) / 0.20);                       // TRANSLATION, round 2's gap
  if (c.crit.honest) s += 10;
  if (c.crit.honest60) s += 3;
  return s;
}
const shapedMean = cells => cells.reduce((a, c) => a + shaped(c), 0) / cells.length;

// ------------------------------------------------- the "how far is the band"
/**
 * THE FALLBACK NUMBER the brief asks for: the tallest rise at which the trunk
 * is above the tread WITH A FOOT RESTING ON IT. Two readings, kept apart:
 *   simultaneous — both true at the scored instant, so it is one posture
 *   overEpisode  — a foot rested on the tread at SOME tick and the trunk peaked
 *                  above the tread at SOME tick; weaker, reported separately
 * Every 9-cell grid computed anywhere in the run feeds this, so it is measured
 * over the whole search, not just over the published bests.
 */
const SIM = { simultaneous_mm: 0, overEpisode_mm: 0, where: null };
function noteSim(G, tunedMM) {
  for (const c of G.cells) {
    const h = Math.round(c.rise * 1000);
    if (c.scored.feetOnTread >= 1 && c.scored.above > 0 && h > SIM.simultaneous_mm) {
      SIM.simultaneous_mm = h;
      SIM.where = { tunedMM, cell: c.cell, above_mm: +mm(c.scored.above),
                    feetOnTread: c.scored.feetOnTread, up: c.scored.up, honest: c.crit.honest };
    }
    if (c.feetOnTreadMax >= 1 && c.maxZ > c.rise && h > SIM.overEpisode_mm) SIM.overEpisode_mm = h;
  }
}

// ---------------------------------------------------------------- seeds
/** Round 2's own corner-stem gene, remapped onto the round-3 stage names. */
function seedFromR2(path) {
  let j; try { j = JSON.parse(fs.readFileSync(path, 'utf8')); } catch { return null; }
  const q = j.gene || {};
  return clampGene({
    gap: q.gap ?? 0.02, side: q.side ?? 0.03, blend: q.blend ?? 1.2,
    hipF: q.hipF, kneeF: q.kneeF, ankF: q.ankF, rollF: q.rollF, yawF: 0,
    hipH: q.hipLean, kneeH: q.kneeFold, ankH: q.ankLean, neckH: q.neck, headH: q.headP,
    rollW: q.rollW, yawW: q.yawW,
    hipP: q.hipT, kneeP: q.kneeT, ankP: q.ankT,
    npR: q.npEnd, hpR: q.hpEnd, hipR: q.hipEnd, kneeR: q.kneeEnd,
    hipK: 0.9, kneeK: -0.5, ankK: 0.2,
    hipS: 0.9, kneeS: -0.5, ankS: 0.2, rollS: 0,
    hipE: 0.15, kneeE: 0, ankE: 0, npE: 0.35,
    d1: q.d1, d2: q.d2, d3: q.d3, d4: q.d4, d5: q.d5, d6: q.d6,
    d7: 0.3, d8: 0.3, d9: 0.35,
    order: q.order ?? 2, lead: q.lead ?? 1, pr: 0, useW: 1, endHome: 0,
  });
}
/**
 * A hand prior for the mechanism the brief describes: lead foot high on the
 * riser face, beak on the tread, trail leg drives, neck hauls, lead foot rolls
 * up onto the tread, trail foot follows, stand.
 */
function seedHand() {
  return clampGene({
    gap: 0.01, side: 0.04, blend: 1.6,
    hipF: 1.15, kneeF: -0.95, ankF: -0.55, rollF: 0.05, yawF: 0.0,
    hipH: 0.95, kneeH: -0.45, ankH: -0.85, neckH: -0.55, headH: 0.55,
    rollW: 0.18, yawW: 0.12,
    hipP: -0.55, kneeP: 0.45, ankP: 0.55,
    npR: 0.75, hpR: -0.35, hipR: 1.20, kneeR: -0.85,
    hipK: 1.35, kneeK: -1.10, ankK: 0.55,
    hipS: 1.30, kneeS: -1.00, ankS: 0.50, rollS: -0.10,
    hipE: 0.25, kneeE: -0.10, ankE: 0.05, npE: 0.35,
    d1: 0.28, d2: 0.24, d3: 0.20, d4: 0.22, d5: 0.20, d6: 0.26, d7: 0.24, d8: 0.28, d9: 0.32,
    order: 0, lead: 1, pr: 0, useW: 1, endHome: 0,
  });
}

// ---------------------------------------------------------------- the arm
const CELL_SCREEN = { dh: 0, drop: 0.120, fmul: 1.0 };
/** A diagonal 3-cell screen: it moves BOTH grid axes for a third of the cost. */
const CELL_TRI = [
  { dh: 0.000, drop: 0.120, fmul: 1.0 },
  { dh: -0.010, drop: 0.130, fmul: 0.7 },
  { dh: 0.010, drop: 0.125, fmul: 1.3 },
];

async function searchArm(riseMM, seed, armBudget) {
  const rise = riseMM / 1000;
  const r = mulberry32(seed);
  const tag = `c${riseMM}`;
  const tmp = `${TMP}/r3_${tag}.json`;
  const T0 = Date.now();
  const left = () => armBudget - (Date.now() - T0) / 1000;
  const log = (...a) => console.log(`[${el().toFixed(0)}s ${riseMM}mm]`, ...a);
  let nEp = 0;

  const one = async (g, cell) => {
    saveIntent(intentOf(g), tmp);
    const c = await scoreCell(tmp, { rise, dh: cell.dh, drop: cell.drop, fmul: cell.fmul });
    nEp++; return c;
  };
  const tri = async g => {
    saveIntent(intentOf(g), tmp);
    const out = [];
    for (const cell of CELL_TRI) {
      out.push(await scoreCell(tmp, { rise, dh: cell.dh, drop: cell.drop, fmul: cell.fmul }));
      nEp++;
    }
    return out;
  };
  const grid = async g => {
    saveIntent(intentOf(g), tmp); nEp += 9;
    const G = await scoreRobust(tmp, { rise });
    noteSim(G, riseMM);           // every grid feeds the "how far is the band" number
    return G;
  };

  // ---- PHASE A: random exploration + seeds, screened on the nominal cell
  const pool = [];
  const seeds = [seedHand(), seedFromR2(`../climb/best_r2_cornerstem_${riseMM}mm.json`),
                 seedFromR2('../climb/best_r2_cornerstem_180mm.json')].filter(Boolean);
  let bestShaped = -1, bestG = null, bestGrid = null;
  const aEnd = armBudget * 0.34;
  for (const g of seeds) {
    const c = await one(g, CELL_SCREEN);
    pool.push({ g, s: shaped(c), sig: 0.16, fail: 0 });
    log(`seed shaped=${shaped(c).toFixed(2)} rew=${c.reward.toFixed(2)} peakZ=${mm(c.maxZ)} bothNear=${mm(c.bothNear)}`);
  }
  while ((Date.now() - T0) / 1000 < aEnd) {
    const g = randGene(r);
    const c = await one(g, CELL_SCREEN);
    const s = shaped(c);
    pool.push({ g, s, sig: 0.16, fail: 0 });
    if (s > bestShaped) {
      bestShaped = s; bestG = g;
      const G = await grid(g);
      bestGrid = G;
      log(`A improve shaped=${s.toFixed(2)} k=${G.k}/9 obj=${G.objective.toFixed(3)} peakZ=${mm(G.agg.maxZ)} bothNear=${G.agg.bothNear_mm.toFixed(0)}mm seq=${intentOf(g).sequence}`);
    }
  }
  pool.sort((a, b) => b.s - a.s);
  const elites = pool.slice(0, 14);
  log(`phase A done: ${nEp} episodes, pool ${pool.length}, elite shaped ${elites.map(e => e.s.toFixed(1)).join(' ')}`);

  // ---- PHASE B: hill climb on the elites, 3-cell diagonal screen
  for (const e of elites) { e.cells = await tri(e.g); e.s3 = shapedMean(e.cells); }
  elites.sort((a, b) => b.s3 - a.s3);
  let best3 = elites[0].s3, best3G = elites[0].g;
  {
    const G = await grid(best3G); bestGrid = G; bestShaped = best3; bestG = best3G;
    log(`B start shaped3=${best3.toFixed(2)} k=${G.k}/9 obj=${G.objective.toFixed(3)} peakZ=${mm(G.agg.maxZ)}`);
  }
  const bEnd = armBudget * 0.86;
  let it = 0;
  while ((Date.now() - T0) / 1000 < bEnd) {
    it++;
    const i = Math.floor(r() * elites.length);
    const e = elites[i];
    let g;
    if (r() < 0.20 && elites.length > 1) {
      const jx = Math.floor(r() * elites.length);
      g = jitter(cross(e.g, elites[jx].g, r), e.sig * 0.5, r);
    } else g = jitter(e.g, e.sig, r);
    const cells = await tri(g);
    const s3 = shapedMean(cells);
    if (s3 > e.s3) {
      e.g = g; e.s3 = s3; e.cells = cells; e.fail = 0; e.sig = Math.min(0.30, e.sig * 1.25);
      if (s3 > best3) {
        best3 = s3; best3G = g;
        const G = await grid(g);
        const bk = bestGrid ? bestGrid.k : -1, bo = bestGrid ? bestGrid.objective : -1e9;
        const adopt = G.k > bk || (G.k === bk && G.objective > bo);
        if (adopt) { bestGrid = G; bestG = g; bestShaped = s3; }
        noteSim(G, riseMM);
        log(`B improve it=${it} shaped3=${s3.toFixed(2)} k=${G.k}/9 obj=${G.objective.toFixed(3)} ${adopt ? 'ADOPT' : 'keep '} peakZ=${mm(G.agg.maxZ)} gainBoth=${(G.cells.reduce((a,c)=>a+(c.maxGainBoth||0),0)/9*1000).toFixed(1)}mm head=${G.agg.headFrac.toFixed(2)} riser=${G.agg.riserFrac.toFixed(2)} wall=${G.agg.wallFrac.toFixed(2)} fotMax=${G.agg.feetOnTreadMax} bothNear=${G.agg.bothNear_mm.toFixed(0)}mm seq=${intentOf(g).sequence}${G.agg.meanFeetOnTreadMax > 0 ? ' <-FOOT ON TREAD' : ''}`);
      }
    } else {
      e.fail++;
      if (e.fail >= 6) { e.sig = Math.max(0.03, e.sig * 0.6); e.fail = 0; }
    }
    if (left() < 0) break;
  }
  log(`phase B done: ${it} iterations, ${nEp} episodes total`);

  // ---- PHASE C: full 9-cell grid on the survivors, ranked by the PUBLISHED objective
  elites.sort((a, b) => b.s3 - a.s3);
  const finals = [];
  const top = elites.slice(0, 6);
  if (best3G && !top.some(e => e.g === best3G)) top.unshift({ g: best3G, s3: best3 });
  for (const e of top) {
    if (left() < 12) break;
    const G = await grid(e.g);
    finals.push({ g: e.g, G, s3: e.s3 });
    log(`C k=${G.k}/9 obj=${G.objective.toFixed(3)} meanRew=${G.meanReward.toFixed(3)} peakZ=${mm(G.agg.maxZ)} seq=${intentOf(e.g).sequence}`);
  }
  finals.sort((a, b) => (b.G.k - a.G.k) || (b.G.objective - a.G.objective) || (b.s3 - a.s3));
  const win = finals[0] || (bestGrid ? { g: bestG, G: bestGrid, s3: bestShaped } : null);
  return { rise, riseMM, nEp, finals, win, bestGrid, tmp };
}

// ---------------------------------------------------------------- the ladder
/**
 * If nothing clears, THE number that says how far the band is: the tallest rise
 * at which the trunk is above the tread WITH A FOOT RESTING ON IT. Measured at
 * the scored instant (both facts true at the same tick, by construction) and
 * also as an over-episode conjunction (feetOnTreadMax >= 1 and peak trunk z
 * above the tread, not necessarily simultaneous — reported separately).
 */
async function ladder(path, rise, rises) {
  const rows = [];
  for (const h of rises) {
    let sim = false, ep = false, best = { above: -1e9 };
    for (const p of PLANTS) {
      const c = await scoreCell(path, { rise: h / 1000, drop: p.drop, fmul: p.fmul });
      if (c.scored.feetOnTread >= 1 && c.scored.above > 0) sim = true;
      if (c.feetOnTreadMax >= 1 && c.maxZ > h / 1000) ep = true;
      if (c.scored.above > best.above) best = { above: c.scored.above, fot: c.scored.feetOnTread, peakZ: c.maxZ, honest: c.crit.honest };
    }
    rows.push({ rise_mm: h, simultaneous: sim, overEpisode: ep,
                bestAbove_mm: +mm(best.above), fot: best.fot, peakZ_mm: +mm(best.peakZ), honest: best.honest });
  }
  return rows;
}

// ---------------------------------------------------------------- main
// Guarded so a refinement pass can import the ONE track compiler above instead
// of making a second copy of it.
const isMain = process.argv[1] && process.argv[1].endsWith('family_c.mjs');
if (isMain) {
const arms = [];
const perArm = (BUDGET_S - 260) / 2;
for (const [i, riseMM] of [120, 180].entries()) {
  console.log(`\n=== FAMILY C round 3 — rise ${riseMM} mm, seed ${SEED_BASE + i}, budget ${perArm.toFixed(0)} s ===`);
  arms.push(await searchArm(riseMM, SEED_BASE + i, perArm));
}

const results = { family: 'C_whole_body_corner_climb_r3', seedBase: SEED_BASE,
                  budget_s: BUDGET_S, grid: { DHS, PLANTS, CLEAR_BONUS }, arms: [] };
for (const a of arms) {
  if (!a.win) { console.log(`no winner at ${a.riseMM}`); continue; }
  const G = a.win.G;
  const note = `FAMILY C ROUND 3, whole-body corner climb, rise ${a.riseMM} mm on the REPAIRED canonical 4-step flight. `
    + `9-cell robust grid (rise ${a.riseMM - 10}/${a.riseMM}/${a.riseMM + 10} mm x plant {0.120/x1.0, 0.130/x0.7, 0.125/x1.3}): `
    + `cleared ${G.k} of 9 under climb/rig3.mjs 'honest', tail=policy. objective ${G.objective.toFixed(3)} `
    + `= meanReward ${G.meanReward.toFixed(3)} + ${CLEAR_BONUS}*${G.k}. `
    + `trunk peak z ${mm(G.agg.maxZ)} mm (max over cells), mean peak gain over the settled stand ${(G.agg.meanPeakGain * 1000).toFixed(1)} mm; `
    + `head contact ${G.agg.headFrac.toFixed(3)}, foot-on-riser ${G.agg.riserFrac.toFixed(3)}, wall contact ${G.agg.wallFrac.toFixed(3)}, `
    + `wall load-bearing ${G.agg.wallBearFrac.toFixed(3)}, head+foot both bearing ${G.agg.bothFrac.toFixed(3)}, sustained (>20 mm up while both bear) ${G.agg.sustainFrac.toFixed(3)}; `
    + `feet resting on the tread: max over episode ${G.agg.feetOnTreadMax}, at the scored instant ${G.agg.meanFeetOnTreadFinal.toFixed(2)} of 2; `
    + `worse foot's closest approach to a landing spot on the tread ${G.agg.bothNear_mm.toFixed(0)} mm (best foot ${G.agg.footNear_mm.toFixed(0)} mm). `
    + `contact sequence ${intentOf(a.win.g).sequence}, opening order ${ORDERS[a.win.g.order].join('')}, lead foot ${a.win.g.lead ? 'R' : 'L'}. `
    + `Reproduce: cd ~/projects/duck-sounds/sim && node -e "import('../climb/robust.mjs').then(async m=>console.log((await m.scoreRobust('../climb/best_r3_cornerclimb_${a.riseMM}mm.json',{rise:${(a.riseMM / 1000).toFixed(3)}})).k))"`;
  const path = `../climb/best_r3_cornerclimb_${a.riseMM}mm.json`;
  saveIntent(intentOf(a.win.g, note), path);
  const re = await scoreRobust(path, { rise: a.rise });
  console.log(`\nSAVED ${path}  re-scored from the published file: k=${re.k}/9 obj=${re.objective.toFixed(3)} (search said k=${G.k} obj=${G.objective.toFixed(3)})`);
  const lad = await ladder(path, a.rise, [40, 50, 60, 70, 80, 90, 100, 110, 120, 140, 160, 180]);
  console.log(`ladder ${a.riseMM}: ` + lad.map(x => `${x.rise_mm}:${x.simultaneous ? 'SIM' : x.overEpisode ? 'ep' : '-'}(above ${x.bestAbove_mm}mm fot ${x.fot})`).join(' '));
  results.arms.push({
    riseMM: a.riseMM, episodes: a.nEp, published: path,
    k: re.k, objective: re.objective, meanReward: re.meanReward,
    searchK: G.k, searchObjective: G.objective,
    agg: re.agg, verdicts: re.verdicts, sequence: intentOf(a.win.g).sequence,
    order: ORDERS[a.win.g.order].join(''), lead: a.win.g.lead ? 'R' : 'L',
    useW: !!a.win.g.useW, pr: a.win.g.pr ? 'R_before_P' : 'P_before_R',
    ladder: lad,
    runnersUp: a.finals.slice(0, 5).map(f => ({ k: f.G.k, objective: +f.G.objective.toFixed(3),
      peakZ_mm: +mm(f.G.agg.maxZ), sequence: intentOf(f.g).sequence,
      headFrac: +f.G.agg.headFrac.toFixed(3), riserFrac: +f.G.agg.riserFrac.toFixed(3),
      wallFrac: +f.G.agg.wallFrac.toFixed(3), feetOnTreadMax: f.G.agg.feetOnTreadMax,
      bothNear_mm: +f.G.agg.bothNear_mm.toFixed(0) })),
  });
}
// round-2 baselines on the repaired flight, for the record
results.round2Baseline = {};
for (const h of [120, 180]) {
  const p = `../climb/best_r2_cornerstem_${h}mm.json`;
  if (!fs.existsSync(p)) continue;
  const G = await scoreRobust(p, { rise: h / 1000 });
  results.round2Baseline[h] = { k: G.k, objective: +G.objective.toFixed(3),
    peakZ_mm: +mm(G.agg.maxZ), headFrac: +G.agg.headFrac.toFixed(3),
    riserFrac: +G.agg.riserFrac.toFixed(3), wallFrac: +G.agg.wallFrac.toFixed(3),
    feetOnTreadMax: G.agg.feetOnTreadMax, bothNear_mm: +G.agg.bothNear_mm.toFixed(0) };
}
results.reach = SIM;
console.log(`\nHOW FAR IS THE BAND — tallest rise with the trunk above the tread AND a foot resting on it:`);
console.log(`  simultaneous (one posture, at the scored instant): ${SIM.simultaneous_mm} mm  ${SIM.where ? JSON.stringify(SIM.where) : '(never)'}`);
console.log(`  over-episode (foot rested at some tick, trunk peaked above the tread at some tick): ${SIM.overEpisode_mm} mm`);
fs.writeFileSync(OUT, JSON.stringify(results, null, 2));
console.log(`\nwrote ${OUT}   total ${el().toFixed(0)} s`);
for (const a of results.arms) console.log(`  ${a.riseMM} mm: k=${a.k}/9 objective=${a.objective.toFixed(3)} peakZ=${(a.agg.maxZ * 1000).toFixed(1)}mm seq=${a.sequence}`);
}
