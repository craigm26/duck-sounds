// FAMILY C — ROUND 3, REFINEMENT PASS AT 120 mm.
//
// The main pass (climb/family_c.mjs, climb/r3_cornerclimb-results.json) left
// the 120 mm arm one axis short. Its published best clears 0 of 9, but it is
// not flat: in 6 of the 9 cells the episode ENDS with one foot RESTING on the
// tread (rig3's tightened footResting: past the riser line, inside the flight,
// within 3 mm of a step geom), and in the best cell the trunk is 37.2 mm above
// the tread at that instant. Round 2's corner stem put a foot on the tread in
// ZERO cells at either rise.
//
// What is missing is not height and not the first foot. It is:
//   upright   projected-gravity z < -0.90 at the scored instant: 0 of 9 cells
//   the 2nd   foot on the tread: 1 of 2, never 2
// So this pass keeps the SAME track compiler (imported from family_c.mjs — one
// compiler, no second copy) and the SAME 9-cell scorer (robust.mjs), and only
// changes what the search is ranked on: `up` and the second foot are weighted
// far above lift and contact time, which the main pass had already saturated.
//
// Seeded from the published 120 mm best plus jittered restarts around it.
// mulberry32, seed 16840 (the family's seed base 16838 + 2, after the two arms).
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/family_c2.mjs
import fs from 'node:fs';
import { scoreCell, scoreRobust, saveIntent, CLEAR_BONUS } from '../climb/robust.mjs';
import { intentOf, jitter, cross, clampGene, mulberry32, ORDERS } from '../climb/family_c.mjs';

const RISE_MM = +(process.env.RISE_MM || 120);
const RISE = RISE_MM / 1000;
const SEED = +(process.env.SEED || 16840);
const BUDGET_S = +(process.env.BUDGET_S || 700);
const TMP = '/tmp/claude-1000/-home-craigm26/43f07dce-e62f-464d-ae1f-2fe020620950/scratchpad';
const SRC = `../climb/best_r3_cornerclimb_${RISE_MM}mm.json`;
const OUT = '../climb/r3_cornerclimb2-results.json';
const T0 = Date.now();
const el = () => (Date.now() - T0) / 1000;
const mm = v => (v * 1000).toFixed(1);
const clip = v => Math.max(0, Math.min(1, v));
const tmp = `${TMP}/r3_c2_${RISE_MM}.json`;

/**
 * The refinement objective for ONE cell. rig3's reward is still the base, and
 * the two axes the main pass never moved carry the weight:
 *   up                    0 of 9 in the incumbent — the single biggest gap
 *   scored.feetOnTread    1 of 2 in the incumbent; `honest` needs 2
 * Lift and contact-time terms are deliberately small here: the incumbent
 * already peaks 160 mm and rests a foot on the tread, so rewarding them again
 * only buys more of what is not the blocker. maxAbsDY is penalised because the
 * incumbent drifts 114.5 mm off centre inside a 170 mm gate.
 */
function shaped2(c) {
  if (c.maxAbsDY > 0.17) return 0;
  let s = c.reward;
  s += 8.0 * (c.scored.up ? 1 : 0);
  s += 5.0 * c.scored.feetOnTread;
  s += 2.0 * c.feetOnTreadMax;
  s += 5.0 * clip(c.scored.above / 0.095);
  s += 3.0 * clip(1 - c.bothNear / 0.32);
  s += 2.0 * clip((c.maxZ - c.z0Settle) / 0.12);
  s += 2.0 * clip((c.maxGainBoth || 0) / 0.10);
  s -= 2.0 * clip(c.maxAbsDY / 0.17);
  if (c.crit.honest) s += 12;
  if (c.crit.honest60) s += 4;
  return s;
}
const CELL_TRI = [
  { dh: 0.000, drop: 0.120, fmul: 1.0 },
  { dh: -0.010, drop: 0.130, fmul: 0.7 },
  { dh: 0.010, drop: 0.125, fmul: 1.3 },
];

const r = mulberry32(SEED);
let nEp = 0;
const tri = async g => {
  saveIntent(intentOf(g), tmp);
  const out = [];
  for (const c of CELL_TRI) { out.push(await scoreCell(tmp, { rise: RISE, dh: c.dh, drop: c.drop, fmul: c.fmul })); nEp++; }
  return out;
};
const grid = async g => { saveIntent(intentOf(g), tmp); nEp += 9; return scoreRobust(tmp, { rise: RISE }); };
const mean3 = cs => cs.reduce((a, c) => a + shaped2(c), 0) / cs.length;

const seedG = clampGene(JSON.parse(fs.readFileSync(SRC, 'utf8')).gene);
const pop = [{ g: seedG, sig: 0.10, fail: 0 }];
for (let i = 0; i < 7; i++) pop.push({ g: jitter(seedG, 0.10 + 0.05 * i, r), sig: 0.14, fail: 0 });
for (const e of pop) { e.cells = await tri(e.g); e.s = mean3(e.cells); }
pop.sort((a, b) => b.s - a.s);
console.log(`=== FAMILY C refinement @${RISE_MM} mm, seed ${SEED}, budget ${BUDGET_S} s ===`);
console.log(`start shaped2 ${pop.map(e => e.s.toFixed(2)).join(' ')}`);

let best = { s: pop[0].s, g: pop[0].g, G: await grid(pop[0].g) };
console.log(`[${el().toFixed(0)}s] incumbent k=${best.G.k}/9 obj=${best.G.objective.toFixed(3)} up=${best.G.agg.upFinal}/9 fotFinal=${best.G.agg.meanFeetOnTreadFinal.toFixed(2)} above=${best.G.agg.meanAbove_mm.toFixed(1)}mm`);

let it = 0;
while (el() < BUDGET_S) {
  it++;
  const i = Math.floor(r() * pop.length), e = pop[i];
  const g = (r() < 0.20 && pop.length > 1)
    ? jitter(cross(e.g, pop[Math.floor(r() * pop.length)].g, r), e.sig * 0.5, r)
    : jitter(e.g, e.sig, r);
  const cells = await tri(g);
  const s = mean3(cells);
  if (s > e.s) {
    e.g = g; e.s = s; e.cells = cells; e.fail = 0; e.sig = Math.min(0.24, e.sig * 1.2);
    if (s > best.s) {
      const G = await grid(g);
      const adopt = G.k > best.G.k || (G.k === best.G.k && G.objective > best.G.objective);
      if (adopt) best = { s, g, G };
      console.log(`[${el().toFixed(0)}s] it=${it} shaped2=${s.toFixed(2)} k=${G.k}/9 obj=${G.objective.toFixed(3)} ${adopt ? 'ADOPT' : 'keep '} up=${G.agg.upFinal}/9 fotFinal=${G.agg.meanFeetOnTreadFinal.toFixed(2)} fotMax=${G.agg.feetOnTreadMax} above=${G.agg.meanAbove_mm.toFixed(1)}mm peakZ=${mm(G.agg.maxZ)} bothNear=${G.agg.bothNear_mm.toFixed(0)}mm seq=${intentOf(g).sequence}`);
      if (adopt) best.s = Math.max(s, best.s);
    }
  } else {
    e.fail++;
    if (e.fail >= 6) { e.sig = Math.max(0.025, e.sig * 0.6); e.fail = 0; }
  }
}
console.log(`refinement done: ${it} iterations, ${nEp} episodes`);

const G = best.G;
const note = `FAMILY C ROUND 3 REFINEMENT, rise ${RISE_MM} mm on the REPAIRED canonical 4-step flight, seeded from ${SRC} and re-ranked on the two axes that pass left open (upright at the scored instant, second foot on the tread). `
  + `9-cell robust grid: cleared ${G.k} of 9 under climb/rig3.mjs 'honest', tail=policy. objective ${G.objective.toFixed(3)} = meanReward ${G.meanReward.toFixed(3)} + ${CLEAR_BONUS}*${G.k}. `
  + `upright at the scored instant ${G.agg.upFinal} of 9 cells; feet resting on the tread ${G.agg.meanFeetOnTreadFinal.toFixed(2)} of 2 at that instant (max over episode ${G.agg.feetOnTreadMax}); trunk peak z ${mm(G.agg.maxZ)} mm, mean trunk height above the tread at the scored instant ${G.agg.meanAbove_mm.toFixed(1)} mm; `
  + `head contact ${G.agg.headFrac.toFixed(3)}, foot-on-riser ${G.agg.riserFrac.toFixed(3)}, wall ${G.agg.wallFrac.toFixed(3)}, head+foot both bearing ${G.agg.bothFrac.toFixed(3)}; worse foot's closest approach to the landing spot ${G.agg.bothNear_mm.toFixed(0)} mm. `
  + `contact sequence ${intentOf(best.g).sequence}, opening order ${ORDERS[best.g.order].join('')}, lead foot ${best.g.lead ? 'R' : 'L'}.`;
const path = `../climb/best_r3_cornerclimb2_${RISE_MM}mm.json`;
saveIntent(intentOf(best.g, note), path);
const re = await scoreRobust(path, { rise: RISE });
console.log(`SAVED ${path} re-scored from the published file: k=${re.k}/9 obj=${re.objective.toFixed(3)} up=${re.agg.upFinal}/9 fotFinal=${re.agg.meanFeetOnTreadFinal.toFixed(2)}`);
for (const v of re.verdicts) console.log(`   rise=${v.rise_mm} drop=${v.drop} f=${v.fmul} honest=${v.honest} rew=${v.reward} x=${v.x_mm} above=${v.above_mm} up=${v.up} fot=${v.feetOnTread} fotMax=${v.feetOnTreadMax} peakZ=${v.peakZ_mm} head=${v.headFrac} riser=${v.riserFrac} wall=${v.wallFrac}`);

// how far does THIS move carry, rise by rise
const lad = [];
for (const h of [40, 60, 80, 100, 110, 120, 130, 140, 160, 180]) {
  let sim = false, bestAbove = -1e9, fot = 0, up = false, hon = false;
  for (const p of [{ drop: 0.120, fmul: 1.0 }, { drop: 0.130, fmul: 0.7 }, { drop: 0.125, fmul: 1.3 }]) {
    const c = await scoreCell(path, { rise: h / 1000, drop: p.drop, fmul: p.fmul });
    if (c.scored.feetOnTread >= 1 && c.scored.above > 0) sim = true;
    if (c.scored.above > bestAbove) { bestAbove = c.scored.above; fot = c.scored.feetOnTread; up = c.scored.up; hon = c.crit.honest; }
  }
  lad.push({ rise_mm: h, simultaneous: sim, bestAbove_mm: +mm(bestAbove), fot, up, honest: hon });
}
console.log('ladder: ' + lad.map(x => `${x.rise_mm}:${x.simultaneous ? 'SIM' : '-'}(above ${x.bestAbove_mm} fot ${x.fot} up ${x.up})`).join(' '));

fs.writeFileSync(OUT, JSON.stringify({ family: 'C_whole_body_corner_climb_r3_refine', rise_mm: RISE_MM,
  seed: SEED, episodes: nEp, iterations: it, published: path, seededFrom: SRC,
  k: re.k, objective: re.objective, meanReward: re.meanReward, agg: re.agg, verdicts: re.verdicts,
  sequence: intentOf(best.g).sequence, order: ORDERS[best.g.order].join(''), lead: best.g.lead ? 'R' : 'L',
  ladder: lad }, null, 2));
console.log(`wrote ${OUT}  total ${el().toFixed(0)} s`);
