// chase_robust.mjs — THE BALL CHALLENGE'S SCORER. One copy, one grid.
//
// A cell is a coincidence. `chase_rig.mjs` will score one, and one cell tells
// you that an entrant reached A ball, once, from one bearing, on one plant —
// which is exactly the mistake the stairs rail made in round 2 and spent round
// 3 undoing (climb/robust.mjs's header: "a move that passes once is a
// coincidence"). So a REPORTED ball result is always this: fourteen cells, nine
// core and five extended, and "chased k of 9".
//
//   bearing in {−20, 0, +20}°  ×  range in {0.45, 0.70, 0.95} m   — the core 9
//   plus the centre cell on the slippery plant (drop 0.130, friction ×0.7),
//        the centre cell on the grippy plant  (drop 0.125, friction ×1.3),
//        the ball at ±40° and 0.70 m, and the ball at 1.20 m dead ahead.
//
// THE EPISODE IS NOT IN THIS FILE. It is `sim/chase_score.mjs`, which
// `chase_rig.mjs` builds the machine for and `duckbench-core.mjs`'s POST /chase
// answers out of. What is this file's is the GRID, the aggregation and the
// verdict rows. `chase/chase_parity.mjs` proves the bench's cells are these
// cells at full float digits.
//
//   cd ~/projects/duck-sounds/sim && node ../chase/chase_robust.mjs
//   cd ~/projects/duck-sounds/sim && node ../chase/chase_robust.mjs --controls
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chaseCell, readEntrant, entrantHash, secondsOf, PLANT, PLANT_DIGEST, RIG }
  from './chase_rig.mjs';
import { gridCells, verdict, CRITERION_SENTENCE, UPRIGHT_TAIL_MIN, TOUCH_MM, TRAVEL_MIN_MM,
         N_CORE, N_EXT } from '../sim/chase_score.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));

export { gridCells, CRITERION_SENTENCE, UPRIGHT_TAIL_MIN, TOUCH_MM, TRAVEL_MIN_MM };

/** The four bundled entrants, in the order the package's leaderboard lists them. */
// ABSOLUTE, because an entrant path resolves against the working directory
// (chase_rig.mjs readEntrant): a bare name here would be looked for wherever
// the caller stands, and these four are the bundled controls beside this file.
export const CONTROLS = [
  'ctrl_do_nothing.json',
  'ctrl_ball_kick_left.json',
  'ctrl_ball_kick_right.json',
  'ctrl_alpha_walking.json',
].map(name => path.join(HERE, name));

/**
 * THE SCORER. Scores a SAVED entrant file over the fourteen-cell grid.
 *
 *   kChased / nCore   cells `chased` among the CORE nine — what a leaderboard
 *                     sorts on
 *   kStable / nCore   the same nine, minus any where the duck did not stand
 *                     through the fifty-tick tail
 *   kExt   / nExtOnly cells `chased` among the FIVE extended ones
 *
 * There is no shaped objective here and that is deliberate. The stairs rail has
 * one because its search needed a gradient; this challenge is scored by a person
 * reading three counts and a distance, and a weighted sum of nine reward terms
 * that has never been calibrated on this plant would be a number nobody could
 * defend sorting on.
 */
export async function scoreChase(file, { core = false, onCell } = {}) {
  const entrant = readEntrant(file);
  const hash = entrantHash(entrant);
  const seconds = secondsOf(entrant);
  const plan = gridCells({ core });
  const cells = [];
  for (const cell of plan) {
    const r = await chaseCell(entrant, cell, { seconds });
    cells.push(r);
    if (onCell) onCell(r);
  }
  const coreCells = cells.filter(c => c.cell.tier === 'core');
  const extCells = cells.filter(c => c.cell.tier === 'ext');
  const mean = (arr, f) => arr.reduce((a, c) => a + f(c), 0) / Math.max(arr.length, 1);
  // THE CENTRE CELL — bearing 0, range 0.70, nominal plant — is the one number a
  // leaderboard row carries beside its counts, because it is the cell every
  // entrant runs and the one a reader can picture.
  const centre = cells.find(c => c.cell.tier === 'core' && c.cell.bearing === 0 && c.cell.range === 0.70);
  return {
    source: path.basename(file),
    name: entrant.name ?? null,
    kind: entrant.kind,
    policy: entrant.kind === 'policy' ? entrant.policy : null,
    seconds,
    sha256: hash, entrant: hash.slice(0, 12),
    kChased: coreCells.filter(c => c.chased).length,
    kStable: coreCells.filter(c => c.stable).length,
    nCore: coreCells.length,
    kExt: extCells.filter(c => c.chased).length,
    kExtStable: extCells.filter(c => c.stable).length,
    nExtOnly: extCells.length,
    kChasedAll: cells.filter(c => c.chased).length,
    kStableAll: cells.filter(c => c.stable).length,
    nAll: cells.length,
    centreBallTravel_mm: centre ? centre.facts.ballTravel_mm : null,
    touchedCells: cells.filter(c => c.facts.touched).length,
    uprightFinalCells: cells.filter(c => c.facts.upright).length,
    meanClosest_mm: mean(cells, c => c.facts.closest_mm),
    minClosest_mm: Math.min(...cells.map(c => c.facts.closest_mm)),
    maxBallTravel_mm: Math.max(...cells.map(c => c.facts.ballTravel_mm)),
    meanBallTravel_mm: mean(cells, c => c.facts.ballTravel_mm),
    maxBallPeakSpeed_mps: Math.max(...cells.map(c => c.facts.ballPeakSpeed_mps)),
    meanUprightTailTicks: mean(cells, c => c.uprightTailTicks),
    minUprightTailTicks: Math.min(...cells.map(c => c.uprightTailTicks)),
    // The nine terms, averaged over the fourteen cells, so a package can print
    // Pollen's reward beside the verdict without a reader recomputing it.
    terms: cells[0].terms.map((t, i) => ({
      term: t.term, weight: t.weight,
      value: mean(cells, c => c.terms[i].value),
      ...(t.action_rate_l2_source ? { action_rate_l2_source: t.action_rate_l2_source } : {}),
    })),
    refused: cells[0].refused,
    criterion: CRITERION_SENTENCE,
    plantName: PLANT, plantDigest: PLANT_DIGEST,
    cells,
    verdicts: cells.map(c => ({
      bearing: c.cell.bearing, range: c.cell.range, drop: c.cell.drop, fmul: c.cell.fmul,
      tier: c.cell.tier, sha256: hash, entrant: hash.slice(0, 12),
      chased: c.chased, stable: c.stable,
      touched: c.facts.touched,
      ballTravel_mm: +c.facts.ballTravel_mm.toFixed(2),
      ballNet_mm: +c.facts.ballNet_mm.toFixed(2),
      closest_mm: +c.facts.closest_mm.toFixed(2),
      final_mm: +c.facts.final_mm.toFixed(2),
      ballPeakSpeed_mps: +c.facts.ballPeakSpeed_mps.toFixed(4),
      upright: c.facts.upright,
      uprightTailTicks: c.uprightTailTicks, tailTicks: c.tailTicks,
    })),
  };
}

/** One row of the bundled leaderboard, in the shape the kit pins. */
export const leaderboardRow = R => ({
  name: R.name, source: R.source, kind: R.kind, policy: R.policy, seconds: R.seconds,
  sha256: R.sha256, entrant: R.entrant,
  kChased: R.kChased, kStable: R.kStable, nCore: R.nCore,
  kExt: R.kExt, nExt: R.nExtOnly,
  centreBallTravel_mm: R.centreBallTravel_mm,
  touchedCells: R.touchedCells, minUprightTailTicks: R.minUprightTailTicks,
});

// ============================================================== the controls
const isMain = process.argv[1] && process.argv[1].endsWith('chase_robust.mjs');
if (isMain) {
  const args = process.argv.slice(2);
  const files = args.filter(a => !a.startsWith('--'));
  const write = args.includes('--write');
  const targets = files.length ? files : CONTROLS;
  console.log('=== chase_robust — the ball challenge, fourteen cells per entrant ===');
  console.log(`   plant ${PLANT} ${PLANT_DIGEST.slice(0, 12)}   ball geom ${RIG.BALLG}, `
            + `duck geoms ${RIG.DUCKG.length}, substeps ${RIG.SUBSTEPS}`);
  console.log(`   criterion: ${CRITERION_SENTENCE}`);
  const rows = [], all = [];
  const t0 = Date.now();
  for (const file of targets) {
    const started = Date.now();
    const R = await scoreChase(file);
    const dt = (Date.now() - started) / 1000;
    rows.push(leaderboardRow(R));
    all.push(R);
    console.log('');
    console.log(`--- ${R.name}  (${R.kind}${R.policy ? ' ' + R.policy : ''}, ${R.seconds}s, `
      + `${R.entrant})  ${dt.toFixed(1)} s, ${(dt / R.nAll).toFixed(2)} s/cell`);
    console.log(`    chased ${R.kChased}/${R.nCore} core   stable ${R.kStable}/${R.nCore}   `
      + `ext ${R.kExt}/${R.nExtOnly}   centre-cell ballTravel ${R.centreBallTravel_mm.toFixed(1)} mm   `
      + `touched in ${R.touchedCells}/${R.nAll} cells`);
    for (const v of R.verdicts) {
      console.log(`      [${v.tier}] bearing ${String(v.bearing).padStart(3)}  range ${v.range.toFixed(2)}  `
        + `drop ${v.drop} f${v.fmul}   chased=${String(v.chased).padEnd(5)} stable=${String(v.stable).padEnd(5)} `
        + `touch=${String(v.touched).padEnd(5)} travel=${String(v.ballTravel_mm).padStart(8)}mm `
        + `net=${String(v.ballNet_mm).padStart(7)}mm closest=${String(v.closest_mm).padStart(8)}mm `
        + `final=${String(v.final_mm).padStart(7)}mm peak=${v.ballPeakSpeed_mps} up=${v.upright} `
        + `upTail=${v.uprightTailTicks}/${v.tailTicks}`);
    }
    console.log('    terms (per-tick means over the driven span, averaged over the 14 cells):');
    for (const t of R.terms) {
      console.log(`      ${t.term.padEnd(24)} weight ${String(t.weight).padStart(6)}   `
        + `value ${t.value.toPrecision(8)}${t.action_rate_l2_source ? '   [' + t.action_rate_l2_source + ']' : ''}`);
    }
    console.log('    refused: ' + R.refused.map(r => r.term).join(', '));
  }
  console.log('');
  console.log('=== THE BUNDLED LEADERBOARD ===');
  console.log('   entrant                     kind    chased/9  stable/9  ext/5  centre ballTravel');
  for (const r of rows) {
    console.log(`   ${String(r.name).padEnd(26)} ${r.kind.padEnd(7)} ${String(r.kChased).padStart(6)}/9 `
      + `${String(r.kStable).padStart(8)}/9 ${String(r.kExt).padStart(5)}/5 `
      + `${r.centreBallTravel_mm.toFixed(1).padStart(14)} mm`);
  }
  console.log(`   (${((Date.now() - t0) / 1000).toFixed(0)} s for ${targets.length} entrants × 14 cells)`);
  if (write) {
    const out = path.join(HERE, 'chase_controls-results.json');
    fs.writeFileSync(out, JSON.stringify({
      why: 'The four bundled controls, scored over the fourteen-cell ball grid by '
         + 'chase/chase_robust.mjs. These rows are the kit\'s sha-pinned bundled leaderboard.',
      criterion: CRITERION_SENTENCE,
      plantName: PLANT, plantDigest: PLANT_DIGEST,
      nCore: N_CORE, nExt: N_EXT,
      grid: gridCells(),
      leaderboard: rows,
      entrants: all.map(R => ({ ...R, cells: undefined })),
    }, null, 2) + '\n');
    console.log(`   wrote ${out}`);
  }
}
