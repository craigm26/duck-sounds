// THE ACCEPTANCE TEST FOR /chase.
//
// WHAT IS BEING CLAIMED. A phone can score a ball entrant. The claim is not
// "the phone runs some physics"; it is that a cell scored through POST /chase
// is THE SAME CELL `chase/chase_robust.mjs` scores — the scorer the bundled
// leaderboard and the published package are decided by — so that "chased 4 of
// 9" means on a phone what it means in the package. A bench that answered a
// different number under the same name would be worse than a bench that
// answered nothing, because the number would be believed.
//
// SO IT IS MEASURED, NOT ASSERTED. Every bundled entrant, every one of the
// fourteen grid cells, scored twice: once through the bench's own `handle`
// (in-process, no socket, which is exactly what the WebView bridge does) and
// once through `chase_robust.scoreChase`. EVERY numeric field is compared with
// `Object.is` at FULL FLOAT DIGITS — the eight facts, the verdict, the tick
// counts, all nine of Pollen's transcribed reward terms, the frozen heading,
// the ball's start and end, and the entrant hash — because a gate that checks
// only the verdict cannot tell a matching verdict from a matching trajectory.
//
// THEN THE FIRST CONTROL IS CHECKED AGAINST THE THING IT EXISTS TO PROVE.
// `ctrl_do_nothing` must score 0 of 14. A criterion that row passes is not a
// chasing test, and that check runs before any of the others are believed.
//
// AND THE AGGREGATES ARE RECOMPUTED FROM THE /chase ANSWERS ALONE and required
// to equal chase_robust's. That is the half a per-cell comparison cannot do:
// two scorers can agree cell by cell and still be counted differently.
//
//   cd ~/projects/duck-sounds/sim && node ../chase/chase_parity.mjs
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { nodeBench } from '../sim/duckbench-node.mjs';
import { scoreChase, CONTROLS } from './chase_robust.mjs';
import { readEntrant, secondsOf } from './chase_rig.mjs';
const HERE = path.dirname(fileURLToPath(import.meta.url));
import { gridCells, CRITERION_SENTENCE, N_CORE, N_EXT, CHASE_REFUSALS }
  from '../sim/chase_score.mjs';

/**
 * THE FOUR BUNDLED ENTRANTS, AND WHAT WAS PREDICTED OF THEM IN ADVANCE.
 *
 * `mustBeZero` is a HARD requirement and the reason the do-nothing row exists;
 * the other predictions are recorded and REPORTED, not enforced, because a
 * measurement that disagrees with a prediction is a finding to chase down and
 * not a test to fail. chase/REWARD.md §5 is where they were written down before
 * anything was run.
 */
const CASES = [
  { file: 'ctrl_do_nothing.json', label: 'CONTROL — HOME held, nothing commanded',
    mustBeZero: true, predicted: '0 of 14: a criterion this row passes is not a chasing test' },
  { file: 'ctrl_ball_kick_left.json', label: 'Pollen\'s kick policy, left foot, at the config\'s own command',
    predicted: '0 of 14, or very close: blind to the ball by design, trained at 90 mm' },
  { file: 'ctrl_ball_kick_right.json', label: 'Pollen\'s kick policy, right foot',
    predicted: '0 of 14, or very close: the same policy with KICK_FOOT flipped' },
  { file: 'ctrl_alpha_walking.json', label: 'THE NAIVE CHASER — straight ahead at 0.5 m/s',
    predicted: 'roughly 2-4 of the 9 core cells; every far off-bearing cell fails' },
];

/** Every field the two sides must agree about, at full float digits. */
const FIELDS = [
  ['ballTravel_mm',      a => a.ballTravel_mm,      c => c.facts.ballTravel_mm],
  ['ballNet_mm',         a => a.ballNet_mm,         c => c.facts.ballNet_mm],
  ['closest_mm',         a => a.closest_mm,         c => c.facts.closest_mm],
  ['final_mm',           a => a.final_mm,           c => c.facts.final_mm],
  ['touched',            a => a.touched,            c => c.facts.touched],
  ['ballPeakSpeed_mps',  a => a.ballPeakSpeed_mps,  c => c.facts.ballPeakSpeed_mps],
  ['upright',            a => a.upright,            c => c.facts.upright],
  ['uprightTailTicks',   a => a.uprightTailTicks,   c => c.uprightTailTicks],
  ['chased',             a => a.chased,             c => c.chased],
  ['stable',             a => a.stable,             c => c.stable],
  ['drivenTicks',        a => a.drivenTicks,        c => c.drivenTicks],
  ['rateTicks',          a => a.rateTicks,          c => c.rateTicks],
  ['tailTicks',          a => a.tailTicks,          c => c.tailTicks],
  ['yaw0',               a => a.yaw0,               c => c.yaw0],
  ['kickDir.x',          a => a.kickDir[0],         c => c.kickDir[0]],
  ['kickDir.y',          a => a.kickDir[1],         c => c.kickDir[1]],
  ['ball0.x',            a => a.ball0[0],           c => c.ball0[0]],
  ['ball0.y',            a => a.ball0[1],           c => c.ball0[1]],
  ['ballEnd.x',          a => a.ballEnd[0],         c => c.ballEnd[0]],
  ['ballEnd.y',          a => a.ballEnd[1],         c => c.ballEnd[1]],
  ['hash',               a => a.hash,               c => c.hash],
  ['actionRateSource',   a => a.actionRateSource,   c => c.actionRateSource],
];

/** The nine terms, by name, weight and value — the reward, not just the verdict. */
function termFields(nine) {
  const out = [];
  for (let i = 0; i < nine; i++) {
    out.push([`terms[${i}].term`, a => a.terms[i].term, c => c.terms[i].term]);
    out.push([`terms[${i}].weight`, a => a.terms[i].weight, c => c.terms[i].weight]);
    out.push([`terms[${i}].value`, a => a.terms[i].value, c => c.terms[i].value]);
  }
  return out;
}

const CELLS = gridCells();
const bench = await nodeBench();
const t0 = Date.now();
const el = () => `${((Date.now() - t0) / 1000).toFixed(0)}s`;

// The grid the bench publishes must BE the grid the scorer runs, or a client
// that trusted GET /chase/grid is scoring a different fourteen cells.
const published = await bench.handle(new URL('http://bench.local/chase/grid'), {});
const gridSame = JSON.stringify(published.cells) === JSON.stringify(CELLS);
const critSame = published.criterion === CRITERION_SENTENCE;
const refusedSame = JSON.stringify(published.refused) === JSON.stringify(CHASE_REFUSALS);
console.log(`GET /chase/grid lists ${published.cells.length} cells (${published.nCore} core, `
  + `${published.nExt} in all); identical to chase_robust's plan: ${gridSame}`);
console.log(`   criterion identical: ${critSame}   refusals identical (${published.refused.length}): ${refusedSame}`);
console.log(`   plant ${published.plantName} ${published.plantDigest.slice(0, 12)}   chaseable: ${published.chaseable}`);
if (!gridSame) console.log(`   !! published ${JSON.stringify(published.cells)}\n   !! expected  ${JSON.stringify(CELLS)}`);
if (published.nCore !== N_CORE || published.nExt !== N_EXT) {
  console.log(`   !! the bench says ${published.nCore}/${published.nExt} where the module says ${N_CORE}/${N_EXT}`);
}

let rows = 0, exact = 0;
const failures = [];
const aggRows = [];
let zeroGuard = null;

for (const K of CASES) {
  const entrant = readEntrant(path.join(HERE, K.file));   // bundled controls live beside this file
  const seconds = secondsOf(entrant);
  const answers = [];
  for (const cell of CELLS) {
    const a = await bench.handle(new URL('http://bench.local/chase'), {
      entrant, seconds,
      cell: { bearing: cell.bearing, range: cell.range, drop: cell.drop, fmul: cell.fmul },
      tail: 'policy',
    });
    if (a.error) throw new Error(`${K.file} ${JSON.stringify(cell)}: ${a.error}`);
    answers.push(a);
  }
  const R = await scoreChase(path.join(HERE, K.file));

  const compare = FIELDS.concat(termFields(R.cells[0].terms.length));
  for (let i = 0; i < CELLS.length; i++) {
    const a = answers[i], c = R.cells[i];
    rows++;
    const bad = [];
    // The cell each side thinks it ran, before anything measured in it.
    if (a.cell.bearing !== CELLS[i].bearing || a.cell.range !== CELLS[i].range
        || a.cell.drop !== CELLS[i].drop || a.cell.fmul !== CELLS[i].fmul
        || c.cell.bearing !== CELLS[i].bearing || c.cell.range !== CELLS[i].range
        || c.cell.drop !== CELLS[i].drop || c.cell.fmul !== CELLS[i].fmul) {
      bad.push({ field: 'cell', bench: a.cell, robust: c.cell, grid: CELLS[i] });
    }
    if (a.seconds !== c.seconds) bad.push({ field: 'seconds', bench: a.seconds, robust: c.seconds });
    if (JSON.stringify(a.refused) !== JSON.stringify(c.refused)) {
      bad.push({ field: 'refused', bench: a.refused.map(r => r.term), robust: c.refused.map(r => r.term) });
    }
    for (const [name, fromBench, fromRobust] of compare) {
      const x = fromBench(a), y = fromRobust(c);
      if (!Object.is(x, y)) bad.push({ field: name, bench: x, robust: y });
    }
    if (!bad.length) exact++;
    else failures.push({ file: K.file, cell: CELLS[i], bad: bad.slice(0, 6) });
  }

  // THE AGGREGATES, RECOMPUTED FROM THE /chase ANSWERS ALONE.
  const core = answers.filter((_, i) => CELLS[i].tier === 'core');
  const ext = answers.filter((_, i) => CELLS[i].tier === 'ext');
  const agg = {
    kChased: core.filter(a => a.chased).length,
    kStable: core.filter(a => a.stable).length,
    kExt: ext.filter(a => a.chased).length,
    touchedCells: answers.filter(a => a.touched).length,
  };
  const vsRobust = agg.kChased === R.kChased && agg.kStable === R.kStable
                && agg.kExt === R.kExt && agg.touchedCells === R.touchedCells;
  const hashSame = answers.every(a => a.hash === R.sha256);
  if (K.mustBeZero) {
    zeroGuard = agg.kChased === 0 && agg.kExt === 0 && agg.touchedCells === 0;
  }
  aggRows.push({ ...K, agg, vsRobust, hashSame,
                 robust: { kChased: R.kChased, kStable: R.kStable, kExt: R.kExt },
                 centre: R.centreBallTravel_mm, entrant: R.entrant });
  console.log(`   [${el().padStart(5)}] ${K.file.padEnd(28)} /chase ${agg.kChased}/${agg.kStable}/${agg.kExt}`
    + `   chase_robust ${R.kChased}/${R.kStable}/${R.kExt}`
    + `   touched ${agg.touchedCells}/14   entrant ${R.entrant}`
    + `   ${vsRobust && hashSame ? 'AGREES' : '!! DISAGREES'}`);
}

console.log('');
console.log(`per-cell rows: ${rows}   EXACT on all ${FIELDS.length + 27} compared fields + the cell itself: ${exact}/${rows}`);
for (const f of failures.slice(0, 8)) console.log(`   !! ${f.file} ${JSON.stringify(f.cell)}: ${JSON.stringify(f.bad)}`);
const allExact = exact === rows;
const allAgg = aggRows.every(r => r.vsRobust && r.hashSame);
console.log(`aggregates recomputed from /chase equal chase_robust on every entrant: ${allAgg}`);
console.log(`ctrl_do_nothing scores 0 of 14 and touches nothing: ${zeroGuard}`);
console.log(`the grid the bench publishes is the grid the scorer runs: ${gridSame}`);
console.log('');
console.log('=== the measured leaderboard, against what was predicted in advance ===');
for (const r of aggRows) {
  console.log(`   ${r.file.replace('.json', '').padEnd(24)} chased ${r.agg.kChased}/9  stable ${r.agg.kStable}/9  `
    + `ext ${r.agg.kExt}/5  centre travel ${r.centre.toFixed(1)} mm`);
  console.log(`      predicted: ${r.predicted}`);
}
console.log('');
const pass = allExact && allAgg && gridSame && critSame && refusedSame && zeroGuard === true;
console.log(pass
  ? `CHASE PARITY PASS — /chase is chase_robust.mjs, cell for cell, at full float digits.  [${el()}]`
  : `CHASE PARITY FAIL  [${el()}]`);
process.exit(pass ? 0 : 1);
