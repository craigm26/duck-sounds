// THE ACCEPTANCE TEST FOR /climb.
//
// WHAT IS BEING CLAIMED. A phone can score a stairs move. The claim is not
// "the phone runs some physics"; it is that a cell scored through POST /climb
// is THE SAME CELL climb/robust.mjs scores — the scorer every verdict in
// climb/r6_judge-results.json was decided by — so that "cleared 5 of 9 stably
// at 60 mm" means on a phone what it means in the audit. A bench that answered
// a different number under the same name would be worse than a bench that
// answered nothing, because the number would be believed.
//
// SO IT IS MEASURED, NOT ASSERTED. Five published files, every one of the
// fourteen grid cells, scored twice: once through the bench's own `handle`
// (in-process, no socket, which is exactly what the WebView bridge does) and
// once through robust.scoreRobust. Seven fields per cell are compared with
// Object.is at FULL FLOAT DIGITS — honest, stable, above, x, feetOnTread,
// peakAboveTread, maxTq — plus the move hash, the lateral excursion, both
// penetration figures and the upright tail count, because a gate that checks
// only the verdict cannot tell a matching verdict from a matching trajectory.
//
// THEN THE AGGREGATES ARE HELD TO THE AUDIT. kCore, kCoreStable, kExt,
// kExtStable and ceilingCore are recomputed from the /climb answers alone and
// required to equal climb/r6_judge-results.json phaseG. That is the half a
// per-cell comparison cannot do: two scorers can agree cell by cell and still
// be counted differently.
//
// This passing is what licenses the sentence "this is the audit's criterion and
// grid, scored on this bench's plant".
//
//   cd ~/projects/duck-sounds/sim && node climb_parity.mjs
import fs from 'node:fs';
import { nodeBench } from './duckbench-node.mjs';
import { scoreRobust } from '../climb/robust.mjs';
import { gridCells, CEILING_ABOVE, UPRIGHT_TAIL_MIN } from './climb_score.mjs';

const P = '../climb/';
const RISE = 0.060;

/**
 * THE FIVE FILES, AND THE NUMBERS THEY ARE HELD TO.
 *
 * The first three are the record, the standing round-3 vault and the round-4
 * event variant, straight out of phaseG. The last two are the CONTROLS, and
 * they are here because a grid that only agrees about successes is not a grid:
 * do-nothing must be 0 everywhere and the duck spawned already standing on the
 * tread must be 14 of 14, or the criterion is not measuring what it says.
 */
const CASES = [
  { file: 'best_r6_ceilvaultC_60mm.json', move: 'a56d459fb649',
    label: 'the record', kCore: 5, kCoreStable: 5, kExt: 5, kExtStable: 5, ceilingCore: 5, maxTq: 0.6405 },
  { file: 'best_r3_vault_60mm.json', move: '4b9110c448ec',
    label: 'the round-3 beak-strut vault', kCore: 4, kCoreStable: 4, kExt: 4, kExtStable: 4, ceilingCore: 5, maxTq: 0.6405 },
  { file: 'best_r4_famA_60mm.json', move: '7b790070b010',
    label: 'the round-4 event variant', kCore: 4, kCoreStable: 4, kExt: 4, kExtStable: 4, ceilingCore: 5, maxTq: 0.6405 },
  { file: 'ctrl_do_nothing.json',
    label: 'CONTROL — HOME held, vx 0', kCore: 0, kCoreStable: 0, kExt: 0, kExtStable: 0, ceilingCore: 0 },
  { file: 'r4_ctrl_on_tread_60mm.json',
    label: 'CONTROL — spawned standing on the tread', kCore: 9, kCoreStable: 9, kExt: 14, kExtStable: 14, ceilingCore: 9 },
];

/** The seven the contract names, plus the four that make a match a trajectory. */
const FIELDS = [
  ['honest',                   a => a.honest,                    c => c.crit.honest],
  ['stable',                   a => a.stable,                    c => c.crit.honest && c.uprightTailTicks >= UPRIGHT_TAIL_MIN],
  ['above_mm',                 a => a.above_mm,                  c => c.scored.above * 1000],
  ['x_mm',                     a => a.x_mm,                      c => c.scored.x * 1000],
  ['feetOnTread',              a => a.feetOnTread,               c => c.scored.feetOnTread],
  ['peakAboveTread_mm',        a => a.peakAboveTread_mm,         c => (c.maxZ - c.rise) * 1000],
  ['maxTq',                    a => a.maxTq,                     c => c.maxTq],
  ['dy_mm',                    a => a.dy_mm,                     c => c.scored.dy * 1000],
  ['maxAbsDY_mm',              a => a.maxAbsDY_mm,               c => c.maxAbsDY * 1000],
  ['uprightTailTicks',         a => a.uprightTailTicks,          c => c.uprightTailTicks],
  ['penetrationAtScore_mm',    a => a.penetrationAtScore_mm,     c => c.penetrationAtScore === null ? null : c.penetrationAtScore * 1000],
  ['minPenetrationEpisode_mm', a => a.minPenetrationEpisode_mm,  c => c.minPenetrationEpisode === null ? null : c.minPenetrationEpisode * 1000],
  ['feetOnTreadMax',           a => a.feetOnTreadMax,            c => c.feetOnTreadMax],
  ['reachedFlight',            a => a.reachedFlight,             c => (c.maxX > 0.12) || (c.feetOnTreadMax > 0)],
];

const CELLS = gridCells();
const bench = await nodeBench();
const t0 = Date.now();
const el = () => `${((Date.now() - t0) / 1000).toFixed(0)}s`;

// The grid the bench publishes must BE the grid the scorer runs, or a client
// that trusted GET /climb/grid is scoring a different fourteen cells.
const published = await bench.handle(new URL('http://bench.local/climb/grid'), {});
const gridSame = JSON.stringify(published.cells) === JSON.stringify(CELLS);
console.log(`GET /climb/grid lists ${published.cells.length} cells; identical to robust.mjs's plan: ${gridSame}`);
console.log(`   bar ${published.bar} of 9 stable; upright-tail minimum ${published.uprightTailMin}; plant ${published.plantDigest.slice(0, 12)}`);
if (!gridSame) console.log(`   !! published ${JSON.stringify(published.cells)}\n   !! expected  ${JSON.stringify(CELLS)}`);

let rows = 0, exact = 0;
const failures = [];
const aggRows = [];

for (const K of CASES) {
  const intent = JSON.parse(fs.readFileSync(P + K.file, 'utf8'));
  const answers = [];
  for (const cell of CELLS) {
    const a = await bench.handle(new URL('http://bench.local/climb'),
      { intent, rise: RISE, cell: { dh: cell.dh, drop: cell.drop, fmul: cell.fmul }, tail: 'policy' });
    if (a.error) throw new Error(`${K.file} ${JSON.stringify(cell)}: ${a.error}`);
    if (a.invalid) throw new Error(`${K.file}: ${a.why}`);
    answers.push(a);
  }
  const R = await scoreRobust(P + K.file, { rise: RISE });
  if (R.invalid) throw new Error(`${K.file} is out of its declared bounds and cannot be a parity case`);

  const hashSame = answers.every(a => a.hash === R.sha256);
  for (let i = 0; i < CELLS.length; i++) {
    const a = answers[i], c = R.cells[i];
    rows++;
    const bad = [];
    // The cell each side thinks it ran, before anything measured in it.
    if (Math.round(c.cell.rise_mm) !== Math.round((RISE + CELLS[i].dh) * 1000)
        || c.cell.drop !== CELLS[i].drop || c.cell.fmul !== CELLS[i].fmul) {
      bad.push({ field: 'cell', bench: CELLS[i], robust: c.cell });
    }
    for (const [name, fromBench, fromRobust] of FIELDS) {
      const x = fromBench(a), y = fromRobust(c);
      if (!Object.is(x, y)) bad.push({ field: name, bench: x, robust: y });
    }
    if (!bad.length) exact++;
    else failures.push({ file: K.file, cell: CELLS[i], bad: bad.slice(0, 6) });
  }

  // THE AGGREGATES, RECOMPUTED FROM THE /climb ANSWERS ALONE.
  const core = answers.filter((_, i) => CELLS[i].tier === 'core');
  const agg = {
    kCore: core.filter(a => a.honest).length,
    kCoreStable: core.filter(a => a.stable).length,
    kExt: answers.filter(a => a.honest).length,
    kExtStable: answers.filter(a => a.stable).length,
    ceilingCore: core.filter(a => a.peakAboveTread_mm / 1000 > CEILING_ABOVE).length,
    maxTq: Math.max(...answers.map(a => a.maxTq)),
  };
  const vsRobust = agg.kCore === R.kCore && agg.kCoreStable === R.kCoreStable
                && agg.kExt === R.kExt && agg.kExtStable === R.kExtStable;
  const vsAudit = agg.kCore === K.kCore && agg.kCoreStable === K.kCoreStable
               && agg.kExt === K.kExt && agg.kExtStable === K.kExtStable
               && agg.ceilingCore === K.ceilingCore
               && (K.maxTq === undefined || Math.abs(agg.maxTq - K.maxTq) < 5e-5)
               && (K.move === undefined || answers[0].move === K.move);
  aggRows.push({ ...K, agg, robust: { kCore: R.kCore, kCoreStable: R.kCoreStable, kExt: R.kExt, kExtStable: R.kExtStable },
                 hashSame, vsRobust, vsAudit, move: answers[0].move });
  console.log(`   [${el().padStart(5)}] ${K.file.padEnd(32)} /climb ${agg.kCore}/${agg.kCoreStable}/${agg.kExt}/${agg.kExtStable} ceil ${agg.ceilingCore}/9`
    + `   robust ${R.kCore}/${R.kCoreStable}/${R.kExt}/${R.kExtStable}   audit ${K.kCore}/${K.kCoreStable}/${K.kExt}/${K.kExtStable} ceil ${K.ceilingCore}/9`
    + `   move ${answers[0].move}${K.move ? (answers[0].move === K.move ? '' : ' !! NOT ' + K.move) : ''}`
    + `   ${vsRobust && vsAudit && hashSame ? 'AGREES' : '!! DISAGREES'}`);
}

console.log('');
console.log(`per-cell rows: ${rows}   EXACT on all ${FIELDS.length} compared fields + the cell itself: ${exact}/${rows}`);
for (const f of failures.slice(0, 8)) console.log(`   !! ${f.file} ${JSON.stringify(f.cell)}: ${JSON.stringify(f.bad)}`);
const allExact = exact === rows;
const allAgg = aggRows.every(r => r.vsRobust && r.vsAudit && r.hashSame);
console.log(`aggregates equal robust.mjs AND climb/r6_judge-results.json phaseG on every case: ${allAgg}`);
console.log(`the grid the bench publishes is the grid the scorer runs: ${gridSame}`);
console.log('');
console.log(allExact && allAgg && gridSame
  ? `CLIMB PARITY PASS — /climb is robust.mjs, cell for cell, at full float digits.  [${el()}]`
  : `CLIMB PARITY FAIL  [${el()}]`);
process.exit(allExact && allAgg && gridSame ? 0 : 1);
