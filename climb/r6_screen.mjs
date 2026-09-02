// r6_screen.mjs — ROUND 6, PHASE 1: THE CEILING OF THE WHOLE CORPUS.
//
// rig3 criteria(): `honest` requires scored.above > 0.095 m, and
// above = trunk z - tread height. A core cell in which the trunk NEVER gets
// 95 mm above the tread at ANY tick cannot be cleared by ANY landing law —
// timed, event-triggered, or servoed — because the landing only decides what
// the feet do once the trunk is already up there. So
//
//   ceilingCore = # of the 9 core cells with max(trunk z) - riseOfThatCell > 95 mm
//
// is a HARD UPPER BOUND on that move's kCore. This scores every published
// launch in the corpus at 60 mm through the ONE shared scorer
// (climb/robust.mjs scoreRobust, {core:true} = the round-3 9 cells) and prints
// the table sorted by ceilingCore. It adds no field to the scorer: peakZ_mm and
// rise_mm are already on every verdict row.
//
// Run from sim/:  node ../climb/r6_screen.mjs
import fs from 'node:fs';
import { scoreRobust } from './robust.mjs';

const P = '../climb/';
const RISE = 0.060;
const T0 = Date.now();
const el = () => (Date.now() - T0) / 1000;

const files = fs.readdirSync(P).filter(f =>
  /^best_r[2345]_.*\.json$/.test(f) || /^best_[012]_.*\.json$/.test(f)).sort();

console.log(`SCREEN: ceilingCore at 60 mm for ${files.length} published launches`);
console.log(`  rule: core cell counts if max(trunk z over the whole episode) - cellRise > 95 mm`);
console.log(`  scorer: climb/robust.mjs scoreRobust({rise:0.060, core:true}) — the round-3 9 cells\n`);

const rows = [];
for (const f of files) {
  const r = await scoreRobust(P + f, { rise: RISE, core: true });
  if (r.invalid) {
    rows.push({ file: f, sha256: r.sha256, move: r.move, invalid: true,
                boundViolations: r.boundViolations, ceilingCore: null });
    console.log(`  [${el().toFixed(0).padStart(4)}s] ${f.padEnd(30)} INVALID (out of declared bounds)`);
    continue;
  }
  const core = r.verdicts.filter(v => v.tier === 'core');
  const peaks = core.map(v => +(v.peakZ_mm - v.rise_mm).toFixed(1));
  const ceilingCore = peaks.filter(p => p > 95).length;
  const row = { file: f, sha256: r.sha256, move: r.move, invalid: false,
    ceilingCore, kCore: r.kCore, kCoreStable: r.kCoreStable,
    peakAboveTread_mm: peaks,
    cells: core.map((v, i) => ({ rise_mm: v.rise_mm, drop: v.drop, fmul: v.fmul,
      peakAbove_mm: peaks[i], over95: peaks[i] > 95, honest: v.honest,
      stable: v.stableClear, above_mm: v.above_mm })),
    maxTq: r.agg.maxTq, minPenetrationEpisode_mm: +r.agg.minPenetrationEpisode_mm.toFixed(2),
    meanPeakAbove_mm: +(peaks.reduce((a, b) => a + b, 0) / peaks.length).toFixed(1) };
  rows.push(row);
  console.log(`  [${el().toFixed(0).padStart(4)}s] ${f.padEnd(30)} ceil=${ceilingCore}/9 kCore=${r.kCore}/9 kStable=${r.kCoreStable}/9  ${row.move}`);
}

rows.sort((a, b) => (b.ceilingCore ?? -1) - (a.ceilingCore ?? -1) || (b.kCore ?? -1) - (a.kCore ?? -1));
console.log('\n=== THE SCREEN, sorted by ceilingCore (all at 60 mm) ===');
console.log('  ceil kCore kStab  move          maxTq   minPen(mm)  peak trunk above tread, core 9 (mm)                  file');
for (const r of rows) {
  if (r.invalid) { console.log(`  INVALID                ${r.move}                                                                     ${r.file}`); continue; }
  console.log(`  ${String(r.ceilingCore).padStart(2)}/9  ${String(r.kCore).padStart(2)}/9  ${String(r.kCoreStable).padStart(2)}/9  ${r.move}  ${r.maxTq.toFixed(4)}  ${String(r.minPenetrationEpisode_mm).padStart(9)}  [${r.peakAboveTread_mm.map(x => String(x).padStart(6)).join(' ')}]  ${r.file}`);
}
const valid = rows.filter(r => !r.invalid);
const maxCeil = Math.max(...valid.map(r => r.ceilingCore));
console.log(`\n  HIGHEST ceilingCore over the corpus at 60 mm: ${maxCeil} of 9.`);
console.log(`  Distinct vectors: ${new Set(valid.map(r => r.sha256)).size} of ${valid.length} files.`);

fs.writeFileSync(`${P}r6_screen-results.json`, JSON.stringify({
  generated: new Date().toISOString(),
  question: 'ceilingCore at 60 mm for every published launch in the corpus',
  rule: 'a core cell counts when max(trunk z) over the whole episode minus that cell\'s tread height exceeds 95 mm',
  why: 'rig3 criteria() honest needs above > 0.095 at the scored instant, so ceilingCore is a hard upper bound on kCore under ANY landing law',
  scorer: 'climb/robust.mjs scoreRobust({rise:0.060, core:true})',
  rise_mm: 60, files: files.length, maxCeilingCore: maxCeil,
  distinctVectors: new Set(valid.map(r => r.sha256)).size,
  wall_s: +el().toFixed(1), rows,
}, null, 1) + '\n');
console.log(`  wrote climb/r6_screen-results.json  (${el().toFixed(0)}s)`);
