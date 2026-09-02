// r6_table.mjs — print the round-6 tables from the saved JSON. No physics runs
// here; it only formats climb/r6_screen-results.json and
// climb/r6_ceiling-results.json. (The search process wrote both and then threw
// on its own final console line — a stale `gen` in a template literal, after
// every file was already flushed. The bug is fixed in r6_ceiling_search.mjs;
// this file exists so the table does not need 35 minutes of compute to reprint.)
import fs from 'node:fs';
const P = '../climb/';
const S = JSON.parse(fs.readFileSync(P + 'r6_screen-results.json', 'utf8'));
const R = JSON.parse(fs.readFileSync(P + 'r6_ceiling-results.json', 'utf8'));
const pad = (v, n) => String(v).padStart(n);

console.log('=== SCREEN: ceilingCore at 60 mm, every published launch in the corpus ===');
console.log('  rule: a core cell counts when max(trunk z) - that cell\'s tread height > 95 mm');
console.log('  ceil kCore kStab  move          maxTq   minPen(mm)  peak trunk above tread, core 9 (mm)                  file');
for (const r of S.rows) {
  if (r.invalid) { console.log(`  INVALID (declared bounds)  ${r.move}  ${' '.repeat(62)}  ${r.file}`); continue; }
  console.log(`  ${pad(r.ceilingCore, 2)}/9  ${pad(r.kCore, 2)}/9  ${pad(r.kCoreStable, 2)}/9  ${r.move}  ${r.maxTq.toFixed(4)}  ${pad(r.minPenetrationEpisode_mm, 9)}  [${r.peakAboveTread_mm.map(x => pad(x, 6)).join(' ')}]  ${r.file}`);
}
const valid = S.rows.filter(r => !r.invalid);
const h1 = {}; for (const r of valid) h1[r.ceilingCore] = (h1[r.ceilingCore] || 0) + 1;
console.log(`  ${valid.length} scorable files, ${S.distinctVectors} distinct vectors; ceilingCore histogram ${JSON.stringify(h1)}; highest ${S.maxCeilingCore}/9\n`);

console.log(`=== SEARCH: every distinct move, sorted by ceilingCore (${R.distinct} distinct vectors, ${R.fullEvals} full 9-cell evaluations, ${R.evals} candidates, ${R.generationsRun ?? R.generations.length} generations, ${R.wall_s}s) ===`);
console.log('  ceil  objective  kCore kStab  move          maxTq   minPen(mm)  satFrac  peak trunk above tread, core 9 (mm)');
for (const r of R.distinctMoves.slice(0, +(process.argv[2] || 40)))
  console.log(`  ${pad(r.ceilingCore, 2)}/9  ${pad(r.objective.toFixed(4), 9)}  ${pad(r.kCore, 2)}/9 ${pad(r.kCoreStable, 4)}/9  ${r.move}  ${r.maxTq.toFixed(4)}  ${pad(r.minPenetrationEpisode_mm, 9)}  ${r.satFrac.toFixed(4)}  [${r.peakAboveTread_mm.map(x => pad(x, 6)).join(' ')}]`);
const h2 = {}; for (const r of R.distinctMoves) h2[r.ceilingCore] = (h2[r.ceilingCore] || 0) + 1;
console.log(`  ceilingCore histogram over the ${R.distinct} distinct moves: ${JSON.stringify(h2)}`);
console.log(`  BEST ceilingCore: ${R.bestCeilingCore} of 9`);
console.log(`  KILL CONDITION (needs >= 7 of 9 at 60 mm): ${R.killCondition.result}`);
console.log(`  published: ${R.published.map(f => `${f.file} (${f.sha256.slice(0, 12)}, ceil ${f.ceilingCore})`).join('\n             ')}`);
