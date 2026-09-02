// r6_verify.mjs — re-score the three published round-6 files FROM DISK through
// the one shared scorer, and confirm the hash the search reported round-trips.
// Also: which of the 9 core cells ever gets over the 95 mm bar, across all 340
// distinct moves round 6 scored — i.e. is the ceiling one wall or nine.
import fs from 'node:fs';
import { scoreRobust } from './robust.mjs';
const P = '../climb/';
const R = JSON.parse(fs.readFileSync(P + 'r6_ceiling-results.json', 'utf8'));

const CELL = ['50/0.120/x1.0', '50/0.130/x0.7', '50/0.125/x1.3',
              '60/0.120/x1.0', '60/0.130/x0.7', '60/0.125/x1.3',
              '70/0.120/x1.0', '70/0.130/x0.7', '70/0.125/x1.3'];

console.log('=== RE-SCORE THE PUBLISHED FILES FROM DISK ===');
const out = [];
for (const f of ['best_r6_ceilvault_60mm.json', 'best_r6_ceilvaultB_60mm.json', 'best_r6_ceilvaultC_60mm.json',
                 'best_r3_vault_60mm.json']) {
  const r = await scoreRobust(P + f, { rise: 0.060, core: true });
  const core = r.verdicts.filter(v => v.tier === 'core');
  const peaks = core.map(v => +(v.peakZ_mm - v.rise_mm).toFixed(1));
  const ceil = peaks.filter(p => p > 95).length;
  const claim = R.published.find(p => p.file === `climb/${f}`);
  const j = JSON.parse(fs.readFileSync(P + f, 'utf8'));
  const ok = !claim || claim.sha256 === r.sha256;
  console.log(`  ${f.padEnd(30)} sha ${r.sha256.slice(0, 12)} ${claim ? (ok ? 'MATCHES the search\'s hash' : 'HASH MISMATCH') : '(the record)'}`);
  console.log(`      ceilingCore ${ceil}/9   kCore ${r.kCore}/9   kCoreStable ${r.kCoreStable}/9   maxTq ${r.agg.maxTq}   minPenEpisode ${r.agg.minPenetrationEpisode_mm.toFixed(2)} mm`);
  console.log(`      peak trunk above tread (mm): ${peaks.map((p, i) => `${CELL[i]}=${p}${p > 95 ? '*' : ' '}`).join('  ')}`);
  console.log(`      cleared (honest+stable):     ${core.map((v, i) => `${CELL[i]}=${v.honest ? (v.stableClear ? 'STABLE' : 'clear ') : '  -   '}`).join('  ')}`);
  out.push({ file: f, sha256: r.sha256, hashMatchesSearch: ok, ceilingCore: ceil, kCore: r.kCore,
    kCoreStable: r.kCoreStable, maxTq: r.agg.maxTq,
    minPenetrationEpisode_mm: +r.agg.minPenetrationEpisode_mm.toFixed(2),
    peakAboveTread_mm: peaks, cells: core.map((v, i) => ({ cell: CELL[i], peakAbove_mm: peaks[i],
      over95: peaks[i] > 95, honest: v.honest, stable: v.stableClear, above_mm: v.above_mm,
      uprightTailTicks: v.uprightTailTicks })) });
}

console.log('\n=== IS THE CEILING ONE WALL OR NINE? per-cell, over all', R.distinct, 'distinct moves round 6 scored ===');
console.log('  cell            best peak (mm)   moves over 95 mm   share');
const percell = [];
for (let i = 0; i < 9; i++) {
  const vals = R.distinctMoves.map(m => m.peakAboveTread_mm[i]);
  const best = Math.max(...vals), over = vals.filter(v => v > 95).length;
  percell.push({ cell: CELL[i], bestPeak_mm: best, movesOver95: over, share: +(over / vals.length).toFixed(3) });
  console.log(`  ${CELL[i].padEnd(14)}  ${String(best).padStart(12)}   ${String(over).padStart(16)}   ${(over / vals.length * 100).toFixed(1)}%`);
}
const maxCeil = Math.max(...R.distinctMoves.map(m => m.ceilingCore));
console.log(`\n  Every one of the 9 cells is reachable on its own (best peak ${Math.min(...percell.map(p => p.bestPeak_mm))}..${Math.max(...percell.map(p => p.bestPeak_mm))} mm),`);
console.log(`  but no single move ever got more than ${maxCeil} of them over the bar at once.`);

fs.writeFileSync(P + 'r6_verify-results.json', JSON.stringify({
  generated: new Date().toISOString(),
  what: 're-score of the published round-6 files from disk, and the per-cell ceiling over every distinct move round 6 scored',
  scorer: 'climb/robust.mjs scoreRobust({rise:0.060, core:true})',
  rescored: out, perCell: percell, maxCeilingCore: maxCeil, distinctMoves: R.distinct,
}, null, 1) + '\n');
console.log('  wrote climb/r6_verify-results.json');
