// Merge the per-rise runs of climb/search_0.mjs into one results file + table.
// Run:  cd ~/projects/duck-sounds/sim && node ../climb/search_0-collect.mjs
import fs from 'node:fs';
const dir = '../climb';
const files = fs.readdirSync(dir).filter(f => /^search_0-\d+mm-seed\d+\.json$/.test(f));
const rows = files.map(f => JSON.parse(fs.readFileSync(`${dir}/${f}`, 'utf8')))
  .sort((a, b) => a.riseMM - b.riseMM);
console.log('rise  evals  bestS    onTop  cleared  peakZ   trunkX  above  feetUp  head%  footRiserTicks  failure');
for (const r of rows) {
  const b = r.best;
  console.log(`${String(r.riseMM).padStart(4)}  ${String(r.evals).padStart(5)}  ${b.score.toFixed(3).padStart(7)}  `
    + `${String(b.onTop).padEnd(5)}  ${r.cleared}/${r.of}      ${String(b.peakZ).padStart(6)}  `
    + `${String(b.x).padStart(6)}  ${String(b.above).padStart(5)}  ${b.feetUp}       `
    + `${(b.headFrac * 100).toFixed(1).padStart(5)}  ${String(b.footRiserTicks).padStart(3)}             ${r.failure}`);
}
fs.writeFileSync(`${dir}/search_0-results.json`, JSON.stringify({
  strategy: 'A - beak hook + wall walk',
  criterion: 'sim/climb_lib.mjs:150 - up && x>0.12 && (z-h)>0.095 && feetUp>=2, re-checked 1 s after the track',
  runs: rows.map(r => ({
    riseMM: r.riseMM, seed: r.seed, seconds: r.seconds, evals: r.evals,
    cleared: r.cleared, of: r.of, failure: r.failure,
    best: r.best, offsets: r.offsets, command: r.command,
  })),
}, null, 2));
console.log('\nwrote ../climb/search_0-results.json');
