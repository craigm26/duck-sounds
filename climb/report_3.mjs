// Format climb/search_3-results.json into the round-2 table.
// Pure reader: no MuJoCo, no ONNX, no episode. Run AFTER search_3.mjs.
//   cd ~/projects/duck-sounds/climb && node report_3.mjs
import fs from 'node:fs';
const R = JSON.parse(fs.readFileSync('search_3-results.json', 'utf8'));
const p = (v, n) => String(v).padStart(n);

console.log('GUARD (instrumented rig3 still reproduces rig3.log Phase E):', JSON.stringify(R.guard));
console.log('total episodes', R.totalEvals, 'elapsed_s', R.elapsed_s, '\n');

console.log('=== ROUND 2, FAMILY C (corner stem, contact order) — rig3 `honest`, tail=policy ===');
console.log('rise steps  episodes  objective  cleared/of  peakZ_mm  head  riser  bothFrac  gainBoth_mm  failure mode');
let prev = 0;
for (const k of Object.keys(R.arms)) {
  const a = R.arms[k], b = a.reScoredFromFile;
  const n = a.evals - prev; prev = a.evals;
  const clears = a.improvements.filter(i => i.honest).length
    + a.perOrderBest.filter(x => x && x.honest).length;
  console.log(p(a.rise_mm, 4) + p(a.stepCount, 6) + p(n, 10) + p(b.reward.toFixed(3), 11) +
    p(`${clears ? '>=1' : 0}/${n}`, 12) + p(b.peakZ_mm.toFixed(1), 10) +
    p(b.headFrac.toFixed(2), 6) + p(b.riserFrac.toFixed(2), 7) + p(b.bothFrac.toFixed(2), 10) +
    p(String(b.maxGainBoth_mm), 13) + '  ' + b.why);
}

console.log('\n=== CONTACT ORDER — best objective per order (the thing nobody searched) ===');
for (const k of Object.keys(R.arms)) {
  const a = R.arms[k];
  console.log(`  ${a.rise_mm} mm (${a.stepCount}-step):`);
  console.log('    order  best_r   maxX_mm  peakZ_mm  head  riser  gainBoth_mm  feetTreadMax  failure mode');
  for (const e of a.perOrderBest.filter(Boolean).sort((x, y) => y.reward - x.reward))
    console.log('    ' + e.order.padEnd(7) + p(e.reward.toFixed(2), 6) + p(e.maxX_mm.toFixed(1), 10) +
      p(e.peakZ_mm.toFixed(1), 10) + p(e.headFrac.toFixed(2), 6) + p(e.riserFrac.toFixed(2), 7) +
      p(String(e.maxGainBoth_mm), 13) + p(e.feetOnTreadMax, 14) + '  ' + e.why);
}

console.log('\n=== BEST OF EACH ARM, re-scored from the published file ===');
for (const k of Object.keys(R.arms)) {
  const a = R.arms[k];
  console.log(`  ${a.bestPath}`);
  console.log(`    order ${a.best.order}  full sequence ${a.best.sequence}  lead ${a.best.lead}`);
  console.log(`    search r=${a.best.reward}  file r=${a.reScoredFromFile.reward}  reproduces=${a.reproduces}`);
  console.log(`    ${JSON.stringify(a.reScoredFromFile)}`);
}

console.log('\n=== -10 / 0 / +10 mm, both flights, scored from the published files ===');
console.log('file                                 rise steps  objective  x_mm  peakZ_mm above_mm feetTread head riser  HONEST  mode');
for (const r of R.robustness)
  console.log(r.file.replace('../climb/', '').padEnd(37) + p(r.rise_mm, 4) + p(r.stepCount, 6) +
    p(r.reward.toFixed(2), 11) + p(r.x_mm.toFixed(1), 6) + p(r.peakZ_mm.toFixed(1), 10) +
    p(r.above_mm.toFixed(1), 9) + p(r.feetOnTread, 10) + p(r.headFrac.toFixed(2), 5) +
    p(r.riserFrac.toFixed(2), 6) + p(r.honest ? 'PASS' : '.', 8) + '  ' + r.why);
