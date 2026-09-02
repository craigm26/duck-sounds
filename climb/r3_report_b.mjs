// r3_report_b.mjs — read climb/r3_landvault-results.json and print the round-3
// family-B table: rise -> best k of 9, objective, per-cell verdicts, trunk peak
// z, head contact fraction, foot-on-riser fraction, failure mode.
// Run from anywhere:  node climb/r3_report_b.mjs
import fs from 'node:fs';
const P = new URL('./r3_landvault-results.json', import.meta.url).pathname;
const R = JSON.parse(fs.readFileSync(P, 'utf8'));

const pad = (s, n) => String(s).padEnd(n);
const rp = (s, n) => String(s).padStart(n);

console.log('BASELINES (9-cell grid, climb/robust.mjs)');
console.log(pad('label', 34) + rp('rise', 5) + rp('k/9', 5) + rp('obj', 8) + rp('peakZ', 8) + rp('gain', 7) + rp('head', 7) + rp('riser', 7) + rp('wall', 7) + rp('fotMax', 7) + rp('bothNear', 9));
for (const b of R.baselines)
  console.log(pad(b.label, 34) + rp(b.rise_mm, 5) + rp(b.k, 5) + rp(b.objective, 8) + rp(b.peakZ_mm ?? '-', 8) +
    rp(b.peakGain_mm, 7) + rp(b.headFrac, 7) + rp(b.riserFrac, 7) + rp(b.wallFrac, 7) + rp(b.feetOnTreadMax, 7) + rp(b.bothNear_mm ?? '-', 9));

for (const run of R.runs) {
  console.log(`\n=== ${run.rise_mm} mm — ${run.tag} : ${run.gens} generations, ${run.evals} episodes screened, ${run.restarts ?? 0} restarts ===`);
  if (!run.best) { console.log('  no candidate ever reached the full grid'); continue; }
  const b = run.best;
  console.log(`  BEST  cleared ${b.k} of 9   objective ${b.objective}   meanReward ${b.meanReward}   contact order: ${b.order}`);
  console.log(`  peak trunk z ${(b.agg.meanMaxZ * 1000).toFixed(1)} mm (mean over cells; max cell ${(b.agg.maxZ * 1000).toFixed(1)} mm), gain over settle ${(b.agg.meanPeakGain * 1000).toFixed(1)} mm`);
  console.log(`  head contact ${b.agg.headFrac}   foot-on-riser ${b.agg.riserFrac}   wall contact ${b.agg.wallFrac}   wall load-bearing ${b.agg.wallBearFrac}`);
  console.log(`  sustained load transfer ${b.agg.sustainFrac}   two-contact ${b.agg.bothFrac}   feet-on-tread max ${b.agg.feetOnTreadMax}   final feet-on-tread ${b.agg.meanFeetOnTreadFinal}`);
  console.log(`  closest both-feet approach to a tread landing spot ${b.agg.bothNear_mm?.toFixed?.(1) ?? b.agg.bothNear_mm} mm ; best single foot ${b.agg.footNear_mm?.toFixed?.(1) ?? b.agg.footNear_mm} mm`);
  console.log(`  saturation ${b.agg.satFrac}   max |actuator force| ${b.agg.maxTq} N.m   max |y-STAIR_Y| ${b.agg.maxAbsDY_mm} mm   tread drift ${b.agg.maxTreadDriftX_mm} mm`);
  console.log('  per-cell verdicts:');
  console.log('   ' + pad('rise', 6) + pad('drop', 7) + pad('fric', 6) + pad('honest', 8) + rp('rew', 7) + rp('x_mm', 9) + rp('above', 8) + rp('dy', 8) + rp('up', 6) + rp('fot', 5) + rp('fotMax', 7) + rp('peakZ', 8) + rp('head', 7) + rp('riser', 7) + rp('wall', 7));
  for (const v of b.verdicts)
    console.log('   ' + pad(v.rise_mm, 6) + pad(v.drop, 7) + pad(v.fmul, 6) + pad(v.honest, 8) + rp(v.reward, 7) +
      rp(v.x_mm, 9) + rp(v.above_mm, 8) + rp(v.dy_mm, 8) + rp(v.up, 6) + rp(v.feetOnTread, 5) + rp(v.feetOnTreadMax, 7) +
      rp(v.peakZ_mm, 8) + rp(v.headFrac, 7) + rp(v.riserFrac, 7) + rp(v.wallFrac, 7));
}

console.log('\nIMPROVEMENT TRAIL (every one graded on the full 9 cells)');
console.log(pad('t_s', 6) + pad('rise', 6) + pad('gen', 5) + pad('k/9', 5) + rp('obj', 8) + rp('fit', 8) + pad('  order', 8) + rp('peakZ', 8) + rp('gain', 7) + rp('head', 7) + rp('riser', 7) + rp('wall', 7) + rp('sust', 7) + rp('fotMax', 7) + rp('bothNear', 9));
for (const i of R.improvements)
  console.log(pad(i.t_s, 6) + pad(i.rise_mm, 6) + pad(i.gen, 5) + pad(i.k, 5) + rp(i.objective, 8) + rp(i.searchFit, 8) +
    pad('  ' + ['beak', 'riser', 'both'][i.order], 8) + rp(i.peakZ_mm, 8) + rp(i.peakGain_mm, 7) + rp(i.headFrac, 7) +
    rp(i.riserFrac, 7) + rp(i.wallFrac, 7) + rp(i.sustainFrac, 7) + rp(i.feetOnTreadMax, 7) + rp(i.bothNear_mm ?? '-', 9));

if (R.cross?.length) {
  console.log('\nCROSS-SCORES');
  for (const c of R.cross) console.log(`  ${pad(c.label, 34)} k=${c.k}/9  objective ${c.objective}`);
}
console.log(`\ntotals: ${R.totals.evals} episodes screened, ${R.totals.fullGrids} full 9-cell grids, ${R.totals.wall_s} s wall`);
