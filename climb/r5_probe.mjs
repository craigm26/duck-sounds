// r5_probe.mjs — round-5 phase 0: baseline the warm start on the round-5
// instrument, measure the cost of one 14-cell evaluation, and measure how high
// the trunk actually gets in every cell (the number a leg law cannot change).
import fs from 'node:fs';
import { scoreRobust, intentHashOfFile } from '../climb/robust.mjs';

const t0 = Date.now();
const g = await scoreRobust('../climb/best_r3_vault_60mm.json', { rise: 0.060 });
const dt = (Date.now() - t0) / 1000;
console.log(`WARM START best_r3_vault_60mm @60mm  sha=${g.move}`);
console.log(`  kCore=${g.kCore}/9 kCoreStable=${g.kCoreStable}/9 kExt=${g.kExt}/14 kExtStable=${g.kExtStable}/14`);
console.log(`  meanReward=${g.meanReward.toFixed(4)} objective=${g.objective.toFixed(4)} objectiveCore=${g.objectiveCore.toFixed(4)} objectiveR3=${g.objectiveR3.toFixed(4)}`);
console.log(`  minPenAtScore=${g.agg.minPenetrationAtScore_mm.toFixed(2)}mm minPenEpisode=${g.agg.minPenetrationEpisode_mm.toFixed(2)}mm ticksScanned=${g.cells[0].penetrationTicksScanned}`);
console.log(`  14-cell wall time ${dt.toFixed(1)} s  (${(dt/14).toFixed(2)} s/cell)`);
console.log('  cell                       honest stable  peakAbove_mm  aboveAtScore_mm  x_mm  fotMax  minPenEp_mm');
for (const c of g.cells) {
  const h = c.rise;
  console.log(`   r${String(c.cell.rise_mm).padStart(3)} d${c.cell.drop} f${c.cell.fmul} ${c.cell.tier.padEnd(4)}  ${String(c.crit.honest).padEnd(6)} ${String(c.stableClear).padEnd(6)}  ${((c.maxZ-h)*1000).toFixed(1).padStart(10)}  ${(c.scored.above*1000).toFixed(1).padStart(13)}  ${(c.scored.x*1000).toFixed(1).padStart(7)}  ${String(c.feetOnTreadMax).padStart(4)}  ${c.minPenetrationEpisode===null?'null':(c.minPenetrationEpisode*1000).toFixed(2).padStart(8)}`);
}
const t1 = Date.now();
const gc = await scoreRobust('../climb/best_r3_vault_60mm.json', { rise: 0.060, core: true });
console.log(`  CORE-ONLY 9-cell wall time ${((Date.now()-t1)/1000).toFixed(1)} s  objectiveCore=${gc.objectiveCore.toFixed(4)}`);
