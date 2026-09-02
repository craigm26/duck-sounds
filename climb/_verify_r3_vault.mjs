// Independent re-score of every round-3 family A best, from its saved file:
// the 9-cell grid via climb/robust.mjs, and the nominal cell cross-checked
// against climb/rig3.mjs scoreSaved() (the instrument) at full float digits.
// Run from sim/:  node ../climb/_verify_r3_vault.mjs
import fs from 'node:fs';
import { scoreRobust } from './robust.mjs';
import { scoreSaved } from './rig3.mjs';
const P = '../climb/';
const files = fs.readdirSync('/home/craigm26/projects/duck-sounds/climb')
  .filter(f => /^best_r3_vault_\d+mm\.json$/.test(f)).sort();
const out = [];
for (const f of files) {
  const h = parseInt(f.match(/_(\d+)mm/)[1], 10) / 1000;
  const g = await scoreRobust(P + f, { rise: h, isolate: true });
  const a = await scoreSaved(P + f, { rise: h, tail: 'policy' });
  const nom = g.verdicts.find(v => v.rise_mm === Math.round(h * 1000) && v.drop === 0.120 && v.fmul === 1.0);
  const match = nom.honest === a.crit.honest && Math.abs(nom.reward - a.reward) < 1e-3;
  console.log(`\n${f}  rise ${h * 1000} mm  ->  cleared ${g.k} of 9   objective ${g.objective.toFixed(4)}  meanReward ${g.meanReward.toFixed(4)}`);
  console.log(`  rig3 cross-check on the nominal cell: rig3 honest=${a.crit.honest} reward=${a.reward.toFixed(4)} | robust honest=${nom.honest} reward=${nom.reward}  MATCH=${match}`);
  console.log('   rise  drop  fric | honest reward     x_mm  above_mm  dy_mm fot fotMax peakZ_mm headFrac riserFrac wallFrac up');
  for (const v of g.verdicts) {
    console.log(`   ${String(v.rise_mm).padStart(4)} ${v.drop.toFixed(3)} ${v.fmul.toFixed(2)} | ${String(v.honest).padStart(6)} ${String(v.reward).padStart(6)} ${String(v.x_mm).padStart(8)} ${String(v.above_mm).padStart(9)} ${String(v.dy_mm).padStart(6)} ${v.feetOnTread}   ${v.feetOnTreadMax}   ${String(v.peakZ_mm).padStart(8)} ${String(v.headFrac).padStart(8)} ${String(v.riserFrac).padStart(9)} ${String(v.wallFrac).padStart(8)} ${v.up}`);
  }
  console.log(`  agg: maxTq ${g.agg.maxTq.toFixed(4)} N.m  satFrac ${g.agg.satFrac.toFixed(3)}  headFrac ${g.agg.headFrac.toFixed(3)}  riserFrac ${g.agg.riserFrac.toFixed(3)}  meanPeakGain ${(g.agg.meanPeakGain * 1000).toFixed(1)} mm  maxAbsDY ${g.agg.maxAbsDY_mm.toFixed(1)} mm  treadDrift ${g.agg.maxTreadDriftX_mm.toFixed(2)} mm`);
  out.push({ file: f, rise_mm: h * 1000, k: g.k, objective: +g.objective.toFixed(4),
             meanReward: +g.meanReward.toFixed(4), rig3Match: match, verdicts: g.verdicts, agg: g.agg });
}
fs.writeFileSync(P + 'r3_vault-verify.json', JSON.stringify({ generated: new Date().toISOString(), rows: out }, null, 2));
console.log(`\nwrote ${P}r3_vault-verify.json`);
