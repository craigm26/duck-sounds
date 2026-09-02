// famB_table.mjs — render climb/r4_famB-results.json as the round-4 table.
// Read-only: it scores nothing, it prints what the run already wrote.
import fs from 'node:fs';
const R = JSON.parse(fs.readFileSync('/home/craigm26/projects/duck-sounds/climb/r4_famB-results.json', 'utf8'));
const p = (...a) => console.log(...a);

p('PARITY (climb/rig3_prefamB.mjs / robust_prefamB.mjs vs the extended instrument):', R.instrument.parityAll);
for (const r of R.instrument.parity) p(`  ${r.file.padEnd(30)} rig3=${r.rig3EXACT} robust=${r.robustEXACT} hash=${r.hashEXACT} (${r.sha256_8})`);
p('');
p('BEAT 1 — from the floor. Terminal = the handoff instant (last keyframe + 0.8 s).');
p('rise  move      trunk x/z (mm)   dy    beak  foot-riser  feetOnTread  upright  pen(mm)  peakZ   maxDY  evals');
for (const b of R.beat1) {
  const t = b.terminal;
  p(`${String(b.rise_mm).padStart(4)}  ${b.sha256.slice(0, 8)}  ${String(t.trunk_x_mm).padStart(6)} /${String(t.trunk_z_mm).padStart(6)}  ${String(t.trunk_dy_mm).padStart(6)}  ${String(t.beakOnTread).padEnd(5)} ${String(t.footOnRiser).padEnd(10)} ${String(t.feetRestingOnTread).padEnd(11)} ${String(t.upright).padEnd(7)} ${String(t.penetration_mm).padStart(7)}  ${String(b.peakZ_mm).padStart(6)} ${String(b.maxAbsDY_mm).padStart(6)}  ${b.evals}`);
}
p('');
p('BEAT 2 — FROM A HANDOFF SPAWN. Not a climb.');
p('rise  move      kCore  kCoreStable  objectiveR3  objectiveCore  meanReward  minUpTail  meanAbove(mm)  meanX(mm)  evals');
for (const b of R.beat2) {
  p(`${String(b.rise_mm).padStart(4)}  ${b.sha256.slice(0, 8)}  ${String(b.kCore).padStart(2)}/9   ${String(b.kCoreStable).padStart(2)}/9        ${b.objectiveR3.toFixed(3).padStart(7)}      ${b.objectiveCore.toFixed(3).padStart(7)}     ${b.meanReward.toFixed(3).padStart(7)}    ${String(b.agg.minUprightTailTicks).padStart(2)}/50     ${b.agg.meanAbove_mm.toFixed(1).padStart(7)}     ${b.agg.meanX_mm.toFixed(1).padStart(7)}   ${b.evals}`);
}
p('');
p('BEAT 1+2 CONCATENATED — one file, spawned on the floor. This is the climb.');
p('rise  move      kCore kCoreStable kExt kExtStable  objective  objR3   meanReward  maxZ(mm) meanAbove meanX  feetOnTreadMax minUpTail minPen  maxDY  evals');
for (const b of R.concat) {
  p(`${String(b.rise_mm).padStart(4)}  ${b.sha256.slice(0, 8)}  ${String(b.kCore).padStart(2)}/9   ${String(b.kCoreStable).padStart(2)}/9    ${String(b.kExt).padStart(2)}/14  ${String(b.kExtStable).padStart(2)}/14   ${b.objective.toFixed(3).padStart(7)}  ${b.objectiveR3.toFixed(3).padStart(7)} ${b.meanReward.toFixed(3).padStart(7)}   ${(b.agg.maxZ * 1000).toFixed(1).padStart(7)} ${b.agg.meanAbove_mm.toFixed(1).padStart(8)} ${b.agg.meanX_mm.toFixed(1).padStart(6)}  ${String(b.agg.feetOnTreadMax).padStart(3)}          ${String(b.agg.minUprightTailTicks).padStart(2)}/50  ${b.agg.minPenetrationAtScore_mm.toFixed(2).padStart(6)} ${b.agg.maxAbsDY_mm.toFixed(1).padStart(6)}  ${b.evals}`);
}
p('');
p('BEAT 1 ALONE on the shared grid (it is a saved file like any other):');
for (const g of R.grids) p(`${String(g.rise_mm).padStart(4)}  ${g.sha256.slice(0, 8)}  kCore=${g.kCore}/9 kExt=${g.kExt}/14 objective=${g.objective.toFixed(3)} objR3=${g.objectiveR3.toFixed(3)}`);
p('');
p('PER-CELL VERDICTS, concatenated files:');
for (const b of R.concat) {
  p(`  rise ${b.rise_mm} mm, move ${b.sha256.slice(0, 8)}:`);
  for (const v of b.verdicts) {
    p(`    [${v.tier}] rise=${String(v.rise_mm).padStart(3)} drop=${v.drop} f=${v.fmul}  honest=${String(v.honest).padEnd(5)} stable=${String(v.stableClear).padEnd(5)} upTail=${String(v.uprightTailTicks).padStart(2)}/50 rew=${String(v.reward).padStart(6)} x=${String(v.x_mm).padStart(6)} above=${String(v.above_mm).padStart(6)} fot=${v.feetOnTread} fotMax=${v.feetOnTreadMax} peakZ=${String(v.peakZ_mm).padStart(6)} pen=${v.penetrationAtScore_mm} maxDY=${v.maxAbsDY_mm}`);
  }
}
const allSha = [...R.beat1, ...R.beat2, ...R.concat].map(b => b.sha256);
p('');
p(`DISTINCT VECTORS: ${new Set(allSha).size} of ${allSha.length} files (a repeated hash would be one move under two labels)`);
