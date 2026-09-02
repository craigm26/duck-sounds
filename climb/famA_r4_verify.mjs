// Family A round-4 VERIFICATION, run as its own process AFTER the search:
// does the exported file replay, is it inside its declared bounds, and is it a
// behaviourally NEW move or the round-3 vector wearing an event?
import fs from 'node:fs';
import { scoreRobust, intentHashOfFile, checkBounds } from './robust.mjs';
const P = '../climb/';
const A = await scoreRobust(P + 'best_r4_famA_60mm.json', { rise: 0.060, isolate: true });
const B = await scoreRobust(P + 'best_r3_vault_60mm.json', { rise: 0.060, isolate: true });
const out = { generated: new Date().toISOString() };
for (const [tag, r, f] of [['r4_famA_60mm', A, 'best_r4_famA_60mm.json'], ['r3_vault_60mm', B, 'best_r3_vault_60mm.json']]) {
  out[tag] = { sha256: r.sha256, move: r.move, kCore: r.kCore, kCoreStable: r.kCoreStable,
    kExt: r.kExt, kExtStable: r.kExtStable, objective: +r.objective.toFixed(4),
    objectiveCore: +r.objectiveCore.toFixed(4), objectiveR3: +r.objectiveR3.toFixed(4),
    meanReward: +r.meanReward.toFixed(4), minPen_mm: +r.agg.minPenetrationAtScore_mm.toFixed(2),
    maxAbsDY_mm: +r.agg.maxAbsDY_mm.toFixed(1), lateralEscapeCells: r.agg.lateralEscapeCells,
    minUprightTailTicks: r.agg.minUprightTailTicks, bounds: checkBounds(JSON.parse(fs.readFileSync(P + f, 'utf8'))) };
}
// cell-for-cell: is the round-4 file the same TRAJECTORY as the round-3 one?
let maxdx = 0, maxdz = 0, sameHonest = true;
const rows = A.verdicts.map((v, i) => {
  const w = B.verdicts[i];
  maxdx = Math.max(maxdx, Math.abs(v.x_mm - w.x_mm)); maxdz = Math.max(maxdz, Math.abs(v.z_mm - w.z_mm));
  if (v.honest !== w.honest) sameHonest = false;
  return { cell: `${v.rise_mm}mm d${v.drop} f${v.fmul} [${v.tier}]`, r4Honest: v.honest, r3Honest: w.honest,
           dx_mm: +(v.x_mm - w.x_mm).toFixed(3), dz_mm: +(v.z_mm - w.z_mm).toFixed(3),
           eventFired: v.eventFired, eventT: v.eventT, eventE_mm: v.eventE_mm,
           upTail: `${v.uprightTailTicks}/${v.tailTicks}`, pen_mm: v.penetrationAtScore_mm, maxDY_mm: v.maxAbsDY_mm };
});
out.sameTrajectory = { maxAbsDx_mm: +maxdx.toFixed(4), maxAbsDz_mm: +maxdz.toFixed(4), sameHonestPattern: sameHonest,
  verdict: (maxdx === 0 && maxdz === 0) ? 'IDENTICAL physics — the round-4 file is the round-3 vector re-expressed as an event, a distinct hash but NOT a new move' : 'different trajectory' };
out.cells = rows;
out.hashes = { r4: intentHashOfFile(P + 'best_r4_famA_60mm.json'), r3: intentHashOfFile(P + 'best_r3_vault_60mm.json') };
fs.writeFileSync(P + 'r4_famA_verify.json', JSON.stringify(out, null, 2));
console.log(JSON.stringify({ sameTrajectory: out.sameTrajectory, r4: out.r4_famA_60mm, r3: out.r3_vault_60mm }, null, 2));
