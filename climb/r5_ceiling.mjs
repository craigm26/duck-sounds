// r5_ceiling.mjs — THE CEILING A LANDING LAW CANNOT RAISE.
//
// rig3 criteria(): honest requires scored.above > 0.095 m. A cell whose trunk
// never gets 95 mm above the tread at ANY tick cannot be cleared by ANY
// landing, servoed or thrown — the landing only decides what happens to the
// feet once the trunk is up there. So for every move scored in round 5 this
// counts, per cell, whether peak trunk height above the tread ever exceeded
// 95 mm. That count is an UPPER BOUND on that move's kCore.
import fs from 'node:fs';
const R = JSON.parse(fs.readFileSync('../climb/r5_servoland-results.json', 'utf8'));
const rows = [];
const one = (label, sha, verdicts, kCore, kCoreStable, obj) => {
  const core = verdicts.filter(v => v.tier === 'core');
  const over = core.filter(v => v.peakZ_mm - v.rise_mm > 95).length;
  const all = verdicts.filter(v => v.peakZ_mm - v.rise_mm > 95).length;
  rows.push({ label, move: sha.slice(0, 12), kCore, kCoreStable, objective: obj,
    coreCellsWhereTrunkEverReached95mm: over, extCellsWhereTrunkEverReached95mm: all,
    peakAbove_mm_core: core.map(v => +(v.peakZ_mm - v.rise_mm).toFixed(1)) });
};
one(R.warmStart.file, R.warmStart.sha256, R.warmStart.verdicts, R.warmStart.kCore, R.warmStart.kCoreStable, R.warmStart.objective);
for (const t of R.table) one(t.label || '', t.sha256, t.verdicts, t.kCore, t.kCoreStable, t.objective);
rows.sort((a, b) => b.coreCellsWhereTrunkEverReached95mm - a.coreCellsWhereTrunkEverReached95mm || b.kCore - a.kCore);
console.log('THE CEILING: core cells (of 9) in which the trunk EVER reached 95 mm above the tread');
console.log('  move          kCore kStable objective  ceiling(core/9)  ext/14   peak trunk above tread, core 9 (mm)');
for (const r of rows.slice(0, 30))
  console.log(`  ${r.move}  ${String(r.kCore).padStart(2)}/9 ${String(r.kCoreStable).padStart(4)}/9 ${r.objective.toFixed(3).padStart(9)}  ${String(r.coreCellsWhereTrunkEverReached95mm).padStart(9)}/9  ${String(r.extCellsWhereTrunkEverReached95mm).padStart(2)}/14   [${r.peakAbove_mm_core.join(', ')}]  ${(r.label || '').slice(0, 46)}`);
const maxCeil = Math.max(...rows.map(r => r.coreCellsWhereTrunkEverReached95mm));
console.log(`\n  HIGHEST CEILING OVER EVERY MOVE ROUND 5 SCORED: ${maxCeil} of 9 core cells.`);
console.log(`  The kill gate needs 7 of 9. ${maxCeil >= 7 ? 'The ceiling does not block it.' : 'NO MOVE ROUND 5 SCORED COULD REACH 7 EVEN WITH A PERFECT LANDING.'}`);
fs.writeFileSync('../climb/r5_ceiling-results.json', JSON.stringify({ rule: 'peak trunk z - rise > 95 mm at any tick; honest requires above > 95 mm at the scored instant', maxCeilingCore: maxCeil, rows }, null, 2));
console.log('  wrote climb/r5_ceiling-results.json');
