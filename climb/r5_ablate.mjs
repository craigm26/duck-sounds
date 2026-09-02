// r5_ablate.mjs — THE ABLATION. Round 5's family changes two things at once:
// the launch is retuned AND the keyframe landing is replaced by the servo. So
// for each saved round-5 best, score the SAME FILE with the `servo` block
// deleted. Whatever the servo actually contributes is the difference.
import fs from 'node:fs';
import { scoreRobust, intentHash, saveIntent } from '../climb/robust.mjs';
const rows = [];
for (const f of ['best_r5_servoland_60mm.json', 'best_r5_servoland_kcore_60mm.json']) {
  const j = JSON.parse(fs.readFileSync('../climb/' + f, 'utf8'));
  const withServo = await scoreRobust('../climb/' + f, { rise: 0.060 });
  const k = { ...j }; delete k.servo; delete k.robust;
  k.name = (j.name || f) + '_LAUNCH_ONLY';
  k.note = 'ABLATION ONLY: ' + f + ' with the servo block deleted. Same launch, the round-3 keyframe landing back in place.';
  const p = '../climb/_r5_ablate_noservo.json';
  saveIntent(k, p);
  const noServo = await scoreRobust(p, { rise: 0.060 });
  const line = { file: f, withServo: { move: withServo.move, kCore: withServo.kCore, kCoreStable: withServo.kCoreStable, kExt: withServo.kExt, objective: +withServo.objective.toFixed(4), meanUpTail: +withServo.agg.meanUprightTailTicks.toFixed(1), minPenEp_mm: +withServo.agg.minPenetrationEpisode_mm.toFixed(2) },
    launchOnly: { move: noServo.move, kCore: noServo.kCore, kCoreStable: noServo.kCoreStable, kExt: noServo.kExt, objective: +noServo.objective.toFixed(4), meanUpTail: +noServo.agg.meanUprightTailTicks.toFixed(1), minPenEp_mm: +noServo.agg.minPenetrationEpisode_mm.toFixed(2) } };
  rows.push(line);
  console.log(`${f}`);
  console.log(`   with the servo : ${line.withServo.move} kCore=${line.withServo.kCore}/9 kCoreStable=${line.withServo.kCoreStable}/9 kExt=${line.withServo.kExt}/14 obj=${line.withServo.objective} meanUpTail=${line.withServo.meanUpTail}/50 minPenEp=${line.withServo.minPenEp_mm}mm`);
  console.log(`   launch only    : ${line.launchOnly.move} kCore=${line.launchOnly.kCore}/9 kCoreStable=${line.launchOnly.kCoreStable}/9 kExt=${line.launchOnly.kExt}/14 obj=${line.launchOnly.objective} meanUpTail=${line.launchOnly.meanUpTail}/50 minPenEp=${line.launchOnly.minPenEp_mm}mm`);
  console.log(`   honest pattern with servo : ${withServo.verdicts.map(v => v.honest ? 1 : 0).join('')}`);
  console.log(`   honest pattern launch only: ${noServo.verdicts.map(v => v.honest ? 1 : 0).join('')}`);
  console.log(`   peakAbove_mm with servo   : [${withServo.verdicts.filter(v => v.tier === 'core').map(v => (v.peakZ_mm - v.rise_mm).toFixed(1)).join(', ')}]`);
  console.log(`   peakAbove_mm launch only  : [${noServo.verdicts.filter(v => v.tier === 'core').map(v => (v.peakZ_mm - v.rise_mm).toFixed(1)).join(', ')}]`);
  console.log(`   upTail with servo         : [${withServo.verdicts.filter(v => v.tier === 'core').map(v => v.uprightTailTicks).join(', ')}]`);
  console.log(`   upTail launch only        : [${noServo.verdicts.filter(v => v.tier === 'core').map(v => v.uprightTailTicks).join(', ')}]`);
}
fs.writeFileSync('../climb/r5_ablation-results.json', JSON.stringify({ note: 'each round-5 best scored with, and without, its servo block; the launch and every other field identical', rows }, null, 2));
console.log('wrote climb/r5_ablation-results.json');
