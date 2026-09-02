// r5_save_kcore.mjs — save the SECOND distinct round-5 vector: the servoed move
// with the most core clears (3 of 9, none stable), which is not the vector with
// the highest objective. Rebuilt from its recorded params through the same
// builder, re-scored from the saved file, and refused if the hash moves.
import fs from 'node:fs';
import { scoreRobust, intentHash, saveIntent, HOME } from '../climb/robust.mjs';
const R = JSON.parse(fs.readFileSync('../climb/r5_servoland-results.json', 'utf8'));
const TARGET = process.argv[2] || '880a120ef649';
const row = R.table.find(t => t.move === TARGET);
if (!row) throw new Error('no such move ' + TARGET);
const P = row.params;
const F1 = [0,-0.21255,-0.36327,0.4539,0.48228,-0.32872,-0.7469,0,0,0,-0.03795,0.36327,-0.4539,-0.48228];
const F2 = [0,-0.21255,-0.24628,-0.50213,0.12444,-0.32872,-0.7469,0,0,0,-0.03795,0.24628,0.50213,-0.12444];
const F3 = [0,-0.21255,-0.03892,0.67747,0.50507,-0.32872,-0.7469,0,0,0,-0.03795,0.03892,-0.67747,-0.50507];
const F4 = [0,-0.21255,0.03891,-1.05397,0.83188,-0.32872,-0.7469,0,0,0,-0.03795,-0.03891,1.05397,-0.83188];
const F5 = [0,-0.0873,-0.26283,-0.25249,0.8677,0.58128,-0.28134,0,0,0,0.0873,0.26283,0.25249,-0.8677];
const F6 = [0,-0.0873,-0.4579,-0.0049,0.453,0.3491,0.3491,0,0,0,0.0873,0.4579,0.0049,-0.453];
const r4 = v => +v.toFixed(4), r6 = v => +v.toFixed(6);
const t1 = r4(P.dReach), t2 = r4(t1 + P.dPre), t3 = r4(t2 + P.dVault), t4 = r4(t3 + P.dTuck);
const t5 = r4(t4 + 0.5827), t6 = r4(t5 + 0.7);
const at = r4(Math.min(Math.max(t3 + P.atFrac * (t4 - t3), t3), t6));
const mix = F4.map((v, i) => v + (F5[i] - v) * P.baseMix);
const B = mix.map((v, i) => HOME[i] + (v - HOME[i]) * P.blend).map(r6);
const j = {
  name: 'servoed_landing_r5_kcore_60mm',
  family: 'Round 5: the round-3 beak-strut LAUNCH + a per-tick servoed landing (climb/servo.mjs)',
  keyframes: [{ t: t1, pose: F1 }, { t: t2, pose: F2 }, { t: t3, pose: F3 },
              { t: t4, pose: F4 }, { t: t5, pose: F5 }, { t: t6, pose: F6 }],
  blend: P.blend, gap: P.gap, side: P.side, approach: 0.1663, isolate: true, stepCount: 4,
  servo: { at, base: { hip: [B[2], B[11]], knee: [B[3], B[12]], ankle: [B[4], B[13]] },
    yawRoll: [B[0], B[1], B[9], B[10]],
    zTarget: P.zTarget, xFoot: P.xFoot, fz: P.fz, pitchRef: 0,
    kHipZ: P.kHipZ, kHipPitch: P.kHipPitch, kHipX: P.kHipX,
    kKneeZ: P.kKneeZ, kKneeFz: P.kKneeFz, kKneeX: P.kKneeX,
    kAnkPitch: P.kAnkPitch, kAnkFz: P.kAnkFz, rate: P.rate, span: P.span },
  params: P, bounds: { blend: [0.7, 2.4], side: [-0.02, 0.09] },
};
const h = intentHash(j);
console.log(`rebuilt ${h.slice(0, 12)} vs recorded ${row.sha256.slice(0, 12)}  MATCH=${h === row.sha256}`);
if (h !== row.sha256) throw new Error('rebuild does not reproduce the searched vector');
const PATH = '../climb/best_r5_servoland_kcore_60mm.json';
if (fs.existsSync(PATH)) { console.log('REFUSING to overwrite ' + PATH); process.exit(0); }
saveIntent(j, PATH);
const g = await scoreRobust(PATH, { rise: 0.060 });
console.log(`re-scored from the saved file: kCore=${g.kCore}/9 kCoreStable=${g.kCoreStable}/9 kExt=${g.kExt}/14 obj=${g.objective.toFixed(4)} minPenEpisode=${g.agg.minPenetrationEpisode_mm.toFixed(2)}mm  sameAsSearch=${g.kCore === row.kCore && g.objective === row.objective}`);
for (const v of g.verdicts)
  console.log(`   [${v.tier}] rise=${v.rise_mm} drop=${v.drop} f=${v.fmul} honest=${v.honest} stable=${v.stableClear} upTail=${v.uprightTailTicks}/50 penScore=${v.penetrationAtScore_mm}mm minPenEp=${v.minPenetrationEpisode_mm}mm rew=${v.reward} x=${v.x_mm} above=${v.above_mm} fot=${v.feetOnTread} peakAbove=${(v.peakZ_mm - v.rise_mm).toFixed(1)}`);
j.note = `ROUND 5, the servoed landing at 60 mm: the vector with the MOST core clears of any servoed move (kCore ${g.kCore}/9), none of them stable (kCoreStable ${g.kCoreStable}/9). It is not the vector with the highest objective (that is best_r5_servoland_60mm.json). Saved as a second distinct vector.`;
j.robust = { kCore: g.kCore, nCore: 9, kCoreStable: g.kCoreStable, kExt: g.kExt, nExt: 14,
  kExtStable: g.kExtStable, objective: g.objective, objectiveCore: g.objectiveCore,
  objectiveR3: g.objectiveR3, meanReward: g.meanReward, sha256: g.sha256,
  minPenetrationEpisode_mm: g.agg.minPenetrationEpisode_mm,
  minPenetrationAtScore_mm: g.agg.minPenetrationAtScore_mm, maxTq: g.agg.maxTq,
  verdicts: g.verdicts, agg: g.agg };
const h2 = intentHash(j);
saveIntent(j, PATH);
console.log(`saved ${PATH}  sha256 ${g.sha256}  (hash unchanged by the stamp: ${h2 === g.sha256})`);
