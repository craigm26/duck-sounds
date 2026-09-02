// verify_feet.mjs — is "feetOnTread" ever a STANCE?
// rig3's foot clause is z > h - 0.005 with no ceiling. Re-score every saved
// best (and the on-tread control) also counting feet that are within a foot's
// thickness of the tread (z < h + 0.045), and ticks where BOTH such feet are
// down AND the duck is upright.  cd sim && node ../climb/verify_feet.mjs
import fs from 'node:fs';
import { scoreSaved, STAIR_Y } from '../climb/blockrig.mjs';
const mm = v => (v*1000).toFixed(1);
const rows = [];
const jobs = [
  ['../climb/best_r2_blockclimb_90mm.json', 0.090],
  ['../climb/best_r2_blockclimb_180mm.json', 0.180],
  ['../climb/ablate_noblock_90mm.json', 0.090],
  ['../climb/ablate_noblock_180mm.json', 0.180],
];
for (const [path, h] of jobs) {
  const v = await scoreSaved(path, h, {});
  rows.push({ path, rise_mm: h*1000, honest: v.crit.honest,
    feetOnTreadMax_rig3: v.feetOnTreadMax, feetRestingMax: v.feetRestingMax,
    uprightBothRestingTicks: v.uprightBothRestingTicks,
    scored_feetOnTread: v.scored.feetOnTread, up: v.scored.up,
    x_mm:+mm(v.scored.x), z_mm:+mm(v.scored.z), upFrac: v.upFrac });
  console.log(`${path.split('/').pop().padEnd(36)} ${(h*1000).toFixed(0)}mm  honest=${v.crit.honest}  rig3 feetOnTreadMax=${v.feetOnTreadMax}  RESTING max=${v.feetRestingMax}  upright&bothResting ticks=${v.uprightBothRestingTicks}  upFrac=${(v.upFrac*100).toFixed(1)}%`);
}
fs.writeFileSync('../climb/verify_feet-results.json', JSON.stringify({ generated:new Date().toISOString(), rows }, null, 2));
console.log('WROTE ../climb/verify_feet-results.json');
