// famB_seam.mjs — the SEAM CHECK. No search, no physics loop: it asks one
// question of rig3's own poseAt(), the function the episode uses to turn a
// track into a commanded pose.
//
// CLAIM: for every tick of beat 1's episode, the concatenated track commands
// EXACTLY the pose beat 1's track commands. If that holds, the state rig3
// reports as beat 1's `terminal` is the state the concatenated move is in when
// beat 2 begins, and the handoff spawn is a faithful reproduction of it rather
// than a convenient fiction.
import fs from 'node:fs';
import { poseAt } from '../climb/rig3.mjs';
const DT = 1 / 50;
let allOK = true;
for (const hm of [90, 80, 120]) {
  const b1 = JSON.parse(fs.readFileSync(`../climb/best_r4_famB_beat1_${hm}mm.json`, 'utf8'));
  const cc = JSON.parse(fs.readFileSync(`../climb/best_r4_famB_concat_${hm}mm.json`, 'utf8'));
  const T0 = b1.keyframes[b1.keyframes.length - 1].t;
  const total = T0 + 0.8;                       // rig3: total = last.t + 0.8
  let worst = 0, ticks = 0;
  for (let t = 0; t * DT < total; t++) {
    const a = poseAt(b1.keyframes, t * DT), b = poseAt(cc.keyframes, t * DT);
    for (let k = 0; k < 14; k++) worst = Math.max(worst, Math.abs(a[k] - b[k]));
    ticks++;
  }
  // and the handoff instant itself
  const a = poseAt(b1.keyframes, total), b = poseAt(cc.keyframes, total);
  let seam = 0; for (let k = 0; k < 14; k++) seam = Math.max(seam, Math.abs(a[k] - b[k]));
  const ok = worst === 0 && seam === 0;
  if (!ok) allOK = false;
  console.log(`rise ${hm} mm: beat-1 track ends t=${T0.toFixed(4)}s, episode total=${total.toFixed(4)}s, ${ticks} commanded ticks; ` +
    `max |concat pose - beat1 pose| over those ticks = ${worst}; at the seam instant = ${seam}; IDENTICAL=${ok}`);
}
console.log('seam identical at every commanded tick of beat 1, all three rises:', allOK);
