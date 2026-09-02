#!/usr/bin/env node
// measure_drift.mjs — WHY THE NAIVE CHASER PASSES THE WRONG COLUMN.
//
// The leaderboard says `ctrl_alpha_walking` clears the whole bearing −20°
// column and MISSES bearing 0 at 0.70 m and 0.95 m. That is the opposite of
// what was predicted in advance ("passes some of the three bearing-0 cells,
// fails every off-bearing cell"), so the card owes the reader a cause rather
// than a shrug.
//
// This script measures it. It re-scores the naive chaser on the four
// bearing-0 cells and records, per cell, the frozen initial heading `yaw0`
// (the vector `ballTravel_mm` projects onto and the same vector Pollen's
// `kick_dir` freezes) and the duck's ROOT POSITION AT THE END of the driven
// span. The angle between the two is the gait's open-loop drift.
//
// It measures ONE control on FOUR cells of the published grid through the same
// `chase_rig.mjs` the leaderboard came from — no new scorer, no new criterion.
//
//   cd ~/projects/duck-sounds/sim && node ../challenge-ball/harness/measure_drift.mjs
//
// Writes results/chase_drift-results.json next to this package.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { scoreSaved, PLANT, PLANT_DIGEST } from '../../chase/chase_rig.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.join(HERE, '..', 'results', 'chase_drift-results.json');
const ENTRANT = path.join(HERE, '..', '..', 'chase', 'ctrl_alpha_walking.json');

const CELLS = [
  { bearing: 0, range: 0.45, drop: 0.120, fmul: 1.0, tier: 'core' },
  { bearing: 0, range: 0.70, drop: 0.120, fmul: 1.0, tier: 'core' },
  { bearing: 0, range: 0.95, drop: 0.120, fmul: 1.0, tier: 'core' },
  { bearing: 0, range: 1.20, drop: 0.120, fmul: 1.0, tier: 'ext' },
];

const DEG = 180 / Math.PI;
const rows = [];
for (const cell of CELLS) {
  const r = await scoreSaved(ENTRANT, cell);
  const [x, y] = r.rootEnd;
  const headingOfPath = Math.atan2(y, x);          // the duck starts at the origin in xy
  rows.push({
    cell,
    yaw0_rad: r.yaw0,
    yaw0_deg: r.yaw0 * DEG,
    kickDir: r.kickDir,
    rootEnd: r.rootEnd,
    pathHeading_deg: headingOfPath * DEG,
    driftFromInitialHeading_deg: (headingOfPath - r.yaw0) * DEG,
    walkedDistance_m: Math.hypot(x, y),
    closest_mm: r.facts.closest_mm,
    ballTravel_mm: r.facts.ballTravel_mm,
    touched: r.facts.touched,
    chased: r.chased,
    hash: r.hash,
  });
}

const out = {
  why: 'The open-loop drift of ctrl_alpha_walking, measured on the four bearing-0 cells of the '
     + 'published grid through chase/chase_rig.mjs. The card cites this to explain why the naive '
     + 'chaser clears the bearing -20 column and misses two of the three dead-ahead cells.',
  entrant: 'ctrl_alpha_walking.json',
  policy: 'alpha_walking.onnx',
  schedule: [[0, { vx: 0.5, vy: 0, vyaw: 0 }]],
  seconds: 4,
  plantName: PLANT,
  plantDigest: PLANT_DIGEST,
  note: 'rootEnd is the duck root at the END of the driven span, in the world frame; the duck '
      + 'starts at the plant origin in xy, so atan2(y, x) is the heading of the whole walk. '
      + 'yaw0 is the heading FROZEN at the first driven tick — the axis ballTravel_mm projects '
      + 'onto. Their difference is the drift. Positive is LEFT, the grid convention.',
  rows,
};
fs.writeFileSync(OUT, JSON.stringify(out, null, 1) + '\n');
for (const r of rows) {
  console.log(`bearing ${r.cell.bearing}  range ${r.cell.range}  drift ${r.driftFromInitialHeading_deg.toFixed(3)} deg  `
            + `walked ${r.walkedDistance_m.toFixed(4)} m  closest ${r.closest_mm.toFixed(2)} mm  chased ${r.chased}`);
}
console.log('wrote', OUT);
