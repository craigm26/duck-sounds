// What the four step moves actually do, measured against the staircase the
// page actually renders: four steps, 280 mm run, flush to the wall. Every
// number previously printed on the page was measured against a lone 450 mm
// block on an open floor, which is a podium and not a stair, so none of them
// carried over. This replays the SHIPPED keyframes, so the number and the
// thing a visitor presses are the same move.
import fs from 'node:fs';
import { replay } from './climb_lib.mjs';
const LADDER = [0.010, 0.016, 0.024, 0.034, 0.046, 0.060, 0.080];
const MOVES = [
  ['step_up',  '../site/intent-stepup.json'],
  ['lever_up', '../site/intent-lever.json'],
  ['riser_up', '../site/intent-riser.json'],
  ['climb',    '../site/intent-climb.json'],
];
console.log('strict: both feet up, standing height above the tread, upright, still there a second later');
for (const [name, path] of MOVES) {
  const m = JSON.parse(fs.readFileSync(path, 'utf8'));
  const opts = { blend: m.blend, approach: m.approach ?? 0, gap: m.gap ?? 0.06, side: m.side ?? 0 };
  let best = 0, bestOk = 0;
  for (const h of LADDER) {
    let ok = 0;
    for (let i = 0; i < 3; i++) if ((await replay(m.keyframes, opts, h)).onTop) ok++;
    if (ok >= 2) { best = h; bestOk = ok; } else break;
  }
  console.log(`  ${name.padEnd(9)} ${best ? (best*1000).toFixed(0)+' mm  '+bestOk+'/3' : 'clears nothing'}`);
}
