// WHY a stair move fails, not just that it does.
//
// ladder.mjs prints a pass/fail. `onTop` in sim/climb_lib.mjs is a conjunction
// of four things — upright, trunk past the first riser, both feet at or above
// the first tread, trunk 95 mm above that tread — and which of the four is
// false is the whole diagnosis: a duck that toppled and a duck that stayed
// neatly standing on the floor in front of the step are the same "no".
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/why.mjs
import fs from 'node:fs';
import { replay } from '../sim/climb_lib.mjs';
import { buildTrack } from '../site/intent.mjs';
import { makeLoop } from '../site/duckloop.mjs';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const { HOME } = makeLoop(C);
const MOVES = [
  ['step_up',  '../site/intent-stepup.json'],
  ['lever_up', '../site/intent-lever.json'],
  ['riser_up', '../site/intent-riser.json'],
  ['climb',    '../site/intent-climb.json'],
];
const HEIGHTS = (process.env.HEIGHTS || '2,10,40,90,170').split(',').map(s => +s / 1000);

console.log('move       rise   onTop  upright  trunkX   trunkZ  above  feetUp   verdict');
for (const [name, path] of MOVES) {
  const j = JSON.parse(fs.readFileSync(path, 'utf8'));
  const p = j.params || {};
  const track = j.keyframes ?? buildTrack(p, HOME);
  const opts = { blend: j.blend ?? p.blend ?? 1, approach: j.approach ?? p.approach ?? 0,
                 gap: j.gap ?? p.gap ?? 0.06, side: j.side ?? 0 };
  for (const h of HEIGHTS) {
    const r = await replay(track, opts, h);
    // The first false conjunct, in the order that matters: a duck on its back
    // is not "short of the riser", it is on its back.
    const verdict = !r.up ? 'toppled'
      : r.x <= 0.12 ? 'upright, still short of the riser face'
      : r.feetUp < 2 ? `upright and past the face but ${r.feetUp}/2 feet on the tread`
      : (r.z - h) <= 0.095 ? 'feet up but crouched below standing height'
      : 'on the step';
    console.log(`${name.padEnd(9)} ${(h * 1000).toFixed(0).padStart(4)} mm  `
      + `${String(r.onTop).padEnd(6)} ${String(r.up).padEnd(7)}  `
      + `${(r.x * 1000).toFixed(0).padStart(5)}mm ${(r.z * 1000).toFixed(0).padStart(6)}mm `
      + `${(r.above * 1000).toFixed(0).padStart(5)}mm  ${r.feetUp}      ${verdict}`);
  }
}
