// What does one attempt() actually cost on this Pi? The budget for any search
// is this number, not the mj_step figure: attempt() also runs an ONNX forward
// pass every tick (sim/climb_lib.mjs:126).
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/episode-cost.mjs
import fs from 'node:fs';
import { replay } from '../sim/climb_lib.mjs';
import { makeLoop } from '../site/duckloop.mjs';
const C = JSON.parse(fs.readFileSync('duckkit-constants.json','utf8'));
const { HOME } = makeLoop(C);
for (const [lbl, tEnd] of [['2.2 s track', 2.2], ['3.2 s track', 3.2]]) {
  const a = HOME.slice(); a[5] = -1.2; a[6] = 0.6;
  const track = [{ t: tEnd*0.4, pose: a }, { t: tEnd, pose: HOME.slice() }];
  const t0 = Date.now();
  const N = 10;
  for (let i = 0; i < N; i++) await replay(track, { blend: 1.4, approach: 0, gap: 0.03, side: 0 }, 0.09);
  const ms = (Date.now() - t0) / N;
  console.log(`${lbl}: ${ms.toFixed(0)} ms per episode  ->  ${(60000/ms).toFixed(0)} episodes/minute/process`);
}
