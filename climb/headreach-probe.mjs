// PRECONDITION for every head-anchored strategy: can the head reach the step
// AT ALL from where attempt() starts the duck?
//
// attempt() places the trunk at x = 0.12 - 0.07 - gap (sim/climb_lib.mjs:113),
// so gap alone decides the start. climb/reach-max.mjs measures the head's
// maximum forward reach as 114 mm (trunk level) to 129 mm (trunk pitched 45-60
// deg nose-down) ahead of the trunk origin. The riser face is at x = 0.120.
// Arithmetic says the authored gaps start the duck out of range. This is the
// dynamic version: reach the head out and hold it, do nothing else, and print
// where the criterion says the trunk ended up.
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/headreach-probe.mjs
import fs from 'node:fs';
import { replay } from '../sim/climb_lib.mjs';
import { makeLoop } from '../site/duckloop.mjs';
const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const { HOME } = makeLoop(C);
const J = { np: 5, hp: 6 };

// two keyframes: reach out, then hold. nothing else moves.
function reachTrack(np, hp) {
  const a = HOME.slice(); a[J.np] = np; a[J.hp] = hp;
  return [{ t: 0.6, pose: a }, { t: 2.2, pose: a.slice() }];
}
const GAPS = [0.02, 0.04, 0.06, 0.0481, 0.1266];
const RISES = [0.020, 0.040, 0.060, 0.090, 0.120, 0.180];
console.log('head fully extended (neck_pitch -1.50, head_pitch 0.80), blend 1.6, approach 0');
console.log('start trunk x = 0.12 - 0.07 - gap.  riser face is at x = 0.120 m.\n');
console.log('  gap    startX   rise |  trunkX  trunkZ   above  feetUp upright onTop');
const rows = [];
for (const gap of GAPS) {
  const startX = 0.12 - 0.07 - gap;
  for (const h of RISES) {
    const r = await replay(reachTrack(-1.50, 0.80), { blend: 1.6, approach: 0, gap, side: 0 }, h);
    console.log(`${gap.toFixed(4)}  ${(startX*1000).toFixed(0).padStart(6)}mm ${(h*1000).toFixed(0).padStart(5)}mm |`
      + `${(r.x*1000).toFixed(0).padStart(7)}mm${(r.z*1000).toFixed(0).padStart(8)}mm`
      + `${(r.above*1000).toFixed(0).padStart(7)}mm    ${r.feetUp}    ${String(r.up).padEnd(6)} ${r.onTop}`);
    rows.push({ gap, startXmm: +(startX*1000).toFixed(1), riseMm: +(h*1000).toFixed(0),
      trunkXmm: +(r.x*1000).toFixed(1), trunkZmm: +(r.z*1000).toFixed(1),
      aboveMm: +(r.above*1000).toFixed(1), feetUp: r.feetUp, upright: r.up, onTop: r.onTop });
  }
}
fs.writeFileSync('../climb/headreach-results.json', JSON.stringify({
  what: 'can the head reach the step from attempt()\'s start? head extended and held, nothing else moves',
  criterion: 'sim/climb_lib.mjs:150', plant: 'sim/scene.mjb', policy: 'BEST_alpha_stand.onnx',
  startX: 'sim/climb_lib.mjs:113 — qpos.x = 0.12 - 0.07 - gap', when: new Date().toISOString(), rows,
}, null, 1) + '\n');
console.log('\nwrote climb/headreach-results.json');
