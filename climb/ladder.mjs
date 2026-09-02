// How tall a step does each authored stair motion actually clear?
//
// WHY THIS IS NOT AN HTTP CLIENT. The task asked for this against the bench at
// 100.122.199.6:8770. That bench cannot host a step: `duckbench-core.mjs`
// contains no reference to stairs.js, layoutStairs or the step joints — grep it
// — so every rollout it serves happens on a flat floor with all fourteen step
// blocks parked wherever the model left them. POST /stairs, /scene, /step and
// /world all answer `no ... here`, and POST /reset with a `stairs` body is
// accepted and ignored. Its /perform is also dead in the process that is
// actually listening: the running unit started 2026-09-01 12:09 and reports
// `duck-bench/4`, from before the split that fixed the `expertName is not
// defined` bug; every /perform today returns that 400.
//
// So the step ladder is run IN PROCESS, on the same machine (this is the bench
// — hostname `robot`, tailscale 100.122.199.6), against the same `scene.mjb`
// the bench serves, through the same MuJoCo WASM build in sim/node_modules and
// the same BEST_alpha_stand.onnx. It reuses sim/climb_lib.mjs's `replay`, which
// is the harness sim/stepverify.mjs already used, so these numbers are
// comparable to the ones in sim/climb.log, lever.log, riser.log and search.log.
//
// RUN IT FROM sim/, because climb_lib.mjs reads scene.mjb, duckkit-constants
// and the ONNX by cwd-relative path:
//     cd ~/projects/duck-sounds/sim && node ../climb/ladder.mjs
//
// REPEATS ARE NOT NOISE HERE. climb_lib's attempt() is deterministic: fixed
// spawn, fixed drop of 0.12 m, no push, no friction scaling, and ORT inference
// is deterministic — so running it three times returns the same answer three
// times, which is what stepverify.mjs and climbverify.mjs were doing. The one
// axis this harness lets a caller vary without touching sim/ is where the duck
// starts, so a "repeat" here is a start-position offset of -10, 0 and +10 mm
// on `gap`, the same knob climbverify.mjs perturbed. 3/3 means the move cleared
// that rise from all three starts.
import fs from 'node:fs';
import { replay } from '../sim/climb_lib.mjs';
import { buildTrack } from '../site/intent.mjs';
import { makeLoop } from '../site/duckloop.mjs';

// HOME is the 14-joint home pose with mouth removed — duckloop derives it from
// duckkit-constants.json, which is where every other script in sim/ gets it.
const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const { HOME } = makeLoop(C);
if (!Array.isArray(HOME) || HOME.length !== 14) throw new Error(`HOME is ${HOME && HOME.length}`);

const SITE = '../site';
const MOVES = [
  ['step_up',  `${SITE}/intent-stepup.json`],
  ['lever_up', `${SITE}/intent-lever.json`],
  ['riser_up', `${SITE}/intent-riser.json`],
  ['climb',    `${SITE}/intent-climb.json`],
];

// mm. 10 is where the shipped staging sits (site/intent-specs.js STAGE_STAIRS);
// 170 is the bottom of a code stair riser.
const LADDER = (process.env.LADDER || '10,16,20,24,30,40,55,70,90,120,170')
  .split(',').map(s => +s / 1000);
const OFFSETS = [-0.010, 0, 0.010];

const rows = [];
for (const [name, path] of MOVES) {
  const j = JSON.parse(fs.readFileSync(path, 'utf8'));
  const p = j.params || {};
  // step_up ships searched PARAMETERS, not a track; the page and the recorder
  // both build its track with site/intent.mjs buildTrack, so this does too.
  const track = j.keyframes ?? buildTrack(p, HOME);
  const opts = {
    blend: j.blend ?? p.blend ?? 1,
    approach: j.approach ?? p.approach ?? 0,
    gap: j.gap ?? p.gap ?? 0.06,
    side: j.side ?? 0,
  };
  console.log(`\n${name}  blend=${opts.blend.toFixed(3)} approach=${opts.approach.toFixed(3)} `
            + `gap=${opts.gap.toFixed(3)} side=${(opts.side).toFixed(3)}  (${path})`);
  let best = 0;
  for (const h of LADDER) {
    let ok = 0; const above = [];
    for (const d of OFFSETS) {
      const r = await replay(track, { ...opts, gap: Math.max(0.01, opts.gap + d) }, h);
      if (r.onTop) ok++;
      above.push(r.above);
    }
    const mm = (h * 1000).toFixed(0).padStart(3);
    const av = (Math.max(...above) * 1000).toFixed(0);
    console.log(`  ${mm} mm  ${ok}/3   best trunk-above-tread ${av} mm`);
    rows.push({ move: name, riseMm: +(h * 1000).toFixed(0), cleared: ok, of: 3, bestAboveMm: +av });
    if (ok >= 2) best = h;
    if (ok === 0 && best) break;   // it has already stopped clearing
  }
  console.log(`  => ${name}: ${best ? (best * 1000).toFixed(0) + ' mm' : 'clears nothing on this flight'}`);
}
fs.writeFileSync('../climb/ladder-results.json', JSON.stringify({
  what: 'authored stair motions vs step height, strict criterion from sim/climb_lib.mjs',
  criterion: 'upright, trunk past the first riser, both feet at or above the first tread, '
           + 'trunk 95 mm above that tread, still there one second after the track ends',
  flight: { count: 4, run: 0.28, start: 0.12, note: 'four steps, 280 mm run, flush to the north wall' },
  repeats: 'start-position offsets of -10, 0, +10 mm on gap (attempt() is otherwise deterministic)',
  plant: 'sim/scene.mjb', policy: 'BEST_alpha_stand.onnx',
  when: new Date().toISOString(), rows,
}, null, 1) + '\n');
console.log('\nwrote climb/ladder-results.json');
