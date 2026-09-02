// IS THE APP'S "measured at 10 mm" REPRODUCIBLE TODAY?
//
// DuckScene.swift:341 pins `measuredStepCeiling = 0.010` and the Unreachable
// line quotes it. Its stated source is site/intent-riser.json's note: "10 mm
// standing on a real stair, 3/3 from these keyframes". climb/ladder.mjs
// (2026-09-02) got riser_up 0/3 at 10 mm from the same keyframes. One of the
// two is wrong and the app's only stair sentence rests on it.
//
// This runs the EXACT configuration the note claims — the json's own gap and
// blend, no start offset — through sim/climb_lib.mjs's replay, and then walks
// the gap out either side to see whether 10 mm clears anywhere at all.
//
// RUN FROM sim/ (climb_lib reads scene.mjb and the ONNX by cwd-relative path):
//     cd ~/projects/duck-sounds/sim && node ../climb/ceiling.mjs
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
// mm of start offset applied to `gap`, either side of the json's own value.
const OFFSETS = [-0.030, -0.020, -0.010, -0.005, 0, 0.005, 0.010, 0.020, 0.030];
const RISE = 0.010;   // the number DuckScene.swift prints

const rows = [];
for (const [name, path] of MOVES) {
  const j = JSON.parse(fs.readFileSync(path, 'utf8'));
  const p = j.params || {};
  const track = j.keyframes ?? buildTrack(p, HOME);
  const base = { blend: j.blend ?? p.blend ?? 1, approach: j.approach ?? p.approach ?? 0,
                 gap: j.gap ?? p.gap ?? 0.06, side: j.side ?? 0 };
  let ok = 0; const detail = [];
  for (const d of OFFSETS) {
    const r = await replay(track, { ...base, gap: Math.max(0.01, base.gap + d) }, RISE);
    if (r.onTop) ok++;
    detail.push({ offsetMm: +(d * 1000).toFixed(0), onTop: r.onTop,
                  aboveMm: +(r.above * 1000).toFixed(0), feetUp: r.feetUp, upright: r.up });
  }
  console.log(`${name.padEnd(9)} at 10 mm: ${ok}/${OFFSETS.length}   `
            + detail.map(d => `${d.offsetMm >= 0 ? '+' : ''}${d.offsetMm}:${d.onTop ? 'Y' : 'n'}`).join(' '));
  rows.push({ move: name, riseMm: 10, cleared: ok, of: OFFSETS.length, gap: base.gap, detail });
}
fs.writeFileSync('../climb/ceiling-results.json', JSON.stringify({
  question: 'does anything clear the 10 mm rise DuckScene.measuredStepCeiling asserts?',
  criterion: 'sim/climb_lib.mjs:150 — upright, x > 0.12 m past the first riser, both feet at or '
           + 'above the first tread, trunk > 95 mm above that tread',
  flight: { count: 4, rise: 0.010, run: 0.28, start: 0.12 },
  varied: 'start offset on `gap`, -30..+30 mm around each json\'s own value',
  plant: 'sim/scene.mjb', policy: 'BEST_alpha_stand.onnx',
  when: new Date().toISOString(), rows,
}, null, 1) + '\n');
console.log('\nwrote climb/ceiling-results.json');
