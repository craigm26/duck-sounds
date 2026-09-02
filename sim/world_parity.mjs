// THE GATE THAT LICENSES /world.
//
// WHAT IS BEING CLAIMED. The live world can be given a different room — a
// flight of steps, a ball somewhere else, a prop moved — and NOTHING ELSE THIS
// BENCH ANSWERS CHANGES. Every published number in this repo came out of
// /record, /measure, /perform, /capture, /tune, /climb or /chase, and every one
// of those runs in an mjData the live lane never touches. But they all share
// one MODEL, and laying a flight means zeroing the step geoms' conaffinity in
// that shared model. If that loan ever leaked, a /climb cell scored while
// somebody was steering a duck would be scored on a different plant, silently,
// and the number would be believed.
//
// SO IT IS MEASURED, NOT ASSERTED, IN FOUR PHASES:
//
//   1. `placeSteps` lays the stairs challenge's own grid EXACTLY where
//      `layoutStairs` lays it — all twenty-eight qpos slots and all
//      twenty-eight qvel slots, `Object.is`, at full float digits, for every
//      rise in the grid. `placeSteps` is additive and `layoutStairs` is
//      untouched precisely so this comparison can exist; if the two ever
//      disagree, the world a caller describes is not the flight the audit
//      scored.
//
//   2. A four-step 60 mm flight, posted and read back, against a checked-in
//      fixture. The same bytes are the kit's `Fixtures/bench/world.json`, so
//      the Swift reader and this bench are held to one document.
//
//   3. /reset re-lays it. `mj_resetData` takes every slide back to qpos0, which
//      for the bank means the stack it boots in; a world that survived a settle
//      and not a reset would vanish at the start of the first trial. The flight
//      has to still be there, the ball has to be where the world put it and the
//      duck has to be on its feet.
//
//   4. THE ONE THAT LICENSES THE DESIGN. With that world standing, a /climb
//      cell and a /record answer byte-for-byte what they answered with no world
//      set. Then the world is read back again and must be unchanged, because
//      the climb episode parks ITS bank in ITS own mjData and must not have
//      touched this one.
//
//   cd ~/projects/duck-sounds/sim && node world_parity.mjs
//   ... --out parity/world-v1.json      to (re)capture the phase 2 fixture
//   ... --against parity/world-v1.json  (the default) to hold it to one
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { placeSteps, layoutStairs, STAIR_COUNT, STEP_HALF_DEPTH } from './stairs.js';
import { gridCells, STAIR_RUN, STAIR_START, DEFAULT_STEP_COUNT } from './climb_score.mjs';
import { gridCells as chaseGridCells } from './chase_score.mjs';
import { nodeBench } from './duckbench-node.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (name, fallback) => {
  const i = args.indexOf(name);
  return i < 0 ? fallback : args[i + 1];
};
const outFile = opt('--out', null);
const against = opt('--against', 'parity/world-v1.json');

let failures = 0;
const ok = (label, good, detail = '') => {
  console.log(`${good ? '  ok  ' : '  FAIL'} ${label}${detail ? ' — ' + detail : ''}`);
  if (!good) failures++;
};

/** Deterministic JSON: keys sorted, so a reordered object is not a diff. */
function canon(value) {
  if (Array.isArray(value)) return value.map(canon);
  if (value && typeof value === 'object') {
    const out = {};
    for (const k of Object.keys(value).sort()) out[k] = canon(value[k]);
    return out;
  }
  return value;
}
/** Every leaf of a canonical answer, as pointer -> value. */
function leaves(value, at, into) {
  if (Array.isArray(value)) { value.forEach((v, i) => leaves(v, `${at}/${i}`, into)); return into; }
  if (value && typeof value === 'object') {
    for (const k of Object.keys(value)) leaves(value[k], `${at}/${k}`, into);
    return into;
  }
  into.set(at, value);
  return into;
}
function diffLeaves(a, b) {
  const L = leaves(canon(a), '', new Map()), R = leaves(canon(b), '', new Map());
  const diffs = [];
  for (const key of new Set([...L.keys(), ...R.keys()])) {
    const l = L.has(key) ? L.get(key) : '<<absent>>';
    const r = R.has(key) ? R.get(key) : '<<absent>>';
    if (!Object.is(l, r)) diffs.push(`${key || '/'}: ${JSON.stringify(l)} -> ${JSON.stringify(r)}`);
  }
  return diffs;
}

// ---------------------------------------------------------------------------
// PHASE 1 — placeSteps IS layoutStairs on the grid's own configurations.
//
// NO MUJOCO IN THIS PHASE ON PURPOSE. Both functions do nothing but write into
// `data.qpos` and `data.qvel` at addresses they are handed, so a pair of plain
// Float64Arrays and a fabricated address table is the whole plant they need —
// and comparing arrays a MuJoCo build never touched leaves no room for the
// engine to be the thing that agreed.
// ---------------------------------------------------------------------------
console.log('PHASE 1 — placeSteps vs layoutStairs, 28 qpos and 28 qvel slots, Object.is');
{
  const addr = Array.from({ length: STAIR_COUNT }, (_, i) => ({ x: i * 2, z: i * 2 + 1,
                                                                dx: i * 2, dz: i * 2 + 1 }));
  const fresh = () => ({ qpos: new Float64Array(STAIR_COUNT * 2).fill(NaN),
                         qvel: new Float64Array(STAIR_COUNT * 2).fill(NaN) });
  // Every rise the stairs challenge is scored at, at the grid's own run, start
  // and step count — plus the counts either side, because a flight of one and a
  // flight of fourteen park a different number of blocks.
  const rises = [...new Set(gridCells().map(c => c.rise))].sort((a, b) => a - b);
  let compared = 0, cells = 0;
  for (const rise of rises) {
    for (const count of [1, 2, DEFAULT_STEP_COUNT, 8, STAIR_COUNT]) {
      const cfg = { count, rise, run: STAIR_RUN, start: STAIR_START };
      const A = fresh(), B = fresh();
      const nA = layoutStairs(A, addr, cfg);
      // The blocks the same flight is made of, said the way a world describes
      // one: a centre and a tread height.
      const blocks = Array.from({ length: count }, (_, i) => ({
        x: STAIR_START + i * STAIR_RUN + STEP_HALF_DEPTH,
        top: (i + 1) * rise,
      }));
      const nB = placeSteps(B, addr, blocks);
      cells++;
      if (nA !== nB) { ok(`rise ${rise} count ${count}`, false, `laid ${nA} vs ${nB}`); continue; }
      const bad = [];
      for (let k = 0; k < STAIR_COUNT * 2; k++) {
        compared += 2;
        if (!Object.is(A.qpos[k], B.qpos[k])) bad.push(`qpos[${k}] ${A.qpos[k]} vs ${B.qpos[k]}`);
        if (!Object.is(A.qvel[k], B.qvel[k])) bad.push(`qvel[${k}] ${A.qvel[k]} vs ${B.qvel[k]}`);
      }
      if (bad.length) ok(`rise ${rise} count ${count}`, false, bad.slice(0, 3).join('; '));
    }
  }
  ok(`${cells} configurations, ${compared} slots identical`, failures === 0);
}

// ---------------------------------------------------------------------------
const bench = await nodeBench();
const { handle } = bench;
const call = async (p, b) => {
  const url = new URL(p, 'http://bench.local');
  const value = await handle(url, b ?? {});
  if (value === undefined) throw new Error(`no ${p} here`);
  return JSON.parse(JSON.stringify(value));
};
const refused = async (p, b) => {
  try { await call(p, b); return null; } catch (e) { return String(e.message || e); }
};

// THE WORLD THIS GATE IS ABOUT: the stairs challenge's own four-step 60 mm
// flight, at the grid's run and start, with the ball put in front of the duck.
const RISE = 0.060;
const FLIGHT = Array.from({ length: DEFAULT_STEP_COUNT }, (_, i) => ({
  x: STAIR_START + i * STAIR_RUN + STEP_HALF_DEPTH,
  top: (i + 1) * RISE,
}));
// AND IT IS DESCRIBED THE WAY A SCENE DESCRIBES IT, not the way the bank can
// express it. `DuckScene.staircase(count: 4, rise: 0.06, run: 0.28, start: 0.12)`
// gives every step a y of 0 and a half-height that grows with the flight; this
// bank has neither, and the fixture is worth having precisely because it is the
// document where those five gaps are written down. A broom and a fifth wall are
// asked for too, because a plant that has neither should say which bodies it
// does have rather than failing.
const WORLD_BODY = {
  name: 'Stairs challenge, 60 mm',
  steps: FLIGHT.map((s, i) => ({ ...s, y: 0, halfHeight: Math.max(0.10, (i + 1) * RISE) })),
  ball: { x: 0.8, y: 0 },
  props: [{ name: 'block_a', x: 0.45, y: -0.30 }, { name: 'broom', x: 1.0, y: 0.5 }],
  walls: [{ name: 'partition', x: 0, y: 0.6 }],
};

// ---------------------------------------------------------------------------
// PHASE 4's BASELINE, TAKEN FIRST, BEFORE ANY WORLD EXISTS.
// ---------------------------------------------------------------------------
console.log('PHASE 4a — /climb and /record with NO world, the baseline');
// The record, on the grid's first cell — the same file and the same cell
// `sim/climb_parity.mjs` scores it on, so a leak here would show up as a
// disagreement with a number the audit published rather than as a private one.
const CLIMB_INTENT = JSON.parse(
  fs.readFileSync(path.resolve(HERE, '../climb/best_r6_ceilvaultC_60mm.json'), 'utf8'));
const CELL0 = gridCells()[0];
const climbBody = { intent: CLIMB_INTENT, rise: RISE, tail: 'policy',
                    cell: { dh: CELL0.dh, drop: CELL0.drop, fmul: CELL0.fmul } };
const recordBody = { policy: 'alpha_walking.onnx', seconds: 1, schedule: [[0, { vx: 0.2 }]] };
// AND A /chase CELL, because the ball challenge borrows the SAME shared model
// for the same reason — its plant axis multiplies the foot geoms' friction —
// and shares `climbLane` with the stairs challenge. A gate that only asked the
// staircase would be asking the one that already knew about the step bank.
const CHASE_ENTRANT = JSON.parse(
  fs.readFileSync(path.resolve(HERE, '../chase/ctrl_ball_kick_right.json'), 'utf8'));
const CHASE_CELL0 = chaseGridCells()[0];
const chaseBody = { entrant: CHASE_ENTRANT, tail: 'policy',
                    cell: { bearing: CHASE_CELL0.bearing, range: CHASE_CELL0.range,
                            drop: CHASE_CELL0.drop, fmul: CHASE_CELL0.fmul } };
// WALL CLOCK IS NOT A TRAJECTORY. `/climb` and `/record` time themselves, so
// two identical runs report two different durations; they are named here, in
// the file, rather than tolerated by a fuzzy comparison — everything else is
// Object.is at full float digits.
const TIMING = ['/seconds', '/elapsedSeconds', '/tookSeconds', '/ms'];
const stripTiming = d => d.filter(line => !TIMING.some(t => line.startsWith(t + ':')));
const climb0 = await call('/climb', climbBody);
const chase0 = await call('/chase', chaseBody);
const record0 = await call('/record', recordBody);
if (climb0.error) throw new Error(`/climb refused the baseline: ${climb0.error}`);
ok('/climb answered', climb0.invalid === false,
   `move ${climb0.move}, honest ${climb0.honest}, stable ${climb0.stable}, above ${climb0.above_mm} mm`);
if (chase0.error) throw new Error(`/chase refused the baseline: ${chase0.error}`);
ok('/chase answered', chase0.chased !== undefined || chase0.verdict !== undefined,
   `entrant ${chase0.entrant ?? chase0.hash?.slice(0, 12) ?? '?'}`);
ok('/record answered', !!record0 && !record0.error);
console.log(`         allowed to differ (wall clock only): ${TIMING.join(', ')}`);

// ---------------------------------------------------------------------------
// PHASE 2 — the readback, against the fixture.
// ---------------------------------------------------------------------------
console.log('PHASE 2 — POST /world, then GET /world, against the fixture');
const posted = await call('/world', WORLD_BODY);
const readback = await call('/world');
ok('POST and GET agree', diffLeaves(posted, readback).length === 0,
   diffLeaves(posted, readback).slice(0, 3).join('; '));
ok('four steps standing', readback.steps.length === DEFAULT_STEP_COUNT,
   `${readback.steps.length} steps`);
ok('ten blocks parked', readback.bank.parked === STAIR_COUNT - DEFAULT_STEP_COUNT,
   `${readback.bank.parked} parked`);
ok('the tread heights are the flight\'s',
   readback.steps.every((s, i) => s.top === Math.round((i + 1) * RISE * 10000) / 10000),
   readback.steps.map(s => s.top).join(', '));
ok('y is the bank\'s and nothing else', readback.steps.every(s => s.y === readback.bank.y));
// The five things this plant could not express, counted rather than described:
// three half-heights, four y's, one broom and one wall.
ok('every gap is named', readback.unexpressed.length === 9,
   readback.unexpressed.map(u => `${u.what}${u.index === undefined ? '' : ' ' + u.index}`).join(', '));
ok('block_a moved where it was asked to',
   readback.props.some(p => p.name === 'block_a' && p.at[0] === 0.45 && p.at[1] === -0.3),
   JSON.stringify(readback.props.find(p => p.name === 'block_a')));

if (outFile) {
  fs.writeFileSync(path.resolve(HERE, outFile), JSON.stringify(canon(readback), null, 1) + '\n');
  console.log(`  wrote ${outFile}`);
}
if (against) {
  const file = path.resolve(HERE, against);
  if (!fs.existsSync(file)) ok(`fixture ${against}`, false, 'not there — run with --out to capture it');
  else {
    const diffs = diffLeaves(JSON.parse(fs.readFileSync(file, 'utf8')), readback);
    ok(`readback matches ${against}`, diffs.length === 0,
       diffs.slice(0, 6).join(' | ') + (diffs.length > 6 ? ` | +${diffs.length - 6} more` : ''));
  }
}

// The refusals, on the same standing bench, because a door that never says no
// is not a door.
console.log('PHASE 2b — the refusals');
ok('fifteen steps refused',
   /bank of 14/.test(await refused('/world', { steps: [...FLIGHT, ...FLIGHT, ...FLIGHT, ...FLIGHT] }) || ''));
ok('a fifth step at this run refused (it crosses wall_e)',
   /wall_e/.test(await refused('/world', {
     steps: Array.from({ length: 5 }, (_, i) => ({ x: STAIR_START + i * STAIR_RUN + STEP_HALF_DEPTH,
                                                   top: (i + 1) * RISE })) }) || ''));
ok('a ball outside the arena refused',
   /outside this arena/.test(await refused('/world', { ball: { x: 2.0, y: 0 } }) || ''));
ok('an unknown duck refused',
   /unknown duck/.test(await refused('/world', { duck: 'goose', steps: [] }) || ''));
{
  // A world is still the one that was standing before the refusals.
  const after = await call('/world');
  ok('a refused world changed nothing', diffLeaves(readback, after).length === 0,
     diffLeaves(readback, after).slice(0, 3).join('; '));
}

// ---------------------------------------------------------------------------
// PHASE 3 — /reset re-lays it.
// ---------------------------------------------------------------------------
console.log('PHASE 3 — POST /reset, then GET /world');
// Walk the duck first, so the reset has something to undo — and check the
// flight survived the WALK, not only the reset. `stepLive` writes the blocks
// and then steps, so a readback taken after a tick sees them two millimetres
// below where they were written; the parked ones drift the same way. A
// readback that could not tell that from a moved world would report ten new
// steps every time somebody pressed forward.
await call('/intent', { vx: 0.3, hold: 0.6 });
{
  const driven = await call('/world');
  ok('the flight survived a second of driving',
     diffLeaves(readback.steps, driven.steps).length === 0
     && driven.bank.parked === STAIR_COUNT - DEFAULT_STEP_COUNT,
     `${driven.steps.length} steps, ${driven.bank.parked} parked`);
}
const afterReset = await call('/reset', {});
const world3 = await call('/world');
ok('the flight is still there', world3.steps.length === DEFAULT_STEP_COUNT,
   `${world3.steps.length} steps`);
ok('the flight is where the world put it',
   diffLeaves(readback.steps, world3.steps).length === 0,
   diffLeaves(readback.steps, world3.steps).slice(0, 3).join('; '));
ok('the ball is where the world put it',
   world3.ball.x === WORLD_BODY.ball.x && world3.ball.y === WORLD_BODY.ball.y,
   `(${world3.ball.x}, ${world3.ball.y})`);
ok('the world kept its name', world3.world.set === true
   && world3.world.name === WORLD_BODY.name, JSON.stringify(world3.world));
ok('the prop is still where the world put it',
   world3.props.some(p => p.name === 'block_a' && p.at[0] === 0.45 && p.at[1] === -0.3),
   JSON.stringify(world3.props.find(p => p.name === 'block_a')));
ok('and the gaps are still named', world3.unexpressed.length === 9,
   `${world3.unexpressed.length} entries`);
ok('the duck is on its feet', afterReset.upright === true,
   `height ${afterReset.height}, upright ${afterReset.upright}`);

// ---------------------------------------------------------------------------
// PHASE 4 — the isolation cannot leak.
// ---------------------------------------------------------------------------
console.log('PHASE 4 — /climb and /record WITH that world standing');
const climb1 = await call('/climb', climbBody);
const chase1 = await call('/chase', chaseBody);
const record1 = await call('/record', recordBody);
{
  const d = stripTiming(diffLeaves(climb0, climb1));
  ok('/climb identical with a world standing', d.length === 0,
     d.slice(0, 6).join(' | ') + (d.length > 6 ? ` | +${d.length - 6} more` : ''));
  const leafCount = leaves(canon(climb0), '', new Map()).size;
  console.log(`         ${leafCount} leaves compared`);
}
{
  const d = stripTiming(diffLeaves(chase0, chase1));
  ok('/chase identical with a world standing', d.length === 0,
     d.slice(0, 6).join(' | ') + (d.length > 6 ? ` | +${d.length - 6} more` : ''));
  console.log(`         ${leaves(canon(chase0), '', new Map()).size} leaves compared`);
}
{
  const d = stripTiming(diffLeaves(record0, record1));
  ok('/record identical with a world standing', d.length === 0,
     d.slice(0, 6).join(' | ') + (d.length > 6 ? ` | +${d.length - 6} more` : ''));
  const leafCount = leaves(canon(record0), '', new Map()).size;
  console.log(`         ${leafCount} leaves compared`);
}
{
  // AND THE CLIMB DID NOT TAKE THE WORLD WITH IT. Its episode parks a bank in
  // its `finally`; if that were the live world's bank, the flight would be gone.
  const world4 = await call('/world');
  const d = diffLeaves(world3.steps, world4.steps);
  ok('the flight survived a /climb and a /record', d.length === 0, d.slice(0, 3).join('; '));
}
{
  // And once more with a BARE FLOOR standing, which is the other end of the
  // range: the bank parked by the live lane rather than laid.
  await call('/world', { name: 'Bare floor', clear: true });
  const climb2 = await call('/climb', climbBody);
  const chase2 = await call('/chase', chaseBody);
  const record2 = await call('/record', recordBody);
  ok('/climb identical with a bare floor standing', stripTiming(diffLeaves(climb0, climb2)).length === 0,
     stripTiming(diffLeaves(climb0, climb2)).slice(0, 4).join(' | '));
  ok('/chase identical with a bare floor standing', stripTiming(diffLeaves(chase0, chase2)).length === 0,
     stripTiming(diffLeaves(chase0, chase2)).slice(0, 4).join(' | '));
  ok('/record identical with a bare floor standing', stripTiming(diffLeaves(record0, record2)).length === 0,
     stripTiming(diffLeaves(record0, record2)).slice(0, 4).join(' | '));
  const bare = await call('/world');
  ok('a bare floor is fourteen parked blocks', bare.steps.length === 0
     && bare.bank.parked === STAIR_COUNT, `${bare.steps.length} standing, ${bare.bank.parked} parked`);
}

console.log(failures ? `WORLD PARITY FAILED: ${failures} check(s)` : 'WORLD PARITY OK');
process.exitCode = failures ? 1 : 0;
