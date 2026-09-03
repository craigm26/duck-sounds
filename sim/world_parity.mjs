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
// SO IT IS MEASURED, NOT ASSERTED, IN FIVE PHASES:
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
//   5. /perform — the endpoint the paragraph above names and the one phases 1
//      to 4 never called. It can now CARRY a room of its own into the mjData
//      /record, /measure, /capture and /tune also step, so it is both the
//      biggest hole in this gate and the biggest new surface. Twelve claims,
//      5a to 5l: a request that carries no world answers exactly what it always
//      answered; a request that carries one lays it where a POST /world lays
//      it, leaf for leaf; the loan is per tick and is handed back; the module
//      global is never written; and — the thing phases 1 to 4 structurally
//      cannot see, because every call in them is awaited before the next is
//      issued — an overlapping request moves no number, PROVED to have actually
//      overlapped by a counter the bench keeps.
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
  //
  // THIS AXIS WAS `undefined` AND THE WHOLE HEIGHT COMPARISON WAS VACUOUS.
  // It read `[...new Set(gridCells().map(c => c.rise))]`, and `gridCells()`
  // returns `{dh, drop, fmul, tier}` — there is no `rise` key on it. So `rises`
  // was `[undefined]`, `layoutStairs` wrote `NaN` into all twenty-eight z slots
  // of both arrays, `Object.is(NaN, NaN)` is true, and every tread-height
  // comparison in the gate that LICENSES `placeSteps` as `layoutStairs` passed
  // by agreeing about nothing. The x slots were the only thing ever proved.
  // The run printed `5 configurations, 280 slots identical`; it now prints
  // 200 and 11,200.
  //
  // THE AXIS IS THE ONE THE HARNESS ACTUALLY COMPUTES: the eight rises the
  // challenge offers, each crossed with the five `dh` the grid runs, added the
  // way the bench adds them (`rise + dh`, `duckbench-core.mjs`'s /climb route).
  //
  // NO `Set` DEDUPE. Deduping doubles is meaningless here — `0.060 - 0.010` is
  // 0.049999999999999996 and `0.050 + 0` is 0.05, two different doubles the
  // harness really does compute and each of which has to be laid identically by
  // both functions — and a dedupe would hide the values under test. No sort
  // either: the order is the order the grid runs them in.
  const rises = [0.040, 0.050, 0.060, 0.070, 0.080, 0.090, 0.120, 0.180]
    .flatMap(r => [-0.010, -0.005, 0, 0.005, 0.010].map(dh => r + dh));
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
// AND A /perform, WHICH IS THE ENDPOINT THIS FILE'S FIRST PARAGRAPH NAMES AND
// THE ONE IT HAS NEVER CALLED. The body is `bench_parity` entry #53's, so the
// answer under test here is the answer that fixture pins.
const HOME14 = [0.0, -0.0873, -0.4579, -0.0049, 0.453, 0.3491, 0.3491, 0.0, 0.0,
                0.0, 0.0873, 0.4579, 0.0049, -0.453];
const NOD = HOME14.map((v, i) => (i === 5 || i === 6 ? v - 0.25 : v));
const TRACK = [{ at: 0.4, pose: NOD }, { at: 0.8, pose: HOME14 }];
const performBody = { track: TRACK, rollouts: 2, seconds: 1 };
const climb0 = await call('/climb', climbBody);
const chase0 = await call('/chase', chaseBody);
const record0 = await call('/record', recordBody);
const perform0 = await call('/perform', performBody);
if (climb0.error) throw new Error(`/climb refused the baseline: ${climb0.error}`);
ok('/climb answered', climb0.invalid === false,
   `move ${climb0.move}, honest ${climb0.honest}, stable ${climb0.stable}, above ${climb0.above_mm} mm`);
if (chase0.error) throw new Error(`/chase refused the baseline: ${chase0.error}`);
ok('/chase answered', chase0.chased !== undefined || chase0.verdict !== undefined,
   `entrant ${chase0.entrant ?? chase0.hash?.slice(0, 12) ?? '?'}`);
ok('/record answered', !!record0 && !record0.error);
ok('/perform answered', !!perform0 && !perform0.error,
   `${perform0.achieves} of ${perform0.rollouts}, ${perform0.frames?.length} frames`);
ok('a /perform that asked for nothing has no `stood` key', perform0.stood === undefined);
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

// ---------------------------------------------------------------------------
// PHASE 5 — /perform, THE ENDPOINT THE LICENCE WAS WRITTEN FOR.
//
// Phase 4's opening paragraph names /perform among the endpoints whose numbers
// must not move, and phase 4 has never called it. It is also the endpoint that
// can now CARRY a world of its own, into an mjData /record, /measure, /capture
// and /tune also step — so it is both the biggest hole in this gate and the
// biggest new surface.
//
// 5g WAS WATCHED GO RED ON 2026-09-02, on purpose. The defect inserted was one
// `await new Promise(r => setImmediate(r));` between `placeSteps` and
// `stepWorld` inside the collision loan, with `tick` awaiting the stepper so
// the loan really did span it — the exact mistake `stepLive`'s header has
// warned about since the live lane was written.
//
//   WITH THE await IN   FAIL 5g a /intent overlapping a /perform moved no leaf
//                       of it — /achieves: 8 -> 7 | /endHeight: 0.176 -> 0.1085
//                       | /endsUpright: true -> false
//                       | /frames/0/0: -0.0017 -> 0.3543
//                       direction 1 overlap counter +7000 of 7000 perform ticks
//                       direction 2 overlap counter  +875 of 7000
//                       WORLD PARITY FAILED: 1 check
//   WITH IT REMOVED     ok, 0 diffs, five times each direction
//                       direction 1 overlap counter +7000 of 7000 perform ticks
//                       direction 2 overlap counter  +875 of 7000
//                       WORLD PARITY OK
//
// IT TOOK TWO GOES TO BUILD ONE THAT COULD FAIL, and that is worth recording:
// the first shape — a duck spawned BESIDE the flight, one 0.4 s /intent hold,
// twenty overlapped ticks of fourteen hundred — passed WITH the defect in. A
// duck standing beside the flight barely feels whether the blocks are isolated.
// A gate nobody has watched fail is not a gate; this file has shipped a vacuous
// one already (phase 1's `undefined` rise axis).
// ---------------------------------------------------------------------------
console.log('PHASE 5 — POST /perform with a world and a spawn');
const r4 = v => Math.round(v * 10000) / 10000;
/** Worked example A: the harness's own four-step 60 mm flight, in ROOM coordinates. */
const EXAMPLE_A = {
  name: 'Stairs challenge, 60 mm',
  steps: FLIGHT,
  props: [],
  walls: [{ name: 'a wall at (0.00, 1.50) m' }],
};
const SPAWN_A = { x: 0.05, y: 1.305 };
const leafCount = v => leaves(canon(v), '', new Map()).size;

// --- 5a — the leak test /perform has never had -----------------------------
{
  await call('/world', WORLD_BODY);
  const perform1 = await call('/perform', performBody);
  await call('/world', { name: 'Bare floor', clear: true });
  const perform2 = await call('/perform', performBody);
  const d1 = stripTiming(diffLeaves(perform0, perform1));
  const d2 = stripTiming(diffLeaves(perform0, perform2));
  ok('5a /perform identical with the flight standing on the live lane', d1.length === 0,
     d1.slice(0, 4).join(' | '));
  ok('5a /perform identical with a bare floor standing', d2.length === 0,
     d2.slice(0, 4).join(' | '));
  ok('5a a no-world /perform grew no key', perform1.stood === undefined
     && perform2.stood === undefined);
  console.log(`         ${leafCount(perform0)} leaves compared`
            + ' (50×14 frames + 50×7 roots + 50×3 commands + 14 joint names + 15 scalars)');
}

// --- 5b — the world that stood is the world that was asked for -------------
let perfA = null;
{
  // A clean slate first, so the POST /world's merge has no older section's
  // notes to keep: a /perform plan is built with `against: null` and inherits
  // nothing, and the comparison is only meaningful against a world in the same
  // state. `ball` is restated by moving it to its own compiled mark.
  await call('/world', { name: 'clean', clear: true, props: [], walls: [],
                         ball: { x: 0.55, y: 0.10 } });
  const postWorld = await call('/world', EXAMPLE_A);
  perfA = await call('/perform', { track: TRACK, rollouts: 1, seconds: 1,
                                   world: EXAMPLE_A, spawn: SPAWN_A });
  if (perfA.error) throw new Error(`/perform refused worked example A: ${perfA.error}`);
  const ds = diffLeaves(perfA.stood.steps, postWorld.steps);
  ok('5b the laid flight is leaf-identical to a POST /world of the same body',
     ds.length === 0, ds.slice(0, 4).join(' | '));
  console.log(`         ${leafCount(perfA.stood.steps)} leaves compared `
            + '(4 steps × 6 fields)');
  const du = diffLeaves(perfA.stood.unexpressed, postWorld.unexpressed);
  ok('5b and so is what neither of them could express', du.length === 0,
     du.slice(0, 4).join(' | '));
  ok('5b ten blocks parked', perfA.stood.bank.parked === 10, `${perfA.stood.bank.parked}`);
  ok('5b the world says it was set', perfA.stood.world.set === true
     && perfA.stood.world.name === EXAMPLE_A.name, JSON.stringify(perfA.stood.world));
}

// --- 5c — tick 0 on the batch lane is layoutStairs in closed form ----------
{
  // `seconds` is clamped to 0.2 by the route, so this is ten ticks and not one;
  // the claim is about EXACTNESS at the read point, not about how many ticks
  // ran after it.
  const one = await call('/perform', { track: TRACK, rollouts: 1, seconds: 0.02,
                                       world: EXAMPLE_A, spawn: SPAWN_A });
  const bad = [];
  one.stood.steps.forEach((s, i) => {
    const x = STAIR_START + i * STAIR_RUN + STEP_HALF_DEPTH;
    if (!Object.is(s.x, r4(x))) bad.push(`x[${i}] ${s.x} vs ${r4(x)}`);
    if (!Object.is(s.top, r4((i + 1) * RISE))) bad.push(`top[${i}] ${s.top} vs ${r4((i + 1) * RISE)}`);
  });
  ok('5c every block is where layoutStairs would have put it', bad.length === 0,
     bad.slice(0, 4).join('; '));
}

// --- 5d — the blocks agree across all three lanes --------------------------
let climbClip = null;
{
  climbClip = await call('/climb', { ...climbBody, clip: true });
  const dx = diffLeaves(climbClip.stood.steps.map(s => s.x), perfA.stood.steps.map(s => s.x));
  ok('5d the climb lane and the batch lane laid the same blocks', dx.length === 0,
     dx.slice(0, 4).join(' | '));
}

// --- 5e — the loan was handed back -----------------------------------------
{
  // The world standing at this instant, so the four re-runs can be shown to
  // have left it alone. (Phase 4a took no GET /world baseline: no world
  // existed then.)
  const worldBefore = await call('/world');
  const climb3 = await call('/climb', climbBody);
  const chase3 = await call('/chase', chaseBody);
  const record3 = await call('/record', recordBody);
  const worldAfter = await call('/world');
  const dc = stripTiming(diffLeaves(climb0, climb3));
  const dh = stripTiming(diffLeaves(chase0, chase3));
  const dr = stripTiming(diffLeaves(record0, record3));
  const dw = diffLeaves(worldBefore, worldAfter);
  ok('5e /climb unmoved after a /perform carried a world', dc.length === 0, dc.slice(0, 4).join(' | '));
  ok('5e /chase unmoved', dh.length === 0, dh.slice(0, 4).join(' | '));
  ok('5e /record unmoved', dr.length === 0, dr.slice(0, 4).join(' | '));
  ok('5e and the standing world is untouched', dw.length === 0, dw.slice(0, 4).join(' | '));
}

// --- 5f — the spawn landed exactly, and its absence is said ----------------
{
  const s = perfA.stood.spawn;
  ok('5f the spawn is where it was written',
     Object.is(s.x, 0.05) && Object.is(s.y, 1.305) && Object.is(s.z, 0.12),
     JSON.stringify(s));
  const noSpawn = await call('/perform', { track: TRACK, rollouts: 1, seconds: 1,
                                           world: EXAMPLE_A });
  ok('5f without one the duck is on its compiled mark',
     noSpawn.stood.spawn.x === 0 && noSpawn.stood.spawn.y === 0,
     JSON.stringify(noSpawn.stood.spawn));
  ok('5f and the room says the duck is beside it, not in it',
     noSpawn.stood.unexpressed.some(n => n.what === 'spawn'),
     noSpawn.stood.unexpressed.map(n => n.what).join(', '));
}

// --- 5g — CONCURRENCY, the thing phase 4 structurally cannot see -----------
{
  // Phase 4 awaits every call before issuing the next, so a loan held across an
  // `await` passes it. These pairs are issued WITHOUT awaiting the first.
  //
  // THE DUCK IS SPAWNED ON THE FIRST TREAD, and that is what makes this able to
  // fail rather than merely able to pass. What the loan protects is the
  // step-step isolation, and a duck standing BESIDE the flight barely feels
  // whether fourteen overlapping 200 kg blocks are shoving each other apart
  // inside a tick; a duck standing ON one feels it immediately (measured: the
  // trunk ends 176 mm up isolated and 54 mm up not). A gate whose victim cannot
  // be hurt is a gate that cannot go red.
  //
  // AND THE STEERING LOOP RUNS FOR THE WHOLE RUN, not for one hold. A single
  // 0.4 s /intent overlaps about twenty of the /perform's fourteen hundred
  // ticks — measured: with the loan deliberately broken, twenty was not enough
  // to move a leaf and a hundred was. Back-to-back holds for as long as the
  // /perform lasts is also the situation this is actually about: somebody
  // steering a duck while an editor run is scored.
  const ON_TREAD = { x: 0.22, y: 1.305, z: 0.18 };
  const longPerform = { track: TRACK, rollouts: 8, seconds: 3,
                        world: EXAMPLE_A, spawn: ON_TREAD };
  const base = await call('/perform', longPerform);
  const lanesAt = async () => (await call('/lanes'));
  let worstPerform = [], overlapTotal = 0;
  const before1 = await lanesAt();
  for (let i = 0; i < 5; i++) {
    let running = true;
    const p = call('/perform', longPerform).then(v => { running = false; return v; });  // NOT awaited
    const steering = (async () => {
      while (running) await call('/intent', { vx: 0.2, hold: 0.4 });
    })();
    const perf = await p;
    await steering;
    const d = stripTiming(diffLeaves(base, perf));
    if (d.length) worstPerform = d;
  }
  const after1 = await lanesAt();
  overlapTotal += after1.performTicksWithAnotherRequestInFlight
                - before1.performTicksWithAnotherRequestInFlight;
  ok('5g a /intent overlapping a /perform moved no leaf of it',
     worstPerform.length === 0, worstPerform.slice(0, 4).join(' | '));
  console.log(`         direction 1: overlap counter +${after1.performTicksWithAnotherRequestInFlight - before1.performTicksWithAnotherRequestInFlight}`
            + ` of ${after1.performTicks - before1.performTicks} perform ticks`);

  const before2 = await lanesAt();
  let worstPerform2 = [], worstClimb = [];
  for (let i = 0; i < 5; i++) {
    const p = call('/perform', longPerform);          // NOT awaited
    const c = call('/climb', climbBody);              // queues onto the batch lane
    const [perf, cl] = await Promise.all([p, c]);
    const dp = stripTiming(diffLeaves(base, perf));
    const dc = stripTiming(diffLeaves(climb0, cl));
    if (dp.length) worstPerform2 = dp;
    if (dc.length) worstClimb = dc;
  }
  const after2 = await lanesAt();
  const delta2 = after2.performTicksWithAnotherRequestInFlight
               - before2.performTicksWithAnotherRequestInFlight;
  overlapTotal += delta2;
  ok('5g a /climb overlapping a /perform moved no leaf of the /perform',
     worstPerform2.length === 0, worstPerform2.slice(0, 4).join(' | '));
  ok('5g nor any leaf of the /climb', worstClimb.length === 0, worstClimb.slice(0, 4).join(' | '));
  console.log(`         direction 2: overlap counter +${delta2}`
            + ` of ${after2.performTicks - before2.performTicks} perform ticks`);
  // AND THE RACE HAS TO HAVE HAPPENED. A green phase that never overlapped
  // proves nothing, which is the failure mode phase 1 shipped for months.
  ok('5g the race actually overlapped', overlapTotal > 0,
     overlapTotal > 0 ? `${overlapTotal} perform ticks stepped with another request in flight`
                      : 'the race did not overlap; this phase proved nothing');
  const w = await call('/world');
  ok('5g and the standing world came through it', w.steps.length === EXAMPLE_A.steps.length,
     `${w.steps.length} steps standing`);
}

// --- 5h — the loan is per tick ---------------------------------------------
{
  ok('5h the blocks were re-pinned every tick', perfA.stood.pinnedEveryTick === true);
  ok('5h and the sag inside one tick is a real, small number',
     perfA.stood.sag_mm > 0 && perfA.stood.sag_mm < 3, `${perfA.stood.sag_mm} mm`);
}

// --- 5i — the one-rollout criterion ----------------------------------------
{
  const one = await call('/perform', { track: TRACK, rollouts: 1, seconds: 1 });
  const two = await call('/perform', { track: TRACK, rollouts: 2, seconds: 1 });
  ok('5i one rollout says it dropped from 0.120 m',
     one.criterion === 'stayed upright to the end, dropped from 0.120 m', one.criterion);
  ok('5i two rollouts still say the sweep',
     two.criterion === 'stayed upright to the end, over drop heights 0.120-0.130 m', two.criterion);
}

// --- 5j — the refusals, and that a refusal changes nothing -----------------
{
  const worldBefore = await call('/world');
  const fifteen = Array.from({ length: 15 }, (_, i) => ({ x: 0, top: 0.06 }));
  const cases = [
    ['a world that says nothing about the bank', { world: {} }, /say what the step bank should do/],
    ['fifteen steps', { world: { steps: fifteen } }, /bank of 14/],
    ['both clear and steps', { world: { clear: true, steps: [] } }, /say one or the other/],
    ['a spawn outside the arena', { spawn: { x: 2, y: 0 } }, /outside this arena/],
    ['a world that is not an object', { world: [1, 2] }, /the same shape POST \/world takes/],
    ['a spawn that is not numbers', { spawn: { x: 'left', y: 0 } }, /give x and y as finite numbers/],
  ];
  for (const [label, extra, re] of cases) {
    const a = await call('/perform', { track: TRACK, rollouts: 1, seconds: 1, ...extra });
    ok(`5j ${label} is refused in the same words as /world`, re.test(a.error || ''),
       JSON.stringify(a.error ?? Object.keys(a)));
    const w = await call('/world');
    const d = diffLeaves(worldBefore, w);
    ok(`5j     ...and changed nothing`, d.length === 0, d.slice(0, 3).join(' | '));
  }
  const still = await call('/perform', performBody);
  const d = stripTiming(diffLeaves(perform0, still));
  ok('5j a no-world /perform is still what it was in 5a', d.length === 0, d.slice(0, 4).join(' | '));
}

// --- 5k — the module global was never written ------------------------------
{
  // A SECOND, FRESH BENCH, because by now this one has a world standing and the
  // claim is about a bench that has never been given one.
  const fresh = await nodeBench();
  const fcall = async (p, b) => JSON.parse(JSON.stringify(await fresh.handle(
    new URL(p, 'http://bench.local'), b ?? {})));
  const perf = await fcall('/perform', { track: TRACK, rollouts: 1, seconds: 1,
                                         world: EXAMPLE_A, spawn: SPAWN_A });
  ok('5k the /perform laid its world', perf.stood?.steps?.length === EXAMPLE_A.steps.length,
     `${perf.stood?.steps?.length} steps`);
  const w = await fcall('/world');
  ok('5k and the live lane still has no world at all', w.world.set === false,
     JSON.stringify(w.world));
  ok('5k nothing is parked, because nothing parked it', w.bank.parked === 0,
     `${w.bank.parked} parked, ${w.steps.length} listed`);
}

// --- 5l — the clip moved no score ------------------------------------------
{
  const extra = Object.keys(climbClip).filter(k => !(k in climb0));
  ok('5l the only new keys are `clip` and `stood`',
     extra.length === 2 && extra.includes('clip') && extra.includes('stood'), extra.join(', '));
  const bad = [];
  for (const k of Object.keys(climb0)) {
    if (TIMING.some(t => t === '/' + k)) continue;
    const d = diffLeaves(climb0[k], climbClip[k]);
    if (d.length) bad.push(`${k}: ${d[0]}`);
  }
  ok('5l every scored key is Object.is-identical with a clip asked for',
     bad.length === 0, bad.slice(0, 4).join(' | '));
  const c = climbClip.clip;
  ok('5l the clip is a clip', c.frames.length === c.roots.length && c.frames.length === c.ticks
     && c.frames.every(f => f.length === 14) && c.roots.every(r => r.length === 7)
     && c.hz === 50 && c.settleExcluded === true,
     `${c.ticks} ticks, frames ${c.frames[0].length} wide, roots ${c.roots[0].length} wide, hz ${c.hz}`);
  // AND ONCE MORE WITH A WORLD STANDING ON THE LIVE LANE, against a bare floor:
  // the climb rig owns its own mjData and must not be able to tell.
  await call('/world', WORLD_BODY);
  const withFlight = await call('/climb', { ...climbBody, clip: true });
  await call('/world', { name: 'Bare floor', clear: true });
  const withFloor = await call('/climb', { ...climbBody, clip: true });
  const d = stripTiming(diffLeaves(withFlight, withFloor));
  ok('5l a clip-bearing /climb is the same with a flight standing and with a bare floor',
     d.length === 0, d.slice(0, 4).join(' | '));
  console.log(`         ${leafCount(withFlight)} leaves compared`);
}

console.log(failures ? `WORLD PARITY FAILED: ${failures} check(s)` : 'WORLD PARITY OK');
process.exitCode = failures ? 1 : 0;
