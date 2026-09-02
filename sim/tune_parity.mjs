// The two things /tune borrows from another language, checked against it.
//
// WHY A THIRD PARITY SCRIPT. `policy_parity.mjs` asks whether this repo's
// forward pass agrees with onnxruntime, and `physics_parity.mjs` asks whether
// the phone's trajectory is the desk's. Neither of them touches /tune, and
// /tune is the endpoint that took two definitions out of somebody else's
// repository and rewrote them here:
//
//   • WHAT A GAIN AND A TRIM MEAN. duckkit's `DuckPolicyWriter.folding` is the
//     definition — it is the code that will actually write the tuned file the
//     robot loads. `foldParameters` in policyforward.mjs is a transcription of
//     it, and a transcription that is 1e-5 out is a DIFFERENT NETWORK from the
//     one the app ships, so a search would be scoring something nobody will
//     ever run. This compares the canonical bytes, and it compares them as
//     BYTES: `Float(gain)` in Swift rounds the multiplier to binary32 before it
//     multiplies, and JavaScript's obvious spelling does not, which is a
//     different last bit on a gain like 1.07.
//
//   • WHAT EACH REWARD TERM IS. StudioKit's `RunMetrics` reads the six terms
//     out of `microduck_velocity_env_cfg.py` and is what every clip in the app
//     is graded by. `rewardSums` in duckbench-core.mjs is a second reading of
//     the same config, and two readings of one config is exactly how a search
//     comes to climb a hill the rest of the app cannot see — with both numbers
//     looking plausible. So both sides score the SAME fifty ticks and must
//     agree to 1e-9.
//
// THE TICKS COME OUT OF THE ENDPOINT, NOT OUT OF A HARNESS. `/tune` with
// `"trace": true` answers with the first drop's ticks, so the fixture behind
// this gate is a recording of the thing clients call. A fixture produced by a
// private code path would prove that the private path agrees with Swift and say
// nothing about the endpoint.
//
//   node sim/tune_parity.mjs            # check both halves
//   node sim/tune_parity.mjs --emit     # re-record the trace fixture first
//   node sim/tune_parity.mjs --bench 100.122.199.6:8770   # emit from a LIVE bench
//
// THE SWIFT HALF HAS TO HAVE RUN. `base.bin`, `folded.bin` and `fold.json` are
// written by `BenchTuneParityTests` in StudioKit — `swift test` produces them.
// If they are not there this script says so and FAILS rather than reporting a
// pass it did not earn.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { rewardSums, travelledAlongCommand, netDisplacement, twistOf,
         TUNE_TERMS } from './duckbench-core.mjs';
import { loadParameters, foldParameters, canonicalBytes } from './policyforward.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const STUDIO = process.env.DUCK_STUDIO
  || path.join(process.env.HOME || '/home/craigm26', 'projects', 'duck-studio');
const FIXTURES = path.join(STUDIO, 'StudioKit/Tests/StudioKitTests/Fixtures/tune');
/**
 * TWO FIXTURES, BECAUSE ONE CANNOT FAIL ON THREE THINGS. `trace.json` commands
 * only vx, so a wrong sign on the commanded yaw rate, a swapped vy index, or
 * the turn-tracking term reading the wrong command would all pass it.
 * `trace-turning.json` commands vx, vy AND vyaw at once, and closes those.
 */
const SCHEDULES = {
  walking: [[0, { vx: 0, vy: 0, vyaw: 0 }], [0.5, { vx: 0.5, vy: 0, vyaw: 0 }]],
  turning: [[0, { vx: 0, vy: 0, vyaw: 0 }], [0.5, { vx: 0.3, vy: 0.2, vyaw: 0.5 }]],
};
const FIXTURE_NAMES = { walking: 'trace.json', turning: 'trace-turning.json' };
const fixturePath = which => path.join(FIXTURES, FIXTURE_NAMES[which]);
const FOLD = path.join(STUDIO, 'StudioKit/.build/fold-fixture');

const C = JSON.parse(fs.readFileSync(path.join(HERE, 'duckkit-constants.json'), 'utf8'));
const POLICY_JOINTS = C.jointNames.filter(n => n !== 'mouth');
const HOME14 = POLICY_JOINTS.map(n => C.homePose[C.jointNames.indexOf(n)]);

/** The weights, as `DuckTuner.terms` reads them out of RunMetrics. */
const WEIGHTS = { upright: 2.0, track_linear_velocity: 2.0, track_angular_velocity: 2.0,
                  pose: 1.0, body_ang_vel: -0.05, action_rate_l2: -1.0 };
const reward = terms => TUNE_TERMS.reduce((t, k) => t + terms[k] * WEIGHTS[k], 0);

const arg = (name, fallback = null) => {
  const i = process.argv.indexOf(name);
  return i < 0 ? fallback : (process.argv[i + 1] ?? true);
};
const TOLERANCE = 1e-9;

let failures = 0;
const fail = line => { failures++; console.log(`  FAIL  ${line}`); };
const pass = line => console.log(`  ok    ${line}`);

// ── recording the fixture ────────────────────────────────────────────────────

/**
 * Fifty ticks of `alpha_walking` under the app's own walking command, with the
 * identity residual — the exact request `TuneView`'s probe sends, one second
 * long, plus `"trace": true`.
 *
 * THE IDENTITY RESIDUAL ON PURPOSE. The fixture is about the REWARD, not about
 * the fold, and a folded network would make a failure ambiguous between the two
 * halves of this script. The fold has its own half below, and it is checked as
 * bytes rather than through a trajectory.
 */
const REQUEST = {
  policy: 'alpha_walking.onnx',
  gain: new Array(14).fill(1),
  offset: new Array(14).fill(0),
  seconds: 1,
  drops: [0.1231],
  schedule: SCHEDULES.walking,
  terms: TUNE_TERMS,
  trace: true,
};
const requestFor = which => ({ ...REQUEST, schedule: SCHEDULES[which] });

async function askTheBench(address, which = 'walking') {
  const REQUEST = requestFor(which);
  if (address) {
    const base = /^https?:\/\//.test(address) ? address : `http://${address}`;
    const response = await fetch(`${base}/tune`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify(REQUEST),
    });
    return { answer: await response.json(), from: base };
  }
  const { nodeBench } = await import('./duckbench-node.mjs');
  const { handle } = await nodeBench();
  return { answer: await handle(new URL('http://bench.local/tune'), REQUEST),
           from: 'this process, through duckbench-core.mjs' };
}

async function emit(address, which = 'walking') {
  const FIXTURE = fixturePath(which);
  const REQUEST = requestFor(which);
  const { answer, from } = await askTheBench(address, which);
  if (answer.error) throw new Error(`the bench refused: ${answer.error}`);
  if (!Array.isArray(answer.trace) || !answer.trace.length) {
    throw new Error('the bench answered without a trace — it is older than `"trace": true`');
  }
  const fixture = {
    why: 'Fifty control ticks the duck bench recorded, WITH the six reward-term values it '
       + 'computed from them. Two implementations of Pollen\'s reward — StudioKit\'s RunMetrics '
       + 'and duckbench-core.mjs\'s rewardSums — score these same ticks and must agree to the '
       + 'tolerance below. Change either transcription and exactly one side goes red.',
    toleranceWhy: 'Measured 2026-09-02 on this Pi, the two languages agree EXACTLY: every one of '
                + 'the six doubles is bit-identical, and the assertion still passes with the '
                + 'tolerance set to 0. It is not left at 0, because five of the six terms end in '
                + '`exp` and neither glibc\'s nor Apple\'s `exp` is correctly rounded — two '
                + 'libms may legitimately differ by an ulp, and this fixture has to keep passing '
                + 'on the phone the app actually ships to. 1e-9 is nine orders of magnitude '
                + 'below the smallest change a search would ever act on.',
    recordedBy: 'duck-sounds sim/tune_parity.mjs --emit, from POST /tune with "trace": true',
    recordedFrom: from,
    recordedOn: new Date().toISOString().slice(0, 10),
    request: { policy: REQUEST.policy, seconds: REQUEST.seconds, drops: REQUEST.drops,
               schedule: REQUEST.schedule, gain: 'identity', offset: 'identity' },
    policy: answer.policy,
    hz: C.tickHz,
    plantName: answer.plantName,
    plantDigest: answer.plantDigest,
    engine: answer.engine,
    tolerance: TOLERANCE,
    terms: answer.terms,
    travelled: answer.travelled,
    netDisplacement: answer.perDrop[0].netDisplacement,
    ticks: answer.trace,
  };
  fs.mkdirSync(path.dirname(FIXTURE), { recursive: true });
  fs.writeFileSync(FIXTURE, JSON.stringify(fixture, null, 1) + '\n');
  console.log(`wrote ${FIXTURE}`);
  console.log(`  ${fixture.ticks.length} ticks, plant ${fixture.plantName} `
            + `${String(fixture.plantDigest).slice(0, 16)}, from ${from}`);
}

// ── half one: the six terms ──────────────────────────────────────────────────

function checkTerms(which = 'walking') {
  const FIXTURE = fixturePath(which);
  console.log(`THE SIX TERMS (${which} schedule) — duckbench-core.mjs's rewardSums against the`);
  console.log('numbers the bench recorded beside the trace, which RunMetrics also has to reproduce.');
  if (!fs.existsSync(FIXTURE)) {
    return fail(`no trace fixture at ${FIXTURE} — run this script with --emit`);
  }
  const fixture = JSON.parse(fs.readFileSync(FIXTURE, 'utf8'));
  const ticks = fixture.ticks;
  console.log(`  ${ticks.length} ticks, plant ${fixture.plantName} `
            + `${String(fixture.plantDigest).slice(0, 16)}, tolerance ${fixture.tolerance}`);

  // The twist the fixture carries has to BE this file's rotation of the raw
  // velocity beside it — the one place both languages could agree on every
  // formula and still be describing different physics.
  let worstTwist = 0;
  for (const tick of ticks) {
    const mine = twistOf(tick.root, tick.qvel);
    for (let k = 0; k < 6; k++) worstTwist = Math.max(worstTwist, Math.abs(mine[k] - tick.twist[k]));
  }
  if (worstTwist < fixture.tolerance) pass(`twist rotation, worst ${worstTwist.toExponential(2)}`);
  else fail(`twist rotation is ${worstTwist.toExponential(2)} out`);

  const { sums, ticks: n, rateTicks } = rewardSums(ticks, POLICY_JOINTS, HOME14);
  const mine = {};
  for (const term of TUNE_TERMS) {
    mine[term] = sums[term] / (term === 'action_rate_l2' ? rateTicks : n);
  }
  for (const term of TUNE_TERMS) {
    const theirs = fixture.terms[term];
    const gap = Math.abs(mine[term] - theirs);
    const line = `${term.padEnd(23)} ${mine[term].toFixed(12)}  vs  ${theirs.toFixed(12)}  `
               + `Δ ${gap.toExponential(2)}`;
    if (theirs === undefined) fail(`${term} is not in the fixture`);
    else if (gap < fixture.tolerance) pass(line);
    else fail(line);
  }
  const gapReward = Math.abs(reward(mine) - reward(fixture.terms));
  if (gapReward < fixture.tolerance) {
    pass(`weighted reward ${reward(mine).toFixed(12)} (DuckTuner's weights)`);
  } else fail(`weighted reward is ${gapReward.toExponential(2)} out`);

  // And the two distance numbers, which are this file's arithmetic as well.
  const travelled = travelledAlongCommand(ticks), net = netDisplacement(ticks);
  const gapTravel = Math.abs(travelled - fixture.travelled);
  if (gapTravel < fixture.tolerance) {
    pass(`travelled ${(travelled * 1000).toFixed(3)} mm along the commanded direction, `
       + `net ${(net * 1000).toFixed(3)} mm`);
  } else fail(`travelled is ${gapTravel.toExponential(2)} out`);
}

// ── half two: the fold, as bytes ─────────────────────────────────────────────

function checkFold() {
  console.log('');
  console.log('THE FOLD — policyforward.mjs\'s foldParameters against the bytes duckkit\'s');
  console.log('DuckPolicyWriter.folding actually wrote, for the same gain and trim.');
  const recipe = path.join(FOLD, 'fold.json');
  if (!fs.existsSync(recipe)) {
    return fail(`no fold fixture at ${FOLD} — it is written by BenchTuneParityTests in `
              + 'StudioKit, so run `swift test` in duck-studio first');
  }
  const { gain, trim, policy, baseFingerprint, foldedFingerprint } =
    JSON.parse(fs.readFileSync(recipe, 'utf8'));
  const base = fs.readFileSync(path.join(FOLD, 'base.bin'));
  const theirs = fs.readFileSync(path.join(FOLD, 'folded.bin'));
  console.log(`  ${policy}, ${base.byteLength} bytes, gain ${gain[0]}…${gain[13]}, `
            + `trim ${trim[0]}…${trim[13]}`);
  console.log(`  base ${String(baseFingerprint).slice(0, 16)} → `
            + `folded ${String(foldedFingerprint).slice(0, 16)}`);

  const mine = canonicalBytes(foldParameters(loadParameters(base), gain, trim));
  if (mine.byteLength !== theirs.byteLength) {
    return fail(`this fold is ${mine.byteLength} bytes and Swift's is ${theirs.byteLength}`);
  }
  let differing = 0, firstAt = -1;
  for (let i = 0; i < mine.length; i++) {
    if (mine[i] !== theirs[i]) { differing++; if (firstAt < 0) firstAt = i; }
  }
  // The float difference too, so a failure says HOW WRONG rather than only THAT.
  const a = new Float32Array(mine.slice().buffer);
  const b = new Float32Array(theirs.buffer, theirs.byteOffset, theirs.byteLength >> 2);
  let worst = 0, worstAt = -1;
  for (let i = 0; i < a.length; i++) {
    const gap = Math.abs(a[i] - b[i]);
    if (gap > worst) { worst = gap; worstAt = i; }
  }
  if (differing === 0) {
    pass(`byte for byte identical over all ${mine.byteLength} bytes — the bench folds the `
       + 'network the app would write, not one that rounds like it');
  } else {
    fail(`${differing} of ${mine.byteLength} bytes differ, first at ${firstAt}; worst float `
       + `gap ${worst.toExponential(3)} at parameter ${worstAt}`);
  }

  // A fold that did nothing would pass the comparison above against a base that
  // also did nothing. Prove the fixture is a real fold.
  let moved = 0;
  for (let i = 0; i < base.length; i++) if (base[i] !== theirs[i]) moved++;
  if (moved > 0) pass(`the fixture is a real fold: ${moved} bytes moved from the base`);
  else fail('base.bin and folded.bin are identical — the fixture proves nothing');
}

// ── the run ──────────────────────────────────────────────────────────────────

if (process.argv.includes('--emit')) {
  for (const which of Object.keys(SCHEDULES)) await emit(arg('--bench', null), which);
}

console.log('');
for (const which of Object.keys(SCHEDULES)) checkTerms(which);
checkFold();
console.log('');
if (failures) {
  console.log(`TUNE PARITY FAILED: ${failures} check(s)`);
  process.exitCode = 1;
} else {
  console.log('TUNE PARITY OK — the bench\'s reward is StudioKit\'s reward, and the bench\'s '
            + 'fold is duckkit\'s fold.');
}
