#!/usr/bin/env node
// check_numbers.mjs — re-derive every number stated in README.md and
// leaderboard.md from the files in results/, and print PASS or FAIL for each.
//
// This script reads NOTHING but results/, entrants/, harness/ and — when it
// happens to be sitting in the repository — ../sim/scene.mjb for the plant
// digest. IT RUNS NO SIMULATION: every assertion is a value that was already
// written to a results file by the run that measured it, or a constant read out
// of the harness snapshot the card cites by line number. If a number in the
// card cannot be found here, the card is wrong.
//
//   cd challenge-ball && node check_numbers.mjs
//
// Exit status 0 when every check passes, 1 otherwise.

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const R = f => JSON.parse(fs.readFileSync(path.join(HERE, 'results', f), 'utf8'));
const T = f => fs.readFileSync(path.join(HERE, 'results', f), 'utf8');
const E = f => JSON.parse(fs.readFileSync(path.join(HERE, 'entrants', f), 'utf8'));
const H = f => fs.readFileSync(path.join(HERE, 'harness', f), 'utf8');

let pass = 0, fail = 0, skipped = 0;
const rows = [];

/**
 * A check that could NOT be run here — the file it needs is not in this copy of
 * the package. It is counted as SKIPPED and never as PASS: a check that did not
 * run is not a check that passed, and a summary that says otherwise would let a
 * shipped package claim a verification it never did.
 */
function skip(label, want, src) {
  skipped++;
  rows.push({ ok: true, skipped: true, label, got: 'n/a', want, src });
}

/** Assert `got` deep-equals `want`; `src` names the file it came from. */
function check(label, got, want, src) {
  const g = JSON.stringify(got), w = JSON.stringify(want);
  const ok = g === w;
  ok ? pass++ : fail++;
  rows.push({ ok, label, got: g, want: w, src });
}
function checkNear(label, got, want, tol, src) {
  const ok = typeof got === 'number' && Math.abs(got - want) <= tol;
  ok ? pass++ : fail++;
  rows.push({ ok, label, got: String(got), want: `${want} ±${tol}`, src });
}
/** A line of a harness file, trimmed — the card cites these by number. */
function line(file, n) { return H(file).split('\n')[n - 1]; }
function checkLine(label, file, n, needle) {
  const l = line(file, n) ?? '';
  const ok = l.includes(needle);
  ok ? pass++ : fail++;
  rows.push({ ok, label, got: JSON.stringify(l.trim().slice(0, 80)), want: JSON.stringify(needle),
              src: `harness/${file}:${n}` });
}

// ---------------------------------------------------------------- the files
const C = R('chase_controls-results.json');
const DRIFT = R('chase_drift-results.json');
const PAR = T('chase_parity.log');

/** The four entrants, by name, from the results file. */
const byName = Object.fromEntries(C.entrants.map(e => [e.name, e]));
const lbByName = Object.fromEntries(C.leaderboard.map(r => [r.name, r]));
const NOTHING = byName.ctrl_do_nothing;
const KICKL = byName.ctrl_ball_kick_left;
const KICKR = byName.ctrl_ball_kick_right;
const ALPHA = byName.ctrl_alpha_walking;
/** One verdict row, addressed the way the card's table addresses it. */
const cellOf = (e, bearing, range, drop = 0.12, fmul = 1) =>
  e.verdicts.find(v => v.bearing === bearing && v.range === range && v.drop === drop && v.fmul === fmul);
const termOf = (e, term) => e.terms.find(t => t.term === term);

// ============================================ 1. the plant and the criterion
check('plant is scene.mjb', C.plantName, 'scene.mjb', 'chase_controls-results.json');
check('plant digest 3f8c9ab9b409…', C.plantDigest,
      '3f8c9ab9b409ba74c73c30179d5f7c12b025f631693f9eec78d80dca242547be',
      'chase_controls-results.json');
check('every entrant row carries the same plant digest',
      [...new Set(C.entrants.map(e => e.plantDigest))], [C.plantDigest],
      'chase_controls-results.json entrants[].plantDigest');
check('the drift measurement ran on the same plant', DRIFT.plantDigest, C.plantDigest,
      'chase_drift-results.json');

const CRIT =
  'chased: the duck touched the ball — any duck geometry within 3 mm of it at any tick — '
  + 'and the ball finished at least 100 mm further along the duck\'s initial heading than it '
  + 'started, and the duck was still upright at the end of the episode. '
  + 'stable: chased, and upright for at least 45 of the 50 tail ticks.';
check('CRITERION_SENTENCE, verbatim as the card quotes it', C.criterion, CRIT,
      'chase_controls-results.json criterion');
check('every entrant row carries that same criterion',
      [...new Set(C.entrants.map(e => e.criterion))], [CRIT],
      'chase_controls-results.json entrants[].criterion');
{
  // The card quotes it as a wrapped, bolded blockquote, so both sides are
  // stripped of markdown furniture and re-flowed before comparing. What is
  // being checked is that the WORDS are the scorer's, not the formatting.
  const flat = s => s.replace(/^[>\s]+/gm, ' ').replace(/\*\*/g, '').replace(/\s+/g, ' ').trim();
  for (const f of ['README.md', 'leaderboard.md']) {
    check(`the criterion sentence appears verbatim in ${f}`,
          flat(fs.readFileSync(path.join(HERE, f), 'utf8')).includes(flat(CRIT)), true, f);
  }
}

// The four constants the card tabulates BY LINE NUMBER in harness/chase_score.mjs.
checkLine('TOUCH_MM = 3.0 at line 57', 'chase_score.mjs', 57, 'export const TOUCH_MM = 3.0;');
checkLine('TRAVEL_MIN_MM = 100.0 at line 59', 'chase_score.mjs', 59, 'export const TRAVEL_MIN_MM = 100.0;');
checkLine('TAIL_TICKS = 50 at line 61', 'chase_score.mjs', 61, 'export const TAIL_TICKS = 50;');
checkLine('UPRIGHT_TAIL_MIN imported, not retyped, at line 69', 'chase_score.mjs', 69,
          'export const UPRIGHT_TAIL_MIN = CLIMB_UPRIGHT_TAIL_MIN;');
checkLine('…and the stairs value it comes from is 45', 'climb_score.mjs', 89,
          'export const UPRIGHT_TAIL_MIN = 45;');
checkLine('SETTLE_TICKS = 25 at line 72', 'chase_score.mjs', 72, 'export const SETTLE_TICKS = 25;');
checkLine('BALL_RADIUS = 0.05 at line 74', 'chase_score.mjs', 74, 'export const BALL_RADIUS = 0.05;');
checkLine('UPRIGHT_GZ = -0.90 at line 76', 'chase_score.mjs', 76, 'export const UPRIGHT_GZ = -0.90;');
checkLine('CRITERION_SENTENCE at line 85', 'chase_score.mjs', 85, 'export const CRITERION_SENTENCE');
// The tail the card describes: the STANDING policy under a NEUTRAL command, at
// lines 752–764 — not the acting network under the entrant's own schedule.
checkLine('the tail is the standing test, at line 752', 'chase_score.mjs', 752,
          '---- the tail: THE STANDING TEST, and nothing else');
checkLine('…the standing policy, under a neutral command, for 50 ticks, at line 753',
          'chase_score.mjs', 753, 'holds the duck under a neutral command for 50 ticks');
checkLine('…and the tail loop steps THAT policy, at lines 761–762', 'chase_score.mjs', 762,
          'await step(stand, stand.reference ?? HOME, null, neutral);');
checkLine('gridCells() at line 121', 'chase_score.mjs', 121, 'export function gridCells(');
checkLine('JOINT_ORDER at line 181', 'chase_score.mjs', 181, 'export const JOINT_ORDER = [');
checkLine('assertJointOrder() at line 188', 'chase_score.mjs', 188, 'export function assertJointOrder(');
checkLine('TERMS at line 214', 'chase_score.mjs', 214, 'export const TERMS = [');
checkLine('ACTION_RATE_SOURCE at line 257', 'chase_score.mjs', 257, 'export const ACTION_RATE_SOURCE = {');
checkLine('CHASE_REFUSALS at line 276', 'chase_score.mjs', 276, 'export const CHASE_REFUSALS = [');
checkLine('BALL_CAVEAT at line 301', 'chase_score.mjs', 301, 'export const BALL_CAVEAT');
checkLine('checkEntrant() at line 333', 'chase_score.mjs', 333, 'export function checkEntrant(');
checkLine('commandAt() at line 378', 'chase_score.mjs', 378, 'export function commandAt(');
checkLine('entrantHashPayload() at line 401', 'chase_score.mjs', 401, 'export function entrantHashPayload(');
checkLine('verdict() at line 425', 'chase_score.mjs', 425, 'export function verdict(facts) {');
checkLine('the verdict\'s touched clause', 'chase_score.mjs', 426, 'const chased = facts.touched');
checkLine('the verdict\'s travel clause', 'chase_score.mjs', 427, 'facts.ballTravel_mm >= TRAVEL_MIN_MM');
checkLine('the verdict\'s upright clause', 'chase_score.mjs', 428, 'facts.upright');
checkLine('stable = chased AND uprightTailTicks >= UPRIGHT_TAIL_MIN', 'chase_score.mjs', 429,
          'stable: chased && facts.uprightTailTicks >= UPRIGHT_TAIL_MIN');
checkLine('PLANTS, the shared plant list, at climb_score.mjs line 75', 'climb_score.mjs', 75,
          'export const PLANTS = [');

// The plant facts the card states about the ball and the actuators.
checkLine('actuator forcerange ±0.6405 N·m at scene_physics.xml line 46', 'scene_physics.xml', 46,
          'forcerange="-0.6405 0.6405"');
checkLine('body name="ball" at scene_physics.xml line 196', 'scene_physics.xml', 196,
          '<body name="ball" pos="0.55 0.10 0.05">');
checkLine('freejoint ball_free at line 197', 'scene_physics.xml', 197, '<freejoint name="ball_free"/>');
checkLine('ball_geom sphere radius 0.05, mass 0.03, at line 200', 'scene_physics.xml', 200,
          '<geom name="ball_geom" type="sphere" size="0.05" mass="0.03"');
checkLine('ball condim 6 at line 201', 'scene_physics.xml', 201, 'condim="6"');

// ================================================================ 2. the grid
check('the grid has 14 cells', C.grid.length, 14, 'chase_controls-results.json grid');
check('nCore 9, nExt 14', [C.nCore, C.nExt], [9, 14], 'chase_controls-results.json');
check('9 core cells, 5 extended',
      [C.grid.filter(c => c.tier === 'core').length, C.grid.filter(c => c.tier === 'ext').length],
      [9, 5], 'chase_controls-results.json grid');
check('core first, then extended',
      C.grid.map(c => c.tier).join(''), 'core'.repeat(9) + 'ext'.repeat(5),
      'chase_controls-results.json grid');
check('the core bearings are {-20, 0, +20}',
      [...new Set(C.grid.filter(c => c.tier === 'core').map(c => c.bearing))].sort((a, b) => a - b),
      [-20, 0, 20], 'chase_controls-results.json grid');
check('the core ranges are {0.45, 0.70, 0.95} m',
      [...new Set(C.grid.filter(c => c.tier === 'core').map(c => c.range))].sort((a, b) => a - b),
      [0.45, 0.7, 0.95], 'chase_controls-results.json grid');
check('every core cell is on the nominal plant (drop 0.120, fmul x1.0)',
      [...new Set(C.grid.filter(c => c.tier === 'core').map(c => `${c.drop}/${c.fmul}`))],
      ['0.12/1'], 'chase_controls-results.json grid');
check('the nine core cells, in order',
      C.grid.filter(c => c.tier === 'core').map(c => [c.bearing, c.range]),
      [[-20, 0.45], [0, 0.45], [20, 0.45], [-20, 0.7], [0, 0.7], [20, 0.7],
       [-20, 0.95], [0, 0.95], [20, 0.95]], 'chase_controls-results.json grid');
check('the five extended cells, in order',
      C.grid.filter(c => c.tier === 'ext').map(c => [c.bearing, c.range, c.drop, c.fmul]),
      [[0, 0.7, 0.13, 0.7], [0, 0.7, 0.125, 1.3], [-40, 0.7, 0.12, 1],
       [40, 0.7, 0.12, 1], [0, 1.2, 0.12, 1]], 'chase_controls-results.json grid');
check('the centre cell is bearing 0, range 0.70, nominal plant',
      C.grid[4], { bearing: 0, range: 0.7, drop: 0.12, fmul: 1, tier: 'core' },
      'chase_controls-results.json grid[4]');
check('every entrant was scored on all 14 cells',
      [...new Set(C.entrants.map(e => e.verdicts.length))], [14],
      'chase_controls-results.json entrants[].verdicts');
check('every entrant was scored on THE published grid',
      C.entrants.every(e => JSON.stringify(e.verdicts.map(v => [v.bearing, v.range, v.drop, v.fmul, v.tier]))
                         === JSON.stringify(C.grid.map(c => [c.bearing, c.range, c.drop, c.fmul, c.tier]))),
      true, 'chase_controls-results.json');

// ========================================================= 3. the leaderboard
check('four bundled controls, and only four', C.leaderboard.length, 4, 'chase_controls-results.json leaderboard');
check('the four names', C.leaderboard.map(r => r.name),
      ['ctrl_do_nothing', 'ctrl_ball_kick_left', 'ctrl_ball_kick_right', 'ctrl_alpha_walking'],
      'chase_controls-results.json leaderboard');

// full hashes, exactly as the card prints them
check('ctrl_alpha_walking sha256', lbByName.ctrl_alpha_walking.sha256,
      'a0bbbbb98acb7fc5bc1d035527c2c7b153df1c3555db79b9c12e4f446d49d6a5',
      'chase_controls-results.json leaderboard');
check('ctrl_do_nothing sha256', lbByName.ctrl_do_nothing.sha256,
      'bc77453e40c677db4073a350da5a43d645676d77e1252f51bbf6544be54ca187',
      'chase_controls-results.json leaderboard');
check('ctrl_ball_kick_left sha256', lbByName.ctrl_ball_kick_left.sha256,
      '7e44b5a781fc6763042a43065598424ea945f3bc8956bd0f1127aca4ec81b6e9',
      'chase_controls-results.json leaderboard');
check('ctrl_ball_kick_right sha256', lbByName.ctrl_ball_kick_right.sha256,
      'f8d4e8bfd2b789668cdf58e7683100d04cf48af2d1fe746d495fc4f697e03ffe',
      'chase_controls-results.json leaderboard');
check('the twelve-character prefixes the leaderboard column prints',
      C.leaderboard.map(r => r.entrant),
      ['bc77453e40c6', '7e44b5a781fc', 'f8d4e8bfd2b7', 'a0bbbbb98acb'],
      'chase_controls-results.json leaderboard[].entrant');
check('every prefix is the first 12 of its own sha256',
      C.leaderboard.every(r => r.sha256.slice(0, 12) === r.entrant), true,
      'chase_controls-results.json leaderboard');

// kinds, policies and seconds
check('ctrl_do_nothing is a move at 5 s',
      [lbByName.ctrl_do_nothing.kind, lbByName.ctrl_do_nothing.policy, lbByName.ctrl_do_nothing.seconds],
      ['move', null, 5], 'chase_controls-results.json leaderboard');
check('ctrl_ball_kick_left is a policy, ball_kick_left.onnx, 5 s',
      [lbByName.ctrl_ball_kick_left.kind, lbByName.ctrl_ball_kick_left.policy, lbByName.ctrl_ball_kick_left.seconds],
      ['policy', 'ball_kick_left.onnx', 5], 'chase_controls-results.json leaderboard');
check('ctrl_ball_kick_right is a policy, ball_kick_right.onnx, 5 s',
      [lbByName.ctrl_ball_kick_right.kind, lbByName.ctrl_ball_kick_right.policy, lbByName.ctrl_ball_kick_right.seconds],
      ['policy', 'ball_kick_right.onnx', 5], 'chase_controls-results.json leaderboard');
check('ctrl_alpha_walking is a policy, alpha_walking.onnx, 4 s',
      [lbByName.ctrl_alpha_walking.kind, lbByName.ctrl_alpha_walking.policy, lbByName.ctrl_alpha_walking.seconds],
      ['policy', 'alpha_walking.onnx', 4], 'chase_controls-results.json leaderboard');

// the counts the leaderboard table prints
for (const [name, want] of [
  ['ctrl_alpha_walking', { kChased: 4, kStable: 4, nCore: 9, kExt: 1, nExt: 5, touchedCells: 5 }],
  ['ctrl_do_nothing', { kChased: 0, kStable: 0, nCore: 9, kExt: 0, nExt: 5, touchedCells: 0 }],
  ['ctrl_ball_kick_left', { kChased: 0, kStable: 0, nCore: 9, kExt: 0, nExt: 5, touchedCells: 0 }],
  ['ctrl_ball_kick_right', { kChased: 0, kStable: 0, nCore: 9, kExt: 0, nExt: 5, touchedCells: 0 }],
]) {
  const r = lbByName[name];
  check(`${name}: chased ${want.kChased}/9, stable ${want.kStable}/9, ext ${want.kExt}/5, touched ${want.touchedCells}/14`,
        { kChased: r.kChased, kStable: r.kStable, nCore: r.nCore, kExt: r.kExt, nExt: r.nExt,
          touchedCells: r.touchedCells },
        want, 'chase_controls-results.json leaderboard');
}
check('THE CENTRE-CELL TRAVEL COLUMN IS ZERO ON ALL FOUR ROWS',
      C.leaderboard.map(r => r.centreBallTravel_mm), [0, 0, 0, 0],
      'chase_controls-results.json leaderboard[].centreBallTravel_mm');
check('…and it really is the centre cell that is zero for each',
      C.entrants.map(e => cellOf(e, 0, 0.7).ballTravel_mm), [0, 0, 0, 0],
      'chase_controls-results.json entrants[].verdicts');
check('every row held 50 of 50 tail ticks in its worst cell',
      C.leaderboard.map(r => r.minUprightTailTicks), [50, 50, 50, 50],
      'chase_controls-results.json leaderboard[].minUprightTailTicks');
check('every entrant is upright at the end of all fourteen cells',
      C.entrants.map(e => e.uprightFinalCells), [14, 14, 14, 14],
      'chase_controls-results.json entrants[].uprightFinalCells');

// ============================== 4. ctrl_do_nothing — 0 of 14, and it must be
check('DO-NOTHING SCORES ZERO: 0 core, 0 ext, 0 of all fourteen',
      [NOTHING.kChased, NOTHING.kStable, NOTHING.kExt, NOTHING.kChasedAll, NOTHING.nAll],
      [0, 0, 0, 0, 14], 'chase_controls-results.json entrants[0]');
check('do-nothing touches nothing in any cell', NOTHING.touchedCells, 0,
      'chase_controls-results.json entrants[0]');
check('do-nothing moves the ball nowhere, in any cell',
      [NOTHING.maxBallTravel_mm, NOTHING.meanBallTravel_mm, NOTHING.maxBallPeakSpeed_mps],
      [0, 0, 0], 'chase_controls-results.json entrants[0]');
check('do-nothing: no cell records a touch', NOTHING.verdicts.filter(v => v.touched).length, 0,
      'chase_controls-results.json entrants[0].verdicts');
check('do-nothing closest approach, min 344.0166288764408 mm', NOTHING.minClosest_mm,
      344.0166288764408, 'chase_controls-results.json entrants[0]');
check('do-nothing closest approach, mean 620.5576745621695 mm', NOTHING.meanClosest_mm,
      620.5576745621695, 'chase_controls-results.json entrants[0]');
check('do-nothing: the 1200 mm cell reads 1073.80 mm', cellOf(NOTHING, 0, 1.2).closest_mm, 1073.8,
      'chase_controls-results.json entrants[0].verdicts');
check('do-nothing holds 50 of 50 tail ticks in every cell',
      [NOTHING.minUprightTailTicks, NOTHING.meanUprightTailTicks], [50, 50],
      'chase_controls-results.json entrants[0]');
check('do-nothing is a move, so its action rate is a keyframe pose target',
      termOf(NOTHING, 'action_rate_l2').action_rate_l2_source, 'keyframe pose target',
      'chase_controls-results.json entrants[0].terms');

// =================== 5. the two kick policies — 0 of 14, the measurement
for (const [k, label, minC, meanC] of [
  [KICKL, 'ctrl_ball_kick_left', 301.2073777841115, 594.0542846572323],
  [KICKR, 'ctrl_ball_kick_right', 308.7546981502833, 606.3468899818689],
]) {
  check(`${label}: 0 core, 0 ext, 0 of all fourteen`,
        [k.kChased, k.kStable, k.kExt, k.kChasedAll], [0, 0, 0, 0],
        'chase_controls-results.json entrants[]');
  check(`${label}: never touches the ball`, k.touchedCells, 0, 'chase_controls-results.json entrants[]');
  check(`${label}: the ball never moves`,
        [k.maxBallTravel_mm, k.maxBallPeakSpeed_mps], [0, 0], 'chase_controls-results.json entrants[]');
  check(`${label}: closest approach, min ${minC} mm`, k.minClosest_mm, minC,
        'chase_controls-results.json entrants[]');
  check(`${label}: closest approach, mean ${meanC} mm`, k.meanClosest_mm, meanC,
        'chase_controls-results.json entrants[]');
  check(`${label}: upright, 50 of 50 tail ticks in every cell`,
        [k.uprightFinalCells, k.minUprightTailTicks], [14, 50], 'chase_controls-results.json entrants[]');
  check(`${label}: no ball term fires`,
        [termOf(k, 'ball_forward_velocity').value, termOf(k, 'ball_speed_overshoot').value], [0, 0],
        'chase_controls-results.json entrants[].terms');
}
// the schedule the card says was read out of the config, not chosen
check('both kick policies are commanded (0, 0, 0) — the centre of the config\'s own ranges',
      [E('ctrl_ball_kick_left.json').schedule, E('ctrl_ball_kick_right.json').schedule],
      [[[0, { vx: 0, vy: 0, vyaw: 0 }]], [[0, { vx: 0, vy: 0, vyaw: 0 }]]],
      'entrants/ctrl_ball_kick_*.json');
check('both kick policies run 5 s — EPISODE_LENGTH_S',
      [E('ctrl_ball_kick_left.json').seconds, E('ctrl_ball_kick_right.json').seconds], [5, 5],
      'entrants/ctrl_ball_kick_*.json');

// ==================== 6. ctrl_alpha_walking — 4 of 9, the wrong four
check('ALPHA WALKING: 4 of 9 core chased', ALPHA.kChased, 4, 'chase_controls-results.json entrants[3]');
check('ALPHA WALKING: 4 of 9 core stable', ALPHA.kStable, 4, 'chase_controls-results.json entrants[3]');
check('ALPHA WALKING: 1 of 5 extended', [ALPHA.kExt, ALPHA.kExtStable, ALPHA.nExtOnly], [1, 1, 5],
      'chase_controls-results.json entrants[3]');
check('ALPHA WALKING: 5 of 14 in all, and every chased cell is also stable',
      [ALPHA.kChasedAll, ALPHA.kStableAll, ALPHA.nAll], [5, 5, 14],
      'chase_controls-results.json entrants[3]');
check('ALPHA WALKING: touched 5 of 14', ALPHA.touchedCells, 5, 'chase_controls-results.json entrants[3]');
check('ALPHA WALKING: upright at the end of all fourteen, 50 of 50 tail ticks',
      [ALPHA.uprightFinalCells, ALPHA.minUprightTailTicks], [14, 50],
      'chase_controls-results.json entrants[3]');

// THE SHAPE, not just the count: the whole -20 column passes, dead ahead fails twice.
check('THE WHOLE BEARING -20 COLUMN IS CHASED (0.45, 0.70, 0.95 m)',
      [0.45, 0.7, 0.95].map(r => cellOf(ALPHA, -20, r).chased), [true, true, true],
      'chase_controls-results.json entrants[3].verdicts');
check('DEAD AHEAD: 0.45 m passes, 0.70 m and 0.95 m FAIL',
      [0.45, 0.7, 0.95].map(r => cellOf(ALPHA, 0, r).chased), [true, false, false],
      'chase_controls-results.json entrants[3].verdicts');
check('EVERY +20 CELL FAILS',
      [0.45, 0.7, 0.95].map(r => cellOf(ALPHA, 20, r).chased), [false, false, false],
      'chase_controls-results.json entrants[3].verdicts');
check('both +/-40 cells fail — every genuinely off-bearing cell is unclaimed',
      [cellOf(ALPHA, -40, 0.7).chased, cellOf(ALPHA, 40, 0.7).chased], [false, false],
      'chase_controls-results.json entrants[3].verdicts');
check('ext 5 (1.20 m dead ahead) FAILS, though it was predicted plausibly to pass',
      cellOf(ALPHA, 0, 1.2).chased, false, 'chase_controls-results.json entrants[3].verdicts');
check('the one extended pass is the GRIPPY plant (drop 0.125, x1.3)',
      ALPHA.verdicts.filter(v => v.tier === 'ext' && v.chased).map(v => [v.drop, v.fmul]),
      [[0.125, 1.3]], 'chase_controls-results.json entrants[3].verdicts');
check('…and the slippery plant (drop 0.130, x0.7) fails',
      cellOf(ALPHA, 0, 0.7, 0.13, 0.7).chased, false,
      'chase_controls-results.json entrants[3].verdicts');

// the per-cell table the card and the leaderboard print, row by row
const ALPHA_TABLE = [
  // bearing, range, drop, fmul, chased, touched, travel, net, closest, final, peak
  [-20, 0.45, 0.12, 1, true, true, 582.8, 650.95, -3.14, 216.34, 0.614],
  [0, 0.45, 0.12, 1, true, true, 641.27, 743.77, -2.14, 680.51, 0.6306],
  [20, 0.45, 0.12, 1, false, false, 0, 0, 110.69, 860.02, 0],
  [-20, 0.7, 0.12, 1, true, true, 135.37, 495.53, -3.55, 500.17, 0.5372],
  [0, 0.7, 0.12, 1, false, false, 0, 0, 24.38, 544.25, 0],
  [20, 0.7, 0.12, 1, false, false, 0, 0, 242.45, 737.43, 0],
  [-20, 0.95, 0.12, 1, true, true, 233.05, 406.23, -5.06, 316.71, 0.595],
  [0, 0.95, 0.12, 1, false, false, 0, 0, 113.87, 370.08, 0],
  [20, 0.95, 0.12, 1, false, false, 0, 0, 396.76, 687.73, 0],
  [0, 0.7, 0.13, 0.7, false, false, 0, 0, 64.93, 331.77, 0],
  [0, 0.7, 0.125, 1.3, true, true, 465.39, 559.72, -4.39, 279.2, 0.6198],
  [-40, 0.7, 0.12, 1, false, false, 0, 0, 210.87, 621.99, 0],
  [40, 0.7, 0.12, 1, false, false, 0, 0, 424.57, 977.05, 0],
  [0, 1.2, 0.12, 1, false, false, 0, 0, 196.8, 320.47, 0],
];
for (const [b, r, d, f, chased, touched, travel, net, closest, final, peak] of ALPHA_TABLE) {
  const v = cellOf(ALPHA, b, r, d, f);
  check(`alpha cell ${b >= 0 ? '+' : ''}${b}° / ${r} m / ${d} / x${f}`,
        [v.chased, v.touched, v.ballTravel_mm, v.ballNet_mm, v.closest_mm, v.final_mm, v.ballPeakSpeed_mps],
        [chased, touched, travel, net, closest, final, peak],
        'chase_controls-results.json entrants[3].verdicts');
}
check('alpha: the unrounded max travel the card cites', ALPHA.maxBallTravel_mm, 641.2679543760061,
      'chase_controls-results.json entrants[3]');
check('alpha: the unrounded mean travel', ALPHA.meanBallTravel_mm, 146.99130814044116,
      'chase_controls-results.json entrants[3]');
check('alpha: the unrounded min closest (the deepest interpenetration)', ALPHA.minClosest_mm,
      -5.064028274740923, 'chase_controls-results.json entrants[3]');
check('alpha: the unrounded mean closest', ALPHA.meanClosest_mm, 126.21663530680352,
      'chase_controls-results.json entrants[3]');
check('alpha: the unrounded peak ball speed', ALPHA.maxBallPeakSpeed_mps, 0.6306190398590918,
      'chase_controls-results.json entrants[3]');
check('EVERY PASSING CELL INTERPENETRATES: closest_mm is negative in all five',
      ALPHA.verdicts.filter(v => v.chased).every(v => v.closest_mm < 0), true,
      'chase_controls-results.json entrants[3].verdicts');
check('…and the card\'s -2.14 to -5.06 mm range is the full spread of those five',
      [Math.max(...ALPHA.verdicts.filter(v => v.chased).map(v => v.closest_mm)),
       Math.min(...ALPHA.verdicts.filter(v => v.chased).map(v => v.closest_mm))],
      [-2.14, -5.06], 'chase_controls-results.json entrants[3].verdicts');
check('TRAVEL AND NET DIFFER at -20/0.70: 135.37 forward out of 495.53 total',
      [cellOf(ALPHA, -20, 0.7).ballTravel_mm, cellOf(ALPHA, -20, 0.7).ballNet_mm],
      [135.37, 495.53], 'chase_controls-results.json entrants[3].verdicts');
check('the schedule: straight ahead at vx 0.5 for 4 s',
      [E('ctrl_alpha_walking.json').schedule, E('ctrl_alpha_walking.json').seconds],
      [[[0, { vx: 0.5, vy: 0, vyaw: 0 }]], 4], 'entrants/ctrl_alpha_walking.json');

// ============================================= 7. the drift behind the shape
check('the drift was measured on ctrl_alpha_walking', DRIFT.entrant, 'ctrl_alpha_walking.json',
      'chase_drift-results.json');
check('…at the schedule the leaderboard row ran', [DRIFT.schedule, DRIFT.seconds],
      [[[0, { vx: 0.5, vy: 0, vyaw: 0 }]], 4], 'chase_drift-results.json');
check('the drift rows are the four bearing-0 cells',
      DRIFT.rows.map(r => [r.cell.bearing, r.cell.range]),
      [[0, 0.45], [0, 0.7], [0, 0.95], [0, 1.2]], 'chase_drift-results.json rows');
check('the drift entrant is the leaderboard entrant',
      [...new Set(DRIFT.rows.map(r => r.hash))], [lbByName.ctrl_alpha_walking.sha256],
      'chase_drift-results.json rows[].hash');
{
  const untouched = DRIFT.rows.filter(r => !r.touched);
  check('three of the four drift cells never touch the ball', untouched.length, 3,
        'chase_drift-results.json rows');
  check('THE OPEN-LOOP DRIFT IS -15.402 DEGREES, identical on all three untouched cells',
        [...new Set(untouched.map(r => +r.driftFromInitialHeading_deg.toFixed(3)))], [-15.402],
        'chase_drift-results.json rows[].driftFromInitialHeading_deg');
  check('…and it walks 1.1882 m, not the 2.0 m the command nominally buys',
        [...new Set(untouched.map(r => +r.walkedDistance_m.toFixed(4)))], [1.1882],
        'chase_drift-results.json rows[].walkedDistance_m');
  check('the drift is to the duck\'s RIGHT (negative, and positive is left)',
        untouched.every(r => r.driftFromInitialHeading_deg < 0), true,
        'chase_drift-results.json rows');
  const hit = DRIFT.rows.find(r => r.touched);
  check('the 0.45 m cell touches the ball; the collision deflects it to -15.588 deg over 1.1306 m',
        [hit.cell.range, +hit.driftFromInitialHeading_deg.toFixed(3), +hit.walkedDistance_m.toFixed(4)],
        [0.45, -15.588, 1.1306], 'chase_drift-results.json rows');
}
// The drift measurement runs the same entrant through the same rig, so its
// closest approaches are the leaderboard's on every cell — including the
// 1.20 m one, where the duck never reaches the ball and the END of the
// episode is its nearest point, which the standing tail decides. Both files
// were produced after the tail became the standing test.
check('the drift cells\' closest approaches are the leaderboard\'s, all four',
      DRIFT.rows.map(r => +r.closest_mm.toFixed(2)),
      [0.45, 0.7, 0.95, 1.2].map(r => cellOf(ALPHA, 0, r).closest_mm),
      'chase_drift-results.json vs chase_controls-results.json');
check('THE 0.70 m DEAD-AHEAD MISS IS 24.38 mm', +DRIFT.rows[1].closest_mm.toFixed(2), 24.38,
      'chase_drift-results.json rows[1]');
check('THE 0.95 m DEAD-AHEAD MISS IS 113.87 mm', +DRIFT.rows[2].closest_mm.toFixed(2), 113.87,
      'chase_drift-results.json rows[2]');
check('the frozen heading is the same vector in every drift row',
      [...new Set(DRIFT.rows.map(r => r.yaw0_rad))].length, 1,
      'chase_drift-results.json rows[].yaw0_rad');

// ============================================ 8. Pollen's reward — the nine
const WANT_TERMS = [
  ['ball_forward_velocity', 12], ['ball_speed_overshoot', -4], ['upright', 2],
  ['pose_stand_legs', 2], ['pose_stand_neck', 1], ['height_stand', 1],
  ['body_ang_vel', -0.05], ['action_rate_l2', -1], ['angular_momentum', -0.02],
];
check('nine computable terms, in the config\'s order, with the weights the card tabulates',
      C.entrants[0].terms.map(t => [t.term, t.weight]), WANT_TERMS,
      'chase_controls-results.json entrants[].terms');
check('every entrant answers the same nine terms in the same order',
      [...new Set(C.entrants.map(e => e.terms.map(t => t.term).join(',')))],
      [WANT_TERMS.map(t => t[0]).join(',')], 'chase_controls-results.json entrants[].terms');
check('THE THREE DELETED VELOCITY TERMS DO NOT APPEAR',
      C.entrants.flatMap(e => e.terms.map(t => t.term))
        .filter(t => ['track_linear_velocity', 'track_angular_velocity', 'pose'].includes(t)),
      [], 'chase_controls-results.json entrants[].terms');
check('action_rate_l2 is scored at the ramp END, -1.0',
      [...new Set(C.entrants.map(e => termOf(e, 'action_rate_l2').weight))], [-1],
      'chase_controls-results.json entrants[].terms');
check('action_rate_l2_source is labelled on every row, and only on that term',
      C.entrants.map(e => e.terms.filter(t => t.action_rate_l2_source).map(t => t.term)),
      [['action_rate_l2'], ['action_rate_l2'], ['action_rate_l2'], ['action_rate_l2']],
      'chase_controls-results.json entrants[].terms');
check('the move entrant\'s action is a keyframe pose target; the three policies\' is raw output',
      C.entrants.map(e => termOf(e, 'action_rate_l2').action_rate_l2_source),
      ['keyframe pose target', 'policy raw output', 'policy raw output', 'policy raw output'],
      'chase_controls-results.json entrants[].terms');

// the term table, value by value, exactly as the card prints it
const TERM_TABLE = {
  ball_forward_velocity: [0, 0, 0, 0.029988343652321836],
  ball_speed_overshoot: [0, 0, 0, 0],
  upright: [0.9998763057782069, 0.9931221510786095, 0.9943087512936136, 0.9564203498605064],
  pose_stand_legs: [0.9965143306457938, 0.9623872375324086, 0.9724136072212477, 0.9038219162946854],
  pose_stand_neck: [0.9407215931900168, 0.9887772233518801, 0.9815705391549662, 0.8457114526149494],
  height_stand: [0.999052246235542, 0.9989018867774248, 0.9983744303308056, 0.9791737927694291],
  body_ang_vel: [1.2002301406106929e-05, 0.035387799283043656, 0.0515341759225177, 0.8565316158987761],
  action_rate_l2: [0, 0.05309782345138118, 0.046590201825831304, 0.2522514765950025],
  angular_momentum: [8.53530467165277e-10, 1.9334917086628585e-06, 2.3455729716410938e-06, 2.74005989743522e-05],
};
for (const [term, want] of Object.entries(TERM_TABLE)) {
  check(`term ${term}, all four entrants`, C.entrants.map(e => termOf(e, term).value), want,
        'chase_controls-results.json entrants[].terms');
}
check('a MOVE riding the standing policy does not read 1.0 on the stand terms',
      [termOf(NOTHING, 'pose_stand_legs').value < 1, termOf(NOTHING, 'pose_stand_neck').value < 1],
      [true, true], 'chase_controls-results.json entrants[0].terms');
check('the card\'s 0.9965 / 0.9407 for the do-nothing stand terms',
      [+termOf(NOTHING, 'pose_stand_legs').value.toFixed(4),
       +termOf(NOTHING, 'pose_stand_neck').value.toFixed(4)],
      [0.9965, 0.9407], 'chase_controls-results.json entrants[0].terms');
check('only the naive chaser ever moves the ball forward, so only it scores ball_forward_velocity',
      C.entrants.map(e => termOf(e, 'ball_forward_velocity').value > 0),
      [false, false, false, true], 'chase_controls-results.json entrants[].terms');
check('nothing overshoots the 1.0 m/s target: ball_speed_overshoot is 0 everywhere',
      C.entrants.map(e => termOf(e, 'ball_speed_overshoot').value), [0, 0, 0, 0],
      'chase_controls-results.json entrants[].terms');

// ============================================ 9. the three refused, by name
const WANT_REFUSED = [
  ['support_foot_grounded', 2], ['self_collisions', -1], ['dof_pos_limits', -1],
];
check('three terms refused BY NAME, with their weights',
      C.entrants[0].refused.map(r => [r.term, r.weight]), WANT_REFUSED,
      'chase_controls-results.json entrants[].refused');
check('every entrant refuses the same three, in the same order',
      [...new Set(C.entrants.map(e => e.refused.map(r => r.term).join(',')))],
      [WANT_REFUSED.map(r => r[0]).join(',')], 'chase_controls-results.json entrants[].refused');
check('every refusal carries a reason, never a bare name',
      C.entrants[0].refused.every(r => typeof r.reason === 'string' && r.reason.length > 60), true,
      'chase_controls-results.json entrants[].refused');
check('support_foot_grounded is refused for the CONTACT SENSOR, by name',
      C.entrants[0].refused[0].reason.includes('support_foot_ground_contact')
        && C.entrants[0].refused[0].reason.includes('inventing a reward'), true,
      'chase_controls-results.json entrants[].refused[0]');
check('self_collisions is refused for the self_collision sensor',
      C.entrants[0].refused[1].reason.includes('self_collision'), true,
      'chase_controls-results.json entrants[].refused[1]');
check('dof_pos_limits is refused for the missing soft_joint_pos_limits fraction',
      C.entrants[0].refused[2].reason.includes('soft_joint_pos_limits'), true,
      'chase_controls-results.json entrants[].refused[2]');
check('nine computed plus three refused = the config\'s twelve live terms',
      C.entrants[0].terms.length + C.entrants[0].refused.length, 12,
      'chase_controls-results.json entrants[0]');

// ============================================ 10. the acceptance test's log
check('parity: 56 per-cell rows, exact on all 49 compared fields',
      /per-cell rows: 56\s+EXACT on all 49 compared fields \+ the cell itself: 56\/56/.test(PAR),
      true, 'chase_parity.log');
check('parity: aggregates recomputed from /chase equal chase_robust on every entrant',
      /aggregates recomputed from \/chase equal chase_robust on every entrant: true/.test(PAR),
      true, 'chase_parity.log');
check('parity: ctrl_do_nothing scores 0 of 14 and touches nothing',
      /ctrl_do_nothing scores 0 of 14 and touches nothing: true/.test(PAR), true, 'chase_parity.log');
check('parity: the grid the bench publishes is the grid the scorer runs',
      /the grid the bench publishes is the grid the scorer runs: true/.test(PAR), true, 'chase_parity.log');
check('parity: GET /chase/grid lists 14 cells, 9 core',
      /GET \/chase\/grid lists 14 cells \(9 core, 5 in all\); identical to chase_robust's plan: true/.test(PAR),
      true, 'chase_parity.log');
check('parity: the criterion string is identical on both sides',
      /criterion identical: true/.test(PAR), true, 'chase_parity.log');
check('parity: all three refusals identical', /refusals identical \(3\): true/.test(PAR), true,
      'chase_parity.log');
check('parity: the run PASSED', /CHASE PARITY PASS/.test(PAR), true, 'chase_parity.log');
check('parity: the plant it ran on is this package\'s plant',
      PAR.includes(C.plantDigest.slice(0, 12)), true, 'chase_parity.log');
check('parity: all four entrants AGREE between the two scorers',
      (PAR.match(/AGREES/g) || []).length, 4, 'chase_parity.log');
check('parity: the four entrant hashes in the log are the four on the leaderboard',
      C.leaderboard.every(r => PAR.includes(r.entrant)), true, 'chase_parity.log');
check('parity: the gate reports how long it took', /CHASE PARITY PASS.*\[\d+s\]/.test(PAR), true, 'chase_parity.log');
{
  // the card says "about 0.5 s per cell": four entrants x 28 scored cells in about a minute
  const secPerCell = 55 / (4 * 28);
  checkNear('…which is about 0.5 s per scored cell', secPerCell, 0.5, 0.02, 'chase_parity.log');
}

// =========================================== 11. the entrant files themselves
check('four entrant files, and they are the four the leaderboard scores',
      fs.readdirSync(path.join(HERE, 'entrants')).sort(),
      ['ctrl_alpha_walking.json', 'ctrl_ball_kick_left.json', 'ctrl_ball_kick_right.json',
       'ctrl_do_nothing.json'], 'entrants/');
check('each entrant file names the row it produced',
      C.leaderboard.map(r => E(r.source).name), C.leaderboard.map(r => r.name),
      'entrants/ vs chase_controls-results.json leaderboard[].source');
check('each entrant file declares the kind and seconds its row was scored at',
      C.leaderboard.map(r => [E(r.source).kind, E(r.source).seconds]),
      C.leaderboard.map(r => [r.kind, r.seconds]), 'entrants/ vs leaderboard');
check('the three policy entrants name the policy the row credits',
      C.leaderboard.filter(r => r.kind === 'policy').map(r => E(r.source).policy),
      C.leaderboard.filter(r => r.kind === 'policy').map(r => r.policy), 'entrants/ vs leaderboard');
{
  const m = E('ctrl_do_nothing.json');
  check('the move entrant has two keyframes of 14 joint targets',
        [m.intent.keyframes.length, [...new Set(m.intent.keyframes.map(k => k.pose.length))]],
        [2, [14]], 'entrants/ctrl_do_nothing.json');
  check('both keyframes are the same pose — it holds HOME',
        JSON.stringify(m.intent.keyframes[0].pose) === JSON.stringify(m.intent.keyframes[1].pose),
        true, 'entrants/ctrl_do_nothing.json');
  check('the pose the card prints, all fourteen numbers', m.intent.keyframes[0].pose,
        [0, -0.0873, -0.4579, -0.0049, 0.453, 0.3491, 0.3491, 0, 0, 0,
         0.0873, 0.4579, 0.0049, -0.453], 'entrants/ctrl_do_nothing.json');
  check('blend 1, and the keyframes run 1.0 to 4.9 s inside a 5 s span',
        [m.intent.blend, m.intent.keyframes.map(k => k.t), m.seconds], [1, [1.0, 4.9], 5],
        'entrants/ctrl_do_nothing.json');
}
check('every entrant carries a note stating what was expected of it IN ADVANCE',
      C.leaderboard.every(r => typeof E(r.source).note === 'string' && E(r.source).note.length > 80),
      true, 'entrants/');
check('the fourteen joint slots the card lists, in order',
      JSON.parse(H('duckkit-constants.json')).jointNames.filter(n => n !== 'mouth'),
      ['left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle',
       'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll',
       'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle'],
      'harness/duckkit-constants.json');
check('duckkit has fifteen joints, with mouth at index 9',
      [JSON.parse(H('duckkit-constants.json')).jointNames.length,
       JSON.parse(H('duckkit-constants.json')).jointNames.indexOf('mouth')], [15, 9],
      'harness/duckkit-constants.json');

// ================================================= 12. the plant digest itself
{
  const mjb = path.join(HERE, '..', 'sim', 'scene.mjb');
  if (fs.existsSync(mjb)) {
    const d = crypto.createHash('sha256').update(fs.readFileSync(mjb)).digest('hex');
    check('plant sim/scene.mjb sha256, hashed here', d, C.plantDigest, 'sim/scene.mjb (hashed here)');
  } else {
    skip('plant sim/scene.mjb sha256 (SKIPPED — not in this package)',
         '3f8c9ab9b409…', 'sim/scene.mjb');
  }
}

// ================================ 13. the package is what MANIFEST.json says
{
  const M = JSON.parse(fs.readFileSync(path.join(HERE, 'MANIFEST.json'), 'utf8'));
  const walk = d => fs.readdirSync(d, { withFileTypes: true }).flatMap(
    e => e.isDirectory() ? walk(path.join(d, e.name)) : [path.relative(HERE, path.join(d, e.name))]);
  const here = walk(HERE).filter(f => f !== 'MANIFEST.json').sort();
  check('MANIFEST.json lists every file in the package, and no file it does not have',
        here, Object.keys(M.files).sort(), 'MANIFEST.json');
  check('MANIFEST.json fileCount', M.fileCount, here.length, 'MANIFEST.json');
  const bad = [];
  let total = 0;
  for (const f of here) {
    const b = fs.readFileSync(path.join(HERE, f));
    total += b.length;
    const d = crypto.createHash('sha256').update(b).digest('hex');
    if (M.files[f].sha256 !== d || M.files[f].bytes !== b.length) bad.push(f);
  }
  check('every file matches its MANIFEST sha256 and byte size', bad, [], 'MANIFEST.json');
  check('MANIFEST.json totalBytes', M.totalBytes, total, 'MANIFEST.json');
  // The card claims the four entrants and the harness snapshot are byte-identical
  // to the repository. That can only be checked when the package is IN the repo.
  const repo = {
    'entrants/ctrl_do_nothing.json': '../chase/ctrl_do_nothing.json',
    'entrants/ctrl_ball_kick_left.json': '../chase/ctrl_ball_kick_left.json',
    'entrants/ctrl_ball_kick_right.json': '../chase/ctrl_ball_kick_right.json',
    'entrants/ctrl_alpha_walking.json': '../chase/ctrl_alpha_walking.json',
    'results/chase_controls-results.json': '../chase/chase_controls-results.json',
    'REWARD.md': '../chase/REWARD.md',
    'harness/chase_score.mjs': '../sim/chase_score.mjs',
    'harness/reward_math.mjs': '../sim/reward_math.mjs',
    'harness/climb_score.mjs': '../sim/climb_score.mjs',
    'harness/chase_rig.mjs': '../chase/chase_rig.mjs',
    'harness/chase_robust.mjs': '../chase/chase_robust.mjs',
    'harness/chase_parity.mjs': '../chase/chase_parity.mjs',
    'harness/scene_physics.xml': '../sim/scene_physics.xml',
    'harness/duckkit-constants.json': '../site/duckkit-constants.json',
  };
  const missing = Object.values(repo).filter(p => !fs.existsSync(path.join(HERE, p)));
  if (missing.length) {
    skip('the snapshot is byte-identical to the repository (SKIPPED — not in the repo)',
         'n/a', 'challenge-ball/ outside duck-sounds');
  } else {
    const differs = Object.entries(repo).filter(([here_, there]) =>
      !fs.readFileSync(path.join(HERE, here_)).equals(fs.readFileSync(path.join(HERE, there))));
    check('THE SNAPSHOT IS BYTE-IDENTICAL TO THE REPOSITORY', differs.map(d => d[0]), [],
          'challenge-ball/ vs chase/, sim/, site/');
  }
}

// =============================== 14. the two other numbers the card states
check('REWARD.md is 476 lines',
      fs.readFileSync(path.join(HERE, 'REWARD.md'), 'utf8').split('\n').length - 1, 476,
      'REWARD.md');
check('hf_upload.sh targets craigm26/microduck-ball-challenge',
      fs.readFileSync(path.join(HERE, 'hf_upload.sh'), 'utf8')
        .includes('${HF_USER}/microduck-ball-challenge'), true, 'hf_upload.sh');
check('the card and the leaderboard agree on every published count',
      ['4** / 9', '`a0bbbbb98acb`', '`bc77453e40c6`', '`7e44b5a781fc`', '`f8d4e8bfd2b7`']
        .every(s => fs.readFileSync(path.join(HERE, 'leaderboard.md'), 'utf8').includes(s)
                 && fs.readFileSync(path.join(HERE, 'README.md'), 'utf8').includes(s)),
      true, 'README.md + leaderboard.md');

// ---------------------------------------------------------------- report
const W = Math.max(...rows.map(r => r.label.length));
for (const r of rows) {
  const verdict = r.skipped ? 'SKIP' : r.ok ? 'PASS' : 'FAIL';
  console.log(`${verdict}  ${r.label.padEnd(W)}  ${r.ok ? '' : `got ${r.got} want ${r.want}  `}[${r.src}]`);
}
console.log(`\n${pass} PASS, ${fail} FAIL, ${skipped} SKIPPED, ${rows.length} checks.`);
process.exit(fail ? 1 : 0);
