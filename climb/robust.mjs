// robust.mjs — THE ROUND-3 SCORER. One copy, imported by every family.
//
// Round 2 reported single-cell results and then discovered, in audit_r2, that
// a 40 mm clear survived 1 of 7 perturbations and a 60 mm clear 3 of 7. A move
// that passes once is a coincidence. So round 3 scores a SAVED FILE over a
// grid and reports "cleared k of N". ROUND 4 extended that grid to 14 cells
// and added an upright-through-the-tail term; the core 9 below are unchanged:
//
//   rise  in { h-10 mm, h, h+10 mm }
//   plant in { (drop 0.120, friction x1.0),      -- nominal
//              (drop 0.130, friction x0.7),      -- higher fall, slippery
//              (drop 0.125, friction x1.3) }     -- higher fall, grippy
//
// THE EPISODE LOOP BELOW IS audit_r2.mjs's go(), which is itself a verbatim
// copy of rig3.mjs runEpisodeRaw() (7/7 EXACT parity, climb/audit_r2-results
// .json). sim/ is off-limits to edit and importing rig3 for the loop is not
// possible (its loop is not exported), so this is the ONE round-3 copy; every
// family imports scoreRobust from here rather than making a fourth.
//
// Differences from rig3's loop, all of them knobs, none of them physics:
//   - drop      : spawn height (rig3 hard-codes 0.12)
//   - fmul      : multiplier on foot geom friction[0] (nominal read from model)
//   - isolate   : step-step collision isolation. The shipped flight is now
//                 isolated by site/stairs.js findStairJoints(); this file
//                 captures the SHIPPED affinity with {isolate:false} and sets
//                 it per run, so isolate:true reproduces the shipped plant
//                 exactly. Phase P proves it against rig3.scoreSaved.
//
// criteria() and reward() are IMPORTED from rig3.mjs — not copied — so the
// verdict and the shaped reward cannot drift from the instrument.
//
// Contacts via mj_geomDistance only; data.contact.get(i) leaks the WASM heap.
//
// Run from sim/ for the parity phase:
//   cd ~/projects/duck-sounds/sim && node ../climb/robust.mjs
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { makeLoop } from '../site/duckloop.mjs';
import { STAIR_Y } from '../site/stairs.js';
// ================================================================== THE MOVE
// THE EPISODE LOOP IS NO LONGER IN THIS FILE EITHER. go() was audit_r2.mjs's
// copy of rig3.mjs runEpisodeRaw(), and this file's own header called it "the
// ONE round-3 copy" — which was true right up until the bench needed a fourth
// so a phone could score a move. All of it now lives once, in
// sim/climb_score.mjs, and rig3.mjs, this file and duckbench-core.mjs's /climb
// call the same function. The knobs this file added to that loop — drop, fmul,
// isolate, stepCount — went with it, and they are still knobs: at drop 0.120,
// fmul 1.0 and isolate on, the shared episode IS rig3's, which is what PHASE P
// at the bottom of this file still measures.
//
// WHAT IS STILL THIS FILE'S: the 14-cell grid, the bounds enforcement, the move
// hash, the aggregation, and the CELL RECORD's exact field set — climb/
// audit_r6.mjs PHASE P3 walks every leaf of it against a byte copy of this
// file, so the shape of the answer is assembled here and not shared.
import { makeClimbRig, criteria as rig3Criteria, reward as rig3Reward,
         poseAt as sharedPoseAt,
         LATERAL as SHARED_LATERAL, RISER_X as SHARED_RISER_X,
         PLANTS as SHARED_PLANTS, DHS as SHARED_DHS,
         EXT_DHS as SHARED_EXT_DHS, EXT_PLANT as SHARED_EXT_PLANT,
         EXT_CELL_COUNT as SHARED_EXT_CELL_COUNT,
         CLEAR_BONUS as SHARED_CLEAR_BONUS, UPRIGHT_BONUS as SHARED_UPRIGHT_BONUS,
         UPRIGHT_TAIL_MIN as SHARED_UPRIGHT_TAIL_MIN,
         DECLARED_BOUNDS as SHARED_DECLARED_BOUNDS,
         checkBounds as sharedCheckBounds, checkIntent, optsOf as sharedOptsOf,
         intentHashPayload, reachedFlight as sharedReachedFlight,
         intentIsolate, intentStepCount, gridCells }
  from '../sim/climb_score.mjs';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
export const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/r.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const model = mj.MjModel.mj_loadBinary('/r.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model);
const stand = await ort.InferenceSession.create('./BEST_alpha_stand.onnx');

export const LATERAL = SHARED_LATERAL;   // 0.17 m
export const RISER_X = SHARED_RISER_X;
export { STAIR_Y };

/** THIS FILE'S RIG: its own model, its own mjData, its own onnxruntime session. */
const RIG = makeClimbRig({
  mj, model, data, D, HOME, LO, HI, buildObs, projectedGravity, command,
  tickHz: C.tickHz,
  async run(obs) {
    const r = await stand.run({ obs: new ort.Tensor('float32', obs, [1, 61]) });
    return r.actions.data;
  },
});
if (!RIG) throw new Error('scene.mjb has no stair bank: this scorer cannot score a climb');

/** The geom sets PHASE P prints, so the count in the log is the count the rig uses. */
const { JAW, LEGG, WALLN } = RIG;
/** ROUND 4, HOLE 3: every collidable geom that belongs to the DUCK. */
export const DUCKG = RIG.DUCKG;

/** rig3.mjs poseAt / climb_lib.mjs:80-86, verbatim — the shared one, over this HOME. */
export function poseAt(tr, time) { return sharedPoseAt(tr, time, HOME); }

// ---------------------------------------------------------------- the episode
/**
 * One episode, through the shared loop, recorded in THIS file's shape.
 * INTERNAL: nothing outside this file scores an in-memory track.
 */
async function go(track, o, h, { drop = 0.120, fmul = 1.0, isolate = true, stepCount = 4 } = {}) {
  const E = await RIG.runEpisode(track, o, h, 'policy', { drop, fmul, isolate, stepCount });
  const scored = E.afterTail;              // robust always scores after the 50-tick policy tail
  const atTrackEnd = E.atTrackEnd;
  const rec = {
    rise: h, x0: E.x0, scored, atTrackEnd,
    event: E.event,
    penetrationAtScore: scored.penetrationAtScore, penetrationPair: scored.penetrationPair,
    // ROUND 5, THE NEW HOLE: whole-episode penetration (settle + track + tail).
    minPenetrationEpisode: E.penetration.min, minPenetrationPair: E.penetration.pair,
    minPenetrationTick: E.penetration.tick, penetrationTicksScanned: E.penetration.ticksScanned,
    // ROUND 5, THE SERVOED LANDING. null for a file with no `servo` block.
    // The shared episode also carries the servo's base and its trace; this
    // file has never published either, and PHASE P3 walks every leaf.
    servo: E.servo ? { armed: E.servo.armed, tArm: E.servo.tArm, ticks: E.servo.ticks,
                       at: E.servo.at, onEvent: E.servo.onEvent,
                       // ROUND 6: authority asked for, and ticks of it actually run.
                       tailAuthority: E.servo.tailAuthority, tailTicksRun: E.servo.tailTicksRun } : null,
    // ROUND 6: read-only per-tail-tick record; undefined unless asked for.
    tailTrace: E.tailTrace,
    uprightTailTicks: E.uprightTailTicks, tailTicks: E.tailTicks,
    uprightTailFrac: E.uprightTailTicks / Math.max(E.tailTicks, 1),
    terminal: E.terminal,          // ROUND 4, FAMILY B: the handoff state
    crit: rig3Criteria(h, scored), critAtTrackEnd: rig3Criteria(h, atTrackEnd),
    maxX: E.maxX, maxZ: E.maxZ, maxAbsDY: E.maxAbsDY, maxTq: E.maxTq,
    feetOnTreadMax: E.feetOnTreadMax, feetHighMax: E.feetHighMax,
    headFrac: E.headTicks / Math.max(E.ticks, 1),
    riserFrac: E.riserTicks / Math.max(E.ticks, 1),
    wallFrac: E.wallTicks / Math.max(E.ticks, 1),
    wallBearFrac: E.wallBearTicks / Math.max(E.ticks, 1),
    wallGain: E.wallGain === -1e9 ? null : E.wallGain,
    headOnlyFrac: E.headOnlyTicks / Math.max(E.ticks, 1),
    bothFrac: E.bothTicks / Math.max(E.ticks, 1),
    sustainFrac: E.sustainTicks / Math.max(E.ticks, 1),
    liftIntegral: E.liftIntegral,
    maxGainBoth: E.maxGainBoth === -1e9 ? null : E.maxGainBoth,
    upFrac: E.upTicks / Math.max(E.ticks, 1),
    satFrac: E.sat / Math.max(E.ctrls, 1),
    z0Settle: E.z0Settle,
    footNear: E.footNear, bothNear: E.bothNear,
    // THE TREAD DRIFT OVER THE RECORDED TICKS — this file has never counted the
    // settle in it, and rig3.mjs always has. The shared episode keeps both.
    maxTreadDriftX_mm: E.recDriftX_mm,
    minStepGap_mm: E.recGap_mm === 1e9 ? null : E.recGap_mm,
  };
  rec.reward = rig3Reward(rec);      // rig3's own reward(), imported
  return rec;
}

// ---------------------------------------------------------------- the grid
//
// EVERY CONSTANT BELOW NOW LIVES IN sim/climb_score.mjs AND IS RE-EXPORTED HERE
// UNDER THE NAME IT HAS ALWAYS HAD. Nothing about the grid moved: the core 9
// cells are the round-3 9, the extended 5 are round 4's, and the bar is still
// 45 of 50 tail ticks. What changed is that the bench's GET /climb/grid answers
// out of the same list, so a client that asks a phone for the grid gets the
// grid this scorer runs rather than a second copy of it.

/** The three plant settings. Cell 0 is the nominal plant rig3 itself uses. */
export const PLANTS = SHARED_PLANTS;
/** The three rises, as offsets from the target. */
export const DHS = SHARED_DHS;
/** Bonus added to the objective for each CORE cell cleared under 'honest'. */
export const CLEAR_BONUS = SHARED_CLEAR_BONUS;
/** ROUND 4's five extra cells: +/-5 mm nominal, and a slippery plant x 3 rises. */
export const EXT_DHS = SHARED_EXT_DHS;
export const EXT_PLANT = SHARED_EXT_PLANT;
export const EXT_CELL_COUNT = SHARED_EXT_CELL_COUNT;
/**
 * ROUND 4, HOLE 2. A cell's +CLEAR_BONUS is earned only if the duck was upright
 * for at least UPRIGHT_TAIL_MIN of the 50 tail ticks.
 */
export const UPRIGHT_TAIL_MIN = SHARED_UPRIGHT_TAIL_MIN;
export const UPRIGHT_BONUS = SHARED_UPRIGHT_BONUS;
/**
 * THE UPRIGHT TERM IS EARNED, NOT GIVEN. Measured on this plant: an ungated
 * upright-tail term pays the DO-NOTHING control the full +4.00 for standing
 * still on the floor. So a cell pays its upright credit only if the duck got
 * somewhere in it: the trunk crossed the riser line at some tick, or a foot
 * rested on a tread at some tick.
 */
const reachedFlight = sharedReachedFlight;
/** ROUND 4, HOLE 4: the DECLARED search bounds, enforced at scoring time. */
export const DECLARED_BOUNDS = SHARED_DECLARED_BOUNDS;
export const checkBounds = sharedCheckBounds;
/** The 14 cells, in the order scoreRobust runs them. GET /climb/grid's list. */
export { gridCells };

const readIntent = path => checkIntent(JSON.parse(fs.readFileSync(path, 'utf8')), path);
const optsOf = sharedOptsOf;
const isoOf = intentIsolate;
const scOf = intentStepCount;

/**
 * ROUND 4, HOLE 4. THE MOVE'S IDENTITY: sha256 over everything the episode
 * actually reads. One vector published under three rise labels hashes to one
 * value; it is one move, and the hash travels in every result row so a table
 * cannot quietly count it three times.
 *
 * The STRING that gets hashed is sim/climb_score.mjs's, because a browser
 * hashes with `crypto.subtle` and cannot share this line — only what goes into
 * it. A file written before a round has none of that round's keys and hashes to
 * exactly the value the published results carry.
 */
export function intentHash(j) {
  return crypto.createHash('sha256').update(intentHashPayload(j)).digest('hex');
}
export function intentHashOfFile(path) { return intentHash(readIntent(path)); }

/** The loud refusal. Printed to stderr AND stdout so no log can miss it. */
function shoutInvalid(path, viol, B) {
  const bar = '!'.repeat(78);
  const lines = [bar,
    `INVALID INTENT — OUT OF DECLARED BOUNDS: ${path}`,
    ...viol.map(v => `   ${v.param} = ${v.value}  is outside the declared [${v.lo}, ${v.hi}]`),
    `   declared bounds: ${JSON.stringify(B)}`,
    '   This file scores as INVALID. objective = -Infinity, k = 0. Bounds are',
    '   enforced HERE, at scoring time, not only declared in a comment.', bar];
  for (const l of lines) { console.log(l); console.error(l); }
}

/**
 * Score ONE cell of the grid, from a SAVED file. Used for cheap screening;
 * a reported result must always come from scoreRobust().
 */
export async function scoreCell(path, { rise, dh = 0, drop = 0.120, fmul = 1.0, isolate, stepCount, skipBounds = false, tailTrace = false } = {}) {
  const j = readIntent(path);
  const B = checkBounds(j);
  if (B.violations.length && !skipBounds) {
    shoutInvalid(path, B.violations, B.bounds);
    return { invalid: true, boundViolations: B.violations, bounds: B.bounds,
             sha256: intentHash(j), reward: -Infinity, crit: { honest: false, orig: false } };
  }
  const r = await go(j.keyframes, { ...optsOf(j), tailTrace }, rise + dh,
    { drop, fmul, isolate: isolate === undefined ? isoOf(j) : isolate,
      stepCount: stepCount || scOf(j) });
  r.sha256 = intentHash(j);
  return r;
}

/**
 * THE SCORER. Scores a SAVED file over the EXTENDED 14-cell grid (pass
 * {core:true} for the round-3 9 cells only).
 *
 *   kCore / nCore  cells cleared under rig3 `honest` on the round-3 9 cells,
 *                  reported unchanged so round 3's numbers stay comparable
 *   kExt  / nExt   the same count over all 14
 *   *Stable        the same cells, minus any that toppled inside the tail
 *
 *   objective     = meanReward(14) + 4*kExtStable  + 4*meanUprightTailFrac(14)
 *   objectiveCore = meanReward(9)  + 4*kCoreStable + 4*meanUprightTailFrac(9)
 *   objectiveR3   = meanReward(9)  + 4*kCore          <- exactly round 3's
 *
 * A file outside its declared bounds does not get scored: it returns
 * invalid:true with objective -Infinity, after shouting.
 */
export async function scoreRobust(path, { rise, isolate, stepCount, onCell, skipBounds = false, core = false, tailTrace = false } = {}) {
  const j = readIntent(path);
  const sha = intentHash(j);
  const B = checkBounds(j);
  if (B.violations.length && !skipBounds) {
    shoutInvalid(path, B.violations, B.bounds);
    return { source: path, rise, sha256: sha, move: sha.slice(0, 12), invalid: true,
             boundViolations: B.violations, bounds: B.bounds,
             k: 0, kCore: 0, kExt: 0, kCoreStable: 0, kExtStable: 0,
             meanReward: -Infinity, objective: -Infinity, objectiveCore: -Infinity, objectiveR3: -Infinity,
             cells: [], verdicts: [] };
  }
  const o = optsOf(j);
  const iso = isolate === undefined ? isoOf(j) : isolate;
  const sc = stepCount || scOf(j);

  // THE GRID. Core first (so a partial run is still the round-3 grid), then the
  // five extended cells. Every cell is tagged with which half it belongs to.
  const plan = [];
  for (const dh of DHS) for (const p of PLANTS) plan.push({ dh, p, tier: 'core' });
  if (!core) {
    for (const dh of EXT_DHS) plan.push({ dh, p: PLANTS[0], tier: 'ext' });
    for (const dh of DHS) plan.push({ dh, p: EXT_PLANT, tier: 'ext' });
  }

  const cells = [];
  for (const { dh, p, tier } of plan) {
    const r = await go(j.keyframes, tailTrace ? { ...o, tailTrace: true } : o, rise + dh, { drop: p.drop, fmul: p.fmul, isolate: iso, stepCount: sc });
    r.cell = { rise_mm: Math.round((rise + dh) * 1000), drop: p.drop, fmul: p.fmul, tier };
    r.sha256 = sha;
    // ROUND 4, HOLE 2: a clear only counts toward the BONUS if it also stood up
    // through the tail. `honest` itself is untouched, so kCore stays comparable.
    r.stableClear = r.crit.honest && r.uprightTailTicks >= UPRIGHT_TAIL_MIN;
    cells.push(r);
    if (onCell) onCell(r);
  }
  const coreCells = cells.filter(c => c.cell.tier === 'core');
  const kCore = coreCells.filter(c => c.crit.honest).length;
  const kExt = cells.filter(c => c.crit.honest).length;
  const kCoreStable = coreCells.filter(c => c.stableClear).length;
  const kExtStable = cells.filter(c => c.stableClear).length;
  const meanOf = (arr, f) => arr.reduce((a, c) => a + f(c), 0) / Math.max(arr.length, 1);
  const mean = f => meanOf(cells, f);
  const meanReward = mean(c => c.reward);
  const meanRewardCore = meanOf(coreCells, c => c.reward);
  const credit = c => (reachedFlight(c) ? c.uprightTailFrac : 0);
  const upFracExt = mean(credit);
  const upFracCore = meanOf(coreCells, credit);
  const upFracRawExt = mean(c => c.uprightTailFrac);
  const reachedCells = cells.filter(reachedFlight).length;
  return {
    source: path, rise, sha256: sha, move: sha.slice(0, 12), invalid: false,
    bounds: B.bounds, boundViolations: [],
    // round-3-comparable
    k: kCore, kCore, nCore: coreCells.length, kCoreStable,
    // the round-4 extended grid
    kExt, nExt: cells.length, kExtStable,
    meanReward, meanRewardCore,
    uprightTailFrac: upFracExt, uprightTailFracCore: upFracCore,
    uprightTailFracRaw: upFracRawExt, reachedFlightCells: reachedCells,
    // what the objective would be if the upright term were NOT earned
    objectiveUngatedUpright: meanReward + CLEAR_BONUS * kExtStable + UPRIGHT_BONUS * upFracRawExt,
    // THE OBJECTIVE every family optimises, over the extended grid
    objective: meanReward + CLEAR_BONUS * kExtStable + UPRIGHT_BONUS * upFracExt,
    objectiveCore: meanRewardCore + CLEAR_BONUS * kCoreStable + UPRIGHT_BONUS * upFracCore,
    // exactly what round 3 printed, so the re-baseline can show the delta
    objectiveR3: meanRewardCore + CLEAR_BONUS * kCore,
    cells,
    agg: {
      minPenetrationAtScore_mm: Math.min(...cells.map(c => c.penetrationAtScore === null ? 0 : c.penetrationAtScore)) * 1000,
      // ROUND 5: the worst penetration at ANY tick of ANY cell.
      minPenetrationEpisode_mm: Math.min(...cells.map(c => c.minPenetrationEpisode === null ? 0 : c.minPenetrationEpisode)) * 1000,
      meanUprightTailTicks: mean(c => c.uprightTailTicks),
      minUprightTailTicks: Math.min(...cells.map(c => c.uprightTailTicks)),
      lateralEscapeCells: cells.filter(c => c.maxAbsDY > LATERAL).length,
      maxZ: Math.max(...cells.map(c => c.maxZ)),
      meanMaxZ: mean(c => c.maxZ),
      meanPeakGain: mean(c => c.maxZ - c.z0Settle),
      headFrac: mean(c => c.headFrac),
      riserFrac: mean(c => c.riserFrac),
      wallFrac: mean(c => c.wallFrac),
      wallBearFrac: mean(c => c.wallBearFrac),
      bothFrac: mean(c => c.bothFrac),
      sustainFrac: mean(c => c.sustainFrac),
      liftIntegral: mean(c => c.liftIntegral),
      footNear_mm: mean(c => c.footNear) * 1000,
      bothNear_mm: mean(c => c.bothNear) * 1000,
      feetOnTreadMax: Math.max(...cells.map(c => c.feetOnTreadMax)),
      meanFeetOnTreadMax: mean(c => c.feetOnTreadMax),
      meanFeetOnTreadFinal: mean(c => c.scored.feetOnTread),
      meanAbove_mm: mean(c => c.scored.above) * 1000,
      meanX_mm: mean(c => c.scored.x) * 1000,
      upFinal: cells.filter(c => c.scored.up).length,
      satFrac: mean(c => c.satFrac),
      maxTq: Math.max(...cells.map(c => c.maxTq)),
      maxAbsDY_mm: Math.max(...cells.map(c => c.maxAbsDY)) * 1000,
      maxTreadDriftX_mm: Math.max(...cells.map(c => c.maxTreadDriftX_mm)),
    },
    verdicts: cells.map(c => ({ rise_mm: c.cell.rise_mm, drop: c.cell.drop, fmul: c.cell.fmul,
      tier: c.cell.tier, sha256: sha, move: sha.slice(0, 12),
      honest: c.crit.honest, stableClear: c.stableClear,
      eventFired: c.event ? c.event.fired : null, eventT: c.event ? c.event.tFire : null,
      eventE_mm: c.event ? c.event.e_mm : null,
      uprightTailTicks: c.uprightTailTicks, tailTicks: c.tailTicks,
      penetrationAtScore_mm: c.penetrationAtScore === null ? null : +(c.penetrationAtScore * 1000).toFixed(2),
      penetrationPair: c.penetrationPair,
      minPenetrationEpisode_mm: c.minPenetrationEpisode === null ? null : +(c.minPenetrationEpisode * 1000).toFixed(2),
      minPenetrationPair: c.minPenetrationPair, minPenetrationTick: c.minPenetrationTick,
      servoArmed: c.servo ? c.servo.armed : null, servoTicks: c.servo ? c.servo.ticks : null,
      maxAbsDY_mm: +(c.maxAbsDY * 1000).toFixed(1), lateralEpisode: c.crit.lateralEpisode,
      reward: +c.reward.toFixed(3),
      x_mm: +(c.scored.x * 1000).toFixed(1), z_mm: +(c.scored.z * 1000).toFixed(1),
      above_mm: +(c.scored.above * 1000).toFixed(1), dy_mm: +(c.scored.dy * 1000).toFixed(1),
      up: c.scored.up, feetOnTread: c.scored.feetOnTread, feetOnTreadMax: c.feetOnTreadMax,
      peakZ_mm: +(c.maxZ * 1000).toFixed(1), headFrac: +c.headFrac.toFixed(3),
      riserFrac: +c.riserFrac.toFixed(3), wallFrac: +c.wallFrac.toFixed(3) })),
  };
}

/** Write an intent object to disk and hand back the path. Nothing scores memory. */
export function saveIntent(obj, path) { fs.writeFileSync(path, JSON.stringify(obj, null, 2)); return path; }

// ================================================================== PHASE P
const isMain = process.argv[1] && process.argv[1].endsWith('robust.mjs');
if (isMain) {
  const { scoreSaved } = await import('../climb/rig3.mjs');
  console.log('=== robust.mjs PHASE P — is cell 0 (drop 0.120, x1.0, isolate on) rig3? ===');
  console.log(`duck leg/trunk geoms for the wall test: ${LEGG.length}  wall_n geom ${WALLN}  jaw geoms ${JAW.length}`);
  const cases = [['best_r2_vault_40mm.json', 0.040], ['best_r2_vault_60mm.json', 0.060],
                 ['best_r2_vault_90mm.json', 0.090], ['ctrl_on_tread_90mm.json', 0.090],
                 ['ctrl_do_nothing.json', 0.090]];
  let all = true;
  for (const [f, h] of cases) {
    const A = await scoreSaved('../climb/' + f, { rise: h, tail: 'policy' });
    const B = await scoreCell('../climb/' + f, { rise: h, isolate: true });
    const ok = A.scored.x === B.scored.x && A.scored.z === B.scored.z
      && A.scored.feetOnTread === B.scored.feetOnTread && A.crit.honest === B.crit.honest
      && A.reward === B.reward;
    if (!ok) all = false;
    console.log(`  ${f.padEnd(30)} rig3 x=${A.scored.x} z=${A.scored.z} rew=${A.reward.toFixed(4)} | robust x=${B.scored.x} z=${B.scored.z} rew=${B.reward.toFixed(4)} EXACT=${ok}`);
  }
  console.log(`  parityAll = ${all}`);
  const t0 = Date.now();
  const g = await scoreRobust('../climb/best_r2_vault_60mm.json', { rise: 0.060 });
  const dt = (Date.now() - t0) / 1000;
  console.log(`=== extended grid on best_r2_vault_60mm @60mm: kCore=${g.kCore}/${g.nCore} kExt=${g.kExt}/${g.nExt} kExtStable=${g.kExtStable}/${g.nExt} meanReward=${g.meanReward.toFixed(3)} objective=${g.objective.toFixed(3)} objectiveR3=${g.objectiveR3.toFixed(3)}  (${dt.toFixed(1)} s, ${(dt / g.nExt).toFixed(2)} s/cell) ===`);
  for (const v of g.verdicts) console.log(`   [${v.tier}] rise=${v.rise_mm} drop=${v.drop} f=${v.fmul} honest=${v.honest} stable=${v.stableClear} upTail=${v.uprightTailTicks}/${v.tailTicks} pen=${v.penetrationAtScore_mm}mm maxDY=${v.maxAbsDY_mm}mm rew=${v.reward} x=${v.x_mm} above=${v.above_mm} fot=${v.feetOnTread}`);
  const g9 = await scoreRobust('../climb/best_r2_vault_90mm.json', { rise: 0.090 });
  console.log(`=== extended grid on best_r2_vault_90mm @90mm: kCore=${g9.kCore}/${g9.nCore} kExt=${g9.kExt}/${g9.nExt} meanReward=${g9.meanReward.toFixed(3)} objective=${g9.objective.toFixed(3)} minUpTail=${g9.agg.minUprightTailTicks}/50 minPen=${g9.agg.minPenetrationAtScore_mm.toFixed(2)}mm ===`);
}
