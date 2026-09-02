// rig3 — the instrument, fixed.
//
// This is a parity-checked copy of sim/climb_lib.mjs's attempt()/replay()
// (that file's lines 99-152). sim/ is off-limits to edit and importing it runs
// its top level (a second MjModel + a second ONNX session), so this is a COPY.
// Every line of the episode staging below — cfg, spawn, the 25-tick settle, the
// +0.8 s track tail, the 50-tick hold, the ctrl blend, the clamp — is
// reproduced verbatim from climb_lib. Phase P below PROVES that: same track,
// same opts, same rise, through both harnesses, compared at full float digits
// (climb/audit_cross.mjs's method).
//
// What is DIFFERENT, and why:
//
//  (1) A LATERAL GATE. climb_lib's criterion (line 150) has no y term at all,
//      and neither does its feetUp loop (line 144: z and x only). The flight is
//      only 340 mm wide (site/stairs.js STAIR_HALF_WIDTH = 0.17), so a duck
//      that walks 426 mm off the side of it and stands on the floor scores
//      "trunk past the riser" — the podium confound at flight scale. Here
//      |y - STAIR_Y| <= 0.17 gates the criterion, the feetUp test, AND the
//      exported reward.
//
//  (2) THE FOOT-x BUG. climb_lib line 144 accepts a foot as "at/above the
//      tread" when x > 0.05. The first riser is at x = 0.12. A foot at
//      x = 0.06, z = h is 60 mm IN FRONT of the riser face, in mid-air, and it
//      counts. feetOnTread here requires x > 0.12 — the same line the trunk
//      has to cross.
//
//  (3) SCORING READS A FILE. scoreSaved() takes a PATH and JSON.parse()s it.
//      There is no entry point that scores an in-memory candidate, because the
//      exported keyframes are rounded (search_0.mjs:300 pose->4 dp, blend/gap/
//      side->4 dp; search_2.mjs:307 pose->5 dp) while the numbers quoted in the
//      notes came from the unrounded candidate. Phase R measures how far apart
//      those two are.
//
//  (4) THREE TAILS. climb_lib scores 50 ticks of BEST_alpha_stand.onnx AFTER
//      the track ends. Terminal trunk z is 115.3-118.1 mm in 41 of 41 upright
//      replays and a do-nothing control lands in the same band, which is what
//      you would expect if that tail simply walks the duck back to a floor
//      stance. --tail selects:
//        policy : 50 ticks of the stand policy   (climb_lib's own tail)
//        hold   : 50 ticks with data.ctrl frozen at clamp(final track pose),
//                 no ONNX call at all — the servos hold position
//        none   : score at the track's last tick; then run the 50 hold ticks
//                 anyway and report whether it is still upright after them
//      'none' and 'hold' are the same simulation (they differ only in which
//      snapshot is the answer), so one run yields both.
//
//  (4b) ROUND 5, THE SERVOED LANDING. An optional `servo` field (see
//      climb/servo.mjs) that, once armed by an authored track time or by the
//      round-4 event, takes the ten LEG slots off the keyframe track and
//      commands them EVERY CONTROL TICK from measured trunk height above the
//      tread, trunk pitch, trunk x relative to the riser line, and each foot's
//      x and z relative to the tread edge and top. Head and neck keep
//      following the keyframes. Targets are rate-limited and clamped to
//      [LO, HI]; the actuator ceiling is untouched. A file with no `servo`
//      replays byte-identically — climb/_r5_parity.mjs proves that on 318 rows
//      against climb/rig3_pre_r5.mjs (a byte copy of this file at commit
//      e00e1e4) and reproduces the round-4 judge's own PHASE P counts against
//      climb/rig3_pre_r4.mjs.
//
//  (4c) ROUND 5, THE WHOLE-EPISODE PENETRATION FIELD. penetrationAtScore is
//      one instant, so a move that passes THROUGH a block on the way up and
//      arrives clean scored clean (round 3 measured -9.4 mm transiently).
//      `minPenetrationEpisode` is the most negative mj_geomDistance between
//      any duck geom and any step geom at ANY control tick of the episode —
//      settle, track and tail — with the pair and the tick named. Exact, not
//      sampled: the bounding-sphere skip is a lower bound on the distance.
//
//  (5) A SPAWN OVERRIDE. An intent JSON may carry `spawn: {x,y,z}`, which
//      replaces the gap/side spawn. That is the only way to author the row a
//      criterion MUST pass: a duck already standing on the first tread.
//
// Contact detection is mj_geomDistance, never data.contact.get(i) — that
// returns an embind object that leaks the WASM heap to 2 GB in ~20 s even when
// .delete() is called (see climb/rig2.mjs).
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/rig3.mjs
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
import { STAIR_Y } from '../site/stairs.js';
// ================================================================== THE MOVE
// THE EPISODE LOOP IS NO LONGER IN THIS FILE. Everything from the cfg and the
// spawn down to the fiftieth tail tick — the settle, the +0.8 s track tail, the
// ctrl blend, the clamp, the event, the servo, the penetration tracker, the
// criterion and the shaped reward — now lives ONCE, in sim/climb_score.mjs, and
// robust.mjs and the bench's /climb import the same function. It was three
// copies: this one, robust.mjs's go() (its own header called itself "the ONE
// round-3 copy"), and the fourth the bench was about to grow so a phone could
// score a move. Three copies of a loop whose output is a trajectory do not stay
// equal by being read carefully.
//
// WHAT IS STILL THIS FILE'S IS THE RECORD. climb/audit_r6.mjs PHASE P1 compares
// 86 rows of scoreSaved()'s answer by a deep recursive walk over every leaf, so
// the field set below is assembled here, exactly as it always was, out of the
// shared episode's measurements. The physics is shared; the shape of the answer
// is local. Everything above this line still describes what this scorer does —
// it is simply now describing code that is one directory over.
import { makeClimbRig, criteria as sharedCriteria, reward as sharedReward,
         poseAt as sharedPoseAt, LATERAL as SHARED_LATERAL, RISER_X as SHARED_RISER_X }
  from '../sim/climb_score.mjs';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
export const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const model = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model);
const stand = await ort.InferenceSession.create('./BEST_alpha_stand.onnx');

/** The flight is 340 mm wide. Anything outside it is not on the staircase. */
export const LATERAL = SHARED_LATERAL;            // 0.17 m
export const RISER_X = SHARED_RISER_X;            // cfg.start — the first riser face
export { STAIR_Y };

/**
 * THIS FILE'S RIG: this model, this mjData, this onnxruntime session.
 *
 * `run` is the only machine-shaped thing the shared episode is handed. Here it
 * is onnxruntime-node, which is what every audited number was measured through;
 * in a browser the same episode is handed a hand-written forward pass over the
 * canonical parameter bytes, and the two agree to 3.5e-6 per action
 * (sim/policy_parity.mjs) rather than exactly — which is why a phone's /climb
 * answer is its own measurement and says so.
 */
const RIG = makeClimbRig({
  mj, model, data, D, HOME, LO, HI, buildObs, projectedGravity, command,
  tickHz: C.tickHz,
  async run(obs) {
    const r = await stand.run({ obs: new ort.Tensor('float32', obs, [1, 61]) });
    return r.actions.data;
  },
});
if (!RIG) throw new Error('scene.mjb has no stair bank: this rig cannot score a climb');

/** ROUND 4, HOLE 3: every collidable geom that belongs to the DUCK. */
export const DUCKG = RIG.DUCKG;

/** climb_lib.mjs:80-86, verbatim — sim/climb_score.mjs's, over this HOME. */
export function poseAt(tr, time) { return sharedPoseAt(tr, time, HOME); }

/**
 * Every criterion under consideration, evaluated on one snapshot, and the
 * shaped reward with the lateral gate. Both are sim/climb_score.mjs's, so the
 * verdict robust.mjs decides and the verdict the bench answers with cannot
 * drift from the one this instrument prints.
 */
export const criteria = sharedCriteria;
export const reward = sharedReward;

// ---------------------------------------------------------------- the episode

/**
 * One episode, through the shared loop, recorded in THIS file's shape.
 * INTERNAL — nothing outside this file may score an in-memory track; see
 * scoreSaved().
 */
async function runEpisodeRaw(track, opts, h, tail) {
  const E = await RIG.runEpisode(track, opts, h, tail, { stepCount: opts.stepCount });
  const { atTrackEnd, afterTail } = E;
  const scored = (tail === 'none') ? atTrackEnd : afterTail;
  const rec = {
    tail, rise: h, x0: E.x0, ctrlJump: E.ctrlJump,
    event: E.event,
    scored, atTrackEnd, afterTail,
    // ROUND 4 first-class fields of every scored row
    penetrationAtScore: scored.penetrationAtScore,
    penetrationPair: scored.penetrationPair,
    // ROUND 5, THE NEW HOLE: the deepest the duck was EVER inside a step geom,
    // over every control tick of the episode (settle + track + tail), with the
    // pair and the tick named. penetrationAtScore is one instant; this is the
    // whole episode, so a move that passes THROUGH a block and arrives clean
    // can no longer score clean.
    minPenetrationEpisode: E.penetration.min,
    minPenetrationPair: E.penetration.pair,
    minPenetrationTick: E.penetration.tick,
    penetrationTicksScanned: E.penetration.ticksScanned,
    // ROUND 5, THE SERVOED LANDING. null for a file with no `servo` block.
    servo: E.servo,
    // ROUND 6: read-only per-tail-tick record; undefined unless asked for.
    tailTrace: E.tailTrace,
    uprightTailTicks: E.uprightTailTicks, tailTicks: E.tailTicks,
    uprightTailFrac: E.uprightTailTicks / Math.max(E.tailTicks, 1),
    terminal: E.terminal,          // ROUND 4, FAMILY B: the handoff state
    crit: criteria(h, scored),
    critAtTrackEnd: criteria(h, atTrackEnd),
    critAfterTail: criteria(h, afterTail),
    maxX: E.maxX, maxZ: E.maxZ, maxAbsDY: E.maxAbsDY,
    feetOnTreadMax: E.feetOnTreadMax, feetUpRawMax: E.feetUpRawMax,
    // THE TREAD DRIFT OVER EVERY TICK, settle included — this file's own
    // reading. robust.mjs reads the same drift over the RECORDED ticks only,
    // and the shared episode keeps both because both were published.
    maxTreadSag_mm: E.allSag_mm, maxTreadDriftX_mm: E.allDriftX_mm,
    minStepGap_mm: E.allGap_mm === 1e9 ? null : E.allGap_mm, trace: E.trace,
    headFrac: E.headTicks / Math.max(E.ticks, 1),
    riserFrac: E.riserTicks / Math.max(E.ticks, 1),
    upFrac: E.upTicks / Math.max(E.ticks, 1),
    satFrac: E.sat / Math.max(E.ctrls, 1),
    // additive instrumentation (round 2, family C): sustained LOAD TRANSFER
    z0Settle: E.z0Settle,
    bothFrac: E.bothTicks / Math.max(E.ticks, 1),
    maxGainBoth: E.maxGainBoth === -1e9 ? null : E.maxGainBoth,
    sustainFrac: E.sustainTicks / Math.max(E.ticks, 1),
    liftIntegral: E.liftIntegral,
  };
  // climb_lib's own return, for the parity check
  rec.legacy = { onTop: rec.crit.orig, x: scored.x, z: scored.z, above: scored.above,
                 feetUp: scored.feetUpRaw, up: scored.up };
  rec.reward = reward(rec);
  return rec;
}

// ---------------------------------------------------------------- public API

/**
 * THE ONLY SCORER. It takes a PATH, not a track.
 *
 * The intent shape on disk is { keyframes:[{t,pose[14]}], blend, gap, side,
 * approach, spawn? }. Everything scored in this file goes through here, so
 * every number reported is a number the saved file actually produces —
 * candidate-vs-export drift cannot hide.
 */
export async function scoreSaved(path, { rise, tail = 'policy', gapOffset = 0, overrides = {} } = {}) {
  const j = JSON.parse(fs.readFileSync(path, 'utf8'));
  if (!Array.isArray(j.keyframes) || !j.keyframes.length) throw new Error('no keyframes in ' + path);
  for (const f of j.keyframes) if (!Array.isArray(f.pose) || f.pose.length !== 14) throw new Error('bad pose in ' + path);
  const opts = {
    blend: j.blend, approach: j.approach || 0,
    gap: (j.gap || 0) + gapOffset, side: j.side || 0,
    spawn: j.spawn || null,
    // ROUND 4, FAMILY A: the optional event-triggered tail (climb/event.mjs).
    event: j.event || null,
    // ROUND 5: the optional servoed landing (climb/servo.mjs). Absent in every
    // file written before round 5 -> null -> not one line of the law runs.
    servo: j.servo || null,
    // ROUND 4, FAMILY B handoff fields — null/undefined for every older file
    spawnQuat: j.spawnQuat || null, spawnPose: j.spawnPose || null,
    spawnVel: j.spawnVel || null, spawnLastAction: j.spawnLastAction || null,
    settleTicks: j.settleTicks === undefined ? undefined : j.settleTicks,
    ...overrides,
  };
  const rec = await runEpisodeRaw(j.keyframes, opts, rise, tail);
  rec.source = path; rec.opts = { ...opts, gapOffset };
  return rec;
}

/** Author a track, round-trip it through JSON on disk, hand back the path. */
export function saveTrack(obj, path) {
  fs.writeFileSync(path, JSON.stringify(obj, null, 2));
  return path;
}

/** Parity only. Not a scorer — it takes an in-memory track on purpose. */
export async function __parityEpisode(track, opts, h, tail) { return runEpisodeRaw(track, opts, h, tail); }

// ================================================================== EXPERIMENT

const isMain = process.argv[1] && process.argv[1].endsWith('rig3.mjs');
if (!isMain) { /* imported as a library */ } else {

const OUT = '../climb/rig3-results.json';
const RISES = [0.020, 0.040, 0.060, 0.090, 0.120, 0.180];
const TAILS = ['policy', 'hold', 'none'];
const mm = v => (v * 1000).toFixed(1);
const results = { generated: new Date().toISOString(), plant: 'scene.mjb', policy: 'BEST_alpha_stand.onnx',
                  lateralGate_m: LATERAL, riserX_m: RISER_X };
const t00 = Date.now();

// ---------------------------------------------------------------- PHASE P
// Parity: is this the same episode as sim/climb_lib.mjs attempt()?
// audit_cross.mjs's method — same track, same opts, same rise, full digits.
console.log('=== PHASE P — parity against sim/climb_lib.mjs (tail=policy, criterion=orig) ===');
const { replay: libReplay } = await import('../sim/climb_lib.mjs');
const parityCases = [];
{
  const a = HOME.slice(); a[5] = -1.3; a[6] = 0.7; a[7] = 1.4;
  parityCases.push({ label: 'synthetic (climb/parity.mjs track)', h: 0.04,
    track: [{ t: 0.5, pose: a }, { t: 1.6, pose: HOME.slice() }],
    opts: { blend: 1.6, approach: 0, gap: 0.03, side: 0.06 } });
}
for (const f of ['best_0_40mm.json', 'best_1_90mm.json', 'best_2_180mm.json']) {
  const j = JSON.parse(fs.readFileSync('../climb/' + f, 'utf8'));
  parityCases.push({ label: f, h: parseInt(f.match(/_(\d+)mm/)[1], 10) / 1000,
    track: j.keyframes, opts: { blend: j.blend, approach: j.approach || 0, gap: j.gap, side: j.side } });
}
results.parity = [];
let parityAll = true;
for (const c of parityCases) {
  const A = await libReplay(c.track, c.opts, c.h);
  const B = await __parityEpisode(c.track, c.opts, c.h, 'policy');
  const L = B.legacy;
  const match = A.onTop === L.onTop && A.x === L.x && A.z === L.z && A.feetUp === L.feetUp && A.up === L.up;
  parityAll = parityAll && match;
  results.parity.push({ case: c.label, rise_mm: c.h * 1000,
    climb_lib: { onTop: A.onTop, x: A.x, z: A.z, feetUp: A.feetUp, up: A.up },
    rig3: { onTop: L.onTop, x: L.x, z: L.z, feetUp: L.feetUp, up: L.up }, exactMatch: match });
  console.log(`  ${c.label.padEnd(34)} rise ${(c.h*1000).toString().padStart(3)}  ` +
    `climb_lib x=${A.x} z=${A.z} feetUp=${A.feetUp}\n  ${' '.repeat(34)}      ` +
    `rig3      x=${L.x} z=${L.z} feetUp=${L.feetUp}   EXACT=${match}`);
}
results.parityAll = parityAll;
console.log(`  PARITY ${parityAll ? 'PASS — rig3 is climb_lib' : 'FAIL'} (${results.parity.length} cases, full float digits)\n`);

// ---------------------------------------------------------------- PHASE R
// Round-trip: what does the export's rounding cost? (search_0 pose->4 dp,
// search_2 pose->5 dp; the notes quote the unrounded candidate.)
console.log('=== PHASE R — export rounding sensitivity (score the file, never the candidate) ===');
results.roundTrip = [];
for (const f of ['best_0_40mm.json', 'best_2_40mm.json', 'best_2_180mm.json']) {
  const h = parseInt(f.match(/_(\d+)mm/)[1], 10) / 1000;
  const asSaved = await scoreSaved('../climb/' + f, { rise: h, tail: 'policy' });
  const j = JSON.parse(fs.readFileSync('../climb/' + f, 'utf8'));
  j.keyframes = j.keyframes.map(k => ({ t: k.t, pose: k.pose.map(v => +v.toFixed(3)) }));
  const p = saveTrack(j, '/tmp/rig3_rt.json');
  const rounded = await scoreSaved(p, { rise: h, tail: 'policy' });
  const row = { file: f, dx_mm: +mm(rounded.scored.x - asSaved.scored.x), dz_mm: +mm(rounded.scored.z - asSaved.scored.z),
    saved: { x_mm: +mm(asSaved.scored.x), z_mm: +mm(asSaved.scored.z) },
    rounded3dp: { x_mm: +mm(rounded.scored.x), z_mm: +mm(rounded.scored.z) } };
  results.roundTrip.push(row);
  console.log(`  ${f.padEnd(20)} saved x=${mm(asSaved.scored.x).padStart(8)} z=${mm(asSaved.scored.z).padStart(7)}  ` +
    `| pose re-rounded to 3 dp x=${mm(rounded.scored.x).padStart(8)} z=${mm(rounded.scored.z).padStart(7)}  ` +
    `| moved ${Math.abs(row.dx_mm).toFixed(1)} mm in x, ${Math.abs(row.dz_mm).toFixed(1)} mm in z`);
}
console.log();

// ---------------------------------------------------------------- controls
const HOLD_TRACK = (pose) => [{ t: 1.0, pose: pose.slice() }, { t: 2.9, pose: pose.slice() }];

// (a) do-nothing: HOME held, no forward command. gap 0.05 -> spawn x = 0.000.
saveTrack({ name: 'control_do_nothing', keyframes: HOLD_TRACK(HOME), blend: 1, gap: 0.05, side: 0, approach: 0,
  note: 'DO-NOTHING CONTROL. HOME pose for the whole track, vx=0. A criterion that this row PASSES is not a climbing test.' },
  '../climb/ctrl_do_nothing.json');
// (a2) walk-only: HOME held, but the policy is commanded forward. The stronger
// control: if walking alone clears a rise, that rise is not a climbing test.
saveTrack({ name: 'control_walk_only', keyframes: HOLD_TRACK(HOME), blend: 1, gap: 0.05, side: 0, approach: 0.30,
  note: 'WALK-ONLY CONTROL. HOME pose, vx=0.30 — the stand policy walks at the flight with no authored motion at all.' },
  '../climb/ctrl_walk_only.json');
// short episode, for a length-matched pair of controls
const SHORT_TRACK = (pose) => [{ t: 0.2, pose: pose.slice() }, { t: 0.5, pose: pose.slice() }];
saveTrack({ name: 'control_do_nothing_short', keyframes: SHORT_TRACK(HOME), blend: 1, gap: 0.05, side: 0, approach: 0,
  note: 'DO-NOTHING CONTROL, short episode (1.3 s track). Length-matched partner of the short on-tread control.' },
  '../climb/ctrl_do_nothing_short.json');
// (c) the row a criterion MUST pass: already standing on the first tread.
// tread 0 spans x in [0.12, 0.46]; step 1 covers from x=0.40, so the EXPOSED
// tread of step 0 is x in [0.12, 0.40] — 280 mm of it. Spawn at three places
// along it and at two episode lengths, because the first attempt (one place,
// one length) confounds "the criterion cannot express standing on a tread"
// with "this particular placement walks off the front edge".
const ON_TREAD_X = [0.20, 0.26, 0.32];
const ON_TREAD_LEN = { short: SHORT_TRACK, med: HOLD_TRACK };
const onTreadFiles = [];
for (const rmm of [20, 40, 60, 90, 120, 180]) {
  for (const sx of ON_TREAD_X) for (const [ln, mk] of Object.entries(ON_TREAD_LEN)) {
    const p = `../climb/ctrl_on_tread_${rmm}mm_x${Math.round(sx*100)}_${ln}.json`;
    saveTrack({ name: `control_on_tread_${rmm}mm_x${Math.round(sx*100)}_${ln}`, keyframes: mk(HOME),
      blend: 1, gap: 0.05, side: 0, approach: 0,
      spawn: { x: sx, y: STAIR_Y, z: rmm / 1000 + 0.12 },
      note: `ON-TREAD CONTROL. Duck spawned standing on the first tread of a ${rmm} mm flight at x=${sx} `
          + `(exposed tread is x in [0.12, 0.40]); trunk 0.12 m above the tread, the same spawn height `
          + `attempt() uses above the floor. HOME pose, vx=0, ${ln} episode. `
          + `A criterion that NO placement of this row passes cannot express success at all.` }, p);
    onTreadFiles.push({ rmm, sx, ln, path: p });
  }
}

/** Run one saved file at one rise under all three tails (2 simulations). */
async function allTails(path, rise, gapOffset = 0) {
  const pol = await scoreSaved(path, { rise, tail: 'policy', gapOffset });
  const hld = await scoreSaved(path, { rise, tail: 'hold', gapOffset });
  const non = { ...hld, tail: 'none', scored: hld.atTrackEnd, crit: hld.critAtTrackEnd,
                penetrationAtScore: hld.atTrackEnd.penetrationAtScore,
                penetrationPair: hld.atTrackEnd.penetrationPair,
                stillUpAfter50Hold: hld.afterTail.up, stillPassesAfter50Hold: hld.critAfterTail };
  non.reward = reward(non);
  return { policy: pol, hold: hld, none: non };
}

const hdr = 'row                         rise  tail    x_mm     y-Y_mm    z_mm  above_mm  feetRaw feetLat feetTread up  ORIG  LAT  HONEST  H60';
const fmtRow = (label, rise, tail, r) => {
  const s = r.scored, c = r.crit;
  return `${label.padEnd(26)} ${(rise*1000).toString().padStart(4)}  ${tail.padEnd(6)} ` +
    `${mm(s.x).padStart(7)} ${mm(s.dy).padStart(8)} ${mm(s.z).padStart(7)} ${mm(s.above).padStart(8)}  ` +
    `${String(s.feetUpRaw).padStart(6)} ${String(s.feetUpLat).padStart(6)} ${String(s.feetOnTread).padStart(8)}  ` +
    `${s.up ? 'Y' : 'n'}  ${c.orig ? 'PASS' : ' .  '}  ${c.lat ? 'PASS' : ' . '}  ${c.honest ? ' PASS ' : '  .   '}  ${c.honest60 ? 'PASS' : ' . '}`;
};
const slim = r => ({ tail: r.tail, x_mm: +mm(r.scored.x), dy_mm: +mm(r.scored.dy), z_mm: +mm(r.scored.z),
  above_mm: +mm(r.scored.above), up: r.scored.up, feetUpRaw: r.scored.feetUpRaw, feetUpLat: r.scored.feetUpLat,
  feetOnTread: r.scored.feetOnTread, crit: r.crit, reward: +r.reward.toFixed(3),
  peakX_mm: +mm(r.maxX), peakZ_mm: +mm(r.maxZ), maxAbsDY_mm: +mm(r.maxAbsDY),
  feetOnTreadMax: r.feetOnTreadMax, feetUpRawMax: r.feetUpRawMax,
  headFrac: +r.headFrac.toFixed(3), riserFrac: +r.riserFrac.toFixed(3), ctrlJump: +r.ctrlJump.toFixed(4),
  penetrationAtScore_mm: r.penetrationAtScore === null ? null : +mm(r.penetrationAtScore),
  penetrationPair: r.penetrationPair,
  minPenetrationEpisode_mm: r.minPenetrationEpisode === null ? null : +mm(r.minPenetrationEpisode),
  minPenetrationPair: r.minPenetrationPair, minPenetrationTick: r.minPenetrationTick,
  uprightTailTicks: r.uprightTailTicks, tailTicks: r.tailTicks,
  stillUpAfter50Hold: r.stillUpAfter50Hold, stillPassesAfter50Hold: r.stillPassesAfter50Hold });

// ---------------------------------------------------------------- PHASE A
console.log('=== PHASE A — the do-nothing control. A criterion this row passes is not a climbing test. ===');
console.log(hdr);
results.controls = { doNothing: [], walkOnly: [] };
for (const h of RISES) {
  const t = await allTails('../climb/ctrl_do_nothing.json', h);
  for (const k of TAILS) { console.log(fmtRow('do-nothing (HOME, vx=0)', h, k, t[k])); results.controls.doNothing.push({ rise_mm: h*1000, ...slim(t[k]) }); }
}
console.log();
console.log('--- walk-only control: same HOME pose, but vx=0.30 (no authored motion at all) ---');
console.log(hdr);
for (const h of RISES) {
  const t = await allTails('../climb/ctrl_walk_only.json', h);
  for (const k of TAILS) { console.log(fmtRow('walk-only (HOME, vx=0.30)', h, k, t[k])); results.controls.walkOnly.push({ rise_mm: h*1000, ...slim(t[k]) }); }
}
console.log();

// ---------------------------------------------------------------- PHASE C
console.log('=== PHASE C — spawned ON the first tread, HOME pose. A criterion NO placement passes cannot express success. ===');
console.log('--- length-matched do-nothing (short episode) ---');
console.log(hdr);
results.controls.doNothingShort = [];
for (const h of RISES) {
  const t = await allTails('../climb/ctrl_do_nothing_short.json', h);
  for (const k of TAILS) { console.log(fmtRow('do-nothing short', h, k, t[k])); results.controls.doNothingShort.push({ rise_mm: h*1000, ...slim(t[k]) }); }
}
console.log();
console.log(hdr);
results.controls.onTread = [];
for (const c of onTreadFiles) {
  const t = await allTails(c.path, c.rmm / 1000);
  for (const k of TAILS) {
    console.log(fmtRow(`on-tread x=${c.sx.toFixed(2)} ${c.ln}`, c.rmm / 1000, k, t[k]));
    results.controls.onTread.push({ rise_mm: c.rmm, spawnX: c.sx, len: c.ln, ...slim(t[k]) });
  }
}
console.log('\n  best on-tread placement per rise (does ANY placement read as success?)');
console.log('  rise  tail    passes HONEST   best row');
for (const h of RISES) {
  for (const k of TAILS) {
    const rows = results.controls.onTread.filter(r => r.rise_mm === h * 1000 && r.tail === k);
    const win = rows.filter(r => r.crit.honest);
    const b = win[0] || rows.slice().sort((a, c) => c.above_mm - a.above_mm)[0];
    console.log(`  ${(h*1000).toString().padStart(4)}  ${k.padEnd(6)}  ${String(win.length).padStart(2)}/${rows.length}          ` +
      `x=${b.spawnX.toFixed(2)} ${b.len.padEnd(5)} x_mm=${b.x_mm.toFixed(1).padStart(7)} z_mm=${b.z_mm.toFixed(1).padStart(6)} ` +
      `above=${b.above_mm.toFixed(1).padStart(7)} feetTread=${b.feetOnTread} up=${b.up?'Y':'n'}`);
  }
}
console.log();

// ---------------------------------------------------------------- PHASE B
console.log('=== PHASE B — the 18 published bests, re-scored from the SAVED file, under each tail, at -10/0/+10 mm ===');
const bests = fs.readdirSync('../climb').filter(f => /^best_[012]_\d+mm\.json$/.test(f)).sort();
results.bests = [];
let anyClear = { orig: 0, lat: 0, honest: 0, honest60: 0 };
console.log('file                     rise off  tail    x_mm     y-Y_mm    z_mm  above_mm  feetRaw feetTread  up  ORIG  LAT  HONEST  H60   peakX  maxDY');
for (const f of bests) {
  const h = parseInt(f.match(/_(\d+)mm/)[1], 10) / 1000;
  for (const d of [-0.010, 0, 0.010]) {
    const t = await allTails('../climb/' + f, h, d);
    for (const k of TAILS) {
      const r = t[k], s = r.scored, c = r.crit;
      anyClear.orig += c.orig ? 1 : 0; anyClear.lat += c.lat ? 1 : 0;
      anyClear.honest += c.honest ? 1 : 0; anyClear.honest60 += c.honest60 ? 1 : 0;
      results.bests.push({ file: f, rise_mm: h * 1000, startOffset_mm: +(d * 1000).toFixed(0), ...slim(r) });
      console.log(`${f.padEnd(22)} ${(h*1000).toString().padStart(4)} ${(d*1000).toFixed(0).padStart(3)}  ${k.padEnd(6)} ` +
        `${mm(s.x).padStart(7)} ${mm(s.dy).padStart(8)} ${mm(s.z).padStart(7)} ${mm(s.above).padStart(8)}  ` +
        `${String(s.feetUpRaw).padStart(6)} ${String(s.feetOnTread).padStart(8)}   ${s.up?'Y':'n'}  ` +
        `${c.orig?'PASS':' .  '}  ${c.lat?'PASS':' . '}  ${c.honest?' PASS ':'  .   '}  ${c.honest60?'PASS':' . '}  ` +
        `${mm(r.maxX).padStart(7)} ${mm(r.maxAbsDY).padStart(6)}`);
    }
  }
}
results.bestsCleared = anyClear;
results.bestsRows = results.bests.length;
console.log(`\n  18 bests x 3 offsets x 3 tails = ${results.bests.length} scored rows.`);
console.log(`  cleared: ORIG ${anyClear.orig}   LAT ${anyClear.lat}   HONEST ${anyClear.honest}   HONEST60 ${anyClear.honest60}`);

// ---------------------------------------------------------------- PHASE D
// WHY does the on-tread control get thrown off at 20-120 mm but stand still at
// 180 mm? Two candidates: (i) the tread is in free fall between pins, so a
// duck standing on it rides a 50 Hz vibrating platform; (ii) something about
// the low-rise geometry. Measure the tread's motion, then re-run with the
// stairs pinned before EVERY mj_step — site/stairs.js's own stated contract
// ("call after any qpos write, and every tick"), which climb_lib.mjs:132 does
// not honour. The pinned variant is a DIAGNOSTIC, not the plant: it changes no
// physics constant, only how often the harness re-writes the stair joints.
console.log('=== PHASE D — is the tread standing still? (stairs are 200 kg boxes on frictionless slides) ===');
results.treadMotion = [];
console.log('rise  variant             tail    maxTreadSag_mm maxTreadDriftX_mm   x_mm    dy_mm   z_mm  feetTread  up  HONEST');
for (const h of [0.020, 0.040, 0.090, 0.120, 0.180]) {
  const rmm = Math.round(h * 1000);
  const path = `../climb/ctrl_on_tread_${rmm}mm_x26_med.json`;
  for (const [vlabel, ov] of [['plant (pin per ctrl tick)', {}], ['pinned per mj_step', { pinEverySubstep: true }]]) {
    for (const k of ['policy', 'none']) {
      const r = await scoreSaved(path, { rise: h, tail: k === 'none' ? 'hold' : 'policy',
        overrides: { ...ov, trace: true } });
      const rr = (k === 'none') ? { ...r, scored: r.atTrackEnd, crit: r.critAtTrackEnd } : r;
      const s = rr.scored, c = rr.crit;
      results.treadMotion.push({ rise_mm: rmm, variant: vlabel, tail: k,
        maxTreadSag_mm: +r.maxTreadSag_mm.toFixed(2), maxTreadDriftX_mm: +r.maxTreadDriftX_mm.toFixed(2),
        x_mm: +mm(s.x), dy_mm: +mm(s.dy), z_mm: +mm(s.z), above_mm: +mm(s.above),
        feetOnTread: s.feetOnTread, up: s.up, crit: c,
        trace: (vlabel.startsWith('plant') && k === 'none') ? r.trace.slice(0, 30) : undefined });
      console.log(`${rmm.toString().padStart(4)}  ${vlabel.padEnd(26)} ${k.padEnd(6)} ` +
        `${r.maxTreadSag_mm.toFixed(2).padStart(10)} ${r.maxTreadDriftX_mm.toFixed(2).padStart(14)}   ` +
        `${mm(s.x).padStart(7)} ${mm(s.dy).padStart(7)} ${mm(s.z).padStart(6)} ${String(s.feetOnTread).padStart(8)}   ` +
        `${s.up ? 'Y' : 'n'}  ${c.honest ? 'PASS' : ' . '}`);
    }
  }
}
console.log();

// ---------------------------------------------------------------- PHASE E
// The flight collides with itself. Consecutive step blocks overlap by 60 mm in
// x (STEP_HALF_DEPTH 0.17 vs run/2 = 0.14) and by (200 - rise) mm in z, and
// they share collision bits. A box-box normal points along the axis of LEAST
// penetration, so for rise < 140 mm the z-overlap is the larger one and the
// blocks push each other APART ALONG X — the tread lurches, is teleported back
// by the next layoutStairs, and lurches again, 50 times a second.
// Test: lay out ONE step instead of four (nothing to collide with) and see
// whether the on-tread control then stands. Diagnostic only — a 1-step flight
// is not the 4-step flight attempt() scores.
console.log('=== PHASE E — does the flight throw the duck off because the steps collide with each other? ===');
results.stepSelfCollision = [];
console.log('rise  steps  minStepGap_mm  maxTreadDriftX_mm  maxTreadSag_mm    x_mm    dy_mm   z_mm  feetTread  up  HONEST');
for (const h of RISES) {
  const rmm = Math.round(h * 1000);
  const path = `../climb/ctrl_on_tread_${rmm}mm_x26_med.json`;
  for (const n of [4, 1]) {
    const r = await scoreSaved(path, { rise: h, tail: 'policy', overrides: { stepCount: n } });
    const s = r.scored, c = r.crit;
    results.stepSelfCollision.push({ rise_mm: rmm, stepCount: n,
      minStepGap_mm: r.minStepGap_mm === null ? null : +r.minStepGap_mm.toFixed(2),
      maxTreadDriftX_mm: +r.maxTreadDriftX_mm.toFixed(2), maxTreadSag_mm: +r.maxTreadSag_mm.toFixed(2),
      x_mm: +mm(s.x), dy_mm: +mm(s.dy), z_mm: +mm(s.z), above_mm: +mm(s.above),
      feetOnTread: s.feetOnTread, up: s.up, crit: c });
    console.log(`${rmm.toString().padStart(4)}  ${String(n).padStart(5)}  ` +
      `${(r.minStepGap_mm === null ? 'n/a' : r.minStepGap_mm.toFixed(2)).padStart(13)}  ` +
      `${r.maxTreadDriftX_mm.toFixed(2).padStart(17)}  ${r.maxTreadSag_mm.toFixed(2).padStart(14)}   ` +
      `${mm(s.x).padStart(7)} ${mm(s.dy).padStart(7)} ${mm(s.z).padStart(6)} ${String(s.feetOnTread).padStart(8)}   ` +
      `${s.up ? 'Y' : 'n'}  ${c.honest ? 'PASS' : ' . '}`);
  }
}
// Where is the threshold? Predicted at rise = 200 - 60 = 140 mm, the rise at
// which the z-overlap between consecutive blocks stops being the deeper one.
console.log('\n  threshold sweep — the rise at which a duck can stand on its own tread (4 steps, plant as-is)');
console.log('  rise steps z-ovl  minStepGap  driftX   sag     x_mm   z_mm  above  lfootZ-h  rfootZ-h  feetTread up HONEST');
results.thresholdSweep = [];
for (const rmm of [20, 40, 60, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]) {
  const h = rmm / 1000;
  const p = saveTrack({ name: `control_on_tread_${rmm}mm_sweep`, keyframes: HOLD_TRACK(HOME),
    blend: 1, gap: 0.05, side: 0, approach: 0, spawn: { x: 0.26, y: STAIR_Y, z: h + 0.12 },
    note: `ON-TREAD CONTROL for the step-self-collision threshold sweep, rise ${rmm} mm.` },
    `/tmp/rig3_sweep_${rmm}.json`);
  for (const n of [4, 1]) {
    const r = await scoreSaved(p, { rise: h, tail: 'policy', overrides: { stepCount: n } });
    const s = r.scored, c = r.crit;
    const lz = (s.lfoot.z - h) * 1000, rz = (s.rfoot.z - h) * 1000;
    results.thresholdSweep.push({ rise_mm: rmm, stepCount: n, zOverlap_mm: 200 - rmm,
      minStepGap_mm: r.minStepGap_mm === null ? null : +r.minStepGap_mm.toFixed(2),
      maxTreadDriftX_mm: +r.maxTreadDriftX_mm.toFixed(2), maxTreadSag_mm: +r.maxTreadSag_mm.toFixed(2),
      x_mm: +mm(s.x), z_mm: +mm(s.z), above_mm: +mm(s.above),
      lfootZminusRise_mm: +lz.toFixed(1), rfootZminusRise_mm: +rz.toFixed(1),
      feetOnTread: s.feetOnTread, up: s.up, crit: c });
    console.log(`  ${rmm.toString().padStart(4)} ${String(n).padStart(5)} ${(200 - rmm).toString().padStart(5)}  ` +
      `${(r.minStepGap_mm === null ? 'n/a' : r.minStepGap_mm.toFixed(2)).padStart(10)}  ` +
      `${r.maxTreadDriftX_mm.toFixed(2).padStart(6)} ${r.maxTreadSag_mm.toFixed(2).padStart(5)} ` +
      `${mm(s.x).padStart(8)} ${mm(s.z).padStart(6)} ${mm(s.above).padStart(6)} ` +
      `${lz.toFixed(1).padStart(9)} ${rz.toFixed(1).padStart(9)} ${String(s.feetOnTread).padStart(9)}  ${s.up?'Y':'n'}  ${c.honest?'PASS':' . '}`);
  }
}
console.log();

// ---------------------------------------------------------------- summary
const tailVerdict = {};
for (const k of TAILS) {
  const on = results.controls.onTread.filter(r => r.tail === k);
  const dn = results.controls.doNothing.filter(r => r.tail === k).concat(results.controls.doNothingShort.filter(r => r.tail === k));
  const wo = results.controls.walkOnly.filter(r => r.tail === k);
  const risesExpressed = RISES.filter(h => on.some(r => r.rise_mm === h * 1000 && r.crit.honest));
  tailVerdict[k] = {
    onTread_pass_honest: on.filter(r => r.crit.honest).length + '/' + on.length,
    onTread_pass_orig: on.filter(r => r.crit.orig).length + '/' + on.length,
    onTread_rises_expressed_mm: risesExpressed.map(h => h * 1000),
    onTread_upright_frac: on.filter(r => r.up).length + '/' + on.length,
    doNothing_pass_honest: dn.filter(r => r.crit.honest).length + '/' + dn.length,
    doNothing_pass_orig: dn.filter(r => r.crit.orig).length + '/' + dn.length,
    doNothing_upright: dn.filter(r => r.up).length + '/' + dn.length,
    walkOnly_pass_honest: wo.filter(r => r.crit.honest).length + '/' + wo.length,
    walkOnly_pass_orig: wo.filter(r => r.crit.orig).length + '/' + wo.length,
    canExpressSuccess: risesExpressed.length > 0 && dn.filter(r => r.crit.honest).length === 0,
  };
}
results.tailVerdict = tailVerdict;
console.log('\n=== TAIL VERDICT (does the tail let a real climb read as success?) ===');
for (const k of TAILS) {
  const v = tailVerdict[k];
  console.log(`  tail=${k.padEnd(7)} on-tread HONEST ${v.onTread_pass_honest.padStart(6)}  upright ${v.onTread_upright_frac.padStart(6)}  ` +
    `| do-nothing HONEST ${v.doNothing_pass_honest} upright ${v.doNothing_upright}  ` +
    `| walk-only HONEST ${v.walkOnly_pass_honest}  -> CAN EXPRESS SUCCESS: ${v.canExpressSuccess}`);
  console.log(`             rises where a real on-tread stance reads as PASS (mm): [${v.onTread_rises_expressed_mm.join(', ')}]`);
}
// the podium confound, counted
const offFlight = results.bests.filter(r => r.maxAbsDY_mm > LATERAL * 1000);
results.lateralConfound = {
  rows_off_flight: offFlight.length, rows_total: results.bests.length,
  files_off_flight: [...new Set(offFlight.map(r => r.file))].sort(),
  worst_mm: Math.max(...results.bests.map(r => r.maxAbsDY_mm)),
};
console.log(`\n=== LATERAL CONFOUND ===\n  ${offFlight.length}/${results.bests.length} scored rows leave the 340 mm flight ` +
  `(|y-STAIR_Y| > ${LATERAL*1000} mm at some tick); worst excursion ${results.lateralConfound.worst_mm.toFixed(1)} mm.`);
console.log(`  files that ever leave the flight: ${results.lateralConfound.files_off_flight.join(', ') || 'none'}`);

results.elapsed_s = +((Date.now() - t00) / 1000).toFixed(1);
fs.writeFileSync(OUT, JSON.stringify(results, null, 1));
console.log(`\nWROTE ${OUT}  (${results.elapsed_s}s)`);
}
