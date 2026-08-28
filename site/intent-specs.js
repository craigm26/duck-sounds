// What each move needs the world to look like before it will work.
//
// None of these numbers are invented here. Every keyframe move was SEARCHED
// from a specific staging — a duck at a measured distance from a riser face or
// a wall, facing +x, joints at HOME, settled — and it only reproduces from
// that staging. Until now that staging existed in exactly one place,
// sim/record_intents.mjs, where it is used to make the gallery clips. The
// browser had none of it, so pressing "Lever up" in an empty room played a
// track authored against a step that was not there and the duck just thrashed.
//
// This module is that contract, lifted out of the recorder so both can import
// it. If you change staging, change it here.
import { STAIR_Y } from './stairs.js';

/**
 * The flight the stair moves were measured against: FOUR steps of TEN
 * MILLIMETRES. Not a design choice — 10 mm is the tallest step anything in the
 * library has ever climbed, so it is the tallest step worth staging against.
 * The page's own slider goes to 180 mm, which is why most of the time the
 * honest answer to "can it do this move here" is no.
 */
export const STAGE_STAIRS = { count: 4, rise: 0.010, run: 0.28, start: 0.12 };

/** The arena's north wall: plane at y = 1.5, half-thickness 0.025. */
export const WALL_Y = 1.5 - 0.025;

/**
 * Two constants that look like geometry and are not.
 *
 * record_intents.mjs stages a stair move at `x = start - 0.07 - gap`, with a
 * comment calling 0.07 the step's half-depth. It is not: layoutStairs() writes
 * the block CENTRE at `start + STEP_HALF_DEPTH`, so the block spans `start` to
 * `start + 0.34` and the riser face is at `start` exactly. STEP_HALF_DEPTH is
 * 0.17, not 0.07, and it would not belong in this sum anyway.
 *
 * So `gap` in the intent JSONs is not the clearance to the riser — the true
 * clearance is `0.07 + gap`, and the searches absorbed the offset because they
 * only ever compared staging against itself. The moves are valid; the label is
 * not. Anything that has to REASON about where the duck is standing needs the
 * true number, so the offset is named and added here rather than left implicit.
 */
export const STAIR_GAP_BASE = 0.07;
export const WALL_GAP_BASE = 0.05;
/** What the recorder falls back to when an intent JSON carries no gap. */
export const GAP_DEFAULT = 0.06;

/**
 * The settle is not optional.
 *
 * Every track was searched from a duck held for 25 control ticks under the
 * STAND policy with the move's own approach command, and a duck that arrives
 * any other way is not in the state the move was searched from. Measured: the
 * pose matches to five decimals but the joint velocities are out by up to
 * 0.147 — and those velocities are 14 of the 61 observation floats the policy
 * reads. A move fired from the wrong velocity state does not reproduce.
 *
 * This is why walking the duck into position is not enough on its own, and why
 * an approach has to END in the same settle the search used.
 */
export const SETTLE_TICKS = 25;
export const SETTLE_POLICY = 'BEST_alpha_stand.onnx';

/** Which moves need something in the room, and what. */
export const NEEDS = {
  step_up:   'stair',
  lever_up:  'stair',
  riser_up:  'stair',
  climb:     'stair',
  wall_flip: 'wall',
  back_roll: null,     // flat floor, no prop — it is a roll
};

/**
 * Turn an intent's shipped JSON into a staging contract in WORLD coordinates.
 *
 * `standoff` is metres back along -x from the riser face (or -y from the wall
 * face) to where the duck's root must sit. `side` is metres toward the wall
 * from the flight's centre line.
 */
export function stagingFor(id, json) {
  const kind = NEEDS[id];
  if (!kind) return null;
  const p = json.params || {};
  const gap = json.gap ?? p.gap ?? GAP_DEFAULT;
  const approach = json.approach ?? p.approach ?? 0;
  return kind === 'stair'
    ? { kind, rise: STAGE_STAIRS.rise, standoff: STAIR_GAP_BASE + gap, side: json.side ?? 0, approach }
    : { kind, standoff: WALL_GAP_BASE + gap, side: 0, approach };
}

/** How close counts as staged. Generous, because the duck cannot walk slowly. */
export const TOL = { pos: 0.030, yaw: 0.14, riseMm: 1 };

const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));

/** Where the duck must stand for this move, given the room as it is now. */
export function targetFor(staging, stairCfg) {
  if (!staging) return null;
  return staging.kind === 'stair'
    ? { x: stairCfg.start - staging.standoff, y: STAIR_Y + staging.side, yaw: 0 }
    // The flip pushes SIDEWAYS off the wall, so the duck stands beside it
    // facing +x rather than facing into it.
    : { x: 0, y: WALL_Y - staging.standoff, yaw: 0 };
}

/**
 * Can this move run, here, now — and if not, is that fixable and how?
 *
 * `fix` is the machine-readable part: 'stairs' means the room is wrong and only
 * the person can change it, 'approach' means the duck is in the wrong place and
 * we can drive it there. `reason` is what to say out loud.
 */
export function readiness(id, json, world) {
  const staging = stagingFor(id, json);
  if (!staging) return { ok: true, staging: null, target: null, fix: null, reason: '' };

  const { stairCfg, pose } = world;
  if (staging.kind === 'stair') {
    const mm = Math.round(stairCfg.rise * 1000);
    const wantMm = Math.round(staging.rise * 1000);
    if (stairCfg.count < 1 || stairCfg.rise <= 0) {
      return { ok: false, staging, target: null, fix: 'stairs',
               reason: `nothing to climb — set Step height to ${wantMm} mm` };
    }
    if (Math.abs(mm - wantMm) > TOL.riseMm) {
      return { ok: false, staging, target: null, fix: 'stairs',
               reason: `these steps are ${mm} mm; this move was only ever measured at ${wantMm} mm` };
    }
  }

  const target = targetFor(staging, stairCfg);
  const d = Math.hypot(target.x - pose.x, target.y - pose.y);
  const dyaw = Math.abs(wrap(target.yaw - pose.yaw));
  if (d <= TOL.pos && dyaw <= TOL.yaw) {
    return { ok: true, staging, target, fix: null, reason: 'in position' };
  }
  return { ok: false, staging, target, fix: 'approach',
           reason: `${Math.round(d * 1000)} mm from where this move starts` };
}
