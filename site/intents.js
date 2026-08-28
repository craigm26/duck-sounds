// The duck's intents, faithful to the Microduck simulator's own defaults.
//
// Every number here is Pollen's, taken from their constants.js and game.js
// rather than chosen. Where this project ADDS something — the step-up, the
// stairs — it is marked as ours. That line matters: the point of the sim is to
// be a true baseline that new intents can be built on and trusted against.
//
// Sources: microduck-simulator/app/src/game/{constants,game}.js
export const DEFAULTS = {
  // Locomotion command magnitudes, per variant.
  VEL_FWD: 0.25, VEL_BACK: -0.2, VEL_ANG: 1.0,
  RVEL_FWD: 0.6, RVEL_BACK: -0.5, RVEL_ANG: 0.3,
  // One-shots.
  KICK_STEPS: 25,               // 0.5 s at 50 Hz
  POST_KICK_LOCK_STEPS: 20,     // 0.4 s before another input is taken
  GROUND_PICK_PERIOD_S: 4.0, GROUND_PICK_END_PHASE: 0.7,
  CROUCH_PERIOD_S: 5.0, CROUCH_END_PHASE: 0.7,
  ROLL_MIN_STEPS: 40,           // a roll is over when it has tipped AND recovered
  ROLL_EXPIRE_STEPS: 150,       // ...and never runs longer than 3 s
  BALL_RADIUS: 0.05,
};

const TICK = 50;

/**
 * Intents, in the shape the runtime actually plays them.
 *
 * `kind` matters more than it looks:
 *   'oneshot' runs for a fixed number of control steps.
 *   'phase'   drives a clock into the command slots; the policy reads its own
 *             progress rather than a velocity.
 *   'mode'    is not a one-shot at all — it swaps the driving policy until
 *             something swaps it back. Sit is a mode, which is why an earlier
 *             version of this file could sit the duck down and never stand it
 *             up: it timed out and handed a SITTING duck back to the walking
 *             policy, which has no idea what to do with one.
 */
export const INTENTS = [
  {
    key: 'q', id: 'kick_left', label: 'Kick left', policy: 'ball_kick_left.onnx',
    kind: 'oneshot', steps: DEFAULTS.KICK_STEPS, lock: DEFAULTS.POST_KICK_LOCK_STEPS,
    cmd: () => ({}),
  },
  {
    key: 'e', id: 'kick_right', label: 'Kick right', policy: 'ball_kick_right.onnx',
    kind: 'oneshot', steps: DEFAULTS.KICK_STEPS, lock: DEFAULTS.POST_KICK_LOCK_STEPS,
    cmd: () => ({}),
  },
  {
    key: 'f', id: 'ground_pick', label: 'Pick up', policy: 'alpha_ground_pick.onnx',
    kind: 'phase',
    steps: Math.round(DEFAULTS.GROUND_PICK_PERIOD_S * DEFAULTS.GROUND_PICK_END_PHASE * TICK),
    period: DEFAULTS.GROUND_PICK_PERIOD_S,
    // The command slots carry a clock, not a velocity: [cos, sin] of progress.
    cmd: u => ({ vx: Math.cos(2 * Math.PI * u), vy: Math.sin(2 * Math.PI * u) }),
  },
  {
    key: 'x', id: 'roulade', label: 'Forward roll', policy: 'roulade.onnx',
    // Ends on a CONDITION, not a clock: it is done once it has tipped over and
    // come back upright, and it is abandoned at 3 s if it never does.
    kind: 'until', minSteps: DEFAULTS.ROLL_MIN_STEPS, steps: DEFAULTS.ROLL_EXPIRE_STEPS,
    cmd: () => ({}),
  },
  {
    key: 'c', id: 'sit', label: 'Sit', policy: 'BEST_alpha_sitstand.onnx',
    kind: 'mode', flag: 1, cmd: () => ({ vx: 1 }),
  },
  {
    key: 'v', id: 'stand', label: 'Stand up', policy: 'BEST_alpha_sitstand.onnx',
    // Also a mode: it holds the sitstand policy with the flag down until the
    // duck is actually up, then hands back to walking. Handing back early is
    // what left it in a heap.
    kind: 'mode', flag: 0, handBack: true, cmd: () => ({ vx: 0 }),
  },
  {
    key: 'z', id: 'hold', label: 'Hold still', policy: 'BEST_alpha_stand.onnx',
    kind: 'mode', flag: 0, cmd: () => ({}),
  },
];

/** Ours, not Pollen's: the searched head-plant step-up. */
export const STEP_UP_KEY = 'g';

/**
 * Also ours: a backward roll.
 *
 * Pollen ship a forward roll and no backward one, and forward is the easy
 * direction — past the toes, gravity finishes the job. Backwards the duck has
 * to carry its own mass over its heels on +-0.96 N.m per joint. Three
 * hand-authored attempts got to about 60 degrees and rocked back. A search over
 * the same shape found one that reaches 179 degrees — fully inverted — and
 * lands upright, five times out of five.
 */
export const BACK_ROLL_KEY = 'b';

/**
 * Ours: the head used as a single finger.
 *
 * The step-up unweights a leg so a foot can reach the tread, which caps it at
 * the leg's own reach. Levering does not care about leg reach — the head is a
 * pivot and the body rotates over it, so it should beat the step-up and on a
 * loose scorer it did: 40 mm against 26 mm, 3/3.
 *
 * Those figures are WITHDRAWN. See [CLIMB_KEY] for what the loose scorer was
 * accepting. On a real flight — 280 mm treads, flush to the wall — this clears
 * nothing at all.
 */
export const LEVER_KEY = 'h';

/**
 * Ours: a flip off a wall.
 *
 * A roll can borrow the floor; a flip has to get its rotation from somewhere,
 * and 0.96 N.m on a 737 g body is not much to spin with. A wall gives the feet
 * something to push against that is not underneath them, which turns a weak leg
 * extension into angular momentum rather than a hop. 179 degrees, passes
 * through inverted once, lands upright five times out of five. Stand about
 * 0.1 m off a wall for it.
 */
export const WALL_FLIP_KEY = 't';

/**
 * Ours: the wall-flip trick, borrowed for the stairs.
 *
 * A stair riser is a vertical surface and it is already there — so the same
 * thing that makes the wall flip work is available on every step, with no wall
 * needed. A foot planted on the riser face pushes against something that is not
 * underneath it, while the head pivots on the tread. Two points of purchase
 * instead of one.
 *
 *   stepping onto the tread   26 mm
 *   head as a pivot           40 mm
 *   foot on the riser too     55 mm   (fails at 70)
 *
 * Each idea stacked on the one before rather than replacing it — and every one
 * of those three numbers is WITHDRAWN. They came off a podium-staged step with
 * a scorer that a duck draped over the edge on its chest could pass. Under the
 * strict bar this move manages 16 mm, and on a real flight it manages 10 mm,
 * which is the highest anything here reaches. See [CLIMB_KEY].
 */
export const RISER_KEY = 'y';

/**
 * Ours: head on the shelf, then walk the feet up the riser.
 *
 * Every earlier attempt treated the step as something to step ONTO — one foot
 * lifts, reaches the tread, takes weight — which caps at how high a foot can
 * reach while the other leg holds the body up. This does something different:
 * the head takes weight on the tread first, unloading both legs at once, then
 * the feet climb the vertical riser in alternating presses until the body can
 * be brought over the edge. The legs never reach the tread from the floor.
 *
 * Measured on a strict bar — both feet up, standing height above the tread,
 * upright, and STILL THERE a second later:
 *
 *   riser push   16 mm
 *   climbing     24 mm   3/3
 *
 * The older 26 / 40 / 55 mm figures used a looser test that a duck draped over
 * the edge on its chest could pass. They are withdrawn.
 */
export const CLIMB_KEY = 'u';

/** Command magnitudes for a variant, so driving matches the runtime. */
export function speeds(variant) {
  return variant === 'rollers'
    ? { fwd: DEFAULTS.RVEL_FWD, back: DEFAULTS.RVEL_BACK, ang: DEFAULTS.RVEL_ANG }
    : { fwd: DEFAULTS.VEL_FWD,  back: DEFAULTS.VEL_BACK,  ang: DEFAULTS.VEL_ANG };
}
