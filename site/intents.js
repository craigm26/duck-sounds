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

/** Command magnitudes for a variant, so driving matches the runtime. */
export function speeds(variant) {
  return variant === 'rollers'
    ? { fwd: DEFAULTS.RVEL_FWD, back: DEFAULTS.RVEL_BACK, ang: DEFAULTS.RVEL_ANG }
    : { fwd: DEFAULTS.VEL_FWD,  back: DEFAULTS.VEL_BACK,  ang: DEFAULTS.VEL_ANG };
}
