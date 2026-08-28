// The duck's intents, and the keys that fire them.
//
// Each is a separate trained network sharing the same 61-in / 14-out contract
// as the walker, so switching intent is switching which session runs — not a
// different pipeline. Durations are measured, not guessed: see sim/skills.mjs,
// which runs each one and reports what it does to the body.
//
// One-shot and EXCLUSIVE, which is how robotd treats them: while a skill holds
// the robot a second one is refused rather than blended. DuckKit's DuckSkill
// models the same rule; a UI that let you stack them would be lying about the
// robot.
export const INTENTS = [
  {
    key: 'q', id: 'kick_left', label: 'Kick left', policy: 'ball_kick_left.onnx',
    seconds: 1.6, cmd: () => ({}),
  },
  {
    key: 'e', id: 'kick_right', label: 'Kick right', policy: 'ball_kick_right.onnx',
    seconds: 1.6, cmd: () => ({}),
  },
  {
    key: 'f', id: 'ground_pick', label: 'Pick up', policy: 'alpha_ground_pick.onnx',
    seconds: 3.0,
    // This one is phase-driven: the command slots carry a clock, not a
    // velocity — [cos, sin] of the progress through the move.
    cmd: u => ({ vx: Math.cos(2 * Math.PI * u), vy: Math.sin(2 * Math.PI * u) }),
  },
  {
    key: 'x', id: 'roulade', label: 'Forward roll', policy: 'roulade.onnx',
    seconds: 2.6, cmd: () => ({}),
  },
  {
    key: 'c', id: 'sit', label: 'Sit', policy: 'BEST_alpha_sitstand.onnx',
    seconds: 2.4, cmd: () => ({ vx: 1 }), toggles: 'stand',
  },
  {
    key: 'v', id: 'stand', label: 'Stand up', policy: 'BEST_alpha_sitstand.onnx',
    seconds: 2.4, cmd: () => ({ vx: 0 }),
  },
  {
    key: 'z', id: 'hold', label: 'Hold still', policy: 'BEST_alpha_stand.onnx',
    seconds: 1.5, cmd: () => ({}),
  },
];

/** The authored one — an offset on the policy rather than a policy of its own. */
export const STEP_UP_KEY = 'g';
