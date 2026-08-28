// Drive the duck to a pose, then hand over.
//
// This is a servo around the WALKING POLICY, not around the joints: the only
// handles are the same two numbers a person has, a forward velocity and a yaw
// rate, and the policy decides what the legs do with them. That constraint is
// the whole design. Three properties of this duck shape it, all of them stated
// on the page itself because they are ours and not the robot's:
//
//   it will not walk slowly    — the policy needs about 0.3 of command before
//                                it produces a gait at all, so there is no
//                                creeping up on a target. You arrive at speed
//                                or you do not arrive.
//   it drifts to one side      — heading is not preserved by walking, so the
//                                bearing has to be corrected continuously
//                                rather than aimed once and trusted.
//   it covers about half the   — so distance-to-go cannot be turned into a
//   ground you ask it to         time-to-go, and the loop has to be closed on
//                                position rather than run open for N ticks.
//
// Together those mean overshoot is normal and must be designed for: the
// controller re-approaches rather than treating a miss as failure, and gives
// up out loud instead of orbiting forever.

const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

/** Yaw about z from a MuJoCo [w,x,y,z] quaternion. */
export function yawOf(q) {
  const [w, x, y, z] = q;
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}

export const PHASES = ['turn', 'walk', 'align', 'done', 'failed'];

/**
 * @param target {x, y, yaw}
 * @param tol    {pos, yaw}
 * @param speed  {fwd, back, ang} — the variant's own command magnitudes
 */
export function planApproach(target, tol, speed, opts = {}) {
  // Budgeted in control ticks, not seconds, so it is the same on any machine.
  const budget = opts.budget ?? 1600;          // 32 s at 50 Hz
  // Enter/exit bands differ on purpose. One threshold makes the duck dither on
  // the boundary: it turns until the bearing is acceptable, walks, drifts back
  // over the line, turns again, and never covers any ground.
  const AIM_ENTER = 0.35;                      // stop walking, turn, if worse than this
  const AIM_EXIT  = 0.12;                      // resume walking once better than this
  let phase = 'turn';
  let ticks = 0;
  let best = Infinity;
  let sinceBest = 0;

  return {
    get phase() { return phase; },
    target,
    /**
     * @param pose {x, y, yaw}
     * @returns {vx, vyaw, phase, done, failed, note}
     */
    step(pose) {
      ticks++;
      const dx = target.x - pose.x, dy = target.y - pose.y;
      const dist = Math.hypot(dx, dy);
      const bearing = wrap(Math.atan2(dy, dx) - pose.yaw);
      const dyaw = wrap(target.yaw - pose.yaw);

      // Progress watchdog. Orbiting a target forever looks identical to
      // walking towards one if you only ever check the budget.
      if (dist < best - 0.005) { best = dist; sinceBest = 0; } else { sinceBest++; }
      if (ticks > budget || sinceBest > 500) {
        phase = 'failed';
        return { vx: 0, vyaw: 0, phase, done: false, failed: true,
                 note: `could not get there — ${Math.round(dist * 1000)} mm out after ${ticks} ticks` };
      }

      if (phase !== 'align' && dist > tol.pos) {
        // Far away: aim, then go. Aiming is a separate phase rather than a
        // correction term because turning while walking is what makes the duck
        // arc past the target instead of reaching it.
        if (phase === 'turn' && Math.abs(bearing) < AIM_EXIT) phase = 'walk';
        else if (phase === 'walk' && Math.abs(bearing) > AIM_ENTER) phase = 'turn';
        if (phase === 'turn') {
          return { vx: 0, vyaw: clamp(bearing * 3, -1, 1) * speed.ang,
                   phase, done: false, failed: false,
                   note: `turning to face the mark, ${Math.round(bearing * 57.3)}° off` };
        }
        return { vx: speed.fwd, vyaw: clamp(bearing * 1.5, -0.6, 0.6) * speed.ang,
                 phase, done: false, failed: false,
                 note: `walking, ${Math.round(dist * 1000)} mm to go` };
      }

      // Arrived. Now the heading, which matters as much as the position: the
      // track is a set of joint angles in the duck's own frame, so a duck
      // standing in the right spot facing the wrong way runs the move sideways.
      phase = 'align';
      if (Math.abs(dyaw) > tol.yaw) {
        return { vx: 0, vyaw: clamp(dyaw * 3, -1, 1) * speed.ang,
                 phase, done: false, failed: false,
                 note: `squaring up, ${Math.round(Math.abs(dyaw) * 57.3)}° off` };
      }
      // Drifted back out of the position band while turning on the spot.
      if (dist > tol.pos * 1.6) {
        phase = 'turn';
        return { vx: 0, vyaw: 0, phase, done: false, failed: false, note: 'drifted off the mark, going again' };
      }
      phase = 'done';
      return { vx: 0, vyaw: 0, phase, done: true, failed: false,
               note: `on the mark, ${Math.round(dist * 1000)} mm out` };
    },
  };
}
