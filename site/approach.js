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
  // Measured: 0.12 rad — 6.9 degrees — is finer than this duck can hold while
  // turning on the spot. It steers BY walking, so a command of 1.0 carries
  // angular momentum the policy does not shed on demand, and the aim phase
  // oscillated across a 7-degree window without ever entering the walk phase.
  // Convergence went from start-dependent to 6/6 by widening the window and
  // letting the walking correction do the fine work.
  const AIM_ENTER = 0.75;                      // stop walking, turn, if worse than this
  const AIM_EXIT  = 0.30;                      // resume walking once better than this
  const TURN_PATIENCE = 120;                   // ...and never aim for longer than this
  let phase = 'turn';
  let ticks = 0;
  let turning = 0;
  // Latched, not recomputed. A bare threshold on the bearing chattered: the
  // duck reversed, drifted, the bearing fell under the line, it began turning,
  // which moved the bearing back over it, and so on — 114 degrees out and
  // still arguing with itself. Enter at 2.84 rad, leave at 2.2.
  let rev = false;
  let best = Infinity;
  let sinceBest = 0;
  // Where the duck must end up POINTING, as a unit vector. Used to decide
  // which side to arrive from.
  const face = { x: Math.cos(target.yaw), y: Math.sin(target.yaw) };
  // Waypoints, decided on the first step because that is when a pose exists.
  let route = null;

  // Position alone is the wrong thing to watch. A duck that has arrived and is
  // turning on the spot makes no positional progress at all, so a distance-only
  // watchdog kills it mid-turn — measured, it gave up 15 mm from the mark and
  // 131 degrees off. Heading counts as progress; 0.06 m per radian puts a full
  // half-turn on the same scale as a 20 cm walk.
  const err = (dist, ang) => dist + Math.abs(ang) * 0.06;

  return {
    get phase() { return phase; },
    get route() { return route; },
    target,
    /**
     * @param pose {x, y, yaw}
     * @returns {vx, vyaw, phase, done, failed, note}
     */
    step(pose) {
      ticks++;
      if (!route) {
        // Arrive from BEHIND. Walking straight at a mark you have to leave
        // facing +x, from somewhere in front of it, means arriving pointed the
        // wrong way and doing a half-turn on the spot — which this duck is bad
        // at, because turning is something the walking policy does while
        // walking. One waypoint behind the mark turns that into a straight
        // final leg that ends already pointing the right way.
        // ALWAYS approach along the target's own heading, not just when the
        // duck starts in front of the mark.
        //
        // The alternative — walk straight at the mark, then square up on the
        // spot — does not work on this robot, and the measurements say so: from
        // six starts the position converged to 9-18 mm every time while the
        // heading stuck 30 degrees out and timed out spinning. It steers by
        // walking. So the last leg is always a straight run down the target's
        // facing, and the duck arrives already pointing the right way instead
        // of arriving and then trying to turn.
        const d0 = Math.hypot(pose.x - target.x, pose.y - target.y);
        route = d0 > 0.15
          ? [{ x: target.x - 0.38 * face.x, y: target.y - 0.38 * face.y, via: true },
             { x: target.x, y: target.y, via: false }]
          : [{ x: target.x, y: target.y, via: false }];
      }

      const wp = route[0];
      const dx = wp.x - pose.x, dy = wp.y - pose.y;
      const dist = Math.hypot(dx, dy);
      const bearing = wrap(Math.atan2(dy, dx) - pose.yaw);
      const dyaw = wrap(target.yaw - pose.yaw);
      // With the mark almost directly behind it, turning round is the worst way
      // to get there: a half-turn on the spot is the one manoeuvre this duck
      // cannot close, and it timed out 173 degrees out trying. It has a
      // backward gait — the runtime's own -0.2 against 0.25 forward — so it
      // reverses instead, and arrives still facing the way it was.
      // 2.84 rad, not 2.4: reversing only pays when the mark is ALMOST exactly
      // behind — within about 17 degrees of straight back. Measured at 2.4, a
      // mark 154 degrees off sent the duck reversing on a 0.2 gait it barely
      // moves on, and it stalled 472 mm out; turning to it forwards had worked.
      // The latch leaves on the REVERSE aim error, not on the bearing. Exiting
      // when |bearing| fell under 2.2 rad meant a normal amount of sideways
      // drift — 54 degrees over 0.7 m, which this duck does — looked like a
      // decision to stop reversing, and it abandoned a manoeuvre that was
      // working. |aim| > 1.2 is the honest condition: the line is actually lost.
      const revAim = Math.abs(wrap(bearing - Math.sign(bearing || 1) * Math.PI));
      if (!rev && Math.abs(bearing) > 2.84 && dist < 1.2) rev = true;
      else if (rev && revAim > 1.2) rev = false;
      const aim = rev ? wrap(bearing - Math.sign(bearing) * Math.PI) : bearing;
      // A waypoint on the way is only a hint; stopping dead on it wastes ticks.
      const posTol = wp.via ? 0.11 : tol.pos;

      // WHICH heading counts as progress depends on what the duck is doing.
      // In transit the useful one is the bearing to the waypoint — turning to
      // face it is progress, and scoring that turn against the FINAL heading
      // made a duck starting behind the mark look stuck the whole time it
      // turned around. Only once it is standing on the mark does the final
      // heading become the thing being improved.
      const closing = phase !== 'align' && dist > posTol;
      const e = err(dist, closing ? aim : dyaw);
      if (e < best - 0.005) { best = e; sinceBest = 0; } else { sinceBest++; }
      if (ticks > budget || sinceBest > 600) {
        phase = 'failed';
        return { vx: 0, vyaw: 0, phase, done: false, failed: true,
                 note: `could not get there — ${Math.round(dist * 1000)} mm and ${Math.round(Math.abs(dyaw) * 57.3)}° out after ${ticks} ticks` };
      }

      if (phase !== 'align' && dist > posTol) {
        // Far away: aim, then go. Aiming is a separate phase rather than a
        // correction term because turning while walking is what makes the duck
        // arc past the target instead of reaching it.
        // The patience escape is for a duck dithering on a small error. Firing
        // it on a near-180-degree bearing would set it walking away from the
        // mark, so a big turn is allowed to take as long as it takes — the
        // watchdog scores bearing as progress, which keeps that honest.
        // The escape stays limited to modest aim errors. Letting it fire at any
        // angle sent a duck standing beside the north wall arcing away to find
        // room it did not have — 1136 mm out. Reversing is the manoeuvre for
        // that case; this one is only for dithering.
        if (phase === 'turn' && (Math.abs(aim) < AIM_EXIT
                                 || (turning > TURN_PATIENCE && Math.abs(aim) < 0.9))) {
          phase = 'walk'; turning = 0;
        } else if (phase === 'walk' && Math.abs(aim) > AIM_ENTER) {
          phase = 'turn'; turning = 0;
        }
        if (phase === 'turn') {
          turning++;
          return { vx: 0, vyaw: clamp(aim * 3, -1, 1) * speed.ang,
                   phase, done: false, failed: false,
                   note: `lining up, ${Math.round(aim * 57.3)}° off` };
        }
        return { vx: rev ? speed.back : speed.fwd, vyaw: clamp(aim * 1.5, -0.8, 0.8) * speed.ang,
                 phase, done: false, failed: false,
                 note: `${rev ? 'backing up' : 'walking'}, ${Math.round(dist * 1000)} mm to go` };
      }

      if (route.length > 1) {
        route.shift();
        best = Infinity; sinceBest = 0; phase = 'turn'; rev = false;
        return { vx: 0, vyaw: 0, phase, done: false, failed: false, note: 'lined up behind the mark' };
      }

      // Arrived. Now the heading, which matters as much as the position: the
      // track is a set of joint angles in the duck's own frame, so a duck
      // standing in the right spot facing the wrong way runs the move sideways.
      phase = 'align';
      if (Math.abs(dyaw) > tol.yaw * 0.6) {
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
