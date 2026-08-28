// A "step up" intent, and the search that tunes it.
//
// The walking policy tops out at a 2 mm step (measured). The idea under test is
// the one a long-jumper uses in reverse: plant something that is not a leg,
// then unload the legs against it. The duck has a head on a 50 mm neck with
// pitch, yaw and roll, and a beak — enough to be a third point of contact.
//
// The move is open-loop keyframes on the fourteen joints. Open-loop on purpose:
// the walking policy is a closed-loop controller for walking, and the whole
// point here is to do something it was never trained to do.
export const PHASES = ['plant', 'push', 'swing', 'transfer', 'recover'];

/** Joint indices within the 14 policy joints. */
export const J = {
  lHipYaw: 0, lHipRoll: 1, lHipPitch: 2, lKnee: 3, lAnkle: 4,
  neckPitch: 5, headPitch: 6, headYaw: 7, headRoll: 8,
  rHipYaw: 9, rHipRoll: 10, rHipPitch: 11, rKnee: 12, rAnkle: 13,
};

/**
 * Turn parameters into a keyframe track: [{ t, pose[14] }].
 * `lead` is the leg that goes up first (0 = left, 1 = right).
 */
export function buildTrack(p, HOME) {
  const lead = p.lead ? {
    hip: J.rHipPitch, knee: J.rKnee, ankle: J.rAnkle, roll: J.rHipRoll,
  } : { hip: J.lHipPitch, knee: J.lKnee, ankle: J.lAnkle, roll: J.lHipRoll };
  const trail = p.lead ? {
    hip: J.lHipPitch, knee: J.lKnee, ankle: J.lAnkle, roll: J.lHipRoll,
  } : { hip: J.rHipPitch, knee: J.rKnee, ankle: J.rAnkle, roll: J.rHipRoll };
  // Left and right joints are exact negations of each other on this robot.
  const sgnLead = p.lead ? -1 : 1, sgnTrail = p.lead ? 1 : -1;

  const base = () => HOME.slice();
  const frames = [];
  let t = 0;

  // 1. PLANT — reach the head down and forward onto the tread.
  t += p.tPlant;
  const plant = base();
  plant[J.neckPitch] = p.neckPlant;
  plant[J.headPitch] = p.headPlant;
  frames.push({ t, pose: plant });

  // 2. PUSH — press down through the head while both knees flex, so the head
  //    carries load the legs would otherwise have to lift.
  t += p.tPush;
  const push = plant.slice();
  push[J.neckPitch] = p.neckPush;
  push[J.headPitch] = p.headPush;
  push[lead.hip] = HOME[lead.hip] + sgnLead * p.leadHipPush;
  push[lead.knee] = HOME[lead.knee] + sgnLead * p.leadKneePush;
  push[trail.hip] = HOME[trail.hip] + sgnTrail * p.trailHipPush;
  push[trail.knee] = HOME[trail.knee] + sgnTrail * p.trailKneePush;
  frames.push({ t, pose: push });

  // 3. SWING — the lead foot comes up and over the riser.
  t += p.tSwing;
  const swing = push.slice();
  swing[lead.hip] = HOME[lead.hip] + sgnLead * p.leadHipSwing;
  swing[lead.knee] = HOME[lead.knee] + sgnLead * p.leadKneeSwing;
  swing[lead.ankle] = HOME[lead.ankle] + sgnLead * p.leadAnkleSwing;
  frames.push({ t, pose: swing });

  // 4. TRANSFER — plant the lead foot on the tread and take weight off the head.
  t += p.tTransfer;
  const transfer = swing.slice();
  transfer[lead.hip] = HOME[lead.hip] + sgnLead * p.leadHipPlant;
  transfer[lead.knee] = HOME[lead.knee] + sgnLead * p.leadKneePlant;
  transfer[J.neckPitch] = p.neckLift;
  transfer[J.headPitch] = p.headLift;
  frames.push({ t, pose: transfer });

  // 5. RECOVER — trailing foot up, back to a stance.
  t += p.tRecover;
  frames.push({ t, pose: base() });
  return frames;
}

/** Sample the track, holding the last pose after the end. */
export function poseAt(track, time, HOME) {
  if (time <= 0) return HOME.slice();
  let prevT = 0, prevPose = HOME;
  for (const f of track) {
    if (time <= f.t) {
      const u = (time - prevT) / Math.max(f.t - prevT, 1e-6);
      const s = u * u * (3 - 2 * u);           // smoothstep: no step in velocity
      return f.pose.map((v, k) => prevPose[k] + (v - prevPose[k]) * s);
    }
    prevT = f.t; prevPose = f.pose;
  }
  return track[track.length - 1].pose.slice();
}

export const DEFAULTS = {
  lead: 0, blend: 1.0, approach: 0.32,
  tPlant: 0.5, tPush: 0.4, tSwing: 0.35, tTransfer: 0.45, tRecover: 0.6,
  neckPlant: 1.0, headPlant: 0.9,
  neckPush: 1.25, headPush: 1.1,
  neckLift: 0.2, headLift: 0.2,
  leadHipPush: 0.25, leadKneePush: 0.30,
  trailHipPush: 0.20, trailKneePush: 0.25,
  leadHipSwing: 0.75, leadKneeSwing: 0.85, leadAnkleSwing: -0.30,
  leadHipPlant: 0.45, leadKneePlant: 0.35,
};

/** Search bounds. Angles in radians, times in seconds. */
export const BOUNDS = {
  blend: [0.2, 1.4], approach: [0.0, 0.55],
  tPlant: [0.2, 0.9], tPush: [0.2, 0.9], tSwing: [0.15, 0.7],
  tTransfer: [0.2, 0.9], tRecover: [0.3, 1.0],
  neckPlant: [0.2, 1.5], headPlant: [-0.2, 1.5],
  neckPush: [0.2, 1.5], headPush: [-0.2, 1.5],
  neckLift: [-0.4, 0.8], headLift: [-0.4, 0.8],
  leadHipPush: [-0.2, 0.9], leadKneePush: [-0.2, 1.0],
  trailHipPush: [-0.2, 0.9], trailKneePush: [-0.2, 1.0],
  leadHipSwing: [0.0, 1.3], leadKneeSwing: [0.0, 1.3], leadAnkleSwing: [-0.9, 0.5],
  leadHipPlant: [-0.2, 1.1], leadKneePlant: [-0.2, 1.1],
};
