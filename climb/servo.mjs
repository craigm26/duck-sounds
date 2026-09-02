// servo.mjs — ROUND 5. THE SERVOED LANDING.
//
// WHY THIS EXISTS. Every move published through round 4 lands its feet on a
// CLOCK. The keyframe track says "at t = 1.34 s the legs are here", and the
// legs go there whether the trunk arrived 10 mm short, 10 mm long, or 40 mm
// low. The round-4 judge measured what that costs: the round-3 beak-strut
// vault (sha256 4b9110c448ec) clears 4 of 9 core cells, its clears are
// ISOLATED POINTS (one control tick of shift takes it 4 -> 1 of 9 and moves
// the trunk 304 mm), and a rise sweep 50/55/60/65/70 mm reads NO/NO/YES/NO/NO.
// Round 4's event mechanism did not fix that, and the judge said why: every
// measured event fires on the SAME tick in every plant, so the event carries
// no information the clock does not, and the axis that decides the verdict
// (rise) is not observable at the event.
//
// A servoed landing is the other thing. It is not a throw at all. Once armed,
// the LEG slots stop following keyframes and are commanded EVERY CONTROL TICK
// from what the trunk and the feet are measured to be doing RIGHT NOW. The
// rise the duck is actually on shows up in the very first reading (the trunk's
// height above the tread), which is exactly the axis a clock cannot see.
//
// -------------------------------------------------------------- THE READINGS
// Five numbers are measured at the top of every tick, all in metres/radians,
// all read-only:
//
//   above     trunk z - h                    (height above the tread)
//   pitch     projectedGravity(trunkQuat)[0] (trunk pitch; 0 = upright)
//   dxTrunk   trunk x - 0.12                 (trunk past the riser line)
//   foot[i].dx  foot geom x - 0.12           (foot past the riser line = the
//                                             tread's front edge)
//   foot[i].dz  foot geom z - h              (foot above the tread's top)
//
// i = 0 is the LEFT foot geom, i = 1 the RIGHT.
//
// ---------------------------------------------------------------- THE LAW
// Per side, three targets, each an affine function of the errors:
//
//   eZ  = zTarget  - above          > 0: trunk too low   -> push up
//   eP  = pitch    - pitchRef       > 0: trunk pitched forward of the reference
//   eTX = xTrunk   - dxTrunk        > 0: trunk short of the riser
//   eFX = xFoot    - foot.dx        > 0: foot short of its landing spot
//   eFZ = fz       - foot.dz        > 0: foot below its landing height
//
//   hip_pitch[i] = base.hip[i]   + kHipZ*eZ  + kHipPitch*eP + kHipX*eFX + kHipTrunkX*eTX
//   knee[i]      = base.knee[i]  + kKneeZ*eZ + kKneeFz*eFZ  + kKneeX*eFX
//   ankle[i]     = base.ankle[i] + kAnkPitch*eP + kAnkFz*eFZ
//
// Then, in this order: a RATE LIMIT of `rate` radians per control tick against
// what that slot was last commanded, and a hard clamp to [LO, HI]. The
// actuator ceiling is untouched — this file never sees forcerange; it only
// authors position targets, exactly like a keyframe does.
//
// The four hip yaw / hip roll slots are also taken over (they are LEG slots)
// but they carry no feedback: `yawRoll` is 'hold' (freeze at whatever the
// track had them at when the servo armed, the default), 'track' (leave them
// on the keyframes), or four explicit numbers.
//
// The head and neck slots (5 neck_pitch, 6 head_pitch, 7 head_yaw, 8 head_roll)
// are NEVER touched by the servo. In the beak-strut vault the neck IS the
// strut; a servo that let go of it would be a different move.
//
// ---------------------------------------------------------------- ARMING
//   servo.at        track-seconds. Arms at the first tick with time >= at.
//   servo.onEvent   true: also arms on the tick the file's `event` fires.
// Whichever comes first wins. A file with neither never arms (and is therefore
// identical to a file with no servo at all).
//
// ---------------------------------------------------------------- INERTNESS
// `servo` is an OPTIONAL top-level field of the saved intent JSON. normServo()
// returns null for a file that does not carry one, and on a null every call
// site in rig3.mjs and robust.mjs is a single `if (SV && ...)` that is false,
// so not one number changes. climb/_r5_parity.mjs proves that at full float
// digits on the round-4 judge's 86 rows and on a 225-row set.
//
// Everything here is pure: it takes measured numbers and returns numbers. It
// touches no ctrl, no qpos, no criterion.

/** The 14 policy slots, mouth removed (site/duckloop.mjs POLICY). */
export const SLOT = {
  L_HIP_YAW: 0, L_HIP_ROLL: 1, L_HIP_PITCH: 2, L_KNEE: 3, L_ANKLE: 4,
  NECK_PITCH: 5, HEAD_PITCH: 6, HEAD_YAW: 7, HEAD_ROLL: 8,
  R_HIP_YAW: 9, R_HIP_ROLL: 10, R_HIP_PITCH: 11, R_KNEE: 12, R_ANKLE: 13,
};
/** [left, right] slot index per commanded joint. */
export const HIP_PITCH = [SLOT.L_HIP_PITCH, SLOT.R_HIP_PITCH];
export const KNEE = [SLOT.L_KNEE, SLOT.R_KNEE];
export const ANKLE = [SLOT.L_ANKLE, SLOT.R_ANKLE];
export const YAWROLL = [SLOT.L_HIP_YAW, SLOT.L_HIP_ROLL, SLOT.R_HIP_YAW, SLOT.R_HIP_ROLL];
/** Every slot the servo owns once armed. */
export const LEG_SLOTS = [...HIP_PITCH, ...KNEE, ...ANKLE, ...YAWROLL].sort((a, b) => a - b);

const num = (v, d) => (v === undefined || v === null) ? d : +v;

/**
 * Validate and normalise. Returns null for a file that carries no servo —
 * which is every file written before round 5.
 */
export function normServo(sv) {
  if (!sv) return null;
  const at = (sv.at === undefined || sv.at === null) ? null : +sv.at;
  const onEvent = !!sv.onEvent;
  if (at === null && !onEvent) throw new Error('servo needs at: <track seconds> or onEvent: true');
  if (at !== null && !(at >= 0)) throw new Error('servo.at must be >= 0');
  let yawRoll = sv.yawRoll === undefined ? 'hold' : sv.yawRoll;
  if (Array.isArray(yawRoll)) {
    if (yawRoll.length !== 4) throw new Error('servo.yawRoll array must be 4 numbers (Lyaw, Lroll, Ryaw, Rroll)');
    yawRoll = yawRoll.map(Number);
  } else if (yawRoll !== 'hold' && yawRoll !== 'track') {
    throw new Error("servo.yawRoll must be 'hold', 'track', or 4 numbers");
  }
  const base = sv.base ? {
    hip: sv.base.hip, knee: sv.base.knee, ankle: sv.base.ankle,
  } : null;
  if (base) for (const k of ['hip', 'knee', 'ankle'])
    if (!Array.isArray(base[k]) || base[k].length !== 2) throw new Error('servo.base.' + k + ' must be [left, right]');
  return {
    at, onEvent, yawRoll, base,
    // set-points
    zTarget: num(sv.zTarget, 0.115),      // trunk metres above the tread
    xTrunk: num(sv.xTrunk, 0.16),         // trunk metres past the riser line
    xFoot: num(sv.xFoot, 0.10),           // foot metres past the riser line
    fz: num(sv.fz, 0.015),                // foot metres above the tread top
    pitchRef: num(sv.pitchRef, 0),
    // gains, radians per metre (or per unit of projected gravity for eP)
    kHipZ: num(sv.kHipZ, 0), kHipPitch: num(sv.kHipPitch, 0),
    kHipX: num(sv.kHipX, 0), kHipTrunkX: num(sv.kHipTrunkX, 0),
    kKneeZ: num(sv.kKneeZ, 0), kKneeFz: num(sv.kKneeFz, 0), kKneeX: num(sv.kKneeX, 0),
    kAnkPitch: num(sv.kAnkPitch, 0), kAnkFz: num(sv.kAnkFz, 0),
    // THE MIRROR. The duck's left and right joint frames are mirrored: the
    // round-3 vault's own keyframes read L_hip_pitch -0.363 / R_hip_pitch
    // +0.363, L_knee +0.454 / R_knee -0.454. A law with one set of gains and
    // no sign would drive the two legs in OPPOSITE physical directions, so
    // every feedback term (never the base) is multiplied by sign[side].
    sign: Array.isArray(sv.sign) && sv.sign.length === 2 ? sv.sign.map(Number) : [1, -1],
    // authority
    rate: Math.abs(num(sv.rate, 0.12)),   // radians per control tick
    span: Math.abs(num(sv.span, 1.2)),    // |target - base| ceiling, radians
  };
}

/**
 * The base pose the servo departs from: the track pose commanded at the arm
 * tick, unless the file authored `servo.base`. `commanded` is the 14-vector
 * poseAt(track, t) at the instant the servo armed.
 */
export function servoBase(S, commanded) {
  const hip = S.base ? S.base.hip.slice() : HIP_PITCH.map(k => commanded[k]);
  const knee = S.base ? S.base.knee.slice() : KNEE.map(k => commanded[k]);
  const ankle = S.base ? S.base.ankle.slice() : ANKLE.map(k => commanded[k]);
  const yr = Array.isArray(S.yawRoll) ? S.yawRoll.slice() : YAWROLL.map(k => commanded[k]);
  return { hip, knee, ankle, yawRoll: yr };
}

/**
 * ONE TICK OF THE LAW.
 *
 *   S     normServo() output
 *   base  servoBase() output
 *   m     { above, pitch, dxTrunk, feet: [{dx,dz},{dx,dz}] }   measured NOW
 *   prev  the 14-vector of what each slot was last COMMANDED (data.ctrl)
 *   LO,HI the actuator ranges
 *
 * Returns a 14-array: a number for every slot the servo owns, null for every
 * slot it does not (head/neck always; hip yaw+roll when yawRoll === 'track').
 * Rate-limited against `prev`, then clamped to [LO, HI].
 */
export function servoTick(S, base, m, prev, LO, HI) {
  const out = new Array(14).fill(null);
  const eZ = S.zTarget - m.above;
  const eP = m.pitch - S.pitchRef;
  const eTX = S.xTrunk - m.dxTrunk;
  const put = (k, want, b) => {
    // |target - base| ceiling, then rate limit, then LO/HI
    let v = Math.max(b - S.span, Math.min(b + S.span, want));
    const p = prev[k];
    if (v > p + S.rate) v = p + S.rate;
    if (v < p - S.rate) v = p - S.rate;
    out[k] = Math.min(Math.max(v, LO[k]), HI[k]);
  };
  for (let i = 0; i < 2; i++) {
    const g = S.sign[i];
    const eFX = S.xFoot - m.feet[i].dx;
    const eFZ = S.fz - m.feet[i].dz;
    put(HIP_PITCH[i], base.hip[i] + g * (S.kHipZ * eZ + S.kHipPitch * eP + S.kHipX * eFX + S.kHipTrunkX * eTX), base.hip[i]);
    put(KNEE[i], base.knee[i] + g * (S.kKneeZ * eZ + S.kKneeFz * eFZ + S.kKneeX * eFX), base.knee[i]);
    put(ANKLE[i], base.ankle[i] + g * (S.kAnkPitch * eP + S.kAnkFz * eFZ), base.ankle[i]);
  }
  if (S.yawRoll !== 'track') for (let j = 0; j < 4; j++) put(YAWROLL[j], base.yawRoll[j], base.yawRoll[j]);
  return out;
}
