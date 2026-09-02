// event.mjs — ROUND 4, FAMILY A. The EVENT-TRIGGERED tail of a track.
//
// WHY THIS EXISTS. The round-3 beak-strut vault is a single ballistic pivot
// whose tuck/land segment fires at an AUTHORED TIME. Measured in round 3: it
// lands ~10 mm short, in the same direction, in every failing cell, and the
// rise axis alone costs 13 of 21 cells. A move that fires its landing on a
// clock cannot know that this particular plant threw the body 10 mm shorter.
//
// So the tail of the track may instead fire on a MEASURED EVENT, and the
// landing keyframe may be shifted by how far the trunk actually is from where
// the author expected it to be at that instant.
//
// THE INTENT SHAPE. `event` is an OPTIONAL top-level field of the saved JSON.
// A file without it replays EXACTLY as before — normEvent() returns null and
// the replay loop is the pre-round-4 loop, keyframe for keyframe, tick for
// tick. That is what climb/famA_r4.mjs PHASE P proves on every existing file.
//
//   "event": {
//     "type":      "beak" | "pitch" | "trunkZ",   -- the discrete gene
//     "threshold": number,        -- beak: metres of mj_geomDistance jaw<->step
//                                 -- pitch: projected-gravity x, fires on >=
//                                 -- trunkZ: metres of trunk z above the tread
//     "arm":       number,        -- track-seconds before which it cannot fire
//     "fallback":  number,        -- track-seconds at which it fires ANYWAY, so
//                                    the move always completes (and a fallback
//                                    at the round-3 tuck time degenerates the
//                                    move back to the round-3 clock)
//     "delay":     number,        -- seconds held after the event before the
//                                    post-event segment starts
//     "refX":      number,        -- the trunk x the author expected at the event
//     "clamp":     number,        -- |e| ceiling, metres
//     "post": [ { "dt": s-after-(fire+delay), "pose": [14], "adapt"?: [14] } ]
//   }
//
// THE FEEDBACK. e = clamp(refX - trunkX_at_fire, -clamp, +clamp): positive when
// the body is SHORT of where it was supposed to be. Each post keyframe's joint
// target is pose[k] + adapt[k] * e, so `adapt` is a per-joint gain in
// radians per metre of shortfall and the landing keyframe's leg targets aim at
// where the body actually is. adapt absent = the keyframe is not adapted.
//
// Everything here is pure: it reads numbers the caller measured and returns
// numbers. It touches no ctrl, no qpos, no criterion.
export const EVENT_TYPES = ['beak', 'pitch', 'trunkZ'];

/** Validate and normalise. Returns null for a file that carries no event. */
export function normEvent(ev) {
  if (!ev) return null;
  if (!EVENT_TYPES.includes(ev.type)) throw new Error('event.type must be one of ' + EVENT_TYPES.join('/'));
  if (!Array.isArray(ev.post) || !ev.post.length) throw new Error('event.post must be a non-empty array');
  let last = 0;
  for (const p of ev.post) {
    if (!Array.isArray(p.pose) || p.pose.length !== 14) throw new Error('bad event.post pose');
    if (p.adapt !== undefined && (!Array.isArray(p.adapt) || p.adapt.length !== 14)) throw new Error('bad event.post adapt');
    if (!(p.dt > last)) throw new Error('event.post dt must be strictly increasing and > 0');
    last = p.dt;
  }
  return {
    type: ev.type,
    threshold: +ev.threshold,
    arm: Math.max(0, +ev.arm || 0),
    fallback: +ev.fallback,
    delay: Math.max(0, +ev.delay || 0),
    refX: +ev.refX,
    clamp: ev.clamp === undefined ? 0.12 : Math.abs(+ev.clamp),
    post: ev.post.map(p => ({ dt: +p.dt, pose: p.pose.slice(), adapt: p.adapt ? p.adapt.slice() : null })),
  };
}

/**
 * Did the event fire this tick? `m` is what the caller measured RIGHT NOW:
 *   m.beakDist  min mj_geomDistance(any jaw geom, any step geom), metres
 *   m.pitch     projectedGravity(trunk quat)[0]
 *   m.above     trunk z - tread height h, metres
 */
export function eventFires(ev, m) {
  if (ev.type === 'beak') return m.beakDist !== null && m.beakDist < ev.threshold;
  if (ev.type === 'pitch') return m.pitch >= ev.threshold;
  return m.above >= ev.threshold;                       // trunkZ
}

/** e, the shortfall the landing is shifted by. Positive = the body is short. */
export function eventError(ev, trunkX) {
  return Math.max(-ev.clamp, Math.min(ev.clamp, ev.refX - trunkX));
}

/**
 * Rebuild the track at the instant the event fires.
 *
 * Every base keyframe strictly BEFORE tFire is kept, so poseAt()'s running
 * `pp` (which starts at HOME) is the same chain it was; then a keyframe at
 * exactly tFire carrying the pose that was commanded at tFire, so the
 * commanded target is continuous across the switch; then the held delay; then
 * the post keyframes, adapted by e.
 */
export function buildDynTrack(base, ev, tFire, poseAtFire, e) {
  const out = base.filter(f => f.t < tFire).map(f => ({ t: f.t, pose: f.pose.slice() }));
  // NO anchor keyframe at tFire when delay is 0. poseAt()'s smoothstep is
  // anchored on the PREVIOUS keyframe, so inserting a keyframe at tFire
  // restarts the ease and slows the segment that is already in flight:
  // measured on the round-3 60 mm move, an anchor at the round-3 tuck time
  // moved the commanded pose by up to 0.035 rad and took the move from 4 of 9
  // to 0 of 9. Without it the same degenerate event reproduces the timed move.
  // A non-zero `delay` is therefore also the gene that BUYS the anchor: it
  // freezes the pose that was commanded at the event for `delay` seconds
  // before the post-event segment starts.
  if (ev.delay > 0) {
    out.push({ t: tFire, pose: poseAtFire.slice() });
    out.push({ t: tFire + ev.delay, pose: poseAtFire.slice() });
  }
  const t0 = tFire + ev.delay;
  for (const p of ev.post) {
    out.push({ t: t0 + p.dt, pose: p.pose.map((v, k) => v + (p.adapt ? p.adapt[k] * e : 0)) });
  }
  return out;
}
