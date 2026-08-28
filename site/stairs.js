// Stairs, without recompiling anything.
//
// The scene ships a fixed bank of step blocks, each on an x and a z slide
// joint. Reshaping the staircase writes those joints' qpos. Two earlier
// approaches did not work and are worth recording so nobody retries them:
// building geometry at runtime (a meshed MJCF will not compile in the browser —
// MuJoCo's convex-hull pass wants threads it cannot spawn), and moving a static
// geom via model.geom_pos (measured: the duck walks straight through a platform
// placed that way, even with geom_rbound corrected). Position that comes from
// qpos is live.
export const STAIR_COUNT = 14;
/** Half-depth of a step block, metres. Runs longer than 2x this leave gaps. */
export const STEP_HALF_DEPTH = 0.06;
/**
 * Half-height of a step block.
 *
 * Small on purpose. They were 0.30 — tall enough to be solid all the way to the
 * floor — which is invisible in physics but not on screen: rendered, fourteen
 * 60 cm slabs swallowed the room and buried the duck. They can be thin because
 * steps and the floor sit on different collision bits and never meet, so a
 * block hanging below floor level costs nothing.
 */
export const STEP_HALF_HEIGHT = 0.025;

/**
 * qpos AND dof addresses for each step's [x, z] joints, looked up once.
 *
 * The dof addresses matter as much as the qpos ones: a step is a heavy body on
 * a frictionless slide, so setting its position every tick without also zeroing
 * its velocity leaves the solver believing it is travelling. It then behaves
 * like a catapult — measured, it threw the duck half a metre into the air.
 */
export function findStairJoints(model) {
  const addr = [];
  for (let i = 0; i < STAIR_COUNT; i++) {
    let x = -1, z = -1, dx = -1, dz = -1;
    for (let j = 0; j < model.njnt; j++) {
      const n = model.jnt(j).name;
      if (n === `step${i}_x`) { x = model.jnt_qposadr[j]; dx = model.jnt_dofadr[j]; }
      if (n === `step${i}_z`) { z = model.jnt_qposadr[j]; dz = model.jnt_dofadr[j]; }
    }
    if (x < 0 || z < 0) return null;
    addr.push({ x, z, dx, dz });
  }
  return addr;
}

/** Hold every step still. Call after any qpos write, and every tick. */
function pin(data, a) { data.qvel[a.dx] = 0; data.qvel[a.dz] = 0; }

/**
 * Park every step far below the floor: a flat room.
 *
 * Spread along x as well as dropped, because parking them all at the same point
 * stacks fourteen boxes inside each other — they are on a shared collision bit,
 * so that alone produced 366 contacts a tick and cost real frame time.
 */
export function clearStairs(data, addr) {
  addr.forEach((a, i) => { data.qpos[a.x] = i * 1.5; data.qpos[a.z] = -5; pin(data, a); });
}

/**
 * A staircase running along +x, first riser at `start`.
 *
 * Each block is tall and solid rather than a thin tread on stilts: a foot that
 * catches should stub against something, not drop into a gap under it.
 */
export function layoutStairs(data, addr, { count, rise, run, start }) {
  const n = Math.max(0, Math.min(count, STAIR_COUNT));
  for (let i = 0; i < STAIR_COUNT; i++) {
    const a = addr[i];
    if (i >= n) { data.qpos[a.x] = i * 1.5; data.qpos[a.z] = -5; pin(data, a); continue; }
    const top = (i + 1) * rise;
    data.qpos[a.x] = start + i * run + STEP_HALF_DEPTH;
    data.qpos[a.z] = top - STEP_HALF_HEIGHT;   // block top lands on `top`
    pin(data, a);
  }
  return n;
}

/** Height of the tread the duck is standing over, for the HUD. */
export function groundUnder(x, { count, rise, run, start }) {
  if (x < start) return 0;
  const i = Math.floor((x - start) / run);
  if (i < 0) return 0;
  return Math.min(i + 1, count) * rise;
}
