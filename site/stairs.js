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
/**
 * Where the staircase sits across the room.
 *
 * Stairs go against a wall — that is what stairs do, and it also gives the duck
 * a vertical surface beside the steps to brace on. This value was derived as
 * 1.5 - 0.025 - 0.17 on the belief that the north wall is 25 mm half-thick.
 * IT IS NOT: sim/scene_physics.xml gives wall_n a half-thickness of 0.05, so
 * its inner face is at y = 1.45 and the outer 25 mm of every tread sits INSIDE
 * the wall (measured 2026-09-02, climb/audit_r2). The number stays because
 * the step bodies are COMPILED at this y with x and z slides only — there is
 * no y joint to move them — so this is where the blocks actually are, and the
 * lateral gate has to be measured from where they are, not from where they
 * should be. Moving them means recompiling the scene with body y = 1.28.
 */
export const STAIR_HALF_WIDTH = 0.17;
export const STAIR_Y = 1.5 - 0.025 - STAIR_HALF_WIDTH;

/** Half-depth of a step block, metres. Runs longer than 2x this leave gaps. */
export const STEP_HALF_DEPTH = 0.17;   // deeper than the deepest run, so steps overlap into one solid flight
/**
 * Half-height of a step block.
 *
 * 200 mm tall, positioned so the TOP is the tread — so it is solid from below
 * the floor up to the step. That is what a stair looks like, and it is also the
 * only way there is a riser face to push against: a floating tread has nothing
 * behind it. The earlier 600 mm version swallowed the room; this is scaled to a
 * 250 mm duck.
 *
 * Older note, kept because it explains the collision bits:
 *
 * They were 0.30 — tall enough to be solid to the floor — which is invisible in
 * physics but not on screen: rendered, fourteen 60 cm slabs swallowed the room.
 * They can be thin because steps and the floor sit on different collision bits
 * and never meet, so a tread floating above the floor costs nothing and reads
 * as a step marker rather than a wall.
 */
export const STEP_HALF_HEIGHT = 0.10;

/**
 * qpos AND dof addresses for each step's [x, z] joints, looked up once.
 *
 * The dof addresses matter as much as the qpos ones: a step is a heavy body on
 * a frictionless slide, so setting its position every tick without also zeroing
 * its velocity leaves the solver believing it is travelling. It then behaves
 * like a catapult — measured, it threw the duck half a metre into the air.
 */
export function findStairJoints(model, { isolate = true } = {}) {
  if (isolate) isolateSteps(model);
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

/**
 * Stop the step blocks colliding with EACH OTHER. They still meet the duck,
 * the walls and any prop.
 *
 * THE FLIGHT USED TO SHOVE ITSELF APART. Each block is 200 mm tall with its
 * top at the tread, so at any rise under 200 mm adjacent blocks interpenetrate
 * by (200 - rise) mm in z — and by 60 mm in x by design, since a 340 mm-deep
 * block on a 280 mm run is what makes a solid flight. They shipped on the same
 * collision bit (contype 4, conaffinity 4), so they collided, and being 200 kg
 * bodies on frictionless slides the solver pushed them apart: up to 20 mm of
 * tread drift and 16 mm of sag inside ONE control tick, teleported back by
 * `layoutStairs` fifty times a second. Below about 150 mm of rise a duck simply
 * STANDING on the first tread was thrown to the floor within ten ticks, and
 * every stair result measured before 2026-09-02 was measured on that
 * (climb/rig3.log, Phases D and E). Zeroing each step geom's conaffinity is the
 * surgical repair: step-step becomes (4 & 0) = 0, while step-duck (4 & 5),
 * step-wall (4 & 5) and step-prop keep colliding exactly as before, because
 * contype is untouched. No mass, friction, gain or timestep moves. Runs from
 * `findStairJoints` so every consumer of the bank — the browser sim, the climb
 * harness, the audits — gets the same flight; pass `{ isolate: false }` to see
 * the broken one on purpose.
 */
export function isolateSteps(model) {
  let n = 0;
  for (let g = 0; g < model.ngeom; g++) {
    const name = model.geom(g).name || '';
    if (/^step\d+_geom$/.test(name)) { model.geom_conaffinity[g] = 0; n++; }
  }
  return n;
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
export function layoutStairs(data, addr, { count, rise, run, start, y = 0 }) {
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
