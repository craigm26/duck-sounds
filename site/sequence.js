// Sequences: a small program the duck can run.
//
// Every keyframe move in this project is a one-shot fired at a duck that
// happens to be standing in the right place. Nothing until now could GET it
// there, so six of the thirteen buttons were a coin flip on where the duck had
// wandered to. A sequence is the missing verb: put the room in a known state,
// walk to the mark, settle exactly the way the search settled, then fire.
//
// The interpreter is deliberately dumb — one step at a time, advanced once per
// animation frame from the same loop that steps the physics, no scheduler and
// no clock of its own. Physics time is the only time there is here, and a
// sequence that ran on wall-clock would drift away from it the moment a frame
// took longer than 20 ms.
import { planApproach } from './approach.js';
import { SETTLE_TICKS } from './intent-specs.js';

/**
 * Step kinds.
 *
 *   stairs  set the staircase        {rise, count, run}
 *   goto    walk to a pose           {x, y, yaw}
 *   place   teleport to a pose       {x, y}   — staging, not driving; see below
 *   settle  hold the stand policy    {ticks, approach}
 *   move    fire an intent           {id}
 *   drive   hold a command           {vx, vyaw, ticks}
 *   wait    do nothing               {ticks}
 *   reset   stand it back up         {}
 *
 * `place` is a teleport and is labelled as one everywhere it appears. It is
 * honest as a staging aid — it is exactly what the GIF recorder does, and it is
 * the only way to reproduce a searched move exactly — but a sequence that uses
 * it has not shown the duck getting anywhere. `goto` is the one that proves
 * something.
 */
export const KINDS = ['stairs', 'goto', 'place', 'settle', 'move', 'drive', 'wait', 'reset'];

export function describe(step) {
  const mm = v => Math.round(v * 1000) + ' mm';
  switch (step.kind) {
    case 'stairs': return step.rise > 0 ? `stairs ${Math.round(step.rise * 1000)} mm x${step.count}` : 'flat floor';
    case 'goto':   return `walk to ${mm(step.x)}, ${mm(step.y)}`;
    case 'place':  return `place at ${mm(step.x)}, ${mm(step.y)} (teleport)`;
    case 'settle': return `settle ${step.ticks} ticks`;
    case 'move':   return `${step.id.replace(/_/g, ' ')}`;
    case 'drive':  return `drive ${step.vx.toFixed(2)}/${step.vyaw.toFixed(2)} for ${step.ticks}`;
    case 'wait':   return `wait ${step.ticks}`;
    case 'reset':  return 'reset';
    default:       return step.kind;
  }
}

/**
 * Expand "do this move" into the steps that actually make it work.
 *
 * This is the whole point of the feature: a move is not one instruction, it is
 * set the room, get to the mark, settle, fire. `readiness` decides how much of
 * that is needed right now — an already-staged duck skips straight to firing.
 */
export function stageFor(id, ready, opts = {}) {
  const out = [];
  if (!ready || ready.ok) return [{ kind: 'move', id }];
  if (ready.fix === 'stairs' && ready.staging && ready.staging.kind === 'stair') {
    out.push({ kind: 'stairs', rise: ready.staging.rise, count: 4, run: 0.28 });
  }
  if (ready.target) {
    out.push(opts.teleport
      ? { kind: 'place', x: ready.target.x, y: ready.target.y }
      : { kind: 'goto', x: ready.target.x, y: ready.target.y, yaw: ready.target.yaw });
    out.push({ kind: 'settle', ticks: SETTLE_TICKS, approach: ready.staging ? ready.staging.approach : 0 });
  }
  out.push({ kind: 'move', id });
  return out;
}

/**
 * The interpreter.
 *
 * `host` is everything it needs from the simulator, passed in rather than
 * imported so this file has no opinion about the DOM and can be exercised
 * headlessly.
 */
export function makeRunner(host) {
  let program = [], pc = -1, state = 'idle', note = '', plan = null, until = 0;

  const stop = (why = '') => {
    plan = null;
    host.drive(null);
    state = why ? 'failed' : 'idle';
    note = why;
    host.onChange && host.onChange();
  };

  async function enter(step) {
    note = describe(step);
    switch (step.kind) {
      case 'stairs': host.setStairs(step); return true;
      case 'reset':  host.reset(); return true;
      case 'place':  host.place(step.x, step.y); return true;
      case 'wait':   until = host.ticks() + step.ticks; return true;
      case 'drive':  until = host.ticks() + step.ticks; host.drive({ vx: step.vx, vyaw: step.vyaw }); return true;
      case 'settle': await host.startSettle(step.ticks ?? SETTLE_TICKS, step.approach ?? 0); return true;
      case 'goto': {
        const t = { x: step.x, y: step.y, yaw: step.yaw ?? 0 };
        plan = planApproach(t, host.tol(), host.speed());
        return true;
      }
      case 'move': {
        const ok = await host.fire(step.id);
        if (!ok) { stop(`could not fire ${step.id}`); return false; }
        return true;
      }
      default: return true;
    }
  }

  /** @returns true when the current step is finished. */
  function settled(step) {
    switch (step.kind) {
      case 'wait':
      case 'drive':  return host.ticks() >= until;
      case 'settle': return !host.settling();
      case 'move':   return !host.isBusy();
      case 'goto': {
        const r = plan.step(host.pose());
        note = r.note;
        if (r.failed) { stop(r.note); return false; }
        host.drive(r.done ? null : { vx: r.vx, vyaw: r.vyaw });
        return r.done;
      }
      // stairs / place / reset all take effect the instant they are entered,
      // but give the solver a frame to see the new world before moving on.
      default: return true;
    }
  }

  return {
    get state() { return state; },
    get note() { return note; },
    get pc() { return pc; },
    get program() { return program.slice(); },
    load(steps) { program = steps.slice(); pc = -1; state = 'idle'; note = ''; },
    async play(steps) {
      if (steps) program = steps.slice();
      if (!program.length) return;
      pc = -1; state = 'running'; note = '';
      host.onChange && host.onChange();
      await advance();
    },
    stop: () => { pc = -1; stop(); },
    /** Called once per animation frame from the physics loop. */
    async tick() {
      if (state !== 'running' || pc < 0 || pc >= program.length) return;
      if (settled(program[pc])) await advance();
    },
  };

  async function advance() {
    plan = null;
    host.drive(null);
    pc++;
    if (pc >= program.length) {
      state = 'idle'; pc = -1; note = 'done';
      host.onChange && host.onChange();
      return;
    }
    host.onChange && host.onChange();
    await enter(program[pc]);
  }
}
