// chase_rig.mjs — THE BALL CHALLENGE'S INSTRUMENT, on this desk.
//
// WHAT THIS FILE IS, AND WHAT IT IS NOT. It is NOT a second opinion about a
// chase episode: the episode, the criterion, the grid and every one of Pollen's
// nine transcribed reward terms live once, in `sim/chase_score.mjs`, and this
// file only supplies the machine — a MuJoCo module, the plant, an mjData, and
// onnxruntime sessions for the standing policy and for whatever network an
// entrant names. `chase/chase_robust.mjs` runs the fourteen cells over it, and
// `duckbench-core.mjs` answers POST /chase out of the same shared module with a
// different machine underneath.
//
// This is `climb/rig3.mjs`'s role in the stairs rail, kept deliberately thin:
// rig3 grew an audited record shape that had to be frozen, and the lesson of
// that is not to grow one here. A cell's answer is whatever `runEpisode`
// returned, plus the entrant's hash.
//
// THE STANDING POLICY IS `alpha_stand.onnx`, THE ROLE NAME, because that is the
// file `duckbench-core.mjs` resolves first (STAND_TRIED) and a chase scored
// under a different settle would not be the bench's cell.
// `BEST_alpha_stand.onnx` beside it is byte-identical today; the role name is
// the one that stays right if it stops being.
//
//   cd ~/projects/duck-sounds/sim && node ../chase/chase_rig.mjs
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { makeLoop } from '../site/duckloop.mjs';
import { declaredDefaultPoseOf } from '../sim/onnx_meta.mjs';
import { makeChaseRig, checkEntrant, entrantHashPayload, gridCells,
         DEFAULT_SECONDS } from '../sim/chase_score.mjs';

// FOUND BESIDE THIS FILE, NOT BESIDE THE SHELL THAT LAUNCHED IT. `climb/
// robust.mjs` reads its plant out of the working directory and therefore only
// runs from sim/; there is no reason for this one to inherit that.
const HERE = path.dirname(fileURLToPath(import.meta.url));
const SIM = path.join(HERE, '..', 'sim');
const at = name => path.join(SIM, name);

export const C = JSON.parse(fs.readFileSync(at('duckkit-constants.json'), 'utf8'));
export const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);

const mj = await load();
export const SCENE_BYTES = fs.readFileSync(at('scene.mjb'));
export const PLANT = 'scene.mjb';
export const PLANT_DIGEST = crypto.createHash('sha256').update(SCENE_BYTES).digest('hex');
mj.FS.writeFile('/c.mjb', new Uint8Array(SCENE_BYTES));
const model = mj.MjModel.mj_loadBinary('/c.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model);

/**
 * A session, the way `duckbench-node.mjs` makes one — from BYTES, with the
 * neutral pose the file declares.
 *
 * IT MUST BE MADE THE SAME WAY OR THE PARITY GATE IS MEASURING THE SHELL. A
 * reference read here and not there (or there and not here) moves every
 * observation by the difference and the two rigs would disagree about a duck
 * neither of them got wrong.
 */
const sessions = new Map();
export async function session(name) {
  if (sessions.has(name)) return sessions.get(name);
  const file = at(name);
  if (!fs.existsSync(file)) throw new Error(`unknown policy: ${name}`);
  const bytes = fs.readFileSync(file);
  const net = await ort.InferenceSession.create(bytes);
  const output = net.outputNames[0];
  const loaded = {
    name,
    reference: declaredDefaultPoseOf(bytes, HOME) ?? HOME,
    async run(obs) {
      const r = await net.run({ obs: new ort.Tensor('float32', obs, [1, 61]) });
      return r[output].data;
    },
  };
  sessions.set(name, loaded);
  return loaded;
}

export const STAND = 'alpha_stand.onnx';
const stand = await session(STAND);

/** THIS FILE'S RIG: its own model, its own mjData, its own onnxruntime. */
export const RIG = makeChaseRig({
  mj, model, data, D, HOME, LO, HI, jointNames: C.jointNames,
  buildObs, projectedGravity, command, tickHz: C.tickHz,
  stand: { run: obs => stand.run(obs), reference: stand.reference },
});
if (!RIG) throw new Error(`${PLANT} has no ball: this scorer cannot score a chase`);

/** sha256 over the normalised entrant — `chase_score.mjs`'s string, hashed here. */
export function entrantHash(e) {
  return crypto.createHash('sha256').update(entrantHashPayload(e)).digest('hex');
}

/** Read an entrant file and check its shape. Nothing scores an unchecked file. */
export function readEntrant(file) {
  // RELATIVE TO WHERE THE CALLER STANDS, never to this file: resolving against
  // chase/ handed a stranger who scored their own edited entrant from sim/ the
  // bundled control of the same name, silently. The climb harness resolves
  // against the working directory; so does this, and a path that resolves to
  // nothing is an error with the path in it.
  const full = path.isAbsolute(file) ? file : path.resolve(process.cwd(), file);
  if (!fs.existsSync(full)) {
    const bundled = path.join(HERE, path.basename(file));
    const hint = fs.existsSync(bundled)
      ? ` A bundled control of that name exists at ${bundled}; pass that path if it is the one you mean.`
      : '';
    throw new Error(`no entrant file at ${full} (relative paths resolve against the working directory).${hint}`);
  }
  return checkEntrant(JSON.parse(fs.readFileSync(full, 'utf8')), full);
}

/** How long an entrant runs: the file's own seconds, or the shared default. */
export const secondsOf = e => (e.seconds === undefined ? DEFAULT_SECONDS : +e.seconds);

/**
 * SCORE ONE CELL, FOR ONE ENTRANT ALREADY IN MEMORY.
 *
 * The record is exactly what the shared episode returned, plus the hash. There
 * is no local reshaping on purpose: `chase/chase_parity.mjs` compares this
 * against POST /chase field by field, and a shape assembled twice is a shape
 * that can disagree twice.
 */
export async function chaseCell(entrant, cell, { seconds } = {}) {
  checkEntrant(entrant, entrant.name || 'entrant');
  const actor = entrant.kind === 'policy' ? await session(entrant.policy) : null;
  const E = await RIG.runEpisode(entrant, cell, {
    seconds: seconds === undefined ? secondsOf(entrant) : seconds,
    tail: 'policy',
    actor: actor ? { run: obs => actor.run(obs), reference: actor.reference } : null,
  });
  E.hash = entrantHash(entrant);
  E.entrant = E.hash.slice(0, 12);
  E.name = entrant.name ?? null;
  E.kind = entrant.kind;
  E.policy = entrant.kind === 'policy' ? entrant.policy : null;
  E.plantName = PLANT;
  E.plantDigest = PLANT_DIGEST;
  return E;
}

/** The same, from a SAVED file — the screening call. A reported result comes
 *  from `chase_robust.mjs`'s whole grid, never from one cell. */
export async function scoreSaved(file, cell, opts = {}) {
  return chaseCell(readEntrant(file), cell, opts);
}

export { gridCells, model, data, mj, D };

// ============================================================ a smoke reading
const isMain = process.argv[1] && process.argv[1].endsWith('chase_rig.mjs');
if (isMain) {
  console.log(`=== chase_rig — plant ${PLANT} ${PLANT_DIGEST.slice(0, 12)} ===`);
  console.log(`   ball geom ${RIG.BALLG}, duck geoms ${RIG.DUCKG.length}, feet ${RIG.FEET.length}, `
            + `substeps ${RIG.SUBSTEPS}, root_angmom adr ${RIG.ANGMOM.adr}`);
  const cells = gridCells();
  const centre = cells[4];
  for (const file of ['ctrl_do_nothing.json', 'ctrl_alpha_walking.json']) {
    const e = readEntrant(file);
    const r = await chaseCell(e, centre);
    console.log(`   ${String(e.name).padEnd(26)} centre cell  chased=${r.chased} stable=${r.stable} `
      + `travel=${r.facts.ballTravel_mm.toFixed(1)}mm closest=${r.facts.closest_mm.toFixed(1)}mm `
      + `final=${r.facts.final_mm.toFixed(1)}mm upTail=${r.uprightTailTicks}/50`);
  }
}
