// One number the phone has to reproduce.
//
// WHY A TRUNK POSITION AND NOT A TEST SUITE. The claim being made is that the
// bench in a browser on an iPhone is the SAME bench as the one on the desk —
// that a measurement taken on the phone is comparable to one taken here. Every
// structural check in the world (the endpoints answer, the shapes match,
// /health says duck-bench/5) can pass while the two run different physics, and
// the way you find out is a trajectory. A closed loop is a chaotic amplifier:
// a policy that disagrees in the seventh decimal moves a joint that changes a
// contact that moves the whole duck, and by 250 ticks the difference is
// millimetres, not microvolts. So the gate is: drop the duck the same way,
// steer it at the same twist for five seconds, and say where its trunk ended
// up.
//
// IT RUNS THE BROWSER'S FORWARD PASS ON PURPOSE. `--engine onnxruntime` is the
// desk bench; `--engine policyforward` (the default here) is the arithmetic
// `duckbench-web.mjs` runs, so the only thing left different between this
// number and the phone's is the WebAssembly engine underneath — and the .wasm
// is byte-identical (md5 c08b79f7… in both sim/node_modules/mujoco and
// site/vendor), so it should agree exactly rather than approximately.
//
// THE BENCH ROUNDS ITS ANSWERS TO 1e-4 and this reads them through the same
// public /state that everything else does, so the last two of the six decimals
// printed are structurally zero. That is stated rather than hidden: the gate
// is "agrees to the bench's own quantum", and at 1e-4 over a metre of walking
// it is still a gate that a wrong integrator, a wrong plant or a wrong policy
// cannot pass.
import { nodeBench } from './duckbench-node.mjs';

const arg = (name, fallback) => {
  const i = process.argv.indexOf(name);
  return i < 0 ? fallback : process.argv[i + 1];
};
const engine = arg('--engine', 'policyforward');

/**
 * The script the page runs too, kept in one place so the two cannot drift.
 *
 * WALKING AND TURNING AT ONCE, AND NOT SLOWLY. The first draft of this gate
 * steered at vx 0.2 and the duck ended 13 mm from where it started — the bench
 * drives the training path (target = reference + action, scale 1.0, no
 * low-pass), and under that the alpha walker marches in place below about
 * 0.3 m/s. A gate whose duck does not move is a gate that compares two
 * standing ducks and passes whatever the physics did. At vx 0.5 with vyaw 0.8
 * it covers 0.84 m along a curve in five seconds, upright at the end, and
 * every contact along the way is a place the two engines could diverge.
 */
export const TWIST = { vx: 0.5, vy: 0, vyaw: 0.8 };
export const TICKS = 250;
export const HOLD = 0.1;              // 5 ticks a call, 50 calls
export const POLICY = 'alpha_walking.onnx';

export async function trunkAfterRun(handle) {
  const call = (path, body) => handle(new URL('http://bench.local' + path), body ?? {});
  await call('/reset', {});
  await call('/policy', { policy: POLICY });
  let state;
  for (let i = 0; i < TICKS / (HOLD * 50); i++) state = await call('/intent', { ...TWIST, hold: HOLD });
  return state;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const bench = await nodeBench({ engine });
  const state = await trunkAfterRun(bench.handle);
  const six = n => n.toFixed(6);
  console.log(`plant       ${bench.plant.name} ${bench.plant.digest.slice(0, 16)}`);
  console.log(`stand       ${bench.stand}`);
  console.log(`engine      ${engine}`);
  console.log(`script      /reset, /policy ${POLICY}, then ${TICKS} ticks at `
            + `vx ${TWIST.vx} vy ${TWIST.vy} vyaw ${TWIST.vyaw}`);
  console.log(`t           ${six(state.t)} s`);
  console.log(`trunk       ${state.position.map(six).join('  ')}`);
  console.log(`height      ${six(state.height)}   upright ${state.upright}`);
  console.log(`EXPECTED    ${state.position.map(six).join(' ')}`);
}
