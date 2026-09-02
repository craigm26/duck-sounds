// Does the hand-written forward pass answer what onnxruntime answers?
//
// WHY THIS IS THE GATE THAT MATTERS. `policyforward.mjs` is 60 lines of
// arithmetic that will drive a real robot's joints. A transposed weight matrix,
// an ELU applied one layer too far, a normalizer read in the wrong order — none
// of those crash, and all of them produce a duck that stands there twitching
// while every log says the network ran. The only honest test is the one that
// runs the SAME 197,774 numbers through onnxruntime and through this, over
// observations in the range the policy actually sees, and compares every one of
// the fourteen outputs.
//
// THE OBSERVATIONS ARE NOT UNIFORM NOISE. They are drawn as mean + std·z, so
// the network's INPUT after normalisation is standard normal — the regime it was
// trained in and the only regime where an ELU's two branches are both exercised.
// Uniform [0,1) noise on a 61-slot vector whose training std spans 0.0129 to
// 3.03 would sit hundreds of sigma out on some slots and zero on others, and a
// pass there proves nothing about the arithmetic that runs the duck.
//
// THE SEED IS FIXED so a failure is a failure somebody else can reproduce.
//
// AND THE FLOOR IS MEASURED, BECAUSE A TOLERANCE NOBODY PROBED IS A NUMBER
// SOMEBODY PICKED. Baseline disagreement between the two implementations is
// 8e-7 to 3.5e-6 depending on the policy — float32 accumulation in a different
// order, nothing more. Deliberately breaking this file to see what the gate
// catches: an ELU applied after the LAST Gemm moves an action by 0.83 and is
// caught instantly; a 1e-4 error in ONE of the 512 first-layer biases produces
// a worst disagreement of 5.4e-6 and PASSES. So this gate catches structural
// mistakes and not small numeric ones. `physics_parity.mjs` is the one that
// catches those, because 250 ticks of closed loop turn 3.5e-6 per action into
// 32 mm of trunk position.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as ort from 'onnxruntime-node';
import { loadParameters, forward, OBS_WIDTH } from './policyforward.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const TOLERANCE = 1e-5;
const SAMPLES = 200;

/** mulberry32: small, seeded, and the same sequence on every machine. */
function rng(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
/** Box-Muller, so the normalized input is genuinely standard normal. */
function normal(next) {
  const u = Math.max(next(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next());
}

/** Every policy that has a .bin beside its .onnx, by the name /policy takes. */
function pairs() {
  const out = [];
  for (const file of fs.readdirSync(HERE).sort()) {
    if (!file.endsWith('.onnx') || file.startsWith('uploaded-')) continue;
    const bin = path.join(HERE, 'params', file.replace(/\.onnx$/, '.bin'));
    if (fs.existsSync(bin)) out.push({ name: file, onnx: path.join(HERE, file), bin });
  }
  const community = path.join(HERE, 'community');
  if (fs.existsSync(community)) {
    for (const dir of fs.readdirSync(community).sort()) {
      const onnx = path.join(community, dir, 'policy.onnx');
      const bin = path.join(HERE, 'params', `${dir}-policy.bin`);
      if (fs.existsSync(onnx) && fs.existsSync(bin)) out.push({ name: `${dir}/policy.onnx`, onnx, bin });
    }
  }
  return out;
}

const only = process.argv.includes('--only') ? process.argv[process.argv.indexOf('--only') + 1] : null;
const files = pairs().filter(p => !only || p.name === only);
if (!files.length) { console.error('no policy has parameter bytes beside it — run dumpparams'); process.exit(2); }

let worst = 0, worstAt = '', failures = 0;
for (const file of files) {
  const params = loadParameters(fs.readFileSync(file.bin));
  const net = await ort.InferenceSession.create(file.onnx);
  const output = net.outputNames[0];
  const next = rng(20260901);
  let biggest = 0, msTheirs = 0, msOurs = 0;
  for (let s = 0; s < SAMPLES; s++) {
    const obs = new Float32Array(OBS_WIDTH);
    for (let i = 0; i < OBS_WIDTH; i++) obs[i] = params.mean[i] + params.std[i] * normal(next);
    let t = performance.now();
    const answer = await net.run({ obs: new ort.Tensor('float32', obs, [1, 61]) });
    msTheirs += performance.now() - t;
    const theirs = answer[output].data;
    t = performance.now();
    const ours = forward(params, obs);
    msOurs += performance.now() - t;
    for (let k = 0; k < theirs.length; k++) {
      const d = Math.abs(theirs[k] - ours[k]);
      if (d > biggest) biggest = d;
      if (d > worst) { worst = d; worstAt = `${file.name} sample ${s} action ${k}`; }
    }
  }
  const ok = biggest < TOLERANCE;
  if (!ok) failures++;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${file.name.padEnd(34)} max |Δ| ${biggest.toExponential(2)}`
            + `   onnxruntime ${(msTheirs / SAMPLES).toFixed(3)} ms   policyforward ${(msOurs / SAMPLES).toFixed(3)} ms`);
}
console.log(`\n${files.length} policies × ${SAMPLES} observations × 14 actions `
          + `= ${files.length * SAMPLES * 14} comparisons`);
console.log(`worst disagreement anywhere: ${worst.toExponential(3)} at ${worstAt} (tolerance ${TOLERANCE})`);
if (failures) { console.log(`${failures} POLICIES FAILED`); process.exitCode = 1; }
else console.log('POLICY PARITY OK');
