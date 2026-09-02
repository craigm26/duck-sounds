// A Microduck policy read into plain arrays, run forward in JS, and folded.
//
// WHY THIS EXISTS RATHER THAN `ort.InferenceSession`. onnxruntime runs a graph;
// it does not let you change one. The tuning search this file serves evaluates
// thousands of candidate policies that differ only in the last Gemm — a
// per-joint gain on its rows and a per-joint trim on its biases — and the only
// ways to run those through onnxruntime are to re-serialise an ONNX file per
// candidate (a 790 KB write and a session build, per episode) or to apply the
// residual outside the network and hope that is the same thing. It is the same
// thing, and `DuckPolicyWriterFoldTests` in duckkit proves it to 1e-5 at every
// state of a 200-tick closed run — but the search should not have to lean on
// that proof to be correct. Folding here, into a 14×128 matrix, costs 1792
// multiplies per candidate and means the network being searched over IS the
// network that gets written.
//
// THE ARCHITECTURE IS FIXED AND CHECKED, the same nine ops DuckKit's loader
// insists on: (obs − mean) / std, then 61×512 → ELU → 512×256 → ELU → 256×128 →
// ELU → 128×14, with no ELU after the last Gemm. That last fact is what makes
// the fold exact: the final Gemm's output IS the action, so
// `gain ⊙ (W·h + b) + offset` is `(diag(gain)·W)·h + (gain ⊙ b + offset)` —
// another Gemm of the same shape. Fold anywhere else and an ELU sits in the
// way, and ELU does not commute with a scale.
//
// PROVED AGAINST onnxruntime, NOT ASSUMED. `verifyAgainst` runs both on the
// same observations and reports the worst disagreement; `tune_es.mjs` calls it
// before it searches, because a forward pass that quietly differs from the
// runtime everything else in this repo uses would make every number downstream
// a number about a different network.
import fs from 'node:fs';

// ── protobuf, the two wire types an ONNX file needs ──────────────────────────
// Same walk as onnx_meta.mjs, kept here so this file stands alone; that one
// reads metadata_props and this one reads initializers.

function varint(b, i) {
  let v = 0, s = 0;
  for (;;) {
    const x = b[i++];
    v += (x & 0x7f) * 2 ** s;
    if (!(x & 0x80)) return [v, i];
    s += 7;
  }
}

function* fields(b, start = 0, end = b.length) {
  let i = start;
  while (i < end) {
    let key; [key, i] = varint(b, i);
    const f = key >>> 3, wt = key & 7;
    if (wt === 0) { let v; [v, i] = varint(b, i); yield [f, wt, v]; }
    else if (wt === 2) {
      let n; [n, i] = varint(b, i);
      yield [f, wt, b.subarray(i, i + n)]; i += n;
    } else if (wt === 5) { yield [f, wt, b.subarray(i, i + 4)]; i += 4; }
    else if (wt === 1) { yield [f, wt, b.subarray(i, i + 8)]; i += 8; }
    else return;
  }
}

const str = p => Buffer.from(p).toString('utf8');

/** GraphProto.node: op_type (4), inputs (1), output (2). */
function readNode(p) {
  const node = { op: '', inputs: [] };
  for (const [f, wt, v] of fields(p)) {
    if (f === 1 && wt === 2) node.inputs.push(str(v));
    else if (f === 4 && wt === 2) node.op = str(v);
  }
  return node;
}

/** TensorProto: dims (1), data_type (2), name (8), raw_data (9). */
function readTensor(p) {
  const t = { dims: [], dataType: 0, name: '', floats: null };
  for (const [f, wt, v] of fields(p)) {
    if (f === 1 && wt === 0) t.dims.push(v);
    else if (f === 2 && wt === 0) t.dataType = v;
    else if (f === 8 && wt === 2) t.name = str(v);
    else if (f === 9 && wt === 2) {
      // A Buffer subarray is not necessarily 4-byte aligned, and Float32Array
      // over a misaligned offset throws. Copy rather than gamble.
      const copy = Buffer.from(v);
      t.floats = new Float32Array(copy.buffer, copy.byteOffset, copy.length >> 2);
    }
  }
  return t;
}

const WIDTHS = [[61, 512], [512, 256], [256, 128], [128, 14]];
const OPS = ['Sub', 'Div', 'Gemm', 'Elu', 'Gemm', 'Elu', 'Gemm', 'Elu', 'Gemm'];

/**
 * Read a policy file into `{ mean, std, layers }`, layers outermost first with
 * weights row-major `[outputs][inputs]` — the ONNX transB=1 convention and
 * DuckKit's own storage order, so a fold written here means the same thing a
 * fold written in Swift does.
 *
 * Constants are resolved THROUGH THE GRAPH, not by name, exactly as
 * `DuckPolicy.load` does: the mean is the second input of the Sub, the std the
 * second input of the Div, and each Gemm names its own weight and bias. A
 * re-exported file with different tensor names still loads.
 */
export function readPolicy(file) {
  const bytes = fs.readFileSync(file);
  let graph = null;
  for (const [f, wt, v] of fields(bytes)) if (f === 7 && wt === 2) graph = v;
  if (!graph) throw new Error(`${file}: no graph`);

  const nodes = [], byName = new Map();
  for (const [f, wt, v] of fields(graph)) {
    if (f === 1 && wt === 2) nodes.push(readNode(v));
    else if (f === 5 && wt === 2) { const t = readTensor(v); byName.set(t.name, t); }
  }
  const ops = nodes.map(n => n.op);
  if (ops.join(',') !== OPS.join(',')) {
    throw new Error(`${file}: op sequence ${ops.join(',')} is not a Microduck policy`);
  }
  const grab = (name, what) => {
    const t = byName.get(name);
    if (!t) throw new Error(`${file}: missing initializer ${name} (${what})`);
    if (t.dataType !== 1) throw new Error(`${file}: ${name} is dtype ${t.dataType}, not float32`);
    return t;
  };
  const mean = grab(nodes[0].inputs[1], 'normalizer mean');
  const std = grab(nodes[1].inputs[1], 'normalizer std');
  if (mean.floats.length !== 61 || std.floats.length !== 61) {
    throw new Error(`${file}: normalizer is ${mean.floats.length}/${std.floats.length} wide, not 61`);
  }
  const layers = [];
  for (const node of nodes) {
    if (node.op !== 'Gemm') continue;
    const [inputs, outputs] = WIDTHS[layers.length];
    const w = grab(node.inputs[1], 'weight'), b = grab(node.inputs[2], 'bias');
    if (w.floats.length !== inputs * outputs || b.floats.length !== outputs) {
      throw new Error(`${file}: layer ${layers.length} is ${w.floats.length}/${b.floats.length}, `
                    + `not ${inputs * outputs}/${outputs}`);
    }
    layers.push({ inputs, outputs, weights: Float32Array.from(w.floats),
                  biases: Float32Array.from(b.floats) });
  }
  return { mean: Float32Array.from(mean.floats), std: Float32Array.from(std.floats), layers };
}

/**
 * One forward pass, float32 throughout, into a caller-owned output array.
 *
 * `Math.fround` on every accumulation would be the pedantic thing and is not
 * what onnxruntime does either: it accumulates a dot product in whatever the
 * kernel's precision is and rounds on store. Writing into Float32Array rounds
 * on store, which is the same discipline, and `verifyAgainst` is how that claim
 * gets checked rather than argued.
 */
export function makeForward(p) {
  const buffers = p.layers.map(l => new Float32Array(l.outputs));
  const x0 = new Float32Array(61);
  return function forward(obs, out = null) {
    for (let i = 0; i < 61; i++) x0[i] = (obs[i] - p.mean[i]) / p.std[i];
    let x = x0;
    for (let li = 0; li < p.layers.length; li++) {
      const l = p.layers[li], y = buffers[li], W = l.weights, n = l.inputs;
      for (let j = 0; j < l.outputs; j++) {
        let sum = l.biases[j];
        const row = j * n;
        for (let i = 0; i < n; i++) sum += W[row + i] * x[i];
        // ELU (α = 1) on every layer but the last — the ninth op is a Gemm and
        // the chain ends there.
        y[j] = li === p.layers.length - 1 ? sum : (sum < 0 ? Math.expm1(sum) : sum);
      }
      x = y;
    }
    if (out) { for (let j = 0; j < x.length; j++) out[j] = x[j]; return out; }
    return x;
  };
}

/**
 * `policy` with `gain ⊙ a + offset` folded into its last Gemm. 14-wide, policy
 * order, mouth excluded — the mouth has no output row to scale.
 *
 * The returned object shares the first three layers with `policy`: only the
 * last one is copied. A search that reallocated 197,774 floats per candidate
 * would spend most of its time in the allocator.
 */
export function fold(policy, gain, offset) {
  const last = policy.layers[policy.layers.length - 1];
  if (gain.length !== last.outputs || offset.length !== last.outputs) {
    throw new Error(`the gain and offset must be ${last.outputs} wide, mouth excluded`);
  }
  const W = new Float32Array(last.weights), b = new Float32Array(last.outputs);
  for (let j = 0; j < last.outputs; j++) {
    const g = gain[j], row = j * last.inputs;
    for (let i = 0; i < last.inputs; i++) W[row + i] *= g;
    b[j] = last.biases[j] * g + offset[j];
  }
  return {
    mean: policy.mean, std: policy.std,
    layers: [...policy.layers.slice(0, -1),
             { inputs: last.inputs, outputs: last.outputs, weights: W, biases: b }],
  };
}

/**
 * The worst disagreement between this forward pass and onnxruntime's over
 * `count` pseudo-random observations, and the worst between a folded network
 * and the residual applied outside the base one.
 *
 * The observations are drawn around the policy's own training statistics rather
 * than from a uniform box: a network judged on inputs 30 standard deviations
 * out of distribution is being judged where nothing about its agreement
 * matters.
 */
export async function verifyAgainst(session, ort, policy, { count = 64, seed = 7 } = {}) {
  const forward = makeForward(policy);
  let s = seed >>> 0;
  const rand = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  const gain = Array.from({ length: 14 }, () => 0.8 + rand() * 0.5);
  const offset = Array.from({ length: 14 }, () => (rand() - 0.5) * 0.1);
  const folded = fold(policy, gain, offset);
  const forwardFolded = makeForward(folded);
  const IN = session.inputNames[0], OUT = session.outputNames[0];
  let worstRuntime = 0, worstFold = 0;
  for (let n = 0; n < count; n++) {
    const obs = new Float32Array(61);
    for (let i = 0; i < 61; i++) obs[i] = policy.mean[i] + policy.std[i] * (rand() * 4 - 2);
    const mine = Array.from(forward(obs));
    const theirs = Array.from((await session.run({ [IN]: new ort.Tensor('float32', obs, [1, 61]) }))[OUT].data);
    const tuned = Array.from(forwardFolded(obs));
    for (let k = 0; k < 14; k++) {
      worstRuntime = Math.max(worstRuntime, Math.abs(mine[k] - theirs[k]));
      worstFold = Math.max(worstFold, Math.abs(tuned[k] - (gain[k] * theirs[k] + offset[k])));
    }
  }
  return { observations: count, worstRuntime, worstFold };
}
