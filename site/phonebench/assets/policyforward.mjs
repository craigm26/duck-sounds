// The policy's forward pass, over the canonical parameter bytes.
//
// WHY THIS EXISTS AT ALL. A bench in a browser has MuJoCo — it compiles to
// WebAssembly — and it does not have onnxruntime in any form that is worth
// shipping: onnxruntime-web is a 14 MB WASM blob that wants threads and SIMD
// and a cross-origin-isolated page, which is three more things to get right on
// an iPhone than the network itself is. And the network is FOUR MATRIX
// MULTIPLIES. 61→512→256→128→14, ELU between them, nothing else: no
// convolution, no attention, no batching, no dynamic shapes. 197,774 numbers
// and about 200,000 multiply-adds a tick.
//
// IT DOES NOT PARSE ONNX, AND THAT IS THE POINT. The bytes it reads are
// `DuckPolicy.canonicalParameterBytes` — duckkit's own definition of what a
// policy IS, stripped of producer strings, initializer names and graph order:
// the normalizer mean, then the normalizer standard deviation, then for each
// layer outermost-first its weights and then its biases, every value a
// little-endian IEEE-754 binary32. That layout is already the thing DuckEvidence
// fingerprints a policy by, so the phone runs exactly the numbers the app
// attested to, and a second ONNX reader — the classic place for two
// implementations to quietly disagree — never gets written.
//
// THE LAYER WIDTHS ARE NOT IN THE FILE. `canonicalParameterBytes` carries
// numbers and no shape, because DuckPolicy.load refuses any architecture but
// this one, so the shape is a constant on both sides rather than a header. A
// file of the wrong length is refused here rather than reshaped into something
// plausible.

/** Outermost first, as `DuckPolicy.expectedWidths` states them. */
export const WIDTHS = [[61, 512], [512, 256], [256, 128], [128, 14]];
export const OBS_WIDTH = 61;
export const ACTION_WIDTH = 14;
/** 61 + 61 + Σ(in·out + out) = 197,896 floats, 791,584 bytes. */
export const FLOAT_COUNT = 2 * OBS_WIDTH + WIDTHS.reduce((n, [i, o]) => n + i * o + o, 0);

/**
 * Read the canonical bytes into the arrays a forward pass wants.
 *
 * Little-endian is asserted rather than assumed: `DataView` would be the
 * portable-but-slow way, and every machine this runs on is little-endian, so
 * the fast path is taken and the assumption is checked once.
 */
export function loadParameters(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (u8.byteLength !== FLOAT_COUNT * 4) {
    throw new Error(`policy parameters are ${u8.byteLength} bytes; this architecture needs ${FLOAT_COUNT * 4}`);
  }
  // 1.0f is 0x3F800000; on a little-endian host that 0x3f is the LAST byte.
  // Written the other way round first, this check fired on the Pi and refused
  // every policy — which is at least the failure that says so out loud.
  if (new Uint8Array(new Float32Array([1]).buffer)[0] === 0x3f) {
    // A big-endian host would read every weight byte-reversed and produce a
    // duck that falls over for reasons no log would explain.
    throw new Error('this host is big-endian: the canonical bytes are little-endian float32');
  }
  // COPIED, NOT VIEWED. A Uint8Array from `fetch` is not guaranteed to start on
  // a four-byte boundary, and Float32Array over an unaligned offset throws.
  const all = new Float32Array(u8.slice().buffer);
  let at = 0;
  const take = n => all.subarray(at, at += n);
  const mean = take(OBS_WIDTH), std = take(OBS_WIDTH);
  const layers = WIDTHS.map(([inputs, outputs]) => ({
    inputs, outputs, weights: take(inputs * outputs), biases: take(outputs),
  }));
  for (let i = 0; i < std.length; i++) {
    if (!(Number.isFinite(std[i]) && std[i] !== 0)) {
      throw new Error(`normalizer std[${i}] is ${std[i]} — dividing by it would poison every inference`);
    }
  }
  return { mean, std, layers };
}

/**
 * (obs − mean) / std, then the MLP. ELU between the layers and NOT after the
 * last one: the final Gemm's output is the action, and squashing it would clip
 * every negative joint command to −1.
 */
export function forward(params, observation) {
  const { mean, std, layers } = params;
  let x = new Float32Array(OBS_WIDTH);
  for (let i = 0; i < OBS_WIDTH; i++) x[i] = (observation[i] - mean[i]) / std[i];
  for (let l = 0; l < layers.length; l++) {
    const { weights, biases, inputs, outputs } = layers[l];
    const out = new Float32Array(outputs);
    for (let row = 0; row < outputs; row++) {
      // The same row-major `[outputs][inputs]` order duckkit accumulates in, so
      // the two cannot drift from one another in the last bits of a float.
      const base = row * inputs;
      let acc = 0;
      for (let col = 0; col < inputs; col++) acc += weights[base + col] * x[col];
      const v = acc + biases[row];
      // ELU, α = 1: identity above zero, exp(x) − 1 below. Not on the last layer.
      out[row] = (l < layers.length - 1 && v < 0) ? Math.expm1(v) : v;
    }
    x = out;
  }
  return x;
}

/** A policy, ready to run: the shape `duckbench-core.mjs` asks `makeSession` for. */
export function makeForwardSession(bytes, name) {
  const params = loadParameters(bytes);
  return { name, run: observation => forward(params, observation) };
}

/**
 * `gain ⊙ action + offset`, ABSORBED INTO THE LAST LAYER — a transcription of
 * duckkit's `DuckPolicyWriter.folding`, which is the definition of what a gain
 * and a trim mean on a Microduck policy.
 *
 * WHY IT IS HERE AND NOT WHERE THE SEARCH IS. `/tune` scores a candidate the
 * app will later fold into a file with the Swift writer, and the only way those
 * two can be the same network is for the bench to apply the identical
 * arithmetic. The last Gemm is the last op in the graph — no ELU after it — so
 * its output IS the action, and
 *
 *     a' = gain ⊙ (W·h + b) + offset = (diag(gain)·W)·h + (gain ⊙ b + offset)
 *
 * is another Gemm of the same shape. Row `j` of `W` scaled by `gain[j]`, bias
 * `j` scaled and shifted. Fold anywhere else and an ELU sits in the way, and
 * ELU does not commute with a scale.
 *
 * EVERY ROUNDING IS float32, IN THE SAME ORDER SWIFT DOES THEM, and this is
 * the whole reason `Math.fround` appears three times in six lines. Swift's
 * `weights[i] *= Float(gain[j])` rounds the gain to binary32 FIRST and then
 * multiplies two binary32s; JavaScript would multiply the binary32 weight by a
 * full binary64 gain and round once at the store, which is a different number
 * in the last bit for a gain like 1.07 that binary32 cannot hold. The bias is
 * two operations in Swift — a multiply, then an add — so it is two roundings
 * here as well, rather than one rounding of a fused expression.
 *
 * THE FIRST THREE LAYERS ARE SHARED, NOT COPIED. A search that reallocated
 * 197,774 floats per candidate would spend most of its time in the allocator,
 * and nothing here writes to them.
 *
 * `gain` and `offset` are indexed by POLICY SLOT — fourteen wide, mouth
 * excluded, because there is no row of `W` that belongs to the mouth.
 */
export function foldParameters(params, gain, offset) {
  const last = params.layers[params.layers.length - 1];
  if (gain.length !== last.outputs || offset.length !== last.outputs) {
    throw new Error(`the gain and the trim must be ${last.outputs} wide, mouth excluded`);
  }
  for (let j = 0; j < last.outputs; j++) {
    if (!Number.isFinite(gain[j]) || !Number.isFinite(offset[j])) {
      throw new Error('the gain or the trim holds something that is not a number: a fold is '
                    + 'arithmetic on every weight in the last layer, and one NaN in it makes a '
                    + 'network that loads and drives nothing');
    }
  }
  const weights = new Float32Array(last.weights);
  const biases = new Float32Array(last.outputs);
  for (let j = 0; j < last.outputs; j++) {
    const g = Math.fround(gain[j]), o = Math.fround(offset[j]);
    const row = j * last.inputs;
    for (let i = 0; i < last.inputs; i++) weights[row + i] = Math.fround(weights[row + i] * g);
    biases[j] = Math.fround(Math.fround(last.biases[j] * g) + o);
  }
  return {
    mean: params.mean, std: params.std,
    layers: [...params.layers.slice(0, -1),
             { inputs: last.inputs, outputs: last.outputs, weights, biases }],
  };
}

/**
 * The canonical bytes back out, in duckkit's own layout: mean, std, then each
 * layer's weights and biases, outermost first, little-endian binary32.
 *
 * IT EXISTS TO BE COMPARED, NOT TO BE SHIPPED. The fold above claims to be the
 * same arithmetic as the Swift writer's, and the only way to check a claim like
 * that is to put both results side by side as bytes — which means this side has
 * to be able to produce bytes. Nothing in the bench writes a policy file.
 */
export function canonicalBytes(params) {
  const all = new Float32Array(FLOAT_COUNT);
  let at = 0;
  const put = a => { all.set(a, at); at += a.length; };
  put(params.mean); put(params.std);
  for (const layer of params.layers) { put(layer.weights); put(layer.biases); }
  if (at !== FLOAT_COUNT) throw new Error(`wrote ${at} floats, not ${FLOAT_COUNT}`);
  return new Uint8Array(all.buffer);
}

/** A folded policy, ready to run — the same session shape as `makeForwardSession`. */
export function makeFoldedSession(bytes, gain, offset, name) {
  const params = foldParameters(loadParameters(bytes), gain, offset);
  return { name, run: observation => forward(params, observation) };
}
