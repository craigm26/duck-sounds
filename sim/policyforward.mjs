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
