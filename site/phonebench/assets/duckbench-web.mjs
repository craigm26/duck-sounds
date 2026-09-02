// The duck bench, in a browser. Same core, a different machine under it.
//
// WHAT THIS ANSWERS. Duck Studio's whole shape rests on one sentence — "an
// iPhone has no physics engine" — and that sentence has been repeated in
// comments, docs and UI copy until it reads like a property of the hardware. It
// is a property of the BUILD. MuJoCo ships a WebAssembly target; the .wasm in
// site/vendor is byte-identical to the one the Pi bench runs (md5 c08b79f7…),
// and WebAssembly's float arithmetic is IEEE-754 with no fast-math, so it is
// the same integrator producing the same numbers. The policy is four matrix
// multiplies over 197,774 floats. Neither of those needs a desk.
//
// WHAT IT DOES NOT ANSWER, SAID HERE RATHER THAN DISCOVERED LATER:
//
//   • /upload is refused. It takes ONNX bytes, and this shell has no ONNX
//     parser — deliberately: `policyforward.mjs` reads duckkit's CANONICAL
//     PARAMETER BYTES, which is the layout DuckEvidence already fingerprints a
//     policy by, so the phone runs exactly the numbers the app attested to. A
//     second ONNX reader in JavaScript would be a second opinion about what a
//     policy is. The app can produce canonical bytes for anything it can load,
//     so this is a wiring job and not a wall — but it is not wired, and the
//     endpoint says so instead of failing obscurely.
//
//   • A TRAJECTORY FROM HERE IS NOT THE DESK'S TRAJECTORY. Measured on the Pi
//     over the physics-parity script: identical to 1e-4 for the first fifty
//     ticks, then 2 mm apart at tick 100, 9 mm at 150, 19 mm at 200, 32 mm at
//     250. The physics is the same; the INFERENCE differs by up to 3.5e-6 per
//     action between onnxruntime and policyforward (policy_parity.mjs), and a
//     closed loop is a chaotic amplifier. So a /record clip made here is not
//     comparable frame-for-frame with one made on the desk, while /measure —
//     which counts outcomes over randomised drops — is.
import { makeBench } from './duckbench-core.mjs';
import { makeForwardSession, FLOAT_COUNT } from './policyforward.mjs';

const hex = buffer => [...new Uint8Array(buffer)].map(b => b.toString(16).padStart(2, '0')).join('');

/**
 * The JavaScript engine, as far as it can be known — and NOT asserted.
 *
 * This field was written as the literal 'JavaScriptCore/WebKit', which is true
 * of the machine this shell exists for and false of every other one that can
 * run it: the first time it was exercised under Node on the Pi, /health told a
 * reader the arithmetic had been done by WebKit. An engine name is a claim
 * about a measurement's provenance, and a saved measurement carries it
 * forever, so it is derived from what the runtime will actually say and falls
 * back to admitting ignorance.
 */
function jsEngine() {
  const ua = globalThis.navigator?.userAgent ?? '';
  // NOT `\bChrome/`: headless Chrome's UA says `HeadlessChrome/151…`, where the
  // s-to-C join is not a word boundary, so the Chrome test missed and the
  // Safari test — which matches every Chrome UA, because they all still carry
  // `AppleWebKit/537.36 … Safari/537.36` — claimed JavaScriptCore. Found by
  // rendering the probe page in Chromium and reading what /health said.
  if (/Chrome\/|Chromium\/|Edg\/|CriOS\//.test(ua)) return 'V8/Blink';
  if (/\bFirefox\//.test(ua)) return 'SpiderMonkey/Gecko';
  if (/\bSafari\/|\bAppleWebKit\//.test(ua)) return 'JavaScriptCore/WebKit';
  if (globalThis.process?.versions?.v8) return `V8 ${globalThis.process.versions.v8} (not a browser)`;
  return 'unidentified JavaScript engine';
}

/**
 * A bench over the page's own origin.
 *
 * `mujoco` IS PASSED IN, NOT IMPORTED. mujoco.js is an Emscripten module that
 * wants to fetch its .wasm relative to itself and to be instantiated once; the
 * page owns that, and hands the instance here. It is also the one thing that
 * cannot be written the same way in Node and in a browser, which is why the
 * core takes it as `env.mujoco` rather than importing it.
 */
export async function makeWebBench({ mujoco, assetBase = './assets/', sceneName = 'scene.mjb' } = {}) {
  const get = async name => {
    // The page fetches relative to its own origin and nothing here builds a
    // URL out of caller input: policy names are keys of a manifest fetched at
    // boot, and everything else is one of two fixed filenames.
    const response = await fetch(assetBase + name);
    if (!response.ok) throw new Error(`asset ${name}: ${response.status} ${response.statusText}`);
    return new Uint8Array(await response.arrayBuffer());
  };

  // NAME AND FILE ARE DIFFERENT THINGS, and this manifest is the only place
  // that knows both — the same job `scan()` does for the Node shell. A
  // community policy is `flamingo-cycle/policy.onnx` to /policy and
  // `flamingo-cycle-policy.bin` on disk, because a slash in a filename is a
  // directory and not a name.
  const manifest = JSON.parse(new TextDecoder().decode(await get('policies/manifest.json')));
  const files = new Map(manifest.policies.map(p => [p.name, p.file]));
  const scratch = new Map();

  return makeBench({
    sceneName,
    mujoco,
    readAsset: name => (files.has(name) ? get('policies/' + files.get(name)) : get(name)),
    listPolicies: () => [...files.keys(), ...scratch.keys()],
    sha256: async bytes => hex(await crypto.subtle.digest('SHA-256',
      bytes instanceof Uint8Array ? bytes.slice().buffer : bytes)),
    // What the browser will admit to. Safari reports 2 on some devices and the
    // real count on others; it is reported rather than relied on — nothing here
    // runs a worker.
    cores: navigator.hardwareConcurrency || 1,
    makeSession(bytes, name) {
      if (bytes.byteLength !== FLOAT_COUNT * 4) {
        throw new Error('this bench runs canonical parameter bytes, not ONNX: '
                      + `${name} is ${bytes.byteLength} bytes where a policy is ${FLOAT_COUNT * 4}. `
                      + 'Uploading a network to a phone bench is not wired yet.');
      }
      return makeForwardSession(bytes, name);
    },
    scratch: {
      has: name => scratch.has(name),
      get: name => scratch.get(name),
      set: (name, bytes) => scratch.set(name, bytes),
      delete: name => scratch.delete(name),
    },
    host: {
      kind: 'phone',
      device: navigator.userAgent,
      engine: `${jsEngine()}, MuJoCo 3.1.16 (WASM), policyforward.mjs`,
    },
  });
}

/**
 * Put the bench behind one function on the page.
 *
 * `duckbench(pathAndQuery, bodyJSON) -> Promise<JSON string>`, because the app
 * on the other side of the WebView bridge can pass strings and nothing else.
 * The answers are the SAME OBJECTS the HTTP bench sends, including the error
 * shape: a request that would have been a 404 comes back as
 * `{"error":"no /x here"}` and one that would have been a 400 as
 * `{"error":"…"}`, so a client can be written once against both.
 */
export function install(bench, target = globalThis) {
  target.duckbench = async (pathAndQuery, bodyJSON) => {
    try {
      const url = new URL(pathAndQuery, 'http://duck.bench');
      let body = {};
      if (bodyJSON) {
        try { body = JSON.parse(bodyJSON); }
        catch { return JSON.stringify({ error: 'body is not JSON' }); }
      }
      const answer = await bench.handle(url, body);
      if (!answer) return JSON.stringify({ error: `no ${url.pathname} here` });
      return JSON.stringify(answer);
    } catch (error) {
      return JSON.stringify({ error: String(error?.message || error) });
    }
  };
  return target.duckbench;
}
