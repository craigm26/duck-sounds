// The duck bench: this machine's physics, offered to Duck Studio over the LAN.
//
// WHY THIS EXISTS. Duck Studio can import a policy — from Pollen, or from
// Hugging Face — and then do nothing with it. It has no physics: an iPhone has
// no MuJoCo, so every clip it shows was recorded HERE and baked into duckkit
// at build time. Import flamingo-cycle on the phone and there is nothing to
// press. This serves the two things the phone cannot do for itself: RECORD a
// policy into a clip, and MEASURE how often it works.
//
// AND IT IS A TRANSPORT. rokbenko/quackd drives a Microduck through a
// `DuckTransport` — get_state, get_frame, send_intent, stop, and now() — and
// composes verbs on top of it. now() is the load-bearing one: the TRANSPORT
// OWNS TIME, so the same steering loop runs at wall-clock speed against a real
// duck and at sim speed against this one without a verb knowing which it has.
// So /state, /intent, /stop and /now sit alongside /record and /measure, over
// one persistent world that stays standing between requests. /policy is the
// odd one out: quackd's learned-verbs note calls a live policy swap the one
// upstream API it needs and does not have. On hardware that is somebody else's
// firmware; in sim it is a map lookup, so here it exists.
//
// NO /frame HERE. quackd's transport also hands over a camera image, and this
// process has no camera and no renderer: rendering lives in duckvision.py,
// which imports the pip `mujoco`, while this runs the npm one — MUJOCO 3.1.16
// (WASM), the version in sim/node_modules/mujoco/package.json — so that clips
// stay canon. This comment said 3.5.1 until 2026-08-30 and a design was
// written on that number as executed ground truth, which is how a stale
// comment becomes a stated fact. duckvision's own version is not given here
// because nothing this file can read says what it is; ask that process.
// Wiring the two together is a job for whatever composes them, not a stub
// that returns a grey rectangle.
//
// MORE THAN ONE DUCK, AND WHY THAT IS ONLY POSSIBLE HERE. A hardware Microduck
// cannot perceive another Microduck. The observation robotd builds is 61 values
// — 48 of proprioception, 3 of commanded twist, 4 of head pose, 6 of body pose
// (Pollen's `duck-ipc-proto` and the training env, read 2026-09-01) — and there
// is no slot in it for a second robot. Two real ducks in a room are two blind
// agents that happen to share a floor, and whatever coordination they show has
// to be carried over a network, which is where it comes apart: `intents.rs` is
// last-writer-wins on one slot, so two writers at 50 Hz interleave into that
// slot and produce a robot that obeys neither, and the deadman is age-based, so
// a partition does not degrade a duck, it stops one.
//
// Ducks in ONE MuJoCo model have none of that, and they DO perceive each other
// — not through a sensor slot but through the physics itself: contact forces
// when they touch, the floor they both push against, the dynamics of an object
// one of them lifts while the other holds it. One integrator, one clock, no
// link jitter, no last-writer-wins, because there is no link and no writer
// race. So the simulator is not a lesser swarm than a room full of hardware; it
// is the only place a swarm can exist right now, and the only place a genuinely
// multi-duck policy could ever be trained, since training needs exactly the
// shared-state rollout the 61-wide hardware observation cannot supply.
//
// Concretely: a scene may hold N ducks, each with a name taken from its MJCF
// prefix (`build_multiduck.py` writes them; `compile_multiduck.mjs` compiles
// them). /intent, /policy, /state, /stop, /ball and /reset take an optional
// `duck` name and mean the first duck without one, which is what a single-duck
// caller has always been asking for. Every duck has its own policy slot, so two
// of them can run different networks. And ONE STEP OF PHYSICS ADVANCES ALL OF
// THEM: a request addressed to one duck still moves the others under the
// commands they are holding, because there is one clock and that is the entire
// point of putting them in one world.
//
// WHAT IT IS NOT. It does not train. The Hailo on this machine is an inference
// ASIC with no training path at all, and mjlab — what Pollen train with — wants
// a GPU. Training on this box would mean a CPU PPO against plain MuJoCo; at the
// measured 1.66 ms a tick that is roughly 46 core-hours for 100M steps, an
// overnight job rather than an impossibility, but it is not this file.
//
// SAFETY. Policies are chosen from a scan of this directory by NAME. A request
// cannot name a path: no traversal, no URL, nothing fetched — and /policy, the
// one endpoint that lets a caller pick, goes through the same lookup as
// everything else. Set DUCKBENCH_TOKEN to require `Authorization: Bearer
// <token>`; without it the bench is open to whoever is on your network, which
// is fine for a bench on a desk and not fine on a café's wifi.
// WHERE THE PHYSICS WENT. This file used to be all of it — the plant, the
// control loop, the endpoints and the door. It is now the DOOR only:
// `duckbench-core.mjs` holds the physics and knows nothing about Node, and
// `duckbench-node.mjs` describes this machine to it. That split exists so the
// same core can run in a browser on an iPhone, where there is no fs, no os, no
// onnxruntime-node and no socket to listen on — and where, it turns out, there
// is a MuJoCo, because MuJoCo compiles to WebAssembly.
//
// NOTHING ABOUT THE ANSWERS CHANGED. `bench_parity.mjs` replays sixty requests
// and compares 5,751 leaves of the responses; the split moved exactly zero of
// the numbers. The only fields that differ are the ones deliberately added:
// /health's `host` block and the version bump to duck-bench/5.
import http from 'node:http';
import { nodeBench } from './duckbench-node.mjs';

const PORT = +(process.env.DUCKBENCH_PORT || 8770);
const TOKEN = process.env.DUCKBENCH_TOKEN || null;

const bench = await nodeBench();
const { handle } = bench;
// What to print on the way up, asked of the bench rather than recomputed here:
// a shell that keeps its own copy of the plant's numbers is a shell that can
// print a world the core did not run.
const health = await handle(new URL('http://bench.local/health'), {});

http.createServer((req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  const send = (code, value) => {
    const text = JSON.stringify(value);
    res.writeHead(code, { 'content-type': 'application/json',
                          'access-control-allow-origin': '*',
                          'access-control-allow-headers': 'authorization,content-type' });
    res.end(text);
  };
  if (req.method === 'OPTIONS') return send(204, {});
  if (TOKEN && req.headers.authorization !== `Bearer ${TOKEN}`) {
    return send(401, { error: 'this bench wants its token' });
  }
  // ONE MEGABYTE EVERYWHERE EXCEPT THE ONE ENDPOINT THAT CARRIES A FILE.
  // Every other body here is a handful of numbers, and a cap is what stops a
  // stray client filling this process's memory. /upload carries a policy: the
  // shipped ones are about 790 KB, which is ~1.05 MB once base64'd, so the old
  // cap silently destroyed the request — the client saw a 100 and no answer,
  // which took longer to work out than it should have.
  const cap = url.pathname === '/upload' ? 12e6 : 1e6;
  let raw = '';
  req.on('data', chunk => { raw += chunk; if (raw.length > cap) req.destroy(); });
  req.on('end', async () => {
    let body = {};
    if (raw) { try { body = JSON.parse(raw); } catch { return send(400, { error: 'body is not JSON' }); } }
    try {
      const answer = await handle(url, body);
      if (!answer) return send(404, { error: `no ${url.pathname} here` });
      send(200, answer);
    } catch (error) {
      send(400, { error: String(error.message || error) });
    }
  });
}).listen(PORT, '0.0.0.0', () => {
  console.log(`duck bench on http://0.0.0.0:${PORT} — ${TOKEN ? 'token required' : 'OPEN on this network'}`);
  console.log(`plant: ${bench.plant.name} ${bench.plant.digest.slice(0, 12)}, `
            + `${bench.timestep} s timestep, ${bench.substeps} substeps per ${bench.tickHz} Hz tick`);
  console.log(`host: ${health.host.device} — ${health.host.tickMillis} ms a control tick `
            + `(real time is ${(1000 / bench.tickHz).toFixed(0)} ms)`);
  // THE FILE THAT DOES THE STANDING, SAID ON THE WAY UP. It is resolved by
  // role now, and every gated endpoint depends on it having been found, so a
  // bench that resolved a different file than the operator expects should say
  // so before the first request rather than after it.
  console.log(`stands with: ${bench.stand}`);
  console.log(`ducks: ${bench.ducks.map(d => d.name).join(', ')} — one world, one clock`
            + `${bench.ducks.length > 1 ? ' (pass `duck` to /intent /policy /state /stop /ball /reset)' : ''}`);
  console.log('records/measures: /record /measure /perform — steers: /state /intent /stop /now /policy');
  console.log(`policies: ${health.policies.join(', ')}`);
});
