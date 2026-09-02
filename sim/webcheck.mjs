// The browser shell, exercised as far as this machine can take it.
//
// WHY IT IS WORTH RUNNING ON A PI. `duckbench-web.mjs` cannot be reviewed by
// reading it: it is glue, and glue fails at the joints — a manifest key that
// does not match a filename, `crypto.subtle` handed a view instead of a buffer,
// an asset path that resolves in a page and not in a directory. Every one of
// those is invisible until something fetches. So this serves site/phonebench
// over real HTTP, imports the SHIPPED copies of the modules out of that
// directory, and drives the whole physics-parity script through
// `globalThis.duckbench` — the same entry point the app will call.
//
// WHAT IT DOES NOT PROVE, and the reason sim/phonebench.html still has to be
// opened on the phone: this is V8 on a Pi, not JavaScriptCore on an A-series,
// and the questions the probe page exists to answer — SharedArrayBuffer,
// cross-origin isolation, whether Safari's WASM is fast enough — are exactly
// the ones this cannot ask. It proves the wiring. The phone proves the phone.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..', 'site', 'phonebench');
if (!fs.existsSync(ROOT)) { console.error('run scripts/make_phonebench.sh first'); process.exit(2); }

const TYPES = { '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json',
                '.wasm': 'application/wasm', '.html': 'text/html' };
const server = http.createServer((req, res) => {
  const file = path.join(ROOT, decodeURIComponent(new URL(req.url, 'http://x').pathname));
  if (!file.startsWith(ROOT) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
    res.writeHead(404); return res.end('no');
  }
  res.writeHead(200, {
    'content-type': TYPES[path.extname(file)] || 'application/octet-stream',
    // The same three headers site/_headers sets, so this is serving what
    // Cloudflare Pages will serve.
    'cross-origin-opener-policy': 'same-origin',
    'cross-origin-embedder-policy': 'require-corp',
    'cross-origin-resource-policy': 'same-origin',
  });
  fs.createReadStream(file).pipe(res);
});
await new Promise(done => server.listen(0, '127.0.0.1', done));
const base = `http://127.0.0.1:${server.address().port}/`;
console.log(`serving ${path.relative(process.cwd(), ROOT)} at ${base}`);

// THE SHIPPED COPIES, not the sources they were made from. A bundle script that
// copies the wrong file is the failure mode this catches.
const load = (await import(pathToFileURL(path.join(ROOT, 'assets/mujoco.js')))).default;
const { makeWebBench, install } = await import(pathToFileURL(path.join(ROOT, 'assets/duckbench-web.mjs')));

const started = Date.now();
const bench = await makeWebBench({ mujoco: await load(), assetBase: base + 'assets/' });
install(bench);
console.log(`bench built in ${((Date.now() - started) / 1000).toFixed(1)} s`);

const health = JSON.parse(await duckbench('/health'));
console.log(`bench       ${health.bench}`);
console.log(`host.kind   ${health.host.kind}`);
console.log(`host.engine ${health.host.engine}`);
console.log(`tickMillis  ${health.host.tickMillis}`);
console.log(`stand       ${health.stand}`);
console.log(`policies    ${health.policies.join(', ')}`);
console.log(`plant       ${health.plantName} ${health.plantDigest.slice(0, 16)}`);
console.log(`transport   ${health.transport.protocol}`);
console.log(`trainsWhy   ${health.trainsWhy}`);

// The one endpoint that is deliberately refused here, checked rather than
// assumed: a shell that quietly accepted ONNX bytes and ran garbage would be
// worse than one that says no.
const refused = JSON.parse(await duckbench('/upload', JSON.stringify({ onnx: 'AAAA' })));
console.log(`upload      ${refused.error}`);

const { TWIST, TICKS, HOLD, POLICY } = await import('./physics_parity.mjs');
const t0 = Date.now();
await duckbench('/reset', '{}');
await duckbench('/policy', JSON.stringify({ policy: POLICY }));
let state;
for (let i = 0; i < TICKS / (HOLD * health.tickHz); i++) {
  state = JSON.parse(await duckbench('/intent', JSON.stringify({ ...TWIST, hold: HOLD })));
}
const wall = (Date.now() - t0) / 1000;
const six = n => n.toFixed(6);
const EXPECTED = process.argv.includes('--expect')
  ? process.argv[process.argv.indexOf('--expect') + 1].split(/[ ,]+/).map(Number)
  : [0.360500, 0.725400, 0.113500];
console.log(`\nEXPECTED    ${EXPECTED.map(six).join(' ')}`);
console.log(`web shell   ${state.position.map(six).join(' ')}`);
const worst = Math.max(...state.position.map((v, i) => Math.abs(v - EXPECTED[i])));
console.log(`worst |Δ|   ${worst.toExponential(2)} m   (bench rounds to 1e-4)`);
console.log(`wall clock  ${wall.toFixed(2)} s for ${TICKS} ticks `
          + `= ${((TICKS / health.tickHz) / wall).toFixed(1)}× real time`);
// ── THE STAIRS CELL, THROUGH THE BROWSER SHELL ───────────────────────────
//
// WHY IT IS HERE AND WHAT IT IS FOR. /climb pulled four new files into the
// bundle — climb_score.mjs and the three it imports under the names the core
// imports them by — and a missing one of those is invisible until something
// asks: the page boots, /health answers, and the Stairs Challenge dies on its
// first cell. So a cell is actually scored, out of the SHIPPED copies, over
// real HTTP.
//
// IT IS NOT A PARITY CHECK AND MUST NOT BE READ AS ONE. This shell runs
// policyforward.mjs where the desk runs onnxruntime, they agree to 3.5e-6 per
// action (policy_parity.mjs), and 350-odd closed-loop ticks is a chaotic
// amplifier — the trajectory above already measures 32 mm apart by tick 250. A
// cell scored here is THIS MACHINE'S measurement of this move, which is exactly
// what a phone's answer is, and it carries its own plant digest to say so. What
// is checked is that it answers, that it answers about the right move, and that
// the grid it publishes is the fourteen cells.
const grid = JSON.parse(await duckbench('/climb/grid'));
console.log(`\nclimb grid  ${grid.cells.length} cells, bar ${grid.bar} of 9 stable, `
          + `upright-tail minimum ${grid.uprightTailMin}, climbable ${grid.climbable}`);
const intent = JSON.parse(fs.readFileSync(path.join(HERE, '..', 'climb', 'best_r3_vault_60mm.json'), 'utf8'));
const cellT0 = Date.now();
const cell = JSON.parse(await duckbench('/climb', JSON.stringify({
  intent, rise: 0.060, cell: grid.cells[3], tail: 'policy' })));   // cells[3] = 60 mm on the nominal plant
const cellWall = (Date.now() - cellT0) / 1000;
if (cell.error) { console.log(`CLIMB FAILED — ${cell.error}`); process.exitCode = 1; }
else {
  console.log(`climb cell  move ${cell.move} at ${grid.cells[3].dh * 1000 + 60} mm, drop ${grid.cells[3].drop}, `
            + `friction x${grid.cells[3].fmul}`);
  console.log(`            honest ${cell.honest}  stable ${cell.stable}  upright tail ${cell.uprightTailTicks}/50  `
            + `above ${cell.above_mm.toFixed(1)} mm  peak ${cell.peakAboveTread_mm.toFixed(1)} mm  `
            + `feet on tread ${cell.feetOnTread}`);
  console.log(`            ${cellWall.toFixed(2)} s of wall clock for one cell — fourteen of them is a grid`);
  console.log('            the desk answers this same cell honest=true stable=true upright tail 50/50 '
            + 'above=116.1 mm peak=122.5 mm feet on tread 2 (climb/r6_judge-results.json phaseG,\n'
            + '            and sim/climb_parity.mjs re-measures it); a difference here is the forward\n'
            + '            pass, not the physics.');
  if (cell.move !== '4b9110c448ec') { console.log('CLIMB FAILED — wrong move hash'); process.exitCode = 1; }
}

server.close();
if (worst > 1e-4 + 1e-9) { console.log('WEB PARITY FAILED'); process.exitCode = 1; }
else console.log('WEB PARITY OK — the browser shell reproduces the desk trajectory');
