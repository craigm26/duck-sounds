// THE GATE THAT LETS THE BENCH BE TAKEN APART.
//
// WHY THIS EXISTS. duckbench.mjs is about to be split in two — a pure core
// that knows physics and nothing about Node, and a shell that knows Node and
// nothing about physics — so that the same core can run in a browser on a
// phone. A refactor of a file whose whole output is floating-point trajectories
// cannot be reviewed by reading it: a moved line that changes the ORDER two
// numbers are added in changes the eleventh decimal of a walk, and by tick 300
// the duck is somewhere else. So the split is judged by replaying a fixed
// script of requests through the bench and comparing the answers byte for byte.
//
// IT REPLAYS THE SAME SCRIPT TWO WAYS. `--mode server` spawns the bench as a
// process and talks HTTP to it, which is the only way to interrogate the
// CURRENT duckbench.mjs, since importing it starts a server. `--mode core`
// imports duckbench-core.mjs and calls `handle` directly, which is what the
// browser will do. A split that is a pure move makes all three runs — old
// server, new server, new core — identical.
//
// WHAT IS ALLOWED TO DIFFER is passed as --allow, a comma-separated list of
// JSON pointer prefixes. Nothing is ignored by default: a field that changed
// has to be named on the command line, in a diff a reviewer reads.
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));

/** The policies this script drives, by the name /policy takes. Fixed rather
 *  than scanned, so the script is the same script on a machine whose scratch
 *  directory holds a different set of uploads. */
const POLICIES = [
  'alpha_ground_pick.onnx', 'alpha_sitstand.onnx', 'alpha_stand.onnx',
  'alpha_walking.onnx', 'ball_kick_left.onnx', 'ball_kick_right.onnx',
  'BEST_alpha_sitstand.onnx', 'BEST_alpha_stand.onnx', 'BEST_roller.onnx',
  'BEST_roller_crouch.onnx', 'headspin.onnx', 'roller.onnx',
  'roller_crouch.onnx', 'roulade.onnx', 'flamingo-cycle/policy.onnx',
];

/** A pose 14 wide: HOME with the head tipped, so /perform has something to do. */
const HOME14 = [0.0, -0.0873, -0.4579, -0.0049, 0.453, 0.3491, 0.3491, 0.0, 0.0,
                0.0, 0.0873, 0.4579, 0.0049, -0.453];
const NOD = HOME14.map((v, i) => (i === 5 || i === 6 ? v - 0.25 : v));
/** The track the existing /perform entry uses, by name, for the new ones. */
const T = [{ at: 0.4, pose: NOD }, { at: 0.8, pose: HOME14 }];
/** The stairs challenge's own four-step 60 mm flight, in the room's frame. */
const FLIGHT4 = Array.from({ length: 4 },
  (_, i) => ({ x: 0.12 + i * 0.28 + 0.17, top: (i + 1) * 0.060 }));
/** The nominal cell — dh 0, drop 0.120, fmul 1.0 — the one rig3 itself runs. */
const CELL0 = { dh: 0.000, drop: 0.120, fmul: 1.0 };
const CLIMB_INTENT = JSON.parse(fs.readFileSync(
  path.resolve(HERE, '../climb/best_r6_ceilvaultC_60mm.json'), 'utf8'));

function script() {
  const r = [];
  const add = (method, url, body) => r.push({ method, url, body: body ?? null });
  add('GET', '/health');
  add('POST', '/reset', {});
  add('GET', '/state');
  add('GET', '/now');
  add('POST', '/intent', { vx: 0.15 });
  add('POST', '/intent', { vx: 0.15, hold: 0.2 });
  add('POST', '/intent', { vyaw: 0.5, hold: 0.1 });
  add('POST', '/intent', { vx: -0.1, vy: 0.1, hold: 0.3 });
  add('POST', '/intent', { vy: 0.2 });
  add('POST', '/intent', { vx: 0.3, vyaw: -0.4, hold: 0.5 });
  add('POST', '/stop', {});
  add('POST', '/stop', { settle: 0.2 });
  add('GET', '/state');
  add('POST', '/ball', { x: 0.5, y: 0.2 });
  add('POST', '/ball', { bearing: 30, range: 0.7 });
  add('GET', '/state');
  add('POST', '/reset', {});
  for (const p of POLICIES) {
    add('POST', '/policy', { policy: p });
    add('POST', '/intent', { vx: 0.1, hold: 0.1 });
  }
  // The allow-list is the security boundary; a gate that never tests it is
  // testing the happy path of a door.
  add('POST', '/policy', { policy: '../secrets.onnx' });
  add('POST', '/policy', { policy: '__proto__' });
  add('POST', '/policy', { policy: 'https://example.com/p.onnx' });
  add('POST', '/reset', {});
  add('POST', '/record', { policy: 'alpha_walking.onnx', seconds: 1,
                           schedule: [[0, { vx: 0.2 }]] });
  add('POST', '/measure', { policy: 'BEST_alpha_stand.onnx', rollouts: 2, seconds: 1 });
  add('POST', '/perform', { track: [{ at: 0.4, pose: NOD }, { at: 0.8, pose: HOME14 }],
                            rollouts: 2, seconds: 1 });
  add('POST', '/capture', { track: [{ at: 0.3, pose: NOD }], rollouts: 1, seconds: 0.6 });
  add('POST', '/record', {});
  add('POST', '/perform', {});
  add('POST', '/intent', { vx: 'banana' });
  add('GET', '/nowhere');
  add('GET', '/health');
  // ── APPEND-ONLY, PAST THE LAST EXISTING ENTRY ─────────────────────────────
  //
  // Every index above is stable, which is what lets `--prefix` assert that the
  // first sixty entries of the v6 capture are leaf-identical to ALL of v5. A
  // recapture that inserted an entry anywhere else would renumber the fixture
  // and the prefix check would be comparing two different requests.
  //
  // The three /perform entries below are ordered so the middle one — the same
  // body as entry #53 — sits BETWEEN a world-bearing request and a bare-floor
  // one. If a per-request world ever leaked into the module global or into the
  // shared batch mjData, that entry is where it would show.
  add('POST', '/perform', { track: T, rollouts: 1, seconds: 1 });               // the criterion fix
  add('POST', '/world', { name: 'parity flight', steps: FLIGHT4 });
  add('POST', '/perform', { track: T, rollouts: 2, seconds: 1 });               // must equal #53
  add('POST', '/perform', { track: T, rollouts: 2, seconds: 1,
                            world: { steps: FLIGHT4 }, spawn: { x: 0.05, y: 1.305 } });
  add('POST', '/perform', { track: T, rollouts: 2, seconds: 1, world: { clear: true } });
  add('POST', '/perform', { track: T, rollouts: 1, seconds: 1, world: {} });    // refusal, pinned
  add('POST', '/perform', { track: T, rollouts: 1, seconds: 1,
                            spawn: { x: 9, y: 0 } });                           // refusal, pinned
  add('POST', '/climb', { intent: CLIMB_INTENT, rise: 0.06, cell: CELL0,
                          tail: 'policy', clip: true });
  add('POST', '/world', { name: 'parity floor', clear: true });
  return r;
}

/** Deterministic JSON: keys sorted, so a reordered object is not a diff. */
function canon(value) {
  if (Array.isArray(value)) return value.map(canon);
  if (value && typeof value === 'object') {
    const out = {};
    for (const k of Object.keys(value).sort()) out[k] = canon(value[k]);
    return out;
  }
  return value;
}

async function viaServer(entry, port) {
  const env = { ...process.env, DUCKBENCH_PORT: String(port) };
  const child = spawn(process.execPath, [entry], { cwd: HERE, env, stdio: ['ignore', 'pipe', 'pipe'] });
  let log = '';
  child.stdout.on('data', d => { log += d; });
  child.stderr.on('data', d => { log += d; });
  const base = `http://127.0.0.1:${port}`;
  const deadline = Date.now() + 60000;
  for (;;) {
    if (child.exitCode !== null) throw new Error(`bench died before it listened:\n${log}`);
    try { const r = await fetch(base + '/health'); if (r.ok) break; } catch { /* not up yet */ }
    if (Date.now() > deadline) { child.kill('SIGKILL'); throw new Error(`bench never listened:\n${log}`); }
    await new Promise(s => setTimeout(s, 200));
  }
  const out = [];
  try {
    for (const step of script()) {
      const init = { method: step.method };
      if (step.body !== null) {
        init.headers = { 'content-type': 'application/json' };
        init.body = JSON.stringify(step.body);
      }
      const res = await fetch(base + step.url, init);
      out.push({ ...step, status: res.status, answer: canon(await res.json()) });
    }
  } finally { child.kill('SIGKILL'); }
  return out;
}

async function viaCore(port) {
  const { nodeBench } = await import('./duckbench-node.mjs');
  const { handle } = await nodeBench();
  const out = [];
  for (const step of script()) {
    const url = new URL(step.url, 'http://bench.local');
    let status = 200, answer;
    try {
      const value = await handle(url, step.body ?? {});
      if (!value) { status = 404; answer = { error: `no ${url.pathname} here` }; }
      else answer = value;
    } catch (e) { status = 400; answer = { error: String(e.message || e) }; }
    out.push({ ...step, status, answer: canon(JSON.parse(JSON.stringify(answer))) });
  }
  return out;
}

/** Every leaf of a canonical answer, as pointer -> value. */
function leaves(value, at, into) {
  if (Array.isArray(value)) { value.forEach((v, i) => leaves(v, `${at}/${i}`, into)); return into; }
  if (value && typeof value === 'object') {
    for (const k of Object.keys(value)) leaves(value[k], `${at}/${k}`, into);
    return into;
  }
  into.set(at, value);
  return into;
}

function compare(a, b, allow) {
  const diffs = [];
  if (a.length !== b.length) return [`request count ${a.length} vs ${b.length}`];
  for (let i = 0; i < a.length; i++) {
    const tag = `#${i} ${a[i].method} ${a[i].url}`;
    if (a[i].status !== b[i].status) diffs.push(`${tag} status ${a[i].status} -> ${b[i].status}`);
    const L = leaves(a[i].answer, '', new Map()), R = leaves(b[i].answer, '', new Map());
    for (const key of new Set([...L.keys(), ...R.keys()])) {
      // Exact pointer or a child of it — never a prefix match on the STRING,
      // or `--allow /policy` would also excuse every leaf of `/policies`.
      if (allow.some(p => key === p || key.startsWith(p + '/'))) continue;
      const l = L.has(key) ? L.get(key) : '<<absent>>';
      const rr = R.has(key) ? R.get(key) : '<<absent>>';
      if (!Object.is(l, rr)) diffs.push(`${tag} ${key || '/'}: ${JSON.stringify(l)} -> ${JSON.stringify(rr)}`);
    }
  }
  return diffs;
}

const args = process.argv.slice(2);
const opt = (name, fallback) => {
  const i = args.indexOf(name);
  return i < 0 ? fallback : args[i + 1];
};
const mode = opt('--mode', 'server');
const outFile = opt('--out', null);
const against = opt('--against', null);
/**
 * THE OLD FIXTURE, AS A PREFIX OF THE NEW ONE.
 *
 * A recapture is the one act in this repo that can make a gate green by
 * agreeing with itself. `--against` says "the answers are what they were";
 * `--prefix` says "and the fixture I am now holding them to still contains,
 * entry for entry and leaf for leaf, everything the OLD one pinned". The new
 * entries are appended past the end, so the old file is exactly the first N
 * entries of the new one — and if it is not, the recapture moved something it
 * was not supposed to move and the reviewer is told which leaf.
 */
const prefix = opt('--prefix', null);
const entry = opt('--entry', 'duckbench.mjs');
const port = +opt('--port', 8791);
const allow = (opt('--allow', '') || '').split(',').filter(Boolean);

const started = Date.now();
const run = mode === 'core' ? await viaCore() : await viaServer(entry, port);
const secs = ((Date.now() - started) / 1000).toFixed(1);
console.log(`${run.length} requests, ${mode} mode${mode === 'server' ? ` (${entry})` : ''}, ${secs}s`);
if (outFile) {
  fs.writeFileSync(path.resolve(HERE, outFile), JSON.stringify(run, null, 1) + '\n');
  console.log(`wrote ${outFile}`);
}
if (prefix) {
  const older = JSON.parse(fs.readFileSync(path.resolve(HERE, prefix), 'utf8'));
  if (older.length > run.length) {
    console.log(`PREFIX FAILED against ${prefix}: it has ${older.length} entries and this run `
              + `made ${run.length}`);
    process.exitCode = 1;
  } else {
    const head = run.slice(0, older.length);
    // THE REQUESTS FIRST, THEN THE ANSWERS. `compare` diffs answers and uses
    // the method and url only to label a line, so an entry inserted in the
    // middle would produce a wall of leaf diffs under the wrong names instead
    // of the one sentence that explains them.
    const moved = [];
    for (let i = 0; i < older.length; i++) {
      if (older[i].method !== head[i].method || older[i].url !== head[i].url
       || JSON.stringify(older[i].body) !== JSON.stringify(head[i].body)) {
        moved.push(`#${i} ${older[i].method} ${older[i].url} -> ${head[i].method} ${head[i].url}`);
      }
    }
    const diffs = moved.length ? moved : compare(older, head, allow);
    if (!diffs.length) {
      console.log(`PREFIX OK: the first ${older.length} entries are ${prefix}, leaf for leaf`);
    } else {
      console.log(`PREFIX FAILED against ${prefix}: ${diffs.length} difference(s)`);
      for (const d of diffs.slice(0, 40)) console.log('  ' + d);
      if (diffs.length > 40) console.log(`  ... and ${diffs.length - 40} more`);
      process.exitCode = 1;
    }
  }
}
if (against) {
  const before = JSON.parse(fs.readFileSync(path.resolve(HERE, against), 'utf8'));
  const diffs = compare(before, run, allow);
  if (allow.length) console.log(`allowed to differ: ${allow.join(', ')}`);
  if (!diffs.length) { console.log(`PARITY OK against ${against}`); }
  else {
    console.log(`PARITY FAILED against ${against}: ${diffs.length} difference(s)`);
    for (const d of diffs.slice(0, 40)) console.log('  ' + d);
    if (diffs.length > 40) console.log(`  ... and ${diffs.length - 40} more`);
    process.exitCode = 1;
  }
}
