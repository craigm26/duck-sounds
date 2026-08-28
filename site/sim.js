// The Microduck, actually simulated.
//
// MuJoCo (WASM) steps the physics at 0.005 s; the real trained policy runs
// every 4th step, which is the robot's own 50 Hz. Nothing here is an
// animation: the joint angles you see are what alpha_walking.onnx asks for,
// given what the physics says the body is doing.
import loadMuJoCo from './vendor/mujoco.js';
import * as ort from './vendor/ort/ort.wasm.min.mjs';
import { makeLoop } from './duckloop.mjs';
import { createRenderer } from './render.js';

// Absolute, not relative: onnxruntime resolves wasmPaths against its OWN module
// URL, so './vendor/ort/' became /vendor/ort/vendor/ort/... and every backend
// failed to load. Caught by the headless browser check, not by eye.
ort.env.wasm.wasmPaths = new URL('./vendor/ort/', document.baseURI).href;
ort.env.wasm.numThreads = 1;

const DECIMATION = 4;
const cv = document.getElementById('view');
let renderer = null;
const statusEl = document.getElementById('status');
const hud = document.getElementById('hud');

const cmdState = { vx: 0, vy: 0, vyaw: 0 };
let model, data, mj, session, inputName, C, HOME, buildObs, gaitTargets, projectedGravity, command;
let lastAction = new Array(14).fill(0), previous = null, ticks = 0, GYRO = 0;

function reset() {
  mj.mj_resetData(model, data);
  data.qpos[2] = 0.12; data.qpos[3] = 1;
  for (let i = 0; i < 14; i++) { data.qpos[7 + i] = HOME[i]; data.ctrl[i] = HOME[i]; }
  mj.mj_forward(model, data);
  lastAction = new Array(14).fill(0); previous = null; ticks = 0;
}

async function tick() {
  const q = [data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]];
  const gyro = [data.sensordata[GYRO], data.sensordata[GYRO + 1], data.sensordata[GYRO + 2]];
  const jpos = [], jvel = [];
  for (let k = 0; k < 14; k++) { jpos.push(data.qpos[7 + k]); jvel.push(data.qvel[6 + k]); }
  const obs = buildObs(gyro, projectedGravity(q), jpos, jvel, lastAction, command(cmdState));
  const out = await session.run({ [inputName]: new ort.Tensor('float32', obs, [1, 61]) });
  lastAction = Array.from(out[session.outputNames[0]].data);
  // Pollen's simulator drives ctrl = pose + action, scale 1.0, no low-pass.
  // DuckKit's 0.9 + filter is robotd's on-robot behaviour, which is a
  // different thing; matching mjlab here is what makes the gait match.
  for (let k = 0; k < 14; k++) data.ctrl[k] = HOME[k] + lastAction[k];
  for (let s = 0; s < DECIMATION; s++) mj.mj_step(model, data);
  ticks++;
}

// ── drawing ───────────────────────────────────────────────────────────────
function themeColour(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const m = /^#([0-9a-f]{6})$/i.exec(v);
  if (!m) return fallback;
  const n = parseInt(m[1], 16);
  return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255];
}

function draw() {
  if (!renderer) return;
  renderer.render(data, {
    bg: themeColour('--panel', [0.91, 0.92, 0.90]),
    grid: themeColour('--rule', [0.79, 0.82, 0.78]),
  });
  const speed = Math.hypot(data.qvel[0], data.qvel[1]);
  hud.textContent =
    `tick ${String(ticks).padStart(5, '0')}   ` +
    `speed ${speed.toFixed(2)} m/s   ` +
    `height ${data.qpos[2].toFixed(3)} m   ` +
    `contacts ${data.ncon}`;
}

// ── controls ───────────────────────────────────────────────────────────────
const keys = new Set();
addEventListener('keydown', e => {
  if (['ArrowUp','ArrowDown','ArrowLeft','ArrowRight',' '].includes(e.key)) e.preventDefault();
  if (e.key === 'r' || e.key === 'R') reset();
  keys.add(e.key);
});
addEventListener('keyup', e => keys.delete(e.key));

function readControls() {
  const fwd = (keys.has('ArrowUp') || keys.has('w') ? 1 : 0) - (keys.has('ArrowDown') || keys.has('s') ? 1 : 0);
  const turn = (keys.has('ArrowLeft') || keys.has('a') ? 1 : 0) - (keys.has('ArrowRight') || keys.has('d') ? 1 : 0);
  cmdState.vx = fwd * 0.45;
  cmdState.vyaw = turn * 1.0;
}
for (const el of document.querySelectorAll('[data-key]')) {
  const k = el.dataset.key;
  const on = e => { e.preventDefault(); keys.add(k); el.classList.add('on'); };
  const off = () => { keys.delete(k); el.classList.remove('on'); };
  el.addEventListener('pointerdown', on); el.addEventListener('pointerup', off);
  el.addEventListener('pointerleave', off); el.addEventListener('pointercancel', off);
}
document.getElementById('reset').addEventListener('click', reset);

// ── boot ───────────────────────────────────────────────────────────────────
(async function start() {
  try {
    statusEl.textContent = 'loading physics…';
    C = await (await fetch('./duckkit-constants.json')).json();
    ({ HOME, buildObs, gaitTargets, projectedGravity, command } = makeLoop(C));
    mj = await loadMuJoCo();
    statusEl.textContent = 'loading the robot…';
    // A PRECOMPILED model, not XML. Compiling Pollen's meshed MJCF in the
    // browser fails with "thread constructor failed": MuJoCo parallelises the
    // convex-hull computation for mesh collision geoms, and the WASM build
    // cannot spawn those workers here. Compiling once in Node and shipping the
    // .mjb skips hull generation entirely — it also means no STLs to fetch.
    const mjb = await (await fetch('./scene.mjb')).arrayBuffer();
    mj.FS.writeFile('/scene.mjb', new Uint8Array(mjb));
    model = mj.MjModel.mj_loadBinary('/scene.mjb', new mj.MjVFS());
    data = new mj.MjData(model);
    // NOT sensordata[0]: their sensor block opens with a 4-value framequat, so
    // the angular-velocity sensor the runtime reads sits further along. Reading
    // the first three floats was feeding the policy part of a quaternion.
    for (let i = 0; i < model.nsensor; i++) {
      if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;
    }

    statusEl.textContent = 'loading the policy…';
    session = await ort.InferenceSession.create('./alpha_walking.onnx');
    inputName = session.inputNames[0];

    statusEl.textContent = 'loading the robot\u2019s geometry\u2026';
    renderer = await createRenderer(cv, './duck-visual.bin');
    console.log('renderer:', renderer.draws, 'parts,', renderer.triangles, 'triangles');

    reset();
    statusEl.textContent = '';
    document.body.classList.add('ready');

    let acc = 0, last = performance.now(), busy = false;
    const stepMs = 1000 / C.tickHz;
    async function loop(now) {
      acc += Math.min(now - last, 100); last = now;
      readControls();
      if (!busy) {
        busy = true;
        let n = 0;
        while (acc >= stepMs && n < 4) { acc -= stepMs; await tick(); n++; }
        busy = false;
      }
      draw();
      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
  } catch (err) {
    statusEl.textContent = 'The simulator could not start: ' + (err && err.message ? err.message : err);
    console.error(err);
  }
})();
