// The Microduck, actually simulated.
//
// MuJoCo (WASM) steps the physics at 0.005 s; the real trained policy runs
// every 4th step, which is the robot's own 50 Hz. Nothing here is an
// animation: the joint angles you see are what alpha_walking.onnx asks for,
// given what the physics says the body is doing.
import loadMuJoCo from './vendor/mujoco.js';
import * as ort from './vendor/ort/ort.wasm.min.mjs';
import { makeLoop } from './duckloop.mjs';

ort.env.wasm.wasmPaths = './vendor/ort/';
ort.env.wasm.numThreads = 1;

const DECIMATION = 4;
// The kinematic chain, from the model. Body 0 is MuJoCo's world.
const BONES = [
  [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],       // left leg
  [1, 7], [7, 8], [8, 9], [9, 10],              // neck + head
  [1, 11], [11, 12], [12, 13], [13, 14], [14, 15], // right leg
];

const cv = document.getElementById('view');
const ctx = cv.getContext('2d');
const statusEl = document.getElementById('status');
const hud = document.getElementById('hud');

const cmdState = { vx: 0, vy: 0, vyaw: 0 };
let model, data, mj, session, inputName, C, HOME, buildObs, gaitTargets, projectedGravity, command;
let lastAction = new Array(14).fill(0), previous = null, ticks = 0;

function reset() {
  mj.mj_resetData(model, data);
  data.qpos[2] = 0.1231; data.qpos[3] = 1;
  for (let i = 0; i < 14; i++) { data.qpos[7 + i] = HOME[i]; data.ctrl[i] = HOME[i]; }
  mj.mj_forward(model, data);
  lastAction = new Array(14).fill(0); previous = null; ticks = 0;
}

async function tick() {
  const q = [data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]];
  const gyro = [data.qvel[3], data.qvel[4], data.qvel[5]];
  const jpos = [], jvel = [];
  for (let k = 0; k < 14; k++) { jpos.push(data.qpos[7 + k]); jvel.push(data.qvel[6 + k]); }
  const obs = buildObs(gyro, projectedGravity(q), jpos, jvel, lastAction, command(cmdState));
  const out = await session.run({ [inputName]: new ort.Tensor('float32', obs, [1, 61]) });
  lastAction = Array.from(out[session.outputNames[0]].data);
  previous = gaitTargets(lastAction, previous);
  for (let k = 0; k < 14; k++) data.ctrl[k] = previous[k];
  for (let s = 0; s < DECIMATION; s++) mj.mj_step(model, data);
  ticks++;
}

// ── drawing ────────────────────────────────────────────────────────────────
// A fixed three-quarter camera that follows the duck on the ground plane.
function project(p, cam) {
  const dx = p[0] - cam.x, dy = p[1] - cam.y, dz = p[2];
  const yaw = 0.9, pitch = 0.28;
  const rx = dx * Math.cos(yaw) - dy * Math.sin(yaw);
  const ry = dx * Math.sin(yaw) + dy * Math.cos(yaw);
  const depth = ry + 1.05;
  const s = cam.scale / Math.max(depth, 0.25);
  return [cam.w / 2 + rx * s, cam.h * 0.62 - (dz - pitch * ry) * s, depth];
}

function draw() {
  const w = cv.clientWidth, h = cv.clientHeight, dpr = Math.min(devicePixelRatio || 1, 2);
  if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  ctx.clearRect(0, 0, w, h);

  const cam = { x: data.qpos[0], y: data.qpos[1], w, h, scale: Math.min(w, h) * 0.9 };

  // floor grid, so travel is visible rather than asserted
  ctx.strokeStyle = css('--rule'); ctx.lineWidth = 1; ctx.globalAlpha = 0.55;
  const step = 0.1, span = 1.4;
  const gx = Math.round(cam.x / step) * step, gy = Math.round(cam.y / step) * step;
  for (let i = -span; i <= span + 1e-9; i += step) {
    ctx.beginPath();
    let a = project([gx + i, gy - span, 0], cam), b = project([gx + i, gy + span, 0], cam);
    ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    ctx.beginPath();
    a = project([gx - span, gy + i, 0], cam); b = project([gx + span, gy + i, 0], cam);
    ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  }
  ctx.globalAlpha = 1;

  const pts = [];
  for (let b = 0; b < 16; b++) pts.push(project([data.xpos[b * 3], data.xpos[b * 3 + 1], data.xpos[b * 3 + 2]], cam));

  // contact shadow
  ctx.fillStyle = css('--rule'); ctx.globalAlpha = 0.5;
  const sh = project([data.qpos[0], data.qpos[1], 0], cam);
  ctx.beginPath(); ctx.ellipse(sh[0], sh[1], 26, 9, 0, 0, Math.PI * 2); ctx.fill();
  ctx.globalAlpha = 1;

  ctx.strokeStyle = css('--duck'); ctx.lineWidth = 4; ctx.lineCap = 'round';
  for (const [a, b] of BONES) {
    ctx.beginPath(); ctx.moveTo(pts[a][0], pts[a][1]); ctx.lineTo(pts[b][0], pts[b][1]); ctx.stroke();
  }
  ctx.fillStyle = css('--ink');
  for (const [, b] of BONES) { ctx.beginPath(); ctx.arc(pts[b][0], pts[b][1], 2.6, 0, Math.PI * 2); ctx.fill(); }
  // the head, so it reads as a duck facing somewhere
  ctx.fillStyle = css('--duck');
  ctx.beginPath(); ctx.arc(pts[10][0], pts[10][1], 9, 0, Math.PI * 2); ctx.fill();

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
  cmdState.vx = fwd * 0.18;
  cmdState.vyaw = turn * 0.8;
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
    const xml = await (await fetch('./scene.xml')).text();
    mj.FS.writeFile('/scene.xml', xml);
    model = mj.MjModel.mj_loadXML('/scene.xml');
    data = new mj.MjData(model);

    statusEl.textContent = 'loading the policy…';
    session = await ort.InferenceSession.create('./alpha_walking.onnx');
    inputName = session.inputNames[0];

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
