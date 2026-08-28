// The Microduck, actually simulated.
//
// MuJoCo (WASM) steps the physics at 0.005 s; the real trained policy runs
// every 4th step, which is the robot's own 50 Hz. Nothing here is an
// animation: the joint angles you see are what alpha_walking.onnx asks for,
// given what the physics says the body is doing.
import loadMuJoCo from './vendor/mujoco.js';
import * as ort from './vendor/ort/ort.wasm.min.mjs';
import { makeLoop } from './duckloop.mjs';

// Absolute, not relative: onnxruntime resolves wasmPaths against its OWN module
// URL, so './vendor/ort/' became /vendor/ort/vendor/ort/... and every backend
// failed to load. Caught by the headless browser check, not by eye.
ort.env.wasm.wasmPaths = new URL('./vendor/ort/', document.baseURI).href;
ort.env.wasm.numThreads = 1;

const DECIMATION = 4;
// The duck is drawn as solid limbs, not a stick figure. Each link is a real box
// spanning from its own joint to its child's, so the proportions are the
// model's own measured link lengths (thigh 42 mm, shin 49 mm, neck 50 mm)
// rather than anything guessed. Pollen's simulator uses their sculpted meshes;
// those live in a Space with no licence, so this is the honest alternative —
// the right shape, in the robot's colours: white shells, dark brackets, orange
// feet.
//
// MJCF body ids: 1 trunk, 2-6 left leg, 7-10 neck+head, 11-15 right leg.
const LIMB = {  // child id -> how the bone INTO it is drawn
  2:  { r: 0.017, c: 'joint' },  11: { r: 0.017, c: 'joint' },
  3:  { r: 0.015, c: 'joint' },  12: { r: 0.015, c: 'joint' },
  4:  { r: 0.015, c: 'shell' },  13: { r: 0.015, c: 'shell' },
  5:  { r: 0.014, c: 'shell' },  14: { r: 0.014, c: 'shell' },
  6:  { r: 0.012, c: 'joint' },  15: { r: 0.012, c: 'joint' },
  7:  { r: 0.013, c: 'joint' },
  8:  { r: 0.012, c: 'joint' },
  9:  { r: 0.013, c: 'joint' },
  10: { r: 0.014, c: 'joint' },
};
// Leaf volumes the bones cannot express: the head shell and the two feet.
const BLOCKS = [
  { body: 10, w: 0.060, d: 0.068, h: 0.042, up: 0.012, c: 'shell' },
  { body: 6,  w: 0.034, d: 0.052, h: 0.011, up: -0.014, c: 'foot' },
  { body: 15, w: 0.034, d: 0.052, h: 0.011, up: -0.014, c: 'foot' },
  { body: 1,  w: 0.050, d: 0.058, h: 0.040, up: 0.000, c: 'shell' },
];
const FACES = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[0,3,7,4]];
const CORNERS = [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]];

function rotate(q, v) {
  const [w, x, y, z] = q;
  const tx = 2 * (y * v[2] - z * v[1]);
  const ty = 2 * (z * v[0] - x * v[2]);
  const tz = 2 * (x * v[1] - y * v[0]);
  return [v[0] + w * tx + (y * tz - z * ty),
          v[1] + w * ty + (z * tx - x * tz),
          v[2] + w * tz + (x * ty - y * tx)];
}

/** Eight corners of a box spanning a→b with square cross-section 2r. */
function boxBetween(a, b, r) {
  const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const len = Math.hypot(d[0], d[1], d[2]) || 1e-6;
  const u = [d[0] / len, d[1] / len, d[2] / len];
  // any vector not parallel to u
  const seed = Math.abs(u[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  let n1 = [u[1] * seed[2] - u[2] * seed[1], u[2] * seed[0] - u[0] * seed[2], u[0] * seed[1] - u[1] * seed[0]];
  const n1l = Math.hypot(n1[0], n1[1], n1[2]) || 1e-6;
  n1 = n1.map(v => v / n1l);
  const n2 = [u[1] * n1[2] - u[2] * n1[1], u[2] * n1[0] - u[0] * n1[2], u[0] * n1[1] - u[1] * n1[0]];
  const out = [];
  for (const c of CORNERS) {
    const t = c[2] > 0 ? 1 : 0;
    out.push([
      a[0] + d[0] * t + n1[0] * c[0] * r + n2[0] * c[1] * r,
      a[1] + d[1] * t + n1[1] * c[0] * r + n2[1] * c[1] * r,
      a[2] + d[2] * t + n1[2] * c[0] * r + n2[2] * c[1] * r,
    ]);
  }
  return out;
}

const cv = document.getElementById('view');
const ctx = cv.getContext('2d');
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

// ── drawing ────────────────────────────────────────────────────────────────
function project(p, cam) {
  const dx = p[0] - cam.x, dy = p[1] - cam.y, dz = p[2];
  const yaw = 0.9, pitch = 0.30;
  const rx = dx * Math.cos(yaw) - dy * Math.sin(yaw);
  const ry = dx * Math.sin(yaw) + dy * Math.cos(yaw);
  const depth = Math.max(ry + 0.75, 0.15);
  const s = cam.scale / depth;
  return [cam.w / 2 + rx * s, cam.h * 0.70 - (dz - pitch * ry) * s, depth];
}

function draw() {
  const w = cv.clientWidth, h = cv.clientHeight, dpr = Math.min(devicePixelRatio || 1, 2);
  if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  ctx.clearRect(0, 0, w, h);

  const cam = { x: data.qpos[0], y: data.qpos[1], w, h, scale: Math.min(w, h) * 1.15 };

  // floor
  ctx.strokeStyle = css('--rule'); ctx.lineWidth = 1; ctx.globalAlpha = 0.55;
  const step = 0.1, span = 0.8;
  const gx = Math.round(cam.x / step) * step, gy = Math.round(cam.y / step) * step;
  const seg = (a, b) => {
    const p = project(a, cam), q2 = project(b, cam);
    if (p[2] <= 0.16 || q2[2] <= 0.16) return;
    ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q2[0], q2[1]); ctx.stroke();
  };
  for (let i = -span; i <= span + 1e-9; i += step) {
    seg([gx + i, gy - span, 0], [gx + i, gy + span, 0]);
    seg([gx - span, gy + i, 0], [gx + span, gy + i, 0]);
  }
  ctx.globalAlpha = 1;

  // contact shadow
  ctx.fillStyle = css('--rule'); ctx.globalAlpha = 0.55;
  const sh = project([data.qpos[0], data.qpos[1], 0.001], cam);
  ctx.beginPath(); ctx.ellipse(sh[0], sh[1], 42, 14, 0, 0, Math.PI * 2); ctx.fill();
  ctx.globalAlpha = 1;

  // limbs and volumes, painted back to front
  const palette = { shell: css('--shell'), joint: css('--bracket'), foot: css('--duck') };
  const world = b => [data.xpos[b * 3], data.xpos[b * 3 + 1], data.xpos[b * 3 + 2]];
  const PARENT = { 2:1, 3:2, 4:3, 5:4, 6:5, 7:1, 8:7, 9:8, 10:9, 11:1, 12:11, 13:12, 14:13, 15:14 };

  const shapes = [];
  for (const key of Object.keys(LIMB)) {
    const child = +key, spec = LIMB[child];
    shapes.push({ corners: boxBetween(world(PARENT[child]), world(child), spec.r), c: spec.c });
  }
  for (const blk of BLOCKS) {
    const pos = world(blk.body);
    const quat = [data.xquat[blk.body * 4], data.xquat[blk.body * 4 + 1],
                  data.xquat[blk.body * 4 + 2], data.xquat[blk.body * 4 + 3]];
    const half = [blk.w / 2, blk.d / 2, blk.h / 2];
    shapes.push({
      corners: CORNERS.map(c => {
        const local = [c[0] * half[0], c[1] * half[1], c[2] * half[2] + blk.up];
        const wpt = rotate(quat, local);
        return [pos[0] + wpt[0], pos[1] + wpt[1], pos[2] + wpt[2]];
      }), c: blk.c,
    });
  }

  const quads = [];
  for (const shape of shapes) {
    const pts = shape.corners.map(p => project(p, cam));
    for (const f of FACES) {
      const [ax, ay] = pts[f[0]], [bx, by] = pts[f[1]], [cx2, cy2] = pts[f[2]];
      if ((bx - ax) * (cy2 - ay) - (by - ay) * (cx2 - ax) <= 0) continue;  // back-face cull
      const depth = (pts[f[0]][2] + pts[f[1]][2] + pts[f[2]][2] + pts[f[3]][2]) / 4;
      quads.push({ depth, pts: f.map(k => pts[k]), fill: palette[shape.c] });
    }
  }
  quads.sort((a, b) => b.depth - a.depth);
  ctx.lineJoin = 'round';
  ctx.strokeStyle = css('--outline'); ctx.lineWidth = 1;
  for (const q of quads) {
    ctx.beginPath();
    ctx.moveTo(q.pts[0][0], q.pts[0][1]);
    for (let k = 1; k < 4; k++) ctx.lineTo(q.pts[k][0], q.pts[k][1]);
    ctx.closePath();
    ctx.fillStyle = q.fill; ctx.fill(); ctx.stroke();
  }

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
