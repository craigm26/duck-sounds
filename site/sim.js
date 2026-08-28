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
import { findStairJoints, layoutStairs, clearStairs, STAIR_COUNT } from './stairs.js';
import { buildTrack, poseAt } from './intent.mjs';
import { INTENTS, STEP_UP_KEY } from './intents.js';
import { isTouch, makeStick } from './touch.js';

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
let model, data, mj, session, inputName, C, HOME, buildObs, gaitTargets, projectedGravity, command, findDuckJoints;
let lastAction = new Array(14).fill(0), previous = null, ticks = 0, GYRO = 0;
let GEOM_TYPES = null;
let STAIRS = null, DUCK = null, stairCfg = { count: 8, rise: 0, run: 0.09, start: 0.45 };
let manual = null;   // 14 hand-set targets, or null while the policy drives
let intent = null;   // { params, track, t0 } while the step-up move is playing
let stepupParams = null;
let skill = null;           // { intent, session, t0 } while a skill holds the robot
const skillSessions = new Map();   // policies are fetched on first use, not up front

function reset() {
  mj.mj_resetData(model, data);
  data.qpos[DUCK.freeQpos + 2] = 0.12; data.qpos[DUCK.freeQpos + 3] = 1;
  for (let i = 0; i < 14; i++) { data.qpos[DUCK.qpos[i]] = HOME[i]; data.ctrl[i] = HOME[i]; }
  applyStairs();
  mj.mj_forward(model, data);
  lastAction = new Array(14).fill(0); previous = null; ticks = 0;
  skill = null; intent = null;
}

function applyStairs() {
  if (!STAIRS) return;
  if (stairCfg.rise > 0) layoutStairs(data, STAIRS, stairCfg);
  else clearStairs(data, STAIRS);
}

async function tick() {
  // The steps are heavy bodies on frictionless slides: their position has to be
  // re-asserted every tick, velocity included, or they drift and then catapult.
  applyStairs();

  const f = DUCK.freeQpos;
  const q = [data.qpos[f+3], data.qpos[f+4], data.qpos[f+5], data.qpos[f+6]];
  const gyro = [data.sensordata[GYRO], data.sensordata[GYRO + 1], data.sensordata[GYRO + 2]];
  const jpos = [], jvel = [];
  for (let k = 0; k < 14; k++) { jpos.push(data.qpos[DUCK.qpos[k]]); jvel.push(data.qvel[DUCK.dof[k]]); }

  if (manual) {
    for (let k = 0; k < 14; k++) data.ctrl[k] = manual[k];
  } else if (skill) {
    // A skill IS the policy while it runs — a different trained network on the
    // same 61-in/14-out contract, not a modifier on the walker.
    const u = (ticks - skill.t0) / (skill.intent.seconds * C.tickHz);
    const obs = buildObs(gyro, projectedGravity(q), jpos, jvel, lastAction,
                         command(skill.intent.cmd(Math.min(u, 1))));
    const out = await skill.session.run({ [inputName]: new ort.Tensor('float32', obs, [1, 61]) });
    lastAction = Array.from(out[session.outputNames[0]].data);
    for (let k = 0; k < 14; k++) data.ctrl[k] = HOME[k] + lastAction[k];
    if (u >= 1) { skill = null; paintKeys(); }
  } else if (intent) {
    // The intent is an OFFSET on the policy, never a replacement. Replacing it
    // collapses the duck: with kp 0.55 the servos cannot hold a pose, so
    // balance is something the policy does continuously, not something a pose
    // encodes. Measured, an open-loop version fell over on a flat floor.
    const cmd = command({ vx: intent.params.approach });
    const obs = buildObs(gyro, projectedGravity(q), jpos, jvel, lastAction, cmd);
    const out = await session.run({ [inputName]: new ort.Tensor('float32', obs, [1, 61]) });
    lastAction = Array.from(out[session.outputNames[0]].data);
    const elapsed = (ticks - intent.t0) / C.tickHz;
    const offset = poseAt(intent.track, elapsed, HOME);
    for (let k = 0; k < 14; k++) {
      data.ctrl[k] = HOME[k] + lastAction[k] + (offset[k] - HOME[k]) * intent.params.blend;
    }
    if (elapsed > intent.track[intent.track.length - 1].t + 0.6) intent = null;
  } else {
    const obs = buildObs(gyro, projectedGravity(q), jpos, jvel, lastAction, command(cmdState));
    const out = await session.run({ [inputName]: new ort.Tensor('float32', obs, [1, 61]) });
    lastAction = Array.from(out[session.outputNames[0]].data);
    for (let k = 0; k < 14; k++) data.ctrl[k] = HOME[k] + lastAction[k];
  }
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
    root: DUCK.freeQpos,
    model,
    geomTypes: GEOM_TYPES,
  });
  const f = DUCK.freeQpos, fd = DUCK.freeDof;
  const speed = Math.hypot(data.qvel[fd], data.qvel[fd+1]);
  hud.textContent =
    `tick ${String(ticks).padStart(5, '0')}   ` +
    `speed ${speed.toFixed(2)} m/s   ` +
    `height ${data.qpos[f+2].toFixed(3)} m   ` +
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

// Set by the thumbstick on touch devices; the keyboard path leaves it null.
let stickCmd = null;

function readControls() {
  if (stickCmd) { cmdState.vx = stickCmd.vx; cmdState.vyaw = stickCmd.vyaw; return; }
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

// ── stairs ────────────────────────────────────────────────────────────────
const riseEl = document.getElementById('rise'), riseOut = document.getElementById('riseOut');
const countEl = document.getElementById('count'), countOut = document.getElementById('countOut');
const runEl = document.getElementById('run'), runOut = document.getElementById('runOut');
const ctlNote = document.getElementById('ctlNote');
function readStairs() {
  stairCfg = {
    count: +countEl.value,
    rise: +riseEl.value / 1000,
    run: +runEl.value / 1000,
    start: 0.45,
  };
  riseOut.textContent = +riseEl.value === 0 ? 'flat' : riseEl.value + ' mm';
  countOut.textContent = countEl.value;
  runOut.textContent = runEl.value + ' mm';
  // Measured, not guessed: 1 and 2 mm are walkable, 3 mm and up are not.
  ctlNote.textContent = +riseEl.value === 0
    ? 'Flat floor.'
    : (+riseEl.value <= 2
        ? 'It can walk up these.'
        : `It cannot walk up ${riseEl.value} mm. Measured: 2 mm is the limit, 3 mm tips it over.`);
  if (data) applyStairs();
}
for (const el of [riseEl, countEl, runEl]) el.addEventListener('input', readStairs);

// ── servos ────────────────────────────────────────────────────────────────
const modeEl = document.getElementById('mode');
const servoList = document.getElementById('servoList');
const servoInputs = [];
function buildServos() {
  const names = C.jointNames.filter(n => n !== 'mouth');
  const lo = C.rangeLo.filter((_, i) => C.jointNames[i] !== 'mouth');
  const hi = C.rangeHi.filter((_, i) => C.jointNames[i] !== 'mouth');
  names.forEach((name, k) => {
    const row = document.createElement('div');
    row.className = 'ctl';
    const label = document.createElement('label');
    label.textContent = name;
    const input = document.createElement('input');
    input.type = 'range';
    // The real travel limits from the model, so a slider cannot ask for a
    // joint angle the robot does not have.
    input.min = lo[k].toFixed(4); input.max = hi[k].toFixed(4);
    input.step = 0.005; input.value = HOME[k].toFixed(4);
    const out = document.createElement('output');
    const show = () => { out.textContent = (+input.value).toFixed(2); };
    input.addEventListener('input', () => {
      show();
      if (manual) manual[k] = +input.value;
    });
    show();
    row.append(label, input, out);
    servoList.appendChild(row);
    servoInputs.push(input);
  });
}
modeEl.addEventListener('change', () => {
  if (modeEl.value === 'manual') {
    // Start from whatever it is doing right now, so you can catch a pose
    // mid-stride and edit it rather than beginning from the home stance.
    manual = [];
    for (let k = 0; k < 14; k++) {
      const v = data.ctrl[k];
      manual.push(v);
      servoInputs[k].value = v.toFixed(4);
      servoInputs[k].nextElementSibling.textContent = v.toFixed(2);
    }
  } else {
    manual = null;
    previous = null;
  }
});
// ── intents ───────────────────────────────────────────────────────────────
const keyRow = document.getElementById('keys');
const keyButtons = new Map();

function busy() { return skill !== null || intent !== null; }

function paintKeys() {
  for (const [id, el] of keyButtons) {
    const active = (skill && skill.intent.id === id) || (intent && id === 'step_up');
    el.classList.toggle('on', !!active);
    el.disabled = busy() && !active;
  }
}

async function fire(item) {
  // One-shot and exclusive, as robotd treats them: a skill arriving while
  // another holds the robot is refused, not blended into it.
  if (busy()) return;
  manual = null; modeEl.value = 'policy';
  if (item.id === 'step_up') {
    if (!stepupParams) return;
    intent = { params: stepupParams, track: buildTrack(stepupParams, HOME), t0: ticks };
  } else {
    let s = skillSessions.get(item.policy);
    if (!s) {
      keyRow.classList.add('loading');
      s = await ort.InferenceSession.create('./' + item.policy);
      skillSessions.set(item.policy, s);
      keyRow.classList.remove('loading');
    }
    skill = { intent: item, session: s, t0: ticks };
  }
  paintKeys();
}

function buildTouch() {
  if (!isTouch()) return;
  document.body.classList.add('is-touch');
  // Forward is up. The policy needs about 0.3 before it walks at all, so the
  // stick starts there rather than at zero — a thumb barely off centre should
  // move the duck, not sit in a dead band.
  makeStick(document.getElementById('stick'), (x, y) => {
    const push = Math.hypot(x, y);
    stickCmd = push < 0.12
      ? { vx: 0, vyaw: 0 }
      : { vx: -y * 0.45, vyaw: -x * 1.0 };
    window.__stick = stickCmd;
  });
  const all = [...INTENTS, { key: STEP_UP_KEY, id: 'step_up', label: 'Step up' }];
  for (const el of document.querySelectorAll('.pad-btn')) {
    const item = all.find(i => i.id === el.dataset.intent);
    if (item) el.addEventListener('pointerdown', e => { e.preventDefault(); fire(item); });
  }
}

function buildKeys() {
  const all = [...INTENTS, { key: STEP_UP_KEY, id: 'step_up', label: 'Step up' }];
  for (const item of all) {
    const b = document.createElement('button');
    b.innerHTML = `<kbd>${item.key.toUpperCase()}</kbd><span>${item.label}</span>`;
    b.title = item.id === 'step_up'
      ? 'The authored move: plant the head, lift onto a 26 mm step'
      : `${item.policy} for ${item.seconds}s`;
    b.addEventListener('click', () => fire(item));
    keyRow.appendChild(b);
    keyButtons.set(item.id, b);
  }
  const byKey = new Map(all.map(i => [i.key, i]));
  addEventListener('keydown', e => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const item = byKey.get(e.key.toLowerCase());
    if (item) { e.preventDefault(); fire(item); }
  });
}
document.getElementById('copyPose').addEventListener('click', async () => {
  const names = C.jointNames.filter(n => n !== 'mouth');
  const pose = Object.fromEntries(names.map((n, k) => [n, +(manual ? manual[k] : data.ctrl[k]).toFixed(4)]));
  const text = JSON.stringify(pose, null, 2);
  try { await navigator.clipboard.writeText(text); ctlNote.textContent = 'Pose copied.'; }
  catch { console.log(text); ctlNote.textContent = 'Pose logged to the console.'; }
});

// ── boot ───────────────────────────────────────────────────────────────────
(async function start() {
  try {
    statusEl.textContent = 'loading physics…';
    C = await (await fetch('./duckkit-constants.json')).json();
    ({ HOME, buildObs, gaitTargets, projectedGravity, command, findDuckJoints } = makeLoop(C));
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
    // MuJoCo's geom type enum, read from the module rather than hardcoded.
    GEOM_TYPES = {
      plane: mj.mjtGeom.mjGEOM_PLANE.value, sphere: mj.mjtGeom.mjGEOM_SPHERE.value,
      capsule: mj.mjtGeom.mjGEOM_CAPSULE.value, box: mj.mjtGeom.mjGEOM_BOX.value,
      mesh: mj.mjtGeom.mjGEOM_MESH.value,
    };
    DUCK = findDuckJoints(model);
    STAIRS = findStairJoints(model);

    statusEl.textContent = 'loading the policy…';
    session = await ort.InferenceSession.create('./alpha_walking.onnx');
    inputName = session.inputNames[0];

    statusEl.textContent = 'loading the robot\u2019s geometry\u2026';
    renderer = await createRenderer(cv, './duck-visual.bin');
    console.log('renderer:', renderer.draws, 'parts,', renderer.triangles, 'triangles');

    try {
      stepupParams = (await (await fetch('./intent-stepup.json')).json()).params;
    } catch { /* the button simply does nothing without it */ }
    buildTouch();
    buildKeys();
    buildServos();
    readStairs();
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
