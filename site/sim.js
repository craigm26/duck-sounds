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
import { makePad, ACTIONS } from './gamepad.js';
import { xrSupport, startXR } from './xr.js';

// The two robots. Skates are a DIFFERENT MODEL, not a different policy: four
// extra bodies on passive wheel joints, so the physics, the geometry and the
// drive network all have to change together. Everything is fetched on switch,
// never up front — most visitors never touch it.
const VARIANTS = {
  legs:    { mjb: 'scene.mjb',         visual: 'duck-visual.bin',
             drive: 'alpha_walking.onnx', speed: 0.45, note: 'walking' },
  rollers: { mjb: 'scene-rollers.mjb', visual: 'duck-visual-rollers.bin',
             drive: 'BEST_roller.onnx',   speed: 0.45, note: 'skating, ~0.59 m/s' },
};

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
let variant = 'legs';
let driveSpeed = 0.45;
let switching = false;
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
    zoom,
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
let pad = null, padCmd = null;
let xrSession = null;
let zoom = 1;
const SLOT_STORE = 'microduck.slots.v1';

function readControls() {
  // A connected pad wins over the thumbstick, which wins over the keyboard —
  // whichever the person actually touched last is the one they meant.
  if (padCmd && (Math.abs(padCmd.vx) > 0.001 || Math.abs(padCmd.vyaw) > 0.001)) {
    cmdState.vx = padCmd.vx * driveSpeed;
    cmdState.vyaw = padCmd.vyaw * 1.0;
    return;
  }
  if (stickCmd) { cmdState.vx = stickCmd.vx; cmdState.vyaw = stickCmd.vyaw; return; }
  const fwd = (keys.has('ArrowUp') || keys.has('w') ? 1 : 0) - (keys.has('ArrowDown') || keys.has('s') ? 1 : 0);
  const turn = (keys.has('ArrowLeft') || keys.has('a') ? 1 : 0) - (keys.has('ArrowRight') || keys.has('d') ? 1 : 0);
  cmdState.vx = fwd * driveSpeed;
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

// ── zoom ──────────────────────────────────────────────────────────────────
const zoomEl = document.getElementById('zoom'), zoomOut = document.getElementById('zoomOut');
function setZoom(z) {
  zoom = Math.min(Math.max(z, 0.45), 3.2);
  zoomEl.value = Math.round(zoom * 100);
  zoomOut.textContent = zoom.toFixed(1) + '\u00d7';
}
zoomEl.addEventListener('input', () => setZoom(+zoomEl.value / 100));
// Wheel on desktop.
cv.addEventListener('wheel', e => {
  e.preventDefault();
  setZoom(zoom * (e.deltaY < 0 ? 1.12 : 1 / 1.12));
}, { passive: false });
// Pinch on touch. Two pointers on the canvas, distance ratio drives the zoom.
const pinch = new Map();
let pinchBase = null;
cv.addEventListener('pointerdown', e => { pinch.set(e.pointerId, e); });
cv.addEventListener('pointermove', e => {
  if (!pinch.has(e.pointerId)) return;
  pinch.set(e.pointerId, e);
  if (pinch.size !== 2) return;
  const [a, b] = [...pinch.values()];
  const d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
  if (pinchBase === null) { pinchBase = { d, zoom }; return; }
  setZoom(pinchBase.zoom * (d / Math.max(pinchBase.d, 1)));
});
for (const ev of ['pointerup', 'pointercancel', 'pointerleave']) {
  cv.addEventListener(ev, e => { pinch.delete(e.pointerId); if (pinch.size < 2) pinchBase = null; });
}

// ── which intent each on-screen button fires ─────────────────────────────
// The phone layout has room for two action buttons, and which two should not
// be a decision made here for everyone. Saved per browser.
const SLOT_DEFAULT = { A: 'kick_left', B: 'step_up' };
function loadSlots() {
  try { return { ...SLOT_DEFAULT, ...JSON.parse(localStorage.getItem(SLOT_STORE) || '{}') }; }
  catch { return { ...SLOT_DEFAULT }; }
}
function buildSlots() {
  const all = [...INTENTS, { id: 'step_up', label: 'Step up' }];
  const slots = loadSlots();
  for (const name of ['A', 'B']) {
    const sel = document.getElementById('slot' + name);
    for (const item of all) {
      const o = document.createElement('option');
      o.value = item.id; o.textContent = item.label;
      sel.appendChild(o);
    }
    sel.value = slots[name];
    const apply = () => {
      const btn = document.querySelector(`.pad-btn[data-slot="${name}"]`);
      const item = all.find(i => i.id === sel.value);
      btn.dataset.intent = sel.value;
      btn.querySelector('em').textContent = item ? item.label.toLowerCase() : sel.value;
      const saved = loadSlots(); saved[name] = sel.value;
      try { localStorage.setItem(SLOT_STORE, JSON.stringify(saved)); } catch {}
    };
    sel.addEventListener('change', apply);
    apply();
  }
}

const variantEl = document.getElementById('variant');
const variantOut = document.getElementById('variantOut');
variantEl.addEventListener('change', async () => {
  if (switching) { variantEl.value = variant; return; }
  variantEl.disabled = true;
  try {
    await loadVariant(variantEl.value);
    variantOut.textContent = VARIANTS[variant].note;
    readStairs();
  } catch (err) {
    statusEl.textContent = 'Could not load that variant: ' + (err.message || err);
    variantEl.value = variant;
  } finally {
    variantEl.disabled = false;
  }
});

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

function buildPad() {
  const stateEl = document.getElementById('padState');
  const listEl = document.getElementById('padList');
  const all = [...INTENTS, { id: 'step_up', label: 'Step up' }];

  pad = makePad({
    onConnect: id => {
      stateEl.textContent = 'connected';
      stateEl.classList.add('on');
      document.getElementById('padHint').textContent = id.slice(0, 48);
    },
    onAction: id => {
      // Pad actions are named the same as intents where they overlap; the two
      // that are not intents are handled here.
      if (id === 'reset') return reset();
      if (id === 'variant') {
        const el = document.getElementById('variant');
        el.value = el.value === 'legs' ? 'rollers' : 'legs';
        el.dispatchEvent(new Event('change'));
        return;
      }
      if (id === 'sit_toggle') {
        // One button for both directions, as the runtime has it: which one you
        // get depends on whether the duck is already down.
        const down = data.qpos[DUCK.freeQpos + 2] < 0.09;
        return fire(all.find(i => i.id === (down ? 'stand' : 'sit')));
      }
      const item = all.find(i => i.id === id);
      if (item) fire(item);
    },
  });

  for (const a of ACTIONS) {
    const row = document.createElement('div');
    row.className = 'ctl';
    const label = document.createElement('label');
    label.textContent = a.label + (a.hold ? ' (hold)' : '');
    const btn = document.createElement('button');
    btn.className = 'mini';
    btn.textContent = 'set';
    const shown = document.createElement('span');
    shown.className = 'btnid';
    const paint = () => { shown.textContent = 'btn ' + pad.map[a.id]; };
    paint();
    btn.addEventListener('click', async () => {
      btn.textContent = 'press…';
      const b = await pad.listen(a.id);
      btn.textContent = 'set';
      paint();
      if (b === undefined) return;
    });
    row.append(label, btn, shown);
    listEl.appendChild(row);
  }
  document.getElementById('padReset').addEventListener('click', () => {
    pad.reset();
    for (const [i, a] of ACTIONS.entries()) {
      listEl.children[i].querySelector('.btnid').textContent = 'btn ' + pad.map[a.id];
    }
  });
  addEventListener('gamepadconnected', () => { stateEl.textContent = 'connected'; stateEl.classList.add('on'); });
  addEventListener('gamepaddisconnected', () => { stateEl.textContent = 'not connected'; stateEl.classList.remove('on'); });
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
      : { vx: -y * driveSpeed, vyaw: -x * 1.0 };
    window.__stick = stickCmd;
  });
  const all = [...INTENTS, { key: STEP_UP_KEY, id: 'step_up', label: 'Step up' }];
  for (const el of document.querySelectorAll('.pad-btn')) {
    // Read the intent at PRESS time, not at wiring time, so re-assigning a
    // button takes effect without rebuilding the handler.
    el.addEventListener('pointerdown', e => {
      e.preventDefault();
      const item = all.find(i => i.id === el.dataset.intent);
      if (item) fire(item);
    });
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
/**
 * Load a variant: its physics model, its geometry and its drive policy.
 *
 * All three move together — skates are four extra bodies on passive wheel
 * joints, so the mesh pack and the network are as specific to the model as the
 * .mjb is. Joints and bodies are re-resolved by name afterwards, because the
 * two models do not agree on a single index.
 */
async function loadVariant(name) {
  const v = VARIANTS[name];
  switching = true;
  statusEl.textContent = `loading the ${name === 'legs' ? 'robot' : 'skates'}\u2026`;

  const mjbBuf = await (await fetch('./' + v.mjb)).arrayBuffer();
  mj.FS.writeFile('/' + v.mjb, new Uint8Array(mjbBuf));
  model = mj.MjModel.mj_loadBinary('/' + v.mjb, new mj.MjVFS());
  data = new mj.MjData(model);

  GEOM_TYPES = {
    plane: mj.mjtGeom.mjGEOM_PLANE.value, sphere: mj.mjtGeom.mjGEOM_SPHERE.value,
    capsule: mj.mjtGeom.mjGEOM_CAPSULE.value, box: mj.mjtGeom.mjGEOM_BOX.value,
    mesh: mj.mjtGeom.mjGEOM_MESH.value,
  };
  GYRO = 0;
  for (let i = 0; i < model.nsensor; i++) {
    if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;
  }
  DUCK = findDuckJoints(model);
  STAIRS = findStairJoints(model);

  renderer = await createRenderer(cv, './' + v.visual);
  session = await ort.InferenceSession.create('./' + v.drive);
  inputName = session.inputNames[0];
  driveSpeed = v.speed;
  variant = name;
  skillSessions.clear();
  skill = null; intent = null; manual = null;
  reset();
  statusEl.textContent = '';
  switching = false;
}

(async function start() {
  try {
    statusEl.textContent = 'loading physics…';
    C = await (await fetch('./duckkit-constants.json')).json();
    ({ HOME, buildObs, gaitTargets, projectedGravity, command, findDuckJoints } = makeLoop(C));
    mj = await loadMuJoCo();
    await loadVariant('legs');
    try {
      stepupParams = (await (await fetch('./intent-stepup.json')).json()).params;
    } catch { /* the button simply does nothing without it */ }
    // AR, where the browser has it. The button stays hidden otherwise rather
    // than offering something that will fail — Safari has no WebXR at all,
    // which is why this project's iOS path is native.
    const support = await xrSupport();
    const xrBtn = document.getElementById('xr');
    const mode = support.ar ? 'immersive-ar' : support.vr ? 'immersive-vr' : null;
    if (mode) {
      xrBtn.hidden = false;
      xrBtn.textContent = support.ar ? 'view in AR' : 'view in VR';
      xrBtn.addEventListener('click', async () => {
        if (xrSession) { await xrSession.end(); return; }
        try {
          xrSession = await startXR({
            gl: renderer.gl, mode,
            step: () => { /* the 50 Hz loop keeps running below */ },
            onFrame: ({ view, proj, origin }) => {
              renderer.render(data, {
                bg: [0, 0, 0], grid: [0, 0, 0], root: DUCK.freeQpos,
                model, geomTypes: GEOM_TYPES, xr: { view, proj, origin },
              });
            },
            onEnd: () => { xrSession = null; xrBtn.textContent = support.ar ? 'view in AR' : 'view in VR'; },
          });
          xrBtn.textContent = 'leave AR';
        } catch (err) {
          statusEl.textContent = 'AR could not start: ' + (err.message || err);
          setTimeout(() => { statusEl.textContent = ''; }, 4000);
        }
      });
    }

    buildSlots();
    setZoom(1);
    buildPad();
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
      padCmd = pad ? pad.poll(now) : null;
      readControls();
      if (!busy && !switching) {
        busy = true;
        let n = 0;
        while (acc >= stepMs && n < 4) { acc -= stepMs; await tick(); n++; }
        busy = false;
      }
      if (!switching && !xrSession) draw();
      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
  } catch (err) {
    statusEl.textContent = 'The simulator could not start: ' + (err && err.message ? err.message : err);
    console.error(err);
  }
})();
