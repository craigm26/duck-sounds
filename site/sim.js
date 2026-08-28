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
import { INTENTS, STEP_UP_KEY, BACK_ROLL_KEY, LEVER_KEY, WALL_FLIP_KEY, RISER_KEY, CLIMB_KEY, DEFAULTS, speeds } from './intents.js';
import { isTouch, makeStick } from './touch.js';
import { readiness as readyOf, stagingFor, SETTLE_TICKS, SETTLE_POLICY, TOL, NEEDS } from './intent-specs.js';
import { yawOf } from './approach.js';
import { makeRunner, stageFor, describe as describeStep } from './sequence.js';
import { makePad, ACTIONS } from './gamepad.js';
import { xrSupport, startXR } from './xr.js';

// The two robots. Skates are a DIFFERENT MODEL, not a different policy: four
// extra bodies on passive wheel joints, so the physics, the geometry and the
// drive network all have to change together. Everything is fetched on switch,
// never up front — most visitors never touch it.
const VARIANTS = {
  legs:    { mjb: 'scene.mjb',         visual: 'duck-visual.bin',
             drive: 'alpha_walking.onnx', note: 'walking' },
  rollers: { mjb: 'scene-rollers.mjb', visual: 'duck-visual-rollers.bin',
             drive: 'BEST_roller.onnx',   note: 'skating' },
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
let STAIRS = null, DUCK = null, stairCfg = { count: 8, rise: 0, run: 0.28, start: 0.45 };
let manual = null;   // 14 hand-set targets, or null while the policy drives
let intent = null;   // { params, track, t0 } while the step-up move is playing
let stepupParams = null;
let backRoll = null;
const tracks = {};   // shipped keyframe moves, by intent id
let variant = 'legs';
let driveSpeed = speeds('legs');
let switching = false;
let skill = null;           // { intent, session, t0 } while a skill holds the robot
let lockUntil = 0;          // post-kick input lock, as the runtime has it
// A MODE is the policy currently driving — sit, stand, hold. It is not an
// exclusive hold: you can always ask for something else, and that is the whole
// difference between it and a one-shot. Conflating the two meant a sit (which
// by design never ends) held the robot forever and refused to stand up again.
let mode = null;
const skillSessions = new Map();   // policies are fetched on first use, not up front
// A command from the sequence runner, which outranks every human input while it
// holds — otherwise a hand on the arrow keys fights the approach controller.
let autoCmd = null;
// The 25-tick hold under the STAND policy that every searched move begins from.
let settleState = null;   // { until, approach, session }

function reset() {
  mj.mj_resetData(model, data);
  data.qpos[DUCK.freeQpos + 2] = 0.12; data.qpos[DUCK.freeQpos + 3] = 1;
  for (let i = 0; i < 14; i++) { data.qpos[DUCK.qpos[i]] = HOME[i]; data.ctrl[i] = HOME[i]; }
  applyStairs();
  mj.mj_forward(model, data);
  lastAction = new Array(14).fill(0); previous = null; ticks = 0;
  skill = null; intent = null; mode = null; lockUntil = 0;
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
  } else if (settleState) {
    // Exactly what the search did before playing a track: the stand policy,
    // holding the move's own approach command, for a fixed number of control
    // ticks. Not decoration — a duck settled any other way has joint velocities
    // out by up to 0.147, and those are 14 of the 61 observation floats.
    const obs = buildObs(gyro, projectedGravity(q), jpos, jvel, lastAction,
                         command({ vx: settleState.approach }));
    const out = await settleState.session.run({ [inputName]: new ort.Tensor('float32', obs, [1, 61]) });
    lastAction = Array.from(out[settleState.session.outputNames[0]].data);
    for (let k = 0; k < 14; k++) data.ctrl[k] = HOME[k] + lastAction[k];
    if (ticks >= settleState.until) settleState = null;
  } else if (skill) {
    // A skill IS the policy while it runs — a different trained network on the
    // same 61-in/14-out contract, not a modifier on the walker.
    const it = skill.intent;
    const elapsed = ticks - skill.t0;
    // A phase intent feeds a clock into the command slots, over the policy's
    // own period rather than over however long we happen to run it.
    const u = it.kind === 'phase'
      ? (elapsed / C.tickHz) / it.period
      : (it.steps ? elapsed / it.steps : 0);
    const obs = buildObs(gyro, projectedGravity(q), jpos, jvel, lastAction,
                         command(it.cmd(u)));
    const out = await skill.session.run({ [inputName]: new ort.Tensor('float32', obs, [1, 61]) });
    lastAction = Array.from(out[session.outputNames[0]].data);
    for (let k = 0; k < 14; k++) data.ctrl[k] = HOME[k] + lastAction[k];

    const up = projectedGravity(q)[2] < -0.85;
    let done = false;
    if (it.kind === 'until') {
      if (!up) skill.tipped = true;
      done = (skill.tipped && up && elapsed >= it.minSteps) || elapsed >= it.steps;
    } else {
      done = elapsed >= it.steps;
    }
    if (done) {
      lockUntil = ticks + (it.lock || 0);
      skill = null;
      paintKeys();
    }
  } else if (mode) {
    const it = mode.intent;
    const elapsed = ticks - mode.t0;
    const obs = buildObs(gyro, projectedGravity(q), jpos, jvel, lastAction, command(it.cmd(0)));
    const out = await mode.session.run({ [inputName]: new ort.Tensor('float32', obs, [1, 61]) });
    lastAction = Array.from(out[session.outputNames[0]].data);
    for (let k = 0; k < 14; k++) data.ctrl[k] = HOME[k] + lastAction[k];
    // "Stand up" hands back to walking once the duck is actually up. A sit
    // holds until something else is asked for.
    if (it.handBack) {
      const up = projectedGravity(q)[2] < -0.85;
      if (up && data.qpos[DUCK.freeQpos + 2] > 0.10 && elapsed > 40) { mode = null; paintKeys(); }
    }
  } else if (intent) {
    // The intent is an OFFSET on the policy, never a replacement. Replacing it
    // collapses the duck: with kp 0.55 the servos cannot hold a pose, so
    // balance is something the policy does continuously, not something a pose
    // encodes. Measured, an open-loop version fell over on a flat floor.
    const cmd = command({ vx: intent.params.approach });
    const obs = buildObs(gyro, projectedGravity(q), jpos, jvel, lastAction, cmd);
    // A shipped track brings its own policy; the step-up rides the drive one.
    const runner = intent.session || session;
    const out = await runner.run({ [inputName]: new ort.Tensor('float32', obs, [1, 61]) });
    lastAction = Array.from(out[session.outputNames[0]].data);
    const elapsed = (ticks - intent.t0) / C.tickHz;
    const offset = poseAt(intent.track, elapsed, HOME);
    void 0;
    for (let k = 0; k < 14; k++) {
      data.ctrl[k] = HOME[k] + lastAction[k] + (offset[k] - HOME[k]) * (intent.blend ?? intent.params.blend);
    }
    if (elapsed > intent.track[intent.track.length - 1].t + (intent.tail ?? 0.6)) { intent = null; paintKeys(); }
  } else {
    const obs = buildObs(gyro, projectedGravity(q), jpos, jvel, lastAction, command(cmdState));
    const out = await session.run({ [inputName]: new ort.Tensor('float32', obs, [1, 61]) });
    lastAction = Array.from(out[session.outputNames[0]].data);
    for (let k = 0; k < 14; k++) data.ctrl[k] = HOME[k] + lastAction[k];
  }
  const dbg = window.__demo;
  if (dbg && dbg.recording > 0) {
    dbg.trace.push({
      t: ticks,
      obs: Array.from(buildObs(gyro, projectedGravity(q), jpos, jvel, lastAction,
                               command(intent ? { vx: intent.params.approach } : cmdState))).map(v => +v.toFixed(5)),
      ctrl: Array.from({ length: 14 }, (_, k) => +data.ctrl[k].toFixed(5)),
      z: +data.qpos[DUCK.freeQpos + 2].toFixed(5),
      gz: +projectedGravity(q)[2].toFixed(4),
      playing: intent ? 'intent' : skill ? 'skill' : mode ? 'mode' : 'drive',
    });
    dbg.recording--;
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
    shiftX,
    model,
    geomTypes: GEOM_TYPES,
  });
  const f = DUCK.freeQpos, fd = DUCK.freeDof;
  const speed = Math.hypot(data.qvel[fd], data.qvel[fd+1]);
  hudVal.tick.textContent = String(ticks).padStart(5, '0');
  hudVal.speed.textContent = speed.toFixed(2) + ' m/s';
  hudVal.height.textContent = data.qpos[f+2].toFixed(3) + ' m';
  hudVal.contacts.textContent = data.ncon;
}

// The readout was one pre-formatted line, which on a phone ran off the right
// edge into a horizontal scroll nobody would find. As label/value pairs it
// wraps, and the per-frame work is four textContent writes instead of building
// and re-parsing a string sixty times a second.
//
// The trailing space in the label is load-bearing: six headless checks in
// sim/ read this element with /speed ([\d.]+)/ and friends, so the element's
// textContent has to keep saying "speed 0.00 m/s" even though the layout no
// longer comes from the text. The space is inside the label rather than a flex
// gap for exactly that reason.
const hudVal = {};
for (const field of ['tick', 'speed', 'height', 'contacts']) {
  const wrap = document.createElement('span');
  wrap.className = 'hud-f';
  const key = document.createElement('i');
  key.textContent = field + ' ';
  const val = document.createElement('b');
  wrap.append(key, val);
  hud.appendChild(wrap);
  hudVal[field] = val;
}

// ── controls ───────────────────────────────────────────────────────────────
const keys = new Set();
/**
 * A key press means "drive the duck" everywhere EXCEPT inside a control that
 * has its own use for it.
 *
 * This did not matter while every slider and select sat far below the canvas.
 * It matters now that they are docked over it: arrowing the step-height slider
 * also walked the duck AND had its default prevented, so the slider did not
 * move at all, and a <select>'s type-ahead fired an intent per letter.
 */
const FORM_TAG = /^(INPUT|SELECT|TEXTAREA)$/;
export function typingIn(e) {
  const t = e.target;
  return !!t && (t.isContentEditable === true || FORM_TAG.test(t.tagName || ''));
}
// Space activates whatever has focus, so it is only ours when nothing that
// wants it does. Arrows scroll the page, and here they drive, so they are
// always swallowed outside a form control.
const SPACE_IS_THEIRS = /^(BUTTON|SUMMARY|A|DETAILS)$/;
addEventListener('keydown', e => {
  if (typingIn(e)) return;
  if (['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(e.key)) e.preventDefault();
  if (e.key === ' ' && !SPACE_IS_THEIRS.test(e.target && e.target.tagName || '')) e.preventDefault();
  if (e.key === 'r' || e.key === 'R') reset();
  keys.add(e.key);
});
addEventListener('keyup', e => keys.delete(e.key));
// Tab away mid-stride and the keyup never arrives, so the duck walks on with
// nobody driving. Clearing on blur is the only place that can be noticed.
addEventListener('blur', () => keys.clear());

// Set by the thumbstick on touch devices; the keyboard path leaves it null.
let stickCmd = null;
let pad = null, padCmd = null;
let xrSession = null;
let zoom = 1;
const SLOT_STORE = 'microduck.slots.v1';

function readControls() {
  // A sequence in flight outranks all three: the approach controller is closing
  // a loop on position, and a hand resting on an arrow key would fight it.
  if (autoCmd) { cmdState.vx = autoCmd.vx; cmdState.vyaw = autoCmd.vyaw; return; }
  // A connected pad wins over the thumbstick, which wins over the keyboard —
  // whichever the person actually touched last is the one they meant.
  if (padCmd && (Math.abs(padCmd.vx) > 0.001 || Math.abs(padCmd.vyaw) > 0.001)) {
    cmdState.vx = padCmd.vx > 0 ? padCmd.vx * driveSpeed.fwd : padCmd.vx * -driveSpeed.back;
    cmdState.vyaw = padCmd.vyaw * driveSpeed.ang;
    return;
  }
  if (stickCmd) { cmdState.vx = stickCmd.vx; cmdState.vyaw = stickCmd.vyaw; return; }
  const fwd = (keys.has('ArrowUp') || keys.has('w') ? 1 : 0) - (keys.has('ArrowDown') || keys.has('s') ? 1 : 0);
  const turn = (keys.has('ArrowLeft') || keys.has('a') ? 1 : 0) - (keys.has('ArrowRight') || keys.has('d') ? 1 : 0);
  // Forward and back are NOT symmetric in the runtime: 0.25 forward against
  // 0.2 back on legs, 0.6 against 0.5 on skates.
  cmdState.vx = fwd > 0 ? driveSpeed.fwd : fwd < 0 ? driveSpeed.back : 0;
  cmdState.vyaw = turn * driveSpeed.ang;
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
  // Measured, not guessed: 1 and 2 mm are walkable, 3 mm and up are not, and
  // the best move we have stands on 10 mm. The flat case used to read just
  // "Flat floor.", which threw away the one finding this control exists to
  // show — the slider is the argument, so the note has to make the ask.
  const mm = +riseEl.value;
  ctlNote.textContent =
    mm === 0 ? 'Flat floor. Raise the steps to find where it stops: walking alone clears 2 mm, and the best move we have stands on 10 mm.'
    : mm <= 2 ? `${mm} mm. It can walk up these.`
    : mm <= 10 ? `${mm} mm. Too tall to walk up — 2 mm is the walking limit. Try Riser up (Y), the highest we have got a foot onto: 10 mm.`
    : `${mm} mm. Nothing we have clears this. Walking tops out at 2 mm, the best move at 10 mm.`;
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

// ── stage layout ──────────────────────────────────────────────────────────
// All of this exists for one reason: on a desktop the simulator should BE the
// viewport, with the controls floating on it, rather than a 700x440 box in the
// middle of an otherwise empty 1920 px page. The CSS does the layout; these
// three measurements are the parts a stylesheet cannot know.
const rig = document.getElementById('rig');
const rail = document.getElementById('rail');
const railToggle = document.getElementById('railToggle');
const masthead = document.getElementById('masthead');
const fullBtn = document.getElementById('full');
let shiftX = 0;

/** The stage is sized against the masthead, which wraps at every width. */
function measureHead() {
  const h = masthead.offsetHeight + 'px';
  const root = document.documentElement;
  if (root.style.getPropertyValue('--head-h') !== h) root.style.setProperty('--head-h', h);
}

/**
 * How much of the canvas the rail is standing on, as a fraction of its width.
 *
 * The camera centres the duck in the CANVAS, and the rail covers the canvas's
 * right edge — so centred in the canvas is off-centre in what you can actually
 * see. render() shears the projection by this much to put the duck back in the
 * middle of the visible part. Measured from the live boxes rather than derived
 * from the CSS, so a collapsed rail, a fullscreen rig and the stacked layout
 * all fall out of the same two rectangles.
 */
function measureShift() {
  const c = cv.getBoundingClientRect();
  const r = rail.getBoundingClientRect();
  const over = !r.width || !r.height ? 0 : Math.max(0, c.right - r.left);
  const onTop = r.top < c.bottom && r.bottom > c.top;   // beside the canvas, not below it
  shiftX = c.width && onTop ? Math.min(over / c.width, 0.5) : 0;
}

const RAIL_STORE = 'microduck.rail.v1';
function setRail(open) {
  document.body.classList.toggle('rail-off', !open);
  railToggle.setAttribute('aria-expanded', String(open));
  const what = open ? 'Hide the controls' : 'Show the controls';
  railToggle.title = what;
  railToggle.setAttribute('aria-label', what);
  try { localStorage.setItem(RAIL_STORE, open ? '1' : '0'); } catch { /* private mode */ }
  measureShift();
}
railToggle.addEventListener('click', () => setRail(document.body.classList.contains('rail-off')));
try { if (localStorage.getItem(RAIL_STORE) === '0') setRail(false); } catch { /* likewise */ }

// Fullscreen takes the RIG, not the canvas, so every dock comes with it.
if (!document.documentElement.requestFullscreen) {
  fullBtn.hidden = true;
} else {
  fullBtn.addEventListener('click', async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await rig.requestFullscreen();
    } catch { /* refused; the button simply does nothing */ }
  });
  document.addEventListener('fullscreenchange', () => {
    const on = document.fullscreenElement === rig;
    const what = on ? 'Leave full screen' : 'Fill the screen';
    fullBtn.title = what;
    fullBtn.setAttribute('aria-label', what);
    // The rig resizes after the event, not before it.
    requestAnimationFrame(() => { measureHead(); measureShift(); });
  });
}

const relayout = () => { measureHead(); measureShift(); };
if (window.ResizeObserver) {
  const ro = new ResizeObserver(relayout);
  ro.observe(masthead); ro.observe(rig);
}
addEventListener('resize', relayout);
// A collapsed <details> in the rail changes its width when the scrollbar
// appears, which moves the occlusion edge.
rail.addEventListener('toggle', relayout, true);
relayout();

// ── which intent each on-screen button fires ─────────────────────────────
// The phone layout has room for two action buttons, and which two should not
// be a decision made here for everyone. Saved per browser.
const SLOT_DEFAULT = { A: 'kick_left', B: 'step_up' };
function loadSlots() {
  try { return { ...SLOT_DEFAULT, ...JSON.parse(localStorage.getItem(SLOT_STORE) || '{}') }; }
  catch { return { ...SLOT_DEFAULT }; }
}
function buildSlots() {
  const all = ALL_MOVES;
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
    input.id = 'servo-' + name;
    label.htmlFor = input.id;
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

/**
 * Every fireable move, in one place.
 *
 * This list was written out four times — in buildSlots, buildPad, buildTouch
 * and buildKeys — and the four had drifted: the gamepad's copy stopped after
 * step_up and back_roll, so a paired controller could not reach lever_up,
 * wall_flip, riser_up or climb at all, while the on-screen slot selects offered
 * all thirteen.
 */
const TRACK_MOVES = [
  { key: STEP_UP_KEY,   id: 'step_up',   label: 'Step up' },
  { key: BACK_ROLL_KEY, id: 'back_roll', label: 'Back roll' },
  { key: LEVER_KEY,     id: 'lever_up',  label: 'Lever up' },
  { key: WALL_FLIP_KEY, id: 'wall_flip', label: 'Wall flip' },
  { key: RISER_KEY,     id: 'riser_up',  label: 'Riser up' },
  { key: CLIMB_KEY,     id: 'climb',     label: 'Climb' },
];
const ALL_MOVES = [...INTENTS, ...TRACK_MOVES];

/** What a move is, and what it needs — the tooltip that used to say "undefined". */
function titleFor(id) {
  const item = ALL_MOVES.find(i => i.id === id);
  const t = tracks[id];
  const how = item && item.policy
    ? `${item.policy}${item.steps ? `, ${(item.steps / 50).toFixed(1)} s` : ''}`
    : t ? `${t.policy}, ${t.keyframes.length} keyframes`
    : id === 'step_up' ? 'an authored track over the drive policy'
    : 'authored';
  const need = NEEDS[id];
  const what = need === 'stair' ? 'needs a 10 mm step in front of it'
             : need === 'wall'  ? 'needs the arena wall beside it'
             : null;
  return what ? `${how} — ${what}` : how;
}

// Only ONE-SHOTS are exclusive. A mode can always be replaced.
function busy() { return skill !== null || intent !== null || ticks < lockUntil; }

// True while a policy is being fetched. Each one is 794 KB and arrives with no
// feedback, so a second press during the download used to fire the move twice.
let arming = false;

/**
 * A policy, fetched once and cached — with the wait made visible.
 *
 * Every one of these is 794 KB and the first press of any move pays for it.
 * Silently, before: the duck just did nothing for a second or two.
 */
async function policyFor(file) {
  let s = skillSessions.get(file);
  if (s) return s;
  arming = true; paintKeys();
  statusEl.textContent = 'loading the move\u2026';
  try {
    s = await ort.InferenceSession.create('./' + file);
    skillSessions.set(file, s);
    return s;
  } catch (err) {
    statusEl.textContent = 'that move could not be loaded: ' + (err.message || err);
    setTimeout(() => { if (/could not be loaded/.test(statusEl.textContent)) statusEl.textContent = ''; }, 4000);
    return null;
  } finally {
    arming = false;
    if (statusEl.textContent === 'loading the move\u2026') statusEl.textContent = '';
    paintKeys();
  }
}

function paintKeys() {
  for (const [id, el] of keyButtons) {
    // `intent.id`, not "is this id a track" — the old test lit step_up and every
    // shipped track at once, so firing Riser up highlighted six buttons and you
    // could not read which move the duck was actually doing.
    const active = (skill && skill.intent.id === id) || (mode && mode.intent.id === id)
      || (intent && intent.id === id);
    el.classList.toggle('on', !!active);
    el.disabled = (busy() || arming) && !active;
  }
  paintReady();
}

/**
 * Say whether each move can actually run here.
 *
 * Six of the thirteen need a stair or a wall the UI never checked, so pressing
 * them anywhere else played a track authored against something that was not
 * there. Recomputed off the live room rather than cached, but only WRITTEN when
 * the text changes — this is called every few frames and there are thirteen of
 * them.
 */
function paintReady() {
  for (const [id, el] of keyButtons) {
    const r = readyFor(id);
    const blocked = !!(r && !r.ok && r.fix === 'stairs');
    const title = titleFor(id) + (r && !r.ok ? ` — ${r.reason}` : '');
    if (el.title !== title) el.title = title;
    // Not colour alone: the reason is in the accessible name too.
    const label = el.textContent.trim() + (r && !r.ok ? `, ${r.reason}` : '');
    if (el.getAttribute('aria-label') !== label) el.setAttribute('aria-label', label);
    el.classList.toggle('unready', !!(r && !r.ok));
    el.classList.toggle('blocked', blocked);
  }
}

// paintKeys() only ever ran on a state CHANGE we made ourselves, and the
// post-kick input lock expires on the clock instead. So the last repaint after
// a kick happened while `ticks < lockUntil` was still true, every button was
// disabled, and nothing ever repainted again — one kick killed all thirteen
// buttons for good, and reset() cleared lockUntil without repainting. Polling
// the predicate once a frame is the only thing that catches a clock.
let wasBusy = false;
function repaintIfIdle() {
  const now = busy() || arming;
  if (now !== wasBusy) { wasBusy = now; paintKeys(); }
}

async function fire(item) {
  // One-shot and exclusive, as robotd treats them: a skill arriving while
  // another holds the robot is refused, not blended into it.
  if (arming) return;
  if (busy() && item.kind !== 'mode') return;
  manual = null; modeEl.value = 'policy';
  if (item.kind === 'mode') {
    // Modes replace each other freely and do not take the exclusive hold.
    const s = await policyFor(item.policy);
    if (!s) return;
    mode = { intent: item, session: s, t0: ticks };
    paintKeys();
    return;
  }
  // Any shipped keyframe move: back roll, lever, wall flip. They are all the
  // same machinery — an offset track played over a policy — and differ only in
  // which policy, how hard, and what command to hold while it plays.
  if (tracks[item.id]) {
    const t = tracks[item.id];
    const s = await policyFor(t.policy);
    if (!s) return;
    mode = null;
    intent = { id: item.id, track: t.keyframes, blend: t.blend, session: s, t0: ticks,
               params: { approach: t.approach || 0 }, tail: 1.2 };
    paintKeys();
    return;
  }
  if (item.id === 'step_up') {
    if (!stepupParams) return;
    intent = { id: 'step_up', params: stepupParams, track: buildTrack(stepupParams, HOME), t0: ticks };
  } else {
    const s = await policyFor(item.policy);
    if (!s) return;
    skill = { intent: item, session: s, t0: ticks };
  }
  paintKeys();
}

// ── sequences ─────────────────────────────────────────────────────────────
// Everything the runner needs from the simulator, handed over explicitly so
// sequence.js can be exercised without a DOM.
const pose = () => ({
  x: data.qpos[DUCK.freeQpos],
  y: data.qpos[DUCK.freeQpos + 1],
  yaw: yawOf([data.qpos[DUCK.freeQpos + 3], data.qpos[DUCK.freeQpos + 4],
              data.qpos[DUCK.freeQpos + 5], data.qpos[DUCK.freeQpos + 6]]),
});

/** The shipped JSON for an intent, which is where its staging comes from. */
function jsonFor(id) {
  if (id === 'step_up') return stepupParams ? { params: stepupParams } : null;
  return tracks[id] || null;
}

/** Can this move run, here, now? Null when the intent needs nothing staged. */
function readyFor(id) {
  if (!(id in NEEDS) || NEEDS[id] === null) return null;
  const j = jsonFor(id);
  if (!j) return null;
  return readyOf(id, j, { stairCfg, pose: pose() });
}

/**
 * Put the duck exactly where a move was searched from.
 *
 * The same reset the demo seam does, and for the same reason: a duck moved by
 * writing qpos still carries the policy's own feedback — lastAction is 14 of
 * the 61 observation floats — so a placed duck with a stale action history is
 * not in the state the move was searched from.
 */
function placeDuck(x, y) {
  const f = DUCK.freeQpos;
  data.qpos[f] = x; data.qpos[f + 1] = y; data.qpos[f + 2] = 0.12;
  data.qpos[f + 3] = 1; data.qpos[f + 4] = 0; data.qpos[f + 5] = 0; data.qpos[f + 6] = 0;
  for (let i = 0; i < 14; i++) { data.qpos[DUCK.qpos[i]] = HOME[i]; data.ctrl[i] = HOME[i]; }
  for (let i = 0; i < model.nv; i++) data.qvel[i] = 0;
  lastAction = new Array(14).fill(0);
  previous = null;
  skill = null; intent = null; mode = null; lockUntil = 0;
  mj.mj_forward(model, data);
}

async function startSettle(n, approach) {
  const sess = await policyFor(SETTLE_POLICY);
  if (!sess) return;
  skill = null; intent = null; mode = null; manual = null;
  settleState = { until: ticks + n, approach, session: sess };
}

const runner = makeRunner({
  ticks: () => ticks,
  pose,
  speed: () => driveSpeed,
  tol: () => TOL,
  settling: () => settleState !== null,
  isBusy: () => busy(),
  drive: c => { autoCmd = c; },
  place: placeDuck,
  reset,
  setStairs: ({ rise, count, run }) => {
    if (rise !== undefined) riseEl.value = Math.round(rise * 1000);
    if (count !== undefined) countEl.value = count;
    if (run !== undefined) runEl.value = Math.round(run * 1000);
    readStairs();
  },
  startSettle,
  fire: async id => {
    const item = ALL_MOVES.find(i => i.id === id);
    if (!item) return false;
    await fire(item);
    return true;
  },
  onChange: () => paintSequence(),
});

// Advanced from the physics loop, so a sequence measures its waits in control
// ticks rather than in wall-clock milliseconds. They are not the same thing:
// screenshotting or a slow frame stretches the wall clock and leaves the
// simulation exactly where it was.
async function sequenceTick() {
  if (runner.state === 'running') await runner.tick();
  recordDrive();
  // Readiness moves as the duck does, but thirteen DOM writes a frame is waste.
  if ((ticks & 15) === 0) paintReady();
}

// ── asking for a move, rather than just firing one ────────────────────────
/**
 * What a button press means now.
 *
 * `fire()` is still the primitive — it plays a track at a duck and asks no
 * questions. `request()` is the thing a person actually meant: work out whether
 * the move can run here, and if it cannot but could, set the room up and walk
 * the duck to the mark first.
 */
async function request(item) {
  if (runner.state === 'running') { runner.stop(); autoCmd = null; return; }
  const r = readyFor(item.id);
  if (!r || r.ok) {
    if (recording) record({ kind: 'move', id: item.id });
    return fire(item);
  }
  const steps = stageFor(item.id, r, { teleport: seqModeEl && seqModeEl.value === 'place' });
  // Changing the staircase out from under someone is a real side effect, so it
  // is said out loud rather than done quietly.
  const setsStairs = steps.some(st => st.kind === 'stairs');
  say(setsStairs
    ? `${item.label}: ${r.reason}. Setting the steps to 10 mm — the only height this move has ever cleared.`
    : `${item.label}: ${r.reason}. Walking there first.`);
  if (recording) for (const st of steps) record(st);
  await runner.play(steps);
}

function say(text) {
  statusEl.textContent = text;
  clearTimeout(say.t);
  say.t = setTimeout(() => { if (statusEl.textContent === text) statusEl.textContent = ''; }, 5000);
}

// ── recording ─────────────────────────────────────────────────────────────
let recording = false;
let program = [];
// Driving is recorded as SEGMENTS, not as key events: a sequence that replayed
// keystrokes would be replaying the person, and what matters is the command the
// policy saw and for how many control ticks it saw it.
let driveRun = null;
function record(step) { program.push(step); paintSequence(); }
function recordDrive() {
  if (!recording || runner.state === 'running') { driveRun = null; return; }
  const vx = +cmdState.vx.toFixed(3), vyaw = +cmdState.vyaw.toFixed(3);
  if (driveRun && driveRun.vx === vx && driveRun.vyaw === vyaw) return;
  if (driveRun && (vx !== driveRun.vx || vyaw !== driveRun.vyaw)) {
    const n = ticks - driveRun.t0;
    if (n > 2 && (driveRun.vx || driveRun.vyaw)) record({ kind: 'drive', vx: driveRun.vx, vyaw: driveRun.vyaw, ticks: n });
  }
  driveRun = { vx, vyaw, t0: ticks };
}

// ── the sequence panel ────────────────────────────────────────────────────
let seqModeEl = null;
function buildSequence() {
  seqModeEl = document.getElementById('seqMode');
  const rec = document.getElementById('seqRec');
  const play = document.getElementById('seqPlay');
  const clear = document.getElementById('seqClear');
  const copy = document.getElementById('seqCopy');

  rec.addEventListener('click', () => {
    recording = !recording;
    driveRun = recording ? { vx: 0, vyaw: 0, t0: ticks } : null;
    paintSequence();
  });
  play.addEventListener('click', async () => {
    if (runner.state === 'running') { runner.stop(); autoCmd = null; paintSequence(); return; }
    if (!program.length) return;
    recording = false;
    await runner.play(program);
  });
  clear.addEventListener('click', () => { program = []; runner.stop(); autoCmd = null; paintSequence(); });
  copy.addEventListener('click', async () => {
    const text = JSON.stringify(program, null, 2);
    try { await navigator.clipboard.writeText(text); say('Sequence copied.'); }
    catch { console.log(text); say('Sequence logged to the console.'); }
  });
  paintSequence();
}

function paintSequence() {
  const list = document.getElementById('seqList');
  const state = document.getElementById('seqState');
  const rec = document.getElementById('seqRec');
  const play = document.getElementById('seqPlay');
  if (!list) return;
  const running = runner.state === 'running';
  rec.textContent = recording ? 'stop' : 'record';
  rec.classList.toggle('on', recording);
  play.textContent = running ? 'halt' : 'play';
  play.disabled = !program.length && !running;
  state.textContent = running ? (runner.note || 'running')
    : recording ? `recording, ${program.length}`
    : program.length ? `${program.length} steps` : 'empty';
  state.classList.toggle('on', running || recording);

  // Rebuild only when the shape changed; the note updates every frame.
  if (list.children.length !== program.length) {
    list.textContent = '';
    program.forEach((st, i) => {
      const li = document.createElement('li');
      const txt = document.createElement('span');
      txt.textContent = describeStep(st);
      const del = document.createElement('button');
      del.className = 'mini'; del.textContent = '\u00d7';
      del.setAttribute('aria-label', 'remove step ' + (i + 1) + ', ' + describeStep(st));
      del.addEventListener('click', () => { program.splice(i, 1); list.textContent = ''; paintSequence(); });
      li.append(txt, del);
      list.appendChild(li);
    });
  }
  [...list.children].forEach((li, i) => li.classList.toggle('at', running && i === runner.pc));
}

function buildPad() {
  const stateEl = document.getElementById('padState');
  const listEl = document.getElementById('padList');
  const all = ALL_MOVES;

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
        return request(all.find(i => i.id === (down ? 'stand' : 'sit')));
      }
      const item = all.find(i => i.id === id);
      if (item) request(item);
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
    btn.setAttribute('aria-label', 'remap ' + a.label);
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
      : { vx: -y > 0 ? -y * driveSpeed.fwd : -y * -driveSpeed.back,
          vyaw: -x * driveSpeed.ang };
    window.__stick = stickCmd;
  });
  const all = ALL_MOVES;
  for (const el of document.querySelectorAll('.pad-btn')) {
    // Read the intent at PRESS time, not at wiring time, so re-assigning a
    // button takes effect without rebuilding the handler.
    el.addEventListener('pointerdown', e => {
      e.preventDefault();
      const item = all.find(i => i.id === el.dataset.intent);
      if (item) request(item);
    });
  }
}

function buildKeys() {
  const all = ALL_MOVES;
  for (const item of all) {
    const b = document.createElement('button');
    b.innerHTML = `<kbd>${item.key.toUpperCase()}</kbd><span>${item.label}</span>`;
    b.title = titleFor(item.id);
    b.addEventListener('click', () => request(item));
    keyRow.appendChild(b);
    keyButtons.set(item.id, b);
  }
  const byKey = new Map(all.map(i => [i.key, i]));
  addEventListener('keydown', e => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (typingIn(e)) return;
    const item = byKey.get(e.key.toLowerCase());
    if (item) { e.preventDefault(); request(item); }
  });
}
document.getElementById('copyPose').addEventListener('click', async () => {
  const names = C.jointNames.filter(n => n !== 'mouth');
  const pose = Object.fromEntries(names.map((n, k) => [n, +(manual ? manual[k] : data.ctrl[k]).toFixed(4)]));
  const text = JSON.stringify(pose, null, 2);
  // Was writing into #ctlNote, which is the STAIRS caption in another card —
  // it overwrote the stair finding and never restored it.
  try { await navigator.clipboard.writeText(text); say('Pose copied.'); }
  catch { console.log(text); say('Pose logged to the console.'); }
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
  driveSpeed = speeds(name);
  variant = name;
  skillSessions.clear();
  skill = null; intent = null; mode = null; lockUntil = 0; manual = null;
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
    try {
      backRoll = await (await fetch('./intent-backroll.json')).json();
      tracks.back_roll = backRoll;
    } catch { /* likewise */ }
    for (const [id, file] of [['lever_up', 'intent-lever.json'], ['wall_flip', 'intent-wallflip.json'], ['riser_up', 'intent-riser.json'], ['climb', 'intent-climb.json']]) {
      try { tracks[id] = await (await fetch('./' + file)).json(); } catch { /* optional */ }
    }
    // AR, where the browser has it. The button stays hidden otherwise rather
    // than offering something that will fail — Safari has no WebXR at all,
    // which is why this project's iOS path is native.
    const support = await xrSupport();
    const xrBtn = document.getElementById('xr');
    const xrMode = support.ar ? 'immersive-ar' : support.vr ? 'immersive-vr' : null;
    if (xrMode) {
      xrBtn.hidden = false;
      xrBtn.textContent = support.ar ? 'AR' : 'VR';
      xrBtn.title = support.ar ? 'View in AR' : 'View in VR';
      xrBtn.setAttribute('aria-label', xrBtn.title);
      xrBtn.addEventListener('click', async () => {
        if (xrSession) { await xrSession.end(); return; }
        try {
          xrSession = await startXR({
            gl: renderer.gl, mode: xrMode,
            step: () => { /* the 50 Hz loop keeps running below */ },
            onFrame: ({ view, proj, origin }) => {
              renderer.render(data, {
                bg: [0, 0, 0], grid: [0, 0, 0], root: DUCK.freeQpos,
                model, geomTypes: GEOM_TYPES, xr: { view, proj, origin },
              });
            },
            onEnd: () => { xrSession = null; xrBtn.textContent = support.ar ? 'AR' : 'VR'; },
          });
          xrBtn.textContent = 'exit';
        } catch (err) {
          statusEl.textContent = 'AR could not start: ' + (err.message || err);
          setTimeout(() => { statusEl.textContent = ''; }, 4000);
        }
      });
    }

    // A demo-only seam, behind ?demo=1, for recording GIFs: some moves need
    // the duck at a specific distance from a wall or a step, and walking it
    // there approximately is not the same thing. Not present otherwise.
    if (new URLSearchParams(location.search).has('demo')) {
      window.__demo = {
        place(x, y = 0) {
          data.qpos[DUCK.freeQpos] = x;
          data.qpos[DUCK.freeQpos + 1] = y;
          data.qpos[DUCK.freeQpos + 2] = 0.12;
          data.qpos[DUCK.freeQpos + 3] = 1;
          for (let i = 4; i < 7; i++) data.qpos[DUCK.freeQpos + i] = 0;
          for (let i = 0; i < 14; i++) { data.qpos[DUCK.qpos[i]] = HOME[i]; data.ctrl[i] = HOME[i]; }
          for (let i = 0; i < model.nv; i++) data.qvel[i] = 0;
          // Reset the policy's own feedback too. lastAction is 14 of the 61
          // observation floats, so a duck placed with a stale action history is
          // not in the state the move was searched from — which is exactly why
          // the wall flip reproduced headlessly and not from the page.
          lastAction = new Array(14).fill(0);
          previous = null;
          skill = null; intent = null; mode = null; lockUntil = 0;
          ticks = 0;
          mj.mj_forward(model, data);
        },
        x: () => data.qpos[DUCK.freeQpos],
        // Capture the observation and the commanded targets for N control
        // ticks, so the browser's state can be diffed against the search's.
        trace: [],
        record(n) { window.__demo.trace = []; window.__demo.recording = n; },
        /**
         * Settle exactly as the search does: n control ticks under the STAND
         * policy holding the move's own approach command.
         *
         * Measured: settling any other way leaves the pose right to five
         * decimals but the joint velocities out by up to 0.147, and those are
         * 14 of the 61 observation floats. A move searched from one velocity
         * state does not fire from another.
         */
        async settle(n, approach = 0) {
          let sess = skillSessions.get('BEST_alpha_stand.onnx');
          if (!sess) {
            sess = await ort.InferenceSession.create('./BEST_alpha_stand.onnx');
            skillSessions.set('BEST_alpha_stand.onnx', sess);
          }
          skill = null; intent = null; mode = null;
          for (let i = 0; i < n; i++) {
            const f = DUCK.freeQpos;
            const q = [data.qpos[f+3], data.qpos[f+4], data.qpos[f+5], data.qpos[f+6]];
            const jp = [], jv = [];
            for (let k = 0; k < 14; k++) { jp.push(data.qpos[DUCK.qpos[k]]); jv.push(data.qvel[DUCK.dof[k]]); }
            const obs = buildObs(
              [data.sensordata[GYRO], data.sensordata[GYRO+1], data.sensordata[GYRO+2]],
              projectedGravity(q), jp, jv, lastAction, command({ vx: approach }));
            const out = await sess.run({ [inputName]: new ort.Tensor('float32', obs, [1, 61]) });
            lastAction = Array.from(out[sess.outputNames[0]].data);
            for (let k = 0; k < 14; k++) data.ctrl[k] = HOME[k] + lastAction[k];
            for (let s2 = 0; s2 < 4; s2++) mj.mj_step(model, data);
          }
          ticks = 0;
        },
        dump() { return window.__demo.trace; },
        // The sequence machinery, for the headless checks. Behind ?demo=1 with
        // everything else that reaches inside.
        seq: {
          pose, readyFor, runner,
          program: () => program,
          setProgram(p) { program = p.slice(); paintSequence(); },
          stairs: () => ({ ...stairCfg }),
          settling: () => settleState !== null,
        },
      };
    }

    buildSlots();
    setZoom(1);
    buildPad();
    buildTouch();
    buildKeys();
    buildServos();
    buildSequence();
    readStairs();
    reset();
    statusEl.textContent = '';
    document.body.classList.add('ready');

    // NB `stepping`, not `busy` — busy() is the input-lock predicate and a local
    // of that name shadowed it here.
    let acc = 0, last = performance.now(), stepping = false;
    const stepMs = 1000 / C.tickHz;
    async function loop(now) {
      // Everything inside is guarded. loop() is async, so a rejected await
      // anywhere in it used to reject the whole promise and never reach the
      // requestAnimationFrame at the bottom — one bad ONNX run stopped the
      // physics AND the rendering, permanently and silently, somewhere the
      // boot try/catch could not see it.
      try {
        acc += Math.min(now - last, 100); last = now;
        padCmd = pad ? pad.poll(now) : null;
        readControls();
        if (!stepping && !switching) {
          stepping = true;
          try {
            let n = 0;
            while (acc >= stepMs && n < 4) { acc -= stepMs; await tick(); n++; }
          } finally { stepping = false; }
        }
        await sequenceTick();
        // The input lock expires on the clock, so nothing else can notice.
        repaintIfIdle();
        if (!switching && !xrSession) draw();
      } catch (err) {
        console.error('frame failed', err);
        statusEl.textContent = 'a frame failed: ' + (err && err.message ? err.message : err);
      }
      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
  } catch (err) {
    statusEl.textContent = 'The simulator could not start: ' + (err && err.message ? err.message : err);
    console.error(err);
  }
})();
