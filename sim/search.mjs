// Search the step-up intent for the tallest step the duck can get onto.
import load from 'mujoco';
import fs from 'node:fs';
import * as ort from 'onnxruntime-node';
import { makeLoop } from '../site/duckloop.mjs';
import { findStairJoints, layoutStairs, clearStairs } from '../site/stairs.js';
import { buildTrack, poseAt, DEFAULTS, BOUNDS } from './intent.mjs';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/scene.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const model = mj.MjModel.mj_loadBinary('/scene.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model);
const ADDR = findStairJoints(model);
const DT = 1 / C.tickHz, SUB = 4;
const session = await ort.InferenceSession.create('./alpha_walking.onnx');
const IN = session.inputNames[0], OUT = session.outputNames[0];
let GYRO = 0;
for (let i = 0; i < model.nsensor; i++) if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;

/** One control tick of the policy. Returns its 14 raw actions. */
async function policyStep(cmd, lastAction) {
  const f = D.freeQpos;
  const q = [data.qpos[f+3], data.qpos[f+4], data.qpos[f+5], data.qpos[f+6]];
  const jp = [], jv = [];
  for (let k = 0; k < 14; k++) { jp.push(data.qpos[D.qpos[k]]); jv.push(data.qvel[D.dof[k]]); }
  const obs = buildObs([data.sensordata[GYRO], data.sensordata[GYRO+1], data.sensordata[GYRO+2]],
                       projectedGravity(q), jp, jv, lastAction, cmd);
  const out = await session.run({ [IN]: new ort.Tensor('float32', obs, [1, 61]) });
  return Array.from(out[OUT].data);
}

/** One step, tall enough to matter, right in front of the duck. */
function setStep(h) {
  if (h <= 0) { clearStairs(data, ADDR); return; }
  layoutStairs(data, ADDR, { count: 1, rise: h, run: 0.30, start: 0.10 });
}

function reset(h) {
  mj.mj_resetData(model, data);
  data.qpos[D.freeQpos + 2] = 0.12;
  data.qpos[D.freeQpos + 3] = 1;
  for (let i = 0; i < 14; i++) { data.qpos[D.qpos[i]] = HOME[i]; data.ctrl[i] = HOME[i]; }
  setStep(h);
  mj.mj_forward(model, data);
}

const upright = () => projectedGravity([
  data.qpos[D.freeQpos+3], data.qpos[D.freeQpos+4],
  data.qpos[D.freeQpos+5], data.qpos[D.freeQpos+6]])[2] < -0.55;

/**
 * Run the move against a step of height h.
 * Success = ends upright, with both the body and the lead foot up on the tread.
 */
/**
 * Run the move against a step of height h.
 *
 * The scripted track is an OFFSET on the policy's output, not a replacement.
 * Replacing it does not work: with kp 0.55 the servos are far too soft to hold
 * a pose, so an open-loop sequence collapses the duck even on a flat floor —
 * measured, it fell every time at every step height including zero. Balance is
 * the policy's job; the script's job is to reach the head out and lean on it.
 */
async function evaluate(p, h) {
  reset(h);
  const cmd = command({ vx: p.approach });
  let la = new Array(14).fill(0);
  // approach and settle under the policy alone
  for (let t = 0; t < 60; t++) {
    setStep(h);
    la = await policyStep(cmd, la);
    for (let i = 0; i < 14; i++) data.ctrl[i] = HOME[i] + la[i];
    for (let s = 0; s < SUB; s++) mj.mj_step(model, data);
  }
  const startX = data.qpos[D.freeQpos];
  const track = buildTrack(p, HOME);
  const total = track[track.length - 1].t + 0.6;
  let fell = false, peak = -9;
  for (let t = 0; t * DT < total; t++) {
    setStep(h);
    la = await policyStep(cmd, la);
    const offset = poseAt(track, t * DT, HOME);
    for (let i = 0; i < 14; i++) {
      const v = HOME[i] + la[i] + (offset[i] - HOME[i]) * p.blend;
      data.ctrl[i] = Math.min(Math.max(v, LO[i]), HI[i]);
    }
    for (let s = 0; s < SUB; s++) mj.mj_step(model, data);
    if (!upright()) fell = true;
    peak = Math.max(peak, data.qpos[D.freeQpos + 2]);
  }
  const z = data.qpos[D.freeQpos + 2], x = data.qpos[D.freeQpos];
  const up = z - h;
  const onTop = !fell && upright() && x > 0.115 && up > 0.085;
  return { onTop, z, x, up, peak, fell, gained: x - startX };
}

function randomIn([lo, hi]) { return lo + Math.random() * (hi - lo); }
function randomParams(lead) {
  const p = { lead };
  for (const k of Object.keys(BOUNDS)) p[k] = randomIn(BOUNDS[k]);
  return p;
}
function jitter(p, scale) {
  const q = { ...p };
  for (const k of Object.keys(BOUNDS)) {
    const [lo, hi] = BOUNDS[k];
    q[k] = Math.min(hi, Math.max(lo, p[k] + (Math.random() * 2 - 1) * (hi - lo) * scale));
  }
  return q;
}

/**
 * How tall a step can this parameter set manage?
 *
 * A ladder that stops at the first failure, not a bisection. Each evaluation
 * costs about 215 policy inferences, so the number of evaluations per candidate
 * is what decides whether a search finishes at all.
 */
const LADDER = [0.006, 0.010, 0.014, 0.018, 0.022, 0.026, 0.030, 0.035];
async function maxHeight(p) {
  let best = 0;
  for (const h of LADDER) {
    if (!(await evaluate(p, h)).onTop) break;
    best = h;
  }
  return best;
}

const mm = v => (v * 1000).toFixed(0);

// ── baseline ──────────────────────────────────────────────────────────────
const baseL = await maxHeight({ ...DEFAULTS, lead: 0 });
const baseR = await maxHeight({ ...DEFAULTS, lead: 1 });
console.log(`BASELINE hand-tuned: lead-left ${mm(baseL)} mm, lead-right ${mm(baseR)} mm`);

// ── search: random restarts, each hill-climbed ────────────────────────────
const BUDGET = +(process.env.BUDGET || 120);
let best = baseL >= baseR ? { ...DEFAULTS, lead: 0 } : { ...DEFAULTS, lead: 1 };
let bestH = Math.max(baseL, baseR), evals = 0;
const t0 = Date.now();

while (evals < BUDGET) {
  let cand = randomParams(evals % 2 ? 1 : 0);
  let h = await maxHeight(cand); evals++;
  // hill-climb this restart while it keeps paying
  for (let i = 0; i < 8 && evals < BUDGET; i++) {
    const q = jitter(cand, i < 4 ? 0.20 : 0.08);
    const qh = await maxHeight(q); evals++;
    if (qh > h) { cand = q; h = qh; i = 0; }
  }
  if (h > bestH) {
    bestH = h; best = cand;
    console.log(`  new best ${mm(h)} mm  (${evals} evaluations, ${((Date.now()-t0)/1000).toFixed(0)}s)`);
    fs.writeFileSync('intent-stepup.json', JSON.stringify({ maxHeightMm: +mm(bestH), params: best }, null, 2));
  }
}
console.log(`SEARCH best ${mm(bestH)} mm after ${evals} evaluations in ${((Date.now()-t0)/1000).toFixed(0)}s`);
fs.writeFileSync('intent-stepup.json', JSON.stringify({ maxHeightMm: +mm(bestH), params: best }, null, 2));
console.log('WROTE intent-stepup.json');
