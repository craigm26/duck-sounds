// ac_headspin.mjs — per-tick trace of the community headspin.onnx: projected
// gravity z, trunk z, and WHICH BODIES touch the floor, across several
// perturbed drops (z in 0.12–0.13, Pollen's reset_base range).
//
// The author's render shows a vertical headstand: head down, trunk HIGH
// (~119 mm), feet up. Belly-up-on-the-trunk-shell instead shows trunk LOW
// (~45–50 mm) with trunk_base geoms in contact. The floor-contact histogram
// over the hold phase is what separates the two.
//
//   SCENE=ac_scene2.mjb PLANT=1 node ac_headspin.mjs    (new plant, default)
//   SCENE=scene.mjb     PLANT=0 node ac_headspin.mjs    (old plant, for contrast)
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { declaredDefaultPose } from './onnx_meta.mjs';
import { makeLoop } from '../site/duckloop.mjs';
import { clearStairs, findStairJoints } from '../site/stairs.js';
import { makePlant } from './ac_plant.mjs';

const SCENE = process.env.SCENE || 'ac_scene2.mjb';
const USE_PLANT = (process.env.PLANT ?? '1') !== '0';
const SEEDS = +(process.env.SEEDS || 6);

const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);

const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync(SCENE)));
const model = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model), ADDR = findStairJoints(model);
let GYRO = 0;
for (let i = 0; i < model.nsensor; i++)
  if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;
let FLOOR = -1;
for (let g = 0; g < model.ngeom; g++)
  if (model.geom(g).name === 'floor') FLOOR = g;
if (FLOOR < 0) throw new Error('no floor geom');

console.log(`scene ${SCENE}  plant-model ${USE_PLANT ? 'ON (BAM friction + 3-6 substep delay + jv lag)' : 'OFF (plain servo, fresh obs)'}  seeds ${SEEDS}`);

const plant = USE_PLANT ? makePlant(mj, model, data, D) : null;

const HEAD_BODIES = new Set(['jaw_soft']);           // all 3 head shells live here
const TRUNK_BODIES = new Set(['trunk_base']);
const FOOT_BODIES = new Set(['ankle_left', 'ankle_right']);

function floorContacts() {
  const bodies = [];
  for (let i = 0; i < data.ncon; i++) {
    const c = data.contact.get(i);
    let other = -1;
    if (c.geom1 === FLOOR) other = c.geom2;
    else if (c.geom2 === FLOOR) other = c.geom1;
    if (other < 0) continue;
    bodies.push(model.body(model.geom_bodyid[other]).name);
  }
  return bodies;
}

function lcg(seed) {
  let s = seed >>> 0;
  return () => ((s = (1664525 * s + 1013904223) >>> 0) / 2 ** 32) * 2 - 1;
}

const HEADSPIN = 'headspin.onnx';
const STAND = 'BEST_alpha_stand.onnx';
const spinSession = await ort.InferenceSession.create('./' + HEADSPIN);
const standSession = await ort.InferenceSession.create('./' + STAND);
const spinRef = declaredDefaultPose('./' + HEADSPIN, HOME) ?? HOME;
const standRef = declaredDefaultPose('./' + STAND, HOME) ?? HOME;

const quat = () => [data.qpos[D.freeQpos + 3], data.qpos[D.freeQpos + 4],
                    data.qpos[D.freeQpos + 5], data.qpos[D.freeQpos + 6]];

async function run(seedIdx) {
  const rand = lcg(seedIdx * 104729 + 13);
  const dropZ = 0.12 + (rand() * 0.5 + 0.5) * 0.01;   // 0.12–0.13
  mj.mj_resetData(model, data);
  if (ADDR) clearStairs(data, ADDR);
  data.qpos[D.freeQpos] = 0.002 * rand();
  data.qpos[D.freeQpos + 1] = 0.002 * rand();
  data.qpos[D.freeQpos + 2] = dropZ;
  data.qpos[D.freeQpos + 3] = 1;
  for (let k = 0; k < 14; k++) {
    data.qpos[D.qpos[k]] = HOME[k] + 0.01 * rand();
    data.ctrl[k] = HOME[k];
  }
  mj.mj_forward(model, data);
  if (plant) plant.resetDelay(HOME);

  let lastAction = new Array(14).fill(0);
  let prevJv = new Array(14).fill(0);
  const cmd = command({});
  const trace = [];
  // 25 settle ticks standing, then 200 ticks (4 s) of headspin — the
  // recorder's exact schedule.
  for (let t = -25; t < 200; t++) {
    const jp = [], jv = [];
    for (let k = 0; k < 14; k++) { jp.push(data.qpos[D.qpos[k]]); jv.push(data.qvel[D.dof[k]]); }
    const reference = t < 0 ? standRef : spinRef;
    const obs = buildObs(
      [data.sensordata[GYRO], data.sensordata[GYRO + 1], data.sensordata[GYRO + 2]],
      projectedGravity(quat()), jp, USE_PLANT ? prevJv : jv, lastAction, cmd, reference);
    prevJv = jv;
    const session = t < 0 ? standSession : spinSession;
    const out = await session.run({ obs: new ort.Tensor('float32', obs, [1, 61]) });
    lastAction = Array.from(out.actions.data);
    const target = new Array(14);
    for (let k = 0; k < 14; k++)
      target[k] = Math.min(Math.max(reference[k] + lastAction[k], LO[k]), HI[k]);
    if (plant) plant.stepTick(target);
    else {
      for (let k = 0; k < 14; k++) data.ctrl[k] = target[k];
      for (let s = 0; s < 4; s++) mj.mj_step(model, data);
    }
    if (t >= 0) {
      const g = projectedGravity(quat());
      trace.push({ t, z: data.qpos[D.freeQpos + 2], g: g[2], contacts: floorContacts() });
    }
  }

  // Per-tick print every 20 ticks; then the hold-phase summary (last 100 ticks).
  console.log(`\n== seed ${seedIdx}  drop z ${(dropZ * 1000).toFixed(1)}mm`);
  for (const r of trace) {
    if (r.t % 20 === 0) {
      const hist = {};
      for (const b of r.contacts) hist[b] = (hist[b] ?? 0) + 1;
      console.log(`  t=${(r.t / 50).toFixed(2)}s  grav_z=${r.g.toFixed(3).padStart(7)}  trunk z=${(r.z * 1000).toFixed(1).padStart(6)}mm  floor: ${Object.entries(hist).map(([b, n]) => `${b}x${n}`).join(' ') || '(none)'}`);
    }
  }
  const hold = trace.slice(100);
  const meanG = hold.reduce((a, r) => a + r.g, 0) / hold.length;
  const meanZ = hold.reduce((a, r) => a + r.z, 0) / hold.length;
  const hist = {};
  for (const r of hold) for (const b of r.contacts) hist[b] = (hist[b] ?? 0) + 1;
  const headTicks = hold.filter(r => r.contacts.some(b => HEAD_BODIES.has(b))).length;
  const trunkTicks = hold.filter(r => r.contacts.some(b => TRUNK_BODIES.has(b))).length;
  const headOnly = hold.filter(r => r.contacts.length > 0 &&
    r.contacts.every(b => HEAD_BODIES.has(b))).length;
  console.log(`  HOLD (2.0-4.0s): mean grav_z ${meanG.toFixed(3)}  mean trunk z ${(meanZ * 1000).toFixed(1)}mm`);
  console.log(`  floor-contact ticks/100: head(jaw_soft) ${headTicks}, trunk_base ${trunkTicks}, head-ONLY ${headOnly}`);
  console.log(`  contact histogram: ${Object.entries(hist).sort((a, b) => b[1] - a[1]).map(([b, n]) => `${b}:${n}`).join(' ')}`);
  return { meanG, meanZ, headTicks, trunkTicks, headOnly };
}

const rows = [];
for (let s = 1; s <= SEEDS; s++) rows.push(await run(s));
console.log('\n== SUMMARY over seeds ==');
rows.forEach((r, i) => console.log(
  `seed ${i + 1}: hold grav_z ${r.meanG.toFixed(3)}  trunk z ${(r.meanZ * 1000).toFixed(1)}mm  head-contact ticks ${r.headTicks}/100  head-only ${r.headOnly}/100  trunk-contact ${r.trunkTicks}/100`));
