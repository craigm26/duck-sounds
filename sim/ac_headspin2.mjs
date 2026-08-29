// ac_headspin2.mjs — hypothesis probes for the community headspin.onnx on the
// training-parameter plant (ac_scene2.mjb + ac_plant loop):
//
//   CMD=zero   command block all zeros (what the recorder feeds today)
//   CMD=clock  [cos(2*pi*t/4), sin(2*pi*t/4), 0] — the GroundPickPhaseCommand
//              layout the whole cyclic-gesture family (ground_pick,
//              roller_crouch, spin) trains with
//   CMD=vx1    vx = 1 flag (the sitstand trigger convention)
//   CMD=yaw    vyaw = 3 rad/s (a spin-rate command)
//
//   START=stand     settle standing, then run the policy (recorder schedule)
//   START=inverted  START the duck upside down (quat 180 deg about y, trunk at
//                   z=0.119, policy's own neutral joints) — asks whether a
//                   vertical headstand is even an equilibrium of this policy
//                   in this plant, independent of the transition into it.
//
// Prints the hold-phase (last 2 s) posture + floor-contact histogram per seed.
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { declaredDefaultPose } from './onnx_meta.mjs';
import { makeLoop } from '../site/duckloop.mjs';
import { clearStairs, findStairJoints } from '../site/stairs.js';
import { makePlant } from './ac_plant.mjs';

const CMD = process.env.CMD || 'zero';
const START = process.env.START || 'stand';
const SEEDS = +(process.env.SEEDS || 3);
const SCENE = process.env.SCENE || 'ac_scene2.mjb';
const USE_PLANT = (process.env.PLANT ?? '1') !== '0';

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
for (let g = 0; g < model.ngeom; g++) if (model.geom(g).name === 'floor') FLOOR = g;

const plant = USE_PLANT ? makePlant(mj, model, data, D) : null;
const spin = await ort.InferenceSession.create('./headspin.onnx');
const stand = await ort.InferenceSession.create('./BEST_alpha_stand.onnx');
const spinRef = declaredDefaultPose('./headspin.onnx', HOME) ?? HOME;
const standRef = declaredDefaultPose('./BEST_alpha_stand.onnx', HOME) ?? HOME;

const quat = () => [data.qpos[D.freeQpos + 3], data.qpos[D.freeQpos + 4],
                    data.qpos[D.freeQpos + 5], data.qpos[D.freeQpos + 6]];
function floorBodies() {
  const out = [];
  for (let i = 0; i < data.ncon; i++) {
    const c = data.contact.get(i);
    const other = c.geom1 === FLOOR ? c.geom2 : c.geom2 === FLOOR ? c.geom1 : -1;
    if (other >= 0) out.push(model.body(model.geom_bodyid[other]).name);
  }
  return out;
}
function lcg(seed) {
  let s = seed >>> 0;
  return () => ((s = (1664525 * s + 1013904223) >>> 0) / 2 ** 32) * 2 - 1;
}
const cmdFor = t => {
  if (CMD === 'clock') return command({ vx: Math.cos(2 * Math.PI * t / 4), vy: Math.sin(2 * Math.PI * t / 4) });
  if (CMD === 'vx1') return command({ vx: 1 });
  if (CMD === 'yaw') return command({ vyaw: 3 });
  return command({});
};

async function run(seedIdx) {
  const rand = lcg(seedIdx * 104729 + 13);
  mj.mj_resetData(model, data);
  if (ADDR) clearStairs(data, ADDR);
  const settle = START === 'stand' ? 25 : 0;
  if (START === 'stand') {
    data.qpos[D.freeQpos + 2] = 0.12 + (rand() * 0.5 + 0.5) * 0.01;
    data.qpos[D.freeQpos + 3] = 1;
    for (let k = 0; k < 14; k++) data.qpos[D.qpos[k]] = HOME[k] + 0.01 * rand();
  } else if (START === 'nose90') {
    // nose-down 90 deg about +y: a mount intermediate (head under, tail rising)
    const h = Math.SQRT1_2;
    data.qpos[D.freeQpos + 2] = 0.085 + 0.002 * rand();
    data.qpos[D.freeQpos + 3] = h; data.qpos[D.freeQpos + 4] = 0;
    data.qpos[D.freeQpos + 5] = h; data.qpos[D.freeQpos + 6] = 0;
    for (let k = 0; k < 14; k++) data.qpos[D.qpos[k]] = spinRef[k] + 0.01 * rand();
  } else if (START === 'pitch150') {
    // 150 deg about +y: just short of the headstand, inside or outside its basin?
    const a = 150 * Math.PI / 180;
    data.qpos[D.freeQpos + 2] = 0.105 + 0.002 * rand();
    data.qpos[D.freeQpos + 3] = Math.cos(a / 2); data.qpos[D.freeQpos + 4] = 0;
    data.qpos[D.freeQpos + 5] = Math.sin(a / 2); data.qpos[D.freeQpos + 6] = 0;
    for (let k = 0; k < 14; k++) data.qpos[D.qpos[k]] = spinRef[k] + 0.01 * rand();
  } else {
    // upside down: 180 deg about y, trunk high, policy's own neutral joints
    data.qpos[D.freeQpos + 2] = 0.119 + 0.002 * rand();
    data.qpos[D.freeQpos + 3] = 0; data.qpos[D.freeQpos + 4] = 0;
    data.qpos[D.freeQpos + 5] = 1; data.qpos[D.freeQpos + 6] = 0;
    for (let k = 0; k < 14; k++) data.qpos[D.qpos[k]] = spinRef[k] + 0.01 * rand();
  }
  for (let k = 0; k < 14; k++) data.ctrl[k] = data.qpos[D.qpos[k]];
  mj.mj_forward(model, data);
  if (plant) plant.resetDelay(START === 'stand' ? HOME : spinRef);

  let lastAction = new Array(14).fill(0);
  let prevJv = new Array(14).fill(0);
  const trace = [];
  for (let t = -settle; t < 200; t++) {
    const jp = [], jv = [];
    for (let k = 0; k < 14; k++) { jp.push(data.qpos[D.qpos[k]]); jv.push(data.qvel[D.dof[k]]); }
    const reference = t < 0 ? standRef : spinRef;
    const cmd = t < 0 ? command({}) : cmdFor(Math.max(0, t) / 50);
    const obs = buildObs(
      [data.sensordata[GYRO], data.sensordata[GYRO + 1], data.sensordata[GYRO + 2]],
      projectedGravity(quat()), jp, USE_PLANT ? prevJv : jv, lastAction, cmd, reference);
    prevJv = jv;
    const session = t < 0 ? stand : spin;
    const out = await session.run({ obs: new ort.Tensor('float32', obs, [1, 61]) });
    lastAction = Array.from(out.actions.data);
    const target = new Array(14);
    for (let k = 0; k < 14; k++)
      target[k] = Math.min(Math.max(reference[k] + lastAction[k], LO[k]), HI[k]);
    if (plant) plant.stepTick(target);
    else {
      for (let k = 0; k < 14; k++) data.ctrl[k] = target[k];
      for (let sb = 0; sb < 4; sb++) mj.mj_step(model, data);
    }
    if (t >= 0) trace.push({ t, z: data.qpos[D.freeQpos + 2], g: projectedGravity(quat())[2], fb: floorBodies() });
  }
  const hold = trace.slice(100);
  const meanG = hold.reduce((a, r) => a + r.g, 0) / hold.length;
  const meanZ = hold.reduce((a, r) => a + r.z, 0) / hold.length;
  const hist = {};
  for (const r of hold) for (const b of r.fb) if (!/block|ball|cone/.test(b)) hist[b] = (hist[b] ?? 0) + 1;
  const early = trace.slice(0, 50);
  console.log(`seed ${seedIdx}: first-1s grav_z ${early.map(r => r.g.toFixed(2)).filter((_, i) => i % 10 === 0).join('->')}  | HOLD mean grav_z ${meanG.toFixed(3)} trunk z ${(meanZ * 1000).toFixed(1)}mm  robot-floor: ${Object.entries(hist).sort((a, b) => b[1] - a[1]).map(([b, n]) => `${b}:${n}`).join(' ') || '(none)'}`);
  return { meanG, meanZ };
}

console.log(`CMD=${CMD} START=${START} SCENE=${SCENE} PLANT=${USE_PLANT ? 'on' : 'off'}`);
for (let s = 1; s <= SEEDS; s++) await run(s);
