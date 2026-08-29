// ac_refute.mjs — independent chaos check of the "New plant stands 5/5" claim.
// Same protocol as ac_check2.mjs (drop from z=0.125, ±2mm trunk, ±0.02 rad
// joints, 200 ticks BEST_alpha_stand on ac_scene2.mjb + full ac_plant), but:
//   - 10 FRESH perturbation seeds never used by the integrator
//   - 3 different plant (delay/friction) xorshift seeds per perturbation batch
// Also prints actuator forcerange and a dof_frictionloss sample to prove the
// scene deltas and the friction budget are actually live.
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { declaredDefaultPose } from './onnx_meta.mjs';
import { makeLoop } from '../site/duckloop.mjs';
import { clearStairs, findStairJoints } from '../site/stairs.js';
import { makePlant } from './ac_plant.mjs';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);

const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('ac_scene2.mjb')));
const model = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model), ADDR = findStairJoints(model);

// prove scene deltas landed
console.log('iterations', model.opt.iterations, 'ls_iterations', model.opt.ls_iterations);
console.log('forcerange act0:', model.actuator_forcerange[0], model.actuator_forcerange[1]);

let GYRO = -1;
for (let i = 0; i < model.nsensor; i++)
  if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;

const session = await ort.InferenceSession.create('./BEST_alpha_stand.onnx');
const reference = declaredDefaultPose('./BEST_alpha_stand.onnx', HOME) ?? HOME;

function lcg(seed) { let s = seed >>> 0;
  return () => ((s = (1664525 * s + 1013904223) >>> 0) / 2 ** 32) * 2 - 1; }

const quat = () => [data.qpos[D.freeQpos + 3], data.qpos[D.freeQpos + 4],
                    data.qpos[D.freeQpos + 5], data.qpos[D.freeQpos + 6]];

async function run(pseed, plantSeed) {
  const plant = makePlant(mj, model, data, D, plantSeed);
  const rand = lcg(pseed);
  mj.mj_resetData(model, data);
  if (ADDR) clearStairs(data, ADDR);
  data.qpos[D.freeQpos] = 0.002 * rand();
  data.qpos[D.freeQpos + 1] = 0.002 * rand();
  data.qpos[D.freeQpos + 2] = 0.125;
  data.qpos[D.freeQpos + 3] = 1;
  for (let k = 0; k < 14; k++) {
    data.qpos[D.qpos[k]] = HOME[k] + 0.02 * rand();
    data.ctrl[k] = HOME[k];
  }
  mj.mj_forward(model, data);
  plant.resetDelay(HOME);
  let lastAction = new Array(14).fill(0);
  let prevJv = new Array(14).fill(0);
  const cmd = command({});
  let fricSample = 0;
  for (let t = 0; t < 200; t++) {
    const jp = [], jv = [];
    for (let k = 0; k < 14; k++) { jp.push(data.qpos[D.qpos[k]]); jv.push(data.qvel[D.dof[k]]); }
    const obs = buildObs(
      [data.sensordata[GYRO], data.sensordata[GYRO + 1], data.sensordata[GYRO + 2]],
      projectedGravity(quat()), jp, prevJv, lastAction, cmd, reference);
    prevJv = jv;
    const out = await session.run({ obs: new ort.Tensor('float32', obs, [1, 61]) });
    lastAction = Array.from(out.actions.data);
    const target = new Array(14);
    for (let k = 0; k < 14; k++)
      target[k] = Math.min(Math.max(reference[k] + lastAction[k], LO[k]), HI[k]);
    plant.stepTick(target);
    if (t === 100) fricSample = model.dof_frictionloss[D.dof[0]];
  }
  const g = projectedGravity(quat());
  const z = data.qpos[D.freeQpos + 2];
  const ok = z > 0.10 && z < 0.13 && g[2] < -0.9;
  console.log(`pseed=${pseed} plantSeed=0x${plantSeed.toString(16)} z=${(z*1000).toFixed(1)}mm grav_z=${g[2].toFixed(3)} fricloss[0]@t100=${fricSample.toFixed(4)} -> ${ok?'STANDING':'FAIL'}`);
  return ok;
}

const pseeds = [104729, 1299709, 15485863, 32452843, 49979687, 67867967, 86028121, 15487469, 275604541, 982451653];
const plantSeeds = [0xdeadbeef, 0x12345678, 0xcafef00d];
let pass = 0, total = 0;
for (const ps of plantSeeds)
  for (const s of pseeds.slice(0, ps === 0xdeadbeef ? 10 : 5)) {
    total++;
    if (await run(s, ps)) pass++;
  }
console.log(`\n${pass}/${total} independent perturbed drops settled standing.`);
