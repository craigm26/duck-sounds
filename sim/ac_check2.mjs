// ac_check2.mjs — validate ac_scene2.mjb + the ac_plant loop model before any
// recording happens on it. Same 5-drop standing test as ac_check.mjs, but on
// the training-parameter plant:
//   - ac_scene2.mjb (solver 10/20, feet priority 1, training floor friction,
//     forcerange ±0.6405, stiff friction solref, stairs/walls/props present)
//   - BAM m6 friction budget written per substep (ac_plant.mjs)
//   - actuation delay Uniform{3..6} physics steps on the position target
//   - joint_vel observation delayed exactly 1 control tick (the Dynamixel
//     firmware latency model every task trained with; base_ang_vel/gravity
//     keep lag 0, which is inside their trained Uniform{0,1}).
//
// PASS: 5/5 drops settle standing (trunk z in 0.10–0.13, grav z < −0.9 at 4 s).
//
// Run:  node ac_check2.mjs        (from this directory)
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { declaredDefaultPose } from './onnx_meta.mjs';
import { makeLoop } from '../site/duckloop.mjs';
import { clearStairs, findStairJoints } from '../site/stairs.js';
import { makePlant, DELAY_MIN, DELAY_MAX } from './ac_plant.mjs';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);

const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('ac_scene2.mjb')));
const model = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model), ADDR = findStairJoints(model);

console.log('nq', model.nq, 'nv', model.nv, 'nu', model.nu, 'ngeom', model.ngeom,
  'timestep', model.opt.timestep, 'iterations', model.opt.iterations,
  'ls_iterations', model.opt.ls_iterations,
  'delay', `${DELAY_MIN}-${DELAY_MAX} physics steps`);

let GYRO = -1;
for (let i = 0; i < model.nsensor; i++)
  if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;
if (GYRO < 0) throw new Error('imu_ang_vel sensor missing');

const STAND = 'BEST_alpha_stand.onnx';
const session = await ort.InferenceSession.create('./' + STAND);
const reference = declaredDefaultPose('./' + STAND, HOME) ?? HOME;

function lcg(seed) {
  let s = seed >>> 0;
  return () => ((s = (1664525 * s + 1013904223) >>> 0) / 2 ** 32) * 2 - 1;
}

const plant = makePlant(mj, model, data, D);

function resetDuck(rand) {
  mj.mj_resetData(model, data);
  if (ADDR) clearStairs(data, ADDR);
  data.qpos[D.freeQpos] = rand ? 0.002 * rand() : 0;
  data.qpos[D.freeQpos + 1] = rand ? 0.002 * rand() : 0;
  data.qpos[D.freeQpos + 2] = 0.125;
  data.qpos[D.freeQpos + 3] = 1;
  for (let k = 0; k < 14; k++) {
    data.qpos[D.qpos[k]] = HOME[k] + (rand ? 0.02 * rand() : 0);
    data.ctrl[k] = HOME[k];
  }
  mj.mj_forward(model, data);
  plant.resetDelay(HOME);
}

const quat = () => [data.qpos[D.freeQpos + 3], data.qpos[D.freeQpos + 4],
                    data.qpos[D.freeQpos + 5], data.qpos[D.freeQpos + 6]];

async function run(label, rand) {
  resetDuck(rand);
  let lastAction = new Array(14).fill(0);
  let prevJv = new Array(14).fill(0);           // 1-tick joint_vel lag
  const cmd = command({});
  console.log(`\n-- ${label}: drop from z=0.125, 200 ticks of ${STAND}`);
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
    for (let k = 0; k < 14; k++) {
      const v = reference[k] + lastAction[k];
      target[k] = Math.min(Math.max(v, LO[k]), HI[k]);
    }
    plant.stepTick(target);
    if ((t + 1) % 50 === 0) {
      const g = projectedGravity(quat());
      console.log(`  t=${((t + 1) / 50).toFixed(1)}s  trunk z=${(data.qpos[D.freeQpos + 2] * 1000).toFixed(1)}mm  grav z=${g[2].toFixed(3)}`);
    }
  }
  const g = projectedGravity(quat());
  const z = data.qpos[D.freeQpos + 2];
  const ok = z > 0.10 && z < 0.13 && g[2] < -0.9;
  console.log(`  ${label} FINAL z=${(z * 1000).toFixed(1)}mm grav_z=${g[2].toFixed(3)} -> ${ok ? 'STANDING' : 'NOT STANDING'}`);
  return ok;
}

let pass = 0;
if (await run('nominal', null)) pass++;
for (let seed = 1; seed <= 4; seed++)
  if (await run(`perturbed seed ${seed}`, lcg(seed * 7919))) pass++;
console.log(`\n${pass}/5 drops settled standing on the training-parameter plant.`);
process.exit(pass === 5 ? 0 : 1);
