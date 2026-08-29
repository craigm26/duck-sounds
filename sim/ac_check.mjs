// ac_check.mjs — validate ac_scene.mjb (allcollisions robot on an mjlab plane).
//
// Loads the .mjb in the same WASM runtime the recorder uses, prints the model
// inventory (joint order, collision geoms PER BODY — the head matters), then
// drops the duck at the home pose from z = 0.125 and runs BEST_alpha_stand.onnx
// for 200 ticks (4 s at 50 Hz) through the EXACT observation/control path of
// record_intents.mjs: gyro from the `imu_ang_vel` sensor, projected gravity
// from the trunk free-joint quaternion, joint pos/vel by name via
// findDuckJoints, obs deviation from the policy's OWN declared neutral
// (onnx_meta.mjs), ctrl = neutral + action at scale 1.0, no low-pass,
// 4 physics steps of 0.005 s per policy tick.
//
// PASS: trunk z settles ≈ 0.116 m and stays there to the 4 s mark, with
// projected-gravity z ≈ −1 (upright). Because a contact sim is chaotic, the
// drop runs 5 times: nominal + 4 deterministically perturbed starts
// (±2 mm trunk, ±0.02 rad per joint).
//
// Run:  node ac_check.mjs        (from this directory)
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { declaredDefaultPose } from './onnx_meta.mjs';
import { makeLoop } from '../site/duckloop.mjs';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);

const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('ac_scene.mjb')));
const model = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model);

// ── inventory ─────────────────────────────────────────────────────────────
console.log('nq', model.nq, 'nv', model.nv, 'nu', model.nu, 'ngeom', model.ngeom,
  'nbody', model.nbody, 'timestep', model.opt.timestep);
const jnames = [];
for (let j = 0; j < model.njnt; j++) jnames.push(model.jnt(j).name);
console.log('joints in model order:', jnames.join(', '));

const perBody = new Map();
for (let g = 0; g < model.ngeom; g++) {
  const body = model.body(model.geom_bodyid[g]).name;
  const collides = model.geom_contype[g] !== 0 || model.geom_conaffinity[g] !== 0;
  if (!perBody.has(body)) perBody.set(body, { coll: 0, total: 0 });
  perBody.get(body).total++;
  if (collides) perBody.get(body).coll++;
}
console.log('collision geoms per body (coll/total):');
for (let b = 0; b < model.nbody; b++) {
  const name = model.body(b).name;
  const e = perBody.get(name) ?? { coll: 0, total: 0 };
  console.log(`  ${name}: ${e.coll}/${e.total}`);
}

let GYRO = -1;
for (let i = 0; i < model.nsensor; i++)
  if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;
if (GYRO < 0) throw new Error('imu_ang_vel sensor missing');
console.log('imu_ang_vel sensor adr', GYRO);

// ── the drop test ─────────────────────────────────────────────────────────
const STAND = 'BEST_alpha_stand.onnx';
const session = await ort.InferenceSession.create('./' + STAND);
const reference = declaredDefaultPose('./' + STAND, HOME) ?? HOME;

// Deterministic LCG so the perturbed starts are reproducible.
function lcg(seed) {
  let s = seed >>> 0;
  return () => ((s = (1664525 * s + 1013904223) >>> 0) / 2 ** 32) * 2 - 1;
}

function resetDuck(rand) {
  mj.mj_resetData(model, data);
  data.qpos[D.freeQpos] = rand ? 0.002 * rand() : 0;
  data.qpos[D.freeQpos + 1] = rand ? 0.002 * rand() : 0;
  data.qpos[D.freeQpos + 2] = 0.125;
  data.qpos[D.freeQpos + 3] = 1;
  for (let k = 0; k < 14; k++) {
    data.qpos[D.qpos[k]] = HOME[k] + (rand ? 0.02 * rand() : 0);
    data.ctrl[k] = HOME[k];
  }
  mj.mj_forward(model, data);
}

const quat = () => [data.qpos[D.freeQpos + 3], data.qpos[D.freeQpos + 4],
                    data.qpos[D.freeQpos + 5], data.qpos[D.freeQpos + 6]];

async function run(label, rand) {
  resetDuck(rand);
  let lastAction = new Array(14).fill(0);
  const cmd = command({});
  console.log(`\n-- ${label}: drop from z=0.125, ${200} ticks of ${STAND}`);
  for (let t = 0; t < 200; t++) {
    const jp = [], jv = [];
    for (let k = 0; k < 14; k++) { jp.push(data.qpos[D.qpos[k]]); jv.push(data.qvel[D.dof[k]]); }
    const obs = buildObs(
      [data.sensordata[GYRO], data.sensordata[GYRO + 1], data.sensordata[GYRO + 2]],
      projectedGravity(quat()), jp, jv, lastAction, cmd, reference);
    const out = await session.run({ obs: new ort.Tensor('float32', obs, [1, 61]) });
    lastAction = Array.from(out.actions.data);
    for (let k = 0; k < 14; k++) {
      const v = reference[k] + lastAction[k]; // action_scale 1.0, no low-pass
      data.ctrl[k] = Math.min(Math.max(v, LO[k]), HI[k]);
    }
    for (let s = 0; s < 4; s++) mj.mj_step(model, data);
    if ((t + 1) % 25 === 0) {
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
console.log(`\n${pass}/5 drops settled standing.`);
process.exit(pass === 5 ? 0 : 1);
