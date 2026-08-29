// ac_ablate.mjs — which plant change costs roulade its recovery-to-stand?
// 16 DR rollouts (Pollen's own randomisation, same seeded dice per config) of
// roulade.onnx under ablations of the new plant:
//
//   full      ac_scene2 + delay + BAM friction              (the new plant)
//   force96   full, but actuator forcerange restored ±0.96  (undo current limit)
//   nodelay   full, but no actuation delay
//   nofrict   full, but static XML frictionloss (no BAM budget)
//   xmlonly   ac_scene2 with neither delay nor BAM friction (XML deltas only)
//   oldplant  scene.mjb, plain servo loop                   (the old plant)
//
// CFG=<name> node ac_ablate.mjs   (or no CFG: runs all six sequentially)
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
import { clearStairs, findStairJoints } from '../site/stairs.js';
import { makePlant } from './ac_plant.mjs';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);
const DT = 1 / C.tickHz;
const ROLLOUTS = 16;

const CONFIGS = {
  full:     { scene: 'ac_scene2.mjb', delay: true,  friction: true,  force: null },
  force96:  { scene: 'ac_scene2.mjb', delay: true,  friction: true,  force: 0.96 },
  nodelay:  { scene: 'ac_scene2.mjb', delay: false, friction: true,  force: null },
  nofrict:  { scene: 'ac_scene2.mjb', delay: true,  friction: false, force: null },
  xmlonly:  { scene: 'ac_scene2.mjb', delay: false, friction: false, force: null },
  oldplant: { scene: 'scene.mjb',     delay: false, friction: false, force: null, plain: true },
};

const mjMod = await load();

async function runConfig(name) {
  const cfg = CONFIGS[name];
  const mj = mjMod;
  mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync(cfg.scene)));
  const model = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
  const data = new mj.MjData(model);
  const D = findDuckJoints(model), ADDR = findStairJoints(model);
  let GYRO = 0;
  for (let i = 0; i < model.nsensor; i++)
    if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;
  if (cfg.force != null)
    for (let a = 0; a < model.nu; a++) {
      model.actuator_forcerange[a * 2] = -cfg.force;
      model.actuator_forcerange[a * 2 + 1] = cfg.force;
    }
  const plant = cfg.plain ? null
    : makePlant(mj, model, data, D, (process.env.PSEED ? Number(process.env.PSEED) : 0x51ed270b) >>> 0, { delay: cfg.delay, friction: cfg.friction });

  const roulade = await ort.InferenceSession.create('./roulade.onnx');
  const stand = await ort.InferenceSession.create('./BEST_alpha_stand.onnx');

  // Same DR dice as measure_success.mjs, re-seeded per config.
  let seed = (process.env.SEED ? Number(process.env.SEED) : 0x2f6e2b1) >>> 0;
  const rand = () => {
    seed ^= seed << 13; seed >>>= 0; seed ^= seed >> 17; seed ^= seed << 5; seed >>>= 0;
    return seed / 0x100000000;
  };
  const between = (lo, hi) => lo + rand() * (hi - lo);

  const FOOT_GEOMS = [];
  for (let g = 0; g < model.ngeom; g++)
    if (/foot/i.test(model.geom(g).name || '')) FOOT_GEOMS.push(g);
  const BASE_FRICTION = FOOT_GEOMS.map(g => model.geom_friction[g * 3]);
  let TRUNK = -1;
  for (let b = 0; b < model.nbody; b++) if (model.body(b).name === 'trunk_base') TRUNK = b;
  const BASE_COM = [model.body_ipos[TRUNK * 3], model.body_ipos[TRUNK * 3 + 1], model.body_ipos[TRUNK * 3 + 2]];

  const quat = () => [data.qpos[D.freeQpos + 3], data.qpos[D.freeQpos + 4],
                      data.qpos[D.freeQpos + 5], data.qpos[D.freeQpos + 6]];
  const upright = () => projectedGravity(quat())[2] < -0.5;

  let ok = 0;
  const finals = [];
  for (let i = 0; i < ROLLOUTS; i++) {
    const friction = between(0.7, 1.3);
    for (let k = 0; k < FOOT_GEOMS.length; k++)
      model.geom_friction[FOOT_GEOMS[k] * 3] = BASE_FRICTION[k] * friction;
    for (let k = 0; k < 3; k++)
      model.body_ipos[TRUNK * 3 + k] = BASE_COM[k] + between(-0.003, 0.003);
    const dropZ = between(0.12, 0.13);
    const pushAt = Math.round(between(3.0, 6.0) * C.tickHz);

    mj.mj_resetData(model, data);
    if (ADDR) clearStairs(data, ADDR);
    data.qpos[D.freeQpos + 2] = dropZ;
    data.qpos[D.freeQpos + 3] = 1;
    for (let k = 0; k < 14; k++) { data.qpos[D.qpos[k]] = HOME[k]; data.ctrl[k] = HOME[k]; }
    mj.mj_forward(model, data);
    if (plant) plant.resetDelay(HOME);

    let lastAction = new Array(14).fill(0);
    let prevJv = new Array(14).fill(0);
    let dead = false;
    for (let t = -25; t < 150; t++) {
      if (t === pushAt) {
        data.qvel[D.freeDof] += between(-0.3, 0.3);
        data.qvel[D.freeDof + 1] += between(-0.3, 0.3);
      }
      const jp = [], jv = [];
      for (let k = 0; k < 14; k++) { jp.push(data.qpos[D.qpos[k]]); jv.push(data.qvel[D.dof[k]]); }
      const obs = buildObs(
        [data.sensordata[GYRO], data.sensordata[GYRO + 1], data.sensordata[GYRO + 2]],
        projectedGravity(quat()), jp, plant ? prevJv : jv, lastAction, command({}));
      prevJv = jv;
      const session = t < 0 ? stand : roulade;
      const out = await session.run({ obs: new ort.Tensor('float32', obs, [1, 61]) });
      lastAction = Array.from(out.actions.data);
      const target = new Array(14);
      for (let k = 0; k < 14; k++)
        target[k] = Math.min(Math.max(HOME[k] + lastAction[k], LO[k]), HI[k]);
      if (plant) plant.stepTick(target);
      else {
        for (let k = 0; k < 14; k++) data.ctrl[k] = target[k];
        for (let s2 = 0; s2 < 4; s2++) mj.mj_step(model, data);
      }
      if (!Number.isFinite(data.qpos[D.freeQpos + 2])) { dead = true; break; }
    }
    for (let k = 0; k < FOOT_GEOMS.length; k++)
      model.geom_friction[FOOT_GEOMS[k] * 3] = BASE_FRICTION[k];
    for (let k = 0; k < 3; k++) model.body_ipos[TRUNK * 3 + k] = BASE_COM[k];
    if (dead) { finals.push('nan'); continue; }
    const z = data.qpos[D.freeQpos + 2];
    const good = upright() && z >= 0.100;
    if (good) ok++;
    finals.push((z * 1000).toFixed(0));
  }
  console.log(`${name.padEnd(9)} roulade ends-standing ${String(ok).padStart(2)}/${ROLLOUTS}  final z (mm): ${finals.join(' ')}`);
}

const only = process.env.CFG;
for (const name of Object.keys(CONFIGS)) {
  if (only && name !== only) continue;
  await runConfig(name);
}
