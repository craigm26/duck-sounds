import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
import { layoutStairs, clearStairs, findStairJoints } from '../site/stairs.js';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json','utf8'));
const { HOME, buildObs, gaitTargets, projectedGravity, command, findDuckJoints } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/scene.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const model = mj.MjModel.mj_loadBinary('/scene.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const session = await ort.InferenceSession.create('./alpha_walking.onnx');
const inputName = session.inputNames[0];
let GYRO = 0;
for (let i = 0; i < model.nsensor; i++) if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;
const ADDR = findStairJoints(model);
const D = findDuckJoints(model);
console.log('duck free qpos at', D.freeQpos, ' first leg joint qpos at', D.qpos[0]);
console.log('stair joints found:', !!ADDR, ' nq =', model.nq, ' gyro =', GYRO);

async function climb({ rise, run = 0.07, count = 10, start = 0.45, seconds = 16, vx = 0.45 }) {
  mj.mj_resetData(model, data);
  const cfg = { count, rise, run, start };
  if (rise > 0) layoutStairs(data, ADDR, cfg); else clearStairs(data, ADDR);
  data.qpos[D.freeQpos+2] = 0.12; data.qpos[D.freeQpos+3] = 1;
  for (let i = 0; i < 14; i++) { data.qpos[D.qpos[i]] = HOME[i]; data.ctrl[i] = HOME[i]; }
  mj.mj_forward(model, data);
  const cmd = command({ vx });
  let la = new Array(14).fill(0);
  let peak = data.qpos[D.freeQpos+2], fell = false;
  for (let t = 0; t < Math.round(seconds * C.tickHz); t++) {
    // hold the steps in place: they are on free-sliding joints
    if (rise > 0) layoutStairs(data, ADDR, cfg); else clearStairs(data, ADDR);
    const q = [data.qpos[D.freeQpos+3],data.qpos[D.freeQpos+4],data.qpos[D.freeQpos+5],data.qpos[D.freeQpos+6]];
    const jp=[],jv=[];
    for (let k=0;k<14;k++){jp.push(data.qpos[D.qpos[k]]);jv.push(data.qvel[D.dof[k]]);}
    const obs = buildObs([data.sensordata[GYRO],data.sensordata[GYRO+1],data.sensordata[GYRO+2]],
                         projectedGravity(q), jp, jv, la, cmd);
    const out = await session.run({[inputName]: new ort.Tensor('float32', obs, [1,61])});
    la = Array.from(out[session.outputNames[0]].data);
    for (let k=0;k<14;k++) data.ctrl[k] = HOME[k] + la[k];
    for (let s=0;s<4;s++) mj.mj_step(model,data);
    peak = Math.max(peak, data.qpos[D.freeQpos+2]);
    if (projectedGravity([data.qpos[D.freeQpos+3],data.qpos[D.freeQpos+4],data.qpos[D.freeQpos+5],data.qpos[D.freeQpos+6]])[2] > -0.5) fell = true;
  }
  return { x: data.qpos[D.freeQpos], z: data.qpos[D.freeQpos+2], peak, fell };
}
for (const rise of [0.001, 0.002, 0.003, 0.005, 0.008]) {
  const r = await climb({ rise, count: 10, run: 0.09 });
  const up = Math.max(0, r.z - 0.116);
  console.log(`STAIRS rise=${(rise*1000).toFixed(0)}mm run=9cm  x=${r.x.toFixed(2)}m  z=${r.z.toFixed(3)}  up ${(up*100).toFixed(1)}cm  ${r.fell?'FELL':'upright'}`);
}
// a ramp is the fair comparison: same gradient, no lip to catch a toe
for (const rise of [0.003, 0.006]) {
  const r = await climb({ rise, count: 12, run: 0.03 });
  const up = Math.max(0, r.z - 0.116);
  console.log(`RAMPish rise=${(rise*1000).toFixed(0)}mm run=3cm  x=${r.x.toFixed(2)}m  z=${r.z.toFixed(3)}  up ${(up*100).toFixed(1)}cm  ${r.fell?'FELL':'upright'}`);
}
