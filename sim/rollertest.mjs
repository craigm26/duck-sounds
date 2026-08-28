import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
import { findStairJoints, clearStairs } from '../site/stairs.js';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json','utf8'));
const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/r.mjb', new Uint8Array(fs.readFileSync('scene-rollers.mjb')));
const model = mj.MjModel.mj_loadBinary('/r.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model), ADDR = findStairJoints(model);
let GYRO = 0;
for (let i=0;i<model.nsensor;i++) if (model.sensor(i).name==='imu_ang_vel') GYRO = model.sensor(i).adr;
console.log('ROLLERS nq', model.nq, 'trunk free at', D.freeQpos, 'gyro', GYRO, 'stairs', !!ADDR);

async function run(policy, opts, secs) {
  const sess = await ort.InferenceSession.create('./' + policy);
  mj.mj_resetData(model, data);
  data.qpos[D.freeQpos+2] = 0.12; data.qpos[D.freeQpos+3] = 1;
  for (let i=0;i<14;i++){ data.qpos[D.qpos[i]] = HOME[i]; data.ctrl[i] = HOME[i]; }
  if (ADDR) clearStairs(data, ADDR);
  mj.mj_forward(model, data);
  const cmd = command(opts);
  let la = new Array(14).fill(0), fell = false;
  const x0 = data.qpos[D.freeQpos], y0 = data.qpos[D.freeQpos+1];
  for (let t=0;t<Math.round(secs*C.tickHz);t++){
    if (ADDR) clearStairs(data, ADDR);
    const f=D.freeQpos;
    const q=[data.qpos[f+3],data.qpos[f+4],data.qpos[f+5],data.qpos[f+6]];
    const jp=[],jv=[];
    for(let k=0;k<14;k++){jp.push(data.qpos[D.qpos[k]]);jv.push(data.qvel[D.dof[k]]);}
    const obs=buildObs([data.sensordata[GYRO],data.sensordata[GYRO+1],data.sensordata[GYRO+2]],
                       projectedGravity(q),jp,jv,la,cmd);
    const out=await sess.run({obs:new ort.Tensor('float32',obs,[1,61])});
    la=Array.from(out.actions.data);
    for(let k=0;k<14;k++) data.ctrl[k]=Math.min(Math.max(HOME[k]+la[k],LO[k]),HI[k]);
    for(let s=0;s<4;s++) mj.mj_step(model,data);
    if (projectedGravity([data.qpos[f+3],data.qpos[f+4],data.qpos[f+5],data.qpos[f+6]])[2] > -0.5) fell = true;
  }
  const dx=data.qpos[D.freeQpos]-x0, dy=data.qpos[D.freeQpos+1]-y0;
  return { d: Math.hypot(dx,dy), dx, dy, z: data.qpos[D.freeQpos+2], fell };
}

for (const [pol, opts, label] of [
  ['BEST_roller.onnx',        { vx: 0 },    'roller, stopped'],
  ['BEST_roller.onnx',        { vx: 0.45 }, 'roller, forward'],
  ['BEST_roller.onnx',        { vx: 0.8 },  'roller, fast'],
  ['BEST_roller.onnx',        { vx: 0.4, vyaw: 0.6 }, 'roller, turning'],
  ['BEST_roller_crouch.onnx', { vx: 0.45 }, 'roller crouch'],
  ['alpha_walking.onnx',      { vx: 0.45 }, 'WALKING policy on wheels'],
]) {
  const r = await run(pol, opts, 8);
  console.log(`ROLL ${label.padEnd(24)} moved ${r.d.toFixed(3)} m  (${(r.d/8).toFixed(3)} m/s)  z ${r.z.toFixed(3)}  ${r.fell?'FELL':'upright'}`);
}
