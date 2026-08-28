import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
const C = JSON.parse(fs.readFileSync('duckkit-constants.json','utf8'));
const { HOME, buildObs, gaitTargets, projectedGravity, command } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/scene.xml', fs.readFileSync('scene.xml','utf8'));
const model = mj.MjModel.mj_loadXML('/scene.xml');
const data = new mj.MjData(model);
const session = await ort.InferenceSession.create('./alpha_walking.onnx');
const inputName = session.inputNames[0];
const yawOf = q => Math.atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]*q[2]+q[3]*q[3]));

async function run(opts, secs){
  mj.mj_resetData(model,data);
  data.qpos[2]=0.1231; data.qpos[3]=1;
  for(let i=0;i<14;i++){data.qpos[7+i]=HOME[i];data.ctrl[i]=HOME[i];}
  mj.mj_forward(model,data);
  const cmd = command(opts);
  let la=new Array(14).fill(0), prev=null;
  const y0 = yawOf([data.qpos[3],data.qpos[4],data.qpos[5],data.qpos[6]]);
  for(let t=0;t<Math.round(secs*C.tickHz);t++){
    const q=[data.qpos[3],data.qpos[4],data.qpos[5],data.qpos[6]];
    const jp=[],jv=[];
    for(let k=0;k<14;k++){jp.push(data.qpos[7+k]);jv.push(data.qvel[6+k]);}
    const gyro=[data.sensordata[0],data.sensordata[1],data.sensordata[2]];
    const obs=buildObs(gyro,projectedGravity(q),jp,jv,la,cmd);
    const out=await session.run({[inputName]:new ort.Tensor('float32',obs,[1,61])});
    la=Array.from(out[session.outputNames[0]].data);
    prev=gaitTargets(la,prev);
    for(let k=0;k<14;k++) data.ctrl[k]=prev[k];
    for(let s=0;s<4;s++) mj.mj_step(model,data);
  }
  let dy = yawOf([data.qpos[3],data.qpos[4],data.qpos[5],data.qpos[6]]) - y0;
  while(dy> Math.PI) dy-=2*Math.PI;
  while(dy<-Math.PI) dy+=2*Math.PI;
  return { dyaw: dy, x: data.qpos[0], y: data.qpos[1] };
}
// forward: which way is the duck's nose?
const f = await run({vx:0.15}, 6);
console.log(`TEST vx=+0.15        -> x ${f.x.toFixed(3)}  y ${f.y.toFixed(3)}  dyaw ${f.dyaw.toFixed(3)}`);
for (const v of [0.8, -0.8]) {
  const r = await run({vx:0, vyaw:v}, 6);
  console.log(`TEST vyaw=${v>0?'+':''}${v.toFixed(1)}         -> dyaw ${r.dyaw.toFixed(3)} rad (${(r.dyaw/6).toFixed(3)} rad/s)  expected sign ${v>0?'+':'-'}`);
}
for (const v of [0.10, -0.10]) {
  const r = await run({vx:0, vy:v}, 6);
  console.log(`TEST vy=${v>0?'+':''}${v.toFixed(2)}          -> x ${r.x.toFixed(3)}  y ${r.y.toFixed(3)}  (expected y sign ${v>0?'+':'-'})`);
}
