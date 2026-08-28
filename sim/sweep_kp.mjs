import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
const C = JSON.parse(fs.readFileSync('duckkit-constants.json','utf8'));
const { HOME, buildObs, gaitTargets, projectedGravity, command } = makeLoop(C);
const mj = await load();
const session = await ort.InferenceSession.create('./alpha_walking.onnx');
const inputName = session.inputNames[0];
const base = fs.readFileSync('scene.xml','utf8');
const yawOf = q => Math.atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]*q[2]+q[3]*q[3]));

async function trial(kp, kv, opts, secs){
  const xml = base.replace(/kp="[\d.]+" kv="[\d.]+"/g, `kp="${kp}" kv="${kv}"`);
  mj.FS.writeFile('/s.xml', xml);
  const model = mj.MjModel.mj_loadXML('/s.xml');
  const data = new mj.MjData(model);
  data.qpos[2]=0.1231; data.qpos[3]=1;
  for(let i=0;i<14;i++){data.qpos[7+i]=HOME[i];data.ctrl[i]=HOME[i];}
  mj.mj_forward(model,data);
  const cmd=command(opts); let la=new Array(14).fill(0), prev=null;
  const y0=yawOf([data.qpos[3],data.qpos[4],data.qpos[5],data.qpos[6]]);
  let fell=false;
  for(let t=0;t<Math.round(secs*C.tickHz);t++){
    const q=[data.qpos[3],data.qpos[4],data.qpos[5],data.qpos[6]];
    const jp=[],jv=[];
    for(let k=0;k<14;k++){jp.push(data.qpos[7+k]);jv.push(data.qvel[6+k]);}
    const obs=buildObs([data.sensordata[0],data.sensordata[1],data.sensordata[2]],projectedGravity(q),jp,jv,la,cmd);
    const out=await session.run({[inputName]:new ort.Tensor('float32',obs,[1,61])});
    la=Array.from(out[session.outputNames[0]].data);
    prev=gaitTargets(la,prev);
    for(let k=0;k<14;k++) data.ctrl[k]=prev[k];
    for(let s=0;s<4;s++) mj.mj_step(model,data);
    if (projectedGravity([data.qpos[3],data.qpos[4],data.qpos[5],data.qpos[6]])[2] > -0.5) fell=true;
  }
  let dy=yawOf([data.qpos[3],data.qpos[4],data.qpos[5],data.qpos[6]])-y0;
  while(dy>Math.PI) dy-=2*Math.PI; while(dy<-Math.PI) dy+=2*Math.PI;
  return { x:data.qpos[0], y:data.qpos[1], dyaw:dy, fell };
}

for (const [kp,kv] of [[6,0.2],[9,0.25],[14,0.35],[20,0.5],[30,0.7],[45,1.0]]) {
  const f  = await trial(kp,kv,{vx:0.15},6);
  const l  = await trial(kp,kv,{vx:0,vyaw:0.8},6);
  const r  = await trial(kp,kv,{vx:0,vyaw:-0.8},6);
  const ok = !f.fell && !l.fell && !r.fell;
  console.log(`KP ${String(kp).padStart(2)} kv ${kv}  fwd x=${f.x.toFixed(2)} y=${f.y.toFixed(2)} dyaw=${f.dyaw.toFixed(2)}   L=${l.dyaw.toFixed(2)} R=${r.dyaw.toFixed(2)}   ${ok?'upright':'FELL'}`);
}
