import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '/home/craigm26/projects/duck-sounds/site/duckloop.mjs';
import { findStairJoints, clearStairs } from '/home/craigm26/projects/duck-sounds/site/stairs.js';
const C = JSON.parse(fs.readFileSync('duckkit-constants.json','utf8'));
const { HOME, LO, HI, buildObs, projectedGravity, command, gaitTargets, findDuckJoints } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const model = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model), ADDR = findStairJoints(model);
let GYRO=0; for(let i=0;i<model.nsensor;i++) if(model.sensor(i).name==='imu_ang_vel') GYRO=model.sensor(i).adr;
const net = await ort.InferenceSession.create('./alpha_walking.onnx');
for (const [vx, gait, y, drop] of [[0.15,true,0,0.12],[0.15,false,0,0.12],[0.25,true,0,0.12],[0.15,true,1.305,0.12],[0.15,true,0,0.1231]]) {
  mj.mj_resetData(model,data); clearStairs(data, ADDR);
  data.qpos[D.freeQpos]=0; data.qpos[D.freeQpos+1]=y; data.qpos[D.freeQpos+2]=drop; data.qpos[D.freeQpos+3]=1;
  for(let i=0;i<14;i++){data.qpos[D.qpos[i]]=HOME[i];data.ctrl[i]=HOME[i];}
  mj.mj_forward(model,data);
  let la=new Array(14).fill(0), prev=null; const cmd=command({vx});
  for(let t=0;t<300;t++){
    clearStairs(data, ADDR);
    const q=[data.qpos[D.freeQpos+3],data.qpos[D.freeQpos+4],data.qpos[D.freeQpos+5],data.qpos[D.freeQpos+6]];
    const jp=[],jv=[]; for(let k=0;k<14;k++){jp.push(data.qpos[D.qpos[k]]);jv.push(data.qvel[D.dof[k]]);}
    const obs=buildObs([data.sensordata[GYRO],data.sensordata[GYRO+1],data.sensordata[GYRO+2]],projectedGravity(q),jp,jv,la,cmd);
    const r=await net.run({[net.inputNames[0]]: new ort.Tensor('float32',obs,[1,61])});
    la=Array.from(r[net.outputNames[0]].data);
    if(gait){prev=gaitTargets(la,prev); for(let k=0;k<14;k++) data.ctrl[k]=prev[k];}
    else for(let k=0;k<14;k++) data.ctrl[k]=Math.min(Math.max(HOME[k]+la[k],LO[k]),HI[k]);
    for(let s=0;s<4;s++) mj.mj_step(model,data);
  }
  console.log(`vx=${vx} gait=${gait} y=${y} drop=${drop} -> x ${(data.qpos[D.freeQpos]*1000).toFixed(0)}mm z ${(data.qpos[D.freeQpos+2]*1000).toFixed(0)}mm`);
}
