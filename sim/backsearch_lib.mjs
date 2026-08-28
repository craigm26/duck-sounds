// A real attempt at a backward roll: search the move rather than guess it.
// Objective is how far past vertical the trunk gets, with a bonus for coming
// back upright — a roll that ends on its back is a fall, not a roll.
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
import { findStairJoints, clearStairs } from '../site/stairs.js';
const C = JSON.parse(fs.readFileSync('duckkit-constants.json','utf8'));
const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const model = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model), ADDR = findStairJoints(model);
let GYRO = 0;
for (let i=0;i<model.nsensor;i++) if (model.sensor(i).name==='imu_ang_vel') GYRO = model.sensor(i).adr;
const stand = await ort.InferenceSession.create('./BEST_alpha_stand.onnx');
const DT = 1/C.tickHz;
const J = { lhr:1, lhp:2, lk:3, la:4, np:5, hp:6, rhr:10, rhp:11, rk:12, ra:13 };

const B = {
  tCrouch:[0.15,0.7], tThrow:[0.10,0.5], tTuck:[0.15,0.8],
  crouchKnee:[0.0,1.5], crouchHip:[-1.0,1.0], crouchNeck:[-1.0,1.2],
  throwAnkle:[-1.6,0.4], throwHip:[-1.5,1.0], throwKnee:[-1.5,1.0],
  throwNeck:[-1.4,1.0], throwHead:[-1.4,1.0],
  tuckKnee:[-0.5,1.5], tuckHip:[-1.2,1.2], tuckNeck:[-1.2,1.2],
  blend:[0.6,2.2],
};
const rnd = ([a,b]) => a + Math.random()*(b-a);
const randP = () => Object.fromEntries(Object.keys(B).map(k => [k, rnd(B[k])]));
function jitter(p, s){
  const q = {...p};
  for (const k of Object.keys(B)){ const [a,b]=B[k];
    q[k] = Math.min(b, Math.max(a, p[k] + (Math.random()*2-1)*(b-a)*s)); }
  return q;
}
function trackOf(p){
  const f1 = HOME.slice();
  f1[J.lk]=HOME[J.lk]+p.crouchKnee; f1[J.rk]=HOME[J.rk]-p.crouchKnee;
  f1[J.lhp]=HOME[J.lhp]+p.crouchHip; f1[J.rhp]=HOME[J.rhp]-p.crouchHip;
  f1[J.np]=p.crouchNeck;
  const f2 = f1.slice();
  f2[J.la]=HOME[J.la]+p.throwAnkle; f2[J.ra]=HOME[J.ra]-p.throwAnkle;
  f2[J.lhp]=HOME[J.lhp]+p.throwHip;  f2[J.rhp]=HOME[J.rhp]-p.throwHip;
  f2[J.lk]=HOME[J.lk]+p.throwKnee;   f2[J.rk]=HOME[J.rk]-p.throwKnee;
  f2[J.np]=p.throwNeck; f2[J.hp]=p.throwHead;
  const f3 = f2.slice();
  f3[J.lk]=HOME[J.lk]+p.tuckKnee; f3[J.rk]=HOME[J.rk]-p.tuckKnee;
  f3[J.lhp]=HOME[J.lhp]+p.tuckHip; f3[J.rhp]=HOME[J.rhp]-p.tuckHip;
  f3[J.np]=p.tuckNeck;
  return [{t:p.tCrouch,pose:f1},{t:p.tCrouch+p.tThrow,pose:f2},{t:p.tCrouch+p.tThrow+p.tTuck,pose:f3},
          {t:p.tCrouch+p.tThrow+p.tTuck+0.8,pose:HOME.slice()}];
}
function poseAt(tr,time){
  if (time<=0) return HOME.slice();
  let pt=0,pp=HOME;
  for(const f of tr){ if(time<=f.t){ const u=(time-pt)/Math.max(f.t-pt,1e-9),s=u*u*(3-2*u);
      return f.pose.map((v,k)=>pp[k]+(v-pp[k])*s);} pt=f.t; pp=f.pose; }
  return tr[tr.length-1].pose.slice();
}
const quat = () => [data.qpos[D.freeQpos+3],data.qpos[D.freeQpos+4],data.qpos[D.freeQpos+5],data.qpos[D.freeQpos+6]];

async function score(p){
  mj.mj_resetData(model,data);
  data.qpos[D.freeQpos+2]=0.12; data.qpos[D.freeQpos+3]=1;
  for(let i=0;i<14;i++){data.qpos[D.qpos[i]]=HOME[i];data.ctrl[i]=HOME[i];}
  clearStairs(data,ADDR); mj.mj_forward(model,data);
  const tr = trackOf(p);
  let la = new Array(14).fill(0), maxGz = -1;
  const total = tr[tr.length-1].t + 1.2;
  for(let t=0;t*DT<total;t++){
    const f=D.freeQpos;
    const q=[data.qpos[f+3],data.qpos[f+4],data.qpos[f+5],data.qpos[f+6]];
    const jp=[],jv=[];
    for(let k=0;k<14;k++){jp.push(data.qpos[D.qpos[k]]);jv.push(data.qvel[D.dof[k]]);}
    const obs=buildObs([data.sensordata[GYRO],data.sensordata[GYRO+1],data.sensordata[GYRO+2]],
                       projectedGravity(q),jp,jv,la,command({}));
    const out=await stand.run({obs:new ort.Tensor('float32',obs,[1,61])});
    la=Array.from(out.actions.data);
    const off = poseAt(tr,t*DT);
    for(let k=0;k<14;k++){
      const v=HOME[k]+la[k]+(off[k]-HOME[k])*p.blend;
      data.ctrl[k]=Math.min(Math.max(v,LO[k]),HI[k]);
    }
    for(let s=0;s<4;s++) mj.mj_step(model,data);
    maxGz = Math.max(maxGz, projectedGravity(quat())[2]);
  }
  const endUp = projectedGravity(quat())[2] < -0.85;
  // maxGz: -1 upright, 0 on its side, +1 fully inverted. Over the top is > 0.
  return { over: maxGz, endUp, score: maxGz + (endUp ? 0.35 : 0) };
}

export { score, trackOf, poseAt };
