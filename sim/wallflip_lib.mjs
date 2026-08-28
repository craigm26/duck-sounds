// A flip off a wall. Two feet against a vertical surface, push, rotate.
//
// The case for it: a roll can borrow the floor, but a flip has to come from
// somewhere, and +-0.96 N.m on a 737 g body is not much to spin with. A wall
// gives the feet something to push against that is not underneath them, which
// turns a weak leg extension into angular momentum instead of a hop.
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
const J = { lhp:2, lk:3, la:4, np:5, hp:6, rhp:11, rk:12, ra:13 };

// The east wall's inner face. Spawn the duck a body-length from it so it can
// walk in, plant both feet and push.
const WALL_X = 1.5 - 0.05;
const B = {
  startGap:[0.06,0.30],
  tSet:[0.15,0.6], tPush:[0.08,0.35], tTuck:[0.15,0.7],
  setHip:[-1.2,1.2], setKnee:[-1.0,1.4], setNeck:[-1.0,1.2],
  pushHip:[-1.5,1.5], pushKnee:[-1.5,1.5], pushAnkle:[-1.5,1.2],
  pushNeck:[-1.4,1.2], pushHead:[-1.4,1.2],
  tuckHip:[-1.4,1.4], tuckKnee:[-1.4,1.4], tuckNeck:[-1.2,1.2],
  blend:[0.8,2.4], approach:[0.0,0.6],
};
const rnd = ([a,b]) => a + Math.random()*(b-a);
const randP = () => Object.fromEntries(Object.keys(B).map(k => [k, rnd(B[k])]));
const jitter = (p,s) => Object.fromEntries(Object.keys(B).map(k => {
  const [a,b]=B[k]; return [k, Math.min(b, Math.max(a, p[k] + (Math.random()*2-1)*(b-a)*s))]; }));

function trackOf(p){
  const a = HOME.slice();
  a[J.lhp]=HOME[J.lhp]+p.setHip; a[J.rhp]=HOME[J.rhp]-p.setHip;
  a[J.lk]=HOME[J.lk]+p.setKnee;  a[J.rk]=HOME[J.rk]-p.setKnee;
  a[J.np]=p.setNeck;
  const b = a.slice();
  b[J.lhp]=HOME[J.lhp]+p.pushHip; b[J.rhp]=HOME[J.rhp]-p.pushHip;
  b[J.lk]=HOME[J.lk]+p.pushKnee;  b[J.rk]=HOME[J.rk]-p.pushKnee;
  b[J.la]=HOME[J.la]+p.pushAnkle; b[J.ra]=HOME[J.ra]-p.pushAnkle;
  b[J.np]=p.pushNeck; b[J.hp]=p.pushHead;
  const c = b.slice();
  c[J.lhp]=HOME[J.lhp]+p.tuckHip; c[J.rhp]=HOME[J.rhp]-p.tuckHip;
  c[J.lk]=HOME[J.lk]+p.tuckKnee;  c[J.rk]=HOME[J.rk]-p.tuckKnee;
  c[J.np]=p.tuckNeck;
  return [{t:p.tSet,pose:a},{t:p.tSet+p.tPush,pose:b},
          {t:p.tSet+p.tPush+p.tTuck,pose:c},
          {t:p.tSet+p.tPush+p.tTuck+0.9,pose:HOME.slice()}];
}
function poseAt(tr,time){
  if (time<=0) return HOME.slice();
  let pt=0,pp=HOME;
  for(const f of tr){ if(time<=f.t){const u=(time-pt)/Math.max(f.t-pt,1e-9),s=u*u*(3-2*u);
    return f.pose.map((v,k)=>pp[k]+(v-pp[k])*s);} pt=f.t; pp=f.pose; }
  return tr[tr.length-1].pose.slice();
}
const quat = () => [data.qpos[D.freeQpos+3],data.qpos[D.freeQpos+4],data.qpos[D.freeQpos+5],data.qpos[D.freeQpos+6]];

async function attempt(p){
  mj.mj_resetData(model,data);
  clearStairs(data,ADDR);
  data.qpos[D.freeQpos] = WALL_X - p.startGap;
  data.qpos[D.freeQpos+2]=0.12; data.qpos[D.freeQpos+3]=1;
  for(let i=0;i<14;i++){data.qpos[D.qpos[i]]=HOME[i];data.ctrl[i]=HOME[i];}
  mj.mj_forward(model,data);
  const tr = trackOf(p);
  let la = new Array(14).fill(0);
  const cmd = command({ vx: p.approach });
  let maxGz = -1, spins = 0, prevGz = -1;
  const total = tr[tr.length-1].t + 1.2;
  for(let t=0;t<25;t++){
    const q=quat(); const jp=[],jv=[];
    for(let k=0;k<14;k++){jp.push(data.qpos[D.qpos[k]]);jv.push(data.qvel[D.dof[k]]);}
    const obs=buildObs([data.sensordata[GYRO],data.sensordata[GYRO+1],data.sensordata[GYRO+2]],projectedGravity(q),jp,jv,la,cmd);
    const out=await stand.run({obs:new ort.Tensor('float32',obs,[1,61])});
    la=Array.from(out.actions.data);
    for(let k=0;k<14;k++) data.ctrl[k]=Math.min(Math.max(HOME[k]+la[k],LO[k]),HI[k]);
    for(let s=0;s<4;s++) mj.mj_step(model,data);
  }
  for(let t=0;t*DT<total;t++){
    const q=quat(); const jp=[],jv=[];
    for(let k=0;k<14;k++){jp.push(data.qpos[D.qpos[k]]);jv.push(data.qvel[D.dof[k]]);}
    const obs=buildObs([data.sensordata[GYRO],data.sensordata[GYRO+1],data.sensordata[GYRO+2]],projectedGravity(q),jp,jv,la,cmd);
    const out=await stand.run({obs:new ort.Tensor('float32',obs,[1,61])});
    la=Array.from(out.actions.data);
    const off = poseAt(tr,t*DT);
    for(let k=0;k<14;k++){
      const v=HOME[k]+la[k]+(off[k]-HOME[k])*p.blend;
      data.ctrl[k]=Math.min(Math.max(v,LO[k]),HI[k]);
    }
    for(let s=0;s<4;s++) mj.mj_step(model,data);
    const gz = projectedGravity(quat())[2];
    if (prevGz < 0 && gz >= 0) spins++;
    prevGz = gz;
    maxGz = Math.max(maxGz, gz);
  }
  const endUp = projectedGravity(quat())[2] < -0.85;
  const tilt = Math.acos(Math.max(-1,Math.min(1,-maxGz))) * 57.2958;
  return { tilt, endUp, spins, score: tilt/180 + (endUp ? 0.5 : 0) };
}

export { attempt, trackOf, poseAt };
