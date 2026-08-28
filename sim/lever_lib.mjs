// The head as a single finger: plant it on the tread and roll the body up over
// it, rather than lifting a foot onto it.
//
// This is a different move from the step-up. The step-up unweights a leg so a
// foot can reach the tread, which caps it at the leg's own reach. Levering does
// not care about leg reach: the head becomes a pivot and the body rotates over
// it. Whether 0.96 N.m can drive that rotation is the question.
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
import { findStairJoints, layoutStairs, clearStairs } from '../site/stairs.js';
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
const J = { lhy:0, lhr:1, lhp:2, lk:3, la:4, np:5, hp:6, hy:7, hr:8, rhy:9, rhr:10, rhp:11, rk:12, ra:13 };

// A four-phase lever: reach the head onto the tread, load it, drive the body
// forward over that pivot, then catch the landing.
const B = {
  tReach:[0.25,1.0], tLoad:[0.15,0.8], tDrive:[0.2,1.0], tCatch:[0.2,0.9],
  reachNeck:[0.2,1.5], reachHead:[-0.4,1.5],
  loadNeck:[0.2,1.5], loadHead:[-0.6,1.5], loadKnee:[-0.6,1.2], loadHip:[-0.8,1.0],
  driveNeck:[-1.2,1.2], driveHead:[-1.2,1.2],
  driveHip:[-1.4,1.4], driveKnee:[-1.4,1.4], driveAnkle:[-1.4,1.0],
  catchHip:[-1.2,1.2], catchKnee:[-1.2,1.2], catchAnkle:[-1.0,1.0],
  blend:[0.6,2.4], approach:[0.0,0.5],
};
const rnd = ([a,b]) => a + Math.random()*(b-a);
const randP = () => Object.fromEntries(Object.keys(B).map(k => [k, rnd(B[k])]));
const jitter = (p,s) => Object.fromEntries(Object.keys(B).map(k => {
  const [a,b] = B[k];
  return [k, Math.min(b, Math.max(a, p[k] + (Math.random()*2-1)*(b-a)*s))];
}));

function trackOf(p){
  const f = HOME.slice();
  f[J.np]=p.reachNeck; f[J.hp]=p.reachHead;
  const g = f.slice();
  g[J.np]=p.loadNeck; g[J.hp]=p.loadHead;
  g[J.lk]=HOME[J.lk]+p.loadKnee; g[J.rk]=HOME[J.rk]-p.loadKnee;
  g[J.lhp]=HOME[J.lhp]+p.loadHip; g[J.rhp]=HOME[J.rhp]-p.loadHip;
  const h = g.slice();
  h[J.np]=p.driveNeck; h[J.hp]=p.driveHead;
  h[J.lhp]=HOME[J.lhp]+p.driveHip; h[J.rhp]=HOME[J.rhp]-p.driveHip;
  h[J.lk]=HOME[J.lk]+p.driveKnee; h[J.rk]=HOME[J.rk]-p.driveKnee;
  h[J.la]=HOME[J.la]+p.driveAnkle; h[J.ra]=HOME[J.ra]-p.driveAnkle;
  const i = h.slice();
  i[J.lhp]=HOME[J.lhp]+p.catchHip; i[J.rhp]=HOME[J.rhp]-p.catchHip;
  i[J.lk]=HOME[J.lk]+p.catchKnee; i[J.rk]=HOME[J.rk]-p.catchKnee;
  i[J.la]=HOME[J.la]+p.catchAnkle; i[J.ra]=HOME[J.ra]-p.catchAnkle;
  return [{t:p.tReach,pose:f},{t:p.tReach+p.tLoad,pose:g},
          {t:p.tReach+p.tLoad+p.tDrive,pose:h},
          {t:p.tReach+p.tLoad+p.tDrive+p.tCatch,pose:i},
          {t:p.tReach+p.tLoad+p.tDrive+p.tCatch+0.7,pose:HOME.slice()}];
}
function poseAt(tr,time){
  if (time<=0) return HOME.slice();
  let pt=0,pp=HOME;
  for(const fr of tr){ if(time<=fr.t){ const u=(time-pt)/Math.max(fr.t-pt,1e-9),s=u*u*(3-2*u);
      return fr.pose.map((v,k)=>pp[k]+(v-pp[k])*s);} pt=fr.t; pp=fr.pose; }
  return tr[tr.length-1].pose.slice();
}
const quat = () => [data.qpos[D.freeQpos+3],data.qpos[D.freeQpos+4],data.qpos[D.freeQpos+5],data.qpos[D.freeQpos+6]];

async function attempt(p, h){
  mj.mj_resetData(model,data);
  layoutStairs(data, ADDR, { count: 1, rise: h, run: 0.40, start: 0.10 });
  data.qpos[D.freeQpos+2]=0.12; data.qpos[D.freeQpos+3]=1;
  for(let i=0;i<14;i++){data.qpos[D.qpos[i]]=HOME[i];data.ctrl[i]=HOME[i];}
  mj.mj_forward(model,data);
  const tr = trackOf(p);
  let la = new Array(14).fill(0);
  const cmd = command({ vx: p.approach });
  const total = tr[tr.length-1].t + 1.0;
  // settle, then lever
  for(let t=0;t<30;t++){
    const q=quat(); const jp=[],jv=[];
    for(let k=0;k<14;k++){jp.push(data.qpos[D.qpos[k]]);jv.push(data.qvel[D.dof[k]]);}
    const obs=buildObs([data.sensordata[GYRO],data.sensordata[GYRO+1],data.sensordata[GYRO+2]],projectedGravity(q),jp,jv,la,cmd);
    const out=await stand.run({obs:new ort.Tensor('float32',obs,[1,61])});
    la=Array.from(out.actions.data);
    for(let k=0;k<14;k++) data.ctrl[k]=Math.min(Math.max(HOME[k]+la[k],LO[k]),HI[k]);
    for(let s=0;s<4;s++) mj.mj_step(model,data);
  }
  let maxZ = data.qpos[D.freeQpos+2];
  for(let t=0;t*DT<total;t++){
    layoutStairs(data, ADDR, { count: 1, rise: h, run: 0.40, start: 0.10 });
    const q=quat(); const jp=[],jv=[];
    for(let k=0;k<14;k++){jp.push(data.qpos[D.qpos[k]]);jv.push(data.qvel[D.dof[k]]);}
    const obs=buildObs([data.sensordata[GYRO],data.sensordata[GYRO+1],data.sensordata[GYRO+2]],projectedGravity(q),jp,jv,la,cmd);
    const out=await stand.run({obs:new ort.Tensor('float32',obs,[1,61])});
    la=Array.from(out.actions.data);
    const off = poseAt(tr, t*DT);
    for(let k=0;k<14;k++){
      const v = HOME[k]+la[k]+(off[k]-HOME[k])*p.blend;
      data.ctrl[k]=Math.min(Math.max(v,LO[k]),HI[k]);
    }
    for(let s=0;s<4;s++) mj.mj_step(model,data);
    maxZ = Math.max(maxZ, data.qpos[D.freeQpos+2]);
  }
  const x = data.qpos[D.freeQpos], z = data.qpos[D.freeQpos+2];
  const up = projectedGravity(quat())[2] < -0.7;
  // ON the step means: past the riser, and riding at least 60 mm above the
  // tread. Lower than a full stance, because arriving by levering can end in a
  // crouch and still count as up.
  const onTop = up && x > 0.12 && (z - h) > 0.060;
  return { onTop, x, z, above: z - h, maxZ, up };
}

export { attempt, trackOf, poseAt };
