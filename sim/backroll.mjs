// Can it roll backwards? And is a flip within these servos at all?
//
// The shipped roulade is a FORWARD roll, and forward is the easy direction:
// gravity helps once the duck is past its toes. Backwards it has to lift its
// own mass over its heels. With +-0.96 N.m per joint on a 737 g robot that is
// the question, not a formality.
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
const DT = 1/C.tickHz;

function reset(){
  mj.mj_resetData(model,data);
  data.qpos[D.freeQpos+2]=0.12; data.qpos[D.freeQpos+3]=1;
  for(let i=0;i<14;i++){data.qpos[D.qpos[i]]=HOME[i];data.ctrl[i]=HOME[i];}
  clearStairs(data,ADDR);
  mj.mj_forward(model,data);
}
const quat = () => [data.qpos[D.freeQpos+3],data.qpos[D.freeQpos+4],data.qpos[D.freeQpos+5],data.qpos[D.freeQpos+6]];
// pitch of the trunk: how far it has rotated about its own y axis
function pitch(){
  const [w,x,y,z] = quat();
  return Math.asin(Math.max(-1, Math.min(1, 2*(w*y - z*x))));
}

/** Smoothstep track from a table of [t, {joint: value}] deltas on the home pose. */
function track(frames){
  return frames.map(([t, d]) => {
    const pose = HOME.slice();
    for (const k of Object.keys(d)) pose[+k] = HOME[+k] + d[k];
    return { t, pose };
  });
}
function poseAt(tr, time){
  if (time <= 0) return HOME.slice();
  let pt = 0, pp = HOME;
  for (const f of tr){
    if (time <= f.t){
      const u = (time - pt)/Math.max(f.t-pt,1e-9), s = u*u*(3-2*u);
      return f.pose.map((v,k)=>pp[k]+(v-pp[k])*s);
    }
    pt = f.t; pp = f.pose;
  }
  return tr[tr.length-1].pose.slice();
}

const stand = await ort.InferenceSession.create('./BEST_alpha_stand.onnx');
async function run(tr, blend, secs, warm=40){
  reset();
  let la = new Array(14).fill(0);
  const step = async (sess, cmd, offset) => {
    const f=D.freeQpos;
    const q=[data.qpos[f+3],data.qpos[f+4],data.qpos[f+5],data.qpos[f+6]];
    const jp=[],jv=[];
    for(let k=0;k<14;k++){jp.push(data.qpos[D.qpos[k]]);jv.push(data.qvel[D.dof[k]]);}
    const obs=buildObs([data.sensordata[GYRO],data.sensordata[GYRO+1],data.sensordata[GYRO+2]],
                       projectedGravity(q),jp,jv,la,cmd);
    const out=await sess.run({obs:new ort.Tensor('float32',obs,[1,61])});
    la=Array.from(out.actions.data);
    for(let k=0;k<14;k++){
      const v = HOME[k]+la[k]+(offset?(offset[k]-HOME[k])*blend:0);
      data.ctrl[k]=Math.min(Math.max(v,LO[k]),HI[k]);
    }
    for(let s=0;s<4;s++) mj.mj_step(model,data);
  };
  for(let t=0;t<warm;t++) await step(stand, command({}), null);
  let minPitch = 0, maxPitch = 0, minGz = 0, went = false;
  const n = Math.round(secs*C.tickHz);
  for(let t=0;t<n;t++){
    await step(stand, command({}), poseAt(tr, t*DT));
    const p = pitch(), gz = projectedGravity(quat())[2];
    minPitch = Math.min(minPitch, p); maxPitch = Math.max(maxPitch, p);
    minGz = Math.min(minGz, gz);
    if (gz > 0.2) went = true;            // passed beyond horizontal
  }
  const gz = projectedGravity(quat())[2];
  return { minPitch, maxPitch, went, upright: gz < -0.85, z: data.qpos[D.freeQpos+2] };
}

// A backward roll attempt: crouch, then drive the ankles and hips to throw the
// body over its heels, then tuck.
const J = { lhp:2, lk:3, la:4, np:5, hp:6, rhp:11, rk:12, ra:13 };
const attempts = [
  ['gentle',  track([[0.4,{[J.lk]:0.5,[J.rk]:-0.5}], [0.8,{[J.la]:-0.8,[J.ra]:0.8,[J.np]:-0.6}], [1.4,{}]]), 1.0],
  ['harder',  track([[0.35,{[J.lk]:0.9,[J.rk]:-0.9,[J.np]:0.5}], [0.65,{[J.la]:-1.4,[J.ra]:1.4,[J.np]:-1.0,[J.hp]:-0.8}], [1.5,{}]]), 1.4],
  ['hardest', track([[0.3,{[J.lk]:1.2,[J.rk]:-1.2,[J.lhp]:0.6,[J.rhp]:-0.6,[J.np]:0.8}],
                     [0.55,{[J.la]:-1.6,[J.ra]:1.6,[J.lhp]:-0.9,[J.rhp]:0.9,[J.np]:-1.2,[J.hp]:-1.0}],
                     [1.6,{}]]), 1.8],
];
for (const [name, tr, blend] of attempts){
  const r = await run(tr, blend, 4);
  console.log(`BACKROLL ${name.padEnd(8)} pitch ${(r.minPitch*57.3).toFixed(0)}..${(r.maxPitch*57.3).toFixed(0)} deg  past horizontal: ${r.went?'YES':'no'}  ended ${r.upright?'upright':'down'} at ${r.z.toFixed(3)} m`);
}
// For comparison, the trained FORWARD roll.
const roll = await ort.InferenceSession.create('./roulade.onnx');
reset();
let la = new Array(14).fill(0);
let mn=0, mx=0, went=false;
for(let t=0;t<200;t++){
  const f=D.freeQpos;
  const q=[data.qpos[f+3],data.qpos[f+4],data.qpos[f+5],data.qpos[f+6]];
  const jp=[],jv=[];
  for(let k=0;k<14;k++){jp.push(data.qpos[D.qpos[k]]);jv.push(data.qvel[D.dof[k]]);}
  const obs=buildObs([data.sensordata[GYRO],data.sensordata[GYRO+1],data.sensordata[GYRO+2]],projectedGravity(q),jp,jv,la,command({}));
  const out=await roll.run({obs:new ort.Tensor('float32',obs,[1,61])});
  la=Array.from(out.actions.data);
  for(let k=0;k<14;k++) data.ctrl[k]=Math.min(Math.max(HOME[k]+la[k],LO[k]),HI[k]);
  for(let s=0;s<4;s++) mj.mj_step(model,data);
  const p=pitch(), gz=projectedGravity(quat())[2];
  mn=Math.min(mn,p); mx=Math.max(mx,p);
  if (gz > 0.2) went = true;
}
console.log(`ROULADE  trained  pitch ${(mn*57.3).toFixed(0)}..${(mx*57.3).toFixed(0)} deg  past horizontal: ${went?'YES':'no'}`);
