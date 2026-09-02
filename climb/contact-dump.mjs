// Which geom pairs actually make contacts, tick by tick. Written because
// twist-budget.mjs found mj_geomDistance(jaw, step0) = 0.0 mm and ZERO
// jaw<->step contacts in the same episode, which cannot both be true.
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
import { findStairJoints, layoutStairs, STAIR_Y } from '../site/stairs.js';
const C = JSON.parse(fs.readFileSync('duckkit-constants.json','utf8'));
const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const m = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const d = new mj.MjData(m);
const D = findDuckJoints(m), ADDR = findStairJoints(m);
const gn = g => m.geom(g).name || `geom${g}(${m.body(m.geom_bodyid[g]).name})`;
const stand = await ort.InferenceSession.create('./BEST_alpha_stand.onnx');
let GYRO=0; for(let i=0;i<m.nsensor;i++) if(m.sensor(i).name==='imu_ang_vel') GYRO=m.sensor(i).adr;
const quat=()=>[d.qpos[D.freeQpos+3],d.qpos[D.freeQpos+4],d.qpos[D.freeQpos+5],d.qpos[D.freeQpos+6]];

// sanity: does mj_geomDistance work at all in this binding?
mj.mj_resetData(m,d); layoutStairs(d,ADDR,{count:4,rise:0.09,run:0.28,start:0.12});
d.qpos[D.freeQpos]=0; d.qpos[D.freeQpos+1]=STAIR_Y; d.qpos[D.freeQpos+2]=0.12; d.qpos[D.freeQpos+3]=1;
for(let i=0;i<14;i++) d.qpos[D.qpos[i]]=HOME[i];
mj.mj_forward(m,d);
console.log('mj_geomDistance sanity (duck upright at x=0, z=120mm, rise 90mm flight):');
for(const [a,b] of [[29,0],[29,5],[30,5],[31,5],[32,5],[30,0],[26,5]])
  console.log(`  ${gn(a)} <-> ${gn(b)} = ${(mj.mj_geomDistance(m,d,a,b,1.0,null)*1000).toFixed(1)} mm`);
console.log('  (left_foot_collision <-> floor should be ~5 mm; if several read exactly 0.0 the call is not usable here)');

const cfg={count:4,rise:0.09,run:0.28,start:0.12};
mj.mj_resetData(m,d); layoutStairs(d,ADDR,cfg);
d.qpos[D.freeQpos]=0.12-0.07-0.02; d.qpos[D.freeQpos+1]=STAIR_Y; d.qpos[D.freeQpos+2]=0.12; d.qpos[D.freeQpos+3]=1;
for(let i=0;i<14;i++){d.qpos[D.qpos[i]]=HOME[i];d.ctrl[i]=HOME[i];}
mj.mj_forward(m,d);
let la=new Array(14).fill(0); const cmd=command({vx:0.3});
const pairs=new Map();
for(let t=0;t<400;t++){
  layoutStairs(d,ADDR,cfg);
  const q=quat(); const jp=[],jv=[];
  for(let k=0;k<14;k++){jp.push(d.qpos[D.qpos[k]]);jv.push(d.qvel[D.dof[k]]);}
  const obs=buildObs([d.sensordata[GYRO],d.sensordata[GYRO+1],d.sensordata[GYRO+2]],projectedGravity(q),jp,jv,la,cmd);
  const r=await stand.run({obs:new ort.Tensor('float32',obs,[1,61])});
  la=Array.from(r.actions.data);
  const off=HOME.slice(); if(t>60){off[5]=-1.3; off[6]=-0.7;}
  for(let k=0;k<14;k++){ const v=HOME[k]+la[k]+(t>60?(off[k]-HOME[k])*2.0:0);
    d.ctrl[k]=Math.min(Math.max(v,LO[k]),HI[k]); }
  for(let s=0;s<4;s++) mj.mj_step(m,d);
  for(let c=0;c<d.ncon;c++){ const ct=d.contact.get(c);
    const k=`${gn(ct.geom1)} <-> ${gn(ct.geom2)}`; pairs.set(k,(pairs.get(k)||0)+1); }
  if(t===399){
    console.log(`\nfinal tick: ncon=${d.ncon}  trunk x=${(d.qpos[D.freeQpos]*1000).toFixed(1)} z=${(d.qpos[D.freeQpos+2]*1000).toFixed(1)} mm`);
    console.log(`  reached neck_pitch=${d.qpos[D.qpos[5]].toFixed(3)} (cmd ${d.ctrl[5].toFixed(3)}), head_pitch=${d.qpos[D.qpos[6]].toFixed(3)} (cmd ${d.ctrl[6].toFixed(3)})`);
    let jx=-1e9,jz=0; for(const g of [30,31,32]){ if(d.geom_xpos[g*3]>jx){jx=d.geom_xpos[g*3];jz=d.geom_xpos[g*3+2];} }
    console.log(`  furthest jaw geom origin x=${(jx*1000).toFixed(1)} z=${(jz*1000).toFixed(1)} mm; riser face is at x=120 mm, tread top z=90 mm`);
  }
}
console.log('\ncontact pairs seen over 400 ticks (count = tick-appearances):');
[...pairs.entries()].sort((a,b)=>b[1]-a[1]).forEach(([k,v])=>console.log(`  ${String(v).padStart(5)}  ${k}`));
