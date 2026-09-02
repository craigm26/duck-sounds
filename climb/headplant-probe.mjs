// THE PREMISE TEST for every head-anchor strategy: can this duck get its head
// onto a tread at all?
//
// contact-dump.mjs: driving the neck alone, with alpha_stand holding the trunk
// upright, puts the furthest jaw geom at x = 91 mm while the riser is at 120 mm
// and the tread top at 90 mm -- the head goes UP, not over, and 400 ticks
// produced ZERO jaw<->step contacts. jaw-reach.mjs says the 190 mm of forward
// reach exists only at 25-55 deg of nose-down TRUNK pitch, and trunk pitch is
// not a servo: it comes from hip pitch and ankle. So lean the trunk with the
// hips and ankles, drive the neck, and count jaw<->step contacts.
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
const stand = await ort.InferenceSession.create('./BEST_alpha_stand.onnx');
let GYRO=0; for(let i=0;i<m.nsensor;i++) if(m.sensor(i).name==='imu_ang_vel') GYRO=m.sensor(i).adr;
const quat=()=>[d.qpos[D.freeQpos+3],d.qpos[D.freeQpos+4],d.qpos[D.freeQpos+5],d.qpos[D.freeQpos+6]];
const JAW=[30,31,32]; const isStep=g=>g>=5&&g<=18;
const COUNT_CONTACTS = process.env.COUNT_CONTACTS === '1';
const bid=n=>{for(let b=0;b<m.nbody;b++) if(m.body(b).name===n) return b; return -1;};
const TRUNK=bid('trunk_base');
const J={lhy:0,lhr:1,lhp:2,lk:3,la:4,np:5,hp:6,hy:7,hr:8,rhy:9,rhr:10,rhp:11,rk:12,ra:13};
const DT=1/C.tickHz;
// nose-down trunk pitch, degrees: world z-component of the trunk's own +x axis
const trunkPitch=()=> -Math.asin(Math.max(-1,Math.min(1,d.xmat[TRUNK*9+6])))*180/Math.PI;

function smooth(u){u=Math.max(0,Math.min(1,u));return u*u*(3-2*u);}

async function run({rise, gap, hipLean, ankLean, kneeFold, neck, head, blend, tLean=0.8, tHold=2.2}){
  const cfg={count:4,rise,run:0.28,start:0.12};
  mj.mj_resetData(m,d); layoutStairs(d,ADDR,cfg);
  d.qpos[D.freeQpos]=0.12-0.07-gap; d.qpos[D.freeQpos+1]=STAIR_Y; d.qpos[D.freeQpos+2]=0.12; d.qpos[D.freeQpos+3]=1;
  for(let i=0;i<14;i++){d.qpos[D.qpos[i]]=HOME[i];d.ctrl[i]=HOME[i];}
  mj.mj_forward(m,d);
  const goal=HOME.slice();
  goal[J.lhp]=HOME[J.lhp]+hipLean; goal[J.rhp]=HOME[J.rhp]-hipLean;
  goal[J.lk] =HOME[J.lk]+kneeFold; goal[J.rk] =HOME[J.rk]-kneeFold;
  goal[J.la] =HOME[J.la]+ankLean;  goal[J.ra] =HOME[J.ra]-ankLean;
  goal[J.np]=neck; goal[J.hp]=head;
  let la=new Array(14).fill(0); const cmd=command({vx:0});
  let jawTicks=0, minDist=1e9, maxJawX=-1e9, maxPitch=-1e9, jawZatMax=0, headOnTread=0;
  const NT=Math.round((tLean+tHold)/DT);
  for(let t=0;t<25+NT;t++){
    layoutStairs(d,ADDR,cfg);
    const q=quat(); const jp=[],jv=[];
    for(let k=0;k<14;k++){jp.push(d.qpos[D.qpos[k]]);jv.push(d.qvel[D.dof[k]]);}
    const obs=buildObs([d.sensordata[GYRO],d.sensordata[GYRO+1],d.sensordata[GYRO+2]],projectedGravity(q),jp,jv,la,cmd);
    const r=await stand.run({obs:new ort.Tensor('float32',obs,[1,61])});
    la=Array.from(r.actions.data);
    const u = t<25 ? 0 : smooth(((t-25)*DT)/tLean);
    for(let k=0;k<14;k++){
      const off=HOME[k]+(goal[k]-HOME[k])*u;
      const v=HOME[k]+la[k]+(off-HOME[k])*blend;
      d.ctrl[k]=Math.min(Math.max(v,LO[k]),HI[k]);
    }
    for(let s=0;s<4;s++) mj.mj_step(m,d);
    // d.contact.get(c) ALSO leaks -- .delete() does not help; ~50k calls abort the module.
    // Contacts were counted in a short run (contact-dump.mjs / the first sweep) and are real;
    // long sweeps must use the geometric proxy below instead.
    if (COUNT_CONTACTS) for(let c=0;c<d.ncon;c++){const ct=d.contact.get(c);
      const g1=ct.geom1, g2=ct.geom2; ct.delete();
      const a=JAW.includes(g1)?g2:JAW.includes(g2)?g1:-1;
      if(a>=0&&isStep(a)) jawTicks++;
    }
    // NOTE: mj_geomDistance LEAKS in this wasm build -- 20k calls abort the module at its
    // 2 GB heap cap (proved with a 20k-call loop). Do not call it inside an episode loop.
    for(const g of JAW){ if(d.geom_xpos[g*3]>maxJawX){maxJawX=d.geom_xpos[g*3];jawZatMax=d.geom_xpos[g*3+2];} }
    const p=trunkPitch(); if(p>maxPitch) maxPitch=p;
    if(d.geom_xpos[30*3]>0.12 && d.geom_xpos[30*3+2]>rise) headOnTread++;
  }
  return {jawTicks, minDist, maxJawX, jawZatMax, maxPitch, headOnTread,
          x:d.qpos[D.freeQpos], z:d.qpos[D.freeQpos+2], up:projectedGravity(quat())[2],
          np:d.qpos[D.qpos[5]], lhp:d.qpos[D.qpos[2]]};
}

console.log('rise gap hipLean ankLean kneeFold neck head blend | maxTrunkPitch  maxJawX  jawZ  jawXvsRiser  jaw<->step ticks  headOverEdge ticks  end(x,z,up)');
const rows=[];
for (const rise of [0.02,0.04,0.06,0.09,0.12,0.18])
 for (const gap of [0.02,0.06])
  for (const hipLean of [0.9,1.4])
   for (const ankLean of [-0.5])
    for (const neck of [-1.5,-0.5]){
      const p={rise,gap,hipLean,ankLean,kneeFold:0.0,neck,head:0.3,blend:2.0};
      const r=await run(p);
      rows.push({p,r});
      console.log(`${(rise*1000).toFixed(0).padStart(4)} ${(gap*1000).toFixed(0).padStart(3)} ${hipLean.toFixed(2).padStart(7)} ${ankLean.toFixed(2).padStart(7)} ${'0.00'.padStart(8)} ${neck.toFixed(2).padStart(5)} ${'0.30'.padStart(5)} ${'2.0'.padStart(5)} |`
        +` ${r.maxPitch.toFixed(1).padStart(13)}  ${(r.maxJawX*1000).toFixed(0).padStart(7)}  ${(r.jawZatMax*1000).toFixed(0).padStart(4)}  ${'n/a'.padStart(14)}  ${String(r.jawTicks).padStart(16)}  ${String(r.headOnTread).padStart(18)}  ${(r.x*1000).toFixed(0)},${(r.z*1000).toFixed(0)},${r.up.toFixed(2)}`);
    }
fs.writeFileSync('../climb/headplant-results.json', JSON.stringify(rows,null,1));
console.log('\nwrote ../climb/headplant-results.json');
