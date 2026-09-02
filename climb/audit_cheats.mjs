// Cheat audit. Own copy of the loop (sim/ untouched) with three extra probes
// the searches never recorded, plus the two perturbations they said they could
// not run: DROP height and FOOT FRICTION.
//   - lateral escape: |y - STAIR_Y| vs STAIR_HALF_WIDTH (did it go BESIDE the flight?)
//   - torque: max |data.actuator_force| vs the plant's +/-0.6405 N.m ceiling
//   - penetration: most-negative mj_geomDistance(jaw|foot, step0_geom) -> clipping
//   - trunk pitch at the terminal tick -> "past the riser" by FALLING onto the tread
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
import { findStairJoints, layoutStairs, STAIR_Y, STAIR_HALF_WIDTH } from '../site/stairs.js';
const C = JSON.parse(fs.readFileSync('duckkit-constants.json','utf8'));
const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const model = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model), ADDR = findStairJoints(model);
let GYRO=0; for(let i=0;i<model.nsensor;i++) if(model.sensor(i).name==='imu_ang_vel') GYRO=model.sensor(i).adr;
const stand = await ort.InferenceSession.create('./BEST_alpha_stand.onnx');
const DT = 1/C.tickHz;
const gname = g => model.geom(g).name || '';
let STEP0=-1, JAW=[], FEET=[];
for(let g=0;g<model.ngeom;g++){ const n=gname(g);
  if(n==='step0_geom') STEP0=g;
  if(/jaw|beak|head/.test(n)) JAW.push(g);
  if(/foot_collision|sole/.test(n)) FEET.push(g); }
// the plant's own ceiling, read not assumed
let FR=0; for(let a=0;a<model.nu;a++) FR=Math.max(FR, model.actuator_forcerange[a*2+1]);
const FRICT0 = Float64Array.from(FEET.map(g=>model.geom_friction[g*3]));
const quat=()=>[data.qpos[D.freeQpos+3],data.qpos[D.freeQpos+4],data.qpos[D.freeQpos+5],data.qpos[D.freeQpos+6]];
function poseAt(tr,time){ if(time<=0) return HOME.slice(); let pt=0,pp=HOME;
  for(const f of tr){ if(time<=f.t){const u=(time-pt)/Math.max(f.t-pt,1e-9),s=u*u*(3-2*u);
    return f.pose.map((v,k)=>pp[k]+(v-pp[k])*s);} pt=f.t; pp=f.pose; } return tr[tr.length-1].pose.slice(); }

async function go(track, o, h, { drop=0.12, fmul=1.0 } = {}) {
  const cfg={count:4,rise:h,run:0.28,start:0.12};
  FEET.forEach((g,i)=>{ model.geom_friction[g*3] = FRICT0[i]*fmul; });
  mj.mj_resetData(model,data);
  layoutStairs(data,ADDR,cfg);
  data.qpos[D.freeQpos]=0.12-0.07-o.gap;
  data.qpos[D.freeQpos+1]=STAIR_Y+(o.side||0);
  data.qpos[D.freeQpos+2]=drop; data.qpos[D.freeQpos+3]=1;
  for(let i=0;i<14;i++){data.qpos[D.qpos[i]]=HOME[i];data.ctrl[i]=HOME[i];}
  mj.mj_forward(model,data);
  const tr=track.map(f=>({t:f.t,pose:f.pose.slice()}));
  let la=new Array(14).fill(0); const cmd=command({vx:o.approach||0});
  let maxTq=0, minPen=1e9, offFlight=false, maxAbsY=0;
  const step=async(off)=>{
    layoutStairs(data,ADDR,cfg);
    const q=quat(); const jp=[],jv=[];
    for(let k=0;k<14;k++){jp.push(data.qpos[D.qpos[k]]);jv.push(data.qvel[D.dof[k]]);}
    const obs=buildObs([data.sensordata[GYRO],data.sensordata[GYRO+1],data.sensordata[GYRO+2]],projectedGravity(q),jp,jv,la,cmd);
    const r=await stand.run({obs:new ort.Tensor('float32',obs,[1,61])});
    la=Array.from(r.actions.data);
    for(let k=0;k<14;k++){ const v=HOME[k]+la[k]+(off?(off[k]-HOME[k])*o.blend:0);
      data.ctrl[k]=Math.min(Math.max(v,LO[k]),HI[k]); }
    for(let s=0;s<4;s++) mj.mj_step(model,data);
    for(let a=0;a<model.nu;a++) maxTq=Math.max(maxTq,Math.abs(data.actuator_force[a]));
    for(const g of JAW.concat(FEET)) minPen=Math.min(minPen, mj.mj_geomDistance(model,data,g,STEP0,0.05,null));
    const dy=Math.abs(data.qpos[D.freeQpos+1]-STAIR_Y);
    if(dy>maxAbsY) maxAbsY=dy;
    if(dy>STAIR_HALF_WIDTH) offFlight=true;
  };
  for(let t=0;t<25;t++) await step(null);
  const total=tr[tr.length-1].t+0.8;
  for(let t=0;t*DT<total;t++) await step(poseAt(tr,t*DT));
  for(let t=0;t<50;t++) await step(null);
  const x=data.qpos[D.freeQpos], z=data.qpos[D.freeQpos+2];
  const g3=projectedGravity(quat()); const up=g3[2]<-0.90;
  let feetUp=0; for(const g of FEET) if(data.geom_xpos[g*3+2]>h-0.005 && data.geom_xpos[g*3]>0.05) feetUp++;
  return { onTop: up && x>0.12 && (z-h)>0.095 && feetUp>=2, x, z, feetUp, up,
    pitch: g3[0], maxTq, minPen, offFlight, maxAbsY };
}
export { go, FR, STAIR_HALF_WIDTH };

if (process.argv[2]) {
  const mm=v=>(v*1000).toFixed(1);
  console.log(`plant forcerange ceiling read from the model: +/-${FR.toFixed(4)} N.m ; flight half-width ${mm(STAIR_HALF_WIDTH)} mm`);
  for (const f of process.argv.slice(2)) {
    const j=JSON.parse(fs.readFileSync('../climb/'+f,'utf8'));
    const h=parseInt(f.match(/_(\d+)mm/)[1],10)/1000;
    const o={blend:j.blend,approach:j.approach||0,gap:j.gap,side:j.side};
    for (const v of [{},{drop:0.125},{drop:0.13},{fmul:0.7},{fmul:1.3}]) {
      const r=await go(j.keyframes,o,h,v);
      const tag=Object.keys(v).length?JSON.stringify(v):'nominal';
      console.log(`${f.padEnd(20)} ${tag.padEnd(16)} onTop=${r.onTop} x=${mm(r.x).padStart(7)} z=${mm(r.z).padStart(6)} feetUp=${r.feetUp} up=${r.up}` +
        ` | maxTq=${r.maxTq.toFixed(4)} pen=${mm(r.minPen).padStart(6)}mm |y-Y|max=${mm(r.maxAbsY).padStart(6)} offFlight=${r.offFlight} pitchZ=${r.pitch.toFixed(3)}`);
    }
  }
}
