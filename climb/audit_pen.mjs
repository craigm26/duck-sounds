// Which geom penetrates, how deep, and what is the plant's NORMAL contact
// softness? Baseline: duck standing on the flat floor, no stairs.
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
import { findStairJoints, layoutStairs, clearStairs, STAIR_Y } from '../site/stairs.js';
const C=JSON.parse(fs.readFileSync('duckkit-constants.json','utf8'));
const {HOME,LO,HI,buildObs,projectedGravity,command,findDuckJoints}=makeLoop(C);
const mj=await load();
mj.FS.writeFile('/s.mjb',new Uint8Array(fs.readFileSync('scene.mjb')));
const model=mj.MjModel.mj_loadBinary('/s.mjb',new mj.MjVFS());
const data=new mj.MjData(model);
const D=findDuckJoints(model),ADDR=findStairJoints(model);
let GYRO=0;for(let i=0;i<model.nsensor;i++)if(model.sensor(i).name==='imu_ang_vel')GYRO=model.sensor(i).adr;
const stand=await ort.InferenceSession.create('./BEST_alpha_stand.onnx');
const DT=1/C.tickHz; const gname=g=>model.geom(g).name||'';
let STEP0=-1,FLOOR=-1; const WATCH=[];
for(let g=0;g<model.ngeom;g++){const n=gname(g);
  if(n==='step0_geom')STEP0=g; if(/floor|ground/.test(n))FLOOR=g;
  if(/jaw|beak|head|foot_collision|sole/.test(n))WATCH.push(g);}
console.log('watching:',WATCH.map(gname).join(', '));
console.log('floor geom:',gname(FLOOR));
const quat=()=>[data.qpos[D.freeQpos+3],data.qpos[D.freeQpos+4],data.qpos[D.freeQpos+5],data.qpos[D.freeQpos+6]];
function poseAt(tr,t){if(t<=0)return HOME.slice();let pt=0,pp=HOME;
  for(const f of tr){if(t<=f.t){const u=(t-pt)/Math.max(f.t-pt,1e-9),s=u*u*(3-2*u);
    return f.pose.map((v,k)=>pp[k]+(v-pp[k])*s);}pt=f.t;pp=f.pose;}return tr[tr.length-1].pose.slice();}
async function tick(off,blend,cmd,la){
  const q=quat();const jp=[],jv=[];
  for(let k=0;k<14;k++){jp.push(data.qpos[D.qpos[k]]);jv.push(data.qvel[D.dof[k]]);}
  const obs=buildObs([data.sensordata[GYRO],data.sensordata[GYRO+1],data.sensordata[GYRO+2]],projectedGravity(q),jp,jv,la,cmd);
  const r=await stand.run({obs:new ort.Tensor('float32',obs,[1,61])});
  const na=Array.from(r.actions.data);
  for(let k=0;k<14;k++){const v=HOME[k]+na[k]+(off?(off[k]-HOME[k])*blend:0);
    data.ctrl[k]=Math.min(Math.max(v,LO[k]),HI[k]);}
  for(let s=0;s<4;s++)mj.mj_step(model,data);
  return na;
}
// BASELINE: stand on the flat floor for 4 s, measure foot-vs-FLOOR penetration
mj.mj_resetData(model,data); clearStairs(data,ADDR);
data.qpos[D.freeQpos]=0;data.qpos[D.freeQpos+1]=STAIR_Y;data.qpos[D.freeQpos+2]=0.12;data.qpos[D.freeQpos+3]=1;
for(let i=0;i<14;i++){data.qpos[D.qpos[i]]=HOME[i];data.ctrl[i]=HOME[i];}
mj.mj_forward(model,data);
{ let la=new Array(14).fill(0); const cmd=command({}); let worst=1e9,who='';
  for(let t=0;t<200;t++){ clearStairs(data,ADDR); la=await tick(null,0,cmd,la);
    for(const g of WATCH){ const d=mj.mj_geomDistance(model,data,g,FLOOR,0.05,null);
      if(d<worst){worst=d;who=gname(g);} } }
  console.log(`\nBASELINE standing on the flat floor, 4 s: deepest foot-vs-floor penetration ${(worst*1000).toFixed(2)} mm (${who})`);
  console.log('  -> this is the plant\'s normal soft-contact depth. Anything near it is physics, not clipping.\n'); }
// Per-geom penetration into step0_geom for a saved track
for(const f of process.argv.slice(2)){
  const j=JSON.parse(fs.readFileSync('../climb/'+f,'utf8'));
  const h=parseInt(f.match(/_(\d+)mm/)[1],10)/1000;
  const cfg={count:4,rise:h,run:0.28,start:0.12};
  mj.mj_resetData(model,data); layoutStairs(data,ADDR,cfg);
  data.qpos[D.freeQpos]=0.12-0.07-j.gap;data.qpos[D.freeQpos+1]=STAIR_Y+(j.side||0);
  data.qpos[D.freeQpos+2]=0.12;data.qpos[D.freeQpos+3]=1;
  for(let i=0;i<14;i++){data.qpos[D.qpos[i]]=HOME[i];data.ctrl[i]=HOME[i];}
  mj.mj_forward(model,data);
  const tr=j.keyframes.map(k=>({t:k.t,pose:k.pose.slice()}));
  let la=new Array(14).fill(0); const cmd=command({vx:j.approach||0});
  const worst={}; for(const g of WATCH) worst[gname(g)]=1e9;
  const rec=()=>{for(const g of WATCH){const d=mj.mj_geomDistance(model,data,g,STEP0,0.05,null);
    if(d<worst[gname(g)])worst[gname(g)]=d;}};
  for(let t=0;t<25;t++){layoutStairs(data,ADDR,cfg);la=await tick(null,0,cmd,la);rec();}
  const total=tr[tr.length-1].t+0.8;
  for(let t=0;t*DT<total;t++){layoutStairs(data,ADDR,cfg);la=await tick(poseAt(tr,t*DT),j.blend,cmd,la);rec();}
  for(let t=0;t<50;t++){layoutStairs(data,ADDR,cfg);la=await tick(null,j.blend,cmd,la);rec();}
  console.log(f+'  deepest penetration INTO step0_geom, per geom:');
  for(const k of Object.keys(worst)) if(worst[k]<0.004)
    console.log(`   ${k.padEnd(24)} ${(worst[k]*1000).toFixed(2)} mm`);
}
