// Does climb_lib's `feetUp` test distinguish a foot ON A TREAD from a foot ON
// THE FLOOR at the rises this project is reporting?
//
// climb_lib.mjs:143-147 counts a foot as "up" when
//     geom_xpos.z > h - 0.005   AND   geom_xpos.x > 0.05
// so the tolerance is 5 mm. If a foot geom's CENTRE sits above (h - 5 mm) while
// resting on the floor, the test passes with the duck standing on the ground.
// This settles the 25 stand ticks that every attempt() begins with, on a FLAT
// floor, and prints the foot geom heights that test is applied to.
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
import { findStairJoints, layoutStairs, STAIR_Y } from '../site/stairs.js';
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
const quat = () => [data.qpos[D.freeQpos+3],data.qpos[D.freeQpos+4],data.qpos[D.freeQpos+5],data.qpos[D.freeQpos+6]];

const FEET = [];
for (let g=0; g<model.ngeom; g++){
  const n = model.geom(g).name || '';
  if (/foot_collision|sole/.test(n)) FEET.push([g, n]);
}
console.log('foot geoms matched by climb_lib:', FEET.length, FEET.map(f=>f[1]).join(', '));

// exactly attempt()'s flat setup, h = 0
const cfg = { count: 4, rise: 0.0, run: 0.28, start: 0.12 };
mj.mj_resetData(model,data);
layoutStairs(data, ADDR, cfg);
data.qpos[D.freeQpos] = 0.12 - 0.07 - 0.06;
data.qpos[D.freeQpos+1] = STAIR_Y;
data.qpos[D.freeQpos+2]=0.12; data.qpos[D.freeQpos+3]=1;
for(let i=0;i<14;i++){data.qpos[D.qpos[i]]=HOME[i];data.ctrl[i]=HOME[i];}
mj.mj_forward(model,data);
let la = new Array(14).fill(0);
const cmd = command({ vx: 0 });
for(let t=0;t<200;t++){
  layoutStairs(data, ADDR, cfg);
  const q=quat(); const jp=[],jv=[];
  for(let k=0;k<14;k++){jp.push(data.qpos[D.qpos[k]]);jv.push(data.qvel[D.dof[k]]);}
  const obs=buildObs([data.sensordata[GYRO],data.sensordata[GYRO+1],data.sensordata[GYRO+2]],projectedGravity(q),jp,jv,la,cmd);
  const r=await stand.run({obs:new ort.Tensor('float32',obs,[1,61])});
  la=Array.from(r.actions.data);
  for(let k=0;k<14;k++) data.ctrl[k]=Math.min(Math.max(HOME[k]+la[k],LO[k]),HI[k]);
  for(let s=0;s<4;s++) mj.mj_step(model,data);
}
console.log('\nSTANDING ON A FLAT FLOOR, trunk z =', (data.qpos[D.freeQpos+2]*1000).toFixed(1), 'mm, trunk x =', (data.qpos[D.freeQpos]*1000).toFixed(1), 'mm');
for (const [g,n] of FEET){
  console.log(`  ${n.padEnd(28)} z = ${(data.geom_xpos[g*3+2]*1000).toFixed(2)} mm   x = ${(data.geom_xpos[g*3]*1000).toFixed(1)} mm`);
}
console.log('\nWould climb_lib count these as "on the tread"?  (needs z > h-5mm AND x > 50mm)');
for (const hmm of [1,2,3,5,8,10,16,20,40]){
  const h = hmm/1000;
  let n=0, nz=0;
  for (const [g] of FEET){
    if (data.geom_xpos[g*3+2] > h - 0.005) nz++;
    if (data.geom_xpos[g*3+2] > h - 0.005 && data.geom_xpos[g*3] > 0.05) n++;
  }
  console.log(`  rise ${String(hmm).padStart(3)} mm: height test passes for ${nz}/${FEET.length} foot geoms; with the x>50mm gate, feetUp = ${n}  ${n>=2?'<-- FLOOR-STANDING FEET SATISFY "feetUp>=2"':''}`);
}
