// What each shipped policy actually does, measured.
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

function reset(){
  mj.mj_resetData(model,data);
  data.qpos[D.freeQpos+2]=0.12; data.qpos[D.freeQpos+3]=1;
  for(let i=0;i<14;i++){data.qpos[D.qpos[i]]=HOME[i];data.ctrl[i]=HOME[i];}
  clearStairs(data,ADDR);
  mj.mj_forward(model,data);
}
const upright = () => projectedGravity([data.qpos[D.freeQpos+3],data.qpos[D.freeQpos+4],
  data.qpos[D.freeQpos+5],data.qpos[D.freeQpos+6]])[2] < -0.55;

async function run(file, cmdMaker, secs, warm) {
  const stand = await ort.InferenceSession.create('./BEST_alpha_stand.onnx');
  const sess = await ort.InferenceSession.create('./' + file);
  reset();
  let la = new Array(14).fill(0);
  const step = async (s, cmd) => {
    const f=D.freeQpos;
    const q=[data.qpos[f+3],data.qpos[f+4],data.qpos[f+5],data.qpos[f+6]];
    const jp=[],jv=[];
    for(let k=0;k<14;k++){jp.push(data.qpos[D.qpos[k]]);jv.push(data.qvel[D.dof[k]]);}
    const obs=buildObs([data.sensordata[GYRO],data.sensordata[GYRO+1],data.sensordata[GYRO+2]],
                       projectedGravity(q),jp,jv,la,cmd);
    const out=await s.run({obs:new ort.Tensor('float32',obs,[1,61])});
    la=Array.from(out.actions.data);
    for(let k=0;k<14;k++) data.ctrl[k]=Math.min(Math.max(HOME[k]+la[k],LO[k]),HI[k]);
    for(let n=0;n<4;n++) mj.mj_step(model,data);
  };
  for (let t=0;t<warm;t++) await step(stand, command({}));
  const x0=data.qpos[D.freeQpos], y0=data.qpos[D.freeQpos+1], z0=data.qpos[D.freeQpos+2];
  let minZ=z0, maxZ=z0, fell=false;
  const ticks=Math.round(secs*C.tickHz);
  for (let t=0;t<ticks;t++){
    await step(sess, cmdMaker(t/ticks));
    minZ=Math.min(minZ,data.qpos[D.freeQpos+2]); maxZ=Math.max(maxZ,data.qpos[D.freeQpos+2]);
    if(!upright()) fell=true;
  }
  const dx=data.qpos[D.freeQpos]-x0, dy=data.qpos[D.freeQpos+1]-y0;
  return {dx,dy,z:data.qpos[D.freeQpos+2],minZ,maxZ,fell,recovered:upright()};
}

const phase = u => command({ vx: Math.cos(2*Math.PI*u), vy: Math.sin(2*Math.PI*u) });
const SKILLS = [
  ['BEST_alpha_stand.onnx',   () => command({}),            3, 50],
  ['BEST_alpha_sitstand.onnx',() => command({ vx: 1 }),      3, 50],
  ['BEST_alpha_sitstand.onnx',() => command({ vx: 0 }),      3, 50],
  ['ball_kick_left.onnx',     () => command({}),             2, 50],
  ['ball_kick_right.onnx',    () => command({}),             2, 50],
  ['alpha_ground_pick.onnx',  phase,                         3, 50],
  ['roulade.onnx',            () => command({}),             3, 50],
];
for (const [file, cm, secs, warm] of SKILLS) {
  const r = await run(file, cm, secs, warm);
  console.log(`SKILL ${file.replace('.onnx','').padEnd(20)} d=(${r.dx.toFixed(3)},${r.dy.toFixed(3)})  z ${r.minZ.toFixed(3)}..${r.maxZ.toFixed(3)} end ${r.z.toFixed(3)}  ${r.fell?'went over':'stayed up'}  ${r.recovered?'upright at end':'DOWN at end'}`);
}
