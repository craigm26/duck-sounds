import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
const C = JSON.parse(fs.readFileSync('duckkit-constants.json','utf8'));
const { HOME, buildObs, gaitTargets, projectedGravity, command } = makeLoop(C);
const mj = await load();
// their model wants meshdir="assets"
mj.FS.mkdir('/assets');
for (const f of fs.readdirSync('assets')) mj.FS.writeFile('/assets/' + f, fs.readFileSync('assets/' + f));
mj.FS.writeFile('/scene.xml', fs.readFileSync('scene_physics.xml','utf8'));
const model = mj.MjModel.mj_loadXML('/scene.xml');
const data = new mj.MjData(model);
console.log('LOADED nq', model.nq, 'nv', model.nv, 'nu', model.nu, 'ngeom', model.ngeom, 'nsensordata', model.nsensordata);
// The gyro is NOT at sensordata[0]. Their sensor block opens with a 4-value
// framequat, so the angular-velocity sensor the runtime reads sits at adr 7 —
// reading 0..2 was feeding the policy three components of a quaternion.
let GYRO = 0;
for (let i = 0; i < model.nsensor; i++) {
  if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;
}
console.log('GYRO adr =', GYRO, 'of', model.nsensordata);
const session = await ort.InferenceSession.create(process.env.POLICY || './alpha_walking.onnx');
const inputName = session.inputNames[0];
const MODE = process.argv[2] || 'duckkit';
const yawOf = q => Math.atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]*q[2]+q[3]*q[3]));

// settle onto the floor from the home pose
function reset(){
  mj.mj_resetData(model,data);
  // Their spawn: z = 0.12, home pose, ctrl already at the pose. The policy
  // starts immediately — with kp 0.55 the servos are deliberately soft and it
  // is the POLICY that holds the duck up, so settling without it just collapses.
  data.qpos[2]=0.12; data.qpos[3]=1;
  for(let i=0;i<14;i++){data.qpos[7+i]=HOME[i];data.ctrl[i]=HOME[i];}
  mj.mj_forward(model,data);
}
async function run(opts, secs){
  reset();
  const cmd=command(opts); let la=new Array(14).fill(0), prev=null;
  const x0=data.qpos[0], y0=data.qpos[1], yw0=yawOf([data.qpos[3],data.qpos[4],data.qpos[5],data.qpos[6]]);
  let fell=false;
  for(let t=0;t<Math.round(secs*C.tickHz);t++){
    const q=[data.qpos[3],data.qpos[4],data.qpos[5],data.qpos[6]];
    const jp=[],jv=[];
    for(let k=0;k<14;k++){jp.push(data.qpos[7+k]);jv.push(data.qvel[6+k]);}
    const obs=buildObs([data.sensordata[GYRO],data.sensordata[GYRO+1],data.sensordata[GYRO+2]],projectedGravity(q),jp,jv,la,cmd);
    const out=await session.run({[inputName]:new ort.Tensor('float32',obs,[1,61])});
    la=Array.from(out[session.outputNames[0]].data);
    if (MODE === 'mjlab') {
      // Pollen's simulator: ctrl = pose + action, scale 1.0, no low-pass.
      for(let k=0;k<14;k++) data.ctrl[k]=HOME[k]+la[k]*1.0;
    } else {
      prev=gaitTargets(la,prev);
      for(let k=0;k<14;k++) data.ctrl[k]=prev[k];
    }
    for(let s=0;s<4;s++) mj.mj_step(model,data);
    if(projectedGravity([data.qpos[3],data.qpos[4],data.qpos[5],data.qpos[6]])[2] > -0.5) fell=true;
  }
  let dy=yawOf([data.qpos[3],data.qpos[4],data.qpos[5],data.qpos[6]])-yw0;
  while(dy>Math.PI)dy-=2*Math.PI; while(dy<-Math.PI)dy+=2*Math.PI;
  return {dx:data.qpos[0]-x0, dy:data.qpos[1]-y0, dyaw:dy, z:data.qpos[2], fell};
}
reset();
console.log('MODE', MODE, ' spawn z =', data.qpos[2].toFixed(3));
console.log('SETTLED trunk z =', data.qpos[2].toFixed(4), ' contacts', data.ncon);
for (const vyaw of [0.5, 1.0, -0.5, -1.0]) {
  const r = await run({vx:0, vyaw}, 8);
  console.log(`TURN vyaw=${vyaw>0?'+':''}${vyaw.toFixed(1)}  dyaw ${r.dyaw>=0?'+':''}${r.dyaw.toFixed(2)} rad  ${r.fell?'FELL':'upright'}  ${Math.sign(r.dyaw)===Math.sign(vyaw)?'CORRECT SIGN':'wrong sign'}`);
}
for (const [vx,vyaw] of [[0.4,0.5],[0.4,-0.5]]) {
  const r = await run({vx, vyaw}, 8);
  console.log(`ARC  vx=${vx} vyaw=${vyaw>0?'+':''}${vyaw}  d=(${r.dx.toFixed(2)}, ${r.dy.toFixed(2)})  dyaw ${r.dyaw.toFixed(2)}  ${r.fell?'FELL':'upright'}`);
}
