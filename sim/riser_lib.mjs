// Steal the wall-flip trick for the stairs: push off the RISER.
//
// The wall flip works because a vertical surface gives the feet something to
// push against that is not underneath them, turning a weak leg extension into
// rotation. A stair riser is exactly such a surface, and it is already there —
// no extra wall needed. Combined with the head planted on the tread as a pivot,
// the duck has two points of purchase instead of one.
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
import { findStairJoints, layoutStairs } from '../site/stairs.js';
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
const J = { lhy:0, lhr:1, lhp:2, lk:3, la:4, np:5, hp:6, rhy:9, rhr:10, rhp:11, rk:12, ra:13 };

// Five phases: walk in, set the head on the tread, plant a foot on the RISER,
// push off it while the head pivots, then catch the landing.
const B = {
  gap:[0.03,0.16], approach:[0.0,0.5],
  tSet:[0.2,0.9], tPlant:[0.12,0.6], tPush:[0.10,0.5], tCatch:[0.2,0.9],
  setNeck:[0.2,1.5], setHead:[-0.6,1.5],
  plantHip:[-0.4,1.5], plantKnee:[-1.4,1.0], plantAnkle:[-1.4,1.0],
  pushHip:[-1.5,1.5], pushKnee:[-1.5,1.5], pushAnkle:[-1.5,1.2],
  pushNeck:[-1.4,1.2], pushHead:[-1.4,1.2],
  trailHip:[-1.4,1.4], trailKnee:[-1.4,1.4],
  catchHip:[-1.3,1.3], catchKnee:[-1.3,1.3], catchNeck:[-1.2,1.2],
  blend:[0.7,2.4],
};
const rnd = ([a,b]) => a + Math.random()*(b-a);
const randP = () => Object.fromEntries(Object.keys(B).map(k => [k, rnd(B[k])]));
const jitter = (p,s) => Object.fromEntries(Object.keys(B).map(k => {
  const [a,b]=B[k]; return [k, Math.min(b, Math.max(a, p[k]+(Math.random()*2-1)*(b-a)*s))]; }));

function trackOf(p){
  const a = HOME.slice();
  a[J.np]=p.setNeck; a[J.hp]=p.setHead;
  const b = a.slice();                      // lead foot up onto the riser face
  b[J.lhp]=HOME[J.lhp]+p.plantHip; b[J.lk]=HOME[J.lk]+p.plantKnee; b[J.la]=HOME[J.la]+p.plantAnkle;
  const c = b.slice();                      // push off it
  c[J.lhp]=HOME[J.lhp]+p.pushHip; c[J.lk]=HOME[J.lk]+p.pushKnee; c[J.la]=HOME[J.la]+p.pushAnkle;
  c[J.rhp]=HOME[J.rhp]-p.trailHip; c[J.rk]=HOME[J.rk]-p.trailKnee;
  c[J.np]=p.pushNeck; c[J.hp]=p.pushHead;
  const d = c.slice();
  d[J.lhp]=HOME[J.lhp]+p.catchHip; d[J.rhp]=HOME[J.rhp]-p.catchHip;
  d[J.lk]=HOME[J.lk]+p.catchKnee;  d[J.rk]=HOME[J.rk]-p.catchKnee;
  d[J.np]=p.catchNeck;
  return [{t:p.tSet,pose:a},{t:p.tSet+p.tPlant,pose:b},
          {t:p.tSet+p.tPlant+p.tPush,pose:c},
          {t:p.tSet+p.tPlant+p.tPush+p.tCatch,pose:d},
          {t:p.tSet+p.tPlant+p.tPush+p.tCatch+0.7,pose:HOME.slice()}];
}
function poseAt(tr,time){
  if (time<=0) return HOME.slice();
  let pt=0,pp=HOME;
  for(const f of tr){ if(time<=f.t){const u=(time-pt)/Math.max(f.t-pt,1e-9),s=u*u*(3-2*u);
    return f.pose.map((v,k)=>pp[k]+(v-pp[k])*s);} pt=f.t; pp=f.pose; }
  return tr[tr.length-1].pose.slice();
}
const quat = () => [data.qpos[D.freeQpos+3],data.qpos[D.freeQpos+4],data.qpos[D.freeQpos+5],data.qpos[D.freeQpos+6]];

async function attempt(p, h){
  const cfg = { count: 1, rise: h, run: 0.45, start: 0.12 };
  mj.mj_resetData(model,data);
  layoutStairs(data, ADDR, cfg);
  // Start a settable gap from the riser face, which sits at start - halfDepth.
  data.qpos[D.freeQpos] = 0.12 - 0.07 - p.gap;
  data.qpos[D.freeQpos+2]=0.12; data.qpos[D.freeQpos+3]=1;
  for(let i=0;i<14;i++){data.qpos[D.qpos[i]]=HOME[i];data.ctrl[i]=HOME[i];}
  mj.mj_forward(model,data);
  const tr = trackOf(p);
  let la = new Array(14).fill(0);
  const cmd = command({ vx: p.approach });
  const total = tr[tr.length-1].t + 1.0;
  const step = async (off) => {
    layoutStairs(data, ADDR, cfg);
    const q=quat(); const jp=[],jv=[];
    for(let k=0;k<14;k++){jp.push(data.qpos[D.qpos[k]]);jv.push(data.qvel[D.dof[k]]);}
    const obs=buildObs([data.sensordata[GYRO],data.sensordata[GYRO+1],data.sensordata[GYRO+2]],projectedGravity(q),jp,jv,la,cmd);
    const out=await stand.run({obs:new ort.Tensor('float32',obs,[1,61])});
    la=Array.from(out.actions.data);
    for(let k=0;k<14;k++){
      const v=HOME[k]+la[k]+(off?(off[k]-HOME[k])*p.blend:0);
      data.ctrl[k]=Math.min(Math.max(v,LO[k]),HI[k]);
    }
    for(let s=0;s<4;s++) mj.mj_step(model,data);
  };
  for(let t=0;t<25;t++) await step(null);
  for(let t=0;t*DT<total;t++) await step(poseAt(tr,t*DT));
  // SUCCESS MEANS STANDING ON THE STEP, and it is worth being strict about it.
  // The earlier bar — trunk 60 mm above the tread and past the riser — passes a
  // duck draped over the edge on its chest. This one wants both feet up, the
  // body at close to full standing height above the TREAD, upright, and still
  // there a second later. A move that arrives and then slides off has not
  // climbed anything.
  const settle = async () => {
    for (let t = 0; t < 50; t++) await step(null);
  };
  await settle();
  const x = data.qpos[D.freeQpos], z = data.qpos[D.freeQpos+2];
  const up = projectedGravity(quat())[2] < -0.90;
  let feetUp = 0;
  for (let g = 0; g < model.ngeom; g++) {
    const n = model.geom(g).name || '';
    if (!/foot_collision|sole/.test(n)) continue;
    if (data.geom_xpos[g*3+2] > h - 0.005 && data.geom_xpos[g*3] > 0.12 - 0.07) feetUp++;
  }
  const onTop = up && x > 0.12 && (z - h) > 0.095 && feetUp >= 2;
  return { onTop, x, z, above: z-h, up, feetUp };
}

export { attempt, trackOf, poseAt };
