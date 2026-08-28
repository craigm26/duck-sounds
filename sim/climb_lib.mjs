// Head on the shelf, then walk the feet UP the riser.
//
// Every previous attempt treated the step as something to step onto: one foot
// lifts, reaches the tread, takes weight. That caps at how high a foot can
// reach while the other leg holds the body up — about 20 mm in practice.
//
// This is a different thing. The head takes weight on the tread first, which
// unloads both legs at once; then the feet climb the vertical riser face in
// alternating steps, pressing into it hard enough for friction to hold, walking
// the body up until it can be brought over the edge. The legs never have to
// reach the tread from the floor — they get there a bit at a time.
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
let GYRO = 0;
for (let i=0;i<model.nsensor;i++) if (model.sensor(i).name==='imu_ang_vel') GYRO = model.sensor(i).adr;
const stand = await ort.InferenceSession.create('./BEST_alpha_stand.onnx');
const DT = 1/C.tickHz;
const J = { lhy:0, lhr:1, lhp:2, lk:3, la:4, np:5, hp:6, rhy:9, rhr:10, rhp:11, rk:12, ra:13 };

// Six phases: shelf the head, then L, R, L up the riser, then over.
const B = {
  gap:[0.02,0.12], approach:[0.0,0.4],
  // How far off the centreline of the flight to start. The stairs are flush to
  // a wall, so a duck offset towards it has a vertical surface on one side —
  // the thing the wall flip already proved this robot can push off.
  side:[0.0, 0.085],
  tShelf:[0.25,0.9], tA:[0.10,0.45], tB:[0.10,0.45], tC:[0.10,0.45], tOver:[0.2,0.9],
  shelfNeck:[0.3,1.5], shelfHead:[-0.6,1.5],
  // each rung: how far the hip lifts, how much the knee folds, how hard the
  // ankle presses into the face (that is what stops the foot sliding down).
  // Hip ROLL is what presses a leg sideways into the wall. Without it the duck
  // only ever pushes backwards off the riser, and a foot on a riser slides
  // down; wedged between riser and wall it does not.
  rollA:[-0.9,0.9], rollB:[-0.9,0.9], rollC:[-0.9,0.9],
  hipA:[0.0,1.5], kneeA:[-1.5,0.6], ankA:[-1.4,0.9],
  hipB:[0.0,1.5], kneeB:[-1.5,0.6], ankB:[-1.4,0.9],
  hipC:[0.0,1.6], kneeC:[-1.5,0.6], ankC:[-1.4,0.9],
  overHip:[-1.4,1.4], overKnee:[-1.4,1.4], overNeck:[-1.3,1.0], overHead:[-1.3,1.0],
  blend:[0.7,2.4],
};
const rnd = ([a,b]) => a + Math.random()*(b-a);
const randP = () => Object.fromEntries(Object.keys(B).map(k => [k, rnd(B[k])]));
const jitter = (p,s) => Object.fromEntries(Object.keys(B).map(k => {
  const [a,b]=B[k]; return [k, Math.min(b, Math.max(a, p[k]+(Math.random()*2-1)*(b-a)*s))]; }));

function trackOf(p){
  const shelf = HOME.slice();
  shelf[J.np]=p.shelfNeck; shelf[J.hp]=p.shelfHead;
  // left foot up the face
  const a = shelf.slice();
  a[J.lhp]=HOME[J.lhp]+p.hipA; a[J.lk]=HOME[J.lk]+p.kneeA; a[J.la]=HOME[J.la]+p.ankA;
  a[J.lhr]=HOME[J.lhr]+p.rollA; a[J.rhr]=HOME[J.rhr]+p.rollA;
  // right foot up, past it
  const b = a.slice();
  b[J.rhp]=HOME[J.rhp]-p.hipB; b[J.rk]=HOME[J.rk]-p.kneeB; b[J.ra]=HOME[J.ra]-p.ankB;
  b[J.lhr]=HOME[J.lhr]+p.rollB; b[J.rhr]=HOME[J.rhr]+p.rollB;
  // left again, higher
  const c = b.slice();
  c[J.lhp]=HOME[J.lhp]+p.hipC; c[J.lk]=HOME[J.lk]+p.kneeC; c[J.la]=HOME[J.la]+p.ankC;
  c[J.lhr]=HOME[J.lhr]+p.rollC; c[J.rhr]=HOME[J.rhr]+p.rollC;
  // bring the body over the edge
  const d = c.slice();
  d[J.lhp]=HOME[J.lhp]+p.overHip; d[J.rhp]=HOME[J.rhp]-p.overHip;
  d[J.lk]=HOME[J.lk]+p.overKnee;  d[J.rk]=HOME[J.rk]-p.overKnee;
  d[J.np]=p.overNeck; d[J.hp]=p.overHead;
  const t1=p.tShelf, t2=t1+p.tA, t3=t2+p.tB, t4=t3+p.tC, t5=t4+p.tOver;
  return [{t:t1,pose:shelf},{t:t2,pose:a},{t:t3,pose:b},{t:t4,pose:c},{t:t5,pose:d},
          {t:t5+0.7,pose:HOME.slice()}];
}
function poseAt(tr,time){
  if (time<=0) return HOME.slice();
  let pt=0,pp=HOME;
  for(const f of tr){ if(time<=f.t){const u=(time-pt)/Math.max(f.t-pt,1e-9),s=u*u*(3-2*u);
    return f.pose.map((v,k)=>pp[k]+(v-pp[k])*s);} pt=f.t; pp=f.pose; }
  return tr[tr.length-1].pose.slice();
}
const quat = () => [data.qpos[D.freeQpos+3],data.qpos[D.freeQpos+4],data.qpos[D.freeQpos+5],data.qpos[D.freeQpos+6]];

/**
 * Score a move that is already authored — the keyframes as SHIPPED, read from
 * the intent JSON the page loads, not the parameters a search once explored.
 * This is the only honest way to check a claim on the page: the searched
 * parameters and the exported track can disagree, and the page plays the track.
 */
async function replay(keyframes, opts, h){
  return attempt({ ...opts, __track: keyframes }, h);
}

async function attempt(p, h){
  // A real flight, not one isolated block. A single 450 mm-deep step is a
  // podium: the duck can walk clean over it and stand on the floor beyond,
  // which scored as failure for the right reason but wasted the search.
  //
  // The run is 280 mm because that is what a stair run IS — eleven inches,
  // the tread depth building codes pair with the 180 mm rise this page's
  // slider already allowed. The 90 mm run tried first is not a staircase; a
  // duck cannot stand on a 90 mm tread because its own feet are longer than
  // that, so the search was being asked for something geometrically
  // impossible and correctly returned almost nothing.
  const cfg = { count: 4, rise: h, run: 0.28, start: 0.12 };
  mj.mj_resetData(model,data);
  layoutStairs(data, ADDR, cfg);
  data.qpos[D.freeQpos] = 0.12 - 0.07 - p.gap;
  data.qpos[D.freeQpos+1] = STAIR_Y + (p.side || 0);  // offset towards the wall the flight is flush against
  data.qpos[D.freeQpos+2]=0.12; data.qpos[D.freeQpos+3]=1;
  for(let i=0;i<14;i++){data.qpos[D.qpos[i]]=HOME[i];data.ctrl[i]=HOME[i];}
  mj.mj_forward(model,data);
  const tr = p.__track ? p.__track.map(f => ({ t: f.t, pose: f.pose.slice() })) : trackOf(p);
  let la = new Array(14).fill(0);
  const cmd = command({ vx: p.approach });
  const step = async (off) => {
    layoutStairs(data, ADDR, cfg);
    const q=quat(); const jp=[],jv=[];
    for(let k=0;k<14;k++){jp.push(data.qpos[D.qpos[k]]);jv.push(data.qvel[D.dof[k]]);}
    const obs=buildObs([data.sensordata[GYRO],data.sensordata[GYRO+1],data.sensordata[GYRO+2]],projectedGravity(q),jp,jv,la,cmd);
    const r=await stand.run({obs:new ort.Tensor('float32',obs,[1,61])});
    la=Array.from(r.actions.data);
    for(let k=0;k<14;k++){
      const v=HOME[k]+la[k]+(off?(off[k]-HOME[k])*p.blend:0);
      data.ctrl[k]=Math.min(Math.max(v,LO[k]),HI[k]);
    }
    for(let s=0;s<4;s++) mj.mj_step(model,data);
  };
  for(let t=0;t<25;t++) await step(null);
  const total = tr[tr.length-1].t + 0.8;
  for(let t=0;t*DT<total;t++) await step(poseAt(tr,t*DT));
  for(let t=0;t<50;t++) await step(null);          // must still be there after
  const x=data.qpos[D.freeQpos], z=data.qpos[D.freeQpos+2];
  const up = projectedGravity(quat())[2] < -0.90;
  let feetUp = 0;
  for (let g=0; g<model.ngeom; g++){
    const n = model.geom(g).name || '';
    if (!/foot_collision|sole/.test(n)) continue;
    if (data.geom_xpos[g*3+2] > h - 0.005 && data.geom_xpos[g*3] > 0.05) feetUp++;
  }
  // Standing ON the flight: past the first riser, both feet at or above the
  // first tread, trunk a standing height above it, and upright. `above` is
  // measured from the first tread because that is the step being attempted;
  // a duck that made it higher passes this too, which is the correct way round.
  const onTop = up && x > 0.12 && (z - h) > 0.095 && feetUp >= 2;
  return { onTop, x, z, above: z-h, feetUp, up };
}

export { attempt, replay, trackOf, poseAt, B, randP, jitter };
