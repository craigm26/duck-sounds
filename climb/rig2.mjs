// An INSTRUMENTED copy of sim/climb_lib.mjs's attempt().
//
// Why a copy and not an import: attempt() returns only terminal state
// ({onTop,x,z,above,feetUp,up} — sim/climb_lib.mjs:151). Strategy B has to be
// graded on things that only exist DURING the episode — was the head actually
// planted on the tread, how far did the trunk yaw while it was planted, did the
// outboard sole ever get over the tread. Those are per-tick facts. sim/ is
// off-limits to edit (another run is refactoring it), so this file reproduces
// attempt()'s staging BYTE FOR BYTE (same cfg, same spawn, same 25-tick settle,
// same +0.8 s tail, same 50-tick hold, same criterion at line 150) and adds a
// recorder. Any divergence from climb_lib is a bug in this file.
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/<script>.mjs
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
import { findStairJoints, layoutStairs, STAIR_Y } from '../site/stairs.js';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json','utf8'));
export const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const model = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model), ADDR = findStairJoints(model);
let GYRO = 0;
for (let i=0;i<model.nsensor;i++) if (model.sensor(i).name==='imu_ang_vel') GYRO = model.sensor(i).adr;
const stand = await ort.InferenceSession.create('./BEST_alpha_stand.onnx');
const DT = 1/C.tickHz;

const bodyId = n => { for (let b=0;b<model.nbody;b++) if (model.body(b).name===n) return b; return -1; };
const JAWB = bodyId('jaw_soft');
const JAW = []; for (let g=0;g<model.ngeom;g++) if (model.geom_bodyid[g]===JAWB && !(model.geom_contype[g]===0&&model.geom_conaffinity[g]===0)) JAW.push(g);
const STEPG = []; for (let g=0;g<model.ngeom;g++) if (/^step\d+_geom$/.test(model.geom(g).name||'')) STEPG.push(g);
let STEP0=-1; for (let g=0;g<model.ngeom;g++) if (model.geom(g).name==='step0_geom') STEP0=g;
let LFOOT=-1, RFOOT=-1;
for (let g=0;g<model.ngeom;g++){ const n=model.geom(g).name||'';
  if (n==='left_foot_collision') LFOOT=g; if (n==='right_foot_collision') RFOOT=g; }
// the geoms the terminal feetUp test walks, resolved ONCE (climb_lib does it
// per episode; here it is per process, same set)
const FEET = []; for (let g=0;g<model.ngeom;g++) if (/foot_collision|sole/.test(model.geom(g).name||'')) FEET.push(g);
export const GEOMS = { JAW, STEPG, STEP0, LFOOT, RFOOT, FEET, STAIR_Y };

const quat = () => [data.qpos[D.freeQpos+3],data.qpos[D.freeQpos+4],data.qpos[D.freeQpos+5],data.qpos[D.freeQpos+6]];
const yawOf = ([w,x,y,z]) => Math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z));

export function poseAt(tr,time){
  if (time<=0) return HOME.slice();
  let pt=0,pp=HOME;
  for(const f of tr){ if(time<=f.t){const u=(time-pt)/Math.max(f.t-pt,1e-9),s=u*u*(3-2*u);
    return f.pose.map((v,k)=>pp[k]+(v-pp[k])*s);} pt=f.t; pp=f.pose; }
  return tr[tr.length-1].pose.slice();
}

/** Same episode as sim/climb_lib.mjs attempt(), plus a per-tick record (rig2: + left sole + peak feetUp). */
export async function run(track, opts, h){
  const cfg = { count: 4, rise: h, run: 0.28, start: 0.12 };
  mj.mj_resetData(model,data);
  layoutStairs(data, ADDR, cfg);
  data.qpos[D.freeQpos] = 0.12 - 0.07 - opts.gap;
  data.qpos[D.freeQpos+1] = STAIR_Y + (opts.side || 0);
  data.qpos[D.freeQpos+2]=0.12; data.qpos[D.freeQpos+3]=1;
  for(let i=0;i<14;i++){data.qpos[D.qpos[i]]=HOME[i];data.ctrl[i]=HOME[i];}
  mj.mj_forward(model,data);
  const tr = track.map(f => ({ t: f.t, pose: f.pose.slice() }));
  let la = new Array(14).fill(0);
  const cmd = command({ vx: opts.approach || 0 });

  const R = { headTicks:0, trackTicks:0, upTicks:0, satCount:0, ctrlCount:0,
              maxX:-1e9, minX:1e9, maxZ:-1e9, maxY:-1e9, footOver:0,
              yawWhilePlanted:0, riserTicks:0, footRiserTicks:0, headEverTouched:false,
              maxSoleZ:-1e9, maxSoleX:-1e9, maxLSoleZ:-1e9, maxLSoleX:-1e9,
              feetUpMax:0, bothFootOver:0 };

  const step = async (off, record) => {
    layoutStairs(data, ADDR, cfg);
    const q=quat(); const jp=[],jv=[];
    for(let k=0;k<14;k++){jp.push(data.qpos[D.qpos[k]]);jv.push(data.qvel[D.dof[k]]);}
    const obs=buildObs([data.sensordata[GYRO],data.sensordata[GYRO+1],data.sensordata[GYRO+2]],projectedGravity(q),jp,jv,la,cmd);
    const r=await stand.run({obs:new ort.Tensor('float32',obs,[1,61])});
    la=Array.from(r.actions.data);
    for(let k=0;k<14;k++){
      const v=HOME[k]+la[k]+(off?(off[k]-HOME[k])*opts.blend:0);
      const c=Math.min(Math.max(v,LO[k]),HI[k]);
      data.ctrl[k]=c;
      if (record){ R.ctrlCount++; if (c<=LO[k]+1e-9 || c>=HI[k]-1e-9) R.satCount++; }
    }
    for(let s=0;s<4;s++) mj.mj_step(model,data);
    if (!record) return;
    R.trackTicks++;
    const x=data.qpos[D.freeQpos], y=data.qpos[D.freeQpos+1], z=data.qpos[D.freeQpos+2];
    if (x>R.maxX) R.maxX=x; if (x<R.minX) R.minX=x;
    if (z>R.maxZ) R.maxZ=z; if (y>R.maxY) R.maxY=y;
    const qq=quat();
    if (projectedGravity(qq)[2] < -0.90) R.upTicks++;
    // head planted: any jaw geom in contact with any step geom
    // CONTACT DETECTION IS mj_geomDistance, NOT data.contact.
    // data.contact.get(i) returns an embind object that leaks the WASM heap
    // even when .delete() is called on it: measured, a bare loop of
    // `const c = d.contact.get(0); c.delete();` aborts the module with
    // "Cannot enlarge memory, requested 2147487744 bytes" somewhere under
    // 200 000 iterations (climb/parity.mjs's sibling probe). A search doing
    // ~230 ticks x ~35 contacts an episode hits that in a couple of hundred
    // episodes. mj_geomDistance returns a plain number, leaks nothing, and
    // costs 3.4 us a call (200 000 in 671 ms on this Pi), so head and foot
    // contact are distances to step0_geom instead. Threshold 3 mm: an
    // unsettled foot standing on the floor reads 2.82 mm.
    let head=false, footRiser=false;
    for (const g of JAW) if (mj.mj_geomDistance(model,data,g,STEP0,0.05,null) < 0.003) { head=true; break; }
    for (const g of [LFOOT,RFOOT]) {
      if (data.geom_xpos[g*3+2] >= h - 0.005) continue;          // at/above the tread is not the riser FACE
      if (mj.mj_geomDistance(model,data,g,STEP0,0.05,null) < 0.003) { footRiser=true; break; }
    }
    if (head){ R.headTicks++; R.headEverTouched=true;
      const yw=Math.abs(yawOf(qq)); if (yw>R.yawWhilePlanted) R.yawWhilePlanted=yw; }
    if (footRiser) R.footRiserTicks++;
    // outboard (right) sole over the tread
    const sx=data.geom_xpos[RFOOT*3], sz=data.geom_xpos[RFOOT*3+2];
    if (sx>R.maxSoleX) R.maxSoleX=sx; if (sz>R.maxSoleZ) R.maxSoleZ=sz;
    let fo = (sx>0.12 && sz>h-0.005) ? 1 : Math.max(0, Math.min(1, (sx-0.05)/0.07))*0.9;
    if (fo>R.footOver) R.footOver=fo;
    // rig2 additions: the inboard (left) sole, and the criterion's OWN feetUp
    // test evaluated per tick. feetUpMax is the strongest shaping signal there
    // is — it is literally the terminal test, graded while the episode runs.
    const lx=data.geom_xpos[LFOOT*3], lz=data.geom_xpos[LFOOT*3+2];
    if (lx>R.maxLSoleX) R.maxLSoleX=lx; if (lz>R.maxLSoleZ) R.maxLSoleZ=lz;
    let fu=0; for (const g of FEET) if (data.geom_xpos[g*3+2] > h-0.005 && data.geom_xpos[g*3] > 0.05) fu++;
    if (fu>R.feetUpMax) R.feetUpMax=fu;
    const lo = (lx>0.12 && lz>h-0.005) ? 1 : Math.max(0, Math.min(1, (lx-0.05)/0.07))*0.9;
    const bo = Math.min(fo,lo);
    if (bo>R.bothFootOver) R.bothFootOver=bo;
  };

  for(let t=0;t<25;t++) await step(null,false);
  const x0 = data.qpos[D.freeQpos];
  const total = tr[tr.length-1].t + 0.8;
  for(let t=0;t*DT<total;t++) await step(poseAt(tr,t*DT),true);
  for(let t=0;t<50;t++) await step(null,true);

  const x=data.qpos[D.freeQpos], z=data.qpos[D.freeQpos+2];
  const up = projectedGravity(quat())[2] < -0.90;
  let feetUp = 0;
  for (const g of FEET) if (data.geom_xpos[g*3+2] > h - 0.005 && data.geom_xpos[g*3] > 0.05) feetUp++;
  const onTop = up && x > 0.12 && (z - h) > 0.095 && feetUp >= 2;
  return { onTop, x, z, endY: data.qpos[D.freeQpos+1], above: z-h, feetUp, up, x0,
    headFrac: R.headTicks/Math.max(R.trackTicks,1),
    upFrac:   R.upTicks/Math.max(R.trackTicks,1),
    satFrac:  R.satCount/Math.max(R.ctrlCount,1),
    riserFrac:R.footRiserTicks/Math.max(R.trackTicks,1),
    maxX:R.maxX, minX:R.minX, maxZ:R.maxZ, maxY:R.maxY,
    footOver:R.footOver, bothFootOver:R.bothFootOver, feetUpMax:R.feetUpMax,
    maxLSoleZ:R.maxLSoleZ, maxLSoleX:R.maxLSoleX, yawPlanted:R.yawWhilePlanted,
    headTouched:R.headEverTouched, maxSoleZ:R.maxSoleZ, maxSoleX:R.maxSoleX };
}
