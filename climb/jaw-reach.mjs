// EXACT jaw reach, from the collision meshes' own vertices -- not rbound.
// Plus: the servo-headroom table, because a commanded angle is clamped to the
// joint range and torque = 0.55*(clamped_cmd - q). If the pose you want to HOLD
// is already at the joint limit there is no error left and therefore no torque.
import load from 'mujoco';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
import { findStairJoints, layoutStairs, STAIR_Y } from '../site/stairs.js';
const C = JSON.parse(fs.readFileSync('duckkit-constants.json','utf8'));
const { HOME, LO, HI, findDuckJoints } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const m = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const d = new mj.MjData(m);
const D = findDuckJoints(m), ADDR = findStairJoints(m);
const N=['left_hip_yaw','left_hip_roll','left_hip_pitch','left_knee','left_ankle','neck_pitch','head_pitch','head_yaw','head_roll','right_hip_yaw','right_hip_roll','right_hip_pitch','right_knee','right_ankle'];
console.log('=== CONTROL CLAMP (duckloop LO/HI) vs joint range, and the torque headroom ===');
console.log('  joint            LO       HI      HOME    tau at cmd=LO from q=HOME   tau at cmd=HI from q=HOME');
for(let i=0;i<14;i++){
  const tl=Math.max(-0.6405,Math.min(0.6405,0.55*(LO[i]-HOME[i])));
  const th=Math.max(-0.6405,Math.min(0.6405,0.55*(HI[i]-HOME[i])));
  console.log(`  ${N[i].padEnd(16)} ${LO[i].toFixed(3).padStart(7)} ${HI[i].toFixed(3).padStart(7)} ${HOME[i].toFixed(3).padStart(7)}   ${tl.toFixed(4).padStart(10)}                 ${th.toFixed(4).padStart(10)}`);
}
console.log('  (0.55 N.m/rad, saturating at 0.6405; a joint needs 1.165 rad = 67 deg of error to saturate,');
console.log('   so ONLY joints whose range is wider than 67 deg either side of the held angle can ever stall.)');

const JAWG=[30,31,32];
console.log('\n=== JAW MESH EXTENTS (body frame) ===');
for(const g of JAWG){
  const mid=m.geom_dataid[g], adr=m.mesh_vertadr[mid], num=m.mesh_vertnum[mid];
  let lo=[1e9,1e9,1e9], hi=[-1e9,-1e9,-1e9];
  for(let v=0;v<num;v++) for(let k=0;k<3;k++){ const val=m.mesh_vert[(adr+v)*3+k]; if(val<lo[k])lo[k]=val; if(val>hi[k])hi[k]=val; }
  console.log(`  geom ${g}: ${num} verts, local bbox x[${(lo[0]*1000).toFixed(1)},${(hi[0]*1000).toFixed(1)}] y[${(lo[1]*1000).toFixed(1)},${(hi[1]*1000).toFixed(1)}] z[${(lo[2]*1000).toFixed(1)},${(hi[2]*1000).toFixed(1)}] mm`);
}
function setPose(q,zT,pitch){
  mj.mj_resetData(m,d); layoutStairs(d,ADDR,{count:0,rise:0,run:0.28,start:0.12});
  const a=pitch*Math.PI/180;
  d.qpos[D.freeQpos]=0; d.qpos[D.freeQpos+1]=STAIR_Y; d.qpos[D.freeQpos+2]=zT;
  d.qpos[D.freeQpos+3]=Math.cos(a/2); d.qpos[D.freeQpos+4]=0; d.qpos[D.freeQpos+5]=Math.sin(a/2); d.qpos[D.freeQpos+6]=0;
  for(let i=0;i<14;i++) d.qpos[D.qpos[i]]=q[i];
  mj.mj_forward(m,d);
}
// world-space jaw hull points
function jawPts(){
  const out=[];
  for(const g of JAWG){
    const mid=m.geom_dataid[g], adr=m.mesh_vertadr[mid], num=m.mesh_vertnum[mid];
    const p=[d.geom_xpos[g*3],d.geom_xpos[g*3+1],d.geom_xpos[g*3+2]];
    const R=[]; for(let k=0;k<9;k++) R.push(d.geom_xmat[g*9+k]);
    const stride=Math.max(1,Math.floor(num/300));
    for(let v=0;v<num;v+=stride){
      const lx=m.mesh_vert[(adr+v)*3], ly=m.mesh_vert[(adr+v)*3+1], lz=m.mesh_vert[(adr+v)*3+2];
      out.push([p[0]+R[0]*lx+R[1]*ly+R[2]*lz, p[1]+R[3]*lx+R[4]*ly+R[5]*lz, p[2]+R[6]*lx+R[7]*ly+R[8]*lz]);
    }
  }
  return out;
}
const ZT=0.1162;   // measured standing trunk z under alpha_stand (climb/feetup-probe.mjs)
const NP=5,HP=6;
console.log(`\n=== JAW SURFACE REACH, trunk origin at x=0, z=${(ZT*1000).toFixed(0)} mm (the measured standing height) ===`);
console.log('  For every tread height h, the furthest FORWARD any jaw-mesh vertex gets while AT OR ABOVE h.');
console.log('  Search over neck_pitch x head_pitch x trunk pitch. dx is relative to the TRUNK ORIGIN.');
console.log('   h(mm)  max jaw dx(mm)   at z(mm)   neck_pitch  head_pitch  trunk_pitch(deg)');
const cand=[];
for(let th=-40; th<=75; th+=5)
 for(let np=LO[NP]; np<=HI[NP]+1e-9; np+=0.12)
  for(let hp=LO[HP]; hp<=HI[HP]+1e-9; hp+=0.16){
    const q=HOME.slice(); q[NP]=np; q[HP]=hp; setPose(q,ZT,th);
    for(const p of jawPts()) cand.push([p[0],p[2],np,hp,th]);
  }
for(const hmm of [20,40,60,90,120,180]){
  const h=hmm/1000;
  const ok=cand.filter(c=>c[1]>=h);
  if(!ok.length){ console.log(`  ${String(hmm).padStart(6)}   -- the jaw never gets that high from this trunk height --`); continue; }
  const b=ok.reduce((a,c)=>c[0]>a[0]?c:a);
  console.log(`  ${String(hmm).padStart(6)}  ${(b[0]*1000).toFixed(1).padStart(13)}  ${(b[1]*1000).toFixed(1).padStart(9)}   ${b[2].toFixed(2).padStart(10)}  ${b[3].toFixed(2).padStart(10)}  ${String(b[4]).padStart(16)}`);
}
const top=cand.reduce((a,c)=>c[1]>a[1]?c:a);
console.log(`  highest any jaw vertex reaches: z=${(top[1]*1000).toFixed(1)} mm at dx=${(top[0]*1000).toFixed(1)} mm (np ${top[2].toFixed(2)}, hp ${top[3].toFixed(2)}, trunk pitch ${top[4]} deg)`);
console.log(`\n  attempt() starts the trunk at x = 0.12 - 0.07 - gap, gap in [0.02,0.12] -> trunk x in [-70, +30] mm.`);
console.log(`  The first riser face is at x = 120 mm. So the jaw must reach dx >= ${'120 - trunk_x'} = 90..190 mm to touch it,`);
console.log(`  and dx >= 120 - trunk_x + a few mm ONTO the tread.`);
for(const hmm of [20,40,60,90,120,180]){
  const h=hmm/1000; const ok=cand.filter(c=>c[1]>=h);
  if(!ok.length) continue;
  const b=ok.reduce((a,c)=>c[0]>a[0]?c:a);
  const needTrunkX = 0.120 - b[0];
  console.log(`   rise ${String(hmm).padStart(3)} mm: jaw clears the edge only if trunk x >= ${(needTrunkX*1000).toFixed(1)} mm  -> gap <= ${((0.05-needTrunkX)*1000).toFixed(1)} mm  ${needTrunkX<=0.03?'FEASIBLE from the start band':'NEEDS THE DUCK TO WALK IN FIRST'}`);
}
