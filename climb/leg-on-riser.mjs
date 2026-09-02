// A foot pressed on a VERTICAL riser face: what can that leg do?
//
// Fixes two things about the first pass in wholebody-physics.mjs: it picked poses
// by shin angle alone and got a foot BEHIND the trunk at 60 deg, and it never
// asked the question the strategies actually need -- "foot at height z on a face
// x mm in front of the hip: how hard can it press, and how much can it lift?"
//
// Also dumps dof damping / frictionloss / armature and the actuator gain, because
// kp = 0.55 N.m/rad with no damping term means a joint SAGS under load and dry
// friction is doing part of the holding.
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
const TAU=0.6405;
let MASS=0; for(let b=1;b<m.nbody;b++){const n=m.body(b).name||'';if(/step|ball|block|cone|wall|floor/.test(n))continue;MASS+=m.body_mass[b];}
const W=MASS*9.81;

console.log('=== JOINT DOF PROPERTIES (why a pose sags) ===');
console.log('  joint            damping   frictionloss   armature   stiffness   kp(gain)  tau_max');
for(let j=0;j<m.njnt;j++){
  const n=m.jnt(j).name||''; if(/step/.test(n)||!n) continue;
  const dof=m.jnt_dofadr[j];
  console.log(`  ${n.padEnd(16)} ${m.dof_damping[dof].toFixed(5).padStart(8)} ${m.dof_frictionloss[dof].toFixed(5).padStart(13)} ${m.dof_armature[dof].toFixed(6).padStart(10)} ${m.jnt_stiffness[j].toFixed(3).padStart(11)}      0.55    ${TAU}`);
}
console.log(`\n  actuator is position-servo gain ${m.actuator_gainprm[0].toFixed(3)}, bias kp ${m.actuator_biasprm[1].toFixed(3)}, kv ${m.actuator_biasprm[2].toFixed(3)}`);
console.log(`  -> steady torque = 0.55 * (ctrl - q), saturating at ${TAU} N.m, i.e. at ${(TAU/0.55).toFixed(3)} rad = ${(TAU/0.55*180/Math.PI).toFixed(0)} deg of tracking error.`);
console.log(`  -> A COMMANDED KEYFRAME IS NOT A REACHED ANGLE. To pull ${TAU} N.m out of a joint you must`);
console.log(`     command it ${(TAU/0.55*180/Math.PI).toFixed(0)} deg past where you want it. Search spaces must allow that overshoot.`);

function setPose(q, zTrunk=0.12, pitch=0){
  mj.mj_resetData(m,d);
  layoutStairs(d, ADDR, {count:0,rise:0,run:0.28,start:0.12});
  const a=pitch*Math.PI/180;
  d.qpos[D.freeQpos]=0; d.qpos[D.freeQpos+1]=STAIR_Y; d.qpos[D.freeQpos+2]=zTrunk;
  d.qpos[D.freeQpos+3]=Math.cos(a/2); d.qpos[D.freeQpos+4]=0; d.qpos[D.freeQpos+5]=Math.sin(a/2); d.qpos[D.freeQpos+6]=0;
  for(let i=0;i<14;i++) d.qpos[D.qpos[i]]=q[i];
  mj.mj_forward(m,d);
}
const LFOOT=29, RFOOT=35, LHP=2,LK=3,LA=4, LHY=0, LHR=1;
const com=()=>{let cx=0,cz=0,mm=0;for(let b=1;b<m.nbody;b++){const n=m.body(b).name||'';if(/step|ball|block|cone|wall|floor/.test(n))continue;mm+=m.body_mass[b];cx+=m.body_mass[b]*d.xipos[b*3];cz+=m.body_mass[b]*d.xipos[b*3+2];}return [cx/mm,cz/mm];};

// Jacobian of the left foot geom wrt the five LEFT leg joints, finite differenced.
function footJac(q,zTrunk){
  const h=1e-5; setPose(q,zTrunk);
  const p0=[d.geom_xpos[LFOOT*3], d.geom_xpos[LFOOT*3+1], d.geom_xpos[LFOOT*3+2]];
  const rows=[];
  for(const k of [LHY,LHR,LHP,LK,LA]){
    const qq=q.slice(); qq[k]+=h; setPose(qq,zTrunk);
    rows.push([(d.geom_xpos[LFOOT*3]-p0[0])/h,(d.geom_xpos[LFOOT*3+1]-p0[1])/h,(d.geom_xpos[LFOOT*3+2]-p0[2])/h]);
  }
  return {rows,p0};
}
const feas=(rows,f)=>rows.every(r=>Math.abs(r[0]*f[0]+r[1]*f[1]+r[2]*f[2])<=TAU+1e-12);
const scale=(rows,dir)=>{let lo=0,hi=3000;for(let i=0;i<60;i++){const mid=(lo+hi)/2;if(feas(rows,dir.map(v=>v*mid)))lo=mid;else hi=mid;}return lo;};

console.log('\n=== LEFT FOOT ON A VERTICAL FACE, trunk upright at z=120 mm ===');
console.log('  For each target foot height z_foot, the pose (hip_pitch,knee,ankle) that puts the foot');
console.log('  furthest FORWARD at that height, then the force the leg can react there.');
console.log('  z_foot  foot_x  hip_p  knee  ankle | Fx_push(N)  Fz_lift(N)  Fz at Fx=Fz/mu=Fz (mu=1)');
for(const zt of [0.02,0.04,0.06,0.09,0.12,0.15]){
  let best=null;
  for(let hip=-1.55;hip<=1.55;hip+=0.05) for(let knee=-1.55;knee<=1.55;knee+=0.05) for(let ank=-1.5;ank<=1.5;ank+=0.25){
    const q=HOME.slice(); q[LHP]=hip;q[LK]=knee;q[LA]=ank; setPose(q,0.12);
    const z=d.geom_xpos[LFOOT*3+2], x=d.geom_xpos[LFOOT*3];
    if(Math.abs(z-zt)>0.006) continue;
    if(!best||x>best.x) best={x,z,hip,knee,ank,q};
  }
  if(!best){console.log(`  ${(zt*1000).toFixed(0).padStart(6)}  -- no pose within 6 mm of that height --`);continue;}
  const {rows}=footJac(best.q,0.12);
  const push=scale(rows,[1,0,0]), lift=scale(rows,[0,0,1]);
  const dir=[1/Math.SQRT2,0,1/Math.SQRT2]; const both=scale(rows,dir)*dir[2];
  console.log(`  ${(zt*1000).toFixed(0).padStart(6)}  ${(best.x*1000).toFixed(0).padStart(6)}  ${best.hip.toFixed(2).padStart(5)} ${best.knee.toFixed(2).padStart(5)} ${best.ank.toFixed(2).padStart(5)} |`
   +` ${push.toFixed(2).padStart(9)}  ${lift.toFixed(2).padStart(9)}  ${both.toFixed(2).padStart(12)}`);
}
console.log(`  robot weight ${W.toFixed(2)} N; TWO feet wedged need ${(W/2).toFixed(2)} N of friction each, i.e. ${(W/2).toFixed(2)} N of normal each at mu=1.`);

console.log('\n=== TRUNK PITCHED FORWARD (+theta about +y is nose-down here?) -- both signs, CoM vs riser at x=120 mm ===');
console.log('  theta  CoM_x  CoM_z  jaw_x  jaw_z  lfoot_x  lfoot_z   (mm)');
for(const th of [-90,-60,-45,-30,-15,0,15,30,45,60,90]){
  const q=HOME.slice(); q[5]=LO[5]; q[6]=-0.6;
  setPose(q,0.12,th);
  const [cx,cz]=com();
  console.log(`  ${String(th).padStart(5)} ${(cx*1000).toFixed(1).padStart(6)} ${(cz*1000).toFixed(1).padStart(6)} ${(d.xpos[30*3]*1000).toFixed(1).padStart(6)} ${(d.xpos[30*3+2]*1000).toFixed(1).padStart(6)} ${(d.geom_xpos[LFOOT*3]*1000).toFixed(1).padStart(8)} ${(d.geom_xpos[LFOOT*3+2]*1000).toFixed(1).padStart(8)}`);
}
