// Two things the twist strategy stands or falls on, plus one premise check.
//  A. inertias: can a head_yaw whip actually counter-rotate the trunk?
//  B. lateral budget: hip_roll is +/-22 deg -- is that enough to get the CoM
//     over one foot so the other hip can be lifted over the edge?
//  C. PREMISE CHECK: does a jaw-vs-step0 contact ever actually appear? The
//     bitmasks say 5 & 4 != 0, but bitmasks have been wrong about this family
//     before, so this drives a real episode and counts contacts.
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
import { findStairJoints, layoutStairs, STAIR_Y } from '../site/stairs.js';
const C = JSON.parse(fs.readFileSync('duckkit-constants.json','utf8'));
const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const m = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const d = new mj.MjData(m);
const D = findDuckJoints(m), ADDR = findStairJoints(m);
const bid=n=>{for(let b=0;b<m.nbody;b++) if(m.body(b).name===n) return b; return -1;};

console.log('=== INERTIAS (body frame diagonal, kg.m^2) ===');
for(const n of ['trunk_base','jaw_soft','neck','upper_leg_left','ankle_left']){
  const b=bid(n);
  console.log(`  ${n.padEnd(16)} mass ${(m.body_mass[b]*1000).toFixed(1)} g   inertia [${[0,1,2].map(k=>m.body_inertia[b*3+k].toExponential(2)).join(', ')}]`);
}
{ // head yaw whip: I_head about the head_yaw axis vs I_rest about the trunk's vertical
  const jb=bid('jaw_soft'), tb=bid('trunk_base');
  const Ih = m.body_inertia[jb*3+2];
  let Irest=0, mrest=0;
  for(let b=1;b<m.nbody;b++){const n=m.body(b).name||''; if(/step|ball|block|cone|wall|floor/.test(n)||n==='jaw_soft')continue;
    Irest += m.body_inertia[b*3+2] + m.body_mass[b]*0.05*0.05; mrest+=m.body_mass[b];}
  console.log(`\n  head yaw inertia about its own z ~ ${Ih.toExponential(2)} kg.m^2`);
  console.log(`  rest-of-robot yaw inertia (crude, +m*(50mm)^2 offset) ~ ${Irest.toExponential(2)} kg.m^2, mass ${(mrest*1000).toFixed(0)} g`);
  console.log(`  a head_yaw sweep of ${(2*2.967).toFixed(2)} rad (limit to limit) counter-rotates the body by about`);
  console.log(`    ${(2*2.967*Ih/Irest).toFixed(2)} rad = ${(2*2.967*Ih/Irest*180/Math.PI).toFixed(0)} deg IF the feet are free (airborne / low friction).`);
  console.log(`  head_yaw torque cap 0.6405 N.m on ${Ih.toExponential(2)} kg.m^2 -> ${(0.6405/Ih).toFixed(0)} rad/s^2;`);
  console.log(`    limit-to-limit (${(2*2.967).toFixed(2)} rad) in ${(Math.sqrt(2*(2*2.967)/(0.6405/Ih))*1000).toFixed(0)} ms if torque-limited only.`);
}

function setPose(q,zT=0.12,pitch=0,roll=0){
  mj.mj_resetData(m,d); layoutStairs(d,ADDR,{count:0,rise:0,run:0.28,start:0.12});
  const a=pitch*Math.PI/180, r=roll*Math.PI/180;
  const qy=[Math.cos(a/2),0,Math.sin(a/2),0], qx=[Math.cos(r/2),Math.sin(r/2),0,0];
  const w=qy[0]*qx[0]-qy[1]*qx[1]-qy[2]*qx[2]-qy[3]*qx[3];
  const x=qy[0]*qx[1]+qy[1]*qx[0]+qy[2]*qx[3]-qy[3]*qx[2];
  const y=qy[0]*qx[2]-qy[1]*qx[3]+qy[2]*qx[0]+qy[3]*qx[1];
  const z=qy[0]*qx[3]+qy[1]*qx[2]-qy[2]*qx[1]+qy[3]*qx[0];
  d.qpos[D.freeQpos]=0; d.qpos[D.freeQpos+1]=STAIR_Y; d.qpos[D.freeQpos+2]=zT;
  d.qpos[D.freeQpos+3]=w;d.qpos[D.freeQpos+4]=x;d.qpos[D.freeQpos+5]=y;d.qpos[D.freeQpos+6]=z;
  for(let i=0;i<14;i++) d.qpos[D.qpos[i]]=q[i];
  mj.mj_forward(m,d);
}
const LFOOT=29,RFOOT=35;
const comY=()=>{let c=0,mm=0;for(let b=1;b<m.nbody;b++){const n=m.body(b).name||'';if(/step|ball|block|cone|wall|floor/.test(n))continue;mm+=m.body_mass[b];c+=m.body_mass[b]*d.xipos[b*3+1];}return c/mm;};
console.log('\n=== LATERAL BUDGET: can the CoM get over ONE foot? ===');
setPose(HOME);
console.log(`  HOME: left foot y=${((d.geom_xpos[LFOOT*3+1]-STAIR_Y)*1000).toFixed(1)} mm, right foot y=${((d.geom_xpos[RFOOT*3+1]-STAIR_Y)*1000).toFixed(1)} mm  -> stance width ${((d.geom_xpos[LFOOT*3+1]-d.geom_xpos[RFOOT*3+1])*1000).toFixed(1)} mm`);
console.log('  hip_roll(both, rad)  CoM_y offset(mm)   left foot y(mm)  right foot y(mm)   CoM over which foot?');
for(const r of [-0.297,-0.2,-0.1,0,0.1,0.2,0.297]){
  const q=HOME.slice(); q[1]=HOME[1]+r; q[11]=HOME[11]-r;   // WORLD-symmetric roll: the joints are mirrored
  setPose(q);
  const ly=(d.geom_xpos[LFOOT*3+1]-STAIR_Y)*1000, ry=(d.geom_xpos[RFOOT*3+1]-STAIR_Y)*1000;
  const mean=(ly+ry)/2;
  console.log(`  ${r.toFixed(3).padStart(15)}  ${mean.toFixed(1).padStart(16)}  ${ly.toFixed(1).padStart(10)}  ${ry.toFixed(1).padStart(11)}  ${(-mean).toFixed(1).padStart(22)}`);
}
console.log('  Note: hip_roll can never stall (range 0.768 rad < the 1.165 rad of error a 0.6405 N.m needs);');
console.log(`  max hip_roll torque limit-to-limit = 0.55*0.768 = ${(0.55*0.768).toFixed(3)} N.m. Same for hip_yaw: 0.55*0.960 = ${(0.55*0.96).toFixed(3)} N.m.`);

// ---------------------------------------------------- C. does the jaw touch?
// SUPERSEDED, and left out on purpose. The version of this section that ran here
// called mj_geomDistance and data.contact.get() inside the episode loop; BOTH leak
// in this wasm build (proved: 20000 mj_geomDistance calls abort the module at its
// 2 GB heap cap, and contact.get(i) does the same at ~50k even with .delete()).
// The contact evidence now lives in climb/contact-dump.mjs (short run, counts every
// geom pair) and the reach sweep in climb/headplant-probe.mjs (geometric proxy, no
// leaking call). Result of that pair: driving the neck alone with alpha_stand holding
// the trunk upright gives ZERO jaw<->step contacts and the furthest jaw geom at
// x = 91 mm against a riser at 120 mm; adding a hip lean of 0.9 rad AND an ankle
// offset of -0.5 rad gives 26.8 deg of trunk pitch, the jaw at x = 161 mm, and
// 16-22 jaw<->step contact ticks.
console.log('\n=== PREMISE CHECK: moved to climb/contact-dump.mjs + climb/headplant-probe.mjs (see comment) ===');
