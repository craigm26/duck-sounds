// How far sideways can this robot move its CoM, and by which joint?
// twist-budget.mjs showed hip_roll alone buys only ~13 mm against a half-stance
// of 40.7 mm, so a plain weight-shift-and-step-up is arithmetically dead. This
// adds the head: 189 g of the 737 g total, on a +/-170 deg yaw joint.
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
function setPose(q){ mj.mj_resetData(m,d); layoutStairs(d,ADDR,{count:0,rise:0,run:0.28,start:0.12});
  d.qpos[D.freeQpos]=0; d.qpos[D.freeQpos+1]=STAIR_Y; d.qpos[D.freeQpos+2]=0.12; d.qpos[D.freeQpos+3]=1;
  for(let i=0;i<14;i++) d.qpos[D.qpos[i]]=q[i]; mj.mj_forward(m,d); }
const com=()=>{let cy=0,cx=0,mm=0;for(let b=1;b<m.nbody;b++){const n=m.body(b).name||'';if(/step|ball|block|cone|wall|floor/.test(n))continue;mm+=m.body_mass[b];cy+=m.body_mass[b]*d.xipos[b*3+1];cx+=m.body_mass[b]*d.xipos[b*3];}return [cx/mm,(cy/mm-STAIR_Y)];};
setPose(HOME); const [,y0]=com();
console.log('CoM lateral authority, trunk held upright and fixed (mm of CoM y relative to HOME):');
console.log('  half-stance width (foot y) = 40.7 mm -- that is the number to beat to stand on one foot.\n');
const cases = {
 'head_yaw = +1.571 (head swung left)': q=>{q[7]=1.571;},
 'head_yaw = +2.967 (limit)':           q=>{q[7]=2.967;},
 'head_yaw = -2.967 (limit, other way)':q=>{q[7]=-2.967;},
 'head_roll = +0.436 (limit)':          q=>{q[8]=0.436;},
 'head_yaw +1.571 AND neck fully forward (np=-1.571)': q=>{q[7]=1.571;q[5]=-1.571;},
 'head_yaw +1.571, np=-1.571, hp=-0.6': q=>{q[7]=1.571;q[5]=-1.571;q[6]=-0.6;},
 'hip_roll world-symmetric +0.297':     q=>{q[1]=HOME[1]+0.297;q[11]=HOME[11]-0.297;},
 'EVERYTHING: hip_roll +0.297, head_yaw +1.571, np -1.571, hp -0.6, head_roll +0.436':
    q=>{q[1]=HOME[1]+0.297;q[11]=HOME[11]-0.297;q[7]=1.571;q[5]=-1.571;q[6]=-0.6;q[8]=0.436;},
};
for(const [label,f] of Object.entries(cases)){
  const q=HOME.slice(); f(q); setPose(q); const [cx,cy]=com();
  console.log(`  ${label.padEnd(54)} CoM dy = ${((cy-y0)*1000).toFixed(1).padStart(6)} mm   CoM x = ${(cx*1000).toFixed(1).padStart(6)} mm`);
}
console.log('\n  Note: with the trunk PINNED these read as CoM motion; with the feet planted the same');
console.log('  joint motion moves the trunk the other way, so the usable weight shift is the same number.');
