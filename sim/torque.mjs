// What can this robot actually lift? Straight from the model.
import load from 'mujoco';
import fs from 'node:fs';
const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const m = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const data = new mj.MjData(m);
mj.mj_forward(m, data);

console.log('=== actuator limits (N.m at the joint) ===');
for (let a = 0; a < m.nu; a++) {
  const name = m.actuator(a).name;
  const lo = m.actuator_forcerange[a*2], hi = m.actuator_forcerange[a*2+1];
  const gear = m.actuator_gear[a*6];
  const kp = m.actuator_gainprm[a*10];
  console.log(`  ${name.padEnd(16)} force ${lo.toFixed(2)} .. ${hi.toFixed(2)}   gear ${gear.toFixed(2)}   kp ${kp.toFixed(2)}`);
}

console.log('\n=== masses (kg) ===');
let total = 0;
const parts = [];
for (let b = 1; b < m.nbody; b++) {
  const name = m.body(b).name;
  const mass = m.body_mass[b];
  if (/step|ball|block|cone/.test(name)) continue;
  total += mass;
  parts.push([name, mass]);
}
parts.sort((a,b) => b[1]-a[1]);
for (const [n, kg] of parts.slice(0, 6)) console.log(`  ${n.padEnd(18)} ${(kg*1000).toFixed(0)} g`);
console.log(`  ${'TOTAL ROBOT'.padEnd(18)} ${(total*1000).toFixed(0)} g  (weight ${(total*9.81).toFixed(2)} N)`);

// What a head torque can hold: the neck is a lever, so the useful number is the
// force at the beak, not the torque at the joint.
const neck = [...Array(m.nu).keys()].find(a => m.actuator(a).name === 'neck_pitch');
const headPitch = [...Array(m.nu).keys()].find(a => m.actuator(a).name === 'head_pitch');
const trunk = 1;
function bodyId(name){ for (let b=0;b<m.nbody;b++) if (m.body(b).name===name) return b; return -1; }
const jawId = bodyId('jaw_soft');
const dx = data.xpos[jawId*3] - data.xpos[trunk*3];
const dy = data.xpos[jawId*3+1] - data.xpos[trunk*3+1];
const dz = data.xpos[jawId*3+2] - data.xpos[trunk*3+2];
const lever = Math.hypot(dx, dy, dz);
const tq = m.actuator_forcerange[neck*2+1];
console.log(`\n=== the head as a lever ===`);
console.log(`  neck torque limit        ${tq.toFixed(2)} N.m`);
console.log(`  trunk -> beak distance   ${(lever*1000).toFixed(0)} mm`);
console.log(`  force available at beak  ${(tq/lever).toFixed(2)} N  (${(tq/lever/9.81*1000).toFixed(0)} g-force equivalent)`);
console.log(`  robot weight             ${(total*9.81).toFixed(2)} N`);
console.log(`  => the head can support  ${((tq/lever)/(total*9.81)*100).toFixed(0)}% of the robot's weight`);
