// How high can this duck get a foot, at all? A grip is only interesting if the
// foot can then reach the tread.
import load from 'mujoco';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
const C = JSON.parse(fs.readFileSync('duckkit-constants.json','utf8'));
const { HOME, LO, HI, findDuckJoints } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const model = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model);

console.log('nu (actuated joints) =', model.nu);
const actuated = [];
for (let a = 0; a < model.nu; a++) actuated.push(model.actuator(a).name);
console.log('actuated:', actuated.join(', '));
console.log('mouth actuated?', actuated.some(n => /mouth|jaw|beak/i.test(n)) ? 'YES' : 'NO');

// Find the highest the left sole can get relative to the floor, searching the
// left leg's joints inside their real travel limits.
let solIdx = -1, trunkIdx = -1;
for (let g = 0; g < model.ngeom; g++) {
  const n = model.geom(g).name || '';
  if (/left_foot_collision|sole_left/.test(n)) solIdx = g;
}
const names = C.jointNames.filter(n => n !== 'mouth');
const legIdx = [0,1,2,3,4];   // left leg within the 14
let best = -9, bestPose = null;
const steps = 5;
function setAndMeasure(pose) {
  data.qpos[D.freeQpos + 2] = 0.12; data.qpos[D.freeQpos + 3] = 1;
  for (let i = 0; i < 14; i++) data.qpos[D.qpos[i]] = pose[i];
  mj.mj_forward(model, data);
  return data.geom_xpos[solIdx * 3 + 2];
}
// coarse grid over hip pitch, knee, ankle (the three that lift a foot)
for (let a = 0; a <= steps; a++) for (let b = 0; b <= steps; b++) for (let c = 0; c <= steps; c++) {
  const pose = HOME.slice();
  pose[2] = LO[2] + (HI[2] - LO[2]) * a / steps;
  pose[3] = LO[3] + (HI[3] - LO[3]) * b / steps;
  pose[4] = LO[4] + (HI[4] - LO[4]) * c / steps;
  const z = setAndMeasure(pose);
  if (z > best) { best = z; bestPose = pose.slice(); }
}
const standing = setAndMeasure(HOME);
console.log(`sole at home pose:        ${(standing*1000).toFixed(0)} mm`);
console.log(`highest the sole reaches: ${(best*1000).toFixed(0)} mm  (trunk pinned at 120 mm)`);
console.log(`=> foot lift available:   ${((best - standing)*1000).toFixed(0)} mm`);
console.log('   7 inches is 178 mm.');
