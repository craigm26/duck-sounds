// How much of its own weight can this duck put on its head?
//
// WHY NOT sim/torque.mjs. That script prints "trunk -> beak distance 1326 mm"
// for a 250 mm robot. It takes the trunk to be body index 1 (`const trunk = 1`,
// torque.mjs), and in scene.mjb body 1 is a STEP — the exact index-shift trap
// site/duckloop.mjs warns about for joints ("Adding stair bodies to the scene
// put their joints ahead of the duck's, and every index shifted"). Everything
// downstream of that distance — 0.48 N at the beak, "the head can support 7% of
// the robot's weight" — is measured from a staircase to a beak. This finds the
// bodies by NAME and reports the lever from the neck joint's own anchor, which
// is the joint whose torque limit is being divided.
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/lever-arm.mjs
import load from 'mujoco';
import fs from 'node:fs';

const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const m = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const data = new mj.MjData(m);
mj.mj_forward(m, data);

const bodyId = n => { for (let b = 0; b < m.nbody; b++) if (m.body(b).name === n) return b; return -1; };
const jointId = n => { for (let j = 0; j < m.njnt; j++) if (m.jnt(j).name === n) return j; return -1; };
const actId = n => { for (let a = 0; a < m.nu; a++) if (m.actuator(a).name === n) return a; return -1; };
const dist = (a, b) => Math.hypot(data.xpos[a * 3] - data.xpos[b * 3],
                                  data.xpos[a * 3 + 1] - data.xpos[b * 3 + 1],
                                  data.xpos[a * 3 + 2] - data.xpos[b * 3 + 2]);

console.log('body 1 in this plant is:', m.body(1).name, '  <- what torque.mjs calls the trunk');

let mass = 0;
for (let b = 1; b < m.nbody; b++) {
  if (/step|ball|block|cone|wall|floor/.test(m.body(b).name || '')) continue;
  mass += m.body_mass[b];
}
const weight = mass * 9.81;
console.log(`robot mass ${(mass * 1000).toFixed(0)} g, weight ${weight.toFixed(2)} N`);

const trunk = bodyId('trunk_base'), jaw = bodyId('jaw_soft');
const head = bodyId('head') >= 0 ? bodyId('head') : jaw;
const neckJ = jointId('neck_pitch');
const neckAnchor = neckJ >= 0 ? [data.xanchor[neckJ * 3], data.xanchor[neckJ * 3 + 1], data.xanchor[neckJ * 3 + 2]] : null;
const armFrom = p => Math.hypot(data.xpos[jaw * 3] - p[0], data.xpos[jaw * 3 + 1] - p[1],
                                data.xpos[jaw * 3 + 2] - p[2]);

const neckTq = m.actuator_forcerange[actId('neck_pitch') * 2 + 1];
const headTq = m.actuator_forcerange[actId('head_pitch') * 2 + 1];

console.log(`\ntrunk_base -> jaw_soft   ${(dist(trunk, jaw) * 1000).toFixed(0)} mm  (home pose, mj_forward only)`);
if (neckAnchor) console.log(`neck_pitch anchor -> jaw ${(armFrom(neckAnchor) * 1000).toFixed(0)} mm  <- the lever the neck torque acts through`);
console.log(`neck_pitch torque limit  ${neckTq.toFixed(4)} N.m`);
console.log(`head_pitch torque limit  ${headTq.toFixed(4)} N.m`);

if (neckAnchor) {
  const arm = armFrom(neckAnchor);
  const F = neckTq / arm;
  console.log(`\nforce the neck can hold at the beak  ${F.toFixed(2)} N  (${(F / 9.81 * 1000).toFixed(0)} g-force)`);
  console.log(`as a share of the robot's weight     ${(F / weight * 100).toFixed(0)}%`);
  console.log(`\nsim/torque.mjs prints ${(neckTq / dist(1, jaw)).toFixed(2)} N for the same quantity, `
            + `because its lever is ${(dist(1, jaw) * 1000).toFixed(0)} mm from ${m.body(1).name}.`);
}
