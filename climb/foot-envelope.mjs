// How high can a foot be placed AT A GIVEN FORWARD REACH? That is the number
// that decides whether feet can walk up a riser face, and it is not the same
// question as "how high can a foot go" (the answer to which is 75 mm, but
// BEHIND the trunk — climb/reach-envelope.mjs).
//
// Sign convention, measured, not assumed: NEGATIVE left_hip_pitch swings the
// left foot FORWARD (+x). HOME is -0.4579 and puts the sole at x +7 mm.
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/foot-envelope.mjs
import load from 'mujoco';
import fs from 'node:fs';
const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const m = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const d = new mj.MjData(m);
const jntQ = n => { for (let j = 0; j < m.njnt; j++) if (m.jnt(j).name === n) return m.jnt_qposadr[j]; return -1; };
const geomId = n => { for (let g = 0; g < m.ngeom; g++) if (m.geom(g).name === n) return g; return -1; };
let tf = -1; for (let j = 0; j < m.njnt; j++) if (m.jnt_type[j] === 0 && m.body(m.jnt_bodyid[j]).name === 'trunk_base') tf = m.jnt_qposadr[j];
const HOME = { left_hip_yaw:0, left_hip_roll:-0.0873, left_hip_pitch:-0.4579, left_knee:-0.0049, left_ankle:0.4530,
  neck_pitch:0.3491, head_pitch:0.3491, head_yaw:0, head_roll:0,
  right_hip_yaw:0, right_hip_roll:0.0873, right_hip_pitch:0.4579, right_knee:0.0049, right_ankle:-0.4530 };
const LF = geomId('left_foot_collision');
const R = m.geom_rbound[LF];
function fk(over) {
  mj.mj_resetData(m, d);
  d.qpos[tf+2] = 0; d.qpos[tf+3] = 1;
  for (const [k, v] of Object.entries({ ...HOME, ...over })) d.qpos[jntQ(k)] = v;
  mj.mj_forward(m, d);
  return [d.geom_xpos[LF*3], d.geom_xpos[LF*3+1], d.geom_xpos[LF*3+2]];
}
// Grid over the three pitch joints; bin by forward reach, keep the highest foot.
const bins = new Map();
const S = a => a;
for (let hp = -1.5708; hp <= 1.5708; hp += 0.1309)
 for (let kn = -1.5708; kn <= 1.5708; kn += 0.1309)
  for (let an = -1.5708; an <= 1.5708; an += 0.1309) {
    const [x, , z] = fk({ left_hip_pitch: hp, left_knee: kn, left_ankle: an });
    const b = Math.round(x * 1000 / 20) * 20;
    const cur = bins.get(b);
    if (!cur || z > cur.z) bins.set(b, { z, hp, kn, an });
  }
console.log('LEFT SOLE, highest reachable centre at each forward reach, in the TRUNK frame');
console.log('(grid: hip_pitch x knee x ankle, 0.1309 rad = 7.5 deg steps, 25^3 = 15625 poses)\n');
console.log('  fwd x   best sole z   hip_pitch   knee    ankle');
for (const b of [...bins.keys()].sort((a,c)=>a-c)) {
  if (b < -40 || b > 160) continue;
  const v = bins.get(b);
  console.log(`${String(b).padStart(6)} mm  ${(v.z*1000).toFixed(0).padStart(6)} mm    `
    + `${v.hp.toFixed(3).padStart(7)} ${v.kn.toFixed(3).padStart(7)} ${v.an.toFixed(3).padStart(7)}`);
}
console.log(`\nsole geom rbound = ${(R*1000).toFixed(1)} mm; the sole SURFACE is inside that of its centre.`);
console.log('Read a row as: with the trunk origin at height Z0 above the floor, a foot placed');
console.log('at forward reach x can have its sole centre at Z0 + (best sole z).');
