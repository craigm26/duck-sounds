// The two numbers the whole design turns on: how far FORWARD of the trunk
// origin can a foot go, and how far forward can the head go — over the full
// joint travel, and with the trunk itself pitched forward.
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/reach-max.mjs
import load from 'mujoco';
import fs from 'node:fs';
const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const m = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const d = new mj.MjData(m);
const jntQ = n => { for (let j = 0; j < m.njnt; j++) if (m.jnt(j).name === n) return m.jnt_qposadr[j]; return -1; };
const geomId = n => { for (let g = 0; g < m.ngeom; g++) if (m.geom(g).name === n) return g; return -1; };
const bodyId = n => { for (let b = 0; b < m.nbody; b++) if (m.body(b).name === n) return b; return -1; };
let tf = -1; for (let j = 0; j < m.njnt; j++) if (m.jnt_type[j] === 0 && m.body(m.jnt_bodyid[j]).name === 'trunk_base') tf = m.jnt_qposadr[j];
const HOME = { left_hip_yaw:0, left_hip_roll:-0.0873, left_hip_pitch:-0.4579, left_knee:-0.0049, left_ankle:0.4530,
  neck_pitch:0.3491, head_pitch:0.3491, head_yaw:0, head_roll:0,
  right_hip_yaw:0, right_hip_roll:0.0873, right_hip_pitch:0.4579, right_knee:0.0049, right_ankle:-0.4530 };
const LF = geomId('left_foot_collision'), jawB = bodyId('jaw_soft');
const HEADG = []; for (let g = 0; g < m.ngeom; g++) if (m.geom_bodyid[g] === jawB && m.geom_contype[g]) HEADG.push(g);
// trunk pitched nose-down by theta about +y
function fk(over, theta = 0) {
  mj.mj_resetData(m, d);
  d.qpos[tf+2] = 0;
  d.qpos[tf+3] = Math.cos(theta/2); d.qpos[tf+4] = 0; d.qpos[tf+5] = Math.sin(theta/2); d.qpos[tf+6] = 0;
  for (const [k, v] of Object.entries({ ...HOME, ...over })) d.qpos[jntQ(k)] = v;
  mj.mj_forward(m, d);
}
let bestFoot = { x: -1e9 };
for (let hp = -1.5708; hp <= 1.5708; hp += 0.1047)
 for (let kn = -1.5708; kn <= 1.5708; kn += 0.1047)
  for (let an = -1.5708; an <= 1.5708; an += 0.1047) {
    fk({ left_hip_pitch: hp, left_knee: kn, left_ankle: an });
    const x = d.geom_xpos[LF*3];
    if (x > bestFoot.x) bestFoot = { x, z: d.geom_xpos[LF*3+2], hp, kn, an };
  }
console.log(`MAX FORWARD FOOT REACH (sole centre, trunk frame, trunk level): x = ${(bestFoot.x*1000).toFixed(1)} mm `
  + `at z = ${(bestFoot.z*1000).toFixed(1)} mm  (hip_pitch ${bestFoot.hp.toFixed(3)}, knee ${bestFoot.kn.toFixed(3)}, ankle ${bestFoot.an.toFixed(3)})`);
console.log(`  sole rbound ${(m.geom_rbound[LF]*1000).toFixed(1)} mm, so the toe bound is x = ${((bestFoot.x+m.geom_rbound[LF])*1000).toFixed(1)} mm`);

let bestHead = { x: -1e9 };
for (let np = -1.5708; np <= 1.0472; np += 0.0873)
 for (let hp = -1.5708; hp <= 1.5708; hp += 0.0873) {
   fk({ neck_pitch: np, head_pitch: hp });
   const x = d.xpos[jawB*3];
   if (x > bestHead.x) bestHead = { x, z: d.xpos[jawB*3+2], np, hp };
 }
console.log(`MAX FORWARD HEAD REACH (jaw_soft centre, trunk level): x = ${(bestHead.x*1000).toFixed(1)} mm `
  + `at z = ${(bestHead.z*1000).toFixed(1)} mm  (neck_pitch ${bestHead.np.toFixed(3)}, head_pitch ${bestHead.hp.toFixed(3)})`);

console.log('\nWith the TRUNK pitched nose-down (the only way to get the head over a tread edge):');
console.log(' trunk pitch |  best head x  at z   | neck_pitch head_pitch | trailing-foot x when trunk pitched');
for (const deg of [0, 15, 30, 45, 60, 75, 90]) {
  const th = deg * Math.PI / 180;
  let b = { x: -1e9 };
  for (let np = -1.5708; np <= 1.0472; np += 0.0873)
   for (let hp = -1.5708; hp <= 1.5708; hp += 0.0873) {
     fk({ neck_pitch: np, head_pitch: hp }, th);
     const x = d.xpos[jawB*3];
     let lo = 1e9; for (const g of HEADG) lo = Math.min(lo, d.geom_xpos[g*3+2] - m.geom_rbound[g]);
     if (x > b.x) b = { x, z: d.xpos[jawB*3+2], lo, np, hp };
   }
  fk({}, th);
  console.log(`${String(deg).padStart(9)} deg | ${(b.x*1000).toFixed(0).padStart(6)} mm  ${(b.z*1000).toFixed(0).padStart(5)} mm |`
    + ` ${b.np.toFixed(3).padStart(7)} ${b.hp.toFixed(3).padStart(8)}   | home-pose sole at x ${(d.geom_xpos[LF*3]*1000).toFixed(0).padStart(5)} mm z ${(d.geom_xpos[LF*3+2]*1000).toFixed(0).padStart(5)} mm`);
}
