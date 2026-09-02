// Two things a keyframe author has to have right before writing a track:
//   (1) the RIGHT leg's sign convention (the left's was measured in
//       climb/foot-envelope.mjs: more-negative left_hip_pitch = foot forward);
//   (2) the HAUL STROKE — with the jaw pinned in the world by friction on a
//       tread, retracting the neck drags the TRUNK by minus the change in the
//       jaw's position in the trunk frame. This is the only 100 mm-class
//       forward stroke on the robot: max forward FOOT reach is 87 mm
//       (climb/reach-max.mjs) and the stand policy contributes none
//       (climb/steps-results.json: every "walk" row ends at x = -1.5 mm).
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/haul-stroke.mjs
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
const RF = geomId('right_foot_collision'), LF = geomId('left_foot_collision'), jawB = bodyId('jaw_soft');
function fk(over) { mj.mj_resetData(m, d); d.qpos[tf+2]=0; d.qpos[tf+3]=1;
  for (const [k,v] of Object.entries({...HOME,...over})) d.qpos[jntQ(k)] = v; mj.mj_forward(m,d); }
const mm = v => (v*1000).toFixed(0);
console.log('RIGHT-LEG SIGN CHECK (mirror of the left, HOME right_hip_pitch = +0.4579):');
for (const v of [-1.0, 0, 0.4579, 1.0, 1.5]) {
  fk({ right_hip_pitch: v });
  console.log(`  right_hip_pitch ${v.toFixed(4).padStart(8)} -> right sole x ${mm(d.geom_xpos[RF*3]).padStart(5)} z ${mm(d.geom_xpos[RF*3+2]).padStart(5)} mm`);
}
console.log('  => MORE POSITIVE right_hip_pitch swings the RIGHT foot FORWARD; the left is the exact mirror.');

console.log('\nHAUL STROKE: jaw position in the trunk frame, and the trunk motion that');
console.log('follows if the jaw is pinned (trunk delta = -(jaw delta)).\n');
const poses = [
  ['reach   ', -1.5000,  0.8000],
  ['reach hi', -1.5000, -0.8000],
  ['mid     ', -0.5000,  0.5000],
  ['retract ',  0.7000, -0.8000],
  ['retract+',  1.0472, -0.8000],
  ['retract^',  1.0472, -1.5708],
];
const P = poses.map(([lbl, np, hp]) => { fk({ neck_pitch: np, head_pitch: hp });
  return { lbl, np, hp, x: d.xpos[jawB*3], z: d.xpos[jawB*3+2] }; });
for (const p of P) console.log(`  ${p.lbl}  neck_pitch ${p.np.toFixed(4).padStart(8)} head_pitch ${p.hp.toFixed(4).padStart(8)}  jaw in trunk frame  x ${mm(p.x).padStart(5)}  z ${mm(p.z).padStart(5)} mm`);
console.log('\n  from -> to                     trunk moves  dx      dz');
for (const a of [P[0], P[1]]) for (const b of [P[3], P[4], P[5]])
  console.log(`  ${a.lbl} -> ${b.lbl}                        ${mm(a.x-b.x).padStart(6)} mm ${mm(a.z-b.z).padStart(6)} mm`);
console.log('\n  (positive dx = trunk hauled forward; positive dz = trunk lifted)');
