// Where can this duck put its HEAD, and where can it put a FOOT?
//
// Forward kinematics only (mj_forward), so it is exact and costs nothing. Two
// envelopes, both expressed in the TRUNK frame, because that is the frame a
// keyframe search reasons in: the trunk's own pitch is not an actuator, it is
// whatever the hips and ankles leave it at.
//
//   HEAD: the lowest and most-forward point of the three jaw_soft collision
//         geoms (climb/geom-facts.mjs: rbound 85.0 / 62.5 / 83.4 mm), swept
//         over neck_pitch x head_pitch. This is what a head-plant lands on.
//   FOOT: the sole geom centre swept over hip_pitch x knee x ankle. This is
//         how high up a riser face a foot can be placed.
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/reach-envelope.mjs
import load from 'mujoco';
import fs from 'node:fs';

const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const m = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const d = new mj.MjData(m);
const jntQ = n => { for (let j = 0; j < m.njnt; j++) if (m.jnt(j).name === n) return m.jnt_qposadr[j]; return -1; };
const bodyId = n => { for (let b = 0; b < m.nbody; b++) if (m.body(b).name === n) return b; return -1; };
const geomId = n => { for (let g = 0; g < m.ngeom; g++) if (m.geom(g).name === n) return g; return -1; };
let trunkFree = -1;
for (let j = 0; j < m.njnt; j++) if (m.jnt_type[j] === 0 && m.body(m.jnt_bodyid[j]).name === 'trunk_base') trunkFree = m.jnt_qposadr[j];

const HOME = { left_hip_yaw:0, left_hip_roll:-0.0873, left_hip_pitch:-0.4579, left_knee:-0.0049, left_ankle:0.4530,
  neck_pitch:0.3491, head_pitch:0.3491, head_yaw:0, head_roll:0,
  right_hip_yaw:0, right_hip_roll:0.0873, right_hip_pitch:0.4579, right_knee:0.0049, right_ankle:-0.4530 };
const jawB = bodyId('jaw_soft');
const HEADG = []; for (let g = 0; g < m.ngeom; g++) if (m.geom_bodyid[g] === jawB && m.geom_contype[g]) HEADG.push(g);
const LF = geomId('left_foot_collision');

function fk(over = {}) {
  mj.mj_resetData(m, d);
  d.qpos[trunkFree] = 0; d.qpos[trunkFree+1] = 0; d.qpos[trunkFree+2] = 0;
  d.qpos[trunkFree+3] = 1; d.qpos[trunkFree+4] = 0; d.qpos[trunkFree+5] = 0; d.qpos[trunkFree+6] = 0;
  for (const [k, v] of Object.entries({ ...HOME, ...over })) d.qpos[jntQ(k)] = v;
  mj.mj_forward(m, d);
}
// The head's contact point: lowest z, and the most +x, over the head geoms
// (a mesh geom's xpos is its frame origin; rbound is the enclosing radius, so
// "lowest point" is xpos.z - rbound and "furthest forward" is xpos.x + rbound.
// That is a bound, not the mesh surface, and is labelled as such below.)
function head() {
  let lo = 1e9, fwd = -1e9, cx = 0, cz = 0;
  for (const g of HEADG) {
    const z = d.geom_xpos[g*3+2] - m.geom_rbound[g], x = d.geom_xpos[g*3] + m.geom_rbound[g];
    if (z < lo) { lo = z; cz = d.geom_xpos[g*3+2]; }
    if (x > fwd) { fwd = x; cx = d.geom_xpos[g*3]; }
  }
  return { lo, fwd, cx, cz };
}
const mm = v => (v*1000).toFixed(0);

console.log('HEAD ENVELOPE in the trunk frame (trunk origin at 0,0,0, no trunk rotation).');
console.log('"lowest" and "furthest fwd" are rbound BOUNDS on the head collision meshes, so the real');
console.log('shell is inside them; jaw_soft body centre is the exact number beside each.\n');
console.log('neck_pitch  head_pitch |  jaw_soft centre x,z |  lowest head point z | furthest fwd x');
for (const np of [-1.5708, -1.0, -0.5, 0, 0.3491, 0.7, 1.0472]) {
  for (const hp of [-1.5708, -0.8, 0, 0.3491, 0.8, 1.5708]) {
    fk({ neck_pitch: np, head_pitch: hp });
    const h = head();
    console.log(`${np.toFixed(4).padStart(9)} ${hp.toFixed(4).padStart(10)}  | `
      + `${mm(d.xpos[jawB*3]).padStart(6)},${mm(d.xpos[jawB*3+2]).padStart(6)} mm |`
      + `${mm(h.lo).padStart(9)} mm       |${mm(h.fwd).padStart(9)} mm`);
  }
}

console.log('\nhead_yaw sweep at the plant pose neck_pitch=-1.20, head_pitch=0.60 (yaw is the joint NO authored move uses):');
for (const hy of [-2.9671, -1.5708, -0.7854, 0, 0.7854, 1.5708, 2.9671]) {
  fk({ neck_pitch: -1.20, head_pitch: 0.60, head_yaw: hy });
  console.log(`  head_yaw ${hy.toFixed(4).padStart(8)} rad (${(hy*180/Math.PI).toFixed(0).padStart(4)} deg): `
    + `jaw_soft at x ${mm(d.xpos[jawB*3]).padStart(5)}  y ${mm(d.xpos[jawB*3+1]).padStart(5)}  z ${mm(d.xpos[jawB*3+2]).padStart(5)} mm`);
}
console.log('\nhead_roll sweep at the same plant pose:');
for (const hr of [-0.4363, 0, 0.4363]) {
  fk({ neck_pitch: -1.20, head_pitch: 0.60, head_roll: hr });
  console.log(`  head_roll ${hr.toFixed(4).padStart(8)}: jaw_soft y ${mm(d.xpos[jawB*3+1]).padStart(5)} mm  z ${mm(d.xpos[jawB*3+2]).padStart(5)} mm`);
}

console.log('\nFOOT ENVELOPE (left sole geom centre) in the trunk frame:');
fk();
console.log(`  HOME:                       x ${mm(d.geom_xpos[LF*3]).padStart(5)}  y ${mm(d.geom_xpos[LF*3+1]).padStart(5)}  z ${mm(d.geom_xpos[LF*3+2]).padStart(5)} mm`);
for (const [hp, kn, an, lbl] of [
  [ 1.0472, -1.5708, 0.0,  'hip +60 deg, knee folded'],
  [ 1.5708, -1.5708, 0.0,  'hip +90 deg, knee folded (max flexion)'],
  [ 1.5708, -1.5708, -1.0, 'hip +90, knee folded, ankle toe-down'],
  [ 1.5708, -1.5708, 1.5708, 'hip +90, knee folded, ankle toe-up (sole faces a riser)'],
  [ 1.2,    -0.6,     1.2,  'mid: hip +69, knee -34, ankle +69'],
  [-1.0,     0.6,     0.4,  'hip back 57 deg (trailing leg)'],
]) {
  fk({ left_hip_pitch: hp, left_knee: kn, left_ankle: an });
  console.log(`  ${lbl.padEnd(45)} x ${mm(d.geom_xpos[LF*3]).padStart(5)}  y ${mm(d.geom_xpos[LF*3+1]).padStart(5)}  z ${mm(d.geom_xpos[LF*3+2]).padStart(5)} mm`);
}
console.log('\nhip_yaw + hip_roll sweep (the two leg joints NO authored move uses / saturates):');
for (const [hy, hr, lbl] of [[0.5236, 0, 'left_hip_yaw +30 deg (max)'], [-0.4363, 0, 'left_hip_yaw -25 deg (min)'],
                             [0, 0.3840, 'left_hip_roll +22 deg (max)'], [0, -0.3840, 'left_hip_roll -22 deg (min)']]) {
  fk({ left_hip_yaw: hy, left_hip_roll: hr, left_hip_pitch: 1.2, left_knee: -1.0 });
  console.log(`  ${lbl.padEnd(30)} at hip_pitch 1.2 knee -1.0: foot x ${mm(d.geom_xpos[LF*3]).padStart(5)}  y ${mm(d.geom_xpos[LF*3+1]).padStart(5)}  z ${mm(d.geom_xpos[LF*3+2]).padStart(5)} mm`);
}
