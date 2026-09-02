// The geometry three whole-body climb strategies have to be designed against.
//
// Everything here is read from the CANON plant (sim/scene.mjb, PLANT.md) by
// NAME, never by index — sim/torque.mjs's "trunk = body 1" trap (see
// climb/lever-arm.mjs:5-11) is what happens otherwise.
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/geom-facts.mjs
import load from 'mujoco';
import fs from 'node:fs';
import { findStairJoints, layoutStairs, STAIR_Y, STAIR_HALF_WIDTH, STEP_HALF_DEPTH, STEP_HALF_HEIGHT } from '../site/stairs.js';

const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const m = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const d = new mj.MjData(m);
const ADDR = findStairJoints(m);
layoutStairs(d, ADDR, { count: 4, rise: 0.09, run: 0.28, start: 0.12 });
mj.mj_forward(m, d);

const bodyId = n => { for (let b = 0; b < m.nbody; b++) if (m.body(b).name === n) return b; return -1; };
const geomId = n => { for (let g = 0; g < m.ngeom; g++) if (m.geom(g).name === n) return g; return -1; };
const jntId  = n => { for (let j = 0; j < m.njnt; j++) if (m.jnt(j).name === n) return j; return -1; };
const actId  = n => { for (let a = 0; a < m.nu; a++) if (m.actuator(a).name === n) return a; return -1; };
const mm = v => (v * 1000).toFixed(1);

console.log('=== masses (robot only: step/ball/block/cone/wall/floor excluded) ===');
let mass = 0, heavies = [];
for (let b = 1; b < m.nbody; b++) {
  const n = m.body(b).name || '';
  if (/step|ball|block|cone|wall|floor/.test(n)) continue;
  mass += m.body_mass[b];
  heavies.push([n, m.body_mass[b]]);
}
heavies.sort((a, b) => b[1] - a[1]);
console.log(`robot mass ${(mass*1000).toFixed(1)} g, weight ${(mass*9.81).toFixed(3)} N`);
console.log('heaviest links:', heavies.slice(0, 6).map(([n, v]) => `${n} ${(v*1000).toFixed(1)}g`).join(', '));
// how much of the mass is the head assembly (everything under body `neck`)
let headMass = 0;
const under = new Set([bodyId('neck')]);
for (let b = 1; b < m.nbody; b++) if (under.has(m.body_parentid[b])) under.add(b);
for (let b = 1; b < m.nbody; b++) if (under.has(b)) headMass += m.body_mass[b];
console.log(`head+neck assembly (subtree of body "neck"): ${(headMass*1000).toFixed(1)} g `
          + `= ${(headMass/mass*100).toFixed(0)}% of the robot`);

console.log('\n=== the collision geoms that matter, at HOME (mj_forward, trunk free joint at its XML pos) ===');
for (const g of ['left_foot_collision','right_foot_collision']) {
  const i = geomId(g);
  console.log(`${g.padEnd(22)} id ${i}  type ${m.geom_type[i]}  rbound ${mm(m.geom_rbound[i])} mm  `
    + `pos ${mm(d.geom_xpos[i*3])},${mm(d.geom_xpos[i*3+1])},${mm(d.geom_xpos[i*3+2])} mm  `
    + `friction ${m.geom_friction[i*3]}/${m.geom_friction[i*3+1]}/${m.geom_friction[i*3+2]}  priority ${m.geom_priority[i]}  `
    + `contype ${m.geom_contype[i]} conaffinity ${m.geom_conaffinity[i]}`);
}
console.log('\nunnamed head-shell / jaw collision geoms (body jaw_soft), which are what a head-plant lands on:');
const jawB = bodyId('jaw_soft');
for (let g = 0; g < m.ngeom; g++) {
  if (m.geom_bodyid[g] !== jawB) continue;
  if (m.geom_contype[g] === 0 && m.geom_conaffinity[g] === 0) continue;
  console.log(`  geom ${g} name "${m.geom(g).name||'(unnamed)'}" type ${m.geom_type[g]} rbound ${mm(m.geom_rbound[g])} mm  `
    + `friction ${m.geom_friction[g*3]}/${m.geom_friction[g*3+1]}/${m.geom_friction[g*3+2]} priority ${m.geom_priority[g]} `
    + `contype ${m.geom_contype[g]} conaffinity ${m.geom_conaffinity[g]}`);
}
const s0 = geomId('step0_geom');
console.log(`\nstep0_geom  friction ${m.geom_friction[s0*3]}/${m.geom_friction[s0*3+1]}/${m.geom_friction[s0*3+2]} `
  + `priority ${m.geom_priority[s0]} contype ${m.geom_contype[s0]} conaffinity ${m.geom_conaffinity[s0]} `
  + `size ${m.geom_size[s0*3]}/${m.geom_size[s0*3+1]}/${m.geom_size[s0*3+2]} m`);
const wn = geomId('wall_n');
console.log(`wall_n      pos y ${m.geom_pos[wn*3+1]} half-thickness ${m.geom_size[wn*3+1]} -> INNER FACE y = `
  + `${(m.geom_pos[wn*3+1]-m.geom_size[wn*3+1]).toFixed(3)} m, top z = ${(m.geom_pos[wn*3+2]+m.geom_size[wn*3+2]).toFixed(3)} m; `
  + `contype ${m.geom_contype[wn]} conaffinity ${m.geom_conaffinity[wn]}`);
console.log(`stairs.js says STAIR_Y = ${STAIR_Y.toFixed(4)} (it assumes wall half-thickness 0.025), `
  + `tread half-width ${STAIR_HALF_WIDTH} -> tread north edge y = ${(STAIR_Y+STAIR_HALF_WIDTH).toFixed(4)} m`);
console.log(`duck at side=+0.085 sits at y = ${(STAIR_Y+0.085).toFixed(4)} m; clearance to the wall face = `
  + `${mm(m.geom_pos[wn*3+1]-m.geom_size[wn*3+1]-(STAIR_Y+0.085))} mm`);

console.log('\n=== actuators: what a keyframe can actually push with ===');
for (const a of ['neck_pitch','head_pitch','head_yaw','head_roll','left_hip_yaw','left_hip_roll','left_hip_pitch','left_knee','left_ankle']) {
  const i = actId(a), j = jntId(a);
  console.log(`  ${a.padEnd(15)} forcerange ±${m.actuator_forcerange[i*2+1].toFixed(4)} N.m  kp ${m.actuator_gainprm[i*10].toFixed(3)}  `
    + `range [${m.jnt_range[j*2].toFixed(4)}, ${m.jnt_range[j*2+1].toFixed(4)}] rad `
    + `= [${(m.jnt_range[j*2]*180/Math.PI).toFixed(0)}, ${(m.jnt_range[j*2+1]*180/Math.PI).toFixed(0)}] deg`);
}

console.log('\n=== which joints the four authored moves ever move ===');
const NAMES = ['left_hip_yaw','left_hip_roll','left_hip_pitch','left_knee','left_ankle',
  'neck_pitch','head_pitch','head_yaw','head_roll',
  'right_hip_yaw','right_hip_roll','right_hip_pitch','right_knee','right_ankle'];
const HOME14 = [0,-0.0873,-0.4579,-0.0049,0.453,0.3491,0.3491,0,0,0,0.0873,0.4579,0.0049,-0.453];
const span = new Array(14).fill(0);
for (const f of ['intent-stepup.json','intent-lever.json','intent-riser.json','intent-climb.json']) {
  const j = JSON.parse(fs.readFileSync('../site/' + f, 'utf8'));
  if (!j.keyframes) { console.log(`  ${f}: params-only (buildTrack); buildTrack touches only neckPitch, headPitch and the two legs' pitch/knee/ankle/roll`); continue; }
  for (const k of j.keyframes) for (let i = 0; i < 14; i++) span[i] = Math.max(span[i], Math.abs(k.pose[i] - HOME14[i]));
}
NAMES.forEach((n, i) => console.log(`  ${n.padEnd(15)} max |offset from home| over all authored keyframes: ${span[i].toFixed(4)} rad`
  + (span[i] < 1e-9 ? '   <-- NEVER MOVED BY ANY AUTHORED MOVE' : '')));
