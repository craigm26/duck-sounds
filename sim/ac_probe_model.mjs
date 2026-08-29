import load from 'mujoco';
import fs from 'node:fs';
const mj = await load();
mj.FS.writeFile('/scene.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const m = mj.MjModel.mj_loadBinary('/scene.mjb', new mj.MjVFS());
const INT = ['Euler','RK4','implicit','implicitfast'];
const SOL = ['PGS','CG','Newton'];
const CONE = ['pyramidal','elliptic'];
console.log('timestep', m.opt.timestep);
console.log('integrator', m.opt.integrator, INT[m.opt.integrator]);
console.log('solver', m.opt.solver, SOL[m.opt.solver]);
console.log('iterations', m.opt.iterations, 'ls_iterations', m.opt.ls_iterations);
console.log('tolerance', m.opt.tolerance, 'ls_tolerance', m.opt.ls_tolerance);
console.log('cone', m.opt.cone, CONE[m.opt.cone], 'impratio', m.opt.impratio, 'jacobian', m.opt.jacobian);
console.log('gravity', Array.from(m.opt.gravity));
console.log('disableflags', m.opt.disableflags, 'enableflags', m.opt.enableflags);
console.log('nq', m.nq, 'nv', m.nv, 'nu', m.nu, 'njnt', m.njnt, 'ngeom', m.ngeom, 'nbody', m.nbody);
// names
const names = new TextDecoder().decode(m.names);
const nameAt = adr => names.slice(adr, names.indexOf('\0', adr));
console.log('== joints (order) ==');
for (let j=0;j<m.njnt;j++) console.log(j, nameAt(m.name_jntadr[j]), 'type', m.jnt_type[j], 'dofadr', m.jnt_dofadr[j], 'range', m.jnt_range[2*j].toFixed(4), m.jnt_range[2*j+1].toFixed(4));
console.log('== actuators (order) ==');
for (let a=0;a<m.nu;a++) {
  const g = Array.from(m.actuator_gainprm.slice(10*a,10*a+3));
  const b = Array.from(m.actuator_biasprm.slice(10*a,10*a+3));
  const fr = [m.actuator_forcerange[2*a], m.actuator_forcerange[2*a+1]];
  const cr = [m.actuator_ctrlrange[2*a], m.actuator_ctrlrange[2*a+1]];
  console.log(a, nameAt(m.name_actuatoradr[a]), 'trnid', m.actuator_trnid[2*a], '=', nameAt(m.name_jntadr[m.actuator_trnid[2*a]]), 'gain', g, 'bias', b, 'forcerange', fr, 'ctrlrange', cr);
}
console.log('== duck joint dof props ==');
for (let j=0;j<m.njnt;j++) {
  const n = nameAt(m.name_jntadr[j]);
  if (m.jnt_type[j] !== 3) continue; // hinge only
  const d = m.jnt_dofadr[j];
  console.log(n, 'damping', m.dof_damping[d], 'armature', m.dof_armature[d], 'frictionloss', m.dof_frictionloss[d]);
}
console.log('== collision geoms (contype!=0) ==');
for (let g=0; g<m.ngeom; g++) {
  if (m.geom_contype[g]===0 && m.geom_conaffinity[g]===0) continue;
  console.log(g, nameAt(m.name_geomadr[g])||'(unnamed)', 'body', nameAt(m.name_bodyadr[m.geom_bodyid[g]]),
    'type', m.geom_type[g], 'contype', m.geom_contype[g], 'conaffinity', m.geom_conaffinity[g],
    'condim', m.geom_condim[g], 'priority', m.geom_priority[g],
    'friction', Array.from(m.geom_friction.slice(3*g,3*g+3)).map(x=>x.toFixed(3)).join(','),
    'solref', Array.from(m.geom_solref.slice(2*g,2*g+2)).map(x=>x.toFixed(4)).join(','),
    'solimp', Array.from(m.geom_solimp.slice(5*g,5*g+5)).map(x=>x.toFixed(4)).join(','));
}
