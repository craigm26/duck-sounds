// Every geom in the canon plant: name, body, type, size, contype/conaffinity.
// Asked because wholebody-physics.mjs found NO geom named head/jaw/beak -- if the
// head has no collision geom, "press the head on the tread" is not a thing the
// physics can do, and every head-anchor strategy is dead on arrival.
import load from 'mujoco';
import fs from 'node:fs';
const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const m = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const d = new mj.MjData(m); mj.mj_forward(m,d);
const T=['plane','hfield','sphere','capsule','ellipsoid','cylinder','box','mesh','sdf'];
console.log('idx  geom_name                body                     type      size(mm)                 contype/conaff  group  mass_of_body(g)');
for(let g=0;g<m.ngeom;g++){
  const b=m.geom_bodyid[g];
  console.log(`${String(g).padStart(3)}  ${(m.geom(g).name||'(unnamed)').padEnd(24)} ${(m.body(b).name||'?').padEnd(24)} ${T[m.geom_type[g]].padEnd(9)} `
   +`${[0,1,2].map(k=>(m.geom_size[g*3+k]*1000).toFixed(1)).join(',').padEnd(24)} ${m.geom_contype[g]}/${m.geom_conaffinity[g]}`.padEnd(20)
   +`  ${m.geom_group[g]}  ${(m.body_mass[b]*1000).toFixed(1)}`);
}
console.log('\nbodies with NO geom at all:');
const has=new Set(); for(let g=0;g<m.ngeom;g++) has.add(m.geom_bodyid[g]);
for(let b=1;b<m.nbody;b++) if(!has.has(b)) console.log('  '+(m.body(b).name||'#'+b)+`  mass ${(m.body_mass[b]*1000).toFixed(1)} g`);
