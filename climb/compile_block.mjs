// compile_block.mjs — build climb/scene_block.mjb.
//
// A SEPARATE SCENE, for compile_multiduck.mjs's reason: every recorded clip in
// duckkit claims sim/scene.mjb by digest, so scene.mjb is not edited. This one
// differs from sim/scene_physics.xml in exactly two places:
//
//   (1) every step geom's conaffinity 4 -> 0. A step still collides with the
//       duck (class collision, contype 5: step.contype 4 & duck.conaffinity 5)
//       and with the props (contype 1 / conaffinity 5), but two steps no longer
//       see each other. site/stairs.js:24 sets STEP_HALF_DEPTH = 0.17 against a
//       0.28 run, so consecutive 200 kg blocks overlap by 60 mm in x and
//       (0.2 - rise) in z; sharing a collision bit, the flight shoves ITSELF
//       apart below a ~145 mm rise. No physics constant moves: friction,
//       gravity, timestep, masses, actuator gains, solref are untouched.
//   (2) one added free body, prop_block: a 60 mm cube, 100 g, condim 4,
//       friction 0.9 0.02 0.002 — block_a's line with a bigger size and mass.
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/compile_block.mjs
import load from 'mujoco';
import fs from 'node:fs';
const mj = await load();
try { mj.FS.mkdir('/assets'); } catch {}
for (const f of fs.readdirSync('assets')) mj.FS.writeFile('/assets/' + f, fs.readFileSync('assets/' + f));
mj.FS.writeFile('/scene.xml', fs.readFileSync('../climb/scene_block.xml', 'utf8'));
const model = mj.MjModel.mj_loadXML('/scene.xml');
console.log('compiled: nq', model.nq, 'ngeom', model.ngeom, 'nu', model.nu, 'nbody', model.nbody);
let found = 0;
for (let g = 0; g < model.ngeom; g++) {
  const n = model.geom(g).name || '';
  if (n === 'prop_block_geom') { found++; console.log('prop_block_geom contype', model.geom_contype[g], 'conaffinity', model.geom_conaffinity[g]); }
  if (n === 'step0_geom') console.log('step0_geom  contype', model.geom_contype[g], 'conaffinity', model.geom_conaffinity[g]);
  if (n === 'floor') console.log('floor       contype', model.geom_contype[g], 'conaffinity', model.geom_conaffinity[g]);
}
if (!found) throw new Error('prop_block_geom missing');
mj.mj_saveModel(model, '/scene_block.mjb', null);
const bytes = mj.FS.readFile('/scene_block.mjb');
fs.writeFileSync('../climb/scene_block.mjb', Buffer.from(bytes));
console.log('SAVED climb/scene_block.mjb', (bytes.length / 1024).toFixed(1), 'KB');
