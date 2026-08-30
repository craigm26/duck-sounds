// compile_grasp.mjs — build scene_grasp.mjb: the canon scene with graspable
// things in it (a broom, a dowel).
//
// A SEPARATE SCENE ON PURPOSE. Every recorded clip in duckkit was made against
// scene.mjb, and adding free bodies to that file would change the plant those
// recordings claim to come from. A bench that wants a broom asks for one:
//   DUCKBENCH_SCENE=scene_grasp.mjb node duckbench.mjs
import load from 'mujoco';
import fs from 'node:fs';
const mj = await load();
try { mj.FS.mkdir('/assets'); } catch {}
for (const f of fs.readdirSync('assets')) mj.FS.writeFile('/assets/' + f, fs.readFileSync('assets/' + f));
mj.FS.writeFile('/scene.xml', fs.readFileSync('scene_grasp.xml', 'utf8'));
const model = mj.MjModel.mj_loadXML('/scene.xml');
console.log('compiled: nq', model.nq, 'ngeom', model.ngeom, 'nu', model.nu, 'nbody', model.nbody);
mj.mj_saveModel(model, '/scene_grasp.mjb', null);
const bytes = mj.FS.readFile('/scene_grasp.mjb');
fs.writeFileSync('scene_grasp.mjb', Buffer.from(bytes));
console.log('SAVED scene_grasp.mjb', (bytes.length / 1024).toFixed(1), 'KB');
