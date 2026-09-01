// compile_multiduck.mjs — build scene_multiduck.mjb: N ducks sharing one world.
//
// A SEPARATE SCENE, for the reason compile_grasp.mjs is a separate scene. Every
// recorded clip in duckkit was made against scene.mjb and claims that plant by
// digest; putting a second duck into it would change the physics those clips
// say they came from. A bench that wants a swarm asks for one:
//   python3 build_multiduck.py && node compile_multiduck.mjs
//   DUCKBENCH_SCENE=scene_multiduck.mjb node duckbench.mjs
import load from 'mujoco';
import fs from 'node:fs';
const mj = await load();
try { mj.FS.mkdir('/assets'); } catch {}
for (const f of fs.readdirSync('assets')) mj.FS.writeFile('/assets/' + f, fs.readFileSync('assets/' + f));
mj.FS.writeFile('/scene.xml', fs.readFileSync('scene_multiduck.xml', 'utf8'));
const model = mj.MjModel.mj_loadXML('/scene.xml');
console.log('compiled: nq', model.nq, 'ngeom', model.ngeom, 'nu', model.nu, 'nbody', model.nbody);
// The ducks this model actually holds, read back out of the compiled model
// rather than trusted from the XML — the same discovery duckbench.mjs does, so
// a scene that compiles but that the bench cannot find its ducks in fails HERE
// with a count instead of at the bench's first request.
const trunks = [];
for (let b = 0; b < model.nbody; b++) {
  const name = model.body(b).name;
  if (name === 'trunk_base' || name.endsWith('_trunk_base')) trunks.push(name);
}
console.log('ducks:', trunks.join(', '));
mj.mj_saveModel(model, '/scene_multiduck.mjb', null);
const bytes = mj.FS.readFile('/scene_multiduck.mjb');
fs.writeFileSync('scene_multiduck.mjb', Buffer.from(bytes));
console.log('SAVED scene_multiduck.mjb', (bytes.length / 1024).toFixed(1), 'KB');
