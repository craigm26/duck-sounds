import load from 'mujoco';
import fs from 'node:fs';
const mj = await load();
try { mj.FS.mkdir('/assets'); } catch {}
for (const f of fs.readdirSync('assets')) mj.FS.writeFile('/assets/' + f, fs.readFileSync('assets/' + f));
mj.FS.writeFile('/scene.xml', fs.readFileSync('scene_physics.xml', 'utf8'));
const model = mj.MjModel.mj_loadXML('/scene.xml');
console.log('compiled: nq', model.nq, 'ngeom', model.ngeom, 'nu', model.nu);
mj.mj_saveModel(model, '/scene.mjb', null);
const bytes = mj.FS.readFile('/scene.mjb');
fs.writeFileSync('scene.mjb', Buffer.from(bytes));
console.log('SAVED scene.mjb', (bytes.length / 1024).toFixed(1), 'KB');
