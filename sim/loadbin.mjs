import load from 'mujoco';
import fs from 'node:fs';
const mj = await load();
mj.FS.writeFile('/scene.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
let model;
try { model = mj.MjModel.mj_loadBinary('/scene.mjb', new mj.MjVFS()); }
catch (e) { console.log('null vfs failed:', e.message); }
if (model) console.log('BINARY OK nq', model.nq, 'ngeom', model.ngeom, 'nu', model.nu, 'nsensordata', model.nsensordata);
