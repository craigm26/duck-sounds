import load from 'mujoco';
import fs from 'node:fs';
const mj = await load();
try { mj.FS.mkdir('/assets'); } catch {}
for (const f of fs.readdirSync('assets')) mj.FS.writeFile('/assets/' + f, fs.readFileSync('assets/' + f));
mj.FS.writeFile('/scene.xml', fs.readFileSync('scene_pollen.xml', 'utf8'));
const model = mj.MjModel.mj_loadXML('/scene.xml');
console.log('FULL nq', model.nq, 'ngeom', model.ngeom, 'nmesh', model.nmesh,
            'nmeshvert', model.nmeshvert, 'nmeshface', model.nmeshface);
mj.mj_saveModel(model, '/full.mjb', null);
const bytes = mj.FS.readFile('/full.mjb');
fs.writeFileSync('scene_full.mjb', Buffer.from(bytes));
console.log('SAVED scene_full.mjb', (bytes.length / 1048576).toFixed(2), 'MB');
// what the renderer needs, straight from the model
const acc = ['mesh_vertadr','mesh_vertnum','mesh_faceadr','mesh_facenum','mesh_vert','mesh_face',
             'geom_dataid','geom_bodyid','geom_pos','geom_quat','geom_rgba','geom_type','geom_group','geom_contype'];
for (const a of acc) console.log('  ', a, typeof model[a], model[a]?.length ?? '');
