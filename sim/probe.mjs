import load from 'mujoco';
import fs from 'node:fs';
const mj = await load();
mj.FS.writeFile('/scene.xml', fs.readFileSync('scene.xml', 'utf8'));
const model = mj.MjModel.mj_loadXML("/scene.xml");
console.log('PROBE nq', model.nq, 'nv', model.nv, 'nu', model.nu, 'ngeom', model.ngeom, 'nbody', model.nbody);
const data = new mj.MjData(model);
const C = JSON.parse(fs.readFileSync('duckkit-constants.json','utf8'));
const pj = C.jointNames.filter(n=>n!=='mouth');
const home = C.jointNames.map((n,i)=>C.homePose[i]).filter((_,i)=>C.jointNames[i]!=='mouth');
// hold the home pose and let it settle onto the floor
for (let i=0;i<pj.length;i++) data.ctrl[i] = home[i];
for (let s=0;s<2000;s++) mj.mj_step(model, data);
console.log('PROBE after 10 s holding home pose: trunk z =', data.qpos[2].toFixed(4),
            ' xy =', data.qpos[0].toFixed(3), data.qpos[1].toFixed(3));
