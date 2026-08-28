import load from 'mujoco';
import fs from 'node:fs';
const mj = await load();
mj.FS.writeFile('/scene.xml', fs.readFileSync('scene.xml','utf8'));
const model = mj.MjModel.mj_loadXML('/scene.xml');
const data = new mj.MjData(model);
const C = JSON.parse(fs.readFileSync('duckkit-constants.json','utf8'));
const home = C.homePose.filter((_,i)=>C.jointNames[i]!=='mouth');
// place joints at home, resolve kinematics, and see where the soles actually are
for (let i=0;i<14;i++) data.qpos[7+i] = home[i];
mj.mj_forward(model, data);
// geom ids: 0 floor, 1 left_sole, 2 right_sole  (xpos is 3 per geom)
for (const g of [0,1,2]) {
  console.log(`PROBE geom ${g} xpos z = ${data.geom_xpos[g*3+2].toFixed(4)}`);
}
console.log('PROBE trunk z =', data.qpos[2].toFixed(4));
const soleZ = Math.min(data.geom_xpos[1*3+2], data.geom_xpos[2*3+2]);
console.log('PROBE lowest sole centre z =', soleZ.toFixed(4), '-> drop trunk by', (soleZ - 0.006).toFixed(4));
