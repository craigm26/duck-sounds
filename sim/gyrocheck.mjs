import load from 'mujoco';
import fs from 'node:fs';
const mj = await load();
mj.FS.writeFile('/scene.xml', fs.readFileSync('scene.xml','utf8'));
const model = mj.MjModel.mj_loadXML('/scene.xml');
const data = new mj.MjData(model);
const C = JSON.parse(fs.readFileSync('duckkit-constants.json','utf8'));
const HOME = C.homePose.filter((_,i)=>C.jointNames[i]!=='mouth');
mj.mj_resetData(model,data);
data.qpos[2]=0.1231; data.qpos[3]=1;
for(let i=0;i<14;i++){data.qpos[7+i]=HOME[i];data.ctrl[i]=HOME[i];}
// spin the free body about world +z (counter-clockwise = left) and compare
data.qvel[5] = 1.0;
mj.mj_forward(model,data);
console.log('PROBE commanded world +z spin of 1.0 rad/s');
console.log('PROBE qvel[3..5]      =', [3,4,5].map(i=>data.qvel[i].toFixed(3)).join(' '));
console.log('PROBE sensordata[0..2]=', [0,1,2].map(i=>data.sensordata[i].toFixed(3)).join(' '));
console.log('PROBE nsensordata =', model.nsensordata);
