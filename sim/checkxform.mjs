import load from 'mujoco';
import fs from 'node:fs';
import struct from 'node:buffer';
const mj = await load();
mj.FS.writeFile('/scene.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const model = mj.MjModel.mj_loadBinary('/scene.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const C = JSON.parse(fs.readFileSync('duckkit-constants.json','utf8'));
const HOME = C.homePose.filter((_,i)=>C.jointNames[i]!=='mouth');
data.qpos[2]=0.12; data.qpos[3]=1;
for(let i=0;i<14;i++) data.qpos[7+i]=HOME[i];
mj.mj_forward(model,data);
console.log('PHYSICS model nbody =', model.nbody);
for (let b=0;b<model.nbody;b++)
  console.log(`  body ${b} name=${model.body(b).name} xpos=(${data.xpos[b*3].toFixed(3)}, ${data.xpos[b*3+1].toFixed(3)}, ${data.xpos[b*3+2].toFixed(3)})`);
// and what the visual pack expects
const raw = fs.readFileSync('duck-visual.bin');
const hl = raw.readUInt32LE(0);
const meta = JSON.parse(raw.slice(4, 4+hl).toString('utf8'));
const bodies = [...new Set(meta.draws.map(d=>d.body))].sort((a,b)=>a-b);
console.log('VISUAL pack references bodies:', bodies.join(','));
