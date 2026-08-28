import load from 'mujoco';
import fs from 'node:fs';
const mj = await load();
mj.FS.writeFile('/scene.xml', fs.readFileSync('scene.xml','utf8'));
const model = mj.MjModel.mj_loadXML('/scene.xml');
const data = new mj.MjData(model);
const C = JSON.parse(fs.readFileSync('duckkit-constants.json','utf8'));
const home = C.homePose.filter((_,i)=>C.jointNames[i]!=='mouth');
for (let i=0;i<14;i++) data.ctrl[i]=home[i];
for (let s=0;s<=1200;s++){
  if (s%100===0){
    const q=[data.qpos[3],data.qpos[4],data.qpos[5],data.qpos[6]];
    // world -z in trunk frame -> upright when ~ -1
    const [w,x,y,z]=q;
    const gz = -(1-2*(x*x+y*y));
    console.log(`PROBE t=${(s*0.005).toFixed(2)}s z=${data.qpos[2].toFixed(4)} gz=${gz.toFixed(3)} ncon=${data.ncon}`);
  }
  mj.mj_step(model,data);
}
