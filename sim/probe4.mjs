import load from 'mujoco';
import fs from 'node:fs';
const mj = await load();
mj.FS.writeFile('/scene.xml', fs.readFileSync('scene.xml','utf8'));
const model = mj.MjModel.mj_loadXML('/scene.xml');
const data = new mj.MjData(model);
const C = JSON.parse(fs.readFileSync('duckkit-constants.json','utf8'));
const home = C.homePose.filter((_,i)=>C.jointNames[i]!=='mouth');

function reset(){
  mj.mj_resetData(model, data);
  data.qpos[0]=0; data.qpos[1]=0; data.qpos[2]=0.1231;
  data.qpos[3]=1; data.qpos[4]=0; data.qpos[5]=0; data.qpos[6]=0;
  for (let i=0;i<14;i++){ data.qpos[7+i]=home[i]; data.ctrl[i]=home[i]; }
  mj.mj_forward(model, data);
}
reset();
for (let s=0;s<=1600;s++){
  if (s%200===0){
    const x=data.qpos[4], y=data.qpos[5];
    const gz = -(1-2*(x*x+y*y));
    console.log(`PROBE t=${(s*0.005).toFixed(2)}s z=${data.qpos[2].toFixed(4)} gz=${gz.toFixed(3)} ncon=${data.ncon}`);
  }
  mj.mj_step(model,data);
}
