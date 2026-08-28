import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const { HOME, buildObs, gaitTargets, projectedGravity, command } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/scene.xml', fs.readFileSync('scene.xml', 'utf8'));
const model = mj.MjModel.mj_loadXML('/scene.xml');
const data = new mj.MjData(model);
const session = await ort.InferenceSession.create('./alpha_walking.onnx');
const inputName = session.inputNames[0];

function reset() {
  mj.mj_resetData(model, data);
  data.qpos[2] = 0.1231; data.qpos[3] = 1;
  for (let i = 0; i < 14; i++) { data.qpos[7 + i] = HOME[i]; data.ctrl[i] = HOME[i]; }
  mj.mj_forward(model, data);
}

async function run(opts, seconds) {
  reset();
  const cmd = command(opts);
  let lastAction = new Array(14).fill(0), previous = null;
  const trace = [];
  for (let t = 0; t < Math.round(seconds * C.tickHz); t++) {
    const q = [data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]];
    const jpos = [], jvel = [];
    for (let k = 0; k < 14; k++) { jpos.push(data.qpos[7 + k]); jvel.push(data.qvel[6 + k]); }
    const obs = buildObs([data.qvel[3], data.qvel[4], data.qvel[5]],
                         projectedGravity(q), jpos, jvel, lastAction, cmd);
    const out = await session.run({ [inputName]: new ort.Tensor('float32', obs, [1, 61]) });
    lastAction = Array.from(out[session.outputNames[0]].data);
    previous = gaitTargets(lastAction, previous);
    for (let k = 0; k < 14; k++) data.ctrl[k] = previous[k];
    for (let s = 0; s < 4; s++) mj.mj_step(model, data);
    trace.push({ x: data.qpos[0], y: data.qpos[1], z: data.qpos[2],
                 gz: projectedGravity([data.qpos[3],data.qpos[4],data.qpos[5],data.qpos[6]])[2] });
  }
  return trace;
}

let failures = 0;
for (const [name, opts] of [['stand',{vx:0}], ['walk',{vx:0.15}], ['fast',{vx:0.20}], ['turn',{vx:0,vyaw:0.8}], ['walk+turn',{vx:0.12,vyaw:0.5}]]) {
  const tr = await run(opts, 8);
  const last = tr[tr.length - 1];
  const fell = tr.some(f => f.gz > -0.5);
  const dist = Math.hypot(last.x, last.y);
  if (fell) failures++;
  console.log(`RESULT ${name.padEnd(10)} moved ${dist.toFixed(3)} m  speed ${(dist/8).toFixed(3)} m/s  z ${last.z.toFixed(3)}  upright=${!fell}`);
}
console.log(failures === 0 ? 'PASS the duck never fell over' : `FAIL it fell in ${failures} case(s)`);
