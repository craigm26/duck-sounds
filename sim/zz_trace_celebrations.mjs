// Sanity trace: one nominal (no-DR, no-push) run per celebration, logging the
// extreme reached by each driven joint, to prove the offsets really applied.
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
import { clearStairs, findStairJoints } from '../site/stairs.js';
const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);
const DT = 1 / C.tickHz;
const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const model = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model), ADDR = findStairJoints(model);
let GYRO = 0;
for (let i = 0; i < model.nsensor; i++) if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;
const session = await ort.InferenceSession.create('./BEST_alpha_stand.onnx');
function poseAt(track, time) {
  if (time <= 0) return HOME.slice();
  let pt = 0, pp = HOME;
  for (const f of track) {
    if (time <= f.t) {
      const u = (time - pt) / Math.max(f.t - pt, 1e-9), s = u * u * (3 - 2 * u);
      return f.pose.map((v, k) => pp[k] + (v - pp[k]) * s);
    }
    pt = f.t; pp = f.pose;
  }
  return track[track.length - 1].pose.slice();
}
const MOTIONS = {
  TAKE_A_BOW: { seconds: 2.4, watch: [5, 6], frames: [[0,{}],[0.7,{5:0.55,6:0.45}],[1.5,{5:0.55,6:0.45}],[2.2,{}]] },
  VICTORY_BOUNCE: { seconds: 2.6, watch: [2,3,4,11,12,13], frames: [[0,{}],[0.5,{2:-0.25,11:0.25,3:-0.2,12:0.2,4:0.15,13:-0.15}],[0.9,{}],[1.3,{2:-0.25,11:0.25,3:-0.2,12:0.2,4:0.15,13:-0.15}],[1.7,{}],[2.3,{}]] },
  HEAD_SHAKE: { seconds: 2.0, watch: [7], frames: [[0,{}],[0.4,{7:0.8}],[0.9,{7:-0.8}],[1.4,{7:0.5}],[1.8,{}]] },
};
const NAME = C.jointNames.filter(n => n !== 'mouth');
for (const [name, m] of Object.entries(MOTIONS)) {
  const track = m.frames.map(([t, off]) => ({ t, pose: HOME.map((h, k) => h + (off[k] ?? 0)) }));
  mj.mj_resetData(model, data); clearStairs(data, ADDR);
  data.qpos[D.freeQpos + 2] = 0.1231; data.qpos[D.freeQpos + 3] = 1;
  for (let i = 0; i < 14; i++) { data.qpos[D.qpos[i]] = HOME[i]; data.ctrl[i] = HOME[i]; }
  mj.mj_forward(model, data);
  let lastAction = new Array(14).fill(0);
  const lo = HOME.map(v => Infinity), hi = HOME.map(v => -Infinity);
  const ticks = Math.round(m.seconds * C.tickHz);
  for (let t = -25; t < ticks; t++) {
    const jp = [], jv = [];
    for (let k = 0; k < 14; k++) { jp.push(data.qpos[D.qpos[k]]); jv.push(data.qvel[D.dof[k]]); }
    const obs = buildObs([data.sensordata[GYRO], data.sensordata[GYRO+1], data.sensordata[GYRO+2]],
                         projectedGravity([data.qpos[D.freeQpos+3],data.qpos[D.freeQpos+4],data.qpos[D.freeQpos+5],data.qpos[D.freeQpos+6]]),
                         jp, jv, lastAction, command({}));
    const out = await session.run({ obs: new ort.Tensor('float32', obs, [1, 61]) });
    lastAction = Array.from(out.actions.data);
    const offsets = t >= 0 ? poseAt(track, t * DT) : null;
    for (let k = 0; k < 14; k++) {
      const base = HOME[k] + lastAction[k];
      const v = offsets ? base + (offsets[k] - HOME[k]) : base;
      data.ctrl[k] = Math.min(Math.max(v, LO[k]), HI[k]);
    }
    for (let s = 0; s < 4; s++) mj.mj_step(model, data);
    if (t >= 0) for (let k = 0; k < 14; k++) {
      const q = data.qpos[D.qpos[k]];
      if (q < lo[k]) lo[k] = q; if (q > hi[k]) hi[k] = q;
    }
  }
  console.log(name);
  for (const k of m.watch) {
    console.log(`  ${NAME[k].padEnd(16)} home ${HOME[k].toFixed(3)}  reached [${lo[k].toFixed(3)}, ${hi[k].toFixed(3)}]`);
  }
}
