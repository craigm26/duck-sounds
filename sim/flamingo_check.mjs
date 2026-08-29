// Does RemiFabre/microduck-flamingo-cycle do what its manifest claims?
//
// The card publishes numbers — 0.09 m of lifted foot, 1.5 s each way, a 10 s
// hold — so they get checked on the canon plant before the policy is carried
// into the corpus. Command is the twist block: [flag, side, 0].
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const model = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model);
let GYRO = 0;
for (let i = 0; i < model.nsensor; i++) if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;
// The foot sites, to measure the lift the card claims.
const SITE = {};
for (let i = 0; i < model.nsite; i++) SITE[model.site(i).name] = i;

const flamingo = await ort.InferenceSession.create('./community/flamingo-cycle/policy.onnx');
const stand = await ort.InferenceSession.create('./BEST_alpha_stand.onnx');

function reset() {
  mj.mj_resetData(model, data);
  data.qpos[D.freeQpos + 2] = 0.1231; data.qpos[D.freeQpos + 3] = 1;
  for (let i = 0; i < 14; i++) { data.qpos[D.qpos[i]] = HOME[i]; data.ctrl[i] = HOME[i]; }
  mj.mj_forward(model, data);
}

async function run(timeline, seconds) {
  reset();
  let last = new Array(14).fill(0);
  const trace = [];
  const settle = 25;
  for (let t = -settle; t < Math.round(seconds * C.tickHz); t++) {
    const secs = t / C.tickHz;
    let flag = 0, side = 0;
    for (const [at, f, s] of timeline) if (secs >= at) { flag = f; side = s; }
    const cmd = command(t >= 0 ? { vx: flag, vy: side } : {});
    const f = D.freeQpos;
    const q = [data.qpos[f + 3], data.qpos[f + 4], data.qpos[f + 5], data.qpos[f + 6]];
    const jp = [], jv = [];
    for (let k = 0; k < 14; k++) { jp.push(data.qpos[D.qpos[k]]); jv.push(data.qvel[D.dof[k]]); }
    const obs = buildObs([data.sensordata[GYRO], data.sensordata[GYRO + 1], data.sensordata[GYRO + 2]],
                         projectedGravity(q), jp, jv, last, cmd);
    const net = t < 0 ? stand : flamingo;
    const out = await net.run({ obs: new ort.Tensor('float32', obs, [1, 61]) });
    last = Array.from(out[net.outputNames[0]].data);
    for (let k = 0; k < 14; k++) data.ctrl[k] = Math.min(Math.max(HOME[k] + last[k], LO[k]), HI[k]);
    for (let s = 0; s < 4; s++) mj.mj_step(model, data);
    if (t >= 0) {
      const lz = data.site_xpos[SITE['left_foot'] * 3 + 2], rz = data.site_xpos[SITE['right_foot'] * 3 + 2];
      trace.push({ t: secs, flag, side, lz, rz, z: data.qpos[f + 2] });
    }
  }
  return trace;
}

const r = v => Math.round(v * 1000) / 1000;
for (const [label, side] of [['lift LEFT leg (side +1)', 1], ['lift RIGHT leg (side -1)', -1]]) {
  // stand 1 s, lift and hold 8 s, lower 3 s
  const tl = [[0, 0, side], [1.0, 1, side], [9.0, 0, side]];
  const tr = await run(tl, 12.0);
  const lifted = side > 0 ? 'lz' : 'rz';
  const planted = side > 0 ? 'rz' : 'lz';
  const ground = Math.min(...tr.slice(0, 40).map(f => f[lifted]));
  const hold = tr.filter(f => f.t >= 3.0 && f.t <= 9.0);
  const peak = Math.max(...hold.map(f => f[lifted] - ground));
  const mean = hold.reduce((a, f) => a + (f[lifted] - ground), 0) / hold.length;
  const up = tr.find(f => f.t >= 1.0 && (f[lifted] - ground) > 0.07);
  const back = tr.filter(f => f.t > 9.0).find(f => (f[lifted] - ground) < 0.01);
  const fell = Math.min(...tr.map(f => f.z)) < 0.06;
  console.log(`${label}: lift peak ${r(peak)} m, mean over hold ${r(mean)} m, `
    + `up in ${up ? r(up.t - 1.0) : '—'} s, down in ${back ? r(back.t - 9.0) : '—'} s, `
    + `planted foot stays ${r(Math.max(...hold.map(f => f[planted] - ground)))} m, ${fell ? 'FELL' : 'upright'}`);
}
