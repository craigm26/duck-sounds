// The duck bench: this machine's physics, offered to Duck Studio over the LAN.
//
// WHY THIS EXISTS. Duck Studio can import a policy — from Pollen, or from
// Hugging Face — and then do nothing with it. It has no physics: an iPhone has
// no MuJoCo, so every clip it shows was recorded HERE and baked into duckkit
// at build time. Import flamingo-cycle on the phone and there is nothing to
// press. This serves the two things the phone cannot do for itself: RECORD a
// policy into a clip, and MEASURE how often it works.
//
// WHAT IT IS NOT. It does not train. The Hailo on this machine is an inference
// ASIC with no training path at all, and mjlab — what Pollen train with — wants
// a GPU. Training on this box would mean a CPU PPO against plain MuJoCo; at the
// measured 1.66 ms a tick that is roughly 46 core-hours for 100M steps, an
// overnight job rather than an impossibility, but it is not this file.
//
// SAFETY. Policies are chosen from a scan of this directory by NAME. A request
// cannot name a path: no traversal, no URL, nothing fetched. Set
// DUCKBENCH_TOKEN to require `Authorization: Bearer <token>`; without it the
// bench is open to whoever is on your network, which is fine for a bench on a
// desk and not fine on a café's wifi.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import { makeLoop } from '../site/duckloop.mjs';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);
const PORT = +(process.env.DUCKBENCH_PORT || 8770);
const TOKEN = process.env.DUCKBENCH_TOKEN || null;
const STAND = 'BEST_alpha_stand.onnx';

const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const model = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model);
let GYRO = 0;
for (let i = 0; i < model.nsensor; i++) {
  if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;
}

/** Every .onnx this bench will run, by bare name — the whole allow-list. */
function catalogue() {
  const out = new Map();
  for (const file of fs.readdirSync('.')) {
    if (file.endsWith('.onnx')) out.set(file, file);
  }
  const community = 'community';
  if (fs.existsSync(community)) {
    for (const dir of fs.readdirSync(community)) {
      const candidate = path.join(community, dir, 'policy.onnx');
      if (fs.existsSync(candidate)) out.set(`${dir}/policy.onnx`, candidate);
    }
  }
  return out;
}

const sessions = new Map();
async function policy(name) {
  const known = catalogue();
  if (!known.has(name)) throw new Error(`unknown policy: ${name}`);
  const file = known.get(name);
  if (!sessions.has(file)) sessions.set(file, await ort.InferenceSession.create('./' + file));
  return sessions.get(file);
}

function reset(z = 0.1231) {
  mj.mj_resetData(model, data);
  data.qpos[D.freeQpos + 2] = z; data.qpos[D.freeQpos + 3] = 1;
  for (let i = 0; i < 14; i++) { data.qpos[D.qpos[i]] = HOME[i]; data.ctrl[i] = HOME[i]; }
  mj.mj_forward(model, data);
}

/**
 * The canon loop, the same one every recorded clip came from: a settle under
 * the standing policy with a neutral command, then the training path — target
 * = HOME + action at scale 1.0, no filter — and the servo travel as the clamp.
 */
async function rollout({ name, seconds, schedule, settle = 25, drop = 0.1231 }) {
  const net = await policy(name);
  const settling = await policy(STAND);
  reset(drop);
  let last = new Array(14).fill(0);
  const frames = [], roots = [], commands = [];
  const ticks = Math.round(seconds * C.tickHz);
  for (let t = -settle; t < ticks; t++) {
    const secs = t / C.tickHz;
    const wanted = t >= 0 ? commandAt(schedule, secs) : {};
    const cmd = command(wanted);
    const f = D.freeQpos;
    const q = [data.qpos[f + 3], data.qpos[f + 4], data.qpos[f + 5], data.qpos[f + 6]];
    const jp = [], jv = [];
    for (let k = 0; k < 14; k++) { jp.push(data.qpos[D.qpos[k]]); jv.push(data.qvel[D.dof[k]]); }
    const obs = buildObs([data.sensordata[GYRO], data.sensordata[GYRO + 1], data.sensordata[GYRO + 2]],
                         projectedGravity(q), jp, jv, last, cmd);
    const runner = t < 0 ? settling : net;
    const out = await runner.run({ obs: new ort.Tensor('float32', obs, [1, 61]) });
    last = Array.from(out[runner.outputNames[0]].data);
    for (let k = 0; k < 14; k++) {
      data.ctrl[k] = Math.min(Math.max(HOME[k] + last[k], LO[k]), HI[k]);
    }
    for (let s = 0; s < 4; s++) mj.mj_step(model, data);
    if (t >= 0) {
      const after = [];
      for (let k = 0; k < 14; k++) {
        after.push(Math.min(Math.max(data.qpos[D.qpos[k]], LO[k]), HI[k]));
      }
      frames.push(after.map(r4));
      roots.push([data.qpos[f], data.qpos[f + 1], data.qpos[f + 2],
                  data.qpos[f + 3], data.qpos[f + 4], data.qpos[f + 5], data.qpos[f + 6]].map(r4));
      commands.push(cmd.slice(0, 3).map(r4));
    }
  }
  return { frames, roots, commands };
}

/** A schedule is a list of [atSeconds, {vx, vy, vyaw}] — the last one that has begun wins. */
function commandAt(schedule, secs) {
  let current = {};
  for (const [at, values] of schedule || []) if (secs >= at) current = values;
  return current;
}

const r4 = v => Math.round(v * 10000) / 10000;
const upright = root => {
  const [, , , w, x, y] = root;
  return -(1 - 2 * (x * x + y * y)) < -0.5;
};

async function handle(url, body) {
  if (url.pathname === '/health') {
    return {
      bench: 'duck-bench/1',
      plant: 'scene.mjb — Pollen robot_allcollisions, training parameters',
      tickHz: C.tickHz,
      cores: (await import('node:os')).cpus().length,
      policies: [...catalogue().keys()].sort(),
      records: true, measures: true,
      trains: false,
      trainsWhy: 'The accelerator here is an inference ASIC, and mjlab wants a GPU. '
               + 'Recording and measuring are what this machine is for.',
    };
  }
  if (url.pathname === '/record') {
    const seconds = Math.min(Math.max(+body.seconds || 3, 0.2), 30);
    const run = await rollout({ name: body.policy, seconds, schedule: body.schedule });
    const last = run.roots[run.roots.length - 1];
    return {
      format: 'duck-intent-clips/3',
      hz: C.tickHz,
      joints: C.jointNames.filter(n => n !== 'mouth'),
      policy: body.policy,
      frames: run.frames, roots: run.roots, commands: run.commands,
      endsUpright: upright(last), endHeight: last[2],
    };
  }
  if (url.pathname === '/measure') {
    // The randomisation is Pollen's own, as measure_success.mjs uses it: the
    // drop height is what a bench can vary without touching the model.
    const rollouts = Math.min(Math.max(+body.rollouts || 8, 1), 32);
    const seconds = Math.min(Math.max(+body.seconds || 3, 0.2), 30);
    let ok = 0; const heights = [];
    for (let i = 0; i < rollouts; i++) {
      const drop = 0.12 + (0.01 * i) / Math.max(rollouts - 1, 1);
      const run = await rollout({ name: body.policy, seconds, schedule: body.schedule, drop });
      const last = run.roots[run.roots.length - 1];
      heights.push(last[2]);
      if (upright(last) && last[2] >= 0.100) ok++;
    }
    heights.sort((a, b) => a - b);
    return {
      policy: body.policy, rollouts, achieves: ok,
      criterion: 'ends standing, trunk at least 100 mm up',
      randomised: 'drop height 0.12-0.13 m (Pollen’s range)',
      medianHeight: r4(heights[heights.length >> 1]), worstHeight: r4(heights[0]),
    };
  }
  return null;
}

http.createServer((req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  const send = (code, value) => {
    const text = JSON.stringify(value);
    res.writeHead(code, { 'content-type': 'application/json',
                          'access-control-allow-origin': '*',
                          'access-control-allow-headers': 'authorization,content-type' });
    res.end(text);
  };
  if (req.method === 'OPTIONS') return send(204, {});
  if (TOKEN && req.headers.authorization !== `Bearer ${TOKEN}`) {
    return send(401, { error: 'this bench wants its token' });
  }
  let raw = '';
  req.on('data', chunk => { raw += chunk; if (raw.length > 1e6) req.destroy(); });
  req.on('end', async () => {
    let body = {};
    if (raw) { try { body = JSON.parse(raw); } catch { return send(400, { error: 'body is not JSON' }); } }
    try {
      const answer = await handle(url, body);
      if (!answer) return send(404, { error: `no ${url.pathname} here` });
      send(200, answer);
    } catch (error) {
      send(400, { error: String(error.message || error) });
    }
  });
}).listen(PORT, '0.0.0.0', () => {
  console.log(`duck bench on http://0.0.0.0:${PORT} — ${TOKEN ? 'token required' : 'OPEN on this network'}`);
  console.log(`policies: ${[...catalogue().keys()].sort().join(', ')}`);
});
