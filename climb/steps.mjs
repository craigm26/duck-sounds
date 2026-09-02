// A flight whose steps are not all the same height — so a BLOCK can be the
// first one — plus the walking policy's own answer to the same question.
//
// PROVENANCE. The rollout loop here is sim/climb_lib.mjs's `attempt`, copied
// rather than imported because that one hard-codes a uniform flight
// (`{ count: 4, rise: h, run: 0.28, start: 0.12 }`, climb_lib.mjs) and a block
// in front of a step is by definition not uniform. Everything else is the same:
// the same scene.mjb the bench at 100.122.199.6:8770 serves, the same MuJoCo
// WASM in sim/node_modules, the same 25-tick settle under the stand policy that
// site/intent-specs.js calls non-optional, the same `pin the step joints every
// tick` rule from site/stairs.js, and the same strict "on the step" test.
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/steps.mjs
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';
import { findStairJoints, STAIR_Y, STEP_HALF_DEPTH, STEP_HALF_HEIGHT,
         STAIR_COUNT } from '../site/stairs.js';
import { buildTrack } from '../site/intent.mjs';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const { HOME, LO, HI, buildObs, projectedGravity, command, gaitTargets,
        findDuckJoints } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const model = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model), ADDR = findStairJoints(model);
if (!ADDR) throw new Error('this plant has no step joints');
let GYRO = 0;
for (let i = 0; i < model.nsensor; i++) if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;
const DT = 1 / C.tickHz;
const sessions = new Map();
const policy = async (name) => {
  if (!sessions.has(name)) sessions.set(name, await ort.InferenceSession.create(`./${name}`));
  return sessions.get(name);
};

const START = 0.12, RUN = 0.28;
/** Tops of the treads, in metres, front to back. A block is just a short first one. */
function layout(tops) {
  for (let i = 0; i < STAIR_COUNT; i++) {
    const a = ADDR[i];
    if (i >= tops.length) { data.qpos[a.x] = i * 1.5; data.qpos[a.z] = -5; }
    else {
      data.qpos[a.x] = START + i * RUN + STEP_HALF_DEPTH;
      data.qpos[a.z] = tops[i] - STEP_HALF_HEIGHT;
    }
    data.qvel[a.dx] = 0; data.qvel[a.dz] = 0;   // stairs.js: an unpinned step is a catapult
  }
}

const quat = () => [data.qpos[D.freeQpos + 3], data.qpos[D.freeQpos + 4],
                    data.qpos[D.freeQpos + 5], data.qpos[D.freeQpos + 6]];
function poseAt(tr, time) {
  if (time <= 0) return HOME.slice();
  let pt = 0, pp = HOME;
  for (const f of tr) {
    if (time <= f.t) { const u = (time - pt) / Math.max(f.t - pt, 1e-9), s = u * u * (3 - 2 * u);
      return f.pose.map((v, k) => pp[k] + (v - pp[k]) * s); }
    pt = f.t; pp = f.pose;
  }
  return tr[tr.length - 1].pose.slice();
}

/**
 * One attempt. `track` null means the policy drives alone — which is the only
 * way to ask what the TRAINED walk does with a step, as opposed to what an
 * authored gesture does.
 */
async function run({ tops, track = null, blend = 1, approach = 0, gap = 0.06, side = 0,
                     policyName = 'BEST_alpha_stand.onnx', seconds = null, drop = 0.12,
                     gait = false }) {
  // TWO ACTION MAPS, AND USING THE WRONG ONE IS SILENT. climb_lib.mjs drives
  // the STAND policy as `HOME + action`, raw. sim/walk.mjs drives the WALKING
  // policy through `gaitTargets` — `HOME + actionScale * action`, low-passed
  // per joint by ALPHA and clamped. Driven the stand way, alpha_walking simply
  // stands: measured, the trunk moved 1 mm in six seconds at every step height,
  // which reads like "the walk cannot climb" and is really "that was not the
  // walk". `gait` selects the map, and it is on for the walking policy only.
  const net = await policy(policyName);
  mj.mj_resetData(model, data);
  layout(tops);
  data.qpos[D.freeQpos] = START - 0.07 - gap;     // intent-specs.js: true standoff is 0.07 + gap
  data.qpos[D.freeQpos + 1] = STAIR_Y + side;
  data.qpos[D.freeQpos + 2] = drop; data.qpos[D.freeQpos + 3] = 1;
  for (let i = 0; i < 14; i++) { data.qpos[D.qpos[i]] = HOME[i]; data.ctrl[i] = HOME[i]; }
  mj.mj_forward(model, data);
  let la = new Array(14).fill(0), previous = null;
  const cmd = command({ vx: approach });
  let peakZ = 0;
  const step = async (off) => {
    layout(tops);
    const q = quat(); const jp = [], jv = [];
    for (let k = 0; k < 14; k++) { jp.push(data.qpos[D.qpos[k]]); jv.push(data.qvel[D.dof[k]]); }
    const obs = buildObs([data.sensordata[GYRO], data.sensordata[GYRO + 1], data.sensordata[GYRO + 2]],
                         projectedGravity(q), jp, jv, la, cmd);
    const r = await net.run({ [net.inputNames[0]]: new ort.Tensor('float32', obs, [1, 61]) });
    la = Array.from(r[net.outputNames[0]].data);
    if (gait) {
      previous = gaitTargets(la, previous);
      for (let k = 0; k < 14; k++) {
        const v = previous[k] + (off ? (off[k] - HOME[k]) * blend : 0);
        data.ctrl[k] = Math.min(Math.max(v, LO[k]), HI[k]);
      }
    } else {
      for (let k = 0; k < 14; k++) {
        const v = HOME[k] + la[k] + (off ? (off[k] - HOME[k]) * blend : 0);
        data.ctrl[k] = Math.min(Math.max(v, LO[k]), HI[k]);
      }
    }
    for (let s = 0; s < 4; s++) mj.mj_step(model, data);
    if (data.qpos[D.freeQpos + 2] > peakZ) peakZ = data.qpos[D.freeQpos + 2];
  };
  for (let t = 0; t < 25; t++) await step(null);          // the settle, intent-specs.js
  const total = seconds ?? (track ? track[track.length - 1].t + 0.8 : 6);
  for (let t = 0; t * DT < total; t++) await step(track ? poseAt(track, t * DT) : null);
  for (let t = 0; t < 50; t++) await step(null);          // must still be there a second later

  const x = data.qpos[D.freeQpos], z = data.qpos[D.freeQpos + 2];
  const up = projectedGravity(quat())[2] < -0.90;
  // Which tread it is standing on: the highest one whose top the feet are at or
  // above and whose face the trunk is past. 0 means the floor.
  let tread = 0, feet = 0;
  for (let i = tops.length - 1; i >= 0; i--) {
    let n = 0;
    for (let g = 0; g < model.ngeom; g++) {
      const nm = model.geom(g).name || '';
      if (!/foot_collision|sole/.test(nm)) continue;
      if (data.geom_xpos[g * 3 + 2] > tops[i] - 0.005 && data.geom_xpos[g * 3] > START + i * RUN - 0.02) n++;
    }
    if (n >= 2 && x > START + i * RUN && (z - tops[i]) > 0.095 && up) { tread = i + 1; feet = n; break; }
  }
  return { tread, feet, x, z, up, peakZ };
}

// ── the runs ───────────────────────────────────────────────────────────────
const MOVES = {
  step_up: '../site/intent-stepup.json', lever_up: '../site/intent-lever.json',
  riser_up: '../site/intent-riser.json', climb: '../site/intent-climb.json',
};
function moveOf(name) {
  const j = JSON.parse(fs.readFileSync(MOVES[name], 'utf8'));
  const p = j.params || {};
  return { track: j.keyframes ?? buildTrack(p, HOME),
           blend: j.blend ?? p.blend ?? 1, approach: j.approach ?? p.approach ?? 0,
           gap: j.gap ?? p.gap ?? 0.06, side: j.side ?? 0,
           policyName: j.policy || 'BEST_alpha_stand.onnx' };
}
const out = { when: new Date().toISOString(), plant: 'sim/scene.mjb', run: RUN, start: START, rows: [] };
const say = (label, tops, r) => {
  const line = `${label.padEnd(34)} tread ${r.tread}  trunk x ${(r.x * 1000).toFixed(0).padStart(5)}mm `
    + `z ${(r.z * 1000).toFixed(0).padStart(4)}mm  peak z ${(r.peakZ * 1000).toFixed(0).padStart(4)}mm  `
    + (r.up ? 'upright' : 'TOPPLED');
  console.log(line);
  out.rows.push({ label, topsMm: tops.map(t => Math.round(t * 1000)), ...r });
};

// vx = 0.25, NOT the 0.15 sim/walk.mjs uses. In THIS plant (scene.mjb) a
// commanded 0.15 m/s is inside the policy's deadband: measured over 6 s on a
// flat floor, vx 0.15 moves the trunk 8 mm and vx 0.25 moves it 575 mm
// (climb/probe-walk.mjs). walk.mjs gets 0.625 m at vx 0.15 because it runs
// scene.xml, a different plant — that duck also stands 135 mm tall, not 116.
console.log('\n== the trained walk against a flight (alpha_walking, vx = 0.25 m/s, 6 s) ==');
for (const mm of [0, 2, 5, 10, 20, 40]) {
  const tops = [mm, mm * 2, mm * 3, mm * 4].map(v => v / 1000);
  say(`walk, ${String(mm).padStart(3)} mm steps`, tops,
      await run({ tops, policyName: 'alpha_walking.onnx', gait: true, approach: 0.25, gap: 0.06, seconds: 6 }));
}

console.log('\n== a block as the first step, then a real riser ==');
for (const b of [60, 90]) {
  for (const h of [170]) {
    const tops = [b / 1000, h / 1000];
    for (const name of ['lever_up', 'riser_up', 'climb']) {
      const m = moveOf(name);
      say(`${name} onto a ${b} mm block (then ${h})`, tops, await run({ tops, ...m }));
    }
    say(`walk onto a ${b} mm block (then ${h})`, tops,
        await run({ tops, policyName: 'alpha_walking.onnx', gait: true, approach: 0.25, seconds: 6 }));
  }
}

console.log('\n== how tall a FIRST step can be, with nothing behind it ==');
for (const name of ['lever_up', 'riser_up', 'climb']) {
  const m = moveOf(name);
  for (const mm of [2, 5, 10, 20]) {
    say(`${name}, lone ${String(mm).padStart(3)} mm block`, [mm / 1000], await run({ tops: [mm / 1000], ...m }));
  }
}
fs.writeFileSync('../climb/steps-results.json', JSON.stringify(out, null, 1) + '\n');
console.log('\nwrote climb/steps-results.json');
