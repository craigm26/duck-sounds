// Can huey stand on dewey?
//
// WHY THIS IS THE ONLY PLACE THE QUESTION CAN BE ASKED. duckbench.mjs says it:
// a hardware Microduck's observation is 61 values with no slot for another
// duck, so two real ducks are two blind agents sharing a floor. Two ducks in
// ONE MuJoCo model perceive each other through contact, which is what standing
// on one another is. scene_multiduck.mjb is that model (built by
// sim/build_multiduck.py, compiled by sim/compile_multiduck.mjs) — huey and
// dewey, spawned abreast 0.5 m apart along y, both facing +x.
//
// The bench at 100.122.199.6:8770 is NOT running it: /health reports
// `"ducks":[{"name":"duck",...}]` and one plant, scene.mjb. duckbench-node.mjs
// takes DUCKBENCH_SCENE, so a second bench could serve the multiduck world —
// but no endpoint can place one duck on top of another, so this is in-process.
//
// The per-duck address lookup is duckbench-core.mjs's `discoverDucks`
// (every joint, actuator and gyro of a duck is its MJCF prefix followed by the
// single-duck name), reimplemented here rather than imported because that
// function is private to the bench.
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/twoduck.mjs
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const { HOME, LO, HI, buildObs, projectedGravity, command } = makeLoop(C);
const JOINTS = C.jointNames.filter(n => n !== 'mouth');

const mj = await load();
mj.FS.writeFile('/m.mjb', new Uint8Array(fs.readFileSync('scene_multiduck.mjb')));
const model = mj.MjModel.mj_loadBinary('/m.mjb', new mj.MjVFS());
const data = new mj.MjData(model);

function findDucks() {
  const out = [];
  for (let b = 0; b < model.nbody; b++) {
    const name = model.body(b).name || '';
    if (!name.endsWith('trunk_base')) continue;
    const prefix = name.slice(0, -'trunk_base'.length);
    const qpos = [], dof = [], ctrl = [];
    for (const j of JOINTS) {
      let jj = -1; for (let i = 0; i < model.njnt; i++) if (model.jnt(i).name === prefix + j) { jj = i; break; }
      let aa = -1; for (let i = 0; i < model.nu; i++) if (model.actuator(i).name === prefix + j) { aa = i; break; }
      if (jj < 0 || aa < 0) throw new Error(`missing ${prefix}${j}`);
      qpos.push(model.jnt_qposadr[jj]); dof.push(model.jnt_dofadr[jj]); ctrl.push(aa);
    }
    let free = -1;
    for (let i = 0; i < model.njnt; i++) {
      if (model.jnt_type[i] === 0 && model.body(model.jnt_bodyid[i]).name === name) { free = model.jnt_qposadr[i]; break; }
    }
    let gyro = 0;
    for (let i = 0; i < model.nsensor; i++) if (model.sensor(i).name === prefix + 'imu_ang_vel') gyro = model.sensor(i).adr;
    out.push({ name: prefix ? prefix.slice(0, -1) : 'duck', prefix, qpos, dof, ctrl, free, gyro });
  }
  return out;
}
const DUCKS = findDucks();
console.log('ducks in this plant:', DUCKS.map(d => d.name).join(', '));

const sessions = new Map();
const policy = async n => {
  if (!sessions.has(n)) sessions.set(n, await ort.InferenceSession.create(`./${n}`));
  return sessions.get(n);
};
const quatOf = d => [data.qpos[d.free + 3], data.qpos[d.free + 4], data.qpos[d.free + 5], data.qpos[d.free + 6]];
const place = (d, x, y, z) => {
  data.qpos[d.free] = x; data.qpos[d.free + 1] = y; data.qpos[d.free + 2] = z;
  data.qpos[d.free + 3] = 1; data.qpos[d.free + 4] = 0; data.qpos[d.free + 5] = 0; data.qpos[d.free + 6] = 0;
  for (let k = 0; k < 14; k++) { data.qpos[d.qpos[k]] = HOME[k]; data.ctrl[d.ctrl[k]] = HOME[k]; }
};

/**
 * One trial. `stack` is the height huey is dropped from above dewey's trunk.
 * Both ducks run their own policy every tick; ONE mj_step advances both, which
 * is the entire reason this is possible at all.
 */
async function trial({ policies, stack, seconds = 4, vx = 0 }) {
  mj.mj_resetData(model, data);
  const [lower, upper] = DUCKS;                     // huey is index 0, dewey index 1
  place(lower, 0, 0, 0.12);
  place(upper, 0, 0, 0.12 + stack);                 // directly above, not abreast
  mj.mj_forward(model, data);
  const nets = await Promise.all(DUCKS.map((d, i) => policy(policies[i])));
  const last = DUCKS.map(() => new Array(14).fill(0));
  const cmd = command({ vx });
  const trace = [];
  for (let t = 0; t < Math.round(seconds * C.tickHz); t++) {
    for (let i = 0; i < DUCKS.length; i++) {
      const d = DUCKS[i];
      const jp = [], jv = [];
      for (let k = 0; k < 14; k++) { jp.push(data.qpos[d.qpos[k]]); jv.push(data.qvel[d.dof[k]]); }
      const obs = buildObs([data.sensordata[d.gyro], data.sensordata[d.gyro + 1], data.sensordata[d.gyro + 2]],
                           projectedGravity(quatOf(d)), jp, jv, last[i], cmd);
      const r = await nets[i].run({ [nets[i].inputNames[0]]: new ort.Tensor('float32', obs, [1, 61]) });
      last[i] = Array.from(r[nets[i].outputNames[0]].data);
      for (let k = 0; k < 14; k++) {
        data.ctrl[d.ctrl[k]] = Math.min(Math.max(HOME[k] + last[i][k], LO[k]), HI[k]);
      }
    }
    for (let s = 0; s < 4; s++) mj.mj_step(model, data);   // one step advances both
    if (t % 10 === 0) trace.push(DUCKS.map(d => +(data.qpos[d.free + 2] * 1000).toFixed(0)));
  }
  // Where the upper duck's FEET ended up is the number that matters: that is
  // the height a duck reached by standing on another duck, and it is what a
  // riser would have to be no taller than.
  const footZ = d => {
    let lo = Infinity;
    for (let g = 0; g < model.ngeom; g++) {
      const nm = model.geom(g).name || '';
      if (!nm.startsWith(d.prefix) || !/foot_collision|sole/.test(nm)) continue;
      lo = Math.min(lo, data.geom_xpos[g * 3 + 2]);
    }
    return Number.isFinite(lo) ? +(lo * 1000).toFixed(0) : null;
  };
  return DUCKS.map(d => ({
    name: d.name, footMm: footZ(d),
    x: +(data.qpos[d.free] * 1000).toFixed(0),
    y: +(data.qpos[d.free + 1] * 1000).toFixed(0),
    z: +(data.qpos[d.free + 2] * 1000).toFixed(0),
    up: projectedGravity(quatOf(d))[2] < -0.90,
  })).concat([{ trace: trace.slice(0, 6) }]);
}

const STAND = 'BEST_alpha_stand.onnx', SIT = 'BEST_alpha_sitstand.onnx';
const rows = [];
// A settled duck's trunk is 116 mm up (measured on this plant); its back is
// lower than that. The three drops bracket "resting on the back" from just
// touching to a real fall.
// NOTE ON WHO IS ON TOP. DUCKS[0] is huey and DUCKS[1] is dewey, in MJCF
// order, and this places DUCKS[0] on the floor and drops DUCKS[1] on it — so
// it is DEWEY who climbs onto HUEY. Saying it the other way round would be a
// caption that disagrees with the run.
const CASES = [];
for (const mm of (process.env.STACK || '110,115,120,125,130,150').split(',').map(Number)) {
  CASES.push([`both standing, dewey dropped ${mm} mm onto huey`, [STAND, STAND], mm / 1000]);
}
CASES.push(['huey on the sit policy, dewey dropped 100 mm', [SIT, STAND], 0.100]);
CASES.push(['huey on the sit policy, dewey dropped 130 mm', [SIT, STAND], 0.130]);
for (const [label, policies, stack] of CASES) {
  const r = await trial({ policies, stack, seconds: +(process.env.DURATION || 4) });
  const [a, b] = r;
  console.log(`${label.padEnd(46)}  ${a.name}: trunk ${String(a.z).padStart(4)}mm feet ${String(a.footMm).padStart(3)}mm `
            + `${a.up ? 'upright' : 'TOPPLED'}   ${b.name}: trunk ${String(b.z).padStart(4)}mm feet `
            + `${String(b.footMm).padStart(3)}mm ${b.up ? 'upright' : 'TOPPLED'} (moved x ${b.x} y ${b.y})`);
  rows.push({ label, policies, stackMm: stack * 1000, result: r });
}
fs.writeFileSync('../climb/twoduck-results.json', JSON.stringify({
  plant: 'sim/scene_multiduck.mjb', ducks: DUCKS.map(d => d.name),
  note: 'duck 0 placed at the origin, duck 1 dropped directly above it; both under their own policy, '
      + 'one integrator, 4 s',
  when: new Date().toISOString(), rows,
}, null, 1) + '\n');
console.log('\nwrote climb/twoduck-results.json');
