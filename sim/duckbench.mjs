// The duck bench: this machine's physics, offered to Duck Studio over the LAN.
//
// WHY THIS EXISTS. Duck Studio can import a policy — from Pollen, or from
// Hugging Face — and then do nothing with it. It has no physics: an iPhone has
// no MuJoCo, so every clip it shows was recorded HERE and baked into duckkit
// at build time. Import flamingo-cycle on the phone and there is nothing to
// press. This serves the two things the phone cannot do for itself: RECORD a
// policy into a clip, and MEASURE how often it works.
//
// AND IT IS A TRANSPORT. rokbenko/quackd drives a Microduck through a
// `DuckTransport` — get_state, get_frame, send_intent, stop, and now() — and
// composes verbs on top of it. now() is the load-bearing one: the TRANSPORT
// OWNS TIME, so the same steering loop runs at wall-clock speed against a real
// duck and at sim speed against this one without a verb knowing which it has.
// So /state, /intent, /stop and /now sit alongside /record and /measure, over
// one persistent world that stays standing between requests. /policy is the
// odd one out: quackd's learned-verbs note calls a live policy swap the one
// upstream API it needs and does not have. On hardware that is somebody else's
// firmware; in sim it is a map lookup, so here it exists.
//
// NO /frame HERE. quackd's transport also hands over a camera image, and this
// process has no camera and no renderer: rendering lives in duckvision.py, on
// MuJoCo 3.12 (pip), while this runs MuJoCo 3.5.1 (WASM) so that clips stay
// canon. Wiring the two together is a job for whatever composes them, not a
// stub that returns a grey rectangle.
//
// WHAT IT IS NOT. It does not train. The Hailo on this machine is an inference
// ASIC with no training path at all, and mjlab — what Pollen train with — wants
// a GPU. Training on this box would mean a CPU PPO against plain MuJoCo; at the
// measured 1.66 ms a tick that is roughly 46 core-hours for 100M steps, an
// overnight job rather than an impossibility, but it is not this file.
//
// SAFETY. Policies are chosen from a scan of this directory by NAME. A request
// cannot name a path: no traversal, no URL, nothing fetched — and /policy, the
// one endpoint that lets a caller pick, goes through the same lookup as
// everything else. Set DUCKBENCH_TOKEN to require `Authorization: Bearer
// <token>`; without it the bench is open to whoever is on your network, which
// is fine for a bench on a desk and not fine on a café's wifi.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import { makeLoop } from '../site/duckloop.mjs';
import { declaredDefaultPose } from './onnx_meta.mjs';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);
const PORT = +(process.env.DUCKBENCH_PORT || 8770);
const TOKEN = process.env.DUCKBENCH_TOKEN || null;
const STAND = 'BEST_alpha_stand.onnx';
// The settle every recorded clip opens with — half a second under the standing
// policy, so the duck is on its feet before anything asks it to move. The live
// world opens with the same one, or /intent's first tick would be steering a
// duck that is still falling the 1 mm from its drop height.
const SETTLE_TICKS = 25;

const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const model = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model);
// THE BALL IS THE ONLY OTHER THING WORTH ADDRESSING IN THIS WORLD. It is
// Pollen's own (radius 0.05, condim 6 so it actually decelerates), and a
// steering loop needs to put it somewhere and then be scored on reaching it.
const BALL = (() => {
  for (let j = 0; j < model.njnt; j++) {
    if (model.jnt_type[j] !== 0) continue;                 // mjJNT_FREE
    const adr = model.jnt_qposadr[j];
    if (adr === D.freeQpos) continue;                      // the duck
    if (model.body(model.jnt_bodyid[j]).name === 'ball') return { adr, dof: model.jnt_dofadr[j] };
  }
  return null;
})();
const BALL_RADIUS = 0.05;

function ballOf(d) {
  if (!BALL) return null;
  return [d.qpos[BALL.adr], d.qpos[BALL.adr + 1], d.qpos[BALL.adr + 2]].map(r4);
}

/** Put the ball down and stop it dead. */
function placeBall(d, x, y, z = BALL_RADIUS) {
  if (!BALL) throw new Error('this world has no ball');
  d.qpos[BALL.adr] = x; d.qpos[BALL.adr + 1] = y; d.qpos[BALL.adr + 2] = z;
  d.qpos[BALL.adr + 3] = 1;
  for (let k = 4; k < 7; k++) d.qpos[BALL.adr + k] = 0;
  for (let k = 0; k < 6; k++) d.qvel[BALL.dof + k] = 0;
  mj.mj_forward(model, d);
}
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

/**
 * A policy, loaded once, WITH THE NEUTRAL POSE IT WAS TRAINED AGAINST.
 *
 * The reference is not decoration. A policy's action is an offset from its own
 * neutral, and its observation's joint block is a deviation from that same
 * pose; Pollen's ten files all declare one equal to HOME, which is why using
 * HOME everywhere went unnoticed, but the community `headspin.onnx` declares
 * neck_pitch 0.220 and head_pitch 0.680 where HOME has 0.349 and 0.349. Every
 * other runner here already honours it (record_intents.mjs, ac_check*.mjs) and
 * this one did not — which mattered the moment /policy let a caller swap to
 * that very file mid-run.
 */
async function policy(name) {
  const known = catalogue();
  if (!known.has(name)) throw new Error(`unknown policy: ${name}`);
  const file = known.get(name);
  if (!sessions.has(file)) {
    sessions.set(file, {
      name,
      net: await ort.InferenceSession.create('./' + file),
      reference: declaredDefaultPose('./' + file, HOME) ?? HOME,
    });
  }
  return sessions.get(file);
}

function reset(d, z = 0.1231) {
  mj.mj_resetData(model, d);
  d.qpos[D.freeQpos + 2] = z; d.qpos[D.freeQpos + 3] = 1;
  for (let i = 0; i < 14; i++) { d.qpos[D.qpos[i]] = HOME[i]; d.ctrl[i] = HOME[i]; }
  mj.mj_forward(model, d);
}

// The plant's own timestep, read out of the plant rather than assumed: one
// mj_step of the settled duck advances the clock by exactly this. The control
// loop below runs at C.tickHz and takes 1/tickHz worth of substeps per tick —
// which is the 4 that used to sit in the loop as a bare literal, and which is
// now checked against the model every boot instead of hoped for.
reset(data);
mj.mj_step(model, data);
const TIMESTEP = data.time;
const SUBSTEPS = Math.round(1 / C.tickHz / TIMESTEP);
if (!(SUBSTEPS >= 1) || Math.abs(SUBSTEPS * TIMESTEP - 1 / C.tickHz) > 1e-9) {
  throw new Error(`${C.tickHz} Hz control does not divide a ${TIMESTEP} s timestep`);
}

/**
 * One control tick on `d`: observe, run the policy, clamp to servo travel, step.
 *
 * THE SAME TICK FOR EVERY CALLER. /record, /measure and /intent all come
 * through here, so a policy that measured 16/16 behaves identically when
 * quackd steers it live. Two loops would drift apart on the first fix.
 */
async function tick(d, loaded, last, cmd) {
  const { net, reference } = loaded;
  const f = D.freeQpos;
  const q = [d.qpos[f + 3], d.qpos[f + 4], d.qpos[f + 5], d.qpos[f + 6]];
  const jp = [], jv = [];
  for (let k = 0; k < 14; k++) { jp.push(d.qpos[D.qpos[k]]); jv.push(d.qvel[D.dof[k]]); }
  const obs = buildObs([d.sensordata[GYRO], d.sensordata[GYRO + 1], d.sensordata[GYRO + 2]],
                       projectedGravity(q), jp, jv, last, cmd, reference);
  const out = await net.run({ obs: new ort.Tensor('float32', obs, [1, 61]) });
  const action = Array.from(out[net.outputNames[0]].data);
  for (let k = 0; k < 14; k++) {
    d.ctrl[k] = Math.min(Math.max(reference[k] + action[k], LO[k]), HI[k]);
  }
  for (let s = 0; s < SUBSTEPS; s++) mj.mj_step(model, d);
  return action;
}

/**
 * The canon loop, the same one every recorded clip came from: a settle under
 * the standing policy with a neutral command, then the training path — target
 * = HOME + action at scale 1.0, no filter — and the servo travel as the clamp.
 */
async function rollout({ name, seconds, schedule, settle = SETTLE_TICKS, drop = 0.1231 }) {
  const net = await policy(name);
  const settling = await policy(STAND);
  reset(data, drop);
  let last = new Array(14).fill(0);
  const frames = [], roots = [], commands = [];
  const ticks = Math.round(seconds * C.tickHz);
  const f = D.freeQpos;
  for (let t = -settle; t < ticks; t++) {
    const cmd = command(t >= 0 ? commandAt(schedule, t / C.tickHz) : {});
    last = await tick(data, t < 0 ? settling : net, last, cmd);
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

/**
 * ONE PHYSICS CALLER AT A TIME, PER WORLD.
 *
 * Node is single-threaded, but every tick awaits onnxruntime, so two
 * overlapping requests would interleave their steps into the same mjData and
 * produce a duck driven by two policies at once — a heisenbug that only shows
 * up under the load a steering loop actually generates. Each world gets its own
 * lane, and the lanes are separate so that a two-minute /measure cannot stall a
 * 100 ms /intent behind it.
 */
function lane() {
  let tail = Promise.resolve();
  return job => {
    const run = tail.then(job, job);
    tail = run.then(() => {}, () => {});   // a rejection must not poison the lane
    return run;
  };
}
const liveLane = lane(), batchLane = lane();

/**
 * THE LIVE WORLD: one duck that keeps standing between requests.
 *
 * Its own mjData, not the one /record and /measure reset on every call — that
 * is the whole difference between a bench and a transport. A steering loop
 * sends an intent, reads the state it caused, and sends the next one; sharing
 * a world with a rollout that resets to the drop height would teleport the duck
 * back to the origin in the middle of a walk.
 */
const live = {
  world: new mj.MjData(model),
  policy: STAND,
  last: new Array(14).fill(0),
  cmd: { vx: 0, vy: 0, vyaw: 0 },
  standing: false,
};

/**
 * Put the duck on its feet, once, the first time anyone asks anything of it.
 *
 * Lazy rather than at boot because a bench that is only ever asked to /record
 * should not pay half a second of settle it will throw away.
 */
async function ensureStanding() {
  if (live.standing) return;
  const settling = await policy(STAND);
  reset(live.world);
  live.last = new Array(14).fill(0);
  const neutral = command({});
  for (let t = 0; t < SETTLE_TICKS; t++) {
    live.last = await tick(live.world, settling, live.last, neutral);
  }
  live.standing = true;
}

/** Advance the live world under the command it is currently holding. */
async function hold(seconds) {
  const loaded = await policy(live.policy);
  const cmd = command(live.cmd);
  const ticks = Math.max(1, Math.round(seconds * C.tickHz));
  for (let i = 0; i < ticks; i++) live.last = await tick(live.world, loaded, live.last, cmd);
  return ticks;
}

/** What the duck is doing right now, in the live world. */
function stateOf() {
  const d = live.world, f = D.freeQpos;
  const root = [d.qpos[f], d.qpos[f + 1], d.qpos[f + 2],
                d.qpos[f + 3], d.qpos[f + 4], d.qpos[f + 5], d.qpos[f + 6]];
  const joints = [];
  for (let k = 0; k < 14; k++) joints.push(r4(d.qpos[D.qpos[k]]));
  return {
    t: r4(d.time),
    position: root.slice(0, 3).map(r4),
    quaternion: root.slice(3).map(r4),
    height: r4(root[2]),
    upright: upright(root),
    joints,
    policy: live.policy,
    command: live.cmd,
    // NOT A NUMBER, ON PURPOSE. quackd's transport reports a battery because a
    // real duck has one; this world has no battery to read, and inventing a
    // percentage would be a number nobody measured that a verb might one day
    // decide to land on. null is the honest reading, and the field is here so
    // the shape matches.
    // GROUND TRUTH, AND IT IS NOT FOR STEERING. A vision loop must earn its
    // bearing from the camera; this is here so a run can be SCORED — the same
    // split measure_success.mjs already uses.
    ball: ballOf(d),
    ballRadius: BALL_RADIUS,
    battery: null,
    batteryWhy: 'simulated duck: there is nothing to discharge',
  };
}

/**
 * A command component. Finite or the request is refused.
 *
 * DELIBERATELY NOT CLAMPED. Training's own range is the only real limit and
 * this repo's schedules span it unevenly — record_intents drives vx and vy
 * around the full unit circle, while ac_headspin2 commands vyaw 3 to get a
 * spin rate. A tidy-looking clamp at ±1 would be a number nobody measured and
 * would quietly break the one policy that needs a big one.
 */
function speed(v, field) {
  if (v === undefined || v === null) return 0;
  const n = Number(v);
  if (!Number.isFinite(n)) throw new Error(`${field} must be a finite number`);
  return n;
}

async function handle(url, body) {
  if (url.pathname === '/health') {
    return {
      bench: 'duck-bench/2',
      plant: 'scene.mjb — Pollen robot_allcollisions, training parameters',
      tickHz: C.tickHz,
      timestep: TIMESTEP,
      substepsPerTick: SUBSTEPS,
      cores: (await import('node:os')).cpus().length,
      policies: [...catalogue().keys()].sort(),
      records: true, measures: true,
      trains: false,
      trainsWhy: 'The accelerator here is an inference ASIC, and mjlab wants a GPU. '
               + 'Recording and measuring are what this machine is for.',
      // The transport surface, named so a caller can tell at a glance which
      // half of quackd's DuckTransport this speaks and which half it does not.
      steers: true,
      transport: {
        protocol: 'quackd DuckTransport (rokbenko/quackd)',
        clock: 'sim',
        clockWhy: 'now() reads this world\'s own MuJoCo clock, so a verb steers at '
                + 'sim speed here and at wall-clock speed on hardware, unchanged.',
        endpoints: ['GET /state', 'POST /intent', 'POST /stop', 'GET /now',
                  'POST /policy', 'POST /ball', 'POST /reset'],
        frames: false,
        framesWhy: 'No camera and no renderer in this process: perception is duckvision.py, '
                 + 'on a different MuJoCo build so that clips stay canon.',
        activePolicy: live.policy,
        standing: live.standing,
      },
    };
  }
  // GET /state — the duck as it is, in the world /intent has been advancing.
  if (url.pathname === '/state') {
    return liveLane(async () => { await ensureStanding(); return stateOf(); });
  }
  // GET /now — the transport owns time, and this is the clock it owns. A
  // steering loop asks for it instead of Date.now() so the same verb code runs
  // here and on hardware; here it only moves when someone advances physics,
  // which is precisely what makes a sim loop reproducible.
  // POST /ball — put it somewhere, or {"bearing": deg, "range": m} to place it
  // relative to where the duck is looking, which is what a trial wants.
  if (url.pathname === '/ball') {
    return liveLane(async () => {
      await ensureStanding();
      const d = live.world, f = D.freeQpos;
      if (body.bearing !== undefined || body.range !== undefined) {
        const bearing = Number(body.bearing ?? 0), range = Number(body.range ?? 0.8);
        if (!Number.isFinite(bearing) || !Number.isFinite(range)) {
          throw new Error('bearing and range must be finite numbers');
        }
        const q = [d.qpos[f + 3], d.qpos[f + 4], d.qpos[f + 5], d.qpos[f + 6]];
        const yaw = Math.atan2(2 * (q[0] * q[3] + q[1] * q[2]),
                               1 - 2 * (q[2] * q[2] + q[3] * q[3]));
        // Positive bearing is LEFT, the convention duckvision and the robot
        // both use, so a trial reads the same way the detector reports.
        const a = yaw + bearing * Math.PI / 180;
        placeBall(d, d.qpos[f] + range * Math.cos(a), d.qpos[f + 1] + range * Math.sin(a));
      } else {
        const x = Number(body.x), y = Number(body.y);
        if (!Number.isFinite(x) || !Number.isFinite(y)) {
          throw new Error('give {x, y} or {bearing, range}');
        }
        placeBall(d, x, y, Number.isFinite(+body.z) ? +body.z : BALL_RADIUS);
      }
      return stateOf();
    });
  }
  // POST /reset — put the live world back to a known start.
  //
  // A TRIAL THAT BEGINS WHEREVER THE LAST ONE STOPPED IS NOT A TRIAL. Without
  // this, a steering run inherited the previous run's position, heading and
  // half-finished stride, and the second trial of a batch could open already
  // spinning — which looks exactly like a controller that cannot see.
  if (url.pathname === '/reset') {
    return liveLane(async () => {
      live.standing = false;
      live.cmd = { vx: 0, vy: 0, vyaw: 0 };
      await ensureStanding();
      if (BALL) placeBall(live.world, 0.8, 0, BALL_RADIUS);
      return stateOf();
    });
  }
  if (url.pathname === '/now') {
    return liveLane(async () => {
      await ensureStanding();
      return { now: r4(live.world.time), clock: 'sim', tickHz: C.tickHz, timestep: TIMESTEP };
    });
  }
  /*
   * POST /intent — hold {vx, vy, vyaw} for a short window and advance physics.
   *
   * THE DEADMAN COSTS NOTHING HERE. quackd's verbs re-send a move every 100 ms
   * so a dropped link stops a real robot instead of leaving it walking into a
   * wall. This world only moves inside a request: miss the next intent and the
   * duck is not still walking, it is frozen mid-stride, so there is no timer to
   * arm and nothing to fail closed. That makes the repeat call the common case,
   * and it is cheap — no reset, no reload, one cached session and five ticks by
   * default, which is exactly the 100 ms the verbs re-send at.
   */
  if (url.pathname === '/intent') {
    const seconds = Math.min(Math.max(+body.hold || 0.1, 1 / C.tickHz), 2);
    return liveLane(async () => {
      await ensureStanding();
      live.cmd = { vx: speed(body.vx, 'vx'), vy: speed(body.vy, 'vy'),
                   vyaw: speed(body.vyaw, 'vyaw') };
      const ticks = await hold(seconds);
      return { ...stateOf(), held: r4(ticks / C.tickHz), ticks };
    });
  }
  // POST /stop — zero the command and let the duck settle under it. Not a
  // reset: stopping is a thing the policy does, and a duck that had to be
  // teleported upright to stop would be hiding the fall.
  if (url.pathname === '/stop') {
    const seconds = Math.min(Math.max(+body.settle || 0.5, 1 / C.tickHz), 5);
    return liveLane(async () => {
      await ensureStanding();
      live.cmd = { vx: 0, vy: 0, vyaw: 0 };
      const ticks = await hold(seconds);
      return { ...stateOf(), settled: r4(ticks / C.tickHz), ticks };
    });
  }
  /*
   * POST /policy — swap the ACTIVE policy under the standing duck.
   *
   * SAME ALLOW-LIST, NO EXCEPTION. The name goes through `policy()`, which
   * looks it up in a Map built by scanning this directory: '../secrets.onnx',
   * '/etc/passwd' and 'https://example.com/p.onnx' are refused not because
   * anything inspects them for traversal but because they are not keys. A Map
   * is used rather than an object so that '__proto__' is a miss too. Nothing
   * is fetched, nothing is joined onto a path.
   *
   * The duck is NOT reset around the swap: hot means hot, and a policy that
   * cannot pick up another's pose mid-stance is telling you something true.
   */
  if (url.pathname === '/policy') {
    const wanted = typeof body.policy === 'string' ? body.policy : String(body.policy ?? '');
    return liveLane(async () => {
      await ensureStanding();
      const loaded = await policy(wanted);   // throws `unknown policy: …` -> 400
      const was = live.policy;
      live.policy = wanted;
      return {
        ...stateOf(), was,
        // Whether this policy declared its own neutral pose or inherited HOME:
        // the one thing about a hot swap that is not visible in the duck's pose.
        reference: loaded.reference === HOME ? 'HOME' : 'declared by the policy',
      };
    });
  }
  if (url.pathname === '/record') {
    const seconds = Math.min(Math.max(+body.seconds || 3, 0.2), 30);
    const run = await batchLane(() =>
      rollout({ name: body.policy, seconds, schedule: body.schedule }));
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
      const run = await batchLane(() =>
        rollout({ name: body.policy, seconds, schedule: body.schedule, drop }));
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
  console.log(`plant: ${TIMESTEP} s timestep, ${SUBSTEPS} substeps per ${C.tickHz} Hz tick`);
  console.log('records/measures: /record /measure — steers: /state /intent /stop /now /policy');
  console.log(`policies: ${[...catalogue().keys()].sort().join(', ')}`);
});
