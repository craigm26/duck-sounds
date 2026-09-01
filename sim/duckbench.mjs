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
// process has no camera and no renderer: rendering lives in duckvision.py,
// which imports the pip `mujoco`, while this runs the npm one — MUJOCO 3.1.16
// (WASM), the version in sim/node_modules/mujoco/package.json — so that clips
// stay canon. This comment said 3.5.1 until 2026-08-30 and a design was
// written on that number as executed ground truth, which is how a stale
// comment becomes a stated fact. duckvision's own version is not given here
// because nothing this file can read says what it is; ask that process.
// Wiring the two together is a job for whatever composes them, not a stub
// that returns a grey rectangle.
//
// MORE THAN ONE DUCK, AND WHY THAT IS ONLY POSSIBLE HERE. A hardware Microduck
// cannot perceive another Microduck. The observation robotd builds is 61 values
// — 48 of proprioception, 3 of commanded twist, 4 of head pose, 6 of body pose
// (Pollen's `duck-ipc-proto` and the training env, read 2026-09-01) — and there
// is no slot in it for a second robot. Two real ducks in a room are two blind
// agents that happen to share a floor, and whatever coordination they show has
// to be carried over a network, which is where it comes apart: `intents.rs` is
// last-writer-wins on one slot, so two writers at 50 Hz interleave into that
// slot and produce a robot that obeys neither, and the deadman is age-based, so
// a partition does not degrade a duck, it stops one.
//
// Ducks in ONE MuJoCo model have none of that, and they DO perceive each other
// — not through a sensor slot but through the physics itself: contact forces
// when they touch, the floor they both push against, the dynamics of an object
// one of them lifts while the other holds it. One integrator, one clock, no
// link jitter, no last-writer-wins, because there is no link and no writer
// race. So the simulator is not a lesser swarm than a room full of hardware; it
// is the only place a swarm can exist right now, and the only place a genuinely
// multi-duck policy could ever be trained, since training needs exactly the
// shared-state rollout the 61-wide hardware observation cannot supply.
//
// Concretely: a scene may hold N ducks, each with a name taken from its MJCF
// prefix (`build_multiduck.py` writes them; `compile_multiduck.mjs` compiles
// them). /intent, /policy, /state, /stop, /ball and /reset take an optional
// `duck` name and mean the first duck without one, which is what a single-duck
// caller has always been asking for. Every duck has its own policy slot, so two
// of them can run different networks. And ONE STEP OF PHYSICS ADVANCES ALL OF
// THEM: a request addressed to one duck still moves the others under the
// commands they are holding, because there is one clock and that is the entire
// point of putting them in one world.
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
import { createHash } from 'node:crypto';
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
// WHICH WORLD. The canon plant is scene.mjb and every recorded clip in duckkit
// claims to come from it, so it is the default and adding bodies to it is not
// on. A bench that wants something to pick up asks for a different scene:
//   DUCKBENCH_SCENE=scene_grasp.mjb node duckbench.mjs
const SCENE = process.env.DUCKBENCH_SCENE || 'scene.mjb';
const SCENE_BYTES = fs.readFileSync(SCENE);
mj.FS.writeFile('/s.mjb', new Uint8Array(SCENE_BYTES));
// AND WHICH WORLD IT ACTUALLY WAS, SAID OUT LOUD IN EVERY ANSWER THAT CARRIES
// A MEASUREMENT. A caller that keeps a result — Duck Studio keeps them beside
// the draft that caused them, forever — has to be able to say which plant
// produced it, and it can only say what this bench tells it. Until now
// /perform and /record told it nothing, so the app wrote a placeholder of its
// own and then printed the placeholder as though it were a fact about a world.
// A filename alone is not enough either: `sim/scene.mjb` and `site/scene.mjb`
// share a name and differ in bytes (see PLANT.md), and it was that pair that
// made this a bug rather than a nicety. So the digest goes with the name.
const PLANT = path.basename(SCENE);
const PLANT_DIGEST = createHash('sha256').update(SCENE_BYTES).digest('hex');
const model = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const data = new mj.MjData(model);

/** The fourteen joints a policy drives, in the order the observation wants them. */
const DUCK_JOINTS = C.jointNames.filter(n => n !== 'mouth');

const namedIndex = (count, read, name) => {
  for (let i = 0; i < count; i++) if (read(i) === name) return i;
  return -1;
};

/**
 * EVERY DUCK IN THIS WORLD, FOUND BY WALKING THE MODEL.
 *
 * A duck is a body whose name is `trunk_base` or ends in `_trunk_base` and
 * which carries a free joint, and everything else about it — its fourteen
 * joints, its fourteen actuators, its gyro — is that body's prefix followed by
 * the name the single-duck scene uses. That prefix is not decoration: every
 * name in MJCF is a global, so a second copy of the duck subtree collides with
 * the first on all thirty-odd of them, and prefixing is how
 * `build_multiduck.py` gets N of them into one model at all.
 *
 * WALKED RATHER THAN CONFIGURED for the reason GRASPABLES is walked: a scene
 * with three ducks in it says so because the ducks are there. Nothing has to be
 * told twice, `scene.mjb` keeps answering with exactly one duck named `duck`,
 * and a caller that never heard of a second one reads the same answers it
 * always did.
 *
 * A MISSING PIECE IS FATAL HERE RATHER THAN SILENT LATER. A trunk whose gyro
 * cannot be found used to fall back on sensor address 0, which means feeding a
 * policy some other sensor's three numbers as its angular velocity — a lie that
 * produces a plausible-looking rollout. Boot is the place to say so.
 */
function discoverDucks() {
  const found = [];
  for (let b = 0; b < model.nbody; b++) {
    const body = model.body(b).name;
    if (body !== 'trunk_base' && !body.endsWith('_trunk_base')) continue;
    const prefix = body.slice(0, body.length - 'trunk_base'.length);
    // `trunk_base` alone is the canon scene's duck and has no prefix; it is
    // named `duck` so that every answer can name a duck even there.
    const name = prefix ? prefix.slice(0, -1) : 'duck';
    let freeQpos = -1, freeDof = -1;
    for (let j = 0; j < model.njnt; j++) {
      if (model.jnt_type[j] !== 0 || model.jnt_bodyid[j] !== b) continue;
      freeQpos = model.jnt_qposadr[j]; freeDof = model.jnt_dofadr[j];
      break;
    }
    if (freeQpos < 0) throw new Error(`${body} has no free joint: it is bolted to the world`);
    const qpos = [], dof = [], ctrl = [];
    for (const joint of DUCK_JOINTS) {
      const j = namedIndex(model.njnt, i => model.jnt(i).name, prefix + joint);
      if (j < 0) throw new Error(`joint missing from the model: ${prefix}${joint}`);
      qpos.push(model.jnt_qposadr[j]); dof.push(model.jnt_dofadr[j]);
      const a = namedIndex(model.nu, i => model.actuator(i).name, prefix + joint);
      if (a < 0) throw new Error(`actuator missing from the model: ${prefix}${joint}`);
      ctrl.push(a);
    }
    const gyro = namedIndex(model.nsensor, i => model.sensor(i).name, prefix + 'imu_ang_vel');
    if (gyro < 0) throw new Error(`sensor missing from the model: ${prefix}imu_ang_vel`);
    // Where this duck starts: the free joint's own qpos0, which is the `pos`
    // the MJCF gave its trunk. `r4` is declared further down and this runs at
    // load, so the rounding is done longhand rather than reaching into the
    // temporal dead zone.
    const spawn = [0, 1, 2].map(k => Math.round(model.qpos0[freeQpos + k] * 10000) / 10000);
    found.push({ name, prefix, spawn, gyro: model.sensor(gyro).adr,
                 joints: { qpos, dof, freeQpos, freeDof }, ctrl });
  }
  if (!found.length) throw new Error('this world has no duck in it');
  return found;
}
const DUCKS = discoverDucks();
const DUCK_NAMES = DUCKS.map(d => d.name);
/** Every duck's root address, so nothing else in the world mistakes one for a prop. */
const DUCK_ROOTS = new Set(DUCKS.map(d => d.joints.freeQpos));

// THE TWO FINDERS ARE PINNED TO EACH OTHER AT BOOT. `findDuckJoints` is
// duckloop's, shared with the browser and with every other runner in this
// directory, and it knows only about an unprefixed duck; `discoverDucks` above
// is the generalisation. Where both apply — the canon one-duck scene — they
// must agree, or the multi-duck path has quietly started driving different
// joints than the recorded corpus came from.
{
  const plain = DUCKS.find(d => d.prefix === '');
  if (plain) {
    const canon = findDuckJoints(model);
    const same = canon.freeQpos === plain.joints.freeQpos
              && canon.freeDof === plain.joints.freeDof
              && canon.qpos.every((v, i) => v === plain.joints.qpos[i])
              && canon.dof.every((v, i) => v === plain.joints.dof[i]);
    if (!same) throw new Error('discoverDucks disagrees with duckloop findDuckJoints about the duck');
  }
}

// THE BALL IS THE ONLY OTHER THING WORTH ADDRESSING IN THIS WORLD. It is
// Pollen's own (radius 0.05, condim 6 so it actually decelerates), and a
// steering loop needs to put it somewhere and then be scored on reaching it.
const BALL = (() => {
  for (let j = 0; j < model.njnt; j++) {
    if (model.jnt_type[j] !== 0) continue;                 // mjJNT_FREE
    const adr = model.jnt_qposadr[j];
    if (DUCK_ROOTS.has(adr)) continue;                     // a duck, not a prop
    if (model.body(model.jnt_bodyid[j]).name === 'ball') return { adr, dof: model.jnt_dofadr[j] };
  }
  return null;
})();
const BALL_RADIUS = 0.05;

/**
 * Every free body in this world that is not the duck and not the ball — the
 * things a fetch or a drag is about.
 *
 * FOUND BY WALKING THE MODEL, not by a list kept in step with the XML. A scene
 * with a broom in it says so because the broom is there; nothing has to be
 * told twice, and a scene without one reports an empty list rather than a lie.
 */
const GRASPABLES = (() => {
  const found = [];
  for (let j = 0; j < model.njnt; j++) {
    if (model.jnt_type[j] !== 0) continue;                 // mjJNT_FREE
    const adr = model.jnt_qposadr[j];
    // A SECOND DUCK IS NOT A GRASPABLE. Its trunk is a free body like any
    // other and would otherwise be listed as something to pick up, which is
    // both wrong and — since /health publishes this list — a claim about the
    // world that the world does not support.
    if (DUCK_ROOTS.has(adr)) continue;
    const body = model.jnt_bodyid[j];
    const name = model.body(body).name;
    if (name === 'ball') continue;                         // it has its own door
    // `r4` is declared further down and this runs at load, so the rounding is
    // done longhand rather than reaching into the temporal dead zone.
    const mass = Math.round(model.body_mass[body] * 10000) / 10000;
    found.push({ name, adr, dof: model.jnt_dofadr[j], body, mass });
  }
  return found;
})();

/** Where a graspable is now: position and whether it has been moved. */
function graspableState(d) {
  return GRASPABLES.map(g => ({
    name: g.name,
    mass: g.mass,
    at: [r4(d.qpos[g.adr]), r4(d.qpos[g.adr + 1]), r4(d.qpos[g.adr + 2])],
  }));
}

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
  // An uploaded policy is already in `sessions` under its own filename and is
  // not in the catalogue, which only walks what shipped. Check there first.
  if (name.startsWith('uploaded-') && sessions.has(`${name}.onnx`)) {
    return sessions.get(`${name}.onnx`);
  }
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

/**
 * ONE DUCK, PUT BACK WHERE IT STARTED, WITHOUT DISTURBING THE WORLD AROUND IT.
 *
 * `mj_resetData` cannot do this: it resets everything, which in a shared world
 * means teleporting the other ducks and every prop as well. Writing this duck's
 * own qpos and qvel is the only way to give one duck a fresh start while the
 * rest of the scene carries on, and it is what /reset with a `duck` name does.
 *
 * The spawn is the free joint's qpos0, so a duck lands back on the mark its
 * MJCF gave it rather than on the origin, which in a multi-duck scene is
 * somebody else's mark.
 */
function placeDuck(d, duck, z = 0.1231) {
  const j = duck.joints, f = j.freeQpos;
  d.qpos[f] = duck.spawn[0]; d.qpos[f + 1] = duck.spawn[1]; d.qpos[f + 2] = z;
  d.qpos[f + 3] = 1; d.qpos[f + 4] = 0; d.qpos[f + 5] = 0; d.qpos[f + 6] = 0;
  for (let k = 0; k < 6; k++) d.qvel[j.freeDof + k] = 0;
  for (let i = 0; i < 14; i++) {
    d.qpos[j.qpos[i]] = HOME[i];
    d.qvel[j.dof[i]] = 0;
    d.ctrl[duck.ctrl[i]] = HOME[i];
  }
}

/**
 * The whole world back to its start, with every duck dropped from `z`.
 *
 * All of them, not just the one a caller is about to drive: a duck left in
 * whatever heap the last rollout ended in is still in the scene, still touching
 * the floor the driven duck walks on, and would make the run unrepeatable.
 */
function reset(d, z = 0.1231) {
  mj.mj_resetData(model, d);
  for (const duck of DUCKS) placeDuck(d, duck, z);
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
/**
 * One control step's WORTH OF COMMANDS, for ONE duck, WITHOUT ADVANCING TIME.
 *
 * SPLIT OUT OF `tick` BECAUSE A WORLD HAS ONE CLOCK AND MAY HAVE SEVERAL DUCKS.
 * If each duck stepped physics after choosing its own action, then in a
 * two-duck world duck A would act, time would move, and duck B would then act
 * on a world half a control period older than the one A saw — a stagger nobody
 * asked for and which would be invisible in the answers. Every duck writes its
 * ctrl against the same instant, and then time moves once for all of them. That
 * is the difference between N ducks in a world and N worlds.
 *
 * `capture`, when given, collects the pairs a policy would have to reproduce to
 * perform this motion by itself: the 61-wide OBSERVATION the network was shown,
 * and the EFFECTIVE ACTION — what it would have had to output for the joints to
 * end up where the authored track put them. That second number is
 * `ctrl - reference`, because a pure policy's target is `reference + action`
 * while an authored run's is `reference + action + offset`. Cloning the sum is
 * cloning the motion.
 *
 * The observation is the untouched one, for the reason the comment below
 * already gives: a network told what it asked for rather than what it is
 * standing in learns something that only works in a recording.
 */
async function actuate(d, duck, loaded, last, cmd, offsets = null, blend = 1, capture = null,
                       expert = null, teacherShare = 0, jitter = 0) {
  const { net, reference } = loaded;
  const J = duck.joints, f = J.freeQpos, g = duck.gyro;
  const q = [d.qpos[f + 3], d.qpos[f + 4], d.qpos[f + 5], d.qpos[f + 6]];
  const jp = [], jv = [];
  for (let k = 0; k < 14; k++) { jp.push(d.qpos[J.qpos[k]]); jv.push(d.qvel[J.dof[k]]); }
  const obs = buildObs([d.sensordata[g], d.sensordata[g + 1], d.sensordata[g + 2]],
                       projectedGravity(q), jp, jv, last, cmd, reference);
  const out = await net.run({ obs: new ort.Tensor('float32', obs, [1, 61]) });
  const action = Array.from(out[net.outputNames[0]].data);
  for (let k = 0; k < 14; k++) {
    // AUTHORED OFFSETS RIDE ON TOP OF THE POLICY, exactly as record_intents.mjs
    // applies them: the policy keeps its balance and the track leans on it. The
    // OBSERVATION is untouched — the network is told what it is actually
    // standing in, not what the author asked for, because handing a policy its
    // own request back as state is how a motion looks fine in a recording and
    // falls over on a robot.
    const base = reference[k] + action[k];
    const target = offsets ? base + (offsets[k] - HOME[k]) * blend : base;
    // THIS DUCK'S OWN ACTUATORS, BY INDEX LOOKED UP AT BOOT. `ctrl[k]` was
    // right while there was one duck and every actuator in the model belonged
    // to it; with two, actuator 3 is the FIRST duck's left knee whichever duck
    // is being driven, and a second duck steered that way would have moved the
    // first one's legs.
    d.ctrl[duck.ctrl[k]] = Math.min(Math.max(target, LO[k]), HI[k]);
  }
  // AFTER THE CLAMP, NOT BEFORE. What the joints were actually commanded is the
  // thing a clone has to reproduce; a label taken before the clamp teaches the
  // network to ask for angles the hardware refuses.
  // What the next observation will be told the last action was. For a plain
  // rollout that is the acting network's own output; while CAPTURING it must be
  // the EFFECTIVE action instead — see the note where it is assigned.
  let fedBack = action;

  if (capture) {
    // WHAT A POLICY WOULD HAVE TO OUTPUT AT THIS STATE. A pure policy's target
    // is `reference + action`; an authored run's is that plus the offset. So
    // the label is `ctrl - reference`, taken AFTER the clamp, because a label
    // taken before it teaches the network to ask for angles the hardware
    // refuses.
    const effective = [];
    if (expert) {
      // THE TEACHER LABELS EVERY STATE, whoever drove the duck into it. That
      // is the pairing a clone actually needs — not "what happened on the good
      // run" but "what to do from here".
      const out = await expert.net.run({ obs: new ort.Tensor('float32', obs, [1, 61]) });
      const teach = Array.from(out[expert.net.outputNames[0]].data);
      const taught = [];
      for (let k = 0; k < 14; k++) {
        const base = expert.reference[k] + teach[k];
        const target = offsets ? base + (offsets[k] - HOME[k]) * blend : base;
        const clamped = Math.min(Math.max(target, LO[k]), HI[k]);
        taught.push(clamped);
        effective.push(clamped - expert.reference[k]);
      }
      // The teacher can also ACT, on a share of steps. A first clone is bad
      // enough to tear the integrator apart within a few steps — measured:
      // 2591 of 2720 unusable — so letting it drive alone yields nothing to
      // learn from.
      if (Math.random() < teacherShare) {
        for (let k = 0; k < 14; k++) d.ctrl[duck.ctrl[k]] = taught[k];
      }
    } else {
      for (let k = 0; k < 14; k++) effective.push(d.ctrl[duck.ctrl[k]] - reference[k]);
    }

    // A NON-FINITE STEP IS NOT A TRAINING PAIR, AND MUST NOT LEAVE AS ONE.
    // `JSON.stringify` writes NaN and Infinity as `null`, so a diverged step
    // travels as a hole a consumer reads back as NaN and trains on in silence.
    //
    // AND FINITE IS NOT PLAUSIBLE. A diverged MuJoCo state yields enormous
    // DOUBLES — measured at 6.8e37 — which `Number.isFinite` accepts happily;
    // train on those and the normaliser's deviation becomes 1e36, every real
    // observation flattens to zero, and the loss is NaN by the second epoch.
    // The bound is generous: past a thousand is not a duck in any pose.
    const sane = v => Number.isFinite(v) && Math.abs(v) < 1000;
    const row = Array.from(obs);
    if (row.every(sane) && effective.every(sane)) {
      capture.push({ obs: row, action: effective });
    } else {
      capture.rejected = (capture.rejected || 0) + 1;
    }

    // THE LABEL IS ALSO WHAT GETS FED BACK, and getting this wrong is what made
    // three clones in a row saturate a joint within half a second.
    //
    // `lastAction` is part of the observation. During capture the acting
    // network was the teacher, so the raw action fed back carried NO authored
    // offset — while a policy trained on these labels outputs the effective
    // action, offset included, and feeds THAT back. The clone therefore met an
    // observation block it had never seen on its very first step, and the error
    // grew every tick. The file was never wrong: checked offline, it answered
    // the training observations to 0.006 rad. The loop was.
    fedBack = effective;
  }

  // NOISE ON THE WAY OUT, NEVER ON THE LABEL. The label above is what the
  // teacher would do AT THIS STATE; the jitter then pushes the duck slightly
  // off the demonstrated path, so the NEXT step's label is a recovery. That is
  // the difference between a clone that memorises one trajectory and one with a
  // basin around it — measured: without it, a bow clone fell in all 32 of 32.
  if (jitter > 0) {
    for (let k = 0; k < 14; k++) {
      const c = duck.ctrl[k];
      d.ctrl[c] = Math.min(Math.max(d.ctrl[c] + (Math.random() * 2 - 1) * jitter,
                                    LO[k]), HI[k]);
    }
  }
  return fedBack;
}

/**
 * Time, moved once, for everything in the world at once.
 *
 * The plant's own substeps, so a control tick is a control tick whether one
 * duck is standing in this world or three are walking into each other.
 */
function stepWorld(d) {
  for (let s = 0; s < SUBSTEPS; s++) mj.mj_step(model, d);
}

/**
 * The single-duck control tick: choose this duck's commands, then advance time.
 *
 * THE SAME TICK FOR EVERY CALLER that drives one duck — /record, /measure,
 * /perform and /capture all come through here, so a policy that measured 16/16
 * behaves identically wherever it is run. The live world does NOT use this one:
 * it actuates every duck first and calls `stepWorld` once, which is the same
 * sequence with the loop in the right place.
 */
async function tick(d, duck, loaded, last, cmd, offsets = null, blend = 1, capture = null,
                    expert = null, teacherShare = 0, jitter = 0) {
  const fedBack = await actuate(d, duck, loaded, last, cmd, offsets, blend, capture,
                                expert, teacherShare, jitter);
  stepWorld(d);
  return fedBack;
}

/**
 * Interpolate an authored keyframe track, the same smoothstep the phone draws
 * with and `record_intents.mjs` records with. TWENTY copies of this curve now
 * exist in this repo — `grep -rn "function poseAt"` over duck-sounds, counted
 * 2026-08-30: nineteen under sim/ and site/intent.mjs, which the browser
 * preview imports — and they all have to agree, or a motion previews as one
 * shape and runs as another. This comment said three, which is how a change
 * here comes to be costed as touching two other files when it touches
 * nineteen, one of them the recorder duckkit's shipped corpus came from and
 * one of them the preview a browser draws.
 */
function poseAt(track, time) {
  if (!track || !track.length) return null;
  if (time <= 0) return HOME.slice();
  let pt = 0, pp = HOME;
  for (const f of track) {
    if (time <= f.at) {
      const u = (time - pt) / Math.max(f.at - pt, 1e-9), s = u * u * (3 - 2 * u);
      return f.pose.map((v, k) => pp[k] + (v - pp[k]) * s);
    }
    pt = f.at; pp = f.pose;
  }
  return track[track.length - 1].pose.slice();
}

/**
 * The canon loop, the same one every recorded clip came from: a settle under
 * the standing policy with a neutral command, then the training path — target
 * = HOME + action at scale 1.0, no filter — and the servo travel as the clamp.
 *
 * IT DRIVES ONE DUCK. In a multi-duck world the others are reset to HOME with
 * the rest of the scene and then left holding that pose on their position
 * servos for the length of the run — they are furniture, not participants, and
 * a rollout says which duck it drove rather than leaving a reader to assume.
 * Recording a swarm would mean a clip format that carries N sets of frames, and
 * duck-intent-clips/3 carries one; the live world is where several ducks are
 * actually steered at once.
 */
async function rollout({ name, seconds, schedule, settle = SETTLE_TICKS, drop = 0.1231,
                         track = null, blend = 1, capture = null, expertName = null,
                         teacherShare = 0, jitter = 0, duck = DUCKS[0] }) {
  const net = await policy(name);
  const settling = await policy(STAND);
  // The teacher, when one is asked for: the policy the authored motion rides on.
  const expert = expertName ? await policy(expertName) : null;
  reset(data, drop);
  let last = new Array(14).fill(0);
  const frames = [], roots = [], commands = [];
  const ticks = Math.round(seconds * C.tickHz);
  const D = duck.joints;
  const f = D.freeQpos;
  for (let t = -settle; t < ticks; t++) {
    const cmd = command(t >= 0 ? commandAt(schedule, t / C.tickHz) : {});
    // NEUTRAL THROUGH THE SETTLE. The settle exists to let the drop-bounce die;
    // feeding the track through it starts the motion before recording does.
    const offsets = track && t >= 0 ? poseAt(track, t / C.tickHz) : null;
    // NOTHING IS CAPTURED DURING THE SETTLE. Those ticks are the drop bounce
    // dying under a different policy; teaching a clone from them would teach it
    // to recover from a fall it will never be dropped into.
    last = await tick(data, duck, t < 0 ? settling : net, last, cmd, offsets, blend,
                      t >= 0 ? capture : null, t >= 0 ? expert : null, teacherShare,
                      t >= 0 ? jitter : 0);
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
 * THE LIVE WORLD: the ducks that keep standing between requests.
 *
 * Its own mjData, not the one /record and /measure reset on every call — that
 * is the whole difference between a bench and a transport. A steering loop
 * sends an intent, reads the state it caused, and sends the next one; sharing
 * a world with a rollout that resets to the drop height would teleport the duck
 * back to the origin in the middle of a walk.
 *
 * ONE WORLD, ONE CLOCK, A SLOT PER DUCK. `world` is singular on purpose: the
 * ducks are in the same mjData, so they collide, share a floor and disturb the
 * same props. What is per-duck is the STEERING — the policy it is running, the
 * command it is holding, and the last action its next observation will be told
 * about — because those are exactly the things two ducks in one world need to
 * differ in for the world to be worth having.
 */
const live = {
  world: new mj.MjData(model),
  standing: false,
  slots: new Map(DUCKS.map(duck => [duck.name, {
    duck,
    policy: STAND,
    last: new Array(14).fill(0),
    cmd: { vx: 0, vy: 0, vyaw: 0 },
  }])),
};

/**
 * Put every duck on its feet, once, the first time anyone asks anything of it.
 *
 * ALL OF THEM, EVEN WHEN ONE WAS ASKED FOR. A duck left lying where the model
 * dropped it is still in the scene the addressed duck has to walk through, so
 * there is no such thing as settling one of them.
 *
 * Lazy rather than at boot because a bench that is only ever asked to /record
 * should not pay half a second of settle it will throw away.
 */
async function ensureStanding() {
  if (live.standing) return;
  const settling = await policy(STAND);
  reset(live.world);
  const neutral = command({});
  for (const slot of live.slots.values()) slot.last = new Array(14).fill(0);
  for (let t = 0; t < SETTLE_TICKS; t++) {
    for (const slot of live.slots.values()) {
      slot.last = await actuate(live.world, slot.duck, settling, slot.last, neutral);
    }
    stepWorld(live.world);
  }
  live.standing = true;
}

/**
 * Advance the live world under the commands its ducks are currently holding.
 *
 * EVERY DUCK IS DRIVEN, WHOEVER ASKED. This is the sentence a caller most needs
 * to have read: /intent addressed to one duck advances the whole world, and the
 * other ducks keep walking under whatever they were last told, because there is
 * one integrator and it cannot advance half a scene. On hardware the equivalent
 * would be four robots each running their own loop and nothing keeping their
 * timelines together; here they are together by construction, and a caller who
 * wants a duck to stand still says so with /stop rather than by not mentioning
 * it.
 *
 * Each policy is loaded once for the whole span even when several ducks share
 * one — `policy()` caches by file, and this keeps the lookup out of the tick.
 */
async function hold(seconds) {
  const driving = [];
  for (const slot of live.slots.values()) {
    driving.push({ slot, loaded: await policy(slot.policy), cmd: command(slot.cmd) });
  }
  const ticks = Math.max(1, Math.round(seconds * C.tickHz));
  for (let i = 0; i < ticks; i++) {
    for (const d of driving) {
      d.slot.last = await actuate(live.world, d.slot.duck, d.loaded, d.slot.last, d.cmd);
    }
    stepWorld(live.world);
  }
  return ticks;
}

/**
 * A one-line reading of every duck in the world, for the answers that address
 * only one of them.
 *
 * WHY IT RIDES ALONG ON EVERY ANSWER. In a shared world the other ducks are
 * part of what happened to this one — they are what it bumped into — so a
 * caller that reads a position without them is reading half a result. On the
 * canon one-duck scene it is a single entry and costs nothing.
 */
function rollCall() {
  const d = live.world;
  return DUCKS.map(duck => {
    const slot = live.slots.get(duck.name), f = duck.joints.freeQpos;
    const root = [d.qpos[f], d.qpos[f + 1], d.qpos[f + 2],
                  d.qpos[f + 3], d.qpos[f + 4], d.qpos[f + 5], d.qpos[f + 6]];
    return {
      name: duck.name,
      position: root.slice(0, 3).map(r4),
      upright: upright(root),
      policy: slot.policy,
      command: slot.cmd,
    };
  });
}

/** What one duck is doing right now, in the live world. */
function stateOf(slot) {
  const d = live.world, D = slot.duck.joints, f = D.freeQpos;
  const root = [d.qpos[f], d.qpos[f + 1], d.qpos[f + 2],
                d.qpos[f + 3], d.qpos[f + 4], d.qpos[f + 5], d.qpos[f + 6]];
  const joints = [];
  for (let k = 0; k < 14; k++) joints.push(r4(d.qpos[D.qpos[k]]));
  return {
    // WHICH DUCK THIS IS ABOUT, ALWAYS SAID. A caller that never passes `duck`
    // gets the same name every time and can ignore it; a caller steering three
    // of them can check that the answer is about the one it addressed rather
    // than trusting that it asked correctly.
    duck: slot.duck.name,
    t: r4(d.time),
    position: root.slice(0, 3).map(r4),
    quaternion: root.slice(3).map(r4),
    height: r4(root[2]),
    upright: upright(root),
    joints,
    policy: slot.policy,
    command: slot.cmd,
    ducks: rollCall(),
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

/**
 * WHICH DUCK A REQUEST IS ABOUT.
 *
 * `duck` in the body, or `?duck=` on the query string for the endpoints that
 * are GETs. ABSENT MEANS THE FIRST DUCK, which on the canon scene is the only
 * duck and is exactly what every existing caller has always been asking for;
 * in a scene with several, the first is the one the MJCF lists first and the
 * answer names it, so nobody has to guess which one moved.
 *
 * A name that is not in this world is refused with the names that are, rather
 * than silently steering the default — a typo that quietly drives the wrong
 * duck is the multi-duck version of the ball-instead-of-the-duck bug that
 * `findDuckJoints` was written to end.
 */
function pickSlot(url, body) {
  const wanted = body?.duck ?? url.searchParams.get('duck');
  if (wanted === undefined || wanted === null || wanted === '') {
    return live.slots.get(DUCKS[0].name);
  }
  const slot = live.slots.get(String(wanted));
  if (!slot) throw new Error(`unknown duck: ${wanted} — this world holds ${DUCK_NAMES.join(', ')}`);
  return slot;
}

/** The same choice, for the batch endpoints, which address a duck but hold no live slot. */
function pickDuck(url, body) {
  return pickSlot(url, body).duck;
}

async function handle(url, body) {
  if (url.pathname === '/health') {
    return {
  // BUMPED WITH THE PLANT FIELDS. /health, /perform and /record now carry
  // plantName and plantDigest, and a reader that sees duck-bench/2 knows it is
  // talking to a bench that CANNOT say which world it ran, as distinct from one
  // that did not. Additive fields alone leave those two indistinguishable.
  // BUMPED AGAIN FOR THE DUCKS. duck-bench/4 holds N of them in one world and
  // takes a `duck` name on the steering endpoints. The field is what lets a
  // reader tell "this bench has one duck" from "this bench cannot tell you how
  // many it has": a duck-bench/3 answer has no `ducks` key either way, and a
  // caller that guessed would be guessing about the world.
      bench: 'duck-bench/4',
      plant: `${SCENE} — Pollen robot_allcollisions, training parameters`,
      plantName: PLANT,
      plantDigest: PLANT_DIGEST,
      // THE DUCKS PRESENT, walked out of the model at boot rather than
      // configured, so a scene answers with what is actually in it. `prefix` is
      // the MJCF name prefix each one's joints and actuators carry and is here
      // because it is what a caller would need to make sense of the scene XML
      // beside this answer; `spawn` is where /reset puts that duck back.
      ducks: DUCKS.map(d => ({
        name: d.name,
        prefix: d.prefix,
        spawn: d.spawn,
        policy: live.slots.get(d.name).policy,
      })),
      ducksWhy: 'One model, one integrator: a single step advances every duck, so they meet '
              + 'through contact and a shared floor rather than over a link. A hardware duck '
              + 'cannot perceive another duck at all — the observation is 61 values with no '
              + 'slot for one — so this is the only place several of them share a world.',
      // What is in this world that the duck could take hold of. NOT EMPTY ON
      // THE CANON SCENE ANY MORE: scene.mjb as served here on 2026-08-30
      // answers with five — block_a, block_b, block_c, cone_a, cone_b — where
      // this comment said it was bare. The list itself is walked out of the
      // model on every boot, so what is SERVED was right the whole time; it is
      // the comment that lied, and a caller who read it instead of the answer
      // is why it is corrected rather than deleted.
      // Name and mass only: the qpos addresses are this file's business.
      graspables: GRASPABLES.map(g => ({ name: g.name, mass: g.mass })),
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
        // Every steering endpoint except /now, which reads the world's clock
        // and so belongs to all of them at once.
        addressable: '`duck` in the body or ?duck= on a GET; absent means ' + DUCKS[0].name,
        frames: false,
        framesWhy: 'No camera and no renderer in this process: perception is duckvision.py, '
                 + 'on a different MuJoCo build so that clips stay canon.',
        // The first duck's, kept under the name a duck-bench/3 reader already
        // knows; `ducks` above carries one of these per duck.
        activePolicy: live.slots.get(DUCKS[0].name).policy,
        standing: live.standing,
      },
    };
  }
  // GET /state — the duck as it is, in the world /intent has been advancing.
  if (url.pathname === '/state') {
    return liveLane(async () => {
      const slot = pickSlot(url, body);
      await ensureStanding();
      return stateOf(slot);
    });
  }
  // GET /now — the transport owns time, and this is the clock it owns. A
  // steering loop asks for it instead of Date.now() so the same verb code runs
  // here and on hardware; here it only moves when someone advances physics,
  // which is precisely what makes a sim loop reproducible.
  // POST /ball — put it somewhere, or {"bearing": deg, "range": m} to place it
  // relative to where the duck is looking, which is what a trial wants.
  if (url.pathname === '/ball') {
    return liveLane(async () => {
      // BEARING AND RANGE ARE RELATIVE TO A DUCK, so which duck is part of the
      // request even though the ball belongs to nobody. Absent, it is the first
      // one, the same as everywhere else.
      const slot = pickSlot(url, body);
      await ensureStanding();
      const d = live.world, f = slot.duck.joints.freeQpos;
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
      return stateOf(slot);
    });
  }
  // POST /reset — put the live world back to a known start.
  //
  // A TRIAL THAT BEGINS WHEREVER THE LAST ONE STOPPED IS NOT A TRIAL. Without
  // this, a steering run inherited the previous run's position, heading and
  // half-finished stride, and the second trial of a batch could open already
  // spinning — which looks exactly like a controller that cannot see.
  //
  // WITH A `duck` NAME IT RESETS THAT DUCK ONLY, and that is a different act,
  // not a smaller one. `mj_resetData` restarts everything in the world, which
  // in a shared scene means teleporting the other ducks and every prop; a
  // named reset writes one duck's qpos and qvel back to its spawn and leaves
  // the clock, the props and the other ducks exactly where they were. Use it
  // to give one duck another go at something the rest of the world is in the
  // middle of; use the plain form to start a trial.
  if (url.pathname === '/reset') {
    return liveLane(async () => {
      // Asked-for-ness, not truthiness: a duck could legitimately be named
      // `0`, and testing the name for truth would quietly reset the whole
      // world instead of that duck.
      const asked = body?.duck ?? url.searchParams.get('duck');
      const named = (asked === undefined || asked === null || asked === '')
        ? null : pickSlot(url, body);
      if (named) {
        await ensureStanding();
        placeDuck(live.world, named.duck);
        mj.mj_forward(model, live.world);
        named.cmd = { vx: 0, vy: 0, vyaw: 0 };
        named.last = new Array(14).fill(0);
        return { ...stateOf(named), reset: named.duck.name };
      }
      live.standing = false;
      for (const slot of live.slots.values()) slot.cmd = { vx: 0, vy: 0, vyaw: 0 };
      await ensureStanding();
      if (BALL) placeBall(live.world, 0.8, 0, BALL_RADIUS);
      return { ...stateOf(live.slots.get(DUCKS[0].name)), reset: 'the whole world' };
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
      const slot = pickSlot(url, body);
      await ensureStanding();
      slot.cmd = { vx: speed(body.vx, 'vx'), vy: speed(body.vy, 'vy'),
                   vyaw: speed(body.vyaw, 'vyaw') };
      // AND THE HOLD MOVES EVERY DUCK. Only this one's command changed; the
      // others keep the commands they were holding and keep walking under them
      // for the same window, because the world has one clock. Steering three
      // ducks therefore means three calls per window, and between them the
      // world does not wait.
      const ticks = await hold(seconds);
      return { ...stateOf(slot), held: r4(ticks / C.tickHz), ticks };
    });
  }
  // POST /stop — zero the command and let the duck settle under it. Not a
  // reset: stopping is a thing the policy does, and a duck that had to be
  // teleported upright to stop would be hiding the fall.
  // STOPPING ONE DUCK DOES NOT STOP THE WORLD, and that is the honest
  // behaviour rather than a limitation: the settle it asks for is time, and
  // time passing is something the other ducks are also in. `duck` with no name
  // stops the first one, which on the canon scene is the only one.
  if (url.pathname === '/stop') {
    const seconds = Math.min(Math.max(+body.settle || 0.5, 1 / C.tickHz), 5);
    return liveLane(async () => {
      const slot = pickSlot(url, body);
      await ensureStanding();
      slot.cmd = { vx: 0, vy: 0, vyaw: 0 };
      const ticks = await hold(seconds);
      return { ...stateOf(slot), settled: r4(ticks / C.tickHz), ticks };
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
  // A POLICY THAT WAS NOT ALREADY ON THIS DISK.
  //
  // /policy takes a NAME and loads from the bench's own directory, which is
  // right for the nine shipped networks and useless for a network somebody
  // just made. Duck Studio can now write an ONNX — it blends the ones it has —
  // and a blend that cannot be run is a file with nothing behind it.
  //
  // WRITTEN TO A SCRATCH FILE BECAUSE onnxruntime LOADS FROM A PATH. The name
  // is derived from the sha256 of the bytes, never from anything the caller
  // sent: a caller-chosen name is a path traversal waiting to happen, and the
  // digest is also exactly what identifies which network this is.
  //
  // This bench is OPEN on the network it is on, and this endpoint widens what
  // that means — it accepts bytes and runs them. It is a development tool on a
  // private network and was already running arbitrary policies from its own
  // directory, but that is a different sentence from "accepts arbitrary bytes",
  // and anybody putting this on a network they do not trust should know which
  // one they have.
  if (url.pathname === '/upload') {
    const b64 = typeof body.onnx === 'string' ? body.onnx : null;
    if (!b64) return { error: 'upload needs an `onnx` field: the file, base64' };
    let bytes;
    try { bytes = Buffer.from(b64, 'base64'); }
    catch { return { error: 'the `onnx` field is not base64' }; }
    if (!bytes.length) return { error: 'the uploaded policy is empty' };
    if (bytes.length > 8 * 1024 * 1024) {
      return { error: `the uploaded policy is ${bytes.length} bytes; the shipped ones are under 1 MB` };
    }
    const digest = createHash('sha256').update(bytes).digest('hex');
    const name = `uploaded-${digest.slice(0, 12)}`;
    const file = `${name}.onnx`;
    if (!fs.existsSync(file)) fs.writeFileSync(file, bytes);
    try {
      // Load it now rather than at first use, so a file that onnxruntime
      // cannot open is refused HERE with the reason, not three calls later in
      // the middle of a rollout.
      sessions.set(file, {
        name,
        net: await ort.InferenceSession.create('./' + file),
        reference: declaredDefaultPose('./' + file, HOME) ?? HOME,
      });
    } catch (e) {
      fs.unlinkSync(file);
      return { error: `that file did not load as a policy: ${e.message}` };
    }
    return { policy: name, sha256: digest, bytes: bytes.length,
             note: 'loaded and ready — pass this name to /policy, /record, /measure or /perform' };
  }

  if (url.pathname === '/policy') {
    const wanted = typeof body.policy === 'string' ? body.policy : String(body.policy ?? '');
    return liveLane(async () => {
      // ONE SLOT PER DUCK, which is what makes two networks in one world
      // possible: swap huey to a walker and leave dewey standing, and the
      // contact between them is between two different policies. That
      // experiment does not exist on hardware, where each duck is its own
      // process on its own robot and the only thing they share is a room.
      const slot = pickSlot(url, body);
      await ensureStanding();
      const loaded = await policy(wanted);   // throws `unknown policy: …` -> 400
      const was = slot.policy;
      slot.policy = wanted;
      return {
        ...stateOf(slot), was,
        // Whether this policy declared its own neutral pose or inherited HOME:
        // the one thing about a hot swap that is not visible in the duck's pose.
        reference: loaded.reference === HOME ? 'HOME' : 'declared by the policy',
      };
    });
  }
  if (url.pathname === '/record') {
    const seconds = Math.min(Math.max(+body.seconds || 3, 0.2), 30);
    const duck = pickDuck(url, body);
    const run = await batchLane(() =>
      rollout({ name: body.policy, seconds, schedule: body.schedule, duck }));
    const last = run.roots[run.roots.length - 1];
    return {
      format: 'duck-intent-clips/3',
      hz: C.tickHz,
      joints: C.jointNames.filter(n => n !== 'mouth'),
      policy: body.policy,
      // WHICH DUCK THE FRAMES ARE OF. One set of frames, one duck, said out
      // loud — a clip from a multi-duck world that did not name its duck would
      // be a recording of an unidentified robot.
      duck: duck.name,
      // The world this recording came out of, so a clip kept anywhere else
      // can still say where it was made. Same two keys /health reports.
      plantName: PLANT,
      plantDigest: PLANT_DIGEST,
      frames: run.frames, roots: run.roots, commands: run.commands,
      endsUpright: upright(last), endHeight: last[2],
    };
  }
  /*
   * POST /perform — run an AUTHORED motion in real physics.
   *
   * THE HOLE THIS FILLS. Every other endpoint runs a trained policy. An
   * authored motion — keyframes somebody wrote in Duck Studio — could be
   * previewed on a phone and published to the world without any physics engine
   * ever having seen it, because a phone has none. A preview is what you ASKED
   * for; this is what happens.
   *
   * IT RUNS THE MOTION MORE THAN ONCE, and that is the point. A single rollout
   * that stays up proves very little: the four authored stair motions in the
   * corpus get up their flight 0 times in 16. The answer to "does it work" is a
   * count, not a yes.
   *
   * The track rides on the standing policy as offsets from HOME, which is what
   * `record_intents.mjs` does and what the app's own preview draws.
   */
  if (url.pathname === '/perform') {
    const track = Array.isArray(body.track) ? body.track : null;
    if (!track || !track.length) return { error: 'perform needs a track of keyframes' };
    for (const key of track) {
      if (!Array.isArray(key.pose) || key.pose.length !== 14) {
        return { error: 'every keyframe needs a pose of 14 joint angles, mouth excluded' };
      }
      if (!key.pose.every(v => Number.isFinite(v))) {
        return { error: 'a keyframe pose holds something that is not a number' };
      }
      if (!Number.isFinite(+key.at)) return { error: 'every keyframe needs an `at` in seconds' };
    }
    const ordered = track.map(k => ({ at: +k.at, pose: k.pose.map(Number) }))
                         .sort((a, b) => a.at - b.at);
    const blend = Math.min(Math.max(+body.blend || 1, 0), 1);
    // How often the teacher acts rather than the driver. Only meaningful with a
    // driver; 0 means the driver is on its own.
    const teacherShare = expertName
      ? Math.min(Math.max(body.teacherShare ?? 0.9, 0), 1) : 0;
    // Radians of noise added to the commanded joints AFTER labelling, so the
    // duck wanders off the demonstration and the labels teach the way back.
    const jitter = Math.min(Math.max(+body.jitter || 0, 0), 0.2);
    const seconds = Math.min(Math.max(+body.seconds || (ordered[ordered.length - 1].at + 0.5),
                                      0.2), 30);
    const rollouts = Math.min(Math.max(+body.rollouts || 8, 1), 32);
    const name = body.policy || STAND;
    const duck = pickDuck(url, body);

    let first = null, ok = 0;
    const heights = [];
    for (let i = 0; i < rollouts; i++) {
      // Pollen's own randomisation, as measure_success.mjs uses it: the drop
      // height is what a bench can vary without touching the model.
      const drop = 0.12 + (0.01 * i) / Math.max(rollouts - 1, 1);
      const run = await batchLane(() =>
        rollout({ name, seconds, schedule: body.schedule, track: ordered, blend, drop, duck }));
      const last = run.roots[run.roots.length - 1];
      if (upright(last)) ok++;
      heights.push(last[2]);
      if (!first) first = run;
    }
    heights.sort((a, b) => a - b);
    const last = first.roots[first.roots.length - 1];

    // Peak joint rate over the first rollout: the fastest any joint was
    // actually moved, which is the number an authored track most often
    // overruns without noticing.
    let peak = 0;
    for (let t = 1; t < first.frames.length; t++) {
      for (let k = 0; k < 14; k++) {
        const rate = Math.abs(first.frames[t][k] - first.frames[t - 1][k]) * C.tickHz;
        if (rate > peak) peak = rate;
      }
    }

    return {
      format: 'duck-intent-clips/3',
      hz: C.tickHz,
      joints: C.jointNames.filter(n => n !== 'mouth'),
      policy: name,
      duck: duck.name,
      authored: true,
      // THE ANSWER A CALLER FILES AWAY. /perform is the one endpoint whose
      // result is stored and shown months later, so it is the one that most
      // has to name its own world rather than let the caller guess.
      plantName: PLANT,
      plantDigest: PLANT_DIGEST,
      blend,
      frames: first.frames, roots: first.roots, commands: first.commands,
      rollouts, achieves: ok,
      criterion: 'stayed upright to the end, over drop heights 0.120-0.130 m',
      medianHeight: r4(heights[Math.floor(heights.length / 2)]),
      endsUpright: upright(last), endHeight: r4(last[2]),
      peakJointRate: r4(peak),
    };
  }
  /*
   * POST /capture — the pairs a POLICY would have to reproduce to perform an
   * authored motion by itself.
   *
   * THE LAST MILE STARTS HERE, AND IT IS NOT A FILE FORMAT PROBLEM. A motion is
   * a function of time; a policy is a function of state, closed-loop at 50 Hz.
   * robotd runs the second kind and nothing else, so an authored motion reaches
   * a real duck only by being cloned into one. This endpoint produces the
   * training set for that: for every control step, the observation the network
   * was shown and the action it would have had to output.
   *
   * IT RUNS THE MOTION SEVERAL TIMES ON PURPOSE. One rollout teaches a clone a
   * single trajectory from a single drop height; the randomised drops are what
   * give it anything to generalise from. It is still a narrow dataset and the
   * clone is still a hypothesis — /measure is what settles whether it worked.
   */
  if (url.pathname === '/capture') {
    const track = Array.isArray(body.track) ? body.track : null;
    if (!track || !track.length) return { error: 'capture needs a track of keyframes' };
    for (const key of track) {
      if (!Array.isArray(key.pose) || key.pose.length !== 14) {
        return { error: 'every keyframe needs a pose of 14 joint angles, mouth excluded' };
      }
      if (!key.pose.every(v => Number.isFinite(v))) {
        return { error: 'a keyframe pose holds something that is not a number' };
      }
      if (!Number.isFinite(+key.at)) return { error: 'every keyframe needs an `at` in seconds' };
    }
    const ordered = track.map(k => ({ at: +k.at, pose: k.pose.map(Number) }))
                         .sort((a, b) => a.at - b.at);
    const rollouts = Math.min(Math.max(+body.rollouts || 8, 1), 32);
    const seconds = Math.min(Math.max(+body.seconds || (ordered[ordered.length - 1].at + 0.5),
                                      0.2), 30);
    // WHO DRIVES, AND WHO TEACHES. Without `driver` the authored motion runs on
    // the standing policy and labels itself — the first dataset. With one, the
    // named policy drives the duck into states of its OWN, and every label is
    // what the authored motion would have commanded from there. That second
    // dataset is the one a clone that fell over needs.
    const name = body.driver || body.policy || STAND;
    const expertName = body.driver ? (body.policy || STAND) : null;
    const blend = Math.min(Math.max(+body.blend || 1, 0), 1);
    // How often the teacher acts rather than the driver. Only meaningful with a
    // driver; 0 means the driver is on its own.
    const teacherShare = expertName
      ? Math.min(Math.max(body.teacherShare ?? 0.9, 0), 1) : 0;
    // Radians of noise added to the commanded joints AFTER labelling, so the
    // duck wanders off the demonstration and the labels teach the way back.
    const jitter = Math.min(Math.max(+body.jitter || 0, 0), 0.2);
    const duck = pickDuck(url, body);

    const pairs = [];
    // NOT `upright`: that is the module-level predicate, and a counter of the
    // same name shadows it inside this block — "upright is not a function".
    let stoodUp = 0;
    for (let i = 0; i < rollouts; i++) {
      const drop = 0.12 + (0.01 * i) / Math.max(rollouts - 1, 1);
      const run = await batchLane(() => rollout({
        name, seconds, schedule: body.schedule, track: ordered, blend, drop,
        capture: pairs, expertName, teacherShare, jitter, duck }));
      if (endedStanding(run)) stoodUp++;
    }
    function endedStanding(run) {
      const last = run.roots[run.roots.length - 1];
      return upright(last) && last[2] >= 0.100;
    }
    return {
      format: 'duck-clone-pairs/1',
      hz: C.tickHz,
      obsWidth: 61,
      actionWidth: 14,
      rollouts, seconds,
      duck: duck.name,
      ridingOn: expertName ?? name,
      drivenBy: name,
      corrective: expertName != null,
      teacherShare,
      jitter,
      // THE SOURCE MOTION'S OWN RECORD. A clone measured later has to be
      // comparable to the thing it was cloned from, and both are only
      // comparable within one world.
      plantName: PLANT,
      plantDigest: PLANT_DIGEST,
      uprightRollouts: stoodUp,
      criterion: 'the authored run itself ended standing, trunk at least 100 mm up',
      pairs: pairs.length,
      // Steps the physics diverged on, dropped rather than exported as nulls.
      rejected: pairs.rejected || 0,
      obs: pairs.map(p => p.obs.map(r4)),
      actions: pairs.map(p => p.action.map(r4)),
    };
  }

  if (url.pathname === '/measure') {
    // The randomisation is Pollen's own, as measure_success.mjs uses it: the
    // drop height is what a bench can vary without touching the model.
    const rollouts = Math.min(Math.max(+body.rollouts || 8, 1), 32);
    const seconds = Math.min(Math.max(+body.seconds || 3, 0.2), 30);
    const duck = pickDuck(url, body);
    let ok = 0; const heights = [];
    for (let i = 0; i < rollouts; i++) {
      const drop = 0.12 + (0.01 * i) / Math.max(rollouts - 1, 1);
      const run = await batchLane(() =>
        rollout({ name: body.policy, seconds, schedule: body.schedule, drop, duck }));
      const last = run.roots[run.roots.length - 1];
      heights.push(last[2]);
      if (upright(last) && last[2] >= 0.100) ok++;
    }
    heights.sort((a, b) => a - b);
    return {
      policy: body.policy, rollouts, achieves: ok, duck: duck.name,
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
  // ONE MEGABYTE EVERYWHERE EXCEPT THE ONE ENDPOINT THAT CARRIES A FILE.
  // Every other body here is a handful of numbers, and a cap is what stops a
  // stray client filling this process's memory. /upload carries a policy: the
  // shipped ones are about 790 KB, which is ~1.05 MB once base64'd, so the old
  // cap silently destroyed the request — the client saw a 100 and no answer,
  // which took longer to work out than it should have.
  const cap = url.pathname === '/upload' ? 12e6 : 1e6;
  let raw = '';
  req.on('data', chunk => { raw += chunk; if (raw.length > cap) req.destroy(); });
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
  console.log(`ducks: ${DUCK_NAMES.join(', ')} — one world, one clock`
            + `${DUCKS.length > 1 ? ' (pass `duck` to /intent /policy /state /stop /ball /reset)' : ''}`);
  console.log('records/measures: /record /measure /perform — steers: /state /intent /stop /now /policy');
  console.log(`policies: ${[...catalogue().keys()].sort().join(', ')}`);
});
