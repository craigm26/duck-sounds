// ac_build2.mjs — build ac_scene2.mjb: the FULL harness scene (stairs, walls,
// ball, blocks, cones — byte-for-byte the props of scene_physics.xml) with the
// training-plant parameters from the microduck_rl/mjlab/BAM investigation
// applied. This is the scene ac_record.mjs / ac_success.mjs run on.
//
// XML-level deltas vs scene_physics.xml (the source of scene.mjb):
//   1. <option> gains iterations="10" ls_iterations="20" — mjlab
//      SimulationCfg MujocoCfg(iterations=10, ls_iterations=20); the harness
//      was running the XML defaults 100/50.
//   2. floor friction 1.0 0.02 0.001 → 1.0 0.005 0.0001 — mjlab's plane is a
//      bare default plane (terrain_entity.py:_import_ground_plane).
//   3. left/right_foot_collision gain priority="1" friction="1.0 0.005 0.0001"
//      — mjlab FULL_COLLISION CollisionCfg: feet priority 1, sliding 1.0.
//   4. chosen_actuator: position forcerange ±0.96 → ±0.6405. The XL330
//      firmware current limit (max_current = 1.75 A, bam/dynamixel/actuator.py)
//      caps stall torque at kt·1.75 = 0.36601·1.75 = 0.6405 Nm; the ±0.96 of
//      the shipped XML is the no-current-limit PWM bound at 7.35 V and lets
//      the duck be ~50% stronger at stall than the robot Pollen trained.
//   5. chosen_actuator joints gain solreffriction="-50000 -200"
//      solimpfriction="0.99 0.9999 0.001 0.5 2" — BAM stiff_frictionloss=True
//      (bam/mjlab.py _STIFF_SOLREF_FRICTION), so the dry-friction constraint
//      is stiff rather than MuJoCo's soft default (0.02, 1).
//
// NOT in the XML (implemented per physics substep in ac_plant.mjs, because
// they are state-dependent): the BAM m6 friction budget written into
// dof_frictionloss, and the 3–6 physics-step actuation delay on ctrl.
//
// Kept AS the harness had them (they are project props, not training):
//   step blocks contype4/conaffinity4, walls, ball, blocks, cones; the robot
//   collision class stays contype5/conaffinity5 — required for the stairs to
//   collide with the robot but not the floor; robot↔robot and robot↔floor
//   pairing is equivalent to Pollen's 1/1 (established by the plant probe).
//
// Run:  node ac_build2.mjs        (from this directory)
import load from 'mujoco';
import fs from 'node:fs';

let xml = fs.readFileSync('scene_physics.xml', 'utf8');
const edits = [];
function edit(name, from, to) {
  const n = xml.split(from).length - 1;
  if (n !== 1) throw new Error(`edit "${name}": pattern found ${n} times, need exactly 1`);
  xml = xml.replace(from, to);
  edits.push(name);
}

// 1 — solver iterations (training: 10 / 20; XML default was 100 / 50).
edit('solver iterations',
  '<option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast" />',
  '<option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast" iterations="10" ls_iterations="20" />');

// 2 — floor friction: mjlab's plane is all-default.
edit('floor friction',
  `<geom name="floor" type="plane" size="6 6 0.05" pos="0 0 0" rgba="0.16 0.18 0.16 1"
          contype="1" conaffinity="1" condim="3" friction="1.0 0.02 0.001" />`,
  `<geom name="floor" type="plane" size="6 6 0.05" pos="0 0 0" rgba="0.16 0.18 0.16 1"
          contype="1" conaffinity="1" condim="3" friction="1.0 0.005 0.0001" />`);

// 3 — feet: priority 1, sliding friction 1.0 (mjlab FULL_COLLISION).
for (const side of ['left', 'right']) {
  const quat = side === 'left' ? '0.5 -0.5 -0.5 -0.5' : '0.5 -0.5 0.5 0.5';
  const x = side === 'left' ? '-0.022' : '0.022';
  edit(`${side} foot priority`,
    `<geom type="mesh" name="${side}_foot_collision" class="collision" pos="${x} 0.00622291 -0.0647" quat="${quat}" mesh="sole_${side}"/>`,
    `<geom type="mesh" name="${side}_foot_collision" class="collision" pos="${x} 0.00622291 -0.0647" quat="${quat}" mesh="sole_${side}" priority="1" friction="1.0 0.005 0.0001"/>`);
}

// 4 + 5 — chosen_actuator: current-limited forcerange; stiff friction constraint.
edit('chosen_actuator',
  `<joint damping="0.053" frictionloss="0.0048" armature="0.0018"/>
      <!-- 200 kp -->
      <position kp="0.55" kv="0.0" forcerange="-0.96 0.96" ctrlrange="-10.0 10.0"/>`,
  `<joint damping="0.053" frictionloss="0.0048" armature="0.0018" solreffriction="-50000 -200" solimpfriction="0.99 0.9999 0.001 0.5 2"/>
      <!-- 200 kp -->
      <position kp="0.55" kv="0.0" forcerange="-0.6405 0.6405" ctrlrange="-10.0 10.0"/>`);

fs.writeFileSync('ac_scene2.xml', xml);
console.log('ac_scene2.xml written. Edits applied:', edits.join(' | '));

const mj = await load();
try { mj.FS.mkdir('/assets'); } catch {}
for (const f of fs.readdirSync('assets'))
  mj.FS.writeFile('/assets/' + f, fs.readFileSync('assets/' + f));
mj.FS.writeFile('/ac_scene2.xml', xml);
const model = mj.MjModel.mj_loadXML('/ac_scene2.xml');
console.log('compiled: nq', model.nq, 'nv', model.nv, 'nu', model.nu,
  'ngeom', model.ngeom, 'nsensor', model.nsensor,
  'timestep', model.opt.timestep, 'iterations', model.opt.iterations,
  'ls_iterations', model.opt.ls_iterations);
// Verify the four parameter groups actually landed in the compiled model.
for (let g = 0; g < model.ngeom; g++) {
  const n = model.geom(g).name;
  if (n === 'floor' || /foot_collision/.test(n))
    console.log(`  geom ${n}: priority ${model.geom_priority[g]} friction`,
      [0, 1, 2].map(i => model.geom_friction[g * 3 + i]).join(' '));
}
for (let a = 0; a < model.nu; a++) {
  if (a === 0 || a === model.nu - 1)
    console.log(`  actuator ${a}: forcerange ±${model.actuator_forcerange[a * 2 + 1]}`);
}
mj.mj_saveModel(model, '/ac_scene2.mjb', null);
const bytes = mj.FS.readFile('/ac_scene2.mjb');
fs.writeFileSync('ac_scene2.mjb', Buffer.from(bytes));
console.log('SAVED ac_scene2.mjb', (bytes.length / 1024).toFixed(1), 'KB');
