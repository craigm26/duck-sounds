// ac_build.mjs — build ac_scene.mjb: Pollen's ALLCOLLISIONS robot on a plain
// mjlab-style plane, compiled for the WASM 'mujoco' npm package.
//
// PROVENANCE, ESTABLISHED 2026-08-28. The robot source is
// src/mjlab_microduck/robot/microduck/robot_allcollisions.xml from
// pollen-robotics/microduck_rl. Verified byte-identical (md5
// 32e5246a4da023f4b318f02c74c35a04) across: the local checkout of branch
// `develop` in the scratchpad, a fresh fetch of branch `main` from
// raw.githubusercontent.com, and — this was the surprise — this project's own
// pollen_robot.xml. The prior investigation's claim that scene.mjb compiles
// Pollen's WALK model is FALSE: pollen_robot.xml IS robot_allcollisions.xml,
// with head collision (top_head_shell, bottom_head_shell, jaw) included. The
// repo's actual walk model (robot_walk.xml) has only sole_left/sole_right.
//
// What this scene changes vs the existing scene.mjb, to be a clean probe:
//   - no stairs, walls, ball, blocks or cones — just robot + plane, which is
//     Pollen's scene.xml minus visual dressing and keyframes;
//   - the collision default keeps Pollen's own contype/conaffinity (MuJoCo
//     default 1/1), NOT the harness's contype=5/conaffinity=5 stair mask;
//   - the floor is mjlab's: a plane with ALL MuJoCo defaults
//     (friction 1 0.005 0.0001, condim 3, priority 0) — see
//     mjlab/terrains/terrain_entity.py:_import_ground_plane, which adds
//     type=PLANE size=(0,0,0.01) and nothing else. The harness floor's
//     friction="1.0 0.02 0.001" is a (small) deviation from training.
//   - option: timestep 0.005, integrator implicitfast — mjlab SimulationCfg
//     (velocity_env_cfg.py:446 timestep=0.005, decimation=4 → the recorder's
//     50 Hz).
//
// Same stripping as build_physics_only.py: visual geoms out, unused meshes
// out, materials out. Physics is untouched by this — every body carries an
// explicit <inertial>, and visual geoms are contype=0 conaffinity=0.
//
// Run:  node ac_build.mjs        (from this directory)
import load from 'mujoco';
import fs from 'node:fs';
import crypto from 'node:crypto';

const REPO_COPY = '/tmp/claude-1000/-home-craigm26-projects/0319e4f7-7237-4a68-86d3-7a005c2c7514/scratchpad/microduck_rl-develop/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml';
const LOCAL = 'pollen_robot.xml';

// Prefer the repo checkout; fall back to the byte-identical local copy.
const srcPath = fs.existsSync(REPO_COPY) ? REPO_COPY : LOCAL;
const src = fs.readFileSync(srcPath, 'utf8');
const md5 = crypto.createHash('md5').update(src).digest('hex');
console.log('robot source:', srcPath);
console.log('md5:', md5, md5 === '32e5246a4da023f4b318f02c74c35a04'
  ? '(= pollen-robotics/microduck_rl robot_allcollisions.xml, main and develop)'
  : '(WARNING: unexpected hash — source drifted?)');

// 1 — strip visual geoms.
let phys = src.replace(/\s*<geom\b[^>]*class="visual"[^>]*\/>/g, '');

// 2 — drop mesh assets nothing references any more. These <mesh> elements
// carry no name attribute; MuJoCo derives the mesh name from the filename.
const kept = new Set([...phys.matchAll(/mesh="([^"]+)"/g)].map(m => m[1]));
phys = phys.replace(/\s*<mesh\b[^>]*\/>/g, tag => {
  const f = tag.match(/file="([^"]+)\.stl"/);
  return f && kept.has(f[1]) ? tag : '';
});

// 3 — materials reference textures we are not loading.
phys = phys.replace(/\s*<material\b[^>]*\/>/g, '');
phys = phys.replace(/\s+material="[^"]*"/g, '');

// 4 — sim options + mjlab's default plane, ahead of the robot's worldbody.
// (String.replace with a plain string replaces only the FIRST occurrence.)
const add = `  <option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast" />
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.01" pos="0 0 0" rgba="0.16 0.18 0.16 1" />
  </worldbody>
`;
phys = phys.replace('<worldbody>', add + '  <worldbody>');

fs.writeFileSync('ac_scene.xml', phys);
console.log('ac_scene.xml written,', phys.length, 'bytes; meshes kept:',
  [...kept].sort().join(', '));

// 5 — compile in the same WASM runtime the recorder uses, save the .mjb.
const mj = await load();
try { mj.FS.mkdir('/assets'); } catch {}
for (const f of fs.readdirSync('assets'))
  mj.FS.writeFile('/assets/' + f, fs.readFileSync('assets/' + f));
mj.FS.writeFile('/ac_scene.xml', phys);
const model = mj.MjModel.mj_loadXML('/ac_scene.xml');
console.log('compiled: nq', model.nq, 'nv', model.nv, 'nu', model.nu,
  'ngeom', model.ngeom, 'nsensor', model.nsensor, 'timestep', model.opt.timestep);
mj.mj_saveModel(model, '/ac_scene.mjb', null);
const bytes = mj.FS.readFile('/ac_scene.mjb');
fs.writeFileSync('ac_scene.mjb', Buffer.from(bytes));
console.log('SAVED ac_scene.mjb', (bytes.length / 1024).toFixed(1), 'KB');
