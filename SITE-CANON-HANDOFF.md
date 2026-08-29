# Canon handoff for the site session — 2026-08-28

From the canon-accuracy audit (Duck Studio session). `site/` is yours, so these
are findings, not edits. The sim/ side of everything below has already landed.

## 1. index.html:493-497 is live and false (highest priority)

The paragraph tells visitors Pollen's model "is kinematics and mass only: it
has no collision shapes and no motors", the soles are "boxes we drew", the
gains "we picked", friction "is a guess", and the duck "drifts... covers about
half the ground per second". All of that describes the pre-0639d96 invented
body. What the site actually ships today: Pollen's robot_allcollisions.xml
byte-identical (mesh sole collisions, head collision meshes, their measured
chosen_actuator class). Suggested shape for the rewrite: Pollen's model ships
full collision geometry and their servo class; ours are the floor/stairs/props,
the browser port, and — until item 2 lands — the solver/friction/torque deltas.
The "trust the policy, doubt the body" framing survives with the true residuals.

## 2. site/scene.mjb is the superseded plant

md5 4c3f30c8 == pre-bb79f54 sim/scene.mjb. Current canon (02de25ed) carries
training's parameters: solver iterations 10/20, floor friction 1.0/0.005/0.0001,
feet priority=1 with training friction, forcerange ±0.6405, stiff joint-friction
solref. site/sim.js already drives every policy training-path (HOME + action,
scale 1.0 — verified coherent), so it should run on the training plant: copy
sim/scene.mjb in and redeploy. CAVEATS measured on the plant change:
- Pollen's corpus is nearly invariant (16/16 across the board, back_roll 14/16).
- The six AUTHORED keyframe tracks were searched on the old plant — re-verify
  (stepverify) before repeating any height/success claim; headspin dropped
  8/16 → 1/16 across this change, authored moves may shift too.
- alpha_walking has a low-command dead band on the canon plant: vx ≤ 0.2
  marches in place (1-2 mm/s); 0.25 → 0.106 m/s, 0.35 → 0.150 m/s. The site's
  VEL_FWD 0.25 is exactly at the knee — consider 0.3 if drive feels dead.

## 3. site/scene-rollers.mjb has no training-parameter version anywhere

Blocked on a build_rollers.py rebuild (port the bb79f54 post-edits, but take
torque ceilings PER-ACTUATOR from the microduck_rl roller env — pollen_rollers.xml
declares ±0.67/±0.75/±0.91/±0.96/±20.0, do NOT blanket ±0.6405). Then re-verify
skating with BEST_roller.onnx before copying into site/.

## 4. Small
- record.mjs (sim/) now records trajectories on the canon plant, training path,
  with a standing settle; if the site ever replays duck-trajectories.json,
  the new clips are straight (old walk veered ~1 rad/loop).
- sim/BEST_alpha_walking.onnx (inert byte-duplicate) was deleted.
