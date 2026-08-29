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

## 2026-08-29 — the head, the rollers, and the assets you are drawing

Findings from the duckkit side that touch `site/`; reported here, not applied.

1. **The jaw hinge exists now, derived — not vendored.** Pollen's plant fuses
   the lower beak into the head (`scripts/bake-duck-mesh.py` in
   pollen-robotics/microduck: "`mouth` is a servo without an MJCF joint (the
   jaw is a fixed geom)"). `robot_allcollisions.xml` does place the mouth's
   XL330 in the head body at pos (0.003, 0.0255, −0.018) quat (0.707,0,0,−0.707);
   the same servo mesh sits at every real joint with its horn 14.5 mm along
   the STL's +X, which puts the mouth horn — the hinge — at **(0.003, 0.040,
   −0.018) in the head frame, axis lateral (head +Y)**. jaw.stl has a hub of
   vertices 3–9 mm from that line. The plant comments (lines 76–79) say the
   real jaw rides a closed-loop linkage, so a single hinge is an approximation
   of a 4-bar; it is labelled as ours in DuckKinematics. Sense: +angle lowers
   the tip; runtime range −5° (pressed shut) … +30° (wide). If the site ever
   draws the beak opening, use this hinge and say the same thing.
   Moving parts: `jaw` + `jaw_soft` (lower); `soft_mouth_top` stays with the shell.

2. **`sim/assets/*.stl` are NOT the microduck_rl assets.** `git hash-object
   sim/assets/jaw.stl` = c76bfb66… vs the RL repo's d7b0e12b… (develop @
   d424a0c), and the whole set is ~188k triangles where the RL set is ~797k.
   The plant XMLs ARE byte-identical to the RL repo's; only the meshes differ.
   Whatever `scene.mjb` / `duck-visual*.bin` were built from, their provenance
   is not "microduck_rl" unless it was rebuilt from a fresh clone. duckkit
   rebuilt from a sparse clone of `src/mjlab_microduck/robot/microduck`
   (`tools/export_duck_mesh.py`); the same clone is at the RL repo for you.

3. **One colour per body should be the DOMINANT part's**, not the last geom in
   the file: the head is otherwise speaker-grey and the trunk battery-black.
   duckkit's exporter now picks the material of the geom with the most
   triangles; the beak comes out yellow (jaw_material 0.98, 0.71, 0.004).

4. **Rollers are recorded now.** `sim/record_rollers.mjs` (new) runs
   BEST_roller.onnx on `scene-rollers.mjb` and writes `skate_stand`, `skate`,
   `skate_fast`, `skate_back`, `skate_turn` into `duck-trajectories.json`,
   cut to whole swizzle cycles (period ≈31 ticks, worst-joint seam ≤0.06 rad,
   clamped to servo travel after rounding, unwrapped yaw). `record_intents.mjs`
   learns `SCENE=rollers` (roller specs only; settles under BEST_roller; merges)
   and records `roller_crouch` with the ground-pick phase clock over 5 s. Every
   clip carries `variant: "rollers"`. Still on the older rollers scene — the
   training-parameter rebuild of build_rollers.py is still pending — and the
   turn clip's yaw is not to be trusted there (the policy turned the other way
   and much faster than commanded).
