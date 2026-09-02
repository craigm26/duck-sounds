# harness/ — a SNAPSHOT, for reading

**These files are here so you can read the criterion, the grid and the transcribed reward without
cloning anything. They are not runnable as they sit.**

The runnable harness is the GitHub repository:

    git clone https://github.com/craigm26/duck-sounds.git
    cd duck-sounds && git checkout ball-challenge-v1   # or main at/after 2026-09-02 if the tag is missing
    cd sim && npm ci

Running the scorer needs things that are not in this package:

- `mujoco` (the wasm build) and `onnxruntime-node`, installed by `npm ci` in `sim/`;
- the **compiled** plant `sim/scene.mjb`
  (sha256 `3f8c9ab9b409ba74c73c30179d5f7c12b025f631693f9eec78d80dca242547be`). The
  `scene_physics.xml` in this folder is the source it is built from, not the binary the scorer
  loads;
- the policy weights — `alpha_stand.onnx`, which settles every episode and which a *move* entrant
  rides on, and whichever of `alpha_walking.onnx`, `ball_kick_left.onnx`, `ball_kick_right.onnx`
  a *policy* entrant names;
- `site/duckloop.mjs`, which `chase_rig.mjs` imports for the observation layout and the joint
  filter, and `sim/onnx_meta.mjs` for `declaredDefaultPoseOf`;
- `chase/node_modules`, a committed symlink to `../sim/node_modules`. Without it `chase_robust`
  and `chase_parity` cannot resolve `mujoco` or `onnxruntime-node`.

`chase_rig.mjs` reads `duckkit-constants.json` and `scene.mjb` from the **current working
directory**, so everything must be run from `sim/`:

    cd ~/duck-sounds/sim && node ../chase/chase_robust.mjs
    cd ~/duck-sounds/sim && node ../chase/chase_parity.mjs

## What is in here

| file | in the repo | what it is |
|---|---|---|
| `chase_score.mjs` | `sim/chase_score.mjs` | **the instrument.** The four criterion constants at lines 57–69, `CRITERION_SENTENCE` at 85, the fourteen cells in `gridCells()` at 121, `JOINT_ORDER` at 181 and `assertJointOrder()` at 188, Pollen's nine computable `TERMS` at 214, `ACTION_RATE_SOURCE` at 257, the three `CHASE_REFUSALS` at 276, `BALL_CAVEAT` at 301, `checkEntrant` at 333, `commandAt` at 378, `entrantHashPayload` at 401 — the definition of an entrant's identity — and `verdict()` at 425. One shared module: the bench, the desk rig and the grid runner all call it. |
| `reward_math.mjs` | `sim/reward_math.mjs` | `gravityXYSquared`, `rotate`, `unrotate`, `twistOf`, `yawOf`. Extracted out of `duckbench-core.mjs` so the ball rail and the `/tune` rail share one transcription of the three formulas their two Pollen configs have in common, rather than keeping a second copy each. `duckbench-core.mjs` re-exports all of them under their existing names. |
| `climb_score.mjs` | `sim/climb_score.mjs` | the **stairs** challenge's shared module, here because `chase_score.mjs` imports three things from it: `poseAt` (the keyframe interpolation a move entrant rides), `UPRIGHT_TAIL_MIN` (line 89, the 45-of-50 bar, imported rather than retyped) and `PLANTS` (line 75), from which the two extended plant pairs are taken verbatim. |
| `chase_rig.mjs` | `chase/chase_rig.mjs` | the desk machine. Its own mujoco module, its own `mjData`, its own onnxruntime sessions. `chaseCell()` scores one cell of one entrant; `scoreSaved()` does the same from a file. Deliberately thin: it adds the hash and the plant identity and reshapes nothing, because a shape assembled twice is a shape that can disagree twice. |
| `chase_robust.mjs` | `chase/chase_robust.mjs` | the grid. `CONTROLS` at line 36, `scoreChase()` at 58, `leaderboardRow` at 130. `kChased`/`kStable` of 9 core, `kExt` of 5 extended, the centre-cell travel, the nine terms averaged, and the rounded per-cell `verdicts` this package publishes. |
| `chase_parity.mjs` | `chase/chase_parity.mjs` | the acceptance test. Every bundled entrant × all fourteen cells, scored twice — through the bench's in-process `POST /chase` handle and through `chase_robust.scoreChase` — compared with `Object.is` at full float digits on 49 fields, with the aggregates recomputed from the `/chase` answers alone. Its output for the published run is `results/chase_parity.log`. |
| `measure_drift.mjs` | *new in this package* | the measurement behind the card's headline: the naive chaser's open-loop drift on the four bearing-0 cells. Writes `results/chase_drift-results.json`. It re-scores one published control on four published cells through `chase_rig.mjs` — no new scorer, no new criterion. |
| `scene_physics.xml` | `sim/scene_physics.xml` | the plant source: solver, friction, contact priorities, the ±0.6405 N·m actuator `forcerange` (line 46) and the ball — `body name="ball"`, `freejoint ball_free`, `geom ball_geom` sphere radius 0.05, mass 0.03, `condim 6` (lines 196–203). |
| `duckkit-constants.json` | `site/duckkit-constants.json` | joint names, HOME pose, joint limits, the observation constants. 15 joints; the policy and every entrant use 14 (all but `mouth`). This is the file `assertJointOrder()` checks at boot. |

Nothing in this folder has been modified from the repository, except `measure_drift.mjs`, which
does not exist there. Compare with `MANIFEST.json` in the package root.
