# harness/ — a SNAPSHOT, for reading

**These files are here so you can read the criterion, the grid and the landing laws without
cloning anything. They are not runnable as they sit.**

The runnable harness is the GitHub repository:

    git clone https://github.com/craigm26/duck-sounds.git
    cd duck-sounds && git checkout stairs-challenge-v1   # or main at/after 2026-09-02 if the tag is missing
    cd sim && npm ci

Running the scorer needs things that are not in this package:

- `mujoco` (the wasm build) and `onnxruntime-node`, installed by `npm ci` in `sim/`;
- the **compiled** plant `sim/scene.mjb`
  (sha256 `3f8c9ab9b409ba74c73c30179d5f7c12b025f631693f9eec78d80dca242547be`). The
  `scene_physics.xml` in this folder is the source it is built from, not the binary the scorer
  loads;
- the policy weights `sim/BEST_alpha_stand.onnx`, which the 50-tick tail runs;
- `site/duckloop.mjs`, which `rig3.mjs` imports for the observation layout and the joint filter.

`rig3.mjs` reads `duckkit-constants.json` and `scene.mjb` from the **current working directory**,
so it must be run from `sim/`:

    cd ~/duck-sounds/sim && node ../climb/rig3.mjs

## What is in here

| file | in the repo | what it is |
|---|---|---|
| `rig3.mjs` | `climb/rig3.mjs` | the instrument. `criteria()` at line 315 defines `honest` at line 330; `footResting()` at line 145; `scoreSaved()` at line 726. |
| `robust.mjs` | `climb/robust.mjs` | the shared scorer. The grid at lines 494–518, `UPRIGHT_TAIL_MIN = 45` at line 528, `scoreRobust()` at line 656, `intentHash()` — the definition of a vector's identity — above it. |
| `servo.mjs` | `climb/servo.mjs` | the round-5 servoed landing. Its header states why every servoed result is an ORACLE. |
| `audit_r6.mjs` | `climb/audit_r6.mjs` | the round-6 adversarial judge that produced `results/r6_judge-results.json`. `ceilingCore` is computed at line 475. |
| `stairs.js` | `site/stairs.js` | the flight. `STAIR_HALF_WIDTH = 0.17` (the 340 mm gate), `STAIR_Y`, and `isolateSteps()` — the 2026-09-02 repair (commit `279b016`) without which the step blocks interpenetrate at any rise under 200 mm. |
| `scene_physics.xml` | `sim/scene_physics.xml` | the plant source: solver, friction, contact priorities and the ±0.6405 N·m actuator `forcerange`. |
| `duckkit-constants.json` | `site/duckkit-constants.json` | joint names, HOME pose, joint limits, the observation constants. 15 joints; the policy and every intent use 14 (all but `mouth`). |

Nothing in this folder has been modified from the repository. Compare with `MANIFEST.json` in
the package root.
