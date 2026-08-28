# Where the model comes from

**Thanks to Pollen Robotics**, who designed and built the Microduck, trained the
policy, and gave permission for their model to be used here. The robot, the
network, the MuJoCo model, the servo parameters and every mesh are their work.
This project adds a floor, a camera, and the arithmetic joining them up.

The simulator runs **Pollen Robotics' own physics model**, used **with Pollen's
permission** (granted to the operator; not implied by any licence file — the
`microduck-simulator` Space declares none).

| Asset | Source | Note |
|---|---|---|
| `pollen_robot.xml` | `pollen-robotics/microduck-simulator` Space, `app/public/robot/mjlab/robot_allcollisions.xml` | Collision geometry, real actuator gains |
| `assets/*.stl` (38) | same Space, `.../meshes/` | Visual + collision meshes |
| `alpha_walking.onnx` | `pollen-robotics/microduck` (Apache-2.0) | Byte-identical to the Space's `BEST_alpha_walking.onnx` — verified by md5 |

**Do not re-vendor these from the Space into a public repo without checking the
permission still stands and how Pollen wants to be credited.**

## What this replaced, and why it matters

Before this, the scene was a floor, two box soles and position servos whose
gains were guesses, because Pollen's *published* MJCF (in the Apache-2.0 repo)
carries mass and kinematics but **no collision shapes and no actuators**. The
difference the real model makes:

| | invented model | Pollen's model |
|---|---|---|
| Forward speed | 0.078 m/s | 0.23 m/s |
| Walks straight | no, −1.15 rad yaw per loop | yes |
| Turns right | never (no gain setting worked) | yes, correct sign |
| Falls over | at any usable stiffness | never observed |

The real servo class is `kp="0.55" forcerange="±0.96"` — far softer than the
`kp=9, ±2.5` that had been guessed. With servos that soft it is the **policy**
that holds the duck up, which is why settling without the policy running just
collapses it.

## Three things that were wrong in our loop, found by comparing

1. **The gyro is not `sensordata[0]`.** Their sensor block opens with a 4-value
   `framequat`, so the angular-velocity sensor the runtime reads sits at
   address 7. We were feeding the policy three components of a quaternion.
2. **The simulator's loop is not robotd's.** mjlab drives
   `ctrl = pose + action × 1.0` with no low-pass. DuckKit's `0.9` and its
   filter are robotd's on-robot behaviour — correct there, wrong here.
3. **The command has a floor.** Below about 0.3 the policy simply stands.
   0.15 — the robot's documented max in m/s — produces no motion at all, so
   the command units in this model are not m/s.

## Build steps

```bash
node compile.mjs        # scene_physics.xml -> scene.mjb (precompiled)
node pollen_test.mjs    # headless: does it walk, does it turn, does it fall
node record.mjs         # capture clips for DuckKit's DuckTrajectory
```

`scene.mjb` is precompiled deliberately: compiling a meshed MJCF in the browser
fails with `thread constructor failed`, because MuJoCo parallelises convex-hull
generation and the WASM build cannot spawn those workers. Compiling in Node
once sidesteps it — and means the page fetches no STLs at all.
