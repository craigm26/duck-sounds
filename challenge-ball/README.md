---
license: cc-by-4.0
pretty_name: "Microduck Ball Challenge"
tags:
  - mujoco
  - robotics
  - bipedal
  - microduck
  - pollen-robotics
  - benchmark
  - ball
  - locomotion
task_categories:
  - robotics
  - reinforcement-learning
---

# Microduck Ball Challenge

**Simulation only, never on hardware.** Every number in this card was produced in MuJoCo,
in the plant `sim/scene.mjb` (sha256 `3f8c9ab9b409ba74c73c30179d5f7c12b025f631693f9eec78d80dca242547be`,
compiled from `sim/scene_physics.xml`, shipped here as `harness/scene_physics.xml`;
`results/chase_controls-results.json` → `plantDigest`). Nothing in this package has been run on
a physical Microduck.

## What the challenge is

The Pollen Robotics Microduck is a bipedal duck about 250 mm tall (Pollen's published spec, not a
measurement of this plant) with 14 position-controlled servos capped at ±0.6405 N·m
(`harness/scene_physics.xml` line 46). There is a ball in the plant: a 100 mm-diameter, 30 g
sphere on a free joint, `body name="ball"`, `geom name="ball_geom"`, radius 0.05, mass 0.03,
`condim 6` (`harness/scene_physics.xml` lines 196–203).

**The challenge is to get the duck to reach that ball and move it.** Not to kick it — Pollen
already ships policies that kick — but to *reach* it, from 450 to 1200 mm away, at bearings the
duck cannot walk straight to. An entrant is scored on fourteen cells of a bearing × range grid
and the verdict is three facts a person can watch happen.

There are two kinds of entrant, both scoreable from day one:

- an **authored move** — keyframes of 14 joint targets with times, plus a blend, replayed on the
  standing policy exactly as the stairs challenge replays one; and
- a **policy under a command schedule** — a named `.onnx` and a list of
  `[atSeconds, {vx, vy, vyaw}]`.

The policy format is the one that makes "chase" a closed-loop question later: the same field
carries a fixed schedule today and a schedule computed from the ball's bearing tomorrow, with no
change to the entrant format, the hash, the app or this package.

The control tick is 20 ms (50 Hz). Every episode opens with a 25-tick settle under the standing
policy, then runs the entrant for its declared seconds, then a 50-tick tail in which **the
standing policy holds the duck under a neutral command** — the standing test, exactly as the
settle did, and not one more second of the entrant's own schedule
(`harness/chase_score.mjs` lines 61, 72, and the tail itself at 752–764).

**The finding this package exists to publish: nothing bundled can chase a ball.** Pollen's two
ball-kick policies score 0 of 14 cells. A duck commanded to walk straight ahead scores 4 of 9
core cells — and *not the three it was predicted to*. The section *What the four controls
establish* is the result.

## The criterion

The scorer is `sim/chase_score.mjs`, published here as `harness/chase_score.mjs`. It is the one
shared module: `chase/chase_rig.mjs`, `chase/chase_robust.mjs` and the bench's `POST /chase`
all call it, and `chase/chase_parity.mjs` holds them equal at full float digits.

The verdict, verbatim, `harness/chase_score.mjs` lines 425–430:

```js
export function verdict(facts) {
  const chased = facts.touched
              && facts.ballTravel_mm >= TRAVEL_MIN_MM
              && facts.upright;
  return { chased, stable: chased && facts.uprightTailTicks >= UPRIGHT_TAIL_MIN };
}
```

and the sentence it is, exported as `CRITERION_SENTENCE` at `harness/chase_score.mjs` line 85 and
answered by every `GET /chase/grid` and every `POST /chase`
(`results/chase_controls-results.json` → `criterion`):

> **chased: the duck touched the ball — any duck geometry within 3 mm of it at any tick — and the
> ball finished at least 100 mm further along the duck's initial heading than it started, and the
> duck was still upright at the end of the episode. stable: chased, and upright for at least 45 of
> the 50 tail ticks.**

The four constants live only in that file:

| constant | value | line |
|---|---|---|
| `TOUCH_MM` | 3.0 | `harness/chase_score.mjs` 57 |
| `TRAVEL_MIN_MM` | 100.0 | 59 |
| `TAIL_TICKS` | 50 | 61 |
| `UPRIGHT_TAIL_MIN` | 45 | 69 — **imported from `climb_score.mjs` line 89, not retyped**, so the 45-of-50 bar cannot drift between the two challenges |

Upright is the stairs rail's own test: `projectedGravity(quat)[2] < −0.90`
(`UPRIGHT_GZ`, `harness/chase_score.mjs` line 76).

### The eight plain facts each cell answers

| fact | what it is |
|---|---|
| `ballTravel_mm` | the ball's net displacement **projected onto the duck's INITIAL heading**, world frame, heading frozen at the first driven tick. **Signed** — a ball pushed backwards scores negative. |
| `ballNet_mm` | unsigned ‖end − start‖ in the plane. Travel and net differ exactly when the ball went sideways, which is the case worth seeing. |
| `closest_mm` | min over ticks of the smallest `mj_geomDistance` between any duck geom and `ball_geom`. **Negative means interpenetration.** |
| `final_mm` | duck root to ball centre in the plane at the last tick |
| `touched` | `closest_mm <= 3.0` |
| `ballPeakSpeed_mps` | peak ball speed over the episode |
| `upright` | at the last tick |
| `uprightTailTicks` | of the final 50 — the tail the standing policy holds under a neutral command |

Contact is `mj_geomDistance`, never `data.contact.get(i)` — that leaks the WASM heap to 2 GB in
about 20 s even with `.delete()`, per `climb/rig2.mjs`.

### Why each clause, and what it defends against

- **`touched`** rules out a duck that walks past and knocks the ball with a draught of nothing,
  and rules out a ball that moved because the episode started with it penetrating something. If
  the duck never made contact, the ball's motion is not the duck's doing.
- **`ballTravel_mm >= 100`, signed and along the INITIAL heading**, rules out three cheats at
  once: a ball nudged 5 mm and called a kick; a ball driven backwards (negative, so it fails);
  and a duck that turns to face wherever the ball happens to have rolled and calls that
  "forward". It is the same defence, for the same reason, as Pollen's own `kick_dir` freeze
  (`mdp.py` 5700–5702: *"Frozen for the episode so the policy can't redefine 'forward' by turning
  after the kick."*). One vector, read twice: the frozen heading is both `kick_dir` and the axis
  `ballTravel_mm` projects onto.
- **`upright` at the end** rules out the move that would otherwise dominate the leaderboard:
  *fall on the ball*. A toppling duck moves a 30 g sphere a long way.
- **`stable`** separates "the ball moved" from "the duck is still a robot afterwards". The 50 tail
  ticks are the standing policy under a neutral command — the standing test — so `stable` measures
  standing and not one more second of chasing. 45 of 50 is deliberately the stairs challenge's
  bar, so a person who has read one challenge already knows what this one means.

### What the criterion deliberately is not

Not "the ball ended near a goal" — there is no goal in `scene.mjb` and adding one would change the
canon plant. Not "peak ball speed above X" — peak speed is dominated by this ball being twice the
mass of Pollen's, and would be un-comparable to anything Pollen published. Not a threshold on the
shaped sum of the nine reward terms — that sum has never been calibrated on this plant and a bar
on it would be a number nobody could defend. It is three facts a person can watch happen.

**The nine weighted reward terms are REPORTED, not the verdict.** See *Pollen's reward* below.

## The grid

`harness/chase_score.mjs` `gridCells()` at line 121, and the same fourteen cells recorded in
`results/chase_controls-results.json` → `grid`. Positive bearing is **LEFT**, the convention
`POST /ball`, duckvision and the robot all use. Range is metres from duck root to ball centre,
measured after the settle. `drop` is spawn height and `fmul` multiplies foot friction — the same
two knobs `climb_score.mjs` uses, so `fmul` 1.0 is the identity.

**Nine core cells**, nominal plant (`drop` 0.120, `fmul` ×1.0) — bearing {−20, 0, +20}° ×
range {0.45, 0.70, 0.95} m. **The centre cell** is bearing 0, range 0.70.

**Five extended cells:**

| # | cell | what it is |
|---|---|---|
| ext 1 | bearing 0, range 0.70, drop 0.130 / ×0.7 | the centre cell on a slippery, higher-spawning plant |
| ext 2 | bearing 0, range 0.70, drop 0.125 / ×1.3 | the centre cell on a grippy plant |
| ext 3 | bearing −40, range 0.70, nominal | a ball well off the heading, to the duck's right |
| ext 4 | bearing +40, range 0.70, nominal | a ball well off the heading, to the duck's left |
| ext 5 | bearing 0, range 1.20, nominal | straight ahead but far: a walk, not a lunge |

The two extended plant pairs `(0.130, ×0.7)` and `(0.125, ×1.3)` are lifted verbatim from
`harness/climb_score.mjs`'s `PLANTS[1]` and `PLANTS[2]` (line 75), so "the slippery plant" means
the same thing in both challenges.

`GET /chase/grid` answers this list and the criterion in this order — core first, so a partial run
is still the core grid — each cell tagged `tier: 'core' | 'ext'`. The kit pins a fallback copy of
the grid and checks it against what the bench publishes; `results/chase_parity.log` records that
check passing (*"the grid the bench publishes is the grid the scorer runs: true"*).

**Why these axes.** Bearing is the axis that makes this a chase. A ball dead ahead can be reached
by a policy that only walks forward; a ball at ±20° cannot, and at ±40° certainly cannot. The grid
is built so the bundled entrants pass some cells and fail others *by construction* — the
off-bearing cells are what the challenge is actually about, and a leaderboard that only ever ran
bearing 0 would look solved while nothing could chase anything.

**Why these ranges.** Pollen's kick task spawns the ball 90 mm in front of the toe
(`microduck_ball_kick_env_cfg.py` line 84, `BALL_OFFSET_X = 0.09`), so the **nearest cell here is
five times the distance the bundled kick policies were trained at.** That gap is the finding this
challenge exists to expose, not a mistake in the grid.

### Aggregates

`kChased` and `kStable` of the 9 core cells; `kExt` of the 5 extended. A leaderboard row also
carries `centreBallTravel_mm` — the signed travel at the centre cell — because it is the one cell
every entrant runs and the one a reader can picture.

**There is no bar yet.** The stairs challenge has one (7 of 9) because six rounds of search
established what a good result looks like. This challenge publishes its first four rows and the
best of them is 4 of 9, off a control that was not trying. Setting a bar off one open-loop walker
would be inventing a number.

## Leaderboard

The four bundled controls, scored over all fourteen cells by `chase/chase_robust.mjs`. Rows are
from `results/chase_controls-results.json` → `leaderboard`. `sha256` is the entrant hash — a
digest of the normalised entrant, not of the file's bytes (see *How to submit*). `centre travel`
is `ballTravel_mm` at the centre cell, bearing 0 / range 0.70 / nominal plant.

| # | sha256 | entrant | kind | seconds | chased / 9 core | stable / 9 core | ext / 5 | touched / 14 | centre travel | scored |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `a0bbbbb98acb` | `entrants/ctrl_alpha_walking.json` | policy `alpha_walking.onnx` | 4 | **4** / 9 | **4** / 9 | 1 / 5 | 5 | 0.0 mm | 2026-09-02 |
| — | `bc77453e40c6` | `entrants/ctrl_do_nothing.json` | move | 5 | 0 / 9 | 0 / 9 | 0 / 5 | 0 | 0.0 mm | 2026-09-02 |
| — | `7e44b5a781fc` | `entrants/ctrl_ball_kick_left.json` | policy `ball_kick_left.onnx` | 5 | 0 / 9 | 0 / 9 | 0 / 5 | 0 | 0.0 mm | 2026-09-02 |
| — | `f8d4e8bfd2b7` | `entrants/ctrl_ball_kick_right.json` | policy `ball_kick_right.onnx` | 5 | 0 / 9 | 0 / 9 | 0 / 5 | 0 | 0.0 mm | 2026-09-02 |

**All four are reference controls, not entries.** They are ranked only so that a reader can see
which is furthest along. Every one of them is bundled in the app, and none of them was authored
to chase anything.

Full sha256s (`results/chase_controls-results.json` → `leaderboard[].sha256`):

```
ctrl_alpha_walking    a0bbbbb98acb7fc5bc1d035527c2c7b153df1c3555db79b9c12e4f446d49d6a5
ctrl_do_nothing       bc77453e40c677db4073a350da5a43d645676d77e1252f51bbf6544be54ca187
ctrl_ball_kick_left   7e44b5a781fc6763042a43065598424ea945f3bc8956bd0f1127aca4ec81b6e9
ctrl_ball_kick_right  f8d4e8bfd2b789668cdf58e7683100d04cf48af2d1fe746d495fc4f697e03ffe
```

**The centre-cell travel column is 0.0 mm for all four rows.** Nothing bundled moves the ball from
the centre cell. That is stated plainly rather than hidden: the one cell every entrant runs is a
cell nothing has ever solved.

### Every cell, for the one control that scores

`ctrl_alpha_walking`, all fourteen cells, from `results/chase_controls-results.json` →
`entrants[3].verdicts`. Per-cell figures in that file are rounded to 2 dp (4 dp for speed); the
unrounded values are in the same entrant's aggregate fields.

| bearing | range | plant | chased | touched | travel mm | net mm | closest mm | final mm | peak m/s |
|---|---|---|---|---|---|---|---|---|---|
| −20° | 0.45 | nominal | **yes** | yes | 582.80 | 650.95 | −3.14 | 216.34 | 0.6140 |
| 0° | 0.45 | nominal | **yes** | yes | 641.27 | 743.77 | −2.14 | 680.51 | 0.6306 |
| +20° | 0.45 | nominal | no | no | 0.00 | 0.00 | 110.69 | 860.02 | 0.0000 |
| −20° | 0.70 | nominal | **yes** | yes | 135.37 | 495.53 | −3.55 | 500.17 | 0.5372 |
| 0° | 0.70 | nominal | no | no | 0.00 | 0.00 | **24.38** | 544.25 | 0.0000 |
| +20° | 0.70 | nominal | no | no | 0.00 | 0.00 | 242.45 | 737.43 | 0.0000 |
| −20° | 0.95 | nominal | **yes** | yes | 233.05 | 406.23 | −5.06 | 316.71 | 0.5950 |
| 0° | 0.95 | nominal | no | no | 0.00 | 0.00 | 113.87 | 370.08 | 0.0000 |
| +20° | 0.95 | nominal | no | no | 0.00 | 0.00 | 396.76 | 687.73 | 0.0000 |
| 0° | 0.70 | ext, drop 0.130 / ×0.7 | no | no | 0.00 | 0.00 | 64.93 | 331.77 | 0.0000 |
| 0° | 0.70 | ext, drop 0.125 / ×1.3 | **yes** | yes | 465.39 | 559.72 | −4.39 | 279.20 | 0.6198 |
| −40° | 0.70 | ext, nominal | no | no | 0.00 | 0.00 | 210.87 | 621.99 | 0.0000 |
| +40° | 0.70 | ext, nominal | no | no | 0.00 | 0.00 | 424.57 | 977.05 | 0.0000 |
| 0° | 1.20 | ext, nominal | no | no | 0.00 | 0.00 | 196.80 | 320.47 | 0.0000 |

Every one of its five `chased` cells is also `stable`, and it is upright at the end of all
fourteen with 50 of 50 tail ticks in every cell
(`entrants[3]`: `kStableAll` 5, `uprightFinalCells` 14, `minUprightTailTicks` 50).

The `−20° / 0.70 m` row is why `ballTravel_mm` and `ballNet_mm` are two facts and not one: the
ball moved **495.53 mm** in total but only **135.37 mm** of it along the duck's initial heading.
It went sideways, and the criterion counts only the part that went forward.

## What the four controls establish

Each control's expected behaviour was declared **in advance**, in the entrant file's own `note`
and in `chase/chase_parity.mjs`, so that a run which disagrees is a finding to chase down rather
than a number to write down. `results/chase_parity.log` prints the predictions beside the
measurements.

### 1. `ctrl_do_nothing` — 0 of 14, and it must be

A move that holds HOME for five seconds. **Predicted 0 of 14. Measured 0 of 14.**
Touched nothing in any cell; `ballTravel_mm` 0.0 everywhere; `ballPeakSpeed_mps` 0 everywhere;
closest approach ranged from **344.02 mm** (the nearest cell) to **1073.80 mm** (the 1200 mm
cell), mean **620.56 mm**; upright with 50 of 50 tail ticks in all fourteen
(`entrants[0]`: `touchedCells` 0, `maxBallTravel_mm` 0, `maxBallPeakSpeed_mps` 0,
`minClosest_mm` 344.0166288764408, `meanClosest_mm` 620.5576745621695,
`uprightFinalCells` 14, `minUprightTailTicks` 50).

**A criterion this row passes is not a chasing test.** It is the first thing
`chase/chase_parity.mjs` checks, before any other comparison is believed
(`results/chase_parity.log`: *"ctrl_do_nothing scores 0 of 14 and touches nothing: true"*).

### 2 & 3. Pollen's ball-kick policies — 0 of 14, and that is the measurement

`ball_kick_left.onnx` and `ball_kick_right.onnx` at **the config's own command**: schedule
`[[0, {vx: 0, vy: 0, vyaw: 0}]]`, `seconds` 5.

The schedule was read out of the config, not chosen by taste. The ball-kick env keeps a twist
command slot only for observation-shape parity with the unified 61-D actor layout, and its ranges
are `lin_vel_x (−0.01, 0.01)`, `lin_vel_y (−0.01, 0.01)`, `ang_vel_z (−0.05, 0.05)`, with
`rel_standing_envs 0.0`, `heading_command False`, and a resampling period equal to the whole
episode (`microduck_ball_kick_env_cfg.py` 400–408; the block's own comment: *"Command: tiny noise
around zero (obs-shape parity only)"*). Zero is the centre of all three ranges, so
`(0, 0, 0)` held for the episode is the centre of the distribution these two policies were
actually trained under. Commanding them to walk would be commanding them 50× outside their
`lin_vel_x` range and calling the result a measurement of Pollen's policy. `seconds` 5 is
`EPISODE_LENGTH_S` (cfg 74), the episode length they were trained at.

**Predicted 0 of 14, or very close. Measured 0 of 14 — both of them.** Neither touched a ball in
any cell; `ballTravel_mm` 0.0 everywhere; peak ball speed 0 everywhere; both upright with 50 of 50
tail ticks in all fourteen. Closest approach: left `301.21` mm minimum, `594.05` mm mean; right
`308.75` mm minimum, `606.35` mm mean (`entrants[1]`, `entrants[2]`).

**This is the measurement that states the problem.** These policies are trained to stand still and
swing one leg at a ball 90 mm in front of the toe, and they are *blind to the ball by design*
(cfg 11–15: *"The policy is BLIND to the ball (no ball obs in the actor)… the operator aims the
robot at the ball"*). At the nearest cell the ball is 450 mm away and the policy cannot see it,
cannot walk to it, and is not commanded to. **Reaching the ball is the unsolved half, not kicking
it.**

`KICK_FOOT` (cfg 41) is a module-level flag, not a per-policy field: the two ONNX files came from
the same file with the flag flipped. Nothing in the reward differs between the runs — only the
ball spawn side and which foot the (refused) `support_foot_grounded` sensor watches — so both are
correctly scored under one term table.

### 4. `ctrl_alpha_walking` — 4 of 9, and **not the four that were predicted**

`alpha_walking.onnx` commanded straight ahead at `vx` 0.5 m/s for 4 s.

The prediction, written before the run: *"passes some of the three bearing-0 cells, fails every
off-bearing cell… roughly 2–4 of 9 core cells, with ext 5 plausibly passing while ext 3 and ext 4
certainly fail."*

The count landed inside the prediction — **4 of 9** — but **the shape is the opposite of what was
predicted**. It passes the entire **bearing −20° column** (0.45, 0.70 and 0.95 m) plus bearing 0
at 0.45 m, and it **fails bearing 0 at 0.70 m and 0.95 m**. It also fails ext 5 (1.20 m dead
ahead), which was predicted to pass plausibly, and passes ext 2 (the grippy plant) instead.

The cause is in the data, not in the scorer. `results/chase_drift-results.json` measures it: run
open-loop at `vx` 0.5 for 4 s, the gait **drifts to the duck's right by 15.893°** away from the
heading frozen at the first driven tick, and walks **1.3145 m** rather than the commanded 2.0 m
(the three cells where it never touches the ball are identical to the last digit — a walk with
nothing in its way is deterministic). On the 0.45 m cell it reaches the ball, the collision
deflects it, and the drift reads **17.831°** over a **1.2896 m** path.

So the −20° column sits about 4° off the actual walk line and the dead-ahead column sits about
16° off it. At 0.70 m dead ahead the duck misses the ball by **24.38 mm** — a gap narrower than
half the ball's radius — and at 0.95 m by **113.87 mm**.

> **Open-loop forward walking does not solve "the ball is straight ahead". It solves "the ball
> happens to be where this gait drifts."**

That makes the challenge's point harder, not weaker. The line someone is being asked to cross is
not "walk further" — it is *steer*: a person editing a keyframe to make the duck turn, or later a
policy that reads the ball and commands its own `vyaw`. **Every genuinely off-bearing cell — both
±40° cells, all three +20° cells — is unclaimed by anything bundled.**

`alpha_walking.onnx` is the **velocity** config's policy, not the ball-kick config's
(`RunMetrics.Task.forPolicy` maps it to `.velocity`). It is here as a *chaser*, judged by the
criterion; the nine reward terms reported for it are the ball-kick config's terms evaluated on a
policy trained under `microduck_velocity_env_cfg.py`, and that caveat travels with every one of
its term values.

### What the four establish between them

A `ctrl_do_nothing` pass would prove the criterion is broken; its fail proves the criterion
requires the duck to do something. A ball-kick pass would prove Pollen's kick generalises past
90 mm; its fail proves reaching the ball is the unsolved half. An `alpha_walking` pass proves
walking is enough when the ball is where the gait goes; its dead-ahead and off-bearing failures
prove that steering — closing the loop on where the ball actually is — is what those cells need.

## Pollen's reward: reported, never the verdict

The reward transcribed here is `microduck_ball_kick_env_cfg.py` from
`pollen-robotics/microduck_rl`, branch `main`, commit
`1e79c29c97d8b38aee9eefde77a545860ba7658e`, 661 lines, Apache-2.0. The config does not contain
all of its own reward: it calls `make_velocity_env_cfg()` (line 185), deletes eight terms
(cfg 210–221) and adds seven (cfg 238–310), so five of the twelve survivors are defined upstream
in **mjlab v1.3.0** — the version that commit's `pyproject.toml` pins, so every mjlab line number
cited is a v1.3.0 line number. The full transcription, with every source file and trap, is
`REWARD.md` in this package.

**Twelve live terms. Nine computable on this plant, three refused by name.** The terms are
reported because they are the reward the bundled policies were trained on and a person editing a
keyframe deserves to see them move. They are **not** the verdict: a shaped sum of nine weighted
terms is not a thing a person can hold in their head, and a leaderboard sorted on it would reward
a duck that stands beautifully still.

### The nine computed

`harness/chase_score.mjs` `TERMS`, line 214. Values below are per-tick means **over the driven
span only** (the standing tail is the bench's own test, not the entrant's episode), averaged
across all fourteen cells, from `results/chase_controls-results.json` → `entrants[].terms`.

| term | weight | do-nothing | kick left | kick right | alpha walking |
|---|---|---|---|---|---|
| `ball_forward_velocity` | +12.0 | 0 | 0 | 0 | 0.029988343652321836 |
| `ball_speed_overshoot` | −4.0 | 0 | 0 | 0 | 0 |
| `upright` | +2.0 | 0.9998763057782069 | 0.9931221510786095 | 0.9943087512936136 | 0.9564203498605064 |
| `pose_stand_legs` | +2.0 | 0.9965143306457938 | 0.9623872375324086 | 0.9724136072212477 | 0.9038219162946854 |
| `pose_stand_neck` | +1.0 | 0.9407215931900168 | 0.9887772233518801 | 0.9815705391549662 | 0.8457114526149494 |
| `height_stand` | +1.0 | 0.999052246235542 | 0.9989018867774248 | 0.9983744303308056 | 0.9791737927694291 |
| `body_ang_vel` | −0.05 | 1.2002301406106929e-05 | 0.035387799283043656 | 0.0515341759225177 | 0.8565316158987761 |
| `action_rate_l2` | −1.0 | 0 | 0.05309782345138118 | 0.046590201825831304 | 0.2522514765950025 |
| `angular_momentum` | −0.02 | 8.53530467165277e-10 | 1.9334917086628585e-06 | 2.3455729716410938e-06 | 2.74005989743522e-05 |

Sources, weights and formulas are carried in the file itself and answered by `GET /chase/grid`:

- `ball_forward_velocity` — cfg 238–242, `mdp.py` 5761–5784, cfg 95.
  `clamp(ball world linear velocity xy · kick_dir, 0.0, 1.0)`. **One-sided**: backward and lateral
  ball motion earn 0, not a penalty.
- `ball_speed_overshoot` — cfg 243–247, `mdp.py` 5787–5806. `clamp(fwd − 1.0, 0.0, 5.0)` on the
  *unclamped* projection.
- `upright` — cfg 285–287, mjlab `velocity/mdp/rewards.py` 67–110, `velocity_env_cfg.py` 286–293.
  `exp(−‖projected gravity xy‖² / 0.05)` on `trunk_base`.
- `pose_stand_legs` — cfg 262–270 and cfg 100, `mdp.py` 2396–2418. Mean over the ten leg joints of
  `exp(−((q − HOME)/0.5)²)`. **Not** the `pose` term `/tune` reports: that is mjlab's
  `variable_posture`, which this config *deletes* (cfg 217).
- `pose_stand_neck` — cfg 274–282 and cfg 101. The same over the four neck/head joints, std 0.3.
- `height_stand` — cfg 290–298, cfg 98, `mdp.py` 2440–2451. `exp(−((z − 0.115)/0.04)²)`.
- `body_ang_vel` — cfg 302–303, mjlab `velocity/mdp/rewards.py` 184–193. `ωx² + ωy²` of
  `trunk_base` in the world frame; z is deliberately unpenalised.
- `action_rate_l2` — cfg 301 (stage-0 −0.1), cfg 561–574 (curriculum to −1.0 by iteration 1500),
  mjlab `envs/mdp/rewards.py` 58–65. Scored at the **ramp end, −1.0**, because the trained policy
  lived under the final value — the same choice `RunMetrics.Task.actionRateWeight` already makes
  for `.ballKick`. `weightStage0: −0.1` is published beside it so nobody has to guess.
- `angular_momentum` — cfg 304, mjlab `velocity/mdp/rewards.py` 196–206, `velocity_env_cfg.py`
  312–315 (`sensor_name "robot/root_angmom"`). `Σ(angmom²)`. `chase_score.mjs` asserts the sensor
  is type 37 (`mjSENS_SUBTREEANGMOM`), objtype body, object `trunk_base`, dim 3 at boot and
  refuses the term by name if any check fails.

`action_rate_l2` means two different things for the two kinds of entrant, so **every row carries
`action_rate_l2_source`**: `"policy raw output"` for a policy (mjlab's `action_manager.action`,
the network's raw 14-vector) and `"keyframe pose target"` for a move (the interpolated, clamped
pose target the keyframes emit). Same formula; comparable within a kind, not across the two.
Labelled rather than refused, because a move genuinely has an action
(`harness/chase_score.mjs` line 257).

The fourteen joint slots are **asserted, not assumed**: `assertJointOrder()`
(`harness/chase_score.mjs` line 188) throws at boot unless duckkit's fifteen joint names minus
`mouth` are exactly the fourteen that make Pollen's `_LEG_JOINTS = [0,1,2,3,4,9,10,11,12,13]` and
`_NECK_JOINTS = [5,6,7,8]` mean what the config means by them.

### The three refused

Answered in `refused[]` on every row with the weight and the reason — never dropped
(`harness/chase_score.mjs` line 276; `results/chase_controls-results.json` → `entrants[].refused`;
`results/chase_parity.log`: *"refusals identical (3): true"*).

| term | weight | why this plant cannot answer it |
|---|---|---|
| `support_foot_grounded` | +2.0 | Reads a **contact sensor**: cfg 160–171 builds `support_foot_ground_contact`, a `ContactSensorCfg` with primary geom `left_foot_collision` and secondary body `terrain`, reduced to a `found` flag; `single_foot_grounded_reward` (`mdp.py` 5809–5824) returns `clamp(found, 0, 1)`. This plant has six sensors and none is a contact sensor. `mj_geomDistance` could report how far a foot geom is from the floor, but that is a **distance**, and turning it into the config's binary `found` requires choosing a threshold Pollen never wrote — that would be inventing a reward. |
| `self_collisions` | −1.0 | Reads the `self_collision` sensor (cfg 173–180), a subtree-vs-subtree `ContactSensor` on `trunk_base`; `self_collision_cost` (mjlab `velocity/mdp/rewards.py` 162–181) counts its found slots or thresholds a force history at 10 N. No collision sensor in this plant, and no contact forces at all. |
| `dof_pos_limits` | −1.0 | mjlab `velocity_env_cfg.py` 317, **not** deleted by the kick config. `joint_pos_limits` (mjlab `envs/mdp/rewards.py` 81–96) scores travel outside `soft_joint_pos_limits` — a configured fraction of the model's hard range. Neither duckkit nor this bench ships that fraction; scoring against the hard `rangeLo`/`rangeHi` would be a different term wearing this one's name. |

**A shorter list is not a better one.** Each of these could be approximated by picking a threshold
or a softening fraction Pollen never wrote down, and each approximation would be a different term
wearing this one's name.

### One deliberate departure, stated so the two rails do not look contradictory

`duckbench-core.mjs`'s `TUNE_REFUSALS` refuses `angular_momentum` for the **velocity** config, and
its stated reason is not a missing sensor — it says the plant does carry `root_angmom` and that
the refusal is because nothing there reads its weight out of the config. That was a fact about
`/tune`'s six-term transcription, not about the plant. Here the weight **is** read (cfg 304,
−0.02) and the sensor **is** the right one. So `/chase` computes it and `/tune` still refuses it,
consistently: **each answers exactly the terms it has transcribed a weight for.**

Of the six terms `/tune`'s `rewardSums` computes for the velocity config, only **three** survive
this config: `upright`, `body_ang_vel` and `action_rate_l2`. `track_linear_velocity`,
`track_angular_velocity` and `pose` are deleted by cfg lines 211, 212 and 217 and must not appear
in a `/chase` answer — reporting them would be answering the wrong config under the right name.

### Two config defects, recorded rather than silently fixed

1. **The target-speed comment contradicts the constant.** Line 95 sets `BALL_TARGET_SPEED = 1.0`;
   the comment block at 224–237 describes the peak as *"BALL_TARGET_SPEED (0.25 m/s — a gentle
   tap)"* and justifies the weight as *"Weight 12.0 = 3.0/target"* — and 3.0/0.25 = 12.0 while
   3.0/1.0 = 3.0. The comment was written for a target of 0.25 and the constant was later moved to
   1.0 without the rescale the comment itself demands. **The code is what ran, so the code is
   transcribed:** `max_speed` 1.0, `target_speed` 1.0, weights +12.0 and −4.0. Recorded so nobody
   "fixes" it into a third set of numbers.
2. **`KICK_FOOT` is a module-level flag, not a per-policy field.** See control 3 above.

## How to reproduce

From a fresh clone. The runnable harness is the GitHub repository, not this package — the sim
needs `mujoco` (wasm), `onnxruntime-node`, the compiled plant `sim/scene.mjb` and the policy
`.onnx` files, all of which live there.

Verified on Node 24.16.0 / npm 11.13.0 on linux-arm64 (a Raspberry Pi 5); Node 20 or later is
expected to work. `npm ci` runs onnxruntime-node's postinstall, which downloads a native binary,
so it needs network beyond the registry and a platform onnxruntime ships a build for. If the tag
is missing on your mirror, `main` at or after 2026-09-02 carries the same harness.

```bash
git clone --depth 1 --branch ball-challenge-v1 https://github.com/craigm26/duck-sounds.git
cd duck-sounds

cd sim
npm ci            # mujoco, onnxruntime-node
# The scorer lives in chase/ and imports bare `mujoco` and `onnxruntime-node`; it finds them
# through a committed symlink, chase/node_modules -> ../sim/node_modules. If your checkout did
# not create symlinks (git config core.symlinks false, or a zip download), make it by hand:
[ -e ../chase/node_modules ] || ln -s ../sim/node_modules ../chase/node_modules

# 1. Score ONE entrant on ONE grid cell, and print the criterion:
node --input-type=module -e '
  import { scoreSaved } from "../chase/chase_rig.mjs";
  import { CRITERION_SENTENCE } from "./chase_score.mjs";
  const r = await scoreSaved("../chase/ctrl_alpha_walking.json",
                             { bearing: -20, range: 0.45, drop: 0.120, fmul: 1.0 });
  console.log("chased", r.chased, "stable", r.stable,
              "touched", r.facts.touched,
              "ballTravel_mm", r.facts.ballTravel_mm,
              "closest_mm", r.facts.closest_mm,
              "uprightTailTicks", r.uprightTailTicks);
  console.log(CRITERION_SENTENCE);
'

# 2. Score the same entrant over the whole fourteen-cell grid:
node --input-type=module -e '
  import { scoreChase } from "../chase/chase_robust.mjs";
  const g = await scoreChase("../chase/ctrl_alpha_walking.json");
  console.log("chased", g.kChased + "/" + g.nCore, "stable", g.kStable + "/" + g.nCore,
              "ext", g.kExt + "/" + g.nExtOnly, "touched", g.touchedCells + "/" + g.nAll,
              "centre travel", g.centreBallTravel_mm, "sha256", g.sha256.slice(0, 12));
'
```

Expected for `ctrl_alpha_walking.json` (`results/chase_controls-results.json` → `entrants[3]`
and `entrants[3].verdicts[0]`):

- snippet 1 — `chased true`, `stable true`, `touched true`,
  `ballTravel_mm 582.7970533588832`, `closest_mm -3.140311715867726`, `uprightTailTicks 50`;
- snippet 2 — `chased 4/9`, `stable 4/9`, `ext 1/5`, `touched 5/14`, `centre travel 0`,
  `sha256 a0bbbbb98acb`.

Pass `{ core: true }` to `scoreChase` for the 9 core cells only.

**The acceptance test.** `chase/chase_parity.mjs` scores all four entrants on all fourteen cells
**twice** — once through the bench's in-process `POST /chase` handle, which is exactly what the
phone's WebView bridge does, and once through `chase_robust.scoreChase` — and compares every
numeric field with `Object.is` at full float digits:

```bash
cd sim && node ../chase/chase_parity.mjs
```

The run recorded in `results/chase_parity.log`: **56 per-cell rows, EXACT on all 49 compared
fields plus the cell itself, 56/56**; aggregates recomputed from the `/chase` answers alone equal
`chase_robust` on every entrant; `ctrl_do_nothing` scores 0 of 14 and touches nothing; the grid
the bench publishes is the grid the scorer runs; the criterion string and all three refusals
identical on both sides. **55 s** for the whole gate — four entrants × 28 scored cells, about
0.5 s per cell on a Pi 5.

**The drift measurement** behind the headline finding:

```bash
cd sim && node ../challenge-ball/harness/measure_drift.mjs
```

writes `results/chase_drift-results.json` and prints the drift, walked distance and closest
approach for the four bearing-0 cells.

## Reproduce in the app

From Microduck Studio **build 46** or later (TestFlight: <https://testflight.apple.com/join/S36AnsKr>;
source: github.com/craigm26/duck-studio):

1. **Studio → Measure → Challenges → Ball** (or **Behaviours → the discover section →
   Challenges → Ball**). The challenge screen is now a list of two challenges, Stairs and Ball, at
   the same place the Stairs Challenge used to be — the stairs are exactly where they were, one
   row further in. The four controls above are bundled as **Reference controls**, byte for byte
   with `entrants/`.
2. Pick a bench. **This iPhone** is always there — the phone's own bench carries the same
   `chase_score.mjs`, `reward_math.mjs` and plant as the harness — or a Pi bench running the
   current duck-sounds. The screen asks the bench for its grid first; a bench without `/chase`
   says so, in its own words, with **no button beside it**.
3. Open a row and tap **Score on `<bench>`**. The app sends the fourteen cells one request
   each, drawing progress cell by cell (`Cell 5 of 14 — 0°/0.70/.120/x1.0`), and prints
   `k of 9 chased`, `k of 9 stable`, the extended count and whether that matches, beats or misses
   the published row.
4. **Open in the editor** turns a *move* entrant into a Studio motion. Change any keyframe's servo
   values there, come back, and tap **Score your edited version**. The screen says whether the
   edit scored better, the same or worse than what it started from. Keep what helps, put back what
   does not. There is no reward model in that loop: you are the judge and the bench is the
   measurement.
   **The three policy entrants have no keyframes to open.** They can be scored and played but not
   edited, and the app says so rather than showing a dead button.
5. **Submit** is gated on a score being publishable *and* on the bench having run the published
   grid — a partial run or a different grid is not submittable, and the screen says which. When it
   is live it writes one file (the entrant, all fourteen per-cell answers unrounded, the plant
   digest, the date, and how to re-score it), opens a pre-filled GitHub issue titled
   `Ball challenge: <entrant>`, and can commit the file to a dataset under your own Hugging Face
   account as an archive.

Scoring on the phone needs no account, no secret and no Pi. On a Pi 5 bench a cell answers in
about 0.5 s (`results/chase_parity.log`), so a fourteen-cell grid is roughly seven seconds of
progress rows.

## On a real Microduck

The app can play a challenge entrant on a bench, in physics. **Playing one on a physical Microduck
is not wired in build 46**, there is no score off a real robot, and nothing in this package has
been run on hardware.

If you put an entrant on a robot, do it with the same care as any untested motion. Two specific
warnings from these results:

- The one control that scores does so by **walking into the ball**: its `closest_mm` at every
  passing cell is *negative* (−2.14 to −5.06 mm), i.e. the duck and the ball interpenetrate in
  MuJoCo's soft contact. A real ball does not yield, and a real duck walking into one at
  0.5 m/s can be knocked over.
- The `chased` criterion has **no floor on how the ball is moved**. A move that shoves the ball
  with the duck's head, or that succeeds by falling forward and getting up, would satisfy the
  facts. The `upright`-at-the-end clause rules out ending on the floor; it does not rule out
  getting there.

Report what happened in the GitHub issue. A real ball on a real floor is your measurement, not
the harness's.

## How to submit

Open an issue on <https://github.com/craigm26/duck-sounds> titled **`Ball challenge: <name>`**
with the entrant JSON attached (attached as a file, not pasted).

The `sha256` on the leaderboard is the **entrant hash** — a digest of the normalised entrant, not
of the file's bytes, so `sha256sum` on a file will not match it. It is defined by
`entrantHashPayload` (`harness/chase_score.mjs` line 401) and computed like this:

```bash
cd sim && node --input-type=module -e '
  import { entrantHash } from "../chase/chase_rig.mjs"; import fs from "node:fs";
  console.log(entrantHash(JSON.parse(fs.readFileSync(process.argv[1], "utf8"))));
' ../chase/ctrl_alpha_walking.json
# a0bbbbb98acb7fc5bc1d035527c2c7b153df1c3555db79b9c12e4f446d49d6a5
```

**Every key is hashed except `name` and `note`.** Unknown keys are *preserved and hashed* rather
than stripped, so an entrant file that also carries stairs fields is a **different** entrant and
not a silently equivalent one; `name` and `note` are excluded because renaming a move or rewording
its note is not a different move, and the app's edit-score-keep loop would otherwise report a new
entrant every time somebody fixed a typo. Objects are serialised with their keys sorted at every
depth, so a file whose keys were written in a different order hashes to the same value.

An audit re-scores the attached file and checks, in this order:

1. **Shape.** `checkEntrant` (`harness/chase_score.mjs` line 333): `kind` is `"move"` or
   `"policy"`; `seconds`, when present, is `0 < seconds <= 30`; a move has at least one keyframe
   and every `pose` is exactly 14 numbers with a finite `t`; a policy names a policy; the schedule
   is a list of `[atSeconds, {vx, vy, vyaw}]`. An entrant that names a policy this bench has never
   heard of is a **valid entrant and an unknown policy** — the two failures read differently to
   whoever sent it.
2. **All fourteen cells**, reporting `kChased`, `kStable`, `kExt` and the eight facts per cell.
3. **The published grid.** The cells the entrant was scored on must be the cells
   `GET /chase/grid` publishes.
4. **The plant.** Every row carries `plantName` and `plantDigest`; a row from a different plant is
   a different measurement.

## The entrant formats

Two kinds. Both are carried verbatim through the app's `HarnessJSON`, so an entrant file
round-trips byte for byte and hashes to one value.

### A move

```json
{
  "name": "ctrl_do_nothing",
  "kind": "move",
  "seconds": 5,
  "intent": {
    "name": "ctrl_do_nothing",
    "keyframes": [
      { "t": 1.0, "pose": [0, -0.0873, -0.4579, -0.0049, 0.453, 0.3491, 0.3491,
                           0, 0, 0, 0.0873, 0.4579, 0.0049, -0.453] },
      { "t": 4.9, "pose": [0, -0.0873, -0.4579, -0.0049, 0.453, 0.3491, 0.3491,
                           0, 0, 0, 0.0873, 0.4579, 0.0049, -0.453] }
    ],
    "blend": 1
  },
  "note": "…"
}
```

The episode reads `keyframes` and `blend`; anything else inside `intent` is carried through
untouched **and hashed**. `pose` is fourteen numbers; `t` is seconds from the first driven tick.
The move rides on the standing policy under the bench's 25-tick settle, exactly as a stairs cell
does. The pose above is duckkit's HOME with the mouth dropped.

### A policy

```json
{
  "name": "ctrl_alpha_walking",
  "kind": "policy",
  "seconds": 4,
  "policy": "alpha_walking.onnx",
  "schedule": [[0, { "vx": 0.5, "vy": 0, "vyaw": 0 }]],
  "note": "…"
}
```

`schedule` is a list of `[atSeconds, {vx, vy, vyaw}]`; **the last entry that has begun wins**,
which is `duckbench-core.mjs`'s existing `commandAt` contract
(`harness/chase_score.mjs` line 378). A missing `seconds` defaults to 5.

### The fourteen joint slots

In order — the Microduck's joints minus `mouth` (`harness/duckkit-constants.json`,
`harness/chase_score.mjs` `JOINT_ORDER` line 181):

`left_hip_yaw`, `left_hip_roll`, `left_hip_pitch`, `left_knee`, `left_ankle`, `neck_pitch`,
`head_pitch`, `head_yaw`, `head_roll`, `right_hip_yaw`, `right_hip_roll`, `right_hip_pitch`,
`right_knee`, `right_ankle`.

## Caveats

- **Pollen's ball is not this ball.** The ball-kick config trains against a 70 mm-diameter, 15 g
  ball (cfg 76–77, `BALL_RADIUS 0.035`). This plant's ball is 100 mm across and 30 g — **1.43× the
  radius and 2× the mass** (`harness/scene_physics.xml` lines 196–203). Every term is computed
  with the same formula and the same weight, but a speed target in m/s was tuned against a ball
  with half the inertia, so the two ball terms here are the config's function evaluated on a
  **different ball**, not a reproduction of Pollen's training signal. That is why every row carries
  `plantName` and `plantDigest`. Substituting Pollen's ball is not an option: `scene.mjb` is the
  canon plant every recorded duckkit clip claims to come from.
- **The nearest cell is five times Pollen's training distance.** `BALL_OFFSET_X = 0.09` (cfg 84)
  against 0.45 m here. The kick policies' 0-of-14 is a measurement of that gap, not a failure of
  the policies at the task they were trained for.
- **`alpha_walking` is the wrong config's policy.** Its nine term values are the ball-kick
  config's terms evaluated on a policy trained under `microduck_velocity_env_cfg.py`. Its
  *criterion* result is valid — the criterion does not care what a policy was trained on — but its
  term values are not a reading of what it was optimised for.
- **The centre cell is unsolved by everything bundled**, so the leaderboard's centre-travel column
  is all zeros. It is a real column with a real value, not a placeholder.
- **A move entrant's stand terms are not 1.0 even when it does nothing.** A move rides on the
  standing policy, so `ctrl_do_nothing`'s `pose_stand_legs` and `pose_stand_neck` read
  0.9965 and 0.9407 rather than 1.0: the joints are where `alpha_stand` holds them, not exactly at
  HOME. Correct, and stated here because a reader who expects 1.0 will ask.
- **`action_rate_l2` is not comparable across the two kinds of entrant.** See
  `action_rate_l2_source`, above.
- **Per-cell figures in the results file are rounded**, to 2 dp for millimetres and 4 dp for
  speed (`chase/chase_robust.mjs`, `verdicts`). The unrounded values are the aggregate fields on
  the same entrant, and `POST /chase` answers unrounded.
- **Soft contact.** Every passing cell interpenetrates the ball by a few millimetres
  (`closest_mm` −2.14 to −5.06 for the one control that scores). That is MuJoCo's soft contact,
  not a tunnel, but it is a physical softness a real ball does not have.
- **One control, four rows.** This leaderboard is four bundled controls scored once each. It is a
  starting line, not a search: no optimiser has ever been run against this criterion, and nobody
  should read "4 of 9" as a hard result about what the Microduck can do.
- **No hardware.** Nothing here has been run on a Microduck. See the top of this card.

## Files in this package

```
README.md              this card
leaderboard.md         the leaderboard tables alone, for editing
REWARD.md              the full reward transcription: every source file, line reference and trap
MANIFEST.json          sha256 and byte size of every file here
check_numbers.mjs      re-derives every number in this card from results/ and prints PASS/FAIL
hf_upload.sh           the exact Hugging Face upload commands
entrants/              the four bundled control entrants, byte-identical to chase/ in the repo
results/               chase_controls-results.json  the fourteen-cell scores behind every number
                       chase_drift-results.json     the naive chaser's open-loop drift
                       chase_parity.log             the acceptance test's own output
harness/               chase_score.mjs, reward_math.mjs, climb_score.mjs, chase_rig.mjs,
                       chase_robust.mjs, chase_parity.mjs, measure_drift.mjs,
                       scene_physics.xml, duckkit-constants.json + a README.
                       A SNAPSHOT FOR READING. The runnable harness is the GitHub repo.
```

## Provenance

Built on 2026-09-02 in `github.com/craigm26/duck-sounds` under `chase/`, following the shape the
stairs challenge established the day before.

| piece | where | what it is |
|---|---|---|
| the episode and the criterion | `sim/chase_score.mjs` | the one shared module. The bench, the desk rig and the grid runner all call it; nothing is allowed a second opinion on the constants. |
| the reward transcription | `chase/REWARD.md` | 476 lines. Four source files had to be quoted because the config does not contain all of its own reward. |
| the desk rig | `chase/chase_rig.mjs` | one cell, one entrant, its own `mjData` |
| the grid | `chase/chase_robust.mjs` | fourteen cells, the aggregation, the verdict rows |
| the acceptance test | `chase/chase_parity.mjs` | `POST /chase` against `chase_robust`, 56 rows, 49 fields, `Object.is` |
| the bench | `sim/duckbench-core.mjs` | `POST /chase` and `GET /chase/grid`. The ball and every piece of laid-out state are captured before a cell and restored after it, so every existing endpoint answers exactly as before. |
| the kit | duck-studio `StudioKit/Sources/StudioKit/Ball*.swift` | the challenge, the grid with a pinned fallback, the score, the submission, every user-visible sentence with a test |
| the app | duck-studio `DuckStudio/Sources/BallChallengeView.swift` | the screen, reached from `Studio → Measure → Challenges` and the Behaviours discover row |

The bench's five existing gates — `bench_parity`, `policy_parity`, `physics_parity`,
`tune_parity` and `climb_parity` — were all re-run after the ball rail landed and all five still
pass. Their outputs are not shipped here (they belong to the repository and, for the stairs, to
the stairs package); what matters for this card is that **the stairs challenge's numbers are
unchanged by anything in this package.**

`REWARD.md` in this package is 476 lines and is the full transcription record: every source file
quoted, every line reference, the six transcription traps recorded so the code cannot get them
wrong, and the two config defects recorded rather than silently fixed.

## License

The data in this package — every entrant under `entrants/`, every results file under `results/`,
`leaderboard.md`, `REWARD.md`, `MANIFEST.json` and this card — is published under **CC BY 4.0**
(https://creativecommons.org/licenses/by/4.0/). Use it, remix it, redistribute it; credit
"Microduck Ball Challenge, craigm26" and link back here. The harness that scores it — `harness/`
here, and the runnable copy at github.com/craigm26/duck-sounds — is **Apache-2.0**, as are
duck-studio (Microduck Studio) and duckkit. Pollen Robotics' policies (`alpha_*.onnx`,
`ball_kick_*.onnx`), their reward config and their plant come from their repositories under their
own terms, which this package does not grant.
