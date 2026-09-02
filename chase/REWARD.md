# The ball challenge: the reward, the criterion, the grid, the controls

**Written before anything is built.** Nothing in this file has been simulated. Every weight and
every formula below is transcribed from Pollen Robotics' own training config at a pinned commit,
with the line reference beside it; every refusal names the term and gives the reason this plant
cannot answer it. No reward is invented here. Where a number could only be produced by picking a
threshold or a weight that Pollen did not write down, the term is refused rather than approximated.

This is the ball counterpart of `sim/climb_score.mjs` + `challenge/README.md`, and it follows that
rail exactly: one shared episode-and-criterion module, one criterion sentence said the same way on
every bench, one grid every verdict is decided on, and controls whose expected behaviour is
declared in advance so that a surprising result is a finding and not a bug.

---

## 1. Provenance — where the config came from

| | |
|---|---|
| repository | `github.com/pollen-robotics/microduck_rl` (public) |
| file | `src/mjlab_microduck/tasks/microduck_ball_kick_env_cfg.py` |
| branch | `main` |
| commit | `1e79c29c97d8b38aee9eefde77a545860ba7658e` |
| raw URL | `https://raw.githubusercontent.com/pollen-robotics/microduck_rl/1e79c29c97d8b38aee9eefde77a545860ba7658e/src/mjlab_microduck/tasks/microduck_ball_kick_env_cfg.py` |
| length | 661 lines |
| licence | Apache-2.0 |

Three more files are quoted, because the ball-kick config does not contain all of its own reward.
It calls `make_velocity_env_cfg()` (line 185) and then deletes eight terms and adds seven, so five
of the twelve surviving terms are defined upstream:

| | |
|---|---|
| reward functions Pollen wrote | `src/mjlab_microduck/tasks/mdp.py`, same commit, 7188 lines |
| the base reward dict | `mjlab` **v1.3.0**, `src/mjlab/tasks/velocity/velocity_env_cfg.py`, lines 275–373 |
| the base reward functions | `mjlab` v1.3.0, `src/mjlab/tasks/velocity/mdp/rewards.py` and `src/mjlab/envs/mdp/rewards.py` |

The mjlab version is not a guess: `pyproject.toml` at the same microduck_rl commit pins
`"mjlab==1.3.0"`. The v1.3.0 reward dict and the current `main` one differ elsewhere in the file,
so **every mjlab line number below is a v1.3.0 line number** and was read from the v1.3.0 tag.

The identical file also exists on the `develop` and `ball_walk` branches. `main` is quoted because
it is the branch the published `ball_kick_left.onnx` / `ball_kick_right.onnx` correspond to; the
`ball_walk` branch adds a *different* task (`microduck_ball_walk_env_cfg.py`, a bigger ball) which
is not what the bundled policies were trained on and is not transcribed here.

### The plant this reward is being asked of

`sim/scene.mjb`, sha256 `3f8c9ab9b409ba74c73c30179d5f7c12b025f631693f9eec78d80dca242547be`.
Read off the compiled model rather than off any XML:

- **Six sensors, and only six** — `orientation` (framequat on site `imu`), `angular-velocity`
  (gyro, site `imu`), `imu_ang_vel` (gyro, site `imu`), `imu_lin_vel` (velocimeter, site `imu`),
  `imu_accel` (accelerometer, site `imu`), `root_angmom` (**subtree angular momentum, type 37, on
  body `trunk_base`**, sensor address 16, dim 3).
- **No contact sensor of any kind. No collision sensor.** Contact is only ever available as a
  geometric query, `mj_geomDistance`, which is present in this MuJoCo build.
- **The ball**: body `ball`, mass **0.030 kg**, one geom `ball_geom`, sphere, **radius 0.05 m**,
  `condim` 6, on a free joint. The core already exposes it as `BALL` / `ballOf(d)` / `placeBall()`
  and `POST /ball {bearing, range}`.
- Control tick 50 Hz, timestep 0.005 s, 4 substeps per tick.

### The one caveat that shadows every absolute number below

**Pollen's ball is not this ball.** The config trains against a 70 mm-diameter, 15 g ball
(lines 76–77, `BALL_RADIUS = 0.035`, and `ball.xml`). This plant's ball is 100 mm-diameter and
30 g — **1.43× the radius and 2× the mass**. Every term below is still *computable*, and computed
from the same state with the same formula and the same weight; but a speed target expressed in
m/s (`BALL_TARGET_SPEED`, line 95) was tuned against a ball with half the inertia, so
`ball_forward_velocity` and `ball_speed_overshoot` on this plant are **the config's function
evaluated on a different ball**, not a reproduction of Pollen's training signal. Every result
row must carry `plantName` and `plantDigest` for exactly this reason, and the package card must
say this in the caveats section. Substituting Pollen's ball is not an option: `scene.mjb` is the
canon plant that every recorded clip in duckkit claims to come from, and changing it would
invalidate them.

---

## 2. The reward, term by term

The ball-kick config's reward stack is built in three moves:

1. **inherit** the twelve terms of mjlab v1.3.0's velocity template (`velocity_env_cfg.py` 275–373);
2. **delete eight** of them (config lines 210–221): `track_linear_velocity`,
   `track_angular_velocity`, `air_time`, `foot_clearance`, `foot_swing_height`, `foot_slip`,
   `pose`, `soft_landing`;
3. **add seven** and re-weight the survivors (config lines 238–310).

That leaves **twelve live terms**. Nine this plant can compute; three it cannot.

### 2.1 Terms this plant CAN compute — nine

| # | term | weight | formula, verbatim | source line |
|---|---|---|---|---|
| 1 | `ball_forward_velocity` | **+12.0** | `fwd = (ball.root_link_lin_vel_w[:, :2] · kick_dir).sum(); clamp(fwd, 0.0, max_speed)` with `max_speed = BALL_TARGET_SPEED = 1.0` | cfg 238–242 (weight, params); `mdp.py` 5761–5784 (function); cfg 95 (`BALL_TARGET_SPEED`) |
| 2 | `ball_speed_overshoot` | **−4.0** | `over = fwd − target_speed; clamp(over, 0.0, max_penalty)` with `target_speed = 1.0`, `max_penalty = 5.0` (the default, not overridden) | cfg 243–247; `mdp.py` 5787–5806 |
| 3 | `upright` | **+2.0** | `exp(−‖projected_gravity_b[:2]‖² / std²)` on body `trunk_base`, `std = sqrt(0.05)` ⇒ `std² = 0.05` | cfg 285–287 (re-weight); mjlab `velocity/mdp/rewards.py` 67–110 (function); mjlab `velocity_env_cfg.py` 286–293 (base term) |
| 4 | `pose_stand_legs` | **+2.0** | `exp(−((q − target)/std)²).mean()` over `_LEG_JOINTS = [0,1,2,3,4,9,10,11,12,13]`, `std = 0.5`, `target = default_joint_pos` (no overrides ⇒ HOME) | cfg 262–270 (weight, params), cfg 100 (`_LEG_JOINTS`); `mdp.py` 2396–2418 (function) |
| 5 | `pose_stand_neck` | **+1.0** | the same function over `_NECK_JOINTS = [5,6,7,8]`, `std = 0.3` | cfg 274–282, cfg 101; `mdp.py` 2396–2418 |
| 6 | `height_stand` | **+1.0** | `exp(−((z − target_height)/std)²)` on `trunk_base`, `z` = root z minus terrain origin z, `std = 0.04`, `target_height = STAND_Z = 0.115` | cfg 290–298; cfg 98 (`STAND_Z`); `mdp.py` 2440–2451 |
| 7 | `body_ang_vel` | **−0.05** | `Σ(ω_xy²)` of `trunk_base`'s **world-frame** link angular velocity; z is deliberately not penalised | cfg 302–303; mjlab `velocity/mdp/rewards.py` 184–193 |
| 8 | `action_rate_l2` | **−0.1 → −1.0** | `Σ(aₜ − aₜ₋₁)²` over the fourteen **raw** action outputs (before scale/offset) | cfg 301 (stage-0 −0.1); cfg 561–574 (curriculum to −1.0 by iter 1500); mjlab `envs/mdp/rewards.py` 58–65 |
| 9 | `angular_momentum` | **−0.02** | `Σ(angmom²)` — the squared magnitude of the subtree angular-momentum sensor | cfg 304; mjlab `velocity/mdp/rewards.py` 196–206; mjlab `velocity_env_cfg.py` 312–315 (`sensor_name = "robot/root_angmom"`) |

Notes that must survive into the code, because each of them is a place where a careless
transcription produces a plausible wrong number:

- **`kick_dir` is frozen, and it is the heading at reset — not the heading now.**
  `_ball_kick_dir` (`mdp.py` 5697–5707) allocates a per-env unit XY vector; `reset_ball_in_front_of_foot`
  (`mdp.py` 5756–5758) writes `(cos yaw, sin yaw)` of the robot's root yaw **at reset** into it and
  it is never touched again. The docstring says why: *"Frozen for the episode so the policy can't
  redefine 'forward' by turning after the kick."* On this bench that is the duck's yaw at the
  **first driven tick** — after the 25-tick settle, before the entrant acts — and it is the same
  vector `ballTravel_mm` projects onto. One vector, read twice, never recomputed.
- **`ball_forward_velocity` and `ball_speed_overshoot` read the *ball's* world linear velocity**,
  `qvel[BALL.dof .. BALL.dof+1]`, not a finite difference of `ballOf(d)`. The free joint carries it
  directly and a difference would disagree in the last digits.
- **Both ball terms are one-sided.** Backward and lateral ball motion earn **0**, not a penalty
  (`clamp(fwd, 0.0, …)`, `mdp.py` 5784). A mis-hit costs nothing; only overshoot costs.
- **`pose_stand_legs` / `pose_stand_neck` are NOT `rewardSums`' `pose`.** `rewardSums` computes
  mjlab's `variable_posture`, and this config **deletes** `pose` outright (cfg line 217). The two
  stand terms are `pose_target_match`, a different function with a fixed target and a fixed std.
  Reusing the `pose` number here would be reporting a term Pollen removed.
- **Of `rewardSums`' six terms, only three survive this config.** `upright`, `body_ang_vel` and
  `action_rate_l2` are transcribable straight across; `track_linear_velocity`,
  `track_angular_velocity` and `pose` are **deleted by lines 211, 212 and 217** and must not appear
  in a `/chase` answer at all. Answering them would be answering the wrong config under the right
  name — the failure `climb_score.mjs`'s header warns about.
- **The joint indices line up with this bench's 14-vector, and that must be asserted, not assumed.**
  duckkit's `jointNames` is fifteen long (`mouth` at index 9); every intent pose and every policy
  action is fourteen long with `mouth` dropped, giving
  `0 left_hip_yaw, 1 left_hip_roll, 2 left_hip_pitch, 3 left_knee, 4 left_ankle, 5 neck_pitch,
  6 head_pitch, 7 head_yaw, 8 head_roll, 9 right_hip_yaw, 10 right_hip_roll, 11 right_hip_pitch,
  12 right_knee, 13 right_ankle`. Under that ordering `_LEG_JOINTS` is exactly the ten leg joints
  and `_NECK_JOINTS` exactly the four neck/head joints. `chase_score.mjs` must assert this mapping
  at boot from `duckkit-constants.json` and throw if it ever stops holding, rather than carrying
  two index lists that agree today.
- **`action_rate_l2` is weighted at the ramp END, −1.0, and both numbers are published.** The
  config starts it at −0.1 (line 301) and the curriculum ramps it to −1.0 by iteration 1500
  (lines 561–574). The trained policy lived under the final value, so −1.0 is the weight the
  term is scored at — the same choice `RunMetrics.Task.actionRateWeight` already makes for
  `.ballKick`. The stage-0 value is carried in the answer beside it so nobody has to guess which
  was used.
- **`action_rate_l2` means two different things for the two entrant kinds, and the answer says
  which.** For a **policy** entrant, `aₜ` is the network's raw 14-vector output — exactly mjlab's
  `action_manager.action`. For a **move** entrant there is no network; the thing the move emits
  each tick is the interpolated, clamped pose target, and that is its `aₜ`. The formula is
  identical and the number is comparable *between moves* and *between policies*, but not across
  the two. So every row carries `action_rate_l2_source`: `"policy raw output"` or
  `"keyframe pose target"`. It is not refused, because the move genuinely has an action; it is
  labelled, because the two are not the same quantity.
- **`angular_momentum` is computable here, and this is a deliberate departure from `TUNE_REFUSALS`.**
  `sim/duckbench-core.mjs` refuses `angular_momentum` for the velocity config, and its stated
  reason is *not* a missing sensor — it says the plant does carry `root_angmom` and that the
  refusal is because *"nothing here reads its weight out of the config"*. That was a fact about
  `/tune`'s six-term transcription, not about the plant. Here the weight **is** read, from
  ball-kick config line 304 (−0.02), and the sensor **is** the right one: `root_angmom` is
  `mjSENS_SUBTREEANGMOM` rooted on body `trunk_base`, which is what mjlab's
  `"robot/root_angmom"` names. So `/chase` computes it and `/tune` still refuses it, and the two
  are consistent: each answers exactly the terms it has transcribed a weight for. `chase_score.mjs`
  must assert the sensor's type (37) and object (`trunk_base`) at boot and refuse the term by name
  if either check fails, rather than silently summing whatever is at that address.

### 2.2 Terms this plant CANNOT compute — three, refused by name

| term | weight | why it is refused |
|---|---|---|
| `support_foot_grounded` | **+2.0** (cfg 253–257) | It reads a **contact sensor**. The config builds `support_foot_ground_contact` (cfg 160–171): a `ContactSensorCfg` whose primary is the geom `left_foot_collision` and whose secondary is the body `terrain`, reduced to a `found` flag; `single_foot_grounded_reward` (`mdp.py` 5809–5824) returns `clamp(found, 0, 1)`. **This plant has six sensors and not one of them is a contact sensor.** `mj_geomDistance` could report how far a foot geom is from the floor, but that is a *distance*, and turning it into the config's binary `found` requires choosing a threshold Pollen never wrote. Choosing one would be inventing a reward. Refused. |
| `self_collisions` | **−1.0** (cfg 306–310) | It reads the `self_collision` sensor (cfg 173–180), a subtree-vs-subtree `ContactSensor` on `trunk_base`; `self_collision_cost` (mjlab `velocity/mdp/rewards.py` 162–181) counts its `found` slots or thresholds its force history at 10 N. **No collision sensor in this plant, and no contact forces at all.** Same wording as `TUNE_REFUSALS`. Refused. |
| `dof_pos_limits` | **−1.0** (mjlab `velocity_env_cfg.py` 317; *not* deleted by the kick config) | `joint_pos_limits` (mjlab `envs/mdp/rewards.py` 81–96) scores travel outside `soft_joint_pos_limits` — the **soft** limits, which are a configured fraction of the model's hard range. Neither duckkit nor this bench ships that fraction; scoring against the hard `rangeLo`/`rangeHi` would be a different term wearing this one's name, and picking a softening factor would be inventing it. Same wording as `TUNE_REFUSALS`. Refused. |

`/chase` answers these three in a `refused[]` array, each as `{term, weight, reason}`, in exactly
this order — **present and named, never dropped.** A shorter refusal list is not a better one.

### 2.3 Two things in the config that are wrong or stale, recorded rather than silently fixed

1. **The target-speed comment contradicts the constant.** Line 95 sets
   `BALL_TARGET_SPEED = 1.0`. The comment block at lines 224–237 describes the landscape as
   *"peaking at BALL_TARGET_SPEED (0.25 m/s — a gentle tap)"* and justifies the weight as
   *"Weight 12.0 = 3.0/target"* — and `3.0 / 0.25 = 12.0`, while `3.0 / 1.0 = 3.0`. The comment
   was written for a target of 0.25 and the constant was later moved to 1.0 without the rescale
   the comment itself asks for (line 92–94: *"if you change the target, rescale the weights with
   it"*). **The code is what ran, so the code is what is transcribed**: `max_speed = 1.0`,
   `target_speed = 1.0`, weight `+12.0`, weight `−4.0`. The discrepancy is recorded here and in
   the package card so that nobody "fixes" it into a third set of numbers.
2. **`KICK_FOOT` is a module-level flag, not a per-policy field.** Line 41 sets it to `"right"`.
   `ball_kick_left.onnx` was produced from the same file with the flag flipped (lines 37–42,
   143, 454). Nothing in the reward differs between the two runs; only the ball spawn side and
   which foot the (refused) `support_foot_grounded` sensor watches. So both bundled policies are
   scored under **one** term table, and that is correct.

---

## 3. The criterion

Everything in §2 is Pollen's reward. It is reported because it is the reward the policies were
trained on, and because a person editing a keyframe deserves to see it move. **It is not the
verdict.** A shaped sum of nine weighted terms is not a thing a person can hold in their head,
and a leaderboard sorted on it would reward a duck that stands beautifully still. The verdict is
a small number of plain facts, and one sentence over them.

### 3.1 The plain facts, per cell

Recorded per tick: ball position and world linear velocity; the duck's root, twist, fourteen
joints, action and command; and the minimum duck-geom-to-`ball_geom` distance by `mj_geomDistance`
(never `data.contact.get(i)` — that leaks the WASM heap to 2 GB in ~20 s even with `.delete()`,
per `climb/rig2.mjs`). Reduced to:

| fact | definition |
|---|---|
| `ballTravel_mm` | the ball's net displacement **projected onto the duck's initial heading**, in the world frame, in mm. The heading is the yaw at the **first driven tick** (settle over, entrant not yet acting) — the same frozen vector `kick_dir` uses. **Signed**: a ball pushed backwards scores negative. |
| `ballNet_mm` | `‖end − start‖` of the ball in the plane, in mm. Unsigned. Travel and net differ exactly when the ball went sideways, which is the case worth seeing. |
| `closest_mm` | the minimum, over every tick, of the smallest `mj_geomDistance` between any duck geom and `ball_geom`, in mm. Negative means interpenetration. |
| `final_mm` | duck-root-to-ball-centre distance in the plane at the last tick, in mm. |
| `touched` | `closest_mm <= 3.0` — any duck geom within 3 mm of the ball at any tick. |
| `ballPeakSpeed_mps` | the maximum over ticks of `‖ball linear velocity xy‖`. |
| `upright` | the trunk upright at the **last** tick, by the bench's existing test. |
| `uprightTailTicks` | how many of the final 50 ticks were upright, 0–50. |

### 3.2 `CRITERION_SENTENCE` — the sentence the verdict is, said the same way on every bench

> **chased: the duck touched the ball — any duck geometry within 3 mm of it at any tick — and the
> ball finished at least 100 mm further along the duck's initial heading than it started, and the
> duck was still upright at the end of the episode. stable: chased, and upright for at least 45
> of the 50 tail ticks.**

Machine-readable, and the only place these constants live:

```
TOUCH_MM        = 3.0     // touched := closest_mm <= 3.0
TRAVEL_MIN_MM   = 100.0   // chased  := touched && ballTravel_mm >= 100.0 && uprightAtEnd
TAIL_TICKS      = 50
UPRIGHT_TAIL_MIN = 45     // stable  := chased && uprightTailTicks >= 45
```

Why each clause is there, and what it is defending against:

- **`touched`** rules out a duck that walks past and knocks the ball with a draught of nothing —
  and, more usefully, rules out a ball that moved because the *floor* moved or because the episode
  started with the ball penetrating something. If the duck never made contact, the ball's motion
  is not the duck's doing.
- **`ballTravel_mm >= 100`**, signed and along the **initial** heading, rules out three cheats at
  once: a ball nudged 5 mm and called a kick; a ball driven backwards (negative, so it fails); and
  a duck that turns to face wherever the ball happens to have rolled and calls that "forward". It
  is the same defence, for the same reason, that Pollen's own `kick_dir` freeze is (`mdp.py`
  5700–5702).
- **`upright` at the end** rules out the one move that will otherwise dominate the leaderboard:
  fall on the ball. A toppling duck moves a 30 g sphere a long way. This is the ball challenge's
  equivalent of the stairs criterion's `s.up`, and it exists for the same reason.
- **`stable`** separates *the ball moved* from *the duck is still a robot afterwards*. 45 of 50
  is the same bar `climb_score.mjs` uses (`UPRIGHT_TAIL_MIN = 45`), deliberately, so that a
  person who has read one challenge already knows what this one means.

`chased` is the criterion the leaderboard sorts on; `stable` is the stricter one printed beside
it. Both are per cell; the aggregate is `kChased` of 9 core, `kStable` of 9 core, and `kExt` of
the 5 extended, exactly as `StairsScore` aggregates.

**What the criterion deliberately is not.** It is not "the ball ended near a goal", because there
is no goal in `scene.mjb` and adding one would change the canon plant. It is not "peak ball speed
above X", because peak speed is dominated by the 2× mass difference from Pollen's ball and would
make the number un-comparable to anything Pollen published. It is not a threshold on the shaped
reward of §2, because that sum has never been calibrated on this plant and a bar on it would be
a number nobody could defend. It is three facts a person can watch happen.

---

## 4. The grid — fourteen cells

Nine core, five extended. Positive bearing is **LEFT** — the convention `POST /ball` already uses
(`duckbench-core.mjs`: *"Positive bearing is LEFT, the convention duckvision and the robot both
use"*), so a trial reads the same way the detector reports. Range is metres from the duck's root
to the ball's centre, measured after the settle. `drop` is the spawn height; `fmul` multiplies
foot friction[0], both read off the same knobs `climb_score.mjs` uses, so `fmul = 1.0` is the
identity and the model is written back with the number it already had.

### The nine core cells — nominal plant, `drop 0.120`, `fmul 1.0`

| | range 0.45 m | range 0.70 m | range 0.95 m |
|---|---|---|---|
| **bearing −20°** (right) | core 1 | core 4 | core 7 |
| **bearing 0°** | core 2 | core 5 (**the centre cell**) | core 8 |
| **bearing +20°** (left) | core 3 | core 6 | core 9 |

### The five extended cells

| # | cell | what it stresses |
|---|---|---|
| ext 1 | bearing 0°, range 0.70 m, **drop 0.130, fmul 0.7** | the centre cell on a slippery, higher-spawning plant |
| ext 2 | bearing 0°, range 0.70 m, **drop 0.125, fmul 1.3** | the centre cell on a grippy plant |
| ext 3 | **bearing −40°**, range 0.70 m, nominal plant | a ball well off the heading, to the right |
| ext 4 | **bearing +40°**, range 0.70 m, nominal plant | a ball well off the heading, to the left |
| ext 5 | bearing 0°, **range 1.20 m**, nominal plant | straight ahead but far — a walk, not a lunge |

The plant pairs `(0.130, 0.7)` and `(0.125, 1.3)` are lifted verbatim from `climb_score.mjs`'s
`PLANTS[1]` and `PLANTS[2]`, so that "the slippery plant" means the same thing in both challenges.

`GET /chase/grid` answers this list and the criterion, in this order — core first, so a partial
run is still the core grid — each cell tagged `tier: 'core' | 'ext'`, and the kit's pinned
fallback is checked against it, exactly as `DuckBenchClimb` does.

### Why these axes, and why these numbers

**Bearing is the axis that makes this a chase.** A ball dead ahead can be reached by a policy that
only walks forward; a ball at ±20° cannot, and a ball at ±40° certainly cannot. The grid is
therefore built so that the bundled entrants pass some cells and fail others *by construction* —
the off-bearing cells are the ones the challenge is actually about, and a leaderboard that only
ever ran bearing 0 would look solved while nothing could chase anything.

**Range spans reach, a step, and a walk.** 0.45 m is roughly two body lengths and out of reach of
any single lunge from a standing duck; 0.70 m needs travel; 0.95 m and the extended 1.20 m need
sustained walking with the ball still where it was left. None of the three is reachable without
locomotion, which is deliberate: Pollen's kick task spawns the ball **90 mm** in front of the toe
(`BALL_OFFSET_X = 0.09`, cfg line 84), so the nearest cell here is **five times** the distance the
kick policies were trained at. That gap is the finding this challenge is built to expose, not a
mistake in the grid.

---

## 5. The controls

Four bundled entrants, run before anything else, on all fourteen cells. Their measured rows become
the kit's sha-pinned bundled leaderboard, so their expected behaviour is declared **here, in
advance**, and a run that disagrees is a finding to be chased down rather than a number to be
written down.

### 5.1 `ctrl_do_nothing` — a move that stands still

**Entrant**: `kind: "move"`, two keyframes at HOME, `blend: 1`, `seconds: 5`.

The HOME pose is duckkit's `homePose` with the mouth (index 9 of fifteen) dropped:

```
[0, -0.0873, -0.4579, -0.0049, 0.453, 0.3491, 0.3491, 0, 0, 0, 0.0873, 0.4579, 0.0049, -0.453]
```

**Expected: 0 of 14. It must fail every cell.** The duck holds HOME for five seconds and the
nearest ball is 450 mm away; nothing touches it, so `touched` is false, `ballTravel_mm ≈ 0`,
`closest_mm` is on the order of 400 mm, `ballPeakSpeed_mps ≈ 0`. `upright` and `uprightTailTicks`
should both be good — that is the point: **a criterion this row passes is not a chasing test.**
It is the same guard `climb/ctrl_do_nothing.json` is, in the same words, and it is the first
thing `chase_parity.mjs` checks.

### 5.2 `ctrl_ball_kick_left` and `ctrl_ball_kick_right` — the bundled policies at the config's own command

**Entrant**: `kind: "policy"`, `policy: "ball_kick_left.onnx"` / `"ball_kick_right.onnx"`,
`schedule: [[0, {"vx": 0, "vy": 0, "vyaw": 0}]]`, `seconds: 5`.

**The schedule is read out of the config, not chosen by taste.** The ball-kick env does have a
twist command slot — it is kept only for observation-shape parity with the unified 61-D actor
layout — and its ranges are, verbatim (cfg lines 406–408):

```python
command.ranges.lin_vel_x = (-0.01, 0.01)
command.ranges.lin_vel_y = (-0.01, 0.01)
command.ranges.ang_vel_z = (-0.05, 0.05)
```

with `rel_standing_envs = 0.0`, `heading_command = False`, and a resampling period equal to the
whole episode (cfg lines 400–404), i.e. **one sample, held for the episode**. The comment on the
block says it outright: *"Command: tiny noise around zero (obs-shape parity only)"*. The centre
of all three ranges is zero, so `(0, 0, 0)` held for the episode is **the centre of the
distribution these two policies were actually trained under**. Commanding them to walk would be
commanding them 50× outside their `lin_vel_x` range and calling the result a measurement of
Pollen's policy.

`seconds: 5` is `EPISODE_LENGTH_S = 5.0` (cfg line 74) — the episode length they were trained at.

**Expected: 0 of 14, or very close to it, and this is the whole point.** These policies are trained
to stand still and swing one leg at a ball **90 mm in front of the toe**, and they are *blind to
the ball by design* (cfg lines 11–15: *"The policy is BLIND to the ball (no ball obs in the
actor)… the operator aims the robot at the ball"*). At the grid's nearest cell the ball is 450 mm
away and the policy cannot see it, cannot walk to it, and is not commanded to. So the prediction
is: they swing, they stay upright, `touched` is false in every cell, `ballTravel_mm ≈ 0`,
`uprightTailTicks` high.

That is not a failed control. It is the measurement that states the problem: **the best ball
policy Pollen ships cannot chase a ball, because chasing was never the task.** If a kick policy
*does* pass a cell — most plausibly bearing 0 at 0.45 m if the swing carries the duck forward —
that is a genuine and interesting result, and it is exactly why the row is run rather than
assumed.

The kit's "policy rows are not editable" sentence exists for these two: a person can score them
and play them back, but there are no keyframes to open in the editor.

### 5.3 `ctrl_alpha_walking` — the naive chaser

**Entrant**: `kind: "policy"`, `policy: "alpha_walking.onnx"`,
`schedule: [[0, {"vx": 0.5, "vy": 0, "vyaw": 0}]]`, `seconds: 4`.

Straight ahead at 0.5 m/s for four seconds. `alpha_walking.onnx` is the **velocity** config's
policy (`RunMetrics.Task.forPolicy` maps it to `.velocity`), so its terms are a different table
from §2 — it is here as a *chaser*, judged by the criterion, and its §2 term values are reported
with the standing caveat that it was trained under `microduck_velocity_env_cfg.py`.

**Expected: passes some of the three bearing-0 cells, fails every off-bearing cell.** At 0.5 m/s
for 4 s the nominal reach is about 2 m, so the ball at 0.45, 0.70, 0.95 and even 1.20 m is inside
the walk — *if* it is dead ahead. At ±20° a ball at 0.70 m sits about 240 mm off the walk line,
and at ±40° about 450 mm off; the duck walks straight past both. Prediction: **roughly 2–4 of 9
core cells, and ext 5 (range 1.20, bearing 0) plausibly passing while ext 3 and ext 4 (±40°)
certainly fail.**

This is the control that makes the challenge legible. It draws the line the challenge is asking
someone to cross: **open-loop forward walking already solves "the ball is straight ahead"; nothing
bundled solves "the ball is over there".** Every off-bearing cell is unclaimed, and closing them
needs steering — a person editing a keyframe to make the duck turn, or, later, a policy that
reads the ball. That is the closed-loop question this entrant format exists to make askable.

### 5.4 What the four controls establish between them

| control | what a pass would prove | what a fail proves |
|---|---|---|
| `ctrl_do_nothing` | the criterion is broken | the criterion requires the duck to do something |
| `ball_kick_left` / `_right` | Pollen's kick generalises past 90 mm | reaching the ball is the unsolved half, not kicking it |
| `alpha_walking` @ vx 0.5 | walking is enough when the ball is ahead | steering is what the off-bearing cells need |

---

## 6. The entrant formats

Two kinds, both scoreable from day one, both carried **verbatim** through `HarnessJSON` so that an
entrant file round-trips byte-for-byte and hashes to one value. The hash is sha256 over the
normalised entrant; unknown keys are preserved and hashed, never stripped.

### 6.1 An authored move

Keyframes and a blend, run under the bench's 25-tick settle exactly as `/climb` runs one. The
episode reads `keyframes` and `blend`; anything else in `intent` is carried through untouched and
hashed (so a file that also carries stairs fields is a *different* entrant, not a silently
equivalent one). `pose` is fourteen numbers, `t` is seconds from the first driven tick.

```json
{
  "name": "ctrl_do_nothing",
  "kind": "move",
  "seconds": 5,
  "intent": {
    "name": "ctrl_do_nothing",
    "keyframes": [
      { "t": 1.0, "pose": [0, -0.0873, -0.4579, -0.0049, 0.453, 0.3491, 0.3491, 0, 0, 0, 0.0873, 0.4579, 0.0049, -0.453] },
      { "t": 4.9, "pose": [0, -0.0873, -0.4579, -0.0049, 0.453, 0.3491, 0.3491, 0, 0, 0, 0.0873, 0.4579, 0.0049, -0.453] }
    ],
    "blend": 1
  },
  "note": "DO-NOTHING CONTROL. HOME pose held for the whole track. A criterion this row PASSES is not a chasing test."
}
```

### 6.2 A policy under a command schedule

A policy name from the bench's catalogue and a schedule — a list of `[atSeconds, {vx, vy, vyaw}]`,
the last entry that has begun wins, which is the core's existing `commandAt` contract. This is the
format that makes "chase" a closed-loop question later: the same field carries a fixed schedule
today and a schedule computed from the ball's bearing tomorrow, with no change to the entrant
format, the hash, the kit or the app.

```json
{
  "name": "ctrl_alpha_walking_vx05",
  "kind": "policy",
  "seconds": 4,
  "policy": "alpha_walking.onnx",
  "schedule": [[0, { "vx": 0.5, "vy": 0, "vyaw": 0 }]],
  "note": "NAIVE CHASER. Straight ahead at 0.5 m/s. Reaches a ball dead ahead; walks past every off-bearing one."
}
```

### 6.3 What `POST /chase` answers

For **one** cell of **one** entrant: the eight facts of §3.1; the nine computed terms of §2.1 as
`{term, weight, value}` with `action_rate_l2_source` beside the eighth; `refused[]` — the three
of §2.2, each `{term, weight, reason}`; `chased`; `stable`; `uprightTailTicks`; `hash`;
`plantName`; `plantDigest`; `criterion` (the `CRITERION_SENTENCE` string of §3.2); and `seconds`.

The ball and every other piece of laid-out state are captured before the cell and restored after
it, so every existing endpoint answers exactly as it did before and `bench_parity.mjs` passes —
the same discipline `climb_score.mjs` keeps for the shipped step affinity.
