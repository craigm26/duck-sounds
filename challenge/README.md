---
license: cc-by-4.0
pretty_name: "Microduck Stairs Challenge"
tags:
  - mujoco
  - robotics
  - bipedal
  - microduck
  - pollen-robotics
  - benchmark
  - stairs
task_categories:
  - robotics
  - reinforcement-learning
---

# Microduck Stairs Challenge

**Simulation only, never on hardware.** Every number in this card was produced in MuJoCo,
in the plant `sim/scene.mjb` (sha256 `3f8c9ab9b409ba74c73c30179d5f7c12b025f631693f9eec78d80dca242547be`,
compiled from `sim/scene_physics.xml`, shipped here as `harness/scene_physics.xml`). No move
in this package has ever been run on a physical Microduck, and nothing here should be run on
one without a separate safety review: in the 20 cells where the trunk got over the 95 mm bar,
across the four traced vault vectors, at least one of the 14 actuators is pinned at its
0.6405 N·m ceiling for 73.6% of a push-off that averages 1,399 ms
(`results/r6_limits_table-results.json` → `pooled.over95`: `anyOf14AtForceCeiling` 0.7358,
`pushOffMs` 1399), and the record move plants the robot's beak on a step and pivots its whole
body over its own head.

## What the challenge is

The Pollen Robotics Microduck is a bipedal duck about 250 mm tall (Pollen's published spec, not a
measurement of this plant) with 14 position-controlled
servos. Its actuators are capped at ±0.6405 N·m (`harness/scene_physics.xml`; the constant is
carried through every scored row as `forcerange_Nm` in `results/r6_judge-results.json`).
The challenge is to get it from the floor onto a step, in simulation, and leave it standing
there — not to touch the step, not to get a foot up, but to end the episode upright on the
tread with both feet resting on it. A submission is not a policy and not a video: it is a
saved intent JSON, a handful of keyframes of 14 joint targets with times, which the scorer
replays in the plant and judges. The control tick is 20 ms (`tickHz` 50,
`results/r6_judge-results.json` → `constants`).

Six rounds of search ran on 2026-09-01 and 2026-09-02. The best result anyone has produced is
5 of 9 robustness cells at a 60 mm rise, and the round-6 judge concluded that this is a limit
of the robot and the move class rather than of the optimiser — the section *What is closed and
why* gives the measurement that closes it. The bar for the challenge (7 of 9) has never been
met at any rise. This package publishes the criterion, the grid, the record, the negative
results and the harness so that somebody can beat it or show that it cannot be beaten.

## The criterion

The judge is `climb/rig3.mjs`, published here as `harness/rig3.mjs`. There are four criteria in
the file; **`honest` is the one that counts.** Verbatim, `harness/rig3.mjs` line 330:

```js
const honest = s.up && lateral && s.x > RISER_X && s.above > 0.095 && s.feetOnTread >= 2;
```

Every term, with its source line:

| term | line | meaning |
|---|---|---|
| `s.up` | 283 | `projectedGravity(quat())[2] < -0.90` — the trunk is upright |
| `lateral` | 325–327 | `lateralAtScore && lateralEpisode`: `\|y − STAIR_Y\| <= 0.17` at the scored instant **and** as a whole-episode maximum. The flight is 340 mm wide (`harness/stairs.js` `STAIR_HALF_WIDTH = 0.17`) |
| `s.x > RISER_X` | 100 | `RISER_X = 0.12` — the trunk is past the first riser face |
| `s.above > 0.095` | 330 | trunk z minus the tread height exceeds 95 mm |
| `s.feetOnTread >= 2` | 292–293 | both feet are RESTING on the tread, by `footResting` |

`footResting`, verbatim, `harness/rig3.mjs` lines 145–150:

```js
function footResting(g, h) {
  const x = data.geom_xpos[g * 3], y = data.geom_xpos[g * 3 + 1], z = data.geom_xpos[g * 3 + 2];
  if (!(z > h - 0.005 && z < h + 0.045 && x > RISER_X && Math.abs(y - STAIR_Y) <= LATERAL)) return false;
  for (const sg of STEPG) if (mj.mj_geomDistance(model, data, g, sg, 0.05, null) < 0.003) return true;
  return false;
}
```

— a foot counts only when it is past the riser line, inside the flight, from 5 mm below to
45 mm above the tread top, **and** within 3 mm of a step geom.

The criterion is evaluated at the scored instant, which is the end of a 50-tick tail of the
standing policy (`BEST_alpha_stand.onnx`) run after the authored keyframe track ends. A clear
is **STABLE** when the duck was also upright for at least 45 of those 50 tail ticks
(`harness/robust.mjs` line 528 `UPRIGHT_TAIL_MIN = 45`, applied at line 688:
`r.stableClear = r.crit.honest && r.uprightTailTicks >= UPRIGHT_TAIL_MIN`).

The criterion separates the two controls it has to separate: a duck spawned already standing on
the tread passes 14 of 14 cells and a do-nothing duck fails all 14
(`results/r4_judge-results.json` → `phaseC`, `results/r5_judge-results.json` → `phaseC`,
`results/r6_judge-results.json` → `phaseC`; all three record
`"doNothingAlwaysFails": true, "placedDuckAlwaysPasses": true`).

## The grid and the bar

`climb/robust.mjs` (`harness/robust.mjs`) scores one saved file over a grid of rise-and-plant
perturbations. The definition is `harness/robust.mjs` lines 494–518, and the same grid is
recorded in `results/r6_judge-results.json` → `constants.grid`:

**Core, 9 cells** = rise `{h − 10 mm, h, h + 10 mm}` × plant
`{(drop 0.120 m, foot friction ×1.0), (0.130, ×0.7), (0.125, ×1.3)}`.

**Extended, 14 cells** = the core 9, plus rise `{h − 5, h + 5}` at the nominal plant, plus the
slippery plant `(drop 0.140, ×0.5)` crossed with the three core rises.

`kCore` counts core cells cleared under `honest`; `kCoreStable` counts those that also held
45/50 tail ticks; `kExt` / `kExtStable` are the same over all 14.

**THE BAR: 7 of 9 stable core cells at a rise.** Nothing has reached it. Two kill gates were
written in advance and both record `"result": "FAILED"`: round 5's was the bar itself
(`results/r5_judge-results.json` → `killGate`: `kCoreStable >= 7 of 9 at 60 mm`), and round 6's
was the ceiling the bar needs (`results/r6_judge-results.json` → `killCondition`: a distinct
launch reaching `ceilingCore >= 7 of 9 at 60 mm`).

## Leaderboard

One row per **distinct vector** (sha256 of the normalised intent). Where the same vector was
published under more than one rise label, the extra rise rows carry no rank and say so. The rank
column orders by `kCoreStable` only and is not comparable across rises: a lower rise is a strictly
easier task. The `sha256` is `intentHash` over the normalised intent (see *How to submit*), not
the file digest. The `scored` column is the date of the judge run that produced the row, not the
file's creation date (the round-2 files date from 2026-09-01). The table lists vectors that exist
as saved intent files; three further vectors in the rebuilt CEM corpus (`b3f06fb9e903`,
`56b676fab298`, `4886bd27f9a3`, `results/r6_judge-results.json` → `phaseX`) reach `kCoreStable` 4
at 60 mm but were never published as files.
`ceilingCore` = how many of the 9 core cells the trunk's **peak** height ever exceeds the 95 mm
bar in; it is an upper bound on `kCore` under any landing law. `n/m` = not measured at that rise
(the ceiling screen was run at 60 mm only).

**A note on which vector is the record.** Through round 5 the record was the round-3 beak-strut
vault `4b9110c448ec` at 4 of 9 (`results/r5_judge-results.json` → `killGate.bestKCoreStable` 4,
`bestFile` `best_r3_vault_60mm.json`). Round 6's ceiling CEM — whose objective was height, not
landing — produced `a56d459fb649`, which clears **5** of 9 stably at 60 mm and is the only vector
in the corpus where `kCore == ceilingCore == 5`. The round-6 judge records it as the holder:
`results/r6_judge-results.json` → `killCondition.bestKCoreStable` 5,
`bestKCoreStableMove` `a56d459fb649`. It is ranked first here for that reason. Both vectors are
the same move class — the beak-strut vault — and neither reaches the 7-of-9 bar.

| rank | sha256 | file | rise | kCore stable / 9 | kExt / 14 | ceilingCore | who | scored (judge run) | notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `a56d459fb649` | `intents/best_r6_ceilvaultC_60mm.json` | 60 mm | **5** / 9 (kCore 5) | 5 (stable 5) | 5 / 9 | round-6 ceiling CEM | 2026-09-02 | beak-strut vault; floor spawn, no servo, no event; the only vector where `kCore == ceilingCore == 5` |
| 2 | `4b9110c448ec` | `intents/best_r3_vault_60mm.json` | 60 mm | 4 / 9 (kCore 4) | 4 (stable 4) | 5 / 9 | round-3 family A | 2026-09-02 | the beak-strut vault: beak planted on the tread, neck locked as a strut, hips extend, the trunk pivots over the head, the feet land on the tread |
| — | `4b9110c448ec` | `intents/best_r3_vault_70mm.json` | 70 mm | 2 / 9 (kCore 2) | 2 (stable 2) | n/m | round-3 family A | 2026-09-02 | **same vector as rank 2**, scored at a 70 mm rise |
| — | `4b9110c448ec` | `intents/best_r3_vault_80mm.json` | 80 mm | 1 / 9 (kCore 1) | 1 (stable 1) | n/m | round-3 family A | 2026-09-02 | **same vector as rank 2**, scored at an 80 mm rise; its one clear is the 70 mm cell of that grid |
| 3 | `7b790070b010` | `intents/best_r4_famA_60mm.json` | 60 mm | 4 / 9 (kCore 4) | 4 (stable 4) | 5 / 9 | round-4 family A | 2026-09-02 | event-triggered landing on the rank-2 launch; behaviourally identical to it (`maxDx_mm` 0, `maxDz_mm` 0, `results/r4_judge-results.json` → `phaseD`) |
| 4 | `29c97398fe13` | `intents/best_r6_ceilvaultB_60mm.json` | 60 mm | 2 / 9 (kCore 3) | 3 (stable 2) | 5 / 9 | round-6 ceiling CEM | 2026-09-02 | chain B |
| 5 | `7904bf3363c5` | `intents/best_r3_vault_50mm.json` | 50 mm | 2 / 9 (kCore 2) | 2 (stable 2) | 2 / 9 | round-3 family A | 2026-09-02 | |
| 6 | `dff01b0a1906` | `intents/best_r3_vault_40mm.json` | 40 mm | 1 / 9 (kCore 2) | 2 (stable 1) | 3 / 9 | round-3 family A | 2026-09-02 | two `honest` clears, one of which topples inside the tail |
| 7 | `8c57838ee9d0` | `intents/best_r6_ceilvault_60mm.json` | 60 mm | 1 / 9 (kCore 1) | 1 (stable 1) | 5 / 9 | round-6 ceiling CEM | 2026-09-02 | best ceiling objective; five cells over the bar, one landed |
| 8 | `74d35b21ac80` | `intents/best_r2_vault_60mm.json` | 60 mm | 1 / 9 (kCore 2) | 2 (stable 1) | 3 / 9 | round-2 vault | 2026-09-02 | |
| 9 | `86813f9c1ad4` | `intents/best_r2_vault_40mm.json` | 40 mm | 1 / 9 (kCore 1) | 1 (stable 1) | 4 / 9 | round-2 vault | 2026-09-02 | |
| — | `e0434c2c90da` | `intents/best_r5_servo_60mm.json` | 60 mm | 0 / 9 (kCore 4) | 5 (stable 0) | 4 / 9 | round-5 servo | 2026-09-02 | **ORACLE** — the servo law reads tread height and edge from the plant |
| — | `880a120ef649` | `intents/best_r5_servoland_kcore_60mm.json` | 60 mm | 0 / 9 (kCore 3) | 4 (stable 0) | 3 / 9 | round-5 servo | 2026-09-02 | **ORACLE** |
| — | `2524a35672b4` | `intents/best_r4_famB_beat1_90mm.json` | 90 mm | 0 / 9 (kCore 0) | 0 (stable 0) | n/m | round-4 family B | 2026-09-02 | the best 90 mm move in the corpus |
| — | `7c52acef4acf` | `intents/best_r4_famB_beat1_120mm.json` | 120 mm | 0 / 9 (kCore 0) | 0 (stable 0) | n/m | round-4 family B | 2026-09-02 | the best 120 mm move in the corpus |
| — | `725674c1b517` | `intents/best_r3_cornerclimb_180mm.json` | 180 mm | 0 / 9 (kCore 0) | 0 (stable 0) | n/m | round-3 corner climb | 2026-09-02 | the best 180 mm move in the corpus |

**Reference rows — controls, not entries.**

| — | sha256 | file | rise | kCore stable / 9 | kExt / 14 | ceilingCore | what it is |
|---|---|---|---|---|---|---|---|
| ctrl | `d99589396fcb` | `intents/r4_ctrl_on_tread_60mm.json` | 60 mm | 9 / 9 | 14 (stable 14) | 9 / 9 | **PLACED SPAWN — NOT A CLIMB.** A duck spawned already standing on the tread. Proves the criterion can be passed. |
| ctrl | `f5bb2f0476c1` | `intents/r4_ctrl_on_tread_90mm.json` | 90 mm | 9 / 9 | 14 (stable 14) | 9 / 9 | placed spawn at 90 mm |
| ctrl | `c703ee6f5a14` | `intents/ctrl_do_nothing.json` | 40/60/90 mm | 0 / 9 | 0 (stable 0) | 0 / 9 | a duck that stands still. Proves the criterion cannot be passed for free. |

Row sources: ranks 1, 3, 4, 7 and the two oracle rows from `results/r6_judge-results.json` →
`phaseG`; rank 2 and the 70/80 mm rows and ranks 5, 6, 8, 9 and the 90/120/180 mm rows from
`results/r4_judge-results.json` → `phaseG` and `ladder`; `ceilingCore` for ranks 2, 5, 6, 8, 9
from `results/r5_judge-results.json` → `phaseE`, and for the round-6 vectors from
`results/r6_judge-results.json` → `phaseE`; control rows from `results/r4_judge-results.json` →
`phaseC` and `results/r6_judge-results.json` → `phaseC`.

## What is closed, and why

### The ceiling bounds the score, and the ceiling is 5 of 9

`honest` requires the trunk more than 95 mm above the tread at the scored instant. The trunk's
height at that instant can never exceed its peak height over the episode. So the number of core
cells in which the peak ever crosses 95 mm — call it `ceilingCore` — is a hard upper bound on
`kCore` under **any** landing law: timed, event-triggered, servoed, or not yet invented. The
round-6 judge states the identity and checks it on every row it scores
(`results/r6_judge-results.json` → `phaseE`, `"kCoreLEQceiling": true` on every row;
`phaseW.kCoreLEQceilingEverywhere` and `phaseX.kCoreLEQceilingEverywhere` both `true`).

Round 6 then measured that ceiling as hard as it could at a 60 mm rise:

- **394 distinct vectors** were scored at 60 mm
  (`results/r6_judge-results.json` → `killCondition.distinctVectorsScoredAt60mm` = 394). After
  de-duplication by `intentHash` the pool is: the 340-move CEM corpus rebuilt from its parameters
  and re-scored by the judge (`results/r6_judge-results.json` → `phaseX`: `claimedDistinctMoves`
  340, `rebuilt` 340, `hashMatches` 340, `ceilingMatches` 340, `kMatches` 340); the 48 distinct
  vectors among the 64 rounds-0-to-5 `best_*` files (`results/r6_screen-results.json`: `files` 64,
  `distinctVectors` 48, 2 refused as out of declared bounds); and the round-6 claim files (six of
  them servoed oracle entries). The pool's eligibility test for the published files checked torque
  only, so three handoff-spawned family-B vectors that the round-4 judge labelled NOT A CLIMB are
  in it; none of them reaches a ceiling above the record's, so the bound stands either way.
- That CEM's objective **was** `ceilingCore` — height only, no landing term, no servo, no event
  (`results/r6_ceiling-results.json`: `objective` = `ceilingCore + mean over the 9 core cells of
  min(peak trunk height above that cell's tread, 0.12 m)`, `landingTerm` `"NONE"`,
  `servo` `"NONE"`, `event` `"NONE"`).
- **The highest `ceilingCore` anywhere is 5 of 9.** The histogram over all 394 distinct vectors
  is `{0: 64, 1: 136, 2: 106, 3: 65, 4: 18, 5: 5}` and `bestCeilingCore` is 5
  (`results/r6_judge-results.json` → `killCondition`).
- **Every cell is individually reachable, and they never accumulate.** Taking the best peak in
  each of the nine cells separately across the 340-move corpus gives
  `[124.1, 141.6, 133.1, 143.8, 131.7, 129.8, 127.5, 131.0, 124.9]` mm — all nine over the
  95 mm bar, a 124–144 mm range (computed from
  `results/r6_judge-results.json` → `phaseX.rows[].measured.peakAboveTread_mm`). No single
  vector holds more than five of them at once.

So the kill condition — "a distinct floor-spawned launch reaching `ceilingCore` ≥ 7 of 9 at
60 mm" — failed, and the search is finished at this scale.

### The physical limit is the strut lever, not the tuning

The vault's whole lift comes from the beak pressed against the tread with the neck locked as a
strut. Round 6 traced that pose and measured the moment arm
(`results/r6_limits_table-results.json`, a diagnostic gated to reproduce the scorer exactly —
`"parity": "9/9 EXACT"` on all four traced files):

- the longest `neck_pitch` lever to the jaw over the four traced vault vectors is
  **87.3 – 89.5 mm** (`perFile[].neckLeverH_mm[1]` = 89.3, 89.3, 89.5, 87.3);
- 0.6405 N·m through those levers is **7.16 – 7.34 N** at the beak
  (`perFile[].neckMaxForceAtLongestLever_N` = 7.172, 7.172, 7.156, 7.337);
- the body weighs **7.23 N** (`bodyWeight_N`); the neck's stall figure used by the trace is an
  input constant, **7.66 N** (`neckStall_N`), which is `forcerange_Nm` 0.6405 ÷ 7.66 = 0.08362 m of
  lever — shorter than the 87 to 90 mm the vault's pose actually presents, which is why the stall
  never binds and the lever does.

These numbers are reported as the round-6 search agent measured them. The `9/9 EXACT` parity
proves the trace replays the scorer's dynamics tick for tick; it does not re-derive the lever
arithmetic independently, and the round-6 judge says so in `results/r6_judge-results.json` →
`whyItFailed.strutGeometry`.

The push-off force available at the beak straddles the weight it has to lift. That is geometry,
not gains.

Saturation is consistent with that reading, not proof of it. Every scored row that pushes runs
`maxTq` = 0.6405 N·m exactly — the ceiling is pinned, not approached; the only rows below it are
the ones that never push (`results/r6_judge-results.json` → `phaseT`: `claimsSaturating` 21 of
21, `publishedCorpusSaturating` 60 of 62, `rebuiltSearchSaturating` 340 of 340,
`anyRowOverCeilingAnywhere` false). Over the 36 traced cells (4 files × 9 core cells) hip
saturation correlates positively with peak height (`r = +0.254`, n = 36 — not distinguishable
from zero at that n; the knee's r is −0.005), `results/r6_limits_table-results.json` →
`correlationsOverAll36Cells`. Read together with the per-file numbers, per-tick saturation looks
like a symptom of lifting rather than the cause of failing; the lever arithmetic above is the
argument, the correlation only fails to contradict it.

### Above 80 mm it is a lift budget

At a rise `h`, `honest` needs the trunk at `h + 95` mm. A standing duck's trunk sits at
**116.2 mm** (`results/r4_famB-results.json` → `instrument.parity`, the `ctrl_do_nothing.json`
row at a 90 mm rise, `terminal_z_mm` 116.2). So the lift required is
**58.8 mm** at 80 mm, **68.8 mm** at 90 mm and **98.8 mm** at 120 mm. The round-4 family-B
two-beat search — the whole-body approach built specifically for that band — peaked at
**153.8 mm** of trunk height across all 114 `peakZ_mm` values in its results file, i.e. about
**37.6 mm** of lift bought
(computed from every `peakZ_mm` in `results/r4_famB-results.json`). All nine of its published
files clear 0 of 9 core cells at their own rise (`results/r4_judge-results.json` → `phaseG`), and
the best move in the whole corpus at 90 mm and at 120 mm also clears 0 of 9
(`results/r4_judge-results.json` → `ladder`).

### 180 mm

Zero tread contact in **2,829 episodes**: the round-3 corner-climb arm at 180 mm records
`episodes` 2829, `feetOnTreadMax` 0, `meanFeetOnTreadMax` 0, `meanFeetOnTreadFinal` 0,
`sustainFrac` 0 and `meanAbove_mm` −80.6 (`results/r3_cornerclimb-results.json` → `arms[1]`).
No foot ever got onto the tread, let alone the trunk.

### The instrument was broken before 2026-09-02, and everything below 180 mm before the repair is void

The flight is built from 200 mm-tall step blocks on frictionless slides. At any rise under
200 mm consecutive blocks **interpenetrated** and shoved each other apart. `results/rig3.r2.log`
Phase E measures the gap directly: with the flight's four steps, `minStepGap_mm` is −45.57 at a
20 mm rise, −52.57 at 40/60/90 mm, −51.80 at 120 mm, −16.62 at 180 mm, with tread drift up to
19.63 mm and tread sag up to 16.26 mm (Phase D). The consequence, in the same log's threshold
sweep: a duck **placed on the tread** fails `honest` at every rise from 20 mm to 170 mm with the
four-step flight — below 150 mm it is shoved off the tread within ten ticks, and from 150 to
170 mm the shove turns vertical and pushes the tread down into its feet (`minStepGap_mm` −41.52,
−33.22, −24.92 at 150, 160, 170 mm) — and passes at every rise from 40 mm to 140 mm with a single
step. Only at 180 mm does a placed duck pass on the four-step flight. `site/stairs.js`
`isolateSteps` repaired it (commit `279b016`) and `harness/stairs.js` carries the repair. **Every
result produced before that repair is void below 180 mm.** All numbers in this card come from
after it.

## What would change the answer

Not another optimiser over the vault's 29 parameters. The round-6 judge names two things
(`results/r6_judge-results.json` → `whyItFailed.whatWouldHaveToChange`):

1. **The robot.** More torque at the neck than 0.6405 N·m, or the same torque through a shorter
   strut lever than the 87.3–89.5 mm the vault pose forces. The arm is the term that eats it.
2. **The move class.** A move that does not ask the duck to lift its own trunk 95 mm unaided: a
   second duck to push off, a wall or rail to react against, a lever or a ramp placed first.

A third thing would also change it, honestly: a different criterion. `honest` is one of four in
`harness/rig3.mjs`; `honest60` relaxes the height clause to 60 mm and is defined at line 331. This
challenge scores `honest`.

## How to reproduce

From a fresh clone. The runnable harness is the GitHub repository, not this package — the sim
needs `mujoco` (wasm), `onnxruntime-node`, the compiled plant `sim/scene.mjb` and the policy
`.onnx` files, all of which live there.

Verified on Node 24.16.0 / npm 11.13.0 on linux-arm64 (a Raspberry Pi 5); Node 20 or later is
expected to work. `npm ci` runs onnxruntime-node's postinstall, which downloads a native binary,
so it needs network beyond the registry and a platform onnxruntime ships a build for. The full
repository is about 725 MB checked out; the shallow clone below is about 200 MB. If the tag is
missing on your mirror, `main` at or after 2026-09-02 carries the same harness.

```bash
git clone --depth 1 --branch stairs-challenge-v1 https://github.com/craigm26/duck-sounds.git
cd duck-sounds

cd sim
npm ci            # mujoco, onnxruntime-node
# The scorer lives in climb/ and imports bare `mujoco` and `onnxruntime-node`; it finds them
# through a committed symlink, climb/node_modules -> ../sim/node_modules. If your checkout did
# not create symlinks (git config core.symlinks false, or a zip download), make it by hand:
[ -e ../climb/node_modules ] || ln -s ../sim/node_modules ../climb/node_modules

# 1. Score ONE intent on ONE grid cell, and print the criterion:
node --input-type=module -e '
  import { scoreSaved } from "../climb/rig3.mjs";
  const r = await scoreSaved("../climb/best_r3_vault_60mm.json", { rise: 0.060, tail: "policy" });
  console.log("honest", r.crit.honest, "above_mm", (r.scored.above*1000).toFixed(1),
              "feetOnTread", r.scored.feetOnTread, "uprightTailTicks", r.uprightTailTicks);
'

# 2. Score the same intent over the 14-cell robustness grid:
node --input-type=module -e '
  import { scoreRobust } from "../climb/robust.mjs";
  const g = await scoreRobust("../climb/best_r3_vault_60mm.json", { rise: 0.060 });
  const peaks = g.cells.filter(c => c.cell.tier === "core").map(c => c.maxZ - c.rise);
  console.log("kCore", g.kCore + "/9", "kCoreStable", g.kCoreStable + "/9",
              "kExt", g.kExt + "/14", "kExtStable", g.kExtStable + "/14",
              "ceilingCore", peaks.filter(z => z > 0.095).length + "/9",
              "maxTq", g.agg.maxTq);
'
```

Expected for `best_r3_vault_60mm.json` (rank 2) at a 60 mm rise: `kCore 4/9`, `kCoreStable 4/9`,
`kExt 4/14`, `kExtStable 4/14`, `ceilingCore 5/9`, `maxTq 0.6405`
(`results/r6_judge-results.json` → `phaseG["best_r3_vault_60mm.json"]`).

The record is rank 1. Run snippet 2 again with `../climb/best_r6_ceilvaultC_60mm.json` and
expect `kCore 5/9`, `kCoreStable 5/9`, `kExt 5/14`, `kExtStable 5/14`, `ceilingCore 5/9`,
`maxTq 0.6405` (`results/r6_judge-results.json` → `phaseG["best_r6_ceilvaultC_60mm.json"]`).
Both reproductions took about a minute of wall clock from a fresh clone. `intents/X.json` in this
package is byte-identical to `climb/X.json` in the repository (see `MANIFEST.json`); the scorer
takes either path.

Pass `{ rise, core: true }` to `scoreRobust` for the 9 core cells only. `ceilingCore` is not a
field of `scoreRobust`'s return; it is the count above, and `harness/audit_r6.mjs` line 475
computes it exactly that way.

One cell costs about 0.712 s of wall clock (`results/r4_famB-results.json` → `secPerCell`), so
the 9-cell core grid is roughly 6.4 s and the 14-cell grid roughly 10.0 s per intent.

## Reproduce in the app

From Microduck Studio build 45 or later (TestFlight: <https://testflight.apple.com/join/S36AnsKr>;
source: github.com/craigm26/duck-studio):

1. **Studio → Measure → Stairs Challenge** (or Behaviours → the discover section → Stairs
   Challenge). The leaderboard is bundled: the same nineteen intent files as `intents/`, byte for
   byte, plus the two controls.
2. Pick a bench. **This iPhone** is always there — the phone's own bench carries the same
   `climb_score.mjs`, `stairs.js` and plant as the harness — or a Pi bench running the current
   duck-sounds. The screen asks the bench for its grid first; a bench without `/climb` says so
   and what to update.
3. Open a row, choose the rise (60 mm is the default) and tap **Score on this bench**. The app
   sends the fourteen cells one request each and draws the kit's verdict: `k of 9 stable`,
   the ceiling, and whether that matches, beats or misses the published row. The number is the
   audit's number: `sim/climb_parity.mjs` holds `/climb` exact against `robust.scoreRobust` on
   every cell, and the app only aggregates.
4. **Open in the editor** turns the move into a Studio motion. Change any keyframe's servo
   values there, come back, and tap **Score your edited version**. The screen says whether the
   edit scored better, the same or worse than what it started from. Keep what helps, put back
   what does not. There is no reward model in that loop: you are the judge and the bench is the
   measurement.
5. **Submit** appears once a comparable score exists: it writes one file (the intent, all
   fourteen per-cell answers unrounded, the plant digest, the date, and how to re-score it),
   opens the pre-filled GitHub issue, and can commit the file to a dataset under your own
   Hugging Face account as an archive.

Scoring on the phone needs no account, no secret and no Pi.

## On a real Microduck

The app can play a challenge move on a bench, in physics. Playing a harness move on a physical
Microduck is not wired in build 45, there is no score off a bench, and nothing in this package
has been run on hardware. If you put one of these moves on a robot, do it with the same care as
any untested motion: several entries drive every actuator to its torque ceiling for over a
second, and the record plants the beak on the step and pivots the body over the head. Report
what happened in the GitHub issue; a real staircase is your measurement, not the harness's.

## How to submit

Open an issue on <https://github.com/craigm26/duck-sounds> titled **`Stairs challenge: <rise> mm`**
with the intent JSON attached (attached as a file, not pasted — the score is computed from the
file's bytes, and `intentHash` in `harness/robust.mjs` is what defines the vector's identity).
The `sha256` printed on the leaderboard is that `intentHash` — a digest of the normalised intent,
not of the file's bytes, so `sha256sum` on a file will not match it. Compute it from `sim/` with:

```bash
node --input-type=module -e '
  import { intentHash } from "../climb/robust.mjs"; import fs from "node:fs";
  console.log(intentHash(JSON.parse(fs.readFileSync(process.argv[1], "utf8"))));
' ../climb/best_r6_ceilvaultC_60mm.json
```

The audit re-scores the attached file and checks, in this order:

1. **Bounds.** `blend` in `[0.7, 2.4]` and `side` in `[-0.02, 0.09]`. A file outside them is not
   scored at all: `scoreRobust` returns `invalid: true`, `objective: -Infinity`, `k: 0`, after
   printing a refusal to both stdout and stderr (`harness/robust.mjs`, `shoutInvalid`). Two of
   the 64 published files are refused this way — `best_r3_cornerclimb_120mm.json` and
   `best_r3_cornerclimb2_120mm.json` (`results/r6_screen-results.json`).
2. **Spawn on the floor.** An intent may carry an explicit `spawn`, which is how the
   placed-duck control is authored. An entry that spawns anywhere but the floor is a control,
   not a climb, and is labelled that way (`results/r4_judge-results.json` → `phaseC`,
   `"startKind": "PLACED SPAWN — NOT A CLIMB"`).
3. **The 14-cell grid under both landings**, at the declared rise, reporting `kCore`,
   `kCoreStable`, `kExt`, `kExtStable` and `ceilingCore` for the `policy` tail and the `hold`
   tail.
4. **Torque ceiling.** No row may exceed `maxTq` 0.6405 N·m.
5. **Whole-episode penetration.** The most negative `mj_geomDistance` between any duck geom and
   any step geom at **any** control tick — settle, track and tail — is reported per cell, not
   just at the scored instant.
6. **The lateral gate**, over the whole episode: `|y − STAIR_Y| <= 0.17` at every tick, not only
   when scored.

An entry whose intent carries a `servo` block is labelled **ORACLE** and is ranked separately.
The servo law reads the tread's height and its front edge straight out of `data.qpos` and
`data.geom_xpos`; the shipped policy is fed a 61-number observation of which none is
exteroceptive (`results/r5_judge-results.json` → `observationSets`: `policyObs.n` 61,
`exteroceptive` false; `servoLawReads.n` 7, `exteroceptive` true). A servoed result is an upper
bound on what a landing law could do given perfect knowledge of the step, not a move the robot
could run.

## The intent JSON format

A saved intent is a single JSON object. Fields:

| field | type | meaning |
|---|---|---|
| `keyframes` | array | the move. Each element `{ "t": seconds, "pose": [14 joint targets, radians] }`, `t` strictly increasing. The pose is interpolated from HOME with a smoothstep. |
| `blend` | number | how strongly the authored pose overrides the standing policy's action. **Bounded `[0.7, 2.4]`.** |
| `side` | number | lateral spawn offset from `STAIR_Y`, metres. **Bounded `[-0.02, 0.09]`.** |
| `gap` | number | spawn distance in front of the first riser, metres |
| `approach` | number | metres walked toward the flight before the track starts |
| `isolate` | bool | build the flight with `isolateSteps` (the 2026-09-02 repair). Leave `true`. |
| `stepCount` | int | number of step blocks in the flight (4 in every published entry) |
| `spawn` | object | optional `{x, y, z}` override — a placed spawn, i.e. a control, not a climb |
| `event` | object | optional round-4 event-triggered tail (see `harness/rig3.mjs` imports and `climb/event.mjs`) |
| `servo` | object | optional round-5 servoed landing. **Makes the entry an ORACLE.** |
| `name`, `family`, `note`, `params`, `robust` | any | provenance; ignored by the scorer |

The 14 joint slots, in order, are the Microduck's joints minus `mouth`
(`harness/duckkit-constants.json`, filtered by `site/duckloop.mjs` line 8):
`left_hip_yaw`, `left_hip_roll`, `left_hip_pitch`, `left_knee`, `left_ankle`, `neck_pitch`,
`head_pitch`, `head_yaw`, `head_roll`, `right_hip_yaw`, `right_hip_roll`, `right_hip_pitch`,
`right_knee`, `right_ankle`.

### The optional `event` block (round 4)

```json
"event": {
  "type": "beak" | "pitch" | "trunkZ",
  "threshold": 0.004,
  "arm": 0.9,
  "fallback": 1.35,
  "delay": 0.0,
  "refX": 0.171,
  "clamp": 0.12,
  "post": [ { "dt": 0.41, "pose": [14 numbers], "adapt": [14 numbers] } ]
}
```

The tail of the track fires on a measured event instead of a clock. `e = clamp(refX − trunkX)`
at the firing tick; each post keyframe's target is `pose[k] + adapt[k] * e`. A file with no
`event` replays byte-identically to the pre-round-4 loop.

### The optional `servo` block (round 5/6) — ORACLE only

```json
"servo": {
  "at": 1.1484, "onEvent": false, "yawRoll": "hold",
  "zTarget": 0.115, "xTrunk": 0.16, "xFoot": 0.10, "fz": 0.015, "pitchRef": 0,
  "kHipZ": 1.5, "kHipPitch": 0.6, "kHipX": 1.2, "kHipTrunkX": 0,
  "kKneeZ": -2, "kKneeFz": -1.5, "kKneeX": 0,
  "kAnkPitch": 0.4, "kAnkFz": 1,
  "rate": 0.15, "span": 1.2, "sign": [1, -1], "tailTicks": 0
}
```

Once armed the ten leg slots leave the keyframe track and are commanded every 20 ms from
measured trunk height above the tread, trunk pitch, trunk x past the riser line and each foot's
x and z relative to the tread edge and top. Head and neck keep following the keyframes. Targets
are rate-limited and clamped to `[LO, HI]`; the actuator ceiling is untouched.
`tailTicks` (default 0) is how many of the 50 tail ticks the law keeps the legs for.

### A worked example

`intents/best_r3_vault_60mm.json`, the rank-2 beak-strut vault, abridged to its first keyframe
(the file has six; the last is at t 2.4311):

```jsonc
{
  "name": "beak_strut_vault_r3_60mm",
  "family": "A beak-strut vault (round 3)",
  "keyframes": [
    { "t": 0.3119,
      "pose": [0, -0.21255, -0.36327, 0.4539, 0.48228, -0.32872, -0.7469,
               0, 0, 0, -0.03795, 0.36327, -0.4539, -0.48228] }
    /* … four more … */
  ],
  "blend": 2.1572,
  "gap": 0.0187,
  "side": 0.0078,
  "approach": 0.1663,
  "isolate": true,
  "stepCount": 4
}
```

sha256 `4b9110c448ec45b7e9aa9e25e8720ab6e149562dbba2c00dda73f5c86aee8f15`. Scored at a 60 mm
rise: `kCore` 4/9, `kCoreStable` 4/9, `ceilingCore` 5/9, `maxTq` 0.6405 N·m
(`results/r6_judge-results.json` → `phaseG`).

## Caveats

- **The broken-flight history.** Everything measured before the `isolateSteps` repair (commit
  `279b016`, 2026-09-02) is void below a 180 mm rise. `results/rig3.r2.log` Phases D and E are
  the record of the fault; `results/rig3-results.r2.json` is what it produced. They are shipped
  as evidence, not as results.
- **Penetration.** Every clear passes *through* a step block on the way up. For the rank-2
  record the four clears' worst whole-episode interpenetrations are −8.52, −9.27, −8.57 and
  −8.88 mm; its worst single cell anywhere is −13.67 mm
  (`results/r5_judge-results.json` → `phaseN`, `results/r6_judge-results.json` → `phaseN`).
  This is soft contact in MuJoCo's solver, not a tunnel: the judge's threshold for a hard
  refusal is −15 mm, `clearsDeeperThanSoft` is 0 on every row and `cellsDeeperThanSoft` is 0 on
  every row of `results/r6_judge-results.json` → `phaseN`. Across the 340 rebuilt CEM moves the
  deepest single-cell reading anywhere is −17.28 mm, past that line, though no clear is deeper
  than it (`phaseN.deepestRebuiltSearchReading_mm`, `rebuiltSearchClearsDeeperThanSoft` 0). It is
  nevertheless a physical softness a real step does not have.
- **The oracle.** Servoed entries read the world. See *How to submit*. `results/r6_judge-results.json`
  → `oracleCaveat` states it in the judge's own words.
- **The 20 ms tick.** The move is authored and replayed at 50 Hz. One tick of shift is not a
  rounding error. Measured on a servoed oracle move, not on the record: shifting the round-5
  servoland launch by +1 control tick takes `kCore` from 3 to 1 and moves trunk x by 163.8 mm;
  −1 tick keeps `kCore` 3 but moves trunk x by 234.2 mm (`results/r5_judge-results.json` →
  `phaseF`). The record vector's own tick sensitivity is not measured in any results file. The
  clears are isolated points in time, not basins.
- **No hardware.** Nothing here has been run on a Microduck. See the top of this card.
- **`ceilingCore` at other rises.** The exhaustive ceiling screen was run at a 60 mm rise only
  (`results/r6_screen-results.json` → `rise_mm` 60). The 90, 120 and 180 mm rows carry `n/m`.

## Files in this package

```
README.md              this card
leaderboard.md         the leaderboard table alone, for editing
MANIFEST.json          sha256 and byte size of every file here
check_numbers.mjs      re-derives every number in this card from results/ and prints PASS/FAIL
hf_upload.sh           the exact Hugging Face upload commands
intents/               123 saved intent JSONs — every climb/best_*.json, climb/ctrl_*.json
                       and climb/r4_ctrl_on_tread_*.json, filenames unchanged
results/               31 result files and logs — every file this card cites
harness/               rig3.mjs, robust.mjs, servo.mjs, audit_r6.mjs, stairs.js,
                       scene_physics.xml, duckkit-constants.json + a README.
                       A SNAPSHOT FOR READING. The runnable harness is the GitHub repo.
```

## Provenance

Six rounds, 2026-09-01 to 2026-09-02, in `github.com/craigm26/duck-sounds` under `climb/`. Every
published claim was re-scored from its saved file by an adversarial audit written after the
round it audits.

| round | date | what it searched | its audit |
|---|---|---|---|
| 1 | 2026-09-01 | first parameterisations, six rises | — |
| 2 | 2026-09-01/02 | vault, block-climb, corner-stem; the instrument was rebuilt and the flight repaired | `climb/audit_r2.mjs` → `results/audit_r2-results.json` |
| 3 | 2026-09-02 | beak-strut vault, land-vault, corner-climb; the shared 9-cell scorer | `climb/audit_r3.mjs` → `results/audit_r3-results.json` |
| 4 | 2026-09-02 | family A (event landing), family B (two-beat, 80–120 mm) | `climb/audit_r4.mjs` → `results/r4_audit-results.json`; judge → `results/r4_judge-results.json` |
| 5 | 2026-09-02 | the servoed landing and its inertness proof | judge → `results/r5_judge-results.json` |
| 6 | 2026-09-02 | the ceiling search, the tail measurement, the limits trace | `harness/audit_r6.mjs` → `results/r6_judge-results.json` |

Search effort, as the results files themselves count it (an "eval" in rounds 3–6 is a whole
9- or 14-cell grid, so the episode count is several times larger):

| round | file | counter |
|---|---|---|
| 1 | `results/search_0-results.json` | 1,822 evals |
| 1 | `results/search_1-results.json` | 2,072 evals (superseded by `search_2` leg 1, same numbers) |
| 2 | `results/search_2-results.json` | 5,202 evals |
| 2 | `results/search_3-results.json` | 4,974 evals (`totalEvals`) |
| 2 | `results/search_block-results.json` | 4,410 episodes |
| 2 | `results/vault-results.json` | 5,265 episodes (`totals.episodes`) |
| 3 | `results/r3_vault-results.json` | 3,411 evals |
| 3 | `results/r3_landvault-results.json` | 762 evals |
| 3 | `results/r3_cornerclimb-results.json` | 5,582 episodes (2,753 + 2,829) |
| 3 | `results/r3_cornerclimb2-results.json` | 2,220 episodes |
| 4 | `results/r4_famA-results.json` | 416 evals |
| 4 | `results/r4_famB-results.json` | 3,186 evals |
| 5 | `results/r5_servo_search-results.json` | 786 stage-1 candidates, 24 stage-2 |
| 5 | `results/r5_servoland-results.json` | 234 full-grid evaluations, 317 screened |
| 6 | `results/r6_ceiling-results.json` | 1,614 evals, 341 full, 340 distinct moves |

The round-6 judge's own summary, verbatim (`results/r6_judge-results.json` → `honestClaim`):

> "After six rounds and 394 distinct scored launches, the best whole-body move gets the duck onto
> a 60 mm step and leaves it standing in 5 of the 9 robustness cells, no move has ever cleared
> more than 5, and the limit is not tuning: all 14 servos are already pinned at their 0.6405 N.m
> torque limit in every cell of every move."

The card's own numbers are narrower than that last clause: 60 of the 62 scorable published moves
saturate (the two block-drag files never push), and "saturate" means at least one of the 14
actuators at the ceiling for 73.6% of the push-off in the cells that clear the bar, not all 14
pinned throughout (`results/r6_judge-results.json` → `phaseT`;
`results/r6_limits_table-results.json` → `pooled.over95`).

## License

The data in this package — every intent under `intents/`, every results file under
`results/`, `leaderboard.md`, `MANIFEST.json` and this card — is published under
**CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/). Use it, remix it, redistribute
it; credit "Microduck Stairs Challenge, craigm26" and link back here. The harness that
scores it — `harness/` here, and the runnable copy at github.com/craigm26/duck-sounds — is
**Apache-2.0**, as are duck-studio (Microduck Studio) and duckkit. Pollen Robotics' policies
(`alpha_*.onnx`, `ball_kick_*.onnx`) and plant come from their repositories under their own
terms, which this package does not grant.
