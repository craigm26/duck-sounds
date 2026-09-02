# Microduck Ball Challenge — leaderboard

Simulation only, never on hardware. These are the leaderboard tables from `README.md`, alone, so
they can be edited without touching the card. Keep the two in step; `check_numbers.mjs` re-derives
every value in them from `results/`.

The criterion, verbatim from `harness/chase_score.mjs` line 85 and
`results/chase_controls-results.json` → `criterion`:

> **chased: the duck touched the ball — any duck geometry within 3 mm of it at any tick — and the
> ball finished at least 100 mm further along the duck's initial heading than it started, and the
> duck was still upright at the end of the episode. stable: chased, and upright for at least 45 of
> the 50 tail ticks.**

`sha256` is the entrant hash — a digest of the normalised entrant with `name` and `note`
excluded, not of the file's bytes. `centre travel` is `ballTravel_mm` at the centre cell
(bearing 0, range 0.70, nominal plant). Rows from `results/chase_controls-results.json` →
`leaderboard`.

**There is no bar yet.** All four rows are reference controls, ranked only so a reader can see
which is furthest along; none of them was authored to chase anything.

| # | sha256 | entrant | kind | seconds | chased / 9 core | stable / 9 core | ext / 5 | touched / 14 | centre travel | scored |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `a0bbbbb98acb` | `entrants/ctrl_alpha_walking.json` | policy `alpha_walking.onnx` | 4 | **4** / 9 | **4** / 9 | 1 / 5 | 5 | 0.0 mm | 2026-09-02 |
| — | `bc77453e40c6` | `entrants/ctrl_do_nothing.json` | move | 5 | 0 / 9 | 0 / 9 | 0 / 5 | 0 | 0.0 mm | 2026-09-02 |
| — | `7e44b5a781fc` | `entrants/ctrl_ball_kick_left.json` | policy `ball_kick_left.onnx` | 5 | 0 / 9 | 0 / 9 | 0 / 5 | 0 | 0.0 mm | 2026-09-02 |
| — | `f8d4e8bfd2b7` | `entrants/ctrl_ball_kick_right.json` | policy `ball_kick_right.onnx` | 5 | 0 / 9 | 0 / 9 | 0 / 5 | 0 | 0.0 mm | 2026-09-02 |

Full hashes:

```
ctrl_alpha_walking    a0bbbbb98acb7fc5bc1d035527c2c7b153df1c3555db79b9c12e4f446d49d6a5
ctrl_do_nothing       bc77453e40c677db4073a350da5a43d645676d77e1252f51bbf6544be54ca187
ctrl_ball_kick_left   7e44b5a781fc6763042a43065598424ea945f3bc8956bd0f1127aca4ec81b6e9
ctrl_ball_kick_right  f8d4e8bfd2b789668cdf58e7683100d04cf48af2d1fe746d495fc4f697e03ffe
```

**The centre-cell travel column is 0.0 mm on every row.** Nothing bundled moves the ball from the
one cell every entrant runs.

## Every cell, for the one control that scores

`ctrl_alpha_walking`, from `results/chase_controls-results.json` → `entrants[3].verdicts`.
Rounded to 2 dp for millimetres and 4 dp for speed, as that file stores them; the unrounded values
are the same entrant's aggregate fields.

| bearing | range | plant | chased | touched | travel mm | net mm | closest mm | final mm | peak m/s |
|---|---|---|---|---|---|---|---|---|---|
| −20° | 0.45 | nominal | **yes** | yes | 582.80 | 650.95 | −3.14 | 216.34 | 0.6140 |
| 0° | 0.45 | nominal | **yes** | yes | 641.27 | 743.77 | −2.14 | 680.51 | 0.6306 |
| +20° | 0.45 | nominal | no | no | 0.00 | 0.00 | 110.69 | 860.02 | 0.0000 |
| −20° | 0.70 | nominal | **yes** | yes | 135.37 | 495.53 | −3.55 | 500.17 | 0.5372 |
| 0° | 0.70 | nominal | no | no | 0.00 | 0.00 | 24.38 | 544.25 | 0.0000 |
| +20° | 0.70 | nominal | no | no | 0.00 | 0.00 | 242.45 | 737.43 | 0.0000 |
| −20° | 0.95 | nominal | **yes** | yes | 233.05 | 406.23 | −5.06 | 316.71 | 0.5950 |
| 0° | 0.95 | nominal | no | no | 0.00 | 0.00 | 113.87 | 370.08 | 0.0000 |
| +20° | 0.95 | nominal | no | no | 0.00 | 0.00 | 396.76 | 687.73 | 0.0000 |
| 0° | 0.70 | ext, drop 0.130 / ×0.7 | no | no | 0.00 | 0.00 | 64.93 | 331.77 | 0.0000 |
| 0° | 0.70 | ext, drop 0.125 / ×1.3 | **yes** | yes | 465.39 | 559.72 | −4.39 | 279.20 | 0.6198 |
| −40° | 0.70 | ext, nominal | no | no | 0.00 | 0.00 | 210.87 | 621.99 | 0.0000 |
| +40° | 0.70 | ext, nominal | no | no | 0.00 | 0.00 | 424.57 | 977.05 | 0.0000 |
| 0° | 1.20 | ext, nominal | no | no | 0.00 | 0.00 | 196.80 | 320.47 | 0.0000 |

It passes the whole bearing −20° column and misses two of the three dead-ahead cells, which is the
opposite of what was predicted. `results/chase_drift-results.json` measures why: the open-loop
gait drifts **15.893°** to the duck's right and covers **1.3145 m** in the 4 s it was commanded to
walk 2.0 m. Open-loop forward walking does not solve "the ball is straight ahead"; it solves "the
ball happens to be where this gait drifts."

Every genuinely off-bearing cell — both ±40° cells and all three +20° cells — is unclaimed.

## The three that score nothing

From `results/chase_controls-results.json` → `entrants[0..2]`. All three: 0 of 14 chased,
0 of 14 touched, `ballTravel_mm` 0.0 and `ballPeakSpeed_mps` 0 in every cell, upright at the end
of all fourteen with 50 of 50 tail ticks.

| entrant | closest approach, min | closest approach, mean |
|---|---|---|
| `ctrl_do_nothing` | 344.0166288764408 mm | 620.5576745621695 mm |
| `ctrl_ball_kick_left` | 301.2073777841115 mm | 594.0542846572323 mm |
| `ctrl_ball_kick_right` | 308.7546981502833 mm | 606.3468899818689 mm |

`ctrl_do_nothing`'s 0 of 14 is a requirement, not a result: a criterion that row passes is not a
chasing test, and `chase/chase_parity.mjs` checks it before it believes anything else
(`results/chase_parity.log`).

The two kick policies' 0 of 14 is the measurement this challenge exists to publish: Pollen's ball
policies are blind to the ball by design and were trained at 90 mm, and the nearest cell here is
five times that. Reaching the ball is the unsolved half, not kicking it.
