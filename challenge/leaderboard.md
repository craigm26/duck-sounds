# Microduck Stairs Challenge — leaderboard

Simulation only, never on hardware. This is the leaderboard table from `README.md`, alone, so it
can be edited without touching the card. Keep the two in step; `check_numbers.mjs` re-derives
every value in it from `results/`.

One row per **distinct vector** (sha256 of the normalised intent). Where the same vector was
published under more than one rise label, the extra rise rows carry no rank and say so. The rank
column orders by `kCoreStable` only and is not comparable across rises: a lower rise is a strictly
easier task. The `sha256` is `intentHash` over the normalised intent (see the card), not the file
digest. The `scored` column is the date of the judge run that produced the row, not the file's
creation date (the round-2 files date from 2026-09-01). The table lists vectors that exist as saved
intent files; three further vectors in the rebuilt CEM corpus (`b3f06fb9e903`, `56b676fab298`,
`4886bd27f9a3`, `results/r6_judge-results.json` → `phaseX`) reach `kCoreStable` 4 at 60 mm but were
never published as files.
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
