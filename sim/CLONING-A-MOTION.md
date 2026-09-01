# Turning an authored motion into a policy: what happened

Measured 2026-08-31 on this bench, plant `scene.mjb` (3f8c9ab9b409).

## Why anyone would

robotd runs an ONNX policy in its `[policy]` slot and nothing else. An authored
motion is keyframes — a function of *time*. A policy is a function of *state*,
closed-loop at 50 Hz. So a motion reaches a real Microduck only by being turned
into a policy, and that is the whole of the "last mile".

## The harness now exists and is correct

`POST /capture` runs an authored track and returns, for every control step, the
61-wide observation the network saw and the action a policy would have had to
output to produce that motion — `ctrl − reference`, taken after the clamp.

- 32 rollouts of a 1.7 s bow: **5440 pairs, 0 dropped, the source ran stood 32
  of 32.**
- `driver` lets a learner drive while the teacher labels (DAgger).
- `teacherShare` lets the teacher keep the wheel on most steps.
- `jitter` perturbs the commanded joints *after* labelling, so the next label is
  a recovery.

## Three bugs found in it, all fixed

1. **NaN travels as `null`.** `JSON.stringify` writes NaN and Infinity as
   `null`; a diverged step left as a hole a consumer reads back as NaN and
   trains on in silence. Every epoch after the first reported NaN loss.
2. **Finite is not plausible.** A diverged MuJoCo state yields enormous
   *doubles* — measured at 6.8e37 — which `Number.isFinite` accepts. The
   normaliser's deviation becomes 1e36 and every real observation flattens to
   zero. Steps outside ±1000 are now dropped and counted.
3. **The fed-back action was the wrong one.** `lastAction` is part of the
   observation. During capture the teacher was acting, so the action fed back
   carried no authored offset — while a policy trained on these labels outputs
   the offset-laden action and feeds *that* back. The clone met an unseen
   observation block on its first step.

## The result: it does not work, and the file is not why

Four variants were trained and measured: plain behaviour cloning, DAgger,
noise-injected demonstrations, and all of it with the feedback corrected.

| | |
|---|---|
| training fit | RMS **0.0063 rad** (0.36°) |
| the written ONNX, checked offline against the training labels | RMS **0.0062 rad** |
| running alone on the bench | neck to the **−1.920 rad** joint stop within 0.4 s |
| upright | **0 of 16** |

The offline check is the important one. Loaded through `DuckPolicy` and fed the
training observations, the written file answers with the training labels to
0.006 rad. **The fit is right, the writer is right, the reader is right.** It
diverges only when it is the one choosing the states — textbook compounding
error, now isolated rather than assumed.

## Why, and what it means

A clone of "standing policy + authored offset" has to reproduce **the balance
controller too**. That controller was trained with RL over a huge range of
states; 5440 samples from one 1.7-second window is nowhere near enough to
relearn it. The bow is the easy part. The balance underneath it is what fails.

So the obstacle is not a file format, and it is not this pipeline:

- The robot **already has** a balance policy.
- What a gesture needs is that policy **plus an offset** — exactly what
  `/perform` does here, and exactly what robotd cannot do, because it runs one
  network and exposes no way to add to its output.
- Collapsing the two into one network means relearning balance from scratch,
  which is RL-scale work, not a demonstration of one bow.

**The honest conclusion: the last mile is blocked at robotd's interface.** The
useful paths are (a) upstream gaining a way to apply an offset or a trajectory,
or (b) training a gesture policy properly with RL rather than cloning one. This
harness is what (b) would be built on, and it is now correct.
