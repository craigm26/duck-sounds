# What averaging two trained policies actually does

Measured 2026-08-31 on this bench. `duck-bench/3`, plant `scene.mjb`
(sha256 `3f8c9ab9b409ba74c73c30179d5f7c12b025f631693f9eec78d80dca242547be`),
50 Hz, 6-second rollouts, commanded forward at `vx = 0.5` from t = 0.5 s.

`alpha_walking.onnx` and `BEST_alpha_stand.onnx` averaged elementwise —
weights, biases and the observation normaliser — at ratio `t` toward standing,
written out by `DuckPolicyWriter` and uploaded to `/upload`.

| t | travel | path | end height | upright |
|---|--------|------|-----------|---------|
| walking, unmodified | 1.207 m | 1.356 m | 0.115 m | yes |
| 0.00 (writer control) | **1.207 m** | **1.356 m** | **0.115 m** | yes |
| 0.25 | 0.176 m | 0.234 m | 0.044 m | **no — fell** |
| 0.50 | 0.156 m | 0.234 m | 0.040 m | **no — fell** |
| 0.75 | 0.002 m | 0.010 m | 0.117 m | yes |
| 1.00 (writer control) | 0.001 m | 0.003 m | 0.117 m | yes |
| standing, unmodified | 0.001 m | 0.003 m | 0.117 m | yes |

## Two things this settles

**The writer is correct, behaviourally and not just structurally.** Row `t=0.00`
is `alpha_walking` decoded by DuckKit and re-encoded by `DuckPolicyWriter`,
nothing else. It reproduces the original's six-second closed-loop walk to every
digit printed here. A walk is a chaotic feedback loop over 300 control steps —
any discrepancy in any of 197,774 parameters diverges visibly. `t=1.00` does the
same for standing. "It loads" was never the claim worth making; this is.

Getting there took three refusals from onnxruntime that this package's own
reader could not produce, because the reader does not look at the fields that
were wrong:

1. *"Missing opset in the model"* — no `opset_import`. The reader walks straight
   to the graph and never reads the ModelProto header.
2. *"Graph output (actions) does not exist in the graph"* — nodes declared their
   inputs and no outputs. The reader infers the chain from position, so an
   unwired graph round-tripped through it perfectly.
3. *"Field 'type' of 'attr' is required but missing"* — the `transB` attribute
   did not declare `AttributeType.INT`. The reader reads the value and ignores
   the tag.

Each is now asserted in `DuckPolicyWriterTests` by walking the emitted protobuf
rather than by reloading it — reloading is precisely the test that passed while
the file was broken. All three assertions were mutation-checked: reverting each
fix fails the suite.

**Blending separately trained policies does not preserve behaviour.** At 25% and
50% the duck falls over. At 75% it stays up and travels two millimetres — it has
stopped walking and become the standing policy.

## The trap in the 75% row

The bench's own success criterion is *"ends standing, trunk at least 100 mm up"*,
and the 75% blend scores a perfect **16 of 16** against it while doing nothing at
all. Standing still passes an uprightness test trivially.

So a success rate against this criterion **cannot distinguish a preserved
behaviour from a lost one**, and anything quoting that rate without a distance
alongside it is reporting a collapse as a triumph. `PolicyBlend.measured` was
written that way and would have called this row *"a genuinely surprising result
worth keeping"*; it now takes a `Behaviour` carrying both the count and the
travel, plus the travel of the liveliest ingredient as a yardstick, and says
*"it stayed on its feet and stopped doing the thing"* instead.

## Reproducing it

The bench needs a command of at least `vx ≈ 0.3` before `alpha_walking` breaks
into a gait — at `vx = 0.15` it stands, travelling 7 mm in six seconds, which
looks exactly like a broken policy and is not one.

```
curl -s -X POST localhost:8770/record -H 'content-type: application/json' \
  -d '{"policy":"alpha_walking.onnx","seconds":6,
       "schedule":[[0,{"vx":0,"vy":0,"vyaw":0}],[0.5,{"vx":0.5,"vy":0,"vyaw":0}]]}'
```

Travel is the planar distance between the first and last root position in
`roots`; path is the summed step-to-step distance. The gap between them is how
much of the movement was not progress.
