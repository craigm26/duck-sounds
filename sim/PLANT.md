# Which plant is canon

Settled 2026-08-30, because two files called `scene.mjb` existed with the same size and
different bytes, and every measurement in this family is attributed to "the canon plant".

## `sim/scene.mjb` is canon. `site/scene.mjb` is a stale copy.

```
sim/scene.mjb    3f8c9ab9b409ba74c73c30179d5f7c12b025f631693f9eec78d80dca242547be
site/scene.mjb   9d75c2379cefbd3d194b1cbd3ae25579867d457f4cf1f4d40d6d8a8c406b5d6f
```

Both are tracked and neither is dirty, so this is a real divergence in history rather than a
working-copy accident.

**The proof is the corpus, not the dates.** `duckkit/Sources/DuckKit/Resources/duck-intent-clips.json`
is byte-identical to `sim/duck-intent-clips.json` —
`85e4d2fdf3eea5527c786b31a4afde7c6e10b9d4ecbac0e81eb5b3b7005a4be3` — and `site/` does not carry
that file at all. Every recorded clip DuckKit ships, and therefore every clip Duck Studio and
OpenCastor draw, was recorded against the `sim/` plant.

**The history agrees.** `site/scene.mjb` was last written by `aac722a` ("A real staircase, and the
stair numbers were wrong"); `sim/scene.mjb` by `bb79f54` ("The plant now carries training's own
solver, friction, and torque ceiling"). `git merge-base --is-ancestor aac722a bb79f54` returns
true, so the site copy is strictly older. `bb79f54` re-recorded `duck-intent-clips.json` and
`intent-success.json` in `sim/` in the same commit and did not touch `site/`.

**What `bb79f54` changed, which is why the difference matters rather than being cosmetic.** It
moved the plant onto training's own numbers: solver iterations 10/20 where MuJoCo's defaults 100/50
had been running a solver four times stiffer than training's; floor friction 1.0/0.005/0.0001;
feet at contact priority 1 so foot-floor contacts take the foot's parameters; actuator forcerange
±0.6405 N⋅m — the XL330's kt × 1.75 A current limit, not the 0.96 stall figure; and training's
stiff joint-friction `solref` so dry friction is not smeared.

That ±0.6405 N⋅m is the figure Duck Studio prints on its "Where the numbers come from" panel. It
comes from `sim/build_physics_only.py:175-181`. It is not in `site/`.

## The consequence nobody had written down

`site/sim.js:26` loads `scene.mjb` from `site/`, so **the web sim runs an older plant than every
recorded number in the family**. A figure read off the web sim and a figure read off the bench are
not comparable, and until now nothing said so. Either rebuild `site/scene.mjb` from
`build_physics_only.py` and re-record whatever `site/` measures, or accept the divergence and say
so wherever the site quotes a number.

## If this is ever in doubt again

Do not compare dates — the mtimes are checkout artifacts. Compare
`duckkit/Sources/DuckKit/Resources/duck-intent-clips.json` against each candidate's sibling
`duck-intent-clips.json`. The plant that recorded the corpus DuckKit ships is the canon one, by
definition, because that corpus is what every app in the family draws.
