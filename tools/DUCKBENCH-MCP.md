# duckbench as MCP tools

Physics for a Microduck, available to Claude with a terminal. `tools/duckbench-mcp.mjs`
wraps the running bench's HTTP API as MCP tools over stdio — no dependencies, no
second copy of MuJoCo, and nothing to install.

## Why it wraps rather than embeds

duckbench is already a server with a physics lane and a world loaded. A second
process with its own MuJoCo would be a second world, disagreeing with the first
about what happened. So this is a thin client: it can point at any bench through
`DUCKBENCH_URL`, and starting it costs nothing because the bench is already up.

## Add it

```
claude mcp add duckbench -- node /home/craigm26/projects/duck-sounds/tools/duckbench-mcp.mjs
```

Pointing it somewhere else — the desktop bench, or a bench with a token:

```
claude mcp add duckbench \
  -e DUCKBENCH_URL=http://100.95.79.116:8770 \
  -e DUCKBENCH_TOKEN=the-same-string-the-bench-started-with \
  -- node /home/craigm26/projects/duck-sounds/tools/duckbench-mcp.mjs
```

Defaults to `http://127.0.0.1:8770`. On this Pi the bench runs as a systemd user
unit and is on the tailnet at `100.122.199.6:8770` — see `sim/BENCH-SETUP.md`.

## The tools

| tool | what it answers |
|---|---|
| `bench_health` | which world is loaded, its digest, every policy it can run |
| `bench_record` | run one policy: how far it travelled, how far it stepped, did it stay up |
| `bench_measure` | how often it met the criterion — **and how far it got** |
| `bench_perform` | run an AUTHORED motion (keyframes) in real physics, several times |
| `bench_upload_policy` | put a local `.onnx` on the bench and get the name to run it by |
| `bench_state` / `bench_steer` / `bench_stop` | the live duck, driven a step at a time |

## The two rules it enforces on itself

**A success rate is not an answer.** The bench's criterion is *"ends standing,
trunk at least 100 mm up"*, and a duck standing perfectly still passes it every
time. Measured here: `alpha_walking` averaged 75/25 with `BEST_alpha_stand`
scores **16 of 16 while travelling two millimetres**, where the walking policy it
came from covers 1.207 m. So `bench_measure` returns the distance beside the
count by default and says which case it is looking at:

```json
{ "achieves": 4, "rollouts": 4,
  "criterion": "ends standing, trunk at least 100 mm up",
  "travelled": 0.0015, "path": 0.0096,
  "note": "It barely moved. If this policy is meant to walk, the rate above is
           measuring it standing still — that passes the criterion and is not
           the behaviour." }
```

Turning that off costs one rollout and leaves a number that cannot tell a walk
from a duck standing still. See `sim/BLEND-MEASUREMENT.md`.

**Name the plant.** Every result carries `plantName` and `plantDigest`. A number
measured against a different world is not comparable to one measured here,
however alike the filenames look.

## Commands, and the threshold that catches people out

`alpha_walking` travels **7 mm in six seconds at vx 0.15** — below the gait's
threshold, so it just stands, which looks exactly like a broken policy and is
not one. At 0.3 it covers 0.681 m; at 0.5, 1.207 m. So `schedule` defaults to
`"walk"` — forward at vx 0.5 after a half-second settle for the drop bounce.
`"still"` is the other shorthand; a raw `[[atSeconds, {vx, vy, vyaw}], …]` list
also works.

## A refusal is an answer

The bench replies 200 with an `error` key when it will not do something, and the
reason is the useful part. Those come back as `isError` results with the text
intact rather than as transport faults, because *"that file did not load as a
policy: Missing opset"* is exactly what tells you your ONNX writer is wrong.
onnxruntime catches things a hand-written parser does not — a missing opset,
nodes that declare no outputs, an attribute with no type — all three of which
this project shipped past its own reader.

## Verified

Driven over stdio by a test client on 2026-08-31 against the live bench:
initialize (protocol 2025-06-18), `tools/list`, health, measure on
`alpha_walking` (4/4, travelled 1.2068) and on the 75/25 blend (4/4, travelled
0.0015), a local `.onnx` uploaded and run end to end, and both refusal paths —
an unknown policy and a missing file — returning readable answers.

## Note on uploaded policies

`bench_upload_policy` returns a name like `uploaded-eba0fbaa170c`, which works
until the bench restarts. After a restart the bench rebuilds its map by scanning
the directory, so the same policy is addressable as `uploaded-eba0fbaa170c.onnx`.
`bench_health` always lists the names that currently work.
