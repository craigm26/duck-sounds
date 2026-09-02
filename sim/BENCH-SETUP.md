# Getting Duck Studio talking to a bench

WHY THIS PAGE CHANGED. It used to open: *"The phone can read a policy,
fingerprint it and blend one. It cannot **run** one — an iPhone has no physics
engine."* The second half of that is false, and it had been repeated in this
file, in `duckbench.mjs`, in the bundle's README and in shipped UI copy until it
read like a fact about hardware. It is a fact about a build. MuJoCo ships a
WebAssembly target — the `.wasm` in `site/vendor` is byte-identical to the one
this bench runs — and a policy is four matrix multiplies over 197,774 floats.
Both run in a browser.

Measured 2026-09-01, in Chromium on the Pi that serves this repo, through the
real browser shell: **0.35 ms a `mj_step`, 0.50 ms a policy forward pass, 2.7 to 3.5 ms
for a whole control tick against a 20 ms budget — 5.7 to 7.4× real time, two runs on
the same machine a third apart.** The same
250-tick walk gave the same trunk position as the desk bench to the bench's own
1e-4 quantum. `sim/phonebench.html`, deployed at `/phonebench/`, is the page
that asks the same questions of an actual iPhone; until somebody opens it there,
**iOS Safari is still unmeasured** and nothing here claims otherwise.

So there are now two benches, and the reason to use the desk one is no longer
that it is the only one that can run physics:

| | desk bench (this page) | phone bench (`duckbench-web.mjs`) |
|---|---|---|
| physics | MuJoCo 3.1.16 WASM | the same file, byte for byte |
| inference | onnxruntime-node | `policyforward.mjs` over duckkit's canonical bytes |
| `/upload` | yes, ONNX bytes | no — canonical parameter bytes only, and not wired |
| policies | every `.onnx` in `sim/` | whatever the app bundles |
| speed on a Pi 5 | 2.3 ms a tick | 2.7–3.5 ms a tick (Chromium, two runs) |
| trajectories | **not interchangeable** — see below | |

**A clip recorded on one is not a clip recorded on the other.** The two inference
paths agree to 3.5e-6 per action (`node sim/policy_parity.mjs`, 42,000
comparisons), which is nothing in one tick and is not nothing in a closed loop:
measured over the parity script, the two stay identical for the first 50 ticks
and are then 2 mm apart at tick 100, 9 mm at 150, 19 mm at 200 and 32 mm at 250.
Frame-for-frame comparison across the two is not meaningful. `/measure`, which
counts outcomes over randomised drops, is.

## Right now, with nothing to install

A bench is already running on the Pi and is reachable over Tailscale. In
Duck Studio → Policies → **Run on your network**, type:

```
100.122.199.6:8770
```

No token. That is the whole setup. It works from the phone on cellular, not
just on the house Wi-Fi, because it is a tailnet address rather than a 192.168
one.

It runs as a systemd **user** unit (`duckbench.service`) with lingering enabled,
so it comes back after a reboot and restarts if it dies:

```
systemctl --user status duckbench
journalctl --user -u duckbench -f
```

> The unit's `ExecStart` is the full nvm path to node. systemd has no login
> shell, so `node` is not on its PATH and this box has no `/usr/bin/node`; the
> unit crash-looped 34 times before anything said why. If node is upgraded
> through nvm, that path changes and the unit must be updated.

## On the Windows box (100.95.79.116)

Worth doing when the Pi is busy — it is a four-core Pi sharing time with
everything else. **Not** for the GPU: the bench's MuJoCo is the WASM build and
runs on the CPU. The GPU matters for *training*, which is a different program.

The Windows machine has no SSH server, so this part has to be done at that
keyboard. In PowerShell:

```powershell
# 1. Fetch the bundle from the Pi (scp ships with Windows 10+)
scp craigm26@100.122.199.6:duckbench-bundle.zip $HOME\Downloads\
Expand-Archive $HOME\Downloads\duckbench-bundle.zip $HOME\duckbench -Force

# 2. Install Node.js LTS if it is not there: https://nodejs.org

# 3. Start it
cd $HOME\duckbench\duckbench-bundle
.\start.ps1
```

The first start runs `npm install` for MuJoCo and onnxruntime and takes a
couple of minutes. Every start after that is seconds. `start.ps1` prints the
exact address to type into the app — it reads it from `tailscale ip -4`, so it
is right even if the tailnet address ever changes.

**Leave the window open.** Closing it stops the bench, and the app's symptom
for that is "nothing answered", which looks identical to a wrong address.

### If Windows Firewall blocks it

Tailscale traffic usually passes, but if the phone cannot reach it:

```powershell
New-NetFirewallRule -DisplayName "Duck bench" -Direction Inbound `
  -LocalPort 8770 -Protocol TCP -Action Allow
```

### Putting a password on it

```powershell
$env:DUCKBENCH_TOKEN = "something-long"; .\start.ps1
```

and the same string in the app's token field. Without one the bench is open to
anything on your tailnet. A tailnet is not the public internet, so this is a
real choice rather than an emergency — but on a shared machine, set one.

## In the app

**Policies → Run on your network → Set one up** has the same steps, and a
**Test this address** button that says which of the seven failure modes it hit
— nothing typed, an address it will not dial, nothing listening, a token
wanted, something else on the port, a bench that refused, or connected. Each
one names the next action. See `BenchSetup.diagnose`.

The address and token are shared across every bench screen, so they are typed
once.

## The gates

Nothing in this directory is believed because it reads correctly.

```
node sim/bench_parity.mjs --mode core --against parity/core-v5-performfix.json \
                          --allow /host/tickMillis   # 60 requests, 5,751 leaves
node sim/policy_parity.mjs                            # onnxruntime vs policyforward
node sim/physics_parity.mjs --engine policyforward    # the number the phone must match

Run `bench_parity` and `physics_parity` on a bench with NO world set (`systemctl --user
restart duckbench` gives you one): both drive the LIVE lane, and a standing `/world` moves
them by design. `climb_parity`, `chase_parity` and `tune_parity` use their own mjData and are
unmoved by a world; `world_parity` phase 4 is the proof. `/health` deliberately does not
advertise `/world` (its `transport.endpoints` still lists the two older world-shaping verbs):
adding a field would move every leaf of `bench_parity`'s baseline, so a client learns the
route the way it learns `/tune` and `/climb` — by sending one and reading the answer.
node sim/tune_parity.mjs                              # /tune's reward and fold, vs Swift
node sim/world_parity.mjs                             # /world, and that it leaks into nothing
node sim/webcheck.mjs                                 # the browser shell, over real HTTP
node sim/pagecheck.mjs                                # the probe page, in Chromium
```

`tune_parity.mjs` is the gate for `/tune`, and it is the only one here that
compares this repository against **another language**. `/tune` took two
definitions out of duck-studio and rewrote them in JavaScript — what a per-joint
gain and trim mean (`DuckPolicyWriter.folding`) and what each of Pollen's six
reward terms is (`RunMetrics`) — and a second transcription of either is how a
search comes to optimise a hill the app cannot see, with both numbers looking
plausible. So:

* **the reward** is checked on a shared fixture. `POST /tune` with
  `"trace": true` answers with the first drop's control ticks; fifty of them
  live in duck-studio at `StudioKit/Tests/StudioKitTests/Fixtures/tune/trace.json`
  with the six values the bench computed. `BenchTuneParityTests` scores those
  ticks through `RunMetrics` and this script scores them through
  `duckbench-core.mjs`'s own `rewardSums`. Measured 2026-09-02, the two agree
  **exactly** — every one of the six doubles is bit-identical, and the Swift
  assertion still passes with the tolerance set to 0. It is asserted at 1e-9
  rather than 0 because five of the six terms end in `exp` and no libm is
  obliged to round it the way glibc does.
* **the fold** is checked as bytes. `swift test` writes
  `StudioKit/.build/fold-fixture/{base,folded}.bin` — a real policy folded by
  duckkit's own writer at a fixed gain and trim — and this script folds the same
  base with `policyforward.mjs`'s `foldParameters`. All 791,584 bytes match.
  It has been watched to fail: dropping the `Math.fround` that rounds the gain
  to binary32 before multiplying, which is what Swift's `Float(gain[j])` does,
  moves 381 bytes and 1.49e-8 of one weight — invisible to any comparison that
  settles for 1e-6.

`swift test` has to run before this script, because the fold half of it is
consuming what that test writes. With the fixture missing, the script says so
and **fails** rather than reporting a pass it did not earn.

### `/world`, and the fourteen blocks nobody had parked

`POST /world` gives the LIVE world a different room: up to fourteen step blocks
at a chosen x and tread height, the ball somewhere else, a graspable moved.
`GET /world` reads back what is actually standing — out of `qpos`, not out of
the request — together with the bank's fixed facts, the arena's four walls and
an `unexpressed` list of everything the request asked for that this plant cannot
express. It is the only endpoint here that is about the room rather than about a
duck in it, and an older bench 404s it.

**Nothing parks the step bank in the live world.** The climb rig and the chase
rig each build their own `MjData` and park their bank in a `finally`; the live
world has never had one. So `live.world` boots with all fourteen 200 kg blocks
stacked at (0, 1.305, 0), colliding on every `/intent`, and **every live number
this bench has ever published was measured with them there.** They do not stay
stacked either: they are 200 kg bodies on frictionless slides with nothing
pinning them, so the solver throws them apart, and twenty-five ticks of boot
settle is enough to leave the fourteen in a column from a tread height of
**−11.715 m to +1.618 m**, all still at x = 0. That is what `GET /world` on a
fresh bench answers, and it is the first time anything in this repo has looked.
That is why a bare floor is an explicit request (`{"clear": true}` or
`{"steps": []}`) and never a default: parking the bank at boot would move
`physics_parity`'s trunk and every leaf of `bench_parity`'s baseline. A first
`POST /world` that says nothing about the bank is **refused**, with that
paragraph as the refusal, rather than quietly choosing for the caller.

`/reset` re-lays a standing world, because `mj_resetData` wipes it — the flight,
the props and the ball all go back to `qpos0` otherwise, and a world that
survived a settle but not a reset would vanish at the start of the first trial.

`world_parity.mjs` is the gate, in four phases: `placeSteps` lays the stairs
challenge's grid exactly where `layoutStairs` lays it (280 slots, `Object.is`);
a four-step 60 mm flight posted and read back against `parity/world-v1.json`
(the same bytes as duck-studio's `Fixtures/bench/world.json`, so the Swift
reader and this bench are held to one document); `/reset` re-laying it; and the
one that licenses the design — with that world standing, a `/climb` cell, a
`/chase` cell and a `/record` answer **byte-for-byte** what they answered with
no world set (28, 97 and 1,222 leaves, wall-clock timings named and excluded).
The full `climb_parity.mjs` and `chase_parity.mjs` have also been re-run with a
world standing: 70/70 and 56/56 rows, identical to the runs without one.

The isolation that makes that true is one synchronous block. `stepLive` zeroes
the step geoms' `conaffinity` in the SHARED model, writes the flight, steps, and
restores in a `finally` — **with no `await` inside it**. One `await` there and a
`/record` that happened to overlap a steering loop would step its duck through a
plant with the step blocks isolated, silently, in a number somebody keeps.

`bench_parity.mjs` is the one that licensed splitting `duckbench.mjs` into a
core and a shell: it replays a fixed script and compares every leaf of every
answer, and the split moved none of the numbers. Each gate has been shown to
FAIL — a 1e-7 change to the action scale moves four leaves; an ELU on the last
layer moves the policy by 0.83. A gate nobody has watched fail is a gate nobody
should trust.

Its floor is measured too, and it is not infinitely fine: a 1e-4 error in one of
the 512 first-layer biases produces a worst disagreement of 5.4e-6 and PASSES at
the 1e-5 tolerance. `policy_parity.mjs` catches structural mistakes, not tiny
numeric ones — `physics_parity.mjs`, where a closed loop amplifies, is the gate
that catches those.

## Rebuilding the bundle

```
bash scripts/make_bench_bundle.sh ~/duckbench-bundle
cd ~ && zip -qr duckbench-bundle.zip duckbench-bundle
```

The bundle carries `scene.mjb` — the physics world every measurement is made
against — and its digest is in the bundle's README. A number recorded against a
different plant is not comparable to one recorded here, which is why the world
travels with the code instead of just its filename.

> The bundle is built from a copy list, and the imports do not name everything:
> `duckkit-constants.json` is read at startup by name, so the first bundle
> started and died on ENOENT. The script is verified by **starting** the
> bundle, not by reading it.
>
> The list grew again when the bench was split. `duckbench.mjs` is now the door
> only; it needs `duckbench-core.mjs`, `duckbench-node.mjs`, `policyforward.mjs`
> and `sim/duckloop.mjs` — the last of which is a one-line re-export of
> `site/duckloop.mjs` and is load-bearing despite containing no code, because it
> is the name the core imports the control loop under in both Node and a
> browser.

## Deploying the phone probe

```
bash scripts/make_phonebench.sh          # assembles site/phonebench/ (17 MB)
node sim/pagecheck.mjs                   # renders it in Chromium first
cd site && npx wrangler pages deploy . --project-name microduck-sim --branch=main
```

The project name is not a guess: it is what `.wrangler/cache/pages.json` in this
repo records from the last deploy, on the Civqo account
(71d59adbd067633aca3e95f915fbf2b4). Deploying the whole `site/` directory is
correct and is what previous deploys did — `_headers` has to be at the root of
what is uploaded or Pages ignores it, and the walk demo has to come along or the
deploy would remove it.

Then open **`https://microduck-sim.pages.dev/phonebench/`** on the iPhone. The page
measures and prints; there is nothing to type. `site/_headers` carries the
cross-origin-isolation rules it needs, scoped to `/phonebench/` so the walk demo
is served exactly as it was.

**Do not verify the deploy with a status code.** `/phonebench/` returns 200
today, before anything is deployed there, because Pages falls back to the site's
`index.html` for an unknown path — checked 2026-09-01, and what comes back is
the walk demo. Check the title instead:

```
curl -s https://microduck-sim.pages.dev/phonebench/ | grep -o '<title>[^<]*'
#   deployed:      <title>duck bench — phone probe
#   not deployed:  <title>Microduck simulator
```

Pages already serves `.wasm` as `application/wasm` without being asked (measured
on `/vendor/mujoco.wasm`), so that rule in `_headers` is belt and braces. The
COOP/COEP rules are not: the site sends none today.
