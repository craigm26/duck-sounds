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
node sim/webcheck.mjs                                 # the browser shell, over real HTTP
node sim/pagecheck.mjs                                # the probe page, in Chromium
```

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
