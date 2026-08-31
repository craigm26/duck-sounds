# Getting Duck Studio talking to a bench

The phone can read a policy, fingerprint it and blend one. It cannot **run**
one — an iPhone has no physics engine. The bench is the machine that does.

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
