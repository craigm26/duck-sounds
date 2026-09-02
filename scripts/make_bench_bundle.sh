#!/usr/bin/env bash
# Package the duck bench so it can be started on another machine — a Windows
# box with a GPU, a spare laptop, anything that runs Node.
#
# WHY A BUNDLE AND NOT "CLONE THE REPO". The bench needs five things from three
# directories of a private repo, and one of them is a 7 MB binary plant whose
# digest every recorded clip is stamped with. Handing somebody a repo and a list
# of paths is how the wrong scene.mjb ends up in the bundle and every
# measurement afterwards is quietly against a different world.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/tmp/duckbench-bundle}"
rm -rf "$OUT"; mkdir -p "$OUT/sim" "$OUT/site"

# duckkit-constants.json is read at startup by name and appears in no import
# list, so a bundle built from the imports alone starts and dies on
# ENOENT. Found by starting the bundle rather than by reading the copy list.
#
# AND THE IMPORT LIST GREW WHEN THE BENCH WAS SPLIT. duckbench.mjs is now the
# door only: the physics is in duckbench-core.mjs, this machine's description is
# in duckbench-node.mjs, and the core imports the control loop as
# `./duckloop.mjs` — which in sim/ is a one-line re-export of site/duckloop.mjs
# and is therefore load-bearing despite containing no code. Leaving any of the
# four out produces a bundle that installs cleanly and dies on its first import.
cp "$HERE/sim/duckbench.mjs" "$HERE/sim/duckbench-core.mjs" \
   "$HERE/sim/duckbench-node.mjs" "$HERE/sim/duckloop.mjs" \
   "$HERE/sim/policyforward.mjs" "$HERE/sim/onnx_meta.mjs" \
   "$HERE/sim/duckkit-constants.json" "$OUT/sim/"
cp "$HERE/site/duckloop.mjs" "$OUT/site/"
cp "$HERE/sim/scene.mjb" "$OUT/sim/"
# The shipped policies only. Anything uploaded to a running bench is scratch and
# belongs to that machine, not in a bundle somebody else starts.
for f in "$HERE"/sim/*.onnx; do
  case "$(basename "$f")" in uploaded-*) continue;; esac
  cp "$f" "$OUT/sim/"
done
[ -d "$HERE/sim/community" ] && cp -r "$HERE/sim/community" "$OUT/sim/" || true

cat > "$OUT/sim/package.json" <<'JSON'
{ "name": "duckbench", "private": true, "type": "module",
  "dependencies": { "mujoco": "^3.1.16", "onnxruntime-node": "^1.29.0" } }
JSON

# THE PLANT'S DIGEST TRAVELS WITH IT. /health reports this, and a clip recorded
# against a different world is not comparable to one recorded here however
# identical the filenames look.
DIGEST=$(sha256sum "$OUT/sim/scene.mjb" | cut -d' ' -f1)

cat > "$OUT/start.ps1" <<'PS1'
# Start the duck bench. Run this in PowerShell from the folder it sits in.
$ErrorActionPreference = "Stop"
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
  Write-Host "Node.js is not installed. Get it from https://nodejs.org (LTS), then run this again."
  exit 1
}
Set-Location "$PSScriptRoot\sim"
if (-not (Test-Path node_modules)) {
  Write-Host "Installing MuJoCo and onnxruntime (once, a couple of minutes)..."
  npm install --no-audit --no-fund
}
# The phone reaches this over Tailscale, so it has to listen on every
# interface, not just localhost. duckbench already binds 0.0.0.0.
Write-Host ""
Write-Host "Starting the bench. Leave this window open."
Write-Host "In Duck Studio, use this address:" -ForegroundColor Green
$ts = (tailscale ip -4 2>$null)
if ($ts) { Write-Host "    $ts`:8770" -ForegroundColor Green }
else { Write-Host "    <this machine's Tailscale IP>:8770" -ForegroundColor Green }
Write-Host ""
node duckbench.mjs
PS1

cat > "$OUT/start.sh" <<'SH2'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/sim"
command -v node >/dev/null || { echo "Node.js is not installed."; exit 1; }
[ -d node_modules ] || npm install --no-audit --no-fund
echo "In Duck Studio, use: $(tailscale ip -4 2>/dev/null | head -1 || echo '<this machine>'):8770"
exec node duckbench.mjs
SH2
chmod +x "$OUT/start.sh"

sed -e "s/__DIGEST__/$DIGEST/" > "$OUT/README.txt" <<'TXT'
The duck bench
==============

This is a machine with room to run the duck properly. Duck Studio can read a
policy, fingerprint it and blend it; it can now also RUN one, in a browser, on
the phone itself — MuJoCo compiles to WebAssembly and the network is four matrix
multiplies. What a desk gives you is headroom: sixteen rollouts of a measurement
finish while a phone is still on its fourth, and this bench carries every policy
in the repo rather than the two the app bundles.

The phone talks to it over your Tailscale network.

WHAT YOU NEED
  - Node.js (LTS) from https://nodejs.org
  - Tailscale, signed in to the same account as the phone

TO START IT
  Windows:  right-click start.ps1 -> Run with PowerShell
  Mac/Linux: ./start.sh

  The first start installs MuJoCo and onnxruntime and takes a couple of
  minutes. Every start after that is a few seconds.

IN THE APP
  Duck Studio -> Policies -> Run on your network, and enter the address the
  start script prints. It is your Tailscale IP and port 8770 — something like
  100.95.79.116:8770.

  The phone and this machine do NOT need to be on the same Wi-Fi. That is the
  point of using the Tailscale address rather than a 192.168 one: it works from
  a phone on cellular.

IF YOU WANT A PASSWORD ON IT
  Set DUCKBENCH_TOKEN before starting, and put the same string in the app's
  token field:
      Windows:  $env:DUCKBENCH_TOKEN = "something-long"; .\start.ps1
      Mac/Linux: DUCKBENCH_TOKEN=something-long ./start.sh

  Without it the bench is open to anything that can reach it on your tailnet.
  A tailnet is not the public internet, so this is a real choice rather than an
  emergency, but on a shared machine set one.

THE WORLD THIS RUNS
  scene.mjb, sha256 __DIGEST__

  Every measurement the bench reports is against that file, and /health says so.
  A number recorded on a different plant is not comparable to one recorded here,
  which is why the digest travels with the bundle instead of just the filename.
TXT
echo "bundle: $OUT"
du -sh "$OUT"
