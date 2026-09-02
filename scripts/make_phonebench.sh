#!/usr/bin/env bash
# Assemble site/phonebench/ — the Safari probe, as a deployable directory.
#
# WHY IT IS ASSEMBLED AND NOT AUTHORED IN PLACE. Every file it serves already
# exists somewhere that is its home: the core in sim/, the plant in sim/, the
# MuJoCo build in site/vendor/, the parameter bytes in sim/params/. A second
# copy checked in beside the page is a second copy that goes stale, and the one
# that goes stale silently is the plant — a probe running a different scene.mjb
# from the desk bench would print a physics-parity mismatch and be believed.
#
# IT DOES NOT TOUCH site/index.html OR ANYTHING THE WALK DEMO SERVES. The probe
# lives entirely under site/phonebench/, so deploying it cannot change the demo.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$HERE/site/phonebench"
rm -rf "$OUT"; mkdir -p "$OUT/assets/policies"

cp "$HERE/sim/phonebench.html"        "$OUT/index.html"
cp "$HERE/sim/duckbench-core.mjs"     "$OUT/assets/"
cp "$HERE/sim/duckbench-web.mjs"      "$OUT/assets/"
cp "$HERE/sim/policyforward.mjs"      "$OUT/assets/"
# duckloop.mjs, by the name duckbench-core.mjs imports it under. In sim/ that
# name is a re-export of this file; here it IS the file, because a browser has
# no `../site` to reach into.
cp "$HERE/site/duckloop.mjs"          "$OUT/assets/duckloop.mjs"
cp "$HERE/sim/duckkit-constants.json" "$OUT/assets/"
# THE PLANT, FROM sim/. `site/scene.mjb` and `sim/scene.mjb` share a name and
# differ in bytes (PLANT.md), and it is sim/'s that every measurement in duckkit
# is stamped with. Copying the wrong one is the failure this whole script exists
# to prevent, so the digest is printed at the end and can be read against
# /health.
cp "$HERE/sim/scene.mjb"              "$OUT/assets/"
# THE SAME MuJoCo THE DESK BENCH RUNS, CHECKED RATHER THAN ASSUMED. The whole
# physics-parity claim — that a number from the phone is comparable to a number
# from the Pi — rests on these two files being byte-identical to the npm package
# duckbench.mjs imports. They are today (md5 c08b79f7…); `npm update` in sim/
# would change one and not the other, and the probe would then print a parity
# mismatch that reads like a broken phone.
for f in mujoco.js mujoco.wasm; do
  a="$HERE/site/vendor/$f"; b="$HERE/sim/node_modules/mujoco/$f"
  if [ -f "$b" ] && ! cmp -s "$a" "$b"; then
    echo "site/vendor/$f differs from sim/node_modules/mujoco/$f — the probe would not be" >&2
    echo "running the desk bench's physics. Re-vendor before deploying." >&2
    exit 1
  fi
  cp "$a" "$OUT/assets/"
done

# THE POLICIES THE PROBE NEEDS, NOT ALL FIFTEEN. Each is 791,584 bytes and a
# probe that makes a phone download 11.9 MB of networks to answer "can this
# phone run one" is answering a different question. `alpha_stand` is what every
# gated endpoint settles under and `alpha_walking` is what the physics-parity
# script drives; pass --all to ship the rest.
POLICIES=(alpha_stand alpha_walking)
if [ "${1:-}" = "--all" ]; then
  POLICIES=(); for b in "$HERE"/sim/params/*.bin; do POLICIES+=("$(basename "$b" .bin)"); done
fi
{
  printf '{\n  "policies": [\n'
  first=1
  for p in "${POLICIES[@]}"; do
    cp "$HERE/sim/params/$p.bin" "$OUT/assets/policies/$p.bin"
    # `flamingo-cycle-policy.bin` is served under the name /policy takes,
    # which has a slash in it and therefore cannot be a filename.
    case "$p" in
      *-policy) name="${p%-policy}/policy.onnx" ;;
      *)        name="$p.onnx" ;;
    esac
    [ $first -eq 1 ] || printf ',\n'; first=0
    printf '    { "name": "%s", "file": "%s.bin" }' "$name" "$p"
  done
  printf '\n  ]\n}\n'
} > "$OUT/assets/policies/manifest.json"

# THE HEADERS ARE CHECKED, NOT WRITTEN. `site/_headers` is one file for the
# whole site — Pages reads exactly one, from the root of what is uploaded, and
# a second one nested in a folder is ignored in silence — so it is shared with
# the walk demo and this script has no business overwriting it. It is verified
# instead, and a missing rule stops the build rather than producing a probe that
# deploys and then cannot instantiate its own WebAssembly.
HEADERS="$HERE/site/_headers"
for rule in "/phonebench/*" "Cross-Origin-Opener-Policy: same-origin" \
            "Cross-Origin-Embedder-Policy: require-corp" \
            "Cross-Origin-Resource-Policy: same-origin" \
            "/phonebench/assets/*.wasm" "Content-Type: application/wasm"; do
  grep -qF -- "$rule" "$HEADERS" || {
    echo "site/_headers is missing: $rule" >&2
    echo "The probe needs it. Add it to site/_headers by hand — this script will not" >&2
    echo "rewrite a file the walk demo also depends on." >&2
    exit 1
  }
done

DIGEST=$(sha256sum "$OUT/assets/scene.mjb" | cut -d' ' -f1)
echo "site/phonebench assembled"
echo "  plant  scene.mjb $DIGEST"
echo "  policies: ${POLICIES[*]}"
du -sh "$OUT"
find "$OUT" -type f | sed "s|$HERE/||" | sort
