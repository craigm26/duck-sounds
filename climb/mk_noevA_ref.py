#!/usr/bin/env python3
"""Build the FAMILY-A PARITY REFERENCE.

climb/rig3.mjs and climb/robust.mjs are shared and were being edited by more
than one family in the same wall-clock window, so a byte copy taken at an
arbitrary instant is not a clean control for MY change: it also carries (or
misses) another family's. This script instead produces the reference by
MECHANICALLY REVERTING EXACTLY ONE THING from the files as they stand right
now — the round-4 family-A event-triggered tail — and nothing else. Whatever
else is in those files at that moment is in the reference too, so a difference
between the pair can only be the event code.

Writes climb/rig3_noevA.mjs and climb/robust_noevA.mjs (isMain disabled in both
so importing them runs no experiment).
"""
import re, sys, os

ROOT = os.path.dirname(os.path.abspath(__file__))

ORIG_LOOP = ("  const total = tr[tr.length - 1].t + 0.8;\n"
             "  for (let t = 0; t * DT < total; t++) await step(poseAt(tr, t * DT), true);\n")
BLOCK_START = "  // ================================================== ROUND 4, FAMILY A\n"
BLOCK_END = "    await step(poseAt(TR, time), true);\n  }\n"


def revert(src):
    n = 0
    i = src.index(BLOCK_START)
    j = src.index(BLOCK_END, i) + len(BLOCK_END)
    src = src[:i] + ORIG_LOOP + src[j:]; n += 1
    src = src.replace("import { normEvent, eventFires, eventError, buildDynTrack } from '../climb/event.mjs';\n", "")
    src = src.replace("  const finalPose = poseAt(TR, total);   // TR === tr when the file has no event",
                      "  const finalPose = poseAt(tr, total);")
    # the recorded event field, in both files
    src = re.sub(r"^ *event: EV \? \{[^\n]*\n", "", src, flags=re.M)
    # the opts plumbing, in both files
    src = re.sub(r"^ *// ROUND 4, FAMILY A: the optional event-triggered tail[^\n]*\n *event: j\.event \|\| null,\n", "", src, flags=re.M)
    # the hash list
    src = src.replace("for (const k of ['event', 'spawnQuat'", "for (const k of ['spawnQuat'")
    # the verdict fields
    src = re.sub(r"^ *eventFired: c\.event[^\n]*\n *eventE_mm: c\.event[^\n]*\n", "", src, flags=re.M)
    if 'normEvent' in src or 'buildDynTrack' in src or 'eventFires' in src:
        raise SystemExit('revert incomplete: event code survives')
    return src


def go(name, out, mainpat):
    s = open(os.path.join(ROOT, name)).read()
    s = revert(s)
    s = s.replace(mainpat[0], mainpat[1], 1)
    open(os.path.join(ROOT, out), 'w').write(s)
    print(f'{out}: {len(s)} bytes  (reverted from {name})')


go('rig3.mjs', 'rig3_noevA.mjs',
   ("process.argv[1].endsWith('rig3.mjs')", "process.argv[1].endsWith('__noevA_never__')"))
go('robust.mjs', 'robust_noevA.mjs',
   ("process.argv[1].endsWith('robust.mjs')", "process.argv[1].endsWith('__noevA_never__')"))
