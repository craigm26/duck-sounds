#!/usr/bin/env python3
"""Render climb/search_block-results.json as the round-2 family-D table.
Reads only the results file; runs no physics."""
import json, sys
R = json.load(open('/home/craigm26/projects/duck-sounds/climb/search_block-results.json'))

print("PLANT:", R['plant'], "| POLICY:", R['policy'], "| CRITERION:", R['criterion'])
print("elapsed_s:", R.get('elapsed_s'))

print("\n--- PHASE P: collision-bit fix vs rig3 on the untouched plant ---")
print(f"{'rise':>6} {'steps':>5} | {'rig3 x':>12} {'rig3 z':>12} {'gap_mm':>8} {'driftX':>7} | {'fix x':>12} {'fix z':>12} {'gap_mm':>8} {'driftX':>7} | diff")
for r in R['phases']['P']:
    a, b = r['rig3'], r['fixed']
    g1 = '-' if a['minStepGap_mm'] is None else f"{a['minStepGap_mm']:.2f}"
    g2 = '-' if b['minStepGap_mm'] is None else f"{b['minStepGap_mm']:.2f}"
    print(f"{r['rise_mm']:>6.0f} {r['steps']:>5} | {a['x']:>12.9f} {a['z']:>12.9f} {g1:>8} {a['maxTreadDriftX_mm']:>7.2f} | {b['x']:>12.9f} {b['z']:>12.9f} {g2:>8} {b['maxTreadDriftX_mm']:>7.2f} | {r['diff']:.3e}")

print("\n--- PHASE A: controls on the fixed flight, criterion=honest ---")
for r in R['phases']['A']:
    print(f"  {r['control']:<12} {r['rise_mm']:>4.0f}mm  honest={str(r['honest']):<5} x={r['x_mm']:>7.1f}mm z={r['z_mm']:>7.1f}mm feetOnTread={r['feetOnTread']}")

B = R['phases']['B']
print(f"\n--- PHASE B: stage 1 alone (the shove), {B['episodes']} episodes ---")
v = B['verified']
for k in ['blockGap_mm','blockDY_mm','blockX_mm','blockZ_mm','duckX_mm','up','footOnBlockFrac','flushWithin20mm']:
    print(f"  {k:<18} {v[k]}")

print("\n--- PHASE C: two-stage, per rise ---")
hdr = f"{'rise':>6} {'eps':>5} {'clr':>5} {'reward':>7} {'honest':>7} {'x_mm':>8} {'z_mm':>8} {'above':>8} {'feet':>5} {'peakZ':>7} {'head%':>6} {'riser%':>7} {'block%':>7} {'up%':>6} {'blkGap':>7} {'sag':>6}"
print(hdr)
for key in R['phases']:
    if not key.startswith('C_'): continue
    c = R['phases'][key]; v = c['verified']
    print(f"{key[2:]:>6} {c['episodes']:>5} {c['clearedHonest']:>5} {v['reward']:>7.3f} {str(v['honest']):>7} {v['x_mm']:>8.1f} {v['z_mm']:>8.1f} {v['above_mm']:>8.1f} {v['feetOnTread']:>2}/{v['feetOnTreadMax']:<2} {v['peakZ_mm']:>7.1f} {v['headFrac']*100:>5.1f}% {v['riserFrac']*100:>6.1f}% {v['blockFootFrac']*100:>6.1f}% {v['upFrac']*100:>5.1f}% {v['blockEnd_gap_mm']:>7.1f} {v['maxTreadSag_mm']:>6.2f}")
    s1 = v.get('stage1')
    if s1: print(f"        stage1: blockGap={s1['blockGap_mm']:.1f}mm dy={s1['blockDY_mm']:.1f}mm blockZ={s1['blockZ']*1000:.1f}mm duckX={s1['duckX']*1000:.1f}mm up={s1['up']} footOnBlock={s1['footOnBlockFrac']*100:.0f}%")
    print(f"        blockEnd: x={v['blockEnd_x_mm']:.1f}mm z={v['blockEnd_z_mm']:.1f}mm onTread={v['blockOnTread']}  maxAbsDY={v['maxAbsDY_mm']:.1f}mm  path={c['path']}")

print("\n--- PHASE D: +-10 mm re-runs and the no-block ablation ---")
for r in R['phases']['D']:
    tag = r.get('ablation','')
    print(f"  {r['from'].split('/')[-1]:<38} {r['rise_mm']:>4.0f}mm honest={str(r['honest']):<5} reward={r['reward']:>6.3f} x={r['x_mm']:>7.1f} z={r['z_mm']:>7.1f} feet={r['feetOnTread']} {tag}")
