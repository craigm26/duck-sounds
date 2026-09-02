// r6_limits_table.mjs — the discriminator. Formats climb/r6_limits-results.json
// (no physics runs here) into the one comparison that answers "what stops the
// trunk": WITHIN each move, the cells that got over the 95 mm bar against the
// cells that did not. Saturation, push-off duration, neck lever, all measured
// on the same episodes robust.mjs scored (parity 9/9 EXACT per file).
import fs from 'node:fs';
const P = '../climb/';
const L = JSON.parse(fs.readFileSync(P + 'r6_limits-results.json', 'utf8'));
const mean = (a, f) => (a.length ? +(a.reduce((s, x) => s + f(x), 0) / a.length).toFixed(4) : null);
const grp = rows => ({
  n: rows.length,
  peakAboveTread_mm: mean(rows, r => r.peakAboveTread_mm),
  pushOffTicks: mean(rows, r => r.pushOffTicks),
  pushOffMs: mean(rows, r => r.pushOffMs),
  hipPitchAtForceCeiling: mean(rows, r => r.torqueCeilingFrac_pushOff.hipPitch),
  kneeAtForceCeiling: mean(rows, r => r.torqueCeilingFrac_pushOff.knee),
  ankleAtForceCeiling: mean(rows, r => r.torqueCeilingFrac_pushOff.ankle),
  neckPitchAtForceCeiling: mean(rows, r => r.torqueCeilingFrac_pushOff.neckPitch),
  anyOf14AtForceCeiling: mean(rows, r => r.anyActuatorAtCeiling_pushOff),
  positionCommandClipped: mean(rows, r => r.commandClipFrac_pushOff),
  peakDuringTrack: mean(rows, r => (r.peakDuringTrack ? 1 : 0)),
});
const pear = (rows, fx, fy) => {
  const n = rows.length, mx = mean(rows, fx), my = mean(rows, fy);
  let a = 0, b = 0, c = 0;
  for (const r of rows) { a += (fx(r) - mx) * (fy(r) - my); b += (fx(r) - mx) ** 2; c += (fy(r) - my) ** 2; }
  return +(a / Math.sqrt(b * c)).toFixed(3);
};
const all = { over: [], under: [] };
const per = [];
for (const f of L.files) {
  const o = f.rows.filter(r => r.over95), u = f.rows.filter(r => !r.over95);
  all.over.push(...o); all.under.push(...u);
  per.push({ file: f.file, move: f.move, parity: f.parityCells, over95: grp(o), under95: grp(u),
    neckLeverH_mm: [f.summary.neckLeverH_mm_min, f.summary.neckLeverH_mm_max],
    neckMaxForceAtLongestLever_N: +(L.forcerange_Nm / (f.summary.neckLeverH_mm_max / 1000)).toFixed(3) });
}
const rows = [...all.over, ...all.under];
const OUT = { generated: new Date().toISOString(), source: 'climb/r6_limits-results.json',
  forcerange_Nm: L.forcerange_Nm, neckStall_N: L.neckStall_N, bodyWeight_N: L.bodyWeight_N,
  window: 'the PUSH-OFF: from the lowest trunk z during the keyframe track to the tick the trunk peaks',
  perFile: per, pooled: { over95: grp(all.over), under95: grp(all.under) },
  correlationsOverAll36Cells: {
    pushOffTicks_vs_peakHeight: pear(rows, r => r.pushOffTicks, r => r.peakAboveTread_mm),
    hipPitchSaturation_vs_peakHeight: pear(rows, r => r.torqueCeilingFrac_pushOff.hipPitch, r => r.peakAboveTread_mm),
    kneeSaturation_vs_peakHeight: pear(rows, r => r.torqueCeilingFrac_pushOff.knee, r => r.peakAboveTread_mm),
    neckSaturation_vs_peakHeight: pear(rows, r => r.torqueCeilingFrac_pushOff.neckPitch, r => r.peakAboveTread_mm),
  } };
fs.writeFileSync(P + 'r6_limits_table-results.json', JSON.stringify(OUT, null, 1) + '\n');
const hdr = '  move          grp  n  peak(mm) push-off(ms) hip@0.6405 knee@ ankle neck  any14 cmdClip';
console.log(`THE DISCRIMINATOR — within each move, cells over the 95 mm bar vs cells under it`);
console.log(`  push-off window: ${OUT.window}`);
console.log(`  "at 0.6405" = |actuator_force| within 1e-4 N.m of the forcerange, as a fraction of push-off ticks\n`);
console.log(hdr);
const line = (lbl, g) => console.log(`  ${lbl.padEnd(13)} ${g === OUT.pooled.over95 || g.over ? '' : ''}${String(g.n).padStart(4)}  ${String(g.peakAboveTread_mm).padStart(7)} ${String(g.pushOffMs).padStart(12)} ${String(g.hipPitchAtForceCeiling).padStart(10)} ${String(g.kneeAtForceCeiling).padStart(5)} ${String(g.ankleAtForceCeiling).padStart(5)} ${String(g.neckPitchAtForceCeiling).padStart(5)} ${String(g.anyOf14AtForceCeiling).padStart(5)} ${String(g.positionCommandClipped).padStart(7)}`);
for (const p of per) { line(`${p.move} >95`, p.over95); line(`${p.move} <95`, p.under95); }
console.log('  ---');
line('POOLED    >95', OUT.pooled.over95);
line('POOLED    <95', OUT.pooled.under95);
console.log(`\n  correlations over all ${rows.length} cells: ${JSON.stringify(OUT.correlationsOverAll36Cells)}`);
console.log('\n  THE NECK STRUT, per move: horizontal lever from the neck_pitch anchor to the jaw during the push-off,');
console.log('  and the largest vertical force 0.6405 N.m can hold at the beak through the LONGEST of those levers:');
for (const p of per) console.log(`    ${p.move}  lever ${p.neckLeverH_mm[0]}..${p.neckLeverH_mm[1]} mm  ->  ${p.neckMaxForceAtLongestLever_N} N   (body weight ${L.bodyWeight_N} N, neck stall ${L.neckStall_N} N)`);
console.log('  wrote climb/r6_limits_table-results.json');
