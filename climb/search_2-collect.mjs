// Merge the per-rise search_2 shards (one node process wrote each) into one
// climb/search_2-results.json, and print the report table.
//   cd ~/projects/duck-sounds/climb && node search_2-collect.mjs
import fs from 'node:fs';
const dir = '/home/craigm26/projects/duck-sounds/climb';
const out = {}, robust = {};
for (const f of fs.readdirSync(dir).sort()){
  if (/^search_2-results_r\d+\.json$/.test(f)){
    const d = JSON.parse(fs.readFileSync(`${dir}/${f}`,'utf8'));
    for (const k of Object.keys(d)){ d[k].shard=f; robust[k]=d[k]; }
    continue;
  }
  if (!/^search_2-results_[ab]\d\.json$/.test(f)) continue;
  const d = JSON.parse(fs.readFileSync(`${dir}/${f}`,'utf8'));
  for (const k of Object.keys(d)){ d[k].shard = f; if (!out[k] || d[k].bestScore > out[k].bestScore) out[k]=d[k]; }
}
// the first leg, for comparison
const leg1 = {};
for (const f of fs.readdirSync(dir)){
  if (!/^search_1-results.*\.json$/.test(f)) continue;
  const d = JSON.parse(fs.readFileSync(`${dir}/${f}`,'utf8'));
  for (const k of Object.keys(d)) if (!leg1[k] || d[k].bestScore>leg1[k].bestScore) leg1[k]=d[k];
}
fs.writeFileSync(`${dir}/search_2-results.json`,
  JSON.stringify({ strategy:'B — head press + trunk twist (leg 2: forward-drive lever)',
    criterion:'sim/climb_lib.mjs:150 — up && x>0.12 && (z-h)>0.095 && feetUp>=2, held 1 s',
    objectives:{ single_start:'reward() scored at the nominal start (comparable to leg 1)',
      robust:'ROBUST=1 — the same reward, scored as the MINIMUM over start offsets -10/0/+10 mm' },
    leg1_baseline: leg1, leg2_single_start: out, leg2_robust: robust }, null, 2));
// Normalise each best_2_<rise>mm.json to the SITE's intent shape (site/
// intent-lever.json: name / policy / blend / approach / note / keyframes) so a
// track can be dropped straight into the page, and fold the per-rise physics
// and the reproduce command into the note.
for (const r of Object.keys(out)){
  const pick = robust[r] || out[r];      // the robust track is the one that reproduces
  const src = `${dir}/best_2_${r}mm${pick.shard.replace('search_2-results','').replace('.json','')}.json`;
  const dst = `${dir}/best_2_${r}mm.json`;
  if (!fs.existsSync(src)) continue;
  const j = JSON.parse(fs.readFileSync(src,'utf8'));
  fs.writeFileSync(dst, JSON.stringify({
    name: `B2_twist_${r}mm`, policy: 'BEST_alpha_stand.onnx',
    objective: robust[r] ? 'robust (min over start offsets -10/0/+10 mm)' : 'single start',
    blend: j.blend, approach: j.approach, gap: j.gap, side: j.side, twistDir: j.twistDir,
    note: j.note, keyframes: j.keyframes }, null, 2));
}
const rows = [];
for (const r of [20,40,60,90,120,180]){
  const b = out[r]; if (!b) { rows.push(`${r}\t(no run)`); continue; }
  const p = b.physics, t = b.terminal;
  rows.push([r, b.bestScore.toFixed(3), b.cleared, b.evals,
    p.trunkPeakZ_mm, p.trunkMaxX_mm, t.z_mm, t.feetUp, p.peakFeetUp,
    p.headTouchedTread?`yes(${p.headContactFrac})`:'no',
    p.footOnRiserFaceFrac>0?`yes(${p.footOnRiserFaceFrac})`:'no',
    (b.params.approach||0).toFixed(2), b.twistDir>0?'+1':'-1', b.failureMode].join('\t'));
}
console.log('rise\tobj\tcleared\tevals\tpeakZ\tpeakX\tendZ\tfeetUp\tpeakFeetUp\thead\triser\tvx\tdir\tmode');
console.log(rows.join('\n'));
console.log('\nROBUST objective (worst of three start offsets):');
for (const r of [20,40,60,90,120,180]) if (robust[r]){ const v=robust[r], p=v.physics, t=v.terminal;
  console.log(`  ${r}mm obj ${v.bestScore.toFixed(3)} cleared ${v.cleared} evals ${v.evals} endX ${t.x_mm} endZ ${t.z_mm} above ${t.above_mm} feetUp ${t.feetUp} peakFeetUp ${p.peakFeetUp} peakZ ${p.trunkPeakZ_mm} head ${p.headContactFrac} riser ${p.footOnRiserFaceFrac} :: ${v.failureMode}`);
  console.log(`      offsets ` + v.offsetChecks.map(c=>`${c.startOffset_mm>0?'+':''}${c.startOffset_mm}mm:x=${c.x_mm},z=${c.z_mm},feetUp=${c.feetUp},onTop=${c.onTop}`).join('  '));}
console.log('\nleg1 (no forward drive) for comparison:');
for (const r of [20,40,60,90,120,180]) if (leg1[r])
  console.log(`  ${r}mm obj ${leg1[r].bestScore.toFixed(3)} cleared ${leg1[r].cleared} evals ${leg1[r].evals} peakX ${leg1[r].physics.trunkMaxX_mm} peakZ ${leg1[r].physics.trunkPeakZ_mm} head ${leg1[r].physics.headContactFrac} ${leg1[r].failureMode}`);
