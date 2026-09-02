// Merge the per-rise tagged outputs of the two search_1 processes into one
// climb/search_1-results.json and rename each best track to best_1_<rise>mm.json.
// cd ~/projects/duck-sounds/climb && node merge_1.mjs
import fs from 'node:fs';
const dir = '/home/craigm26/projects/duck-sounds/climb';
const out = {};
for (const f of fs.readdirSync(dir).filter(n => /^search_1-results_[abc]\d\.json$/.test(n))) {
  const j = JSON.parse(fs.readFileSync(`${dir}/${f}`,'utf8'));
  for (const k of Object.keys(j)) if (!out[k] || j[k].bestScore > out[k].bestScore) out[k] = j[k];
}
for (const f of fs.readdirSync(dir).filter(n => /^best_1_(\d+)mm_[abc]\d\.json$/.test(n))) {
  const rise = f.match(/best_1_(\d+)mm/)[1];
  if (!out[rise]) continue;
  const j = JSON.parse(fs.readFileSync(`${dir}/${f}`,'utf8'));
  if (Math.abs(+j.note.match(/objective ([-\d.]+)/)[1] - out[rise].bestScore) < 1e-3)
    fs.writeFileSync(`${dir}/best_1_${rise}mm.json`, JSON.stringify(j,null,2));
}
const ord = Object.keys(out).map(Number).sort((a,b)=>a-b);
const sorted = {}; for (const k of ord) sorted[k]=out[k];
fs.writeFileSync(`${dir}/search_1-results.json`, JSON.stringify(sorted,null,2));
console.log('rise  evals  best      onTop  cleared  peakZ  maxX   headFrac riser footOver yaw   upFrac sat   mode');
for (const k of ord){ const r=sorted[k], p=r.physics, t=r.terminal;
  console.log(`${String(k).padStart(4)}  ${String(r.evals).padStart(5)}  ${r.bestScore.toFixed(3).padStart(7)}  `
    +`${String(t.onTop).padEnd(5)}  ${r.cleared.padEnd(7)}  ${String(p.trunkPeakZ_mm).padStart(5)}  ${String(p.trunkMaxX_mm).padStart(5)}  `
    +`${String(p.headContactFrac).padStart(8)} ${String(p.footOnRiserFaceFrac).padStart(5)} ${String(p.outboardFootOverTread).padStart(8)} `
    +`${String(p.maxTrunkYawWhilePlanted_rad).padStart(5)} ${String(p.uprightFrac).padStart(6)} ${String(p.ctrlSatFrac).padStart(5)} ${r.failureMode}`);
}
