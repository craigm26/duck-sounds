// The saved best_1_<rise>mm.json tracks, scored by the OFFICIAL harness
// (sim/climb_lib.mjs replay(), untouched) at start offsets -10/0/+10 mm.
// cd ~/projects/duck-sounds/sim && node ../climb/verify_1.mjs
import fs from 'node:fs';
import { replay } from '../sim/climb_lib.mjs';
for (const rise of [20,40,60,90,120,180]) {
  const j = JSON.parse(fs.readFileSync(`../climb/best_1_${rise}mm.json`,'utf8'));
  let ok=0; const rows=[];
  for (const d of [-0.010,0,0.010]) {
    const r = await replay(j.keyframes, {blend:j.blend, approach:0, gap:Math.max(0.01,j.gap+d), side:j.side}, rise/1000);
    if (r.onTop) ok++;
    rows.push(`${(d*1000).toFixed(0).padStart(3)}mm onTop=${String(r.onTop).padEnd(5)} z=${(r.z*1000).toFixed(1).padStart(6)} above=${(r.above*1000).toFixed(1).padStart(6)} feetUp=${r.feetUp} up=${r.up}`);
  }
  console.log(`${String(rise).padStart(3)} mm  cleared ${ok}/3   ` + rows.join('  |  '));
}
