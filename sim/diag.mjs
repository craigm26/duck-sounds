import fs from 'node:fs';
const best = JSON.parse(fs.readFileSync('climb-best.json','utf8'));
const { attempt } = await import('./climb_lib.mjs');
for (const h of [0.0, 0.016, 0.024]) {
  const r = await attempt(best.p, h);
  console.log(`h=${(h*1000).toFixed(0)}mm  onTop=${r.onTop} x=${r.x.toFixed(3)} z=${r.z.toFixed(3)} above=${r.above.toFixed(3)} feetUp=${r.feetUp} up=${r.up}`);
}
