import fs from 'node:fs';
const b = JSON.parse(fs.readFileSync('climb-best.json','utf8'));
const { attempt } = await import('./climb_lib.mjs');
for (let i=0;i<4;i++){
  const r = await attempt(b.p, 0.010);
  console.log(`run ${i}: onTop=${r.onTop} x=${r.x.toFixed(4)} feetUp=${r.feetUp} up=${r.up}`);
}
