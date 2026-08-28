// Does the shipped riser move actually END STANDING on the step?
import fs from 'node:fs';
const { attempt } = await import('./riser_lib.mjs');
const p = JSON.parse(fs.readFileSync('riser-best.json','utf8')).p;
for (const h of [0.020, 0.035, 0.055]) {
  const r = await attempt(p, h);
  console.log(`STRICT ${(h*1000).toFixed(0)} mm  on top=${r.onTop}  trunk ${(r.above*1000).toFixed(0)} mm above tread  feet up ${r.feetUp}/2  upright=${r.up}`);
}
