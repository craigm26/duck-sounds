import fs from 'node:fs';
const best = JSON.parse(fs.readFileSync('riser-best.json','utf8'));
const { attempt } = await import('./riser_lib.mjs');
for (const h of [0.040, 0.055, 0.070]) {
  let ok = 0;
  for (let i = 0; i < 3; i++) if ((await attempt(best.p, h)).onTop) ok++;
  console.log(`RISERVERIFY ${(h*1000).toFixed(0)} mm  ${ok}/3`);
}
