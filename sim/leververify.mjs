import fs from 'node:fs';
const best = JSON.parse(fs.readFileSync('lever-best.json','utf8'));
const { attempt } = await import('./lever_lib.mjs');
for (const h of [0.026, 0.040, 0.050]) {
  let ok = 0;
  for (let i = 0; i < 3; i++) if ((await attempt(best.p, h)).onTop) ok++;
  console.log(`LEVERVERIFY ${(h*1000).toFixed(0)} mm  ${ok}/3`);
}
