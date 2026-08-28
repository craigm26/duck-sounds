import fs from 'node:fs';
const best = JSON.parse(fs.readFileSync('lever-best-v2.json','utf8'));
const { attempt } = await import('./lever_lib.mjs');
console.log('--- height ---');
for (const h of [0.020, 0.030, 0.040]) {
  let ok = 0; for (let i = 0; i < 3; i++) if ((await attempt(best.p, h)).onTop) ok++;
  console.log(`  ${(h*1000).toFixed(0)} mm  ${ok}/3`);
}
console.log('--- start position (+-20 mm on the approach) ---');
let ok = 0, n = 0;
for (const d of [-0.02, -0.01, 0, 0.01, 0.02]) {
  const q = { ...best.p, approach: Math.max(0, best.p.approach + d) };
  n++; if ((await attempt(q, 0.030)).onTop) ok++;
}
console.log(`  on top in ${ok}/${n} starts`);
