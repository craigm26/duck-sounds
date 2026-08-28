import fs from 'node:fs';
const best = JSON.parse(fs.readFileSync('climb-best.json','utf8'));
const { attempt } = await import('./climb_lib.mjs');   // _lib, never climb.mjs — importing that re-runs the search
console.log('--- height (strict: feet up, standing, still there a second later) ---');
for (const h of [0.016, 0.024, 0.034]) {
  let ok = 0;
  for (let i = 0; i < 3; i++) if ((await attempt(best.p, h)).onTop) ok++;
  console.log(`  ${(h*1000).toFixed(0)} mm  ${ok}/3`);
}
console.log('--- start position ---');
let ok = 0, n = 0;
for (const d of [-0.015, -0.008, 0, 0.008, 0.015]) {
  const q = { ...best.p, gap: Math.max(0.01, best.p.gap + d) };
  n++; if ((await attempt(q, 0.024)).onTop) ok++;
}
console.log(`  on top in ${ok}/${n} starts (+-15 mm)`);
