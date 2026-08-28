// Did it flip, or did it just lean a long way and come back? The two look the
// same in a peak-tilt number, so this counts how many times the duck actually
// passed through inverted, and how far it travelled while doing it.
import fs from 'node:fs';
const best = JSON.parse(fs.readFileSync('wallflip-best.json','utf8'));
const { attempt } = await import('./wallflip_lib.mjs');
let over = 0, up = 0, spins = 0;
for (let i = 0; i < 5; i++) {
  const r = await attempt(best.p);
  if (r.tilt > 150) over++;
  if (r.endUp) up++;
  spins += r.spins;
  console.log(`  run ${i+1}: tilt ${r.tilt.toFixed(0)} deg, passed inverted ${r.spins}x, ends ${r.endUp ? 'upright' : 'down'}`);
}
console.log(`FLIPVERIFY tilt>150 ${over}/5, upright ${up}/5, inversions total ${spins}`);
console.log(spins > 0
  ? 'FLIPVERIFY it genuinely passes through inverted — that is a flip, not a lean'
  : 'FLIPVERIFY it never passes inverted: a deep lean that recovers, NOT a flip');
