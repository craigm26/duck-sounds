// Re-run the searched intent independently. A single successful evaluation
// during a search can be luck; this repeats it and reports how often it holds.
import fs from 'node:fs';
const best = JSON.parse(fs.readFileSync('intent-stepup.json', 'utf8'));
process.env.VERIFY = '1';
const { evaluate, DEFAULTS } = await import('./searchlib.mjs');
for (const h of [0.018, 0.022, 0.026, 0.030]) {
  let ok = 0;
  for (let i = 0; i < 3; i++) if ((await evaluate(best.params, h)).onTop) ok++;
  console.log(`VERIFY ${(h*1000).toFixed(0)} mm  ${ok}/3 succeeded`);
}
let base = 0;
for (let i = 0; i < 3; i++) if ((await evaluate({ ...DEFAULTS, lead: 0 }, 0.010)).onTop) base++;
console.log(`VERIFY hand-tuned at 10 mm  ${base}/3 succeeded`);
