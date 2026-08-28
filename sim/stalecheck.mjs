// Which searched moves still work against the CURRENT scene?
//
// The scene has changed twice since some of these were found: steps went from
// invented physics to Pollen's model, and from thin floating treads to solid
// 200 mm blocks with a riser face. A move tuned against geometry that no longer
// exists is not a capability, it is a fossil.
import fs from 'node:fs';
const { attempt: leverAttempt } = await import('./lever_lib.mjs');
const { attempt: riserAttempt } = await import('./riser_lib.mjs');
const lever = JSON.parse(fs.readFileSync('lever-best.json','utf8')).p;
const riser = JSON.parse(fs.readFileSync('riser-best.json','utf8')).p;
const step  = JSON.parse(fs.readFileSync('intent-stepup.json','utf8')).params;

// step_up used the search harness in searchlib; run it through the lever rig,
// which is the same offset-on-policy shape, to see whether it still lands.
const { evaluate } = await import('./searchlib.mjs');
console.log('--- against the CURRENT scene ---');
for (const [name, fn, p, hs] of [
  ['step up (searched v1)', async (pp, h) => await evaluate(pp, h), step, [0.010, 0.018, 0.026]],
  ['lever up',              leverAttempt, lever, [0.020, 0.030, 0.040]],
  ['riser up',              riserAttempt, riser, [0.030, 0.045, 0.055]],
]) {
  const row = [];
  for (const h of hs) {
    let ok = 0;
    for (let i = 0; i < 2; i++) if ((await fn(p, h)).onTop) ok++;
    row.push(`${(h*1000).toFixed(0)}mm:${ok}/2`);
  }
  console.log(`  ${name.padEnd(22)} ${row.join('  ')}`);
}
