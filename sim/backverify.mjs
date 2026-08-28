import fs from 'node:fs';
const best = JSON.parse(fs.readFileSync('backroll-best.json','utf8'));
process.argv[2] = '';
const mod = await import('./backsearch_lib.mjs');
let over = 0, up = 0;
for (let i = 0; i < 5; i++) {
  const r = await mod.score(best.p);
  if (r.over > 0) over++;
  if (r.endUp) up++;
  console.log(`  run ${i+1}: tilt ${(Math.acos(Math.max(-1,Math.min(1,-r.over)))*57.3).toFixed(0)} deg, ends ${r.endUp?'upright':'down'}`);
}
console.log(`BACKVERIFY over the top ${over}/5, upright ${up}/5`);
