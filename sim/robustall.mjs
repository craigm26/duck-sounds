// Are the searched moves capabilities, or coincidences?
//
// A move tuned by search can be fitted to the exact state it was searched from.
// The wall flip survived a warm-up sweep; the others have never been checked.
// This varies the warm-up AND the starting position, because a move that only
// works from one spot on the floor is not much use on a robot that walks.
import fs from 'node:fs';

const { attempt: flipAttempt } = await import('./wallflip_lib.mjs');
const { attempt: leverAttempt } = await import('./lever_lib.mjs');
const { attempt: riserAttempt } = await import('./riser_lib.mjs');
const { score: backScore }      = await import('./backsearch_lib.mjs');

const flip  = JSON.parse(fs.readFileSync('wallflip-best.json','utf8')).p;
const lever = JSON.parse(fs.readFileSync('lever-best.json','utf8')).p;
const riser = JSON.parse(fs.readFileSync('riser-best.json','utf8')).p;
const back  = JSON.parse(fs.readFileSync('backroll-best.json','utf8')).p;

const pct = (a, b) => `${a}/${b}`;

// 1. warm-up sensitivity
console.log('--- warm-up sweep (settle ticks before the move) ---');
{
  let ok = 0, n = 0;
  for (const w of [20, 23, 25, 28, 32, 40]) { n++; if ((await flipAttempt(flip, w)).spins > 0) ok++; }
  console.log(`  wall flip   inverted in ${pct(ok, n)} warm-ups`);
}
{
  let ok = 0, n = 0;
  for (const w of [20, 23, 25, 28, 32, 40]) { n++; if ((await backScore(back, w)).over > 0) ok++; }
  console.log(`  back roll   past horizontal in ${pct(ok, n)} warm-ups`);
}

// 2. position sensitivity — the same move, started a little nearer or further
console.log('--- start-position sweep ---');
for (const [name, fn, p, h, key] of [
  ['lever up', leverAttempt, lever, 0.040, null],
  ['riser up', riserAttempt, riser, 0.055, 'gap'],
]) {
  let ok = 0, n = 0;
  for (const d of [-0.02, -0.01, 0, 0.01, 0.02]) {
    const q = { ...p };
    if (key) q[key] = p[key] + d;
    else q.approach = Math.max(0, p.approach + d);
    n++;
    if ((await fn(q, h)).onTop) ok++;
  }
  console.log(`  ${name}   on top in ${pct(ok, n)} starts (+-20 mm)`);
}

// 3. how much of the searched height is retained if you ask for less
console.log('--- margin: does it clear an easier step too? ---');
for (const [name, fn, p] of [['lever up', leverAttempt, lever], ['riser up', riserAttempt, riser]]) {
  const row = [];
  for (const h of [0.010, 0.020, 0.030, 0.040, 0.050]) {
    row.push(`${(h*1000).toFixed(0)}mm:${(await fn(p, h)).onTop ? 'y' : 'n'}`);
  }
  console.log(`  ${name}   ${row.join('  ')}`);
}
