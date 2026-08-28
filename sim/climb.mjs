// The search that authored the climb. The move itself, and the one function
// that scores an attempt, live in climb_lib.mjs — imported here rather than
// copied, because the two files having their own copy of `attempt` is exactly
// how a verified number and a searched number drift apart. They did: a move
// scored 24 mm against a lone 450 mm block and 0 mm against the flight the page
// actually renders, and only the duplication hid it.
//
// Random restarts, then hill-climb by jitter. Hand-authoring loses badly on
// this robot — the back roll went 60 degrees hand-written against 179 searched
// — because the useful parameters are couplings between a soft servo, a policy
// still trying to balance, and a friction cone, and none of those are legible.
import fs from 'node:fs';
import { attempt, B, randP, jitter } from './climb_lib.mjs';

// Heights are a ladder, not a target: score a candidate by the highest step it
// still stands on, and stop at the first it fails. A move good for 24 mm is
// good for everything below it, so the ladder short-circuits honestly.
const LADDER = [0.010, 0.016, 0.024, 0.034, 0.046, 0.060, 0.080, 0.100];

// A rung counts only if the move clears it from three different start
// positions. One clean run is not a capability: this sim is a contact sim, and
// a contact sim is chaotic — rounding the exported keyframes to five decimals,
// a 3e-5 perturbation, was enough to turn a move that scored 10 mm into one
// that cleared nothing. Scoring a single start finds trajectories that are
// lucky rather than moves that work, and the export then cannot reproduce them.
// So perturb the stand-off by a couple of millimetres and demand two of three.
const STARTS = [-0.002, 0, 0.002];
async function clears(p, h){
  let ok = 0;
  for (const d of STARTS) {
    if ((await attempt({ ...p, gap: Math.max(0.01, p.gap + d) }, h)).onTop) ok++;
    if (ok >= 2) return true;                    // decided; skip the rest
  }
  return false;
}
async function best(p){ let b=0; for(const h of LADDER){ if(!(await clears(p,h))) break; b=h; } return b; }

let bp=null,bh=0,evals=0;
const BUDGET=+(process.env.BUDGET||520);
while(evals<BUDGET){
  let c=randP(), h=await best(c); evals++;
  for(let k=0;k<7 && evals<BUDGET;k++){
    const q=jitter(c,0.2), qh=await best(q); evals++;
    if(qh>h){ c=q; h=qh; k=0; }
  }
  // Save on every improvement, not at the end. These runs get cut off by a
  // wall-clock limit far more often than they exhaust their budget, and a
  // search that only writes its result on a clean exit throws away every hour
  // it spent when it does not get one.
  if(h>bh){ bh=h; bp=c; console.log(`  climb: ${(bh*1000).toFixed(0)} mm  (${evals} evals)`);
    fs.writeFileSync('climb-best.json', JSON.stringify({ mm: bh*1000, p: bp }, null, 1)); }
}
console.log(`CLIMB best ${(bh*1000).toFixed(0)} mm standing on the flight, against the wall`);

