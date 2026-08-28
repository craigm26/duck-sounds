// Is the flip robust, or is it overfit to the exact warm-up it was searched
// against? Same move, different settle lengths. A move that only works from one
// dynamic state is not a capability, it is a coincidence.
import fs from 'node:fs';
const best = JSON.parse(fs.readFileSync('wallflip-best.json','utf8')).p;
const { attempt } = await import('./wallflip_lib.mjs');
for (const warm of [23, 24, 25, 26, 27, 30, 40]) {
  const r = await attempt(best, warm);
  console.log(`ROBUST warm=${String(warm).padStart(2)}  tilt ${r.tilt.toFixed(0).padStart(3)} deg  inversions ${r.spins}  ends ${r.endUp ? 'upright' : 'down'}`);
}
