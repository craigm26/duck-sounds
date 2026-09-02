// Does the EXPORTED track reproduce the SEARCHED episode? The exported
// keyframes are rounded to 5 decimals (search_2.mjs's writeFileSync); the
// search scored the unrounded ones. Rebuild both from the same parameters and
// compare. Same idea as sim/climb_lib.mjs's replay(): the page plays the track,
// not the parameters.
import fs from 'node:fs';
import { run } from './rig2.mjs';
const S = JSON.parse(fs.readFileSync('../climb/search_2-results.json','utf8')).leg2;
for (const rmm of (process.env.RISE||'20,40').split(',').map(Number)){
  const exp = JSON.parse(fs.readFileSync(`../climb/best_2_${rmm}mm.json`,'utf8'));
  const o = {blend:exp.blend, gap:exp.gap, side:exp.side, approach:exp.approach};
  const rounded = exp.keyframes;
  // un-round: the same frames at full double precision are not recoverable from
  // the JSON, so instead perturb by the rounding quantum to bound sensitivity.
  const jitter = rounded.map(f=>({t:f.t, pose:f.pose.map(v=>v + 1e-5)}));
  const a = await run(rounded, o, rmm/1000);
  const b = await run(jitter,  o, rmm/1000);
  console.log(`${rmm}mm  searched endX ${S[rmm].terminal.x_mm} mm | exported(5dp) endX ${(a.x*1000).toFixed(1)} `
    + `| exported +1e-5 rad on every joint endX ${(b.x*1000).toFixed(1)}  `
    + `-> a 1e-5 rad change moves the terminal trunk x by ${Math.abs((b.x-a.x)*1000).toFixed(1)} mm`);
}
