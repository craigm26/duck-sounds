// Replay a saved best_2 track and report the LATERAL position, which the
// criterion never looks at. A duck that ends at trunk z = 116 mm (free-standing
// height) far down the run has either walked over the flight or walked BESIDE
// it onto the floor — the flight is only STAIR_HALF_WIDTH wide.
import fs from 'node:fs';
import { run, GEOMS } from './rig2.mjs';
import { STAIR_Y, STAIR_HALF_WIDTH } from '../site/stairs.js';
console.log('STAIR_Y', STAIR_Y, 'half width', STAIR_HALF_WIDTH,
  'flight y span', (STAIR_Y-STAIR_HALF_WIDTH).toFixed(4), (STAIR_Y+STAIR_HALF_WIDTH).toFixed(4));
for (const rmm of (process.env.RISE||'20,40,90').split(',').map(Number)){
  const j = JSON.parse(fs.readFileSync(`../climb/best_2_${rmm}mm.json`,'utf8'));
  const r = await run(j.keyframes, {blend:j.blend, gap:j.gap, side:j.side, approach:j.approach}, rmm/1000);
  console.log(`${rmm}mm  endX ${(r.x*1000).toFixed(1)}  endZ ${(r.z*1000).toFixed(1)}  endY ${(r.endY*1000).toFixed(1)}  `
    + `startY ${((STAIR_Y+j.side)*1000).toFixed(1)}  lateral drift ${((r.endY-STAIR_Y-j.side)*1000).toFixed(1)} mm  `
    + `offFlight=${Math.abs(r.endY-STAIR_Y) > STAIR_HALF_WIDTH}  feetUp ${r.feetUp} onTop ${r.onTop}`);
}
