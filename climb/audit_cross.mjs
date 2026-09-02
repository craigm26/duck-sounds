// Does the SAVED track reproduce the numbers the report quoted?
// Runs each saved best through BOTH the untouched harness replay() and the
// instrumented rig2.run(), on the same track, same opts, same rise.
import fs from 'node:fs';
import { replay } from '../sim/climb_lib.mjs';
import { run } from './rig2.mjs';
const mm = v => (v*1000).toFixed(1);
const files = process.argv.slice(2);
for (const f of files) {
  const j = JSON.parse(fs.readFileSync('../climb/'+f,'utf8'));
  const h = parseInt(f.match(/_(\d+)mm/)[1],10)/1000;
  const o = { blend:j.blend, approach:j.approach||0, gap:j.gap, side:j.side };
  for (const d of [-0.010,0,0.010]) {
    const a = await replay(j.keyframes, {...o, gap:o.gap+d}, h);
    const b = await run(j.keyframes, {...o, gap:o.gap+d}, h);
    console.log(`${f} off${(d*1000).toFixed(0).padStart(3)}  climb_lib x=${mm(a.x)} z=${mm(a.z)} feetUp=${a.feetUp} up=${a.up}` +
      `  || rig2 x=${mm(b.x)} z=${mm(b.z)} feetUp=${b.feetUp} up=${b.up}` +
      `  MATCH=${Math.abs(a.x-b.x)<1e-9 && Math.abs(a.z-b.z)<1e-9 && a.feetUp===b.feetUp}` +
      `  || spawnX=${mm(b.x0)} endY-STAIR_Y=${mm(b.endY-1.305)} maxY-STAIR_Y=${mm(b.maxY-1.305)}` +
      ` peakX=${mm(b.maxX)} peakZ=${mm(b.maxZ)} head=${b.headFrac.toFixed(3)} riser=${b.riserFrac.toFixed(3)} feetUpMax=${b.feetUpMax}`);
  }
}
