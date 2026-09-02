// Podium confound at flight scale: did the duck leave the staircase sideways?
import fs from 'node:fs';
import { run } from './rig2.mjs';
import { STAIR_Y, STAIR_HALF_WIDTH } from '../site/stairs.js';
const mm=v=>(v*1000).toFixed(1);
const files=fs.readdirSync('../climb').filter(f=>/^best_[012]_\d+mm(_[a-z]\d*)?\.json$/.test(f)).sort();
console.log('flight spans y = STAIR_Y +/- '+mm(STAIR_HALF_WIDTH)+' mm\n');
for(const f of files){
  const j=JSON.parse(fs.readFileSync('../climb/'+f,'utf8'));
  const h=parseInt(f.match(/_(\d+)mm/)[1],10)/1000;
  const r=await run(j.keyframes,{blend:j.blend,approach:j.approach||0,gap:j.gap,side:j.side},h);
  const dEnd=r.endY-STAIR_Y, dMax=r.maxY-STAIR_Y;
  const off=Math.abs(dEnd)>STAIR_HALF_WIDTH||Math.abs(dMax)>STAIR_HALF_WIDTH;
  console.log(`${f.padEnd(24)} rise ${(h*1000).toString().padStart(3)}  endY-Y=${mm(dEnd).padStart(8)}  maxY-Y=${mm(dMax).padStart(7)}  ${off?'*** OFF FLIGHT ***':'on flight'}   peakX=${mm(r.maxX).padStart(7)} endX=${mm(r.x).padStart(8)} feetUp=${r.feetUp}`);
}
