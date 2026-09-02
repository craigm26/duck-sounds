// drag_probe.mjs — the question search_block.mjs's PHASE B could not ask.
//
// climb_lib.mjs:110 spawns the duck at x = 0.12 - 0.07 - gap with gap <= 0.12,
// i.e. between 70 and 190 mm in front of the riser, and the duck is about
// 250 mm long. There is therefore NO room to put a block "a duck-length in
// front of the riser" and still have the duck behind it: the whole corridor is
// 90-120 mm. PHASE B's bx range [0, 75] mm is that corridor, and the search
// went straight to its far end (bx = 75 mm, a 15 mm shove) — a placement, not
// a drag.
//
// This probe uses rig3's spawn override to move the duck BACK and asks for a
// real haul: block front face 150 mm from the riser, duck a further 150 mm
// behind it. Everything else is the same episode.
//   cd ~/projects/duck-sounds/sim && node ../climb/drag_probe.mjs
import fs from 'node:fs';
import { saveAndScore, scoreSaved, exportIntent, HOME, J, STAIR_Y, BLOCK_HALF } from '../climb/blockrig.mjs';
const SCRATCH = '/tmp/claude-1000/-home-craigm26/43f07dce-e62f-464d-ae1f-2fe020620950/scratchpad';
const CAND = SCRATCH + '/drag.json';
const mm = v => (v*1000).toFixed(1);
const T0 = Date.now(); const el = () => ((Date.now()-T0)/1000).toFixed(0)+'s';
function rng(seed){ return function(){ seed|=0; seed=seed+0x6D2B79F5|0;
  let t=Math.imul(seed^seed>>>15,1|seed); t=t+Math.imul(t^t>>>7,61|t)^t;
  return ((t^t>>>14)>>>0)/4294967296; }; }
const SP = { pt1:[2.0,8.0], pvx:[0.05,0.40], pblend:[0.3,2.0], pneck:[-0.4,1.5], phead:[-1.0,1.5],
             bdy:[-0.05,0.05], side:[0.0,0.085] };
const pick=(r)=>Object.fromEntries(Object.keys(SP).map(k=>[k,SP[k][0]+r()*(SP[k][1]-SP[k][0])]));
const jit=(r,p,s)=>Object.fromEntries(Object.keys(SP).map(k=>{const[a,b]=SP[k];
  return [k, Math.min(b, Math.max(a, p[k]+(r()*2-1)*(b-a)*s))];}));

const BLOCK_X0 = 0.12 - 0.150 - BLOCK_HALF;   // front face 150 mm from the riser
const DUCK_X0  = BLOCK_X0 - 0.150;            // duck a further 150 mm behind it
const HAUL_MM  = (0.12 - (BLOCK_X0 + BLOCK_HALF)) * 1000;
console.log(`block starts at x=${mm(BLOCK_X0)}mm (front face ${mm(BLOCK_X0+BLOCK_HALF)}mm), duck at x=${mm(DUCK_X0)}mm; the haul is ${HAUL_MM.toFixed(0)}mm`);

const intentOf = p => ({
  family: 'block-drag', gap: 0, side: 0, approach: 0, blend: 1.0,
  spawn: { x: DUCK_X0, y: STAIR_Y + p.side, z: 0.12 },
  block: { on: true, x: BLOCK_X0, dy: p.bdy },
  push: { t1: p.pt1, vx: p.pvx, blend: p.pblend, neck: p.pneck, head: p.phead },
  keyframes: [{ t: 0.2, pose: HOME.slice() }],
});

const BUDGET = Number(process.env.DRAG_MIN || 5) * 60 * 1000;
const r = rng(0xD2A6);
let best=null, n=0;
const rows=[];
while (Date.now()-T0 < BUDGET) {
  const p = (best && r()<0.65) ? jit(r,best.p, r()<0.5?0.08:0.25) : pick(r);
  const rec = await saveAndScore(intentOf(p), 0.090, CAND, {}); n++;
  const s1 = rec.stage1;
  const moved = (s1.blockX - BLOCK_X0)*1000;
  const off = Math.abs(s1.blockDY_mm) > 170;
  const obj = off ? -1000 : moved - 0.25*Math.abs(s1.blockDY_mm) + (s1.up?5:0);
  if (!best || obj > best.obj) {
    best = { p, obj, rec };
    console.log(`  [${el()}] n=${n} moved=${moved.toFixed(1)}mm of ${HAUL_MM.toFixed(0)}mm  gapToRiser=${s1.blockGap_mm.toFixed(1)}mm dy=${s1.blockDY_mm.toFixed(1)}mm blockZ=${mm(s1.blockZ)}mm duckX=${mm(s1.duckX)}mm up=${s1.up} touch=${(rec.blockFootFrac*100).toFixed(0)}% onTop=${(rec.footOnBlockTopFrac*100).toFixed(0)}% t1=${p.pt1.toFixed(2)} vx=${p.pvx.toFixed(3)}`);
  }
}
const path='../climb/best_r2_blockdrag.json';
fs.writeFileSync(path, JSON.stringify(exportIntent(intentOf(best.p)), null, 2));
const v = await scoreSaved(path, 0.090, { trace: true });
const out = {
  generated: new Date().toISOString(), episodes: n, haul_mm: HAUL_MM,
  blockStart_x_mm: +mm(BLOCK_X0), duckStart_x_mm: +mm(DUCK_X0), path,
  verified: {
    blockEnd_x_mm: +mm(v.stage1.blockX), moved_mm: (v.stage1.blockX - BLOCK_X0)*1000,
    gapToRiser_mm: v.stage1.blockGap_mm, flushWithin20mm: Math.abs(v.stage1.blockGap_mm) <= 20,
    blockDY_mm: v.stage1.blockDY_mm, blockZ_mm: +mm(v.stage1.blockZ),
    duckX_mm: +mm(v.stage1.duckX), up: v.stage1.up,
    footTouchFrac: v.blockFootFrac, footOnBlockTopFrac: v.footOnBlockTopFrac,
    bothFeetOnBlockFrac: v.bothFeetOnBlockFrac,
    maxZonBlock_mm: v.maxZonBlock === null ? null : +mm(v.maxZonBlock),
  },
  params: best.p, trace: v.trace,
};
fs.writeFileSync('../climb/drag_probe-results.json', JSON.stringify(out, null, 2));
console.log(`re-scored from ${path}: moved=${out.verified.moved_mm.toFixed(1)}mm gap=${out.verified.gapToRiser_mm.toFixed(1)}mm flush<=20mm=${out.verified.flushWithin20mm} onTop=${(out.verified.footOnBlockTopFrac*100).toFixed(0)}%`);
console.log(`WROTE ../climb/drag_probe-results.json after ${el()}, ${n} episodes`);
