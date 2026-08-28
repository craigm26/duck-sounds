import fs from 'node:fs';
const c = JSON.parse(fs.readFileSync('climb-best.json','utf8'));
const { trackOf } = await import('./climb_lib.mjs');
const tr = trackOf(c.p);
fs.writeFileSync('../site/intent-climb.json', JSON.stringify({
  name: 'climb', blend: c.p.blend, policy: 'BEST_alpha_stand.onnx',
  approach: c.p.approach, gap: c.p.gap, side: c.p.side,
  note: 'head on the tread, then the feet walk up the riser, braced against the wall',
  keyframes: tr.map(f => ({ t: +f.t.toFixed(4), pose: f.pose.map(v => +v.toFixed(5)) })),
}, null, 1));
console.log('climb exported,', tr.length, 'keyframes, gap', c.p.gap.toFixed(3));
