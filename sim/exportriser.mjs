import fs from 'node:fs';
const r = JSON.parse(fs.readFileSync('riser-best.json','utf8'));
const { trackOf } = await import('./riser_lib.mjs');
const tr = trackOf(r.p);
fs.writeFileSync('../site/intent-riser.json', JSON.stringify({
  name: 'riser_up', blend: r.p.blend, policy: 'BEST_alpha_stand.onnx',
  approach: r.p.approach,
  note: 'foot on the riser + head on the tread; 55 mm, 3/3 (lever 40, stepping 26)',
  gap: r.p.gap,
  keyframes: tr.map(f => ({ t: +f.t.toFixed(4), pose: f.pose.map(v => +v.toFixed(5)) })),
}, null, 1));
console.log('riser move exported,', tr.length, 'keyframes, start gap', r.p.gap.toFixed(3), 'm');
