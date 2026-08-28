import fs from 'node:fs';
const best = JSON.parse(fs.readFileSync('backroll-best.json','utf8'));
const { trackOf } = await import('./backsearch_lib.mjs');
const tr = trackOf(best.p);
fs.writeFileSync('../site/intent-backroll.json', JSON.stringify({
  name: 'back_roll',
  blend: best.p.blend,
  policy: 'BEST_alpha_stand.onnx',
  note: 'searched; 179 deg past upright, lands upright, 5/5',
  keyframes: tr.map(f => ({ t: +f.t.toFixed(4), pose: f.pose.map(v => +v.toFixed(5)) })),
}, null, 1));
console.log('wrote intent-backroll.json,', tr.length, 'keyframes, duration',
            tr[tr.length-1].t.toFixed(2), 's, blend', best.p.blend.toFixed(2));
