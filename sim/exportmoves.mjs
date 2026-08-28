import fs from 'node:fs';
const lever = JSON.parse(fs.readFileSync('lever-best.json','utf8'));
const flip  = JSON.parse(fs.readFileSync('wallflip-best.json','utf8'));
const L = await import('./lever_lib.mjs');
const W = await import('./wallflip_lib.mjs');
const pack = (name, tr, p, note) => ({
  name, blend: p.blend, policy: 'BEST_alpha_stand.onnx', note,
  approach: p.approach ?? 0,
  keyframes: tr.map(f => ({ t: +f.t.toFixed(4), pose: f.pose.map(v => +v.toFixed(5)) })),
});
fs.writeFileSync('../site/intent-lever.json', JSON.stringify(
  pack('lever_up', L.trackOf(lever.p), lever.p,
       'head planted on the tread as a pivot; 40 mm, 3/3 (stepping managed 26)'), null, 1));
fs.writeFileSync('../site/intent-wallflip.json', JSON.stringify(
  pack('wall_flip', W.trackOf(flip.p), flip.p,
       'two feet against a wall; 179 deg, passes inverted once, lands upright 5/5'), null, 1));
console.log('lever keyframes', L.trackOf(lever.p).length, ' flip keyframes', W.trackOf(flip.p).length);
console.log('flip start gap from wall:', flip.p.startGap.toFixed(3), 'm');
