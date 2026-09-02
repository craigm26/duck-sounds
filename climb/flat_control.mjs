// Control experiment, 2026-09-01. NOT a stairs test — the opposite.
//
// intent-riser.json and intent-lever.json were authored against a stairs
// scene (sim/climb_lib.mjs lays a 4 x rise, 280 mm run flight). The LIVE
// bench at 100.122.199.6:8770 runs scene.mjb, which is a FLAT floor: /health
// names its plant "scene.mjb — Pollen robot_allcollisions" and lists only
// blocks and cones as props. There is no stair in it.
//
// So replaying those tracks here answers one question and only one: how much
// of the claimed height gain is the POSE, and how much is the STAIR. If the
// flat run ends at the same trunk height as a plain stand, the gain in the
// notes is contact with the step, not the choreography.
const BENCH = process.env.BENCH || 'http://100.122.199.6:8770';
const post = async (path, body) => {
  const r = await fetch(BENCH + path, { method: 'POST',
    headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  return r.json();
};
const site = new URL('../site/', import.meta.url);
const read = async n => (await import('node:fs/promises')).readFile(new URL(n, site), 'utf8').then(JSON.parse);

for (const name of ['intent-riser.json', 'intent-lever.json', 'intent-climb.json']) {
  const intent = await read(name);
  const track = intent.keyframes.map(k => ({ at: k.t, pose: k.pose }));
  const out = await post('/perform', {
    track, blend: intent.blend, policy: 'BEST_alpha_stand.onnx', rollouts: 8,
  });
  if (out.error) { console.log(`${name}\tERROR ${out.error}`); continue; }
  console.log([
    name.padEnd(20),
    `achieves ${out.achieves}/${out.rollouts}`,
    `medianHeight ${out.medianHeight}`,
    `endHeight ${out.endHeight}`,
    `endsUpright ${out.endsUpright}`,
    `peakJointRate ${out.peakJointRate} rad/s`,
    `plant ${out.plantDigest.slice(0, 12)}`,
  ].join('\t'));
}
