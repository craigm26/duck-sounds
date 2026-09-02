// ADVERSARIAL AUDIT — replay every saved best_*.json through the UNTOUCHED
// sim/climb_lib.mjs replay(). No copy, no instrumentation: this is the exact
// function the three searches claim to have been graded by.
import fs from 'node:fs';
import { replay } from '../sim/climb_lib.mjs';

const files = fs.readdirSync('../climb').filter(f => /^best_[012]_\d+mm\.json$/.test(f)).sort();
const rows = [];
for (const f of files) {
  const j = JSON.parse(fs.readFileSync('../climb/' + f, 'utf8'));
  const h = parseInt(f.match(/_(\d+)mm/)[1], 10) / 1000;
  const base = { blend: j.blend, approach: j.approach || 0, gap: j.gap, side: j.side };
  const out = [];
  for (const d of [-0.010, 0, 0.010]) {
    const r = await replay(j.keyframes, { ...base, gap: base.gap + d }, h);
    out.push(r);
  }
  const cleared = out.filter(r => r.onTop).length;
  rows.push({ file: f, rise: h * 1000, cleared, out });
  console.log(`${f.padEnd(20)} rise ${(h*1000).toString().padStart(3)}mm  cleared ${cleared}/3  ` +
    out.map(r => `[x=${(r.x*1000).toFixed(1)} z=${(r.z*1000).toFixed(1)} above=${(r.above*1000).toFixed(1)} feetUp=${r.feetUp} up=${r.up}]`).join(' '));
}
fs.writeFileSync('../climb/audit_replay-results.json', JSON.stringify(rows, null, 1));
console.log('\nTOTAL CLEARED:', rows.reduce((a, r) => a + r.cleared, 0), 'of', rows.length * 3);
