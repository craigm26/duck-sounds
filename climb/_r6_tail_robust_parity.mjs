// ROUND 6 — the LAST inertness hole. climb/_r6_tail_parity.mjs proved rig3.mjs
// exact against its own pre-round-6 byte copy, and cell 0 of robust.mjs exact
// against rig3.mjs. Cell 0 is one of nine: the other eight cells run plants
// (drop 0.130 / friction x0.7, drop 0.125 / x1.3) that rig3 cannot produce, so
// nothing above proves the round-6 edit left THOSE cells alone.
//
// This does: every core cell of five saved files, scored by the edited
// robust.mjs and by climb/robust_pre_r6.mjs — a byte copy of robust.mjs taken
// immediately before the round-6 edit — compared at full float digits.
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/_r6_tail_robust_parity.mjs
import fs from 'node:fs';
import { scoreCell as NEW, PLANTS, DHS } from '../climb/robust.mjs';
import { scoreCell as PRE } from '../climb/robust_pre_r6.mjs';

const P = '../climb/';
const LOG = [];
const log = s => { console.log(s); LOG.push(String(s)); };
const R6_ONLY = new Set(['tailTrace']);

function deepDiff(oldV, newV, path, out, skipTop) {
  if (oldV === null || typeof oldV !== 'object') {
    if (!Object.is(oldV, newV)) out.push({ path, pre: oldV, now: newV });
    return;
  }
  if (Array.isArray(oldV)) {
    if (!Array.isArray(newV) || newV.length !== oldV.length) { out.push({ path: path + '.length', pre: oldV.length, now: Array.isArray(newV) ? newV.length : typeof newV }); return; }
    for (let i = 0; i < oldV.length; i++) deepDiff(oldV[i], newV[i], `${path}[${i}]`, out, false);
    return;
  }
  for (const k of Object.keys(oldV)) {
    if (skipTop && R6_ONLY.has(k)) continue;
    deepDiff(oldV[k], newV === null || typeof newV !== 'object' ? undefined : newV[k], `${path}.${k}`, out, false);
  }
}

const CASES = [['best_r3_vault_60mm.json', 0.060],
               ['best_r5_servo_60mm.json', 0.060],
               ['best_r5_servoland_kcore_60mm.json', 0.060],
               ['best_r2_vault_60mm.json', 0.060],
               ['ctrl_do_nothing.json', 0.060]];
const OUT = { generated: new Date().toISOString(), script: '_r6_tail_robust_parity.mjs',
              grid: 'the 9 core cells', rows: 0, exact: 0, diffs: [] };
log('=== ROUND 6 — robust.mjs vs robust_pre_r6.mjs, all 9 core cells, full float digits ===');
for (const [f, h] of CASES) {
  let ex = 0, n = 0;
  for (const dh of DHS) for (const p of PLANTS) {
    const A = await NEW(P + f, { rise: h, dh, drop: p.drop, fmul: p.fmul });
    const B = await PRE(P + f, { rise: h, dh, drop: p.drop, fmul: p.fmul });
    const d = []; deepDiff(B, A, '', d, true);
    n++; OUT.rows++;
    if (!d.length) { ex++; OUT.exact++; }
    else OUT.diffs.push({ file: f, cell: { rise_mm: Math.round((h + dh) * 1000), drop: p.drop, fmul: p.fmul }, nDiffs: d.length, diffs: d.slice(0, 10) });
  }
  log(`  ${f.padEnd(36)} EXACT ${ex}/${n}`);
}
OUT.verdict = OUT.exact === OUT.rows
  ? `INERT — ${OUT.rows}/${OUT.rows} core cells identical to the pre-round-6 robust.mjs at full float digits`
  : `NOT INERT (${OUT.exact}/${OUT.rows})`;
log(`  ${OUT.verdict}`);
for (const d of OUT.diffs) log(`  !! ${JSON.stringify(d)}`);
fs.writeFileSync(P + 'r6_tailrobustparity-results.json', JSON.stringify(OUT, null, 2));
fs.writeFileSync(P + '_r6_tail_robust_parity.log', LOG.join('\n') + '\n');
