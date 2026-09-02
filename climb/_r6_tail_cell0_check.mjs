// ROUND 6 — the tailTicks law is written TWICE (climb/rig3.mjs and the one
// shared scorer climb/robust.mjs, which cannot import rig3's episode loop).
// _r6_parity.mjs proved cell 0 exact on the eight files that existed before
// the experiment; this holds the SAME comparison against the six files the
// experiment wrote, the ones that actually exercise servo.tailTicks.
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/_r6_tail_cell0_check.mjs
import fs from 'node:fs';
import { scoreSaved } from '../climb/rig3.mjs';
import { scoreCell, intentHashOfFile } from '../climb/robust.mjs';

const P = '../climb/';
const LOG = [];
const log = s => { console.log(s); LOG.push(String(s)); };
const FILES = fs.readdirSync(P).filter(f => /^best_r6_tail\d+_(servo|servoland)_60mm\.json$/.test(f)).sort();
const SNAP = ['x', 'y', 'z', 'dy', 'above', 'up', 'feetUpRaw', 'feetUpLat', 'feetOnTread'];
const OUT = { generated: new Date().toISOString(), script: '_r6_tail_cell0_check.mjs', rows: [] };

log('=== ROUND 6 cell-0 parity on the six tailTicks files: robust.mjs vs rig3.mjs ===');
let exact = 0;
for (const f of FILES) {
  const A = await scoreSaved(P + f, { rise: 0.060, tail: 'policy' });
  const B = await scoreCell(P + f, { rise: 0.060, isolate: true });
  const d = [];
  for (const k of SNAP) if (!Object.is(A.scored[k], B.scored[k])) d.push({ field: 'scored.' + k, rig3: A.scored[k], robust: B.scored[k] });
  for (const k of ['orig', 'lat', 'honest', 'honest60']) if (A.crit[k] !== B.crit[k]) d.push({ field: 'crit.' + k, rig3: A.crit[k], robust: B.crit[k] });
  for (const k of ['reward', 'uprightTailTicks', 'tailTicks', 'maxX', 'maxZ', 'maxAbsDY', 'feetOnTreadMax'])
    if (!Object.is(A[k], B[k])) d.push({ field: k, rig3: A[k], robust: B[k] });
  for (const k of ['armed', 'ticks', 'tailAuthority', 'tailTicksRun'])
    if (!Object.is(A.servo[k], B.servo[k])) d.push({ field: 'servo.' + k, rig3: A.servo[k], robust: B.servo[k] });
  if (!d.length) exact++;
  OUT.rows.push({ file: f, sha256: intentHashOfFile(P + f), exact: !d.length, diffs: d,
    tailAuthority: A.servo.tailAuthority, tailTicksRun: A.servo.tailTicksRun,
    x: A.scored.x, z: A.scored.z, honest: A.crit.honest, uprightTailTicks: A.uprightTailTicks });
  log(`  ${f.padEnd(40)} tailAuth=${String(A.servo.tailAuthority).padStart(2)} run=${String(A.servo.tailTicksRun).padStart(2)}  EXACT=${!d.length}  x=${A.scored.x.toFixed(6)} z=${A.scored.z.toFixed(6)} honest=${A.crit.honest} upTail=${A.uprightTailTicks}/50`);
  for (const x of d) log(`     !! ${JSON.stringify(x)}`);
}
OUT.verdict = exact === OUT.rows.length
  ? `EXACT ${exact}/${OUT.rows.length} — the two implementations of servo.tailTicks agree at full float digits`
  : `MISMATCH ${exact}/${OUT.rows.length}`;
log(`  ${OUT.verdict}`);
fs.writeFileSync(P + 'r6_tailcell0-results.json', JSON.stringify(OUT, null, 2));
fs.writeFileSync(P + '_r6_tailcell0_check.log', LOG.join('\n') + '\n');
