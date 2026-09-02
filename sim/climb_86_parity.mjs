// THE 86-ROW PROOF, RE-RUN AGAINST THE SHARED EPISODE.
//
// climb/audit_r6.mjs PHASE P1 is the gate that says round 6 moved nothing on
// the round-4 floor: 43 pre-round-4 intent files x two tails = 86 rows, current
// rig3.mjs against climb/rig3_pre_r5.mjs (a byte copy of rig3.mjs at commit
// e00e1e4), compared by a DEEP RECURSIVE WALK over every leaf with Object.is at
// full float digits. PHASE P3 is the same walk over all 9 core cells of
// robust.mjs against climb/robust_pre_r6.mjs (52b0392).
//
// Extracting the episode loop into sim/climb_score.mjs is exactly the kind of
// change those two phases exist to catch, so they are re-run here rather than
// asserted. This file is that pair of phases and nothing else: same corpus,
// same exclusions, same comparison. audit_r6.mjs's other phases re-measure
// the round-6 search and take a great deal longer; the proof the audits rest on
// is this one.
//
//   cd ~/projects/duck-sounds/sim && node climb_86_parity.mjs
import fs from 'node:fs';
import { scoreSaved as newScore } from '../climb/rig3.mjs';
import { scoreSaved as preR5Score } from '../climb/rig3_pre_r5.mjs';
import { scoreRobust } from '../climb/robust.mjs';
import { scoreRobust as preR6Robust } from '../climb/robust_pre_r6.mjs';

const P = '../climb/';
const ls = re => fs.readdirSync(P).filter(f => re.test(f)).sort();
const riseOf = f => { const m = f.match(/_(\d+)mm/); return m ? parseInt(m[1], 10) / 1000 : null; };

// audit_r6.mjs PHASE P1's corpus, verbatim.
const PRE_R4 = [
  ...ls(/^best_r2_.*\.json$/),
  ...ls(/^best_r3_.*\.json$/),
  ...ls(/^best_[012]_\d+mm\.json$/),
  'ctrl_do_nothing.json', 'ctrl_on_tread_40mm.json', 'ctrl_on_tread_60mm.json',
  'ctrl_on_tread_90mm.json', 'ctrl_on_tread_120mm.json', 'ctrl_walk_only.json',
].filter(f => fs.existsSync(P + f));

// The keys that exist only because of a later round, named so the exclusion is
// visible. audit_r6.mjs's own two sets.
const R6_ONLY = new Set(['tailTrace', 'tailAuthority', 'tailTicksRun']);
const R5_ONLY = new Set(['minPenetrationEpisode', 'minPenetrationPair', 'minPenetrationTick',
                         'penetrationTicksScanned', 'servo']);

function deepDiff(a, b, path, out, skip) {
  if (out.length > 40) return;
  const ta = a === null ? 'null' : typeof a, tb = b === null ? 'null' : typeof b;
  if (ta !== tb) { out.push({ path, now: a, pre: b, why: 'type' }); return; }
  if (a !== null && typeof a === 'object') {
    if (Array.isArray(a) !== Array.isArray(b)) { out.push({ path, why: 'array/object' }); return; }
    if (Array.isArray(a)) {
      if (a.length !== b.length) { out.push({ path, now: a.length, pre: b.length, why: 'length' }); return; }
      for (let i = 0; i < a.length; i++) deepDiff(a[i], b[i], `${path}[${i}]`, out, skip);
      return;
    }
    for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
      if (skip.has(k)) continue;
      deepDiff(a[k], b[k], path ? `${path}.${k}` : k, out, skip);
    }
    return;
  }
  if (!Object.is(a, b)) out.push({ path, now: a, pre: b });
}

const t0 = Date.now();
const el = () => `${((Date.now() - t0) / 1000).toFixed(0)}s`;

console.log('PHASE P1 — CURRENT rig3.mjs (shared episode) vs rig3_pre_r5.mjs on the round-4');
console.log('   judge\'s own 86 rows. DEEP recursive walk over every leaf, Object.is, full float digits.');
const p1 = { rows: 0, exact: 0, diffs: [], penRows: 0, penConsistent: 0, penViolations: [], leaks: [] };
for (const f of PRE_R4) {
  const h = riseOf(f) ?? 0.060;
  for (const tail of ['policy', 'hold']) {
    const A = await newScore(P + f, { rise: h, tail });
    const B = await preR5Score(P + f, { rise: h, tail });
    p1.rows++;
    const dd = []; deepDiff(A, B, '', dd, new Set([...R5_ONLY, ...R6_ONLY]));
    if (!dd.length) p1.exact++; else p1.diffs.push({ file: f, tail, diffs: dd.slice(0, 8) });
    if (A.minPenetrationEpisode !== null && A.penetrationAtScore !== null) {
      p1.penRows++;
      if (A.minPenetrationEpisode <= A.penetrationAtScore + 1e-15) p1.penConsistent++;
      else p1.penViolations.push({ file: f, tail });
    }
    if (A.servo !== null) p1.leaks.push({ file: f, tail, servo: A.servo });
    if (A.tailTrace !== undefined) p1.leaks.push({ file: f, tail, tailTrace: 'PRESENT WITHOUT BEING ASKED FOR' });
  }
}
console.log(`   ${p1.rows} rows (${PRE_R4.length} pre-round-4 files x tails policy+hold)`);
console.log(`   DEEP walk identical: ${p1.exact}/${p1.rows}`);
for (const d of p1.diffs.slice(0, 6)) console.log(`      !! ${d.file} tail=${d.tail}: ${JSON.stringify(d.diffs)}`);
console.log(`   minPenetrationEpisode <= penetrationAtScore : ${p1.penConsistent}/${p1.penRows}`);
console.log(`   servo/tailTrace leaked onto a file that has neither: ${p1.leaks.length}`);
console.log(`   86 rows as the round-4 judge counted them: ${p1.rows === 86}`);
const p1ok = p1.exact === p1.rows && !p1.leaks.length && !p1.penViolations.length && p1.rows === 86;
console.log(`   VERDICT: ${p1ok ? 'THE SHARED EPISODE MOVED NOTHING on the round-4 floor' : 'A PRE-ROUND-5 NUMBER MOVED'}   [${el()}]`);

console.log('');
console.log('PHASE P2 — CURRENT rig3.mjs vs rig3_pre_r6.mjs (52b0392) on the half rig3_pre_r5 cannot');
console.log('   score: the round-5 SERVOED files, the round-6 launches and the record. This is the');
console.log('   only phase that exercises a SERVOED file down the `hold` tail, where the frozen');
console.log('   ctrl comes from what the law last commanded rather than from a keyframe.');
const P2SET = [
  'best_r3_vault_60mm.json', 'best_r4_famA_60mm.json', 'best_r2_vault_60mm.json',
  'best_r5_servo_60mm.json', 'best_r5_servoland_60mm.json', 'best_r5_servoland_kcore_60mm.json',
  'r5_servo_never_60mm.json', 'r5_servo_armed_60mm.json',
  'best_r6_ceilvault_60mm.json', 'best_r6_ceilvaultB_60mm.json', 'best_r6_ceilvaultC_60mm.json',
  'ctrl_do_nothing.json', 'r4_ctrl_on_tread_60mm.json',
].filter(f => fs.existsSync(P + f));
const { scoreSaved: preR6Score } = await import('../climb/rig3_pre_r6.mjs');
const p2 = { rows: 0, exact: 0, diffs: [] };
for (const f of P2SET) {
  for (const tail of ['policy', 'hold']) {
    const A = await newScore(P + f, { rise: 0.060, tail });
    const B = await preR6Score(P + f, { rise: 0.060, tail });
    p2.rows++;
    const dd = []; deepDiff(A, B, '', dd, R6_ONLY);
    if (!dd.length) p2.exact++; else p2.diffs.push({ file: f, tail, diffs: dd.slice(0, 8) });
  }
}
console.log(`   ${p2.rows} rows (${P2SET.length} files x tails policy+hold)   DEEP walk identical: ${p2.exact}/${p2.rows}`);
for (const d of p2.diffs.slice(0, 6)) console.log(`      !! ${d.file} tail=${d.tail}: ${JSON.stringify(d.diffs)}`);
const p2ok = p2.exact === p2.rows;
console.log(`   VERDICT: ${p2ok ? 'THE SHARED EPISODE MOVED NOTHING on the servoed half' : 'A SERVOED NUMBER MOVED'}   [${el()}]`);

console.log('');
console.log('PHASE P3 — CURRENT robust.mjs (shared episode) vs robust_pre_r6.mjs over all 9 CORE');
console.log('   cells. robust.mjs is the scorer that decides every verdict.');
const CELLNAME = [];
for (const dh of [-10, 0, 10]) for (const p of ['.120/x1.0', '.130/x0.7', '.125/x1.3']) CELLNAME.push(`${dh}mm ${p}`);
const P3SET = ['best_r3_vault_60mm.json', 'best_r5_servo_60mm.json',
               'best_r5_servoland_kcore_60mm.json', 'best_r6_ceilvaultC_60mm.json',
               'best_r4_famA_60mm.json', 'ctrl_do_nothing.json'].filter(f => fs.existsSync(P + f));
const p3 = { files: 0, cells: 0, exact: 0, diffs: [], kSame: 0 };
for (const f of P3SET) {
  const A = await scoreRobust(P + f, { rise: 0.060, core: true, skipBounds: true });
  const B = await preR6Robust(P + f, { rise: 0.060, core: true, skipBounds: true });
  p3.files++;
  for (let i = 0; i < A.cells.length; i++) {
    p3.cells++;
    const dd = []; deepDiff(A.cells[i], B.cells[i], '', dd, R6_ONLY);
    if (!dd.length) p3.exact++; else p3.diffs.push({ file: f, cell: CELLNAME[i], diffs: dd.slice(0, 6) });
  }
  const same = A.kCore === B.kCore && A.kCoreStable === B.kCoreStable && A.objective === B.objective
            && A.sha256 === B.sha256;
  if (same) p3.kSame++;
  console.log(`   ${f.padEnd(36)} kCore ${A.kCore}/${B.kCore}  kCoreStable ${A.kCoreStable}/${B.kCoreStable}  `
    + `objective ${A.objective.toFixed(9)}/${B.objective.toFixed(9)}  sha ${A.move}/${B.move}  ${same ? 'SAME' : '!! MOVED'}`);
}
console.log(`   ${p3.cells} core cells (${p3.files} files x 9)   DEEP walk identical: ${p3.exact}/${p3.cells}`);
for (const d of p3.diffs.slice(0, 6)) console.log(`      !! ${d.file} ${d.cell}: ${JSON.stringify(d.diffs)}`);
const p3ok = p3.exact === p3.cells && p3.kSame === p3.files;
console.log(`   VERDICT: ${p3ok ? 'THE SHARED EPISODE MOVED NOTHING in robust.mjs' : 'A CELL MOVED'}   [${el()}]`);

console.log('');
console.log(p1ok && p2ok && p3ok ? 'PARITY PASS — the extraction is a pure move.' : 'PARITY FAIL');
process.exit(p1ok && p2ok && p3ok ? 0 : 1);
