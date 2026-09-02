// ROUND 5 — INERTNESS PROOF for the servoed landing and the whole-episode
// penetration field.
//
// Two instruments are held against the new one, at FULL FLOAT DIGITS:
//
//   climb/rig3_pre_r5.mjs   a byte copy of climb/rig3.mjs at commit e00e1e4
//                           (the round-4 instrument), differing only in its
//                           isMain guard string. Round 5 must be EXACT against
//                           this on every field, criteria included.
//   climb/rig3_pre_r4.mjs   the pre-round-4 copy the round-4 judge used. Round 5
//                           must reproduce the judge's own PHASE P counts on
//                           the same 86 rows — the same 86/86 physical-state
//                           match and the SAME list of criterion rows that the
//                           whole-episode lateral gate moved in round 4.
//
// SET 1 is the judge's 86 rows (43 pre-round-4 files x tails policy+hold).
// SET 2 is every scorable intent JSON in climb/ x tails policy+hold, which
// includes the round-4 event and handoff files that rig3_pre_r4 cannot replay
// and is therefore held only against rig3_pre_r5.
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/_r5_parity.mjs
import fs from 'node:fs';
import { scoreSaved as NEW } from '../climb/rig3.mjs';
import { scoreSaved as PRE5 } from '../climb/rig3_pre_r5.mjs';
import { scoreSaved as PRE4 } from '../climb/rig3_pre_r4.mjs';

const P = '../climb/';
const LOG = [];
const log = s => { console.log(s); LOG.push(String(s)); };
const T0 = Date.now();
const el = () => ((Date.now() - T0) / 1000).toFixed(0) + 's';

// Fields that exist ONLY in the round-5 instrument. They are never compared
// (there is nothing to compare them to); everything else is.
const R5_ONLY = new Set(['minPenetrationEpisode', 'minPenetrationPair', 'minPenetrationTick',
                         'penetrationTicksScanned', 'servo']);

/** Every leaf of OLD, compared to the same path in NEW. Full float digits. */
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
    if (skipTop && R5_ONLY.has(k)) continue;
    if (path === '' && k === 'source') continue;
    if (path === '.opts' && R5_ONLY.has(k)) continue;
    deepDiff(oldV[k], newV === null || typeof newV !== 'object' ? undefined : newV[k], `${path}.${k}`, out, false);
  }
}

const riseOf = f => { const m = f.match(/_(\d+)mm/); return m ? parseInt(m[1], 10) / 1000 : null; };
const ls = re => fs.readdirSync(P).filter(f => re.test(f)).sort();

// ------------------------------------------------------------------- SET 1
// The judge's own list, climb/audit_r4_judge.mjs PHASE P, verbatim.
const PRE_R4_FILES = [
  ...ls(/^best_r2_.*\.json$/),
  ...ls(/^best_r3_.*\.json$/),
  ...ls(/^best_[012]_\d+mm\.json$/),
  'ctrl_do_nothing.json', 'ctrl_on_tread_40mm.json', 'ctrl_on_tread_60mm.json',
  'ctrl_on_tread_90mm.json', 'ctrl_on_tread_120mm.json', 'ctrl_walk_only.json',
].filter(f => fs.existsSync(P + f));

// The judge's field lists, so its PHASE P numbers are reproduced exactly.
const SCAL = ['x0', 'maxX', 'maxZ', 'maxAbsDY', 'feetOnTreadMax', 'feetHighMax',
              'headFrac', 'riserFrac', 'wallFrac', 'upFrac', 'satFrac', 'z0Settle',
              'liftIntegral', 'footNear', 'bothNear', 'reward'];
const SNAP = ['x', 'y', 'z', 'dy', 'above', 'up', 'feetUpRaw', 'feetUpLat', 'feetOnTread'];
const CRIT = ['orig', 'lat', 'honest', 'honest60'];

const OUT = { generated: new Date().toISOString(), script: '_r5_parity.mjs',
              newOnlyFields: [...R5_ONLY] };

log('================================================================');
log(`SET 1 — the round-4 judge's 86 rows (${PRE_R4_FILES.length} pre-round-4 files x tails policy+hold)`);
const s1 = { rows: 0, deepExact: 0, deepDiffs: [], judgeStateExact: 0, judgeRewardExact: 0, judgeCritRows: [] };
for (const f of PRE_R4_FILES) {
  const h = riseOf(f) ?? 0.060;
  for (const tail of ['policy', 'hold']) {
    const A = await NEW(P + f, { rise: h, tail });
    const B5 = await PRE5(P + f, { rise: h, tail });
    const B4 = await PRE4(P + f, { rise: h, tail });
    s1.rows++;
    // (a) round-5 inertness: EVERY field of the round-4 instrument, deep.
    const d = []; deepDiff(B5, A, '', d, true);
    if (!d.length) s1.deepExact++; else s1.deepDiffs.push({ file: f, rise_mm: h * 1000, tail, diffs: d.slice(0, 12), nDiffs: d.length });
    // (b) the judge's own PHASE P comparison, reproduced against rig3_pre_r4
    const jd = [];
    for (const k of SCAL) if (A[k] !== B4[k]) jd.push({ field: k, now: A[k], pre: B4[k] });
    for (const k of SNAP) if (A.scored[k] !== B4.scored[k]) jd.push({ field: 'scored.' + k, now: A.scored[k], pre: B4.scored[k] });
    const stateDiffs = jd.filter(x => x.field !== 'reward');
    if (!stateDiffs.length) s1.judgeStateExact++;
    if (A.reward === B4.reward) s1.judgeRewardExact++;
    const cd = CRIT.filter(k => A.crit[k] !== B4.crit[k]);
    if (cd.length) s1.judgeCritRows.push({ file: f, rise_mm: h * 1000, tail, changed: cd,
      now: Object.fromEntries(cd.map(k => [k, A.crit[k]])), pre: Object.fromEntries(cd.map(k => [k, B4.crit[k]])),
      maxAbsDY_mm: +(A.maxAbsDY * 1000).toFixed(1) });
    if (stateDiffs.length) s1.deepDiffs.push({ file: f, tail, vs: 'pre_r4', diffs: stateDiffs });
  }
  log(`   ${f.padEnd(34)} done  [${el()}]`);
}
log(`   rows: ${s1.rows}`);
log(`   vs rig3_pre_r5 (ROUND-5 INERTNESS, every field deep): EXACT ${s1.deepExact}/${s1.rows}`);
log(`   vs rig3_pre_r4 (the judge's PHASE P): state ${s1.judgeStateExact}/${s1.rows}, reward ${s1.judgeRewardExact}/${s1.rows}, criterion rows moved ${s1.judgeCritRows.length}`);
for (const r of s1.judgeCritRows) log(`      [round-4 gate, not round 5] ${r.file} @${r.rise_mm}mm ${r.tail} ${r.changed.join(',')} pre=${JSON.stringify(r.pre)} now=${JSON.stringify(r.now)} maxAbsDY=${r.maxAbsDY_mm}mm`);
for (const d of s1.deepDiffs) log(`   !! MOVED ${JSON.stringify(d)}`);
OUT.set1 = s1;

// ------------------------------------------------------------------- SET 2
// Every scorable intent in climb/, including round-4 event and handoff files.
const ALL = fs.readdirSync(P).filter(f => f.endsWith('.json')).sort().filter(f => {
  try { const j = JSON.parse(fs.readFileSync(P + f, 'utf8')); return Array.isArray(j.keyframes) && j.keyframes.length && j.keyframes.every(k => Array.isArray(k.pose) && k.pose.length === 14); }
  catch { return false; }
});
log('');
log(`SET 2 — every scorable intent JSON in climb/ (${ALL.length} files) x tails policy+hold = ${ALL.length * 2} rows, vs rig3_pre_r5`);
const s2 = { files: ALL.length, rows: 0, exact: 0, diffs: [] };
for (const f of ALL) {
  const h = riseOf(f) ?? 0.060;
  for (const tail of ['policy', 'hold']) {
    let A, B;
    try { A = await NEW(P + f, { rise: h, tail }); B = await PRE5(P + f, { rise: h, tail }); }
    catch (e) { s2.diffs.push({ file: f, tail, error: String(e.message || e) }); continue; }
    s2.rows++;
    const d = []; deepDiff(B, A, '', d, true);
    if (!d.length) s2.exact++; else s2.diffs.push({ file: f, rise_mm: h * 1000, tail, nDiffs: d.length, diffs: d.slice(0, 12) });
  }
}
log(`   rows ${s2.rows}   EXACT ${s2.exact}/${s2.rows}   [${el()}]`);
for (const d of s2.diffs) log(`   !! MOVED ${JSON.stringify(d)}`);
OUT.set2 = s2;

OUT.verdict = (s1.deepExact === s1.rows && s2.exact === s2.rows)
  ? `INERT — ${s1.rows + s2.rows} rows identical to the round-4 instrument at full float digits`
  : 'NOT INERT';
log('');
log(`VERDICT: ${OUT.verdict}   [${el()}]`);
fs.writeFileSync(P + 'r5_parity-results.json', JSON.stringify(OUT, null, 2));
fs.writeFileSync(P + '_r5_parity.log', LOG.join('\n') + '\n');
