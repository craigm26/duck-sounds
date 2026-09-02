// ROUND 6 — INERTNESS PROOF for servo.tailTicks and the read-only tail trace.
//
// Round 6 adds exactly two things to the instrument:
//   (a) servo.tailTicks (climb/servo.mjs), default 0: how many of the 50 tail
//       ticks the servo law keeps the leg slots for.
//   (b) opts.tailTrace, default false: a read-only per-tail-tick record.
// Both must be invisible when not asked for. Three sets prove it, at FULL
// FLOAT DIGITS:
//
//   SET 1  the round-4 judge's 86 rows (43 pre-round-5 files x tails
//          policy+hold) against climb/rig3_pre_r5.mjs — the same 86/86 the
//          round-5 proof reported, so round 6 has not moved round 5's floor.
//   SET 2  EVERY scorable intent JSON in climb/ x tails policy+hold against
//          climb/rig3_pre_r6.mjs, a byte copy of rig3.mjs taken immediately
//          before this round's edit. This is the set that contains the
//          round-5 SERVO files, which rig3_pre_r5 cannot replay: it is the
//          proof that a servo file with no tailTicks is untouched.
//   SET 3  robust.mjs cell 0 (drop 0.120, x1.0, isolate on) against
//          rig3.scoreSaved, deep on the scored snapshot, the criterion, the
//          tail counts and the reward — the ONE shared scorer must not have
//          drifted from the instrument.
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/_r6_tail_parity.mjs
import fs from 'node:fs';
import { scoreSaved as NEW } from '../climb/rig3.mjs';
import { scoreSaved as PRE5 } from '../climb/rig3_pre_r5.mjs';
import { scoreSaved as PRE6 } from '../climb/rig3_pre_r6.mjs';
import { scoreCell } from '../climb/robust.mjs';

const P = '../climb/';
const LOG = [];
const log = s => { console.log(s); LOG.push(String(s)); };
const T0 = Date.now();
const el = () => ((Date.now() - T0) / 1000).toFixed(0) + 's';

// Fields that exist only in a LATER instrument than the one being compared to.
const R5_ONLY = new Set(['minPenetrationEpisode', 'minPenetrationPair', 'minPenetrationTick',
                         'penetrationTicksScanned', 'servo']);
const R6_ONLY = new Set(['tailTrace']);

function deepDiff(oldV, newV, path, out, skipTop, skipSet) {
  if (oldV === null || typeof oldV !== 'object') {
    if (!Object.is(oldV, newV)) out.push({ path, pre: oldV, now: newV });
    return;
  }
  if (Array.isArray(oldV)) {
    if (!Array.isArray(newV) || newV.length !== oldV.length) { out.push({ path: path + '.length', pre: oldV.length, now: Array.isArray(newV) ? newV.length : typeof newV }); return; }
    for (let i = 0; i < oldV.length; i++) deepDiff(oldV[i], newV[i], `${path}[${i}]`, out, false, skipSet);
    return;
  }
  for (const k of Object.keys(oldV)) {
    if (skipTop && skipSet.has(k)) continue;
    if (path === '' && k === 'source') continue;
    if (path === '.opts' && (skipSet.has(k) || k === 'tailTrace')) continue;
    deepDiff(oldV[k], newV === null || typeof newV !== 'object' ? undefined : newV[k], `${path}.${k}`, out, false, skipSet);
  }
}

const riseOf = f => { const m = f.match(/_(\d+)mm/); return m ? parseInt(m[1], 10) / 1000 : null; };
const ls = re => fs.readdirSync(P).filter(f => re.test(f)).sort();

const PRE_R4_FILES = [
  ...ls(/^best_r2_.*\.json$/),
  ...ls(/^best_r3_.*\.json$/),
  ...ls(/^best_[012]_\d+mm\.json$/),
  'ctrl_do_nothing.json', 'ctrl_on_tread_40mm.json', 'ctrl_on_tread_60mm.json',
  'ctrl_on_tread_90mm.json', 'ctrl_on_tread_120mm.json', 'ctrl_walk_only.json',
].filter(f => fs.existsSync(P + f));

const OUT = { generated: new Date().toISOString(), script: '_r6_tail_parity.mjs',
              round6Fields: ['servo.tailTicks (intent)', 'servo.tailAuthority', 'servo.tailTicksRun', 'tailTrace'] };

log('================================================================');
log(`SET 1 — the round-4 judge's 86 rows vs rig3_pre_r5 (${PRE_R4_FILES.length} files x 2 tails)`);
const s1 = { rows: 0, exact: 0, diffs: [] };
for (const f of PRE_R4_FILES) {
  const h = riseOf(f) ?? 0.060;
  for (const tail of ['policy', 'hold']) {
    const A = await NEW(P + f, { rise: h, tail });
    const B = await PRE5(P + f, { rise: h, tail });
    s1.rows++;
    const d = []; deepDiff(B, A, '', d, true, R5_ONLY);
    if (!d.length) s1.exact++; else s1.diffs.push({ file: f, rise_mm: h * 1000, tail, nDiffs: d.length, diffs: d.slice(0, 12) });
  }
}
log(`   rows ${s1.rows}   EXACT ${s1.exact}/${s1.rows}   [${el()}]`);
for (const d of s1.diffs) log(`   !! MOVED ${JSON.stringify(d)}`);
OUT.set1 = s1;

const ALL = fs.readdirSync(P).filter(f => f.endsWith('.json')).sort().filter(f => {
  try { const j = JSON.parse(fs.readFileSync(P + f, 'utf8')); return Array.isArray(j.keyframes) && j.keyframes.length && j.keyframes.every(k => Array.isArray(k.pose) && k.pose.length === 14); }
  catch { return false; }
});
const SERVO_FILES = ALL.filter(f => { try { return !!JSON.parse(fs.readFileSync(P + f, 'utf8')).servo; } catch { return false; } });
log('');
log(`SET 2 — every scorable intent JSON (${ALL.length} files, ${SERVO_FILES.length} of them carrying a servo block) x 2 tails vs rig3_pre_r6`);
const s2 = { files: ALL.length, servoFiles: SERVO_FILES, rows: 0, exact: 0, diffs: [] };
for (const f of ALL) {
  const h = riseOf(f) ?? 0.060;
  for (const tail of ['policy', 'hold']) {
    let A, B;
    try { A = await NEW(P + f, { rise: h, tail }); B = await PRE6(P + f, { rise: h, tail }); }
    catch (e) { s2.diffs.push({ file: f, tail, error: String(e.message || e) }); continue; }
    s2.rows++;
    const d = []; deepDiff(B, A, '', d, true, R6_ONLY);
    if (!d.length) s2.exact++; else s2.diffs.push({ file: f, rise_mm: h * 1000, tail, nDiffs: d.length, diffs: d.slice(0, 12) });
  }
}
log(`   rows ${s2.rows}   EXACT ${s2.exact}/${s2.rows}   [${el()}]`);
for (const d of s2.diffs) log(`   !! MOVED ${JSON.stringify(d)}`);
OUT.set2 = s2;

log('');
log('SET 3 — robust.mjs cell 0 vs rig3.scoreSaved (the ONE shared scorer)');
const CASES = [['best_r2_vault_40mm.json', 0.040], ['best_r2_vault_60mm.json', 0.060],
               ['best_r2_vault_90mm.json', 0.090], ['ctrl_on_tread_90mm.json', 0.090],
               ['ctrl_do_nothing.json', 0.090],
               ['best_r3_vault_60mm.json', 0.060],
               ['best_r5_servo_60mm.json', 0.060],
               ['best_r5_servoland_kcore_60mm.json', 0.060],
               ...ls(/^best_r6_.*\.json$/).map(f => [f, 0.060])];
const SNAP = ['x', 'y', 'z', 'dy', 'above', 'up', 'feetUpRaw', 'feetUpLat', 'feetOnTread'];
const s3 = { rows: 0, exact: 0, rowsOut: [] };
for (const [f, h] of CASES) {
  if (!fs.existsSync(P + f)) continue;
  const A = await NEW(P + f, { rise: h, tail: 'policy' });
  const B = await scoreCell(P + f, { rise: h, isolate: true });
  const d = [];
  for (const k of SNAP) if (!Object.is(A.scored[k], B.scored[k])) d.push({ field: 'scored.' + k, rig3: A.scored[k], robust: B.scored[k] });
  for (const k of ['orig', 'lat', 'honest', 'honest60']) if (A.crit[k] !== B.crit[k]) d.push({ field: 'crit.' + k, rig3: A.crit[k], robust: B.crit[k] });
  for (const k of ['reward', 'uprightTailTicks', 'tailTicks', 'maxX', 'maxZ', 'maxAbsDY', 'feetOnTreadMax'])
    if (!Object.is(A[k], B[k])) d.push({ field: k, rig3: A[k], robust: B[k] });
  if (A.servo || B.servo) {
    for (const k of ['armed', 'ticks', 'tailAuthority', 'tailTicksRun'])
      if (!Object.is(A.servo && A.servo[k], B.servo && B.servo[k])) d.push({ field: 'servo.' + k, rig3: A.servo && A.servo[k], robust: B.servo && B.servo[k] });
  }
  s3.rows++;
  if (!d.length) s3.exact++;
  s3.rowsOut.push({ file: f, rise_mm: h * 1000, exact: !d.length, diffs: d,
    x: A.scored.x, z: A.scored.z, honest: A.crit.honest, upTail: A.uprightTailTicks,
    servoTailRun: A.servo ? A.servo.tailTicksRun : null });
  log(`   ${f.padEnd(34)} @${(h * 1000).toString().padStart(3)}mm  EXACT=${!d.length}  x=${A.scored.x.toFixed(6)} z=${A.scored.z.toFixed(6)} honest=${A.crit.honest} upTail=${A.uprightTailTicks}/50 svTail=${A.servo ? A.servo.tailTicksRun : '-'}`);
  for (const x of d) log(`      !! ${JSON.stringify(x)}`);
}
log(`   cell-0 parity EXACT ${s3.exact}/${s3.rows}   [${el()}]`);
OUT.set3 = s3;

OUT.verdict = (s1.exact === s1.rows && s2.exact === s2.rows && s3.exact === s3.rows)
  ? `INERT — ${s1.rows + s2.rows} instrument rows identical at full float digits, and ${s3.rows}/${s3.rows} cell-0 rows identical between robust.mjs and rig3.mjs`
  : 'NOT INERT';
log('');
log(`VERDICT: ${OUT.verdict}   [${el()}]`);
fs.writeFileSync(P + 'r6_tailparity-results.json', JSON.stringify(OUT, null, 2));
fs.writeFileSync(P + '_r6_tailparity.log', LOG.join('\n') + '\n');
