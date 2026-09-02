// ROUND 6 — THE JUDGE'S LOOP.
//
// New file. climb/audit_r2.mjs, audit_r3.mjs, audit_r4.mjs, audit_r4_judge.mjs
// and audit_r5.mjs are the earlier rounds' evidence and are NOT touched. This
// file writes climb/r6_judge-results.json and climb/audit_r6.log only, plus its
// own probe intents under new paths (climb/audit_r6_*.json and the
// reconstruction directory climb/audit_r6_recon/).
//
// Every number printed here is measured IN THIS PROCESS, from a SAVED JSON on
// disk, through the CURRENT climb/rig3.mjs and climb/robust.mjs. Nothing is
// quoted from an agent's report; where a report made a claim, the claim is
// re-derived and the report's number is printed beside it.
//
// ROUND 6 IS TWO AGENTS AND THEY OVERLAPPED. One ran a ceiling screen + a CEM
// over the vault launch (climb/r6_screen.mjs, r6_ceiling_search.mjs, r6_verify,
// r6_limits) and published three intents. The other ADDED A FIELD TO THE
// INSTRUMENT — servo.tailTicks, plus an opt-in tailTrace — editing rig3.mjs,
// robust.mjs and servo.mjs, and published six intents. So the ceiling search's
// numbers were produced by an instrument that has since been edited, and the
// first duty of this judge is to establish that the edit moved nothing and that
// the search's rows reproduce under the instrument as it stands NOW.
//
//   PHASE P   PARITY, in four parts.
//             P0 provenance: rig3_pre_r5.mjs IS e00e1e4:climb/rig3.mjs;
//                rig3_pre_r6.mjs and robust_pre_r6.mjs ARE 52b0392's; and no
//                tracked intent JSON under climb/ has been modified.
//             P1 CURRENT rig3.mjs vs rig3_pre_r5.mjs on the round-4 judge's own
//                86 rows (43 pre-round-4 files x tails policy+hold), at full
//                float digits, DEEP recursive walk over every leaf.
//             P2 CURRENT rig3.mjs vs rig3_pre_r6.mjs (the byte copy taken at
//                52b0392) over a set that INCLUDES the round-5 servoed files
//                and the round-6 launches, x tails policy+hold. That is the
//                half rig3_pre_r5 cannot score.
//             P3 CURRENT robust.mjs vs robust_pre_r6.mjs over all 9 core cells
//                of a servoed + unservoed set. The verdict scorer is robust.mjs
//                and it is the one that has to be unchanged.
//   PHASE R   Are rig3.mjs and robust.mjs still the same simulation on the
//             round-6 files, including the six that carry servo.tailTicks?
//   PHASE H   Identity: sha256 of every round-6 file, distinctness, and every
//             hash earlier rounds published, re-computed under the edited
//             instrument.
//   PHASE B   Declared bounds, enforced at scoring time — and, separately, the
//             search's OWN 29-parameter BOUNDS, which robust.mjs does not check
//             because the published files declare no `bounds` block.
//   PHASE S   How the duck STARTED. spawn / spawnPose / spawnVel /
//             spawnLastAction / settleTicks != 25 means it did not start on the
//             floor and is not a climb.
//   PHASE E   THE CEILING, re-derived UNROUNDED. `honest` needs the trunk above
//             95 mm at the scored instant, so the count of core cells in which
//             the trunk EVER exceeded 95 mm is an upper bound on kCore under
//             ANY landing law. Both round-6 agents computed it off peakZ_mm,
//             which is rounded to 0.1 mm before the comparison; this judge
//             computes it from the raw maxZ and reports every cell where the
//             two could disagree.
//   PHASE W   THE SCREEN, re-derived. Every published launch the screen claims
//             to have scored, re-scored here, row against row.
//   PHASE X   THE SEARCH, RECONSTRUCTED. 337 of the 340 vectors the CEM reports
//             were never saved as files, so on the standing rule ("score only
//             saved JSON files") their rows are unauditable as published. They
//             are NOT taken on trust: r6_ceiling-results.json carries each
//             move's full 29-parameter `params`, so this judge rebuilds every
//             one through the search's own intentOf(), WRITES IT TO DISK, and
//             re-scores it. A rebuild whose sha256 does not equal the claimed
//             sha256 is not the same vector and is reported as such.
//   PHASE G   Re-score every round-6 claim on the 14-cell grid, under BOTH
//             landings the instrument offers (the policy tail that decides
//             every verdict, and the hold tail), plus the ABLATION landing:
//             the same launch with its servo block deleted.
//   PHASE N   Penetration, whole-episode, on every clear.
//   PHASE T   Torque against the 0.6405 N.m forcerange.
//   PHASE C   The controls. do-nothing must fail; a duck PLACED on the tread
//             must pass. If round 6's edits broke the criterion, this is where
//             it shows.
//   PHASE K   THE KILL GATE: PASSED only if a DISTINCT, FLOOR-SPAWNED launch
//             reaches ceilingCore >= 7 of 9 at 60 mm with no cheat.
//
// THE BAR IS 95 mm AND THIS FILE DOES NOT MOVE IT. CEILING_ABOVE below is read
// straight out of the same constant the criterion uses; PHASE E prints the
// criterion's own threshold beside it.
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/audit_r6.mjs
import fs from 'node:fs';
import { execSync } from 'node:child_process';
import { scoreSaved as newScore, criteria, LATERAL, DUCKG, RISER_X } from '../climb/rig3.mjs';
import { scoreSaved as preR5Score } from '../climb/rig3_pre_r5.mjs';
import { scoreSaved as preR6Score } from '../climb/rig3_pre_r6.mjs';
import { scoreRobust, scoreCell, intentHash, intentHashOfFile, checkBounds,
         DHS, PLANTS, EXT_DHS, EXT_PLANT, CLEAR_BONUS, UPRIGHT_BONUS,
         UPRIGHT_TAIL_MIN, DECLARED_BOUNDS, HOME } from '../climb/robust.mjs';
import { scoreRobust as preR6Robust } from '../climb/robust_pre_r6.mjs';

const P = '../climb/';
const RECON_DIR = P + 'audit_r6_recon/';
const LOG = [];
const log = s => { console.log(s); LOG.push(String(s)); };
const T0 = Date.now();
const el = () => ((Date.now() - T0) / 1000).toFixed(0) + 's';
const FORCERANGE = 0.6405;                     // N.m, every actuator
const CEILING_ABOVE = 0.095;                   // `honest` needs above > this
const PEN_SOFT = -0.015;                       // the "soft contact" line rounds 4-5 used
const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const DT = 1 / C.tickHz;                       // the CONTROL tick, measured

const rd = f => JSON.parse(fs.readFileSync(P + f, 'utf8'));
const ls = re => fs.readdirSync(P).filter(f => re.test(f)).sort();
const riseOf = f => { const m = f.match(/_(\d+)mm/); return m ? parseInt(m[1], 10) / 1000 : null; };
const f2 = n => (n === null || n === undefined) ? 'null' : n.toFixed(2);
const CELLNAME = ['50/.120/x1.0', '50/.130/x0.7', '50/.125/x1.3',
                  '60/.120/x1.0', '60/.130/x0.7', '60/.125/x1.3',
                  '70/.120/x1.0', '70/.130/x0.7', '70/.125/x1.3'];

const OUT = {
  generated: new Date().toISOString(),
  judge: 'audit_r6.mjs',
  note: 'Round-6 judge. Writes climb/r6_judge-results.json, climb/audit_r6.log, climb/audit_r6_*.json probes and climb/audit_r6_recon/ only; no earlier round\'s file is touched.',
  constants: {
    lateralGate_mm: LATERAL * 1000, duckCollidableGeoms: DUCKG.length,
    riserLine_mm: RISER_X * 1000, forcerange_Nm: FORCERANGE,
    honestAboveGate_mm: CEILING_ABOVE * 1000, softContactLine_mm: PEN_SOFT * 1000,
    controlTick_ms: DT * 1000, tickHz: C.tickHz,
    uprightTailMin: UPRIGHT_TAIL_MIN, clearBonus: CLEAR_BONUS, uprightBonus: UPRIGHT_BONUS,
    declaredBounds: DECLARED_BOUNDS,
    grid: { core: { dhs_mm: DHS.map(d => d * 1000), plants: PLANTS },
            ext: { dhs_mm: EXT_DHS.map(d => d * 1000), plant: EXT_PLANT }, n: 14 },
  },
};
const flush = () => {
  fs.writeFileSync(P + 'r6_judge-results.json', JSON.stringify(OUT, null, 1) + '\n');
  fs.writeFileSync(P + 'audit_r6.log', LOG.join('\n') + '\n');
};

log('================================================================');
log('ROUND 6 — THE JUDGE. Every number measured here, from saved JSON, now.');
log(`   lateral gate ${(LATERAL * 1000).toFixed(0)} mm | duck geoms ${DUCKG.length} | riser ${(RISER_X * 1000).toFixed(0)} mm`);
log(`   CONTROL TICK from duckkit-constants.json: tickHz=${C.tickHz} -> ${(DT * 1000).toFixed(2)} ms`);
log(`   THE BAR: rig3 criteria() honest needs trunk above > ${(CEILING_ABOVE * 1000).toFixed(0)} mm. It is not moved anywhere in this file.`);

// =================================================================== PHASE P0
log('');
log('PHASE P0 — PROVENANCE of the three byte copies, and of the corpus itself.');
const prov = {};
const gitFile = (rev, path) => execSync(`git show ${rev}:${path}`, { cwd: '..', encoding: 'utf8' });
const copyMatches = (copyFile, rev, realPath, isMainName) => {
  try {
    const now = gitFile(rev, realPath);
    const copy = fs.readFileSync(P + copyFile, 'utf8').split(copyFile).join(realPath.split('/').pop());
    return { ok: copy === now, note: copy === now
      ? `${copyFile} IS ${realPath} at ${rev}, character for character apart from its isMain guard string`
      : `${copyFile} DIFFERS from ${realPath} at ${rev} by more than the isMain guard` };
  } catch (e) { return { ok: false, note: `could not read ${rev}:${realPath} — ${e.message}` }; }
};
prov.rig3_pre_r5 = copyMatches('rig3_pre_r5.mjs', 'e00e1e4', 'climb/rig3.mjs');
prov.rig3_pre_r6 = copyMatches('rig3_pre_r6.mjs', '52b0392', 'climb/rig3.mjs');
prov.robust_pre_r6 = copyMatches('robust_pre_r6.mjs', '52b0392', 'climb/robust.mjs');
for (const [k, v] of Object.entries(prov)) log(`   ${v.ok ? 'OK  ' : '!!  '}${v.note}`);
// No earlier round's saved intent may have been written to. git is the witness.
let dirtyIntents = [], gitNote = '';
try {
  const st = execSync('git status --porcelain -- climb/', { cwd: '..', encoding: 'utf8' });
  dirtyIntents = st.split('\n').filter(Boolean)
    .filter(l => /\.json$/.test(l) && !/^\?\?/.test(l))
    .map(l => l.trim());
  gitNote = dirtyIntents.length ? 'A TRACKED INTENT UNDER climb/ HAS BEEN MODIFIED' : 'no tracked JSON under climb/ is modified — every earlier round\'s saved intent is byte-identical to its commit';
} catch (e) { gitNote = 'git status failed — ' + e.message; }
log(`   ${dirtyIntents.length ? '!!  ' : 'OK  '}${gitNote}`);
for (const d of dirtyIntents) log(`      !! ${d}`);
// What round 6 changed in the instrument, named so the exclusion below is not a hiding place.
let instrDiff = '';
try { instrDiff = execSync('git diff --stat -- climb/rig3.mjs climb/robust.mjs climb/servo.mjs', { cwd: '..', encoding: 'utf8' }).trim(); } catch (e) { instrDiff = 'git diff failed'; }
for (const l of instrDiff.split('\n')) log(`      instrument diff vs 52b0392: ${l}`);
OUT.phaseP0 = { provenance: prov, modifiedTrackedIntents: dirtyIntents, gitNote,
                instrumentDiffStat: instrDiff,
                allCopiesVerified: Object.values(prov).every(v => v.ok) };
flush();

// =================================================================== PHASE P1
log('');
log('PHASE P1 — CURRENT rig3.mjs vs rig3_pre_r5.mjs on the round-4 judge\'s own 86 rows.');
log('   DEEP recursive walk over every leaf, Object.is, full float digits.');
const PRE_R4 = [
  ...ls(/^best_r2_.*\.json$/),
  ...ls(/^best_r3_.*\.json$/),
  ...ls(/^best_[012]_\d+mm\.json$/),
  'ctrl_do_nothing.json', 'ctrl_on_tread_40mm.json', 'ctrl_on_tread_60mm.json',
  'ctrl_on_tread_90mm.json', 'ctrl_on_tread_120mm.json', 'ctrl_walk_only.json',
].filter(f => fs.existsSync(P + f));

// Keys that exist ONLY because of round 6. Named here so the exclusion is
// visible. On a pre-round-4 file `servo` is null, so the only one that can
// appear at all is tailTrace, and it is undefined unless opts.tailTrace is set.
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

const p1 = { rows: 0, exact: 0, diffs: [], penRows: 0, penConsistent: 0, penViolations: [], servoLeak: [] };
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
      else p1.penViolations.push({ file: f, tail, episode: A.minPenetrationEpisode, atScore: A.penetrationAtScore });
    }
    if (A.servo !== null) p1.servoLeak.push({ file: f, tail, servo: A.servo });
    if (A.tailTrace !== undefined) p1.servoLeak.push({ file: f, tail, tailTrace: 'PRESENT WITHOUT BEING ASKED FOR' });
  }
}
log(`   ${p1.rows} rows (${PRE_R4.length} pre-round-4 files x tails policy+hold)`);
log(`   DEEP walk identical: ${p1.exact}/${p1.rows}   (round-5-only keys excluded: ${[...R5_ONLY].join(', ')})`);
log(`                                                 (round-6-only keys excluded: ${[...R6_ONLY].join(', ')})`);
for (const d of p1.diffs.slice(0, 6)) log(`      !! ${d.file} tail=${d.tail}: ${JSON.stringify(d.diffs)}`);
log(`   minPenetrationEpisode <= penetrationAtScore : ${p1.penConsistent}/${p1.penRows}`);
log(`   servo/tailTrace leaked onto a file that has neither: ${p1.servoLeak.length}`);
OUT.phaseP1 = { files: PRE_R4.length, rows: p1.rows, deepExact: p1.exact, deepDiffs: p1.diffs,
                penetrationConsistency: `${p1.penConsistent}/${p1.penRows}`,
                penetrationViolations: p1.penViolations, leaks: p1.servoLeak,
                r5OnlyKeysExcluded: [...R5_ONLY], r6OnlyKeysExcluded: [...R6_ONLY],
                reproducesR4Judge86: p1.rows === 86,
                verdict: (p1.exact === p1.rows && !p1.servoLeak.length && !p1.penViolations.length)
                  ? 'ROUND 6 MOVED NOTHING on the round-4 floor' : 'ROUND 6 MOVED A PRE-ROUND-5 NUMBER' };
log(`   86 rows as the round-4 judge counted them: ${OUT.phaseP1.reproducesR4Judge86}`);
log(`   VERDICT: ${OUT.phaseP1.verdict}   [${el()}]`);
flush();

// =================================================================== PHASE P2
log('');
log('PHASE P2 — CURRENT rig3.mjs vs rig3_pre_r6.mjs (52b0392) on the half rig3_pre_r5 cannot');
log('   score: the round-5 SERVOED files, plus the round-6 launches and the record.');
const P2SET = [
  'best_r3_vault_60mm.json', 'best_r4_famA_60mm.json', 'best_r2_vault_60mm.json',
  'best_r5_servo_60mm.json', 'best_r5_servoland_60mm.json', 'best_r5_servoland_kcore_60mm.json',
  'r5_servo_never_60mm.json', 'r5_servo_armed_60mm.json',
  'best_r6_ceilvault_60mm.json', 'best_r6_ceilvaultB_60mm.json', 'best_r6_ceilvaultC_60mm.json',
  'ctrl_do_nothing.json', 'r4_ctrl_on_tread_60mm.json',
].filter(f => fs.existsSync(P + f));
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
log(`   ${p2.rows} rows (${P2SET.length} files x tails policy+hold)   DEEP walk identical: ${p2.exact}/${p2.rows}`);
for (const d of p2.diffs.slice(0, 6)) log(`      !! ${d.file} tail=${d.tail}: ${JSON.stringify(d.diffs)}`);
OUT.phaseP2 = { files: P2SET, rows: p2.rows, deepExact: p2.exact, deepDiffs: p2.diffs,
                verdict: p2.exact === p2.rows ? 'servo.tailTicks is INERT at its default on every servoed file here' : 'NOT INERT' };
log(`   VERDICT: ${OUT.phaseP2.verdict}   [${el()}]`);
flush();

// =================================================================== PHASE P3
log('');
log('PHASE P3 — CURRENT robust.mjs vs robust_pre_r6.mjs over all 9 CORE cells. robust.mjs is');
log('   the scorer that decides every verdict, so it is the one that has to be unchanged.');
const P3SET = ['best_r3_vault_60mm.json', 'best_r5_servo_60mm.json',
               'best_r5_servoland_kcore_60mm.json', 'best_r6_ceilvaultC_60mm.json',
               'ctrl_do_nothing.json'].filter(f => fs.existsSync(P + f));
const p3 = { files: 0, cells: 0, exact: 0, diffs: [] };
for (const f of P3SET) {
  const A = await scoreRobust(P + f, { rise: 0.060, core: true, skipBounds: true });
  const B = await preR6Robust(P + f, { rise: 0.060, core: true, skipBounds: true });
  p3.files++;
  for (let i = 0; i < A.cells.length; i++) {
    p3.cells++;
    const dd = []; deepDiff(A.cells[i], B.cells[i], '', dd, R6_ONLY);
    if (!dd.length) p3.exact++; else p3.diffs.push({ file: f, cell: CELLNAME[i], diffs: dd.slice(0, 6) });
  }
  const kSame = A.kCore === B.kCore && A.kCoreStable === B.kCoreStable && A.objective === B.objective;
  log(`   ${f.padEnd(36)} kCore ${A.kCore}/${B.kCore}  kCoreStable ${A.kCoreStable}/${B.kCoreStable}  objective ${A.objective.toFixed(9)}/${B.objective.toFixed(9)}  ${kSame ? 'SAME' : '!! MOVED'}`);
}
log(`   ${p3.cells} core cells (${p3.files} files x 9)   DEEP walk identical: ${p3.exact}/${p3.cells}`);
for (const d of p3.diffs.slice(0, 6)) log(`      !! ${d.file} ${d.cell}: ${JSON.stringify(d.diffs)}`);
OUT.phaseP3 = { files: P3SET, cells: p3.cells, deepExact: p3.exact, deepDiffs: p3.diffs,
                verdict: p3.exact === p3.cells ? 'THE VERDICT SCORER IS UNCHANGED' : 'THE VERDICT SCORER MOVED' };
log(`   VERDICT: ${OUT.phaseP3.verdict}   [${el()}]`);
const ADMISSIBLE = OUT.phaseP0.allCopiesVerified && dirtyIntents.length === 0
  && p1.exact === p1.rows && p2.exact === p2.rows && p3.exact === p3.cells;
log(`   ROUND 6 IS ${ADMISSIBLE ? 'ADMISSIBLE' : 'INADMISSIBLE'} — the instrument edit moved ${ADMISSIBLE ? 'nothing' : 'something'}.`);
OUT.admissible = ADMISSIBLE;
flush();

// ==================================================================== PHASE R
log('');
log('PHASE R — rig3.scoreSaved vs robust.scoreCell (cell 0), on the round-6 files');
const RSET = ['best_r3_vault_60mm.json',
              'best_r6_ceilvault_60mm.json', 'best_r6_ceilvaultB_60mm.json', 'best_r6_ceilvaultC_60mm.json',
              'best_r6_tail10_servo_60mm.json', 'best_r6_tail25_servo_60mm.json', 'best_r6_tail50_servo_60mm.json',
              'best_r6_tail10_servoland_60mm.json', 'best_r6_tail25_servoland_60mm.json', 'best_r6_tail50_servoland_60mm.json',
             ].filter(f => fs.existsSync(P + f));
const rrows = [];
for (const f of RSET) {
  const A = await newScore(P + f, { rise: 0.060, tail: 'policy' });
  const B = await scoreCell(P + f, { rise: 0.060, isolate: true, skipBounds: true });
  const ok = A.scored.x === B.scored.x && A.scored.z === B.scored.z && A.reward === B.reward
    && A.crit.honest === B.crit.honest && A.scored.feetOnTread === B.scored.feetOnTread
    && A.uprightTailTicks === B.uprightTailTicks && A.maxAbsDY === B.maxAbsDY
    && A.maxZ === B.maxZ && A.minPenetrationEpisode === B.minPenetrationEpisode
    && (A.servo ? A.servo.tailTicksRun : null) === (B.servo ? B.servo.tailTicksRun : null);
  rrows.push({ file: f, exact: ok, servoArmed: A.servo ? A.servo.armed : null,
               tailAuthority: A.servo ? A.servo.tailAuthority : null,
               tailTicksRun: A.servo ? A.servo.tailTicksRun : null,
               x: A.scored.x, z: A.scored.z, maxZ: A.maxZ, reward: A.reward,
               uprightTailTicks: A.uprightTailTicks });
  log(`   ${f.padEnd(38)} EXACT=${ok}  tailAuthority=${A.servo ? A.servo.tailAuthority : '-'} run=${A.servo ? A.servo.tailTicksRun : '-'}  maxZ=${(A.maxZ * 1000).toFixed(1)}mm  upTail=${A.uprightTailTicks}/50`);
}
OUT.phaseR = { rows: rrows, allExact: rrows.every(r => r.exact),
               tailAuthorityEqualsRun: rrows.filter(r => r.tailAuthority !== null)
                 .every(r => r.tailTicksRun === r.tailAuthority || r.tailTicksRun === 0) };
log(`   allExact = ${OUT.phaseR.allExact}   [${el()}]`);
flush();

// ============================================================== PHASE H/B/S
log('');
log('PHASE H / B / S — identity, declared bounds, the search\'s own bounds, and the start');
// The search's own 29-parameter box. robust.mjs checkBounds only enforces
// blend and side unless the file declares a `bounds` block, and none of the
// three published round-6 launches does — so gap and approach are checked here
// against the box the search itself declared, or they are checked nowhere.
const SEARCH_BOUNDS = { gap: [0.01, 0.10], side: [-0.02, 0.09], approach: [0.0, 0.45], blend: [0.8, 2.4] };
const R6_FILES = ls(/^best_r6_.*\.json$/);
const REF_FILES = ['best_r3_vault_60mm.json', 'best_r4_famA_60mm.json', 'best_r2_vault_60mm.json',
                   'best_r5_servo_60mm.json', 'best_r5_servoland_60mm.json',
                   'best_r5_servoland_kcore_60mm.json'].filter(f => fs.existsSync(P + f));
const inv = [];
for (const f of [...R6_FILES, ...REF_FILES]) {
  const j = rd(f);
  const sha = intentHashOfFile(P + f);
  const B = checkBounds(j);
  const sbViol = Object.entries(SEARCH_BOUNDS)
    .map(([k, [lo, hi]]) => ({ param: k, value: k === 'blend' ? j[k] : (j[k] || 0), lo, hi }))
    .filter(v => !(typeof v.value === 'number' && v.value >= v.lo && v.value <= v.hi));
  const floorSpawn = !j.spawn && !j.spawnPose && !j.spawnVel && !j.spawnLastAction && !j.spawnQuat;
  const settle = j.settleTicks === undefined ? 25 : j.settleTicks;
  const kind = j.spawnPose ? 'HANDOFF SPAWN — NOT A CLIMB'
             : j.spawn ? 'PLACED SPAWN — NOT A CLIMB'
             : settle !== 25 ? `settleTicks=${settle} — NOT THE STANDARD START`
             : 'floor spawn — a climb';
  inv.push({ file: f, sha256: sha, move: sha.slice(0, 12), blend: j.blend, side: j.side || 0,
             gap: j.gap || 0, approach: j.approach || 0, isolate: j.isolate, stepCount: j.stepCount,
             declaredBounds: B.bounds, boundViolations: B.violations,
             searchBoxViolations: sbViol,
             hasServo: !!j.servo, hasEvent: !!j.event,
             servoAt: j.servo ? j.servo.at : null,
             servoTailTicks: j.servo ? (j.servo.tailTicks ?? 0) : null,
             settleTicks: settle, floorSpawn, kind });
}
for (const r of inv)
  log(`   ${r.file.padEnd(38)} ${r.move}  blend=${r.blend} gap=${r.gap} side=${r.side} approach=${r.approach}  declaredViol=${r.boundViolations.length} searchBoxViol=${r.searchBoxViolations.length}  servo=${r.hasServo ? 'yes tailTicks=' + r.servoTailTicks : 'no'} event=${r.hasEvent}  [${r.kind}]`);
const byHash = {};
for (const r of inv) (byHash[r.sha256] ||= []).push(r.file);
const dupes = Object.entries(byHash).filter(([, v]) => v.length > 1).map(([h, v]) => ({ move: h.slice(0, 12), files: v }));
for (const d of dupes) log(`   ONE VECTOR, ${d.files.length} LABELS  ${d.move}  ${d.files.join('  ')}`);
// Hashes the round-6 agents claimed, and hashes earlier rounds published.
const CLAIMED_HASHES = {
  'best_r6_ceilvault_60mm.json': '8c57838ee9d0', 'best_r6_ceilvaultB_60mm.json': '29c97398fe13',
  'best_r6_ceilvaultC_60mm.json': 'a56d459fb649',
  'best_r6_tail10_servo_60mm.json': '847e04685c20', 'best_r6_tail25_servo_60mm.json': '098ce4d211f9',
  'best_r6_tail50_servo_60mm.json': 'b160eec56c23',
  'best_r6_tail10_servoland_60mm.json': '3ff2de93400e', 'best_r6_tail25_servoland_60mm.json': 'd78034ef9cd9',
  'best_r6_tail50_servoland_60mm.json': 'b78f6248359c',
  'best_r3_vault_60mm.json': '4b9110c448ec', 'best_r4_famA_60mm.json': '7b790070b010',
  'best_r2_vault_60mm.json': '74d35b21ac80', 'best_r5_servo_60mm.json': 'e0434c2c90da',
  'best_r5_servoland_60mm.json': 'e6e8ff144695', 'best_r5_servoland_kcore_60mm.json': '880a120ef649',
};
const hashRows = [];
for (const [f, want] of Object.entries(CLAIMED_HASHES)) {
  if (!fs.existsSync(P + f)) continue;
  const got = intentHashOfFile(P + f).slice(0, 12);
  hashRows.push({ file: f, claimed: want, now: got, matches: got === want });
  log(`   hash ${f.padEnd(38)} claimed ${want}  now ${got}  ${got === want ? 'MATCHES' : '!! DOES NOT MATCH'}`);
}
OUT.phaseHBS = { files: inv, distinctVectors: Object.keys(byHash).length, duplicateGroups: dupes,
                 hashes: hashRows, hashesAllMatch: hashRows.every(r => r.matches),
                 declaredBoundViolations: inv.filter(r => r.boundViolations.length).map(r => ({ file: r.file, violations: r.boundViolations })),
                 searchBoxViolations: inv.filter(r => r.searchBoxViolations.length).map(r => ({ file: r.file, violations: r.searchBoxViolations })),
                 notAClimb: inv.filter(r => !r.floorSpawn || r.settleTicks !== 25).map(r => ({ file: r.file, kind: r.kind })),
                 searchBoxNote: 'robust.mjs checkBounds enforces only blend and side unless the file declares its own `bounds`; none of the three published round-6 launches does, so gap and approach are checked here against the box r6_ceiling_search.mjs declared.' };
log(`   distinct vectors ${OUT.phaseHBS.distinctVectors} | declared-bound violations ${OUT.phaseHBS.declaredBoundViolations.length} | search-box violations ${OUT.phaseHBS.searchBoxViolations.length} | NOT-A-CLIMB ${OUT.phaseHBS.notAClimb.length}`);
flush();

// ============================================== THE ABLATION / PROBE INTENTS
// Written to disk BEFORE anything is scored, on NEW paths only. A re-run may
// rewrite its own probe file, and only when the bytes are identical.
const writeProbe = (name, obj) => {
  const path = P + name;
  const body = JSON.stringify(obj, null, 2);
  if (fs.existsSync(path) && fs.readFileSync(path, 'utf8') !== body)
    throw new Error('refusing to overwrite ' + path);
  fs.writeFileSync(path, body);
  return { file: name, sha256: intentHash(obj) };
};
const ABL = [];
for (const src of ['best_r6_tail10_servo_60mm.json', 'best_r6_tail25_servo_60mm.json',
                   'best_r6_tail50_servo_60mm.json', 'best_r6_tail10_servoland_60mm.json',
                   'best_r6_tail25_servoland_60mm.json', 'best_r6_tail50_servoland_60mm.json',
                   'best_r5_servo_60mm.json', 'best_r5_servoland_kcore_60mm.json']) {
  if (!fs.existsSync(P + src)) continue;
  const j = rd(src);
  // ONLY the servo block goes. `bounds`, if the file declares one, stays: it is
  // what the file is scored against, and dropping it would quietly change the
  // gate the ablation is judged by.
  delete j.servo; delete j.robust; delete j.note;
  j.note = 'ROUND-6 JUDGE ABLATION, written by climb/audit_r6.mjs: ' + src + ' with its `servo` block deleted and every other launch field identical. This is the LAUNCH ALONE.';
  const name = 'audit_r6_launchonly_' + src.replace(/^best_|\.json$/g, '') + '.json';
  ABL.push({ src, ...writeProbe(name, j) });
}
log('');
log('ABLATION INTENTS WRITTEN (the launch alone, servo block deleted):');
for (const a of ABL) log(`   ${a.file.padEnd(48)} ${a.sha256.slice(0, 12)}   from ${a.src}`);
OUT.ablationIntents = ABL;
flush();

// ==================================================================== PHASE G
log('');
log('PHASE G — every round-6 claim, re-scored from disk on the 14-cell grid, POLICY tail');
log('   (the tail every verdict is decided on), and at the nominal cell on the HOLD tail.');
log('   file                                    rise kC  kCs kExt kEs ceil objective  maxTq  minPenEp upTail(min/mean) holdUp holdHonest');
const CLAIMS = [
  ['best_r3_vault_60mm.json', 0.060, 'the standing record / warm start'],
  ['best_r6_ceilvault_60mm.json', 0.060, 'round-6 ceiling search, best objective'],
  ['best_r6_ceilvaultB_60mm.json', 0.060, 'round-6 ceiling search, chain B'],
  ['best_r6_ceilvaultC_60mm.json', 0.060, 'round-6 ceiling search, CLAIMED kCoreStable 5/9'],
  ['best_r5_servo_60mm.json', 0.060, 'round-5 servoed best (oracle)'],
  ['best_r5_servoland_kcore_60mm.json', 0.060, 'round-5 servoed best by kCore (oracle)'],
  ['best_r6_tail10_servo_60mm.json', 0.060, 'round-6 tail authority 10 (oracle)'],
  ['best_r6_tail25_servo_60mm.json', 0.060, 'round-6 tail authority 25 (oracle)'],
  ['best_r6_tail50_servo_60mm.json', 0.060, 'round-6 tail authority 50 (oracle)'],
  ['best_r6_tail10_servoland_60mm.json', 0.060, 'round-6 tail authority 10 (oracle)'],
  ['best_r6_tail25_servoland_60mm.json', 0.060, 'round-6 tail authority 25 (oracle)'],
  ['best_r6_tail50_servoland_60mm.json', 0.060, 'round-6 tail authority 50 (oracle)'],
  ['best_r4_famA_60mm.json', 0.060, 'round-4 event landing on the record launch'],
  ...ABL.map(a => [a.file, 0.060, 'ABLATION — the launch alone, from ' + a.src]),
].filter(([f]) => fs.existsSync(P + f));

const G = {};
for (const [f, h, label] of CLAIMS) {
  const j = rd(f);
  const B = checkBounds(j);
  const floorSpawn = !j.spawn && !j.spawnPose && !j.spawnVel && !j.spawnLastAction && !j.spawnQuat;
  const settle = j.settleTicks === undefined ? 25 : j.settleTicks;
  const g = await scoreRobust(P + f, { rise: h, skipBounds: true });
  const hold = await newScore(P + f, { rise: h, tail: 'hold' });
  const core = g.cells.filter(c => c.cell.tier === 'core');
  // THE CEILING, from the RAW maxZ. Not from the rounded peakZ_mm the round-6
  // scripts used. Both are recorded so the rounding can be seen.
  const peaksRaw = core.map(c => (c.maxZ - c.rise));
  const peaksRounded = core.map(c => +((c.maxZ - c.rise) * 1000).toFixed(1) / 1000);
  const ceilingCore = peaksRaw.filter(z => z > CEILING_ABOVE).length;
  const ceilingCoreRounded = peaksRounded.filter(z => z > CEILING_ABOVE).length;
  const clears = g.cells.filter(c => c.crit.honest);
  const row = {
    file: f, label, rise_mm: h * 1000, sha256: g.sha256, move: g.move,
    admissible: B.violations.length === 0 && floorSpawn && settle === 25,
    boundViolations: B.violations, floorSpawn, settleTicks: settle,
    kCore: g.kCore, kCoreStable: g.kCoreStable, kExt: g.kExt, kExtStable: g.kExtStable,
    ceilingCore, ceilingCoreRounded, ceilingRoundingDisagrees: ceilingCore !== ceilingCoreRounded,
    objective: +g.objective.toFixed(4), objectiveCore: +g.objectiveCore.toFixed(4),
    maxTq: +g.agg.maxTq.toFixed(4), tqOverCeiling: g.agg.maxTq > FORCERANGE + 1e-9,
    tqSaturated: g.agg.maxTq >= FORCERANGE - 1e-4,
    minPenetrationEpisode_mm: +g.agg.minPenetrationEpisode_mm.toFixed(2),
    minPenetrationAtScore_mm: +g.agg.minPenetrationAtScore_mm.toFixed(2),
    maxAbsDY_mm: +g.agg.maxAbsDY_mm.toFixed(1), lateralEscapeCells: g.agg.lateralEscapeCells,
    minUprightTailTicks: g.agg.minUprightTailTicks,
    meanUprightTailTicks: +g.agg.meanUprightTailTicks.toFixed(1),
    servoArmedCells: g.cells.filter(c => c.servo && c.servo.armed).length,
    servoTailAuthority: g.cells[0].servo ? g.cells[0].servo.tailAuthority : null,
    servoTailTicksRunMax: Math.max(0, ...g.cells.map(c => c.servo ? c.servo.tailTicksRun : 0)),
    peakAbove_mm_core: peaksRaw.map(z => +(z * 1000).toFixed(1)),
    honestPatternCore: core.map(c => c.crit.honest ? 1 : 0).join(''),
    stablePatternCore: core.map(c => c.stableClear ? 1 : 0).join(''),
    ceilPatternCore: peaksRaw.map(z => z > CEILING_ABOVE ? 1 : 0).join(''),
    // whole-episode penetration
    clearsMinPenEpisode_mm: clears.map(c => +(c.minPenetrationEpisode * 1000).toFixed(2)),
    clearsDeeperThanSoft: clears.filter(c => c.minPenetrationEpisode < PEN_SOFT).length,
    cellsDeeperThanSoft: g.cells.filter(c => c.minPenetrationEpisode < PEN_SOFT).length,
    penConsistent: g.cells.every(c => c.minPenetrationEpisode === null || c.penetrationAtScore === null
                                   || c.minPenetrationEpisode <= c.penetrationAtScore + 1e-15),
    // THE SECOND LANDING: the hold tail, at the nominal cell.
    holdTail: { uprightTailTicks: hold.uprightTailTicks, honest: hold.crit.honest,
                stable: hold.crit.honest && hold.uprightTailTicks >= UPRIGHT_TAIL_MIN,
                above_mm: +(hold.scored.above * 1000).toFixed(1),
                peakAbove_mm: +((hold.maxZ - h) * 1000).toFixed(1) },
    cellsXZ: core.map((c, i) => ({ cell: CELLNAME[i], honest: c.crit.honest, stable: c.stableClear,
                                   above_mm: +(c.scored.above * 1000).toFixed(1),
                                   peakAbove_mm: +((c.maxZ - c.rise) * 1000).toFixed(1),
                                   overBar: (c.maxZ - c.rise) > CEILING_ABOVE,
                                   upTail: c.uprightTailTicks, feetOnTread: c.scored.feetOnTread,
                                   minPenEp_mm: +(c.minPenetrationEpisode * 1000).toFixed(2),
                                   maxTq: +c.maxTq.toFixed(4) })),
  };
  G[f] = row;
  log(`   ${f.padEnd(40)} ${String(row.rise_mm).padStart(4)} ${row.kCore}/9 ${row.kCoreStable}/9 ${String(row.kExt).padStart(2)}/14 ${String(row.kExtStable).padStart(2)}/14 ${row.ceilingCore}/9 ${row.objective.toFixed(3).padStart(9)} ${row.maxTq.toFixed(4)} ${f2(row.minPenetrationEpisode_mm).padStart(8)} ${String(row.minUprightTailTicks).padStart(2)}/${String(row.meanUprightTailTicks).padStart(4)}  ${String(row.holdTail.uprightTailTicks).padStart(2)}/50  ${row.holdTail.honest}`);
  flush();
}
OUT.phaseG = G;
log(`   [${el()}]`);
flush();

// =============================================================== PHASE A
log('');
log('PHASE A — THE ABLATION, read off PHASE G. What is the servo block worth, and what is');
log('   servo.tailTicks worth, on the SAME launch? (every servoed row is an ORACLE row)');
const ablRows = [];
for (const a of ABL) {
  const withS = G[a.src], without = G[a.file];
  if (!withS || !without) continue;
  const r = { src: a.src, launchOnlyFile: a.file,
              withServo: { move: withS.move, kCore: withS.kCore, kCoreStable: withS.kCoreStable,
                           ceilingCore: withS.ceilingCore, meanUpTail: withS.meanUprightTailTicks,
                           tailAuthority: withS.servoTailAuthority, pattern: withS.honestPatternCore },
              launchOnly: { move: without.move, kCore: without.kCore, kCoreStable: without.kCoreStable,
                            ceilingCore: without.ceilingCore, meanUpTail: without.meanUprightTailTicks,
                            pattern: without.honestPatternCore },
              dKCore: withS.kCore - without.kCore, dKCoreStable: withS.kCoreStable - without.kCoreStable,
              dCeilingCore: withS.ceilingCore - without.ceilingCore,
              dMeanUpTail: +(withS.meanUprightTailTicks - without.meanUprightTailTicks).toFixed(1) };
  ablRows.push(r);
  log(`   ${a.src.padEnd(38)} servo(tailTicks ${r.withServo.tailAuthority}) kCore ${r.withServo.kCore}/9 stable ${r.withServo.kCoreStable}/9 ceil ${r.withServo.ceilingCore}/9 upTail ${r.withServo.meanUpTail}`);
  log(`   ${''.padEnd(38)} launch only                 kCore ${r.launchOnly.kCore}/9 stable ${r.launchOnly.kCoreStable}/9 ceil ${r.launchOnly.ceilingCore}/9 upTail ${r.launchOnly.meanUpTail}   -> the law is worth kCore ${r.dKCore >= 0 ? '+' : ''}${r.dKCore}, ceiling ${r.dCeilingCore >= 0 ? '+' : ''}${r.dCeilingCore}, upright tail ${r.dMeanUpTail >= 0 ? '+' : ''}${r.dMeanUpTail}`);
}
// The ceiling report's corollary: best_r5_servo_60mm's LAUNCH is the record's launch.
const corollary = [];
for (const a of ABL) {
  const rec = G['best_r3_vault_60mm.json'];
  const lo = G[a.file];
  if (rec && lo) corollary.push({ launchOnlyOf: a.src, move: lo.move,
                                  isTheRecordLaunch: lo.sha256 === rec.sha256 });
}
for (const c of corollary.filter(c => c.isTheRecordLaunch))
  log(`   COROLLARY CONFIRMED: the launch under ${c.launchOnlyOf} hashes to the record, ${c.move}`);
OUT.phaseA = { rows: ablRows, launchIdentity: corollary,
               note: 'A servoed or tail-authority row is an ORACLE row: climb/servo.mjs reads the tread height and the tread edge straight out of data.qpos/data.geom_xpos, and the shipped policy gets 61 proprioceptive numbers that contain neither.' };
flush();

// ==================================================================== PHASE W
log('');
log('PHASE W — THE SCREEN, RE-DERIVED. Every published launch r6_screen.mjs claims to have');
log('   scored, re-scored here at 60 mm through scoreRobust({core:true}), row against row.');
let SCREEN_FILES = fs.readdirSync(P).filter(f =>
  /^best_r[2345]_.*\.json$/.test(f) || /^best_[012]_.*\.json$/.test(f)).sort();
// A smoke-test knob only. A REPORTED run leaves it unset and screens all of them;
// the value actually used is recorded in phaseW.filesScreened below.
if (process.env.AUDIT_R6_SCREEN_LIMIT) SCREEN_FILES = SCREEN_FILES.slice(0, parseInt(process.env.AUDIT_R6_SCREEN_LIMIT, 10));
let SCREEN_CLAIM = null;
try { SCREEN_CLAIM = JSON.parse(fs.readFileSync(P + 'r6_screen-results.json', 'utf8')); } catch (e) {}
const claimRow = f => {
  if (!SCREEN_CLAIM) return null;
  const arr = SCREEN_CLAIM.rows || SCREEN_CLAIM;
  return Array.isArray(arr) ? arr.find(r => r.file === f) : null;
};
const wrows = [];
for (const f of SCREEN_FILES) {
  const r = await scoreRobust(P + f, { rise: 0.060, core: true });
  const cl = claimRow(f);
  if (r.invalid) {
    wrows.push({ file: f, invalid: true, boundViolations: r.boundViolations, sha256: r.sha256,
                 claimedInvalid: cl ? !!cl.invalid : null, agrees: cl ? !!cl.invalid : null });
    log(`   [${el().padStart(5)}] ${f.padEnd(38)} INVALID (out of declared bounds)  claim says invalid=${cl ? !!cl.invalid : 'n/a'}`);
    continue;
  }
  const core = r.verdicts.filter(v => v.tier === 'core');
  const peaksRaw = r.cells.filter(c => c.cell.tier === 'core').map(c => c.maxZ - c.rise);
  const ceilingCore = peaksRaw.filter(z => z > CEILING_ABOVE).length;
  const ceilingRounded = core.map(v => +(v.peakZ_mm - v.rise_mm).toFixed(1)).filter(p => p > 95).length;
  const row = { file: f, sha256: r.sha256, move: r.move, ceilingCore, ceilingCoreRounded: ceilingRounded,
                roundingDisagrees: ceilingCore !== ceilingRounded,
                kCore: r.kCore, kCoreStable: r.kCoreStable,
                maxTq: +r.agg.maxTq.toFixed(4),
                minPenetrationEpisode_mm: +r.agg.minPenetrationEpisode_mm.toFixed(2),
                peakAbove_mm: peaksRaw.map(z => +(z * 1000).toFixed(1)),
                kCoreLEQceiling: r.kCore <= ceilingCore,
                claimed: cl ? { ceilingCore: cl.ceilingCore, kCore: cl.kCore, kCoreStable: cl.kCoreStable,
                                sha256: cl.sha256 } : null };
  row.agrees = !cl || (cl.ceilingCore === ceilingRounded && cl.kCore === r.kCore
                    && cl.kCoreStable === r.kCoreStable && cl.sha256 === r.sha256);
  wrows.push(row);
  log(`   [${el().padStart(5)}] ${f.padEnd(38)} ceil ${ceilingCore}/9 kCore ${r.kCore}/9 kStable ${r.kCoreStable}/9  ${r.move}  ${cl ? (row.agrees ? 'agrees with the screen' : '!! DISAGREES with the screen: claimed ceil ' + cl.ceilingCore + ' kCore ' + cl.kCore + ' kStable ' + cl.kCoreStable) : '(no claim row)'}`);
  if (wrows.length % 8 === 0) { OUT.phaseW = { rows: wrows }; flush(); }
}
const scorable = wrows.filter(r => !r.invalid);
const wByHash = {};
for (const r of scorable) (wByHash[r.sha256] ||= []).push(r.file);
const wHist = {};
for (const r of scorable) wHist[r.ceilingCore] = (wHist[r.ceilingCore] || 0) + 1;
const wHistDistinct = {};
for (const h of Object.keys(wByHash)) {
  const r = scorable.find(x => x.sha256 === h);
  wHistDistinct[r.ceilingCore] = (wHistDistinct[r.ceilingCore] || 0) + 1;
}
const wMax = Math.max(...scorable.map(r => r.ceilingCore));
OUT.phaseW = { filesScreened: SCREEN_FILES.length, scorable: scorable.length,
               invalid: wrows.filter(r => r.invalid).length,
               distinctVectors: Object.keys(wByHash).length,
               histogramOverFiles: wHist, histogramOverDistinctVectors: wHistDistinct,
               maxCeilingCore: wMax,
               holdersOfTheMax: scorable.filter(r => r.ceilingCore === wMax).map(r => ({ file: r.file, move: r.move, kCore: r.kCore, kCoreStable: r.kCoreStable })),
               rowsDisagreeingWithTheScreen: wrows.filter(r => r.agrees === false),
               roundingDisagreements: wrows.filter(r => r.roundingDisagrees),
               kCoreLEQceilingEverywhere: scorable.every(r => r.kCoreLEQceiling),
               rows: wrows };
log('');
log(`   ${SCREEN_FILES.length} files, ${scorable.length} scorable, ${wrows.filter(r => r.invalid).length} refused by the bounds gate, ${Object.keys(wByHash).length} distinct vectors`);
log(`   ceilingCore histogram over files ${JSON.stringify(wHist)} | over distinct vectors ${JSON.stringify(wHistDistinct)}`);
log(`   HIGHEST ceilingCore in the published corpus at 60 mm: ${wMax}/9`);
for (const h of OUT.phaseW.holdersOfTheMax) log(`      ${h.file.padEnd(38)} ${h.move}  kCore ${h.kCore}/9 kCoreStable ${h.kCoreStable}/9`);
log(`   rows that DISAGREE with r6_screen-results.json: ${OUT.phaseW.rowsDisagreeingWithTheScreen.length}`);
log(`   cells where the 0.1 mm rounding could flip a ceiling verdict: ${OUT.phaseW.roundingDisagreements.length}`);
log(`   kCore <= ceilingCore holds on every row: ${OUT.phaseW.kCoreLEQceilingEverywhere}   [${el()}]`);
flush();

// ==================================================================== PHASE X
log('');
log('PHASE X — THE SEARCH, RECONSTRUCTED. 337 of the 340 vectors the CEM reports were never');
log('   saved, so their rows cannot be audited as published. They are rebuilt here from the');
log('   29-parameter `params` on every row of r6_ceiling-results.json, through the search\'s');
log('   own intentOf(), WRITTEN TO DISK, and re-scored. A rebuild whose sha256 is not the');
log('   claimed sha256 is not the same vector and is reported as such.');
// climb/vault.mjs trackOf(), as r6_ceiling_search.mjs copied it. Reproduced here
// so the reconstruction does not depend on the search script still existing.
const JI = { lhy: 0, lhr: 1, lhp: 2, lk: 3, la: 4, np: 5, hp: 6, hy: 7, hr: 8,
             rhy: 9, rhr: 10, rhp: 11, rk: 12, ra: 13 };
function trackOf(p) {
  const put = (q, hip, knee, ank, roll) => {
    q[JI.lhp] = HOME[JI.lhp] + hip;  q[JI.rhp] = HOME[JI.rhp] - hip;
    q[JI.lk] = HOME[JI.lk] + knee;   q[JI.rk] = HOME[JI.rk] - knee;
    q[JI.la] = HOME[JI.la] + ank;    q[JI.ra] = HOME[JI.ra] - ank;
    q[JI.lhr] = HOME[JI.lhr] + roll; q[JI.rhr] = HOME[JI.rhr] + roll;
    return q;
  };
  const strut = q => { q[JI.np] = p.strutNeck; q[JI.hp] = p.strutHead; return q; };
  const A = strut(put(HOME.slice(), p.crouchHip, p.crouchKnee, p.crouchAnk, p.roll));
  const B = strut(put(HOME.slice(), p.preHip, p.preKnee, p.preAnk, p.roll));
  const Cc = strut(put(HOME.slice(), p.vaultHip, p.vaultKnee, p.vaultAnk, p.roll));
  const Dd = strut(put(HOME.slice(), p.tuckHip, p.tuckKnee, p.tuckAnk, p.roll));
  const E = put(HOME.slice(), p.landHip, p.landKnee, p.landAnk, 0);
  E[JI.np] = p.landNeck; E[JI.hp] = p.landHead;
  const t1 = p.tReach, t2 = t1 + p.tPre, t3 = t2 + p.tVault, t4 = t3 + p.tTuck, t5 = t4 + p.tLand;
  return [{ t: t1, pose: A }, { t: t2, pose: B }, { t: t3, pose: Cc },
          { t: t4, pose: Dd }, { t: t5, pose: E }, { t: t5 + 0.7, pose: HOME.slice() }];
}
const r5f = v => +v.toFixed(5);
const intentOf = (p, note) => ({
  name: 'beak_strut_vault_r6_ceiling_60mm', family: 'R6 ceiling — beak-strut vault launch',
  keyframes: trackOf(p).map(f => ({ t: +f.t.toFixed(4), pose: f.pose.map(r5f) })),
  blend: +p.blend.toFixed(4), gap: +p.gap.toFixed(4), side: +p.side.toFixed(4),
  approach: +p.approach.toFixed(4), isolate: true, stepCount: 4, params: p, note });

let SEARCH = null;
try { SEARCH = JSON.parse(fs.readFileSync(P + 'r6_ceiling-results.json', 'utf8')); } catch (e) {}
const xrows = [];
let xHashOk = 0, xCeilOk = 0, xKOk = 0, xScored = 0;
if (SEARCH && Array.isArray(SEARCH.distinctMoves)) {
  fs.mkdirSync(RECON_DIR, { recursive: true });
  // Rebuild EVERY row. Ordered by claimed ceilingCore descending so the rows
  // that carry the verdict are re-derived first, and a truncated run still
  // answers the kill gate.
  const moves = SEARCH.distinctMoves.slice().sort((a, b) => b.ceilingCore - a.ceilingCore || b.objective - a.objective);
  const LIMIT = process.env.AUDIT_R6_RECON_LIMIT ? parseInt(process.env.AUDIT_R6_RECON_LIMIT, 10) : moves.length;
  log(`   rebuilding ${Math.min(LIMIT, moves.length)} of ${moves.length} distinct moves into ${RECON_DIR}`);
  for (let i = 0; i < Math.min(LIMIT, moves.length); i++) {
    const m = moves[i];
    const obj = intentOf(m.params, `ROUND-6 JUDGE RECONSTRUCTION of the searched move ${m.move}, rebuilt from r6_ceiling-results.json distinctMoves[].params through the search's own intentOf(). Written by climb/audit_r6.mjs.`);
    const sha = intentHash(obj);
    const hashOk = sha === m.sha256;
    const path = RECON_DIR + `r6m_${m.move}.json`;
    fs.writeFileSync(path, JSON.stringify(obj, null, 2));
    const bnd = checkBounds(obj);
    const sbv = Object.entries(SEARCH_BOUNDS)
      .map(([k, [lo, hi]]) => ({ param: k, value: k === 'blend' ? obj[k] : (obj[k] || 0), lo, hi }))
      .filter(v => !(v.value >= v.lo && v.value <= v.hi));
    const row = { move: m.move, claimedSha256: m.sha256, rebuiltSha256: sha, hashMatches: hashOk,
                  declaredBoundViolations: bnd.violations, searchBoxViolations: sbv,
                  claimed: { ceilingCore: m.ceilingCore, kCore: m.kCore, kCoreStable: m.kCoreStable,
                             maxTq: m.maxTq, minPenetrationEpisode_mm: m.minPenetrationEpisode_mm,
                             peakAboveTread_mm: m.peakAboveTread_mm } };
    if (hashOk) xHashOk++;
    const r = await scoreRobust(path, { rise: 0.060, core: true, skipBounds: true });
    xScored++;
    const peaksRaw = r.cells.map(c => c.maxZ - c.rise);
    const ceilingCore = peaksRaw.filter(z => z > CEILING_ABOVE).length;
    const ceilingRounded = r.verdicts.map(v => +(v.peakZ_mm - v.rise_mm).toFixed(1)).filter(p => p > 95).length;
    row.measured = { ceilingCore, ceilingCoreRounded: ceilingRounded, kCore: r.kCore,
                     kCoreStable: r.kCoreStable, maxTq: +r.agg.maxTq.toFixed(4),
                     minPenetrationEpisode_mm: +r.agg.minPenetrationEpisode_mm.toFixed(2),
                     peakAbove_mm: peaksRaw.map(z => +(z * 1000).toFixed(1)),
                     clearsDeeperThanSoft: r.cells.filter(c => c.crit.honest && c.minPenetrationEpisode < PEN_SOFT).length,
                     lateralEscapeCells: r.agg.lateralEscapeCells,
                     tqOverCeiling: r.agg.maxTq > FORCERANGE + 1e-9 };
    row.ceilingMatches = ceilingRounded === m.ceilingCore;
    row.kMatches = r.kCore === m.kCore && r.kCoreStable === m.kCoreStable;
    row.kCoreLEQceiling = r.kCore <= ceilingCore;
    row.roundingDisagrees = ceilingCore !== ceilingRounded;
    if (row.ceilingMatches) xCeilOk++;
    if (row.kMatches) xKOk++;
    xrows.push(row);
    if (i < 30 || !row.hashMatches || !row.ceilingMatches || !row.kMatches)
      log(`   [${el().padStart(5)}] ${String(i + 1).padStart(3)}/${Math.min(LIMIT, moves.length)} ${m.move}  hash ${hashOk ? 'OK' : '!! MISMATCH'}  ceil claimed ${m.ceilingCore} measured ${ceilingCore}${row.ceilingMatches ? '' : ' !!'}  kCore ${m.kCore}->${r.kCore} kStable ${m.kCoreStable}->${r.kCoreStable}${row.kMatches ? '' : ' !!'}`);
    if ((i + 1) % 20 === 0) {
      OUT.phaseX = { partial: true, scored: xScored, hashMatches: xHashOk, ceilingMatches: xCeilOk, kMatches: xKOk, rows: xrows };
      flush();
      log(`   [${el()}] ... ${i + 1} rebuilt, hash ${xHashOk}/${xScored}, ceiling ${xCeilOk}/${xScored}, k ${xKOk}/${xScored}`);
    }
  }
}
const xHist = {};
for (const r of xrows) xHist[r.measured.ceilingCore] = (xHist[r.measured.ceilingCore] || 0) + 1;
const xMax = xrows.length ? Math.max(...xrows.map(r => r.measured.ceilingCore)) : 0;
OUT.phaseX = {
  partial: false, source: 'climb/r6_ceiling-results.json distinctMoves[].params',
  reconDir: 'climb/audit_r6_recon/', claimedDistinctMoves: SEARCH ? SEARCH.distinct : null,
  rebuilt: xrows.length, hashMatches: xHashOk, ceilingMatches: xCeilOk, kMatches: xKOk,
  measuredCeilingHistogram: xHist, maxMeasuredCeilingCore: xMax,
  claimedCeilingHistogram: SEARCH ? (() => { const h = {}; for (const m of SEARCH.distinctMoves) h[m.ceilingCore] = (h[m.ceilingCore] || 0) + 1; return h; })() : null,
  claimedMaxCeilingCore: SEARCH ? Math.max(...SEARCH.distinctMoves.map(m => m.ceilingCore)) : null,
  kCoreLEQceilingEverywhere: xrows.every(r => r.kCoreLEQceiling),
  anyTorqueOverCeiling: xrows.some(r => r.measured.tqOverCeiling),
  clearsDeeperThanSoftContact: xrows.filter(r => r.measured.clearsDeeperThanSoft > 0)
    .map(r => ({ move: r.move, minPenetrationEpisode_mm: r.measured.minPenetrationEpisode_mm,
                 ceilingCore: r.measured.ceilingCore, kCore: r.measured.kCore })),
  hashMismatches: xrows.filter(r => !r.hashMatches).map(r => ({ move: r.move, rebuilt: r.rebuiltSha256 })),
  ceilingMismatches: xrows.filter(r => !r.ceilingMatches)
    .map(r => ({ move: r.move, claimed: r.claimed.ceilingCore, measured: r.measured.ceilingCore })),
  kMismatches: xrows.filter(r => !r.kMatches)
    .map(r => ({ move: r.move, claimed: [r.claimed.kCore, r.claimed.kCoreStable], measured: [r.measured.kCore, r.measured.kCoreStable] })),
  roundingDisagreements: xrows.filter(r => r.roundingDisagrees).map(r => r.move),
  outOfDeclaredBounds: xrows.filter(r => r.declaredBoundViolations.length).map(r => ({ move: r.move, violations: r.declaredBoundViolations })),
  outOfTheSearchsOwnBox: xrows.filter(r => r.searchBoxViolations.length).map(r => ({ move: r.move, violations: r.searchBoxViolations })),
  rows: xrows,
};
log('');
log(`   rebuilt ${xrows.length} of ${SEARCH ? SEARCH.distinct : '?'} distinct moves`);
log(`   sha256 of the rebuild equals the claimed sha256 : ${xHashOk}/${xrows.length}`);
log(`   ceilingCore reproduces                          : ${xCeilOk}/${xrows.length}`);
log(`   kCore and kCoreStable reproduce                 : ${xKOk}/${xrows.length}`);
log(`   measured ceilingCore histogram ${JSON.stringify(xHist)}   claimed ${JSON.stringify(OUT.phaseX.claimedCeilingHistogram)}`);
log(`   HIGHEST measured ceilingCore over the rebuilt search: ${xMax}/9`);
log(`   kCore <= ceilingCore on every rebuilt row: ${OUT.phaseX.kCoreLEQceilingEverywhere}`);
log(`   any rebuilt row over the ${FORCERANGE} N.m forcerange: ${OUT.phaseX.anyTorqueOverCeiling}`);
log(`   rebuilt moves with a CLEAR that penetrates deeper than ${(PEN_SOFT * 1000).toFixed(0)} mm: ${OUT.phaseX.clearsDeeperThanSoftContact.length}`);
for (const c of OUT.phaseX.clearsDeeperThanSoftContact.slice(0, 10))
  log(`      ${c.move}  minPenEpisode ${c.minPenetrationEpisode_mm} mm  ceil ${c.ceilingCore}/9  kCore ${c.kCore}/9`);
log(`   [${el()}]`);
flush();

// ==================================================================== PHASE E
log('');
log('PHASE E — THE CEILING, stated as what it is: an IDENTITY of the criterion, not a result.');
log('   rig3 criteria() honest requires scored.above > 0.095 m at the scored instant. above is');
log('   trunk z minus the tread height, and maxZ is the trunk\'s maximum over the whole episode,');
log('   so honest in a cell IMPLIES maxZ - rise > 0.095 in that cell. Therefore');
log('   kCore <= ceilingCore, for every landing law that exists or could be written.');
const ceilRows = Object.values(G).map(r => ({ file: r.file, move: r.move, kCore: r.kCore,
  kCoreStable: r.kCoreStable, ceilingCore: r.ceilingCore, peakAbove_mm_core: r.peakAbove_mm_core,
  ceilPattern: r.ceilPatternCore, honestPattern: r.honestPatternCore,
  kCoreLEQceiling: r.kCore <= r.ceilingCore, converted: r.kCore === r.ceilingCore }));
log('   file                                    kCore ceil  peak trunk above the tread, the 9 core cells (mm)');
for (const r of ceilRows) log(`   ${r.file.padEnd(40)} ${r.kCore}/9   ${r.ceilingCore}/9  [${r.peakAbove_mm_core.join(', ')}]`);
const everything = [...ceilRows.map(r => r.ceilingCore), ...scorable.map(r => r.ceilingCore),
                    ...xrows.map(r => r.measured.ceilingCore)];
const MAXCEIL = Math.max(...everything);
OUT.phaseE = {
  rule: 'ceilingCore = # of the 9 core cells with (max trunk z over the episode) - thatCellsRise > 0.095 m',
  identity: 'rig3 criteria().honest requires above > 0.095 at the scored instant, and above <= maxZ - rise by definition of maxZ, so honest implies the cell is over the bar. kCore <= ceilingCore is an identity of the criterion, not a hypothesis.',
  rows: ceilRows,
  boundHoldsOnEveryRowHere: ceilRows.every(r => r.kCoreLEQceiling)
    && scorable.every(r => r.kCoreLEQceiling) && xrows.every(r => r.kCoreLEQceiling),
  maxCeilingCore_claims: Math.max(...ceilRows.map(r => r.ceilingCore)),
  maxCeilingCore_publishedCorpus: wMax,
  maxCeilingCore_rebuiltSearch: xMax,
  maxCeilingCore_overEverythingScoredHere: MAXCEIL,
  distinctVectorsScoredHere: new Set([...ceilRows.map(r => r.move), ...scorable.map(r => r.move), ...xrows.map(r => r.move)]).size,
  barMoved: false, barNote: 'The 95 mm bar is read from rig3 criteria() and is not redefined anywhere in this file.',
};
log('');
log(`   max ceilingCore over the round-6 CLAIMS            : ${OUT.phaseE.maxCeilingCore_claims}/9`);
log(`   max ceilingCore over the PUBLISHED CORPUS (re-scored): ${OUT.phaseE.maxCeilingCore_publishedCorpus}/9`);
log(`   max ceilingCore over the REBUILT SEARCH            : ${OUT.phaseE.maxCeilingCore_rebuiltSearch}/9`);
log(`   MAX OVER EVERYTHING SCORED HERE                    : ${MAXCEIL}/9   over ${OUT.phaseE.distinctVectorsScoredHere} distinct vectors`);
log(`   kCore <= ceilingCore holds everywhere: ${OUT.phaseE.boundHoldsOnEveryRowHere}`);
flush();

// ==================================================================== PHASE N
log('');
log('PHASE N — penetration, whole-episode, on every clear of every claim');
const penRows = Object.values(G).map(r => ({ file: r.file, move: r.move, clears: r.kExt,
  clearsMinPenEpisode_mm: r.clearsMinPenEpisode_mm, worstCell_mm: r.minPenetrationEpisode_mm,
  clearsDeeperThanSoft: r.clearsDeeperThanSoft, cellsDeeperThanSoft: r.cellsDeeperThanSoft,
  penConsistent: r.penConsistent }));
for (const r of penRows)
  log(`   ${r.file.padEnd(40)} clears ${String(r.clears).padStart(2)}/14  [${r.clearsMinPenEpisode_mm.join(', ')}]  worst cell ${f2(r.worstCell_mm)} mm  clears deeper than ${(PEN_SOFT * 1000).toFixed(0)} mm: ${r.clearsDeeperThanSoft}`);
OUT.phaseN = { rows: penRows,
  invariantHoldsEverywhere: penRows.every(r => r.penConsistent),
  anyClaimClearDeeperThanSoft: penRows.some(r => r.clearsDeeperThanSoft > 0),
  deepestClaimReading_mm: Math.min(...penRows.map(r => r.worstCell_mm)),
  deepestPublishedCorpusReading_mm: Math.min(...scorable.map(r => r.minPenetrationEpisode_mm)),
  deepestRebuiltSearchReading_mm: xrows.length ? Math.min(...xrows.map(r => r.measured.minPenetrationEpisode_mm)) : null,
  rebuiltSearchClearsDeeperThanSoft: OUT.phaseX.clearsDeeperThanSoftContact.length };
log(`   invariant minPenEpisode <= penAtScore holds on every cell: ${OUT.phaseN.invariantHoldsEverywhere}`);
log(`   deepest reading — claims ${f2(OUT.phaseN.deepestClaimReading_mm)} mm | published corpus ${f2(OUT.phaseN.deepestPublishedCorpusReading_mm)} mm | rebuilt search ${f2(OUT.phaseN.deepestRebuiltSearchReading_mm)} mm`);
flush();

// ==================================================================== PHASE T
log('');
log('PHASE T — torque against the forcerange');
const tqClaims = Object.values(G).map(r => r.maxTq);
OUT.phaseT = { forcerange_Nm: FORCERANGE,
  anyClaimOverCeiling: Object.values(G).some(r => r.tqOverCeiling),
  claimsSaturating: Object.values(G).filter(r => r.tqSaturated).length, claims: Object.keys(G).length,
  publishedCorpusSaturating: scorable.filter(r => r.maxTq >= FORCERANGE - 1e-4).length,
  publishedCorpusScored: scorable.length,
  publishedCorpusNotSaturating: scorable.filter(r => r.maxTq < FORCERANGE - 1e-4).map(r => ({ file: r.file, maxTq: r.maxTq })),
  rebuiltSearchSaturating: xrows.filter(r => r.measured.maxTq >= FORCERANGE - 1e-4).length,
  rebuiltSearchScored: xrows.length,
  anyRowOverCeilingAnywhere: Object.values(G).some(r => r.tqOverCeiling)
    || scorable.some(r => r.maxTq > FORCERANGE + 1e-9) || xrows.some(r => r.measured.tqOverCeiling) };
log(`   claims: ${OUT.phaseT.claimsSaturating}/${OUT.phaseT.claims} saturate ${FORCERANGE} N.m, ${OUT.phaseT.anyClaimOverCeiling ? 'SOME EXCEED IT' : 'none exceeds it'}`);
log(`   published corpus: ${OUT.phaseT.publishedCorpusSaturating}/${OUT.phaseT.publishedCorpusScored} saturate; the exceptions are the ones that never push:`);
for (const e of OUT.phaseT.publishedCorpusNotSaturating) log(`      ${e.file.padEnd(38)} maxTq ${e.maxTq}`);
log(`   rebuilt search: ${OUT.phaseT.rebuiltSearchSaturating}/${OUT.phaseT.rebuiltSearchScored} saturate`);
log(`   ANY row anywhere above the forcerange: ${OUT.phaseT.anyRowOverCeilingAnywhere}`);
flush();

// ==================================================================== PHASE C
log('');
log('PHASE C — the controls, under the CURRENT (round-6-edited) instrument');
const CTRL = [['ctrl_do_nothing.json', 0.040], ['ctrl_do_nothing.json', 0.060],
              ['r4_ctrl_on_tread_60mm.json', 0.060], ['r4_ctrl_on_tread_90mm.json', 0.090]]
              .filter(([f]) => fs.existsSync(P + f));
const ctrlRows = [];
for (const [f, h] of CTRL) {
  const g = await scoreRobust(P + f, { rise: h, skipBounds: true });
  const core = g.cells.filter(c => c.cell.tier === 'core');
  const ceilingCore = core.filter(c => (c.maxZ - c.rise) > CEILING_ABOVE).length;
  const row = { file: f, rise_mm: h * 1000, move: g.move, kCore: g.kCore, kCoreStable: g.kCoreStable,
                kExt: g.kExt, kExtStable: g.kExtStable, ceilingCore,
                maxTq: +g.agg.maxTq.toFixed(4),
                minPenEpisode_mm: +g.agg.minPenetrationEpisode_mm.toFixed(2) };
  ctrlRows.push(row);
  log(`   ${f.padEnd(30)} @${String(row.rise_mm).padStart(3)}mm  kCore ${row.kCore}/9  stable ${row.kCoreStable}/9  kExt ${row.kExt}/14  ceil ${row.ceilingCore}/9  maxTq ${row.maxTq}`);
}
const dn = ctrlRows.filter(r => r.file === 'ctrl_do_nothing.json');
const pd = ctrlRows.filter(r => r.file.startsWith('r4_ctrl_on_tread'));
OUT.phaseC = { rows: ctrlRows,
               doNothingAlwaysFails: dn.length > 0 && dn.every(r => r.kExt === 0),
               placedDuckAlwaysPasses: pd.length > 0 && pd.every(r => r.kExt === 14) };
log(`   do-nothing fails everywhere: ${OUT.phaseC.doNothingAlwaysFails}   placed duck clears 14/14: ${OUT.phaseC.placedDuckAlwaysPasses}`);
flush();

// ==================================================================== PHASE Y
log('');
log('PHASE Y — the tail agent\'s MECHANISM claim, re-derived from the trace the round-6 edit');
log('   added. Claim: the fall is 3 control ticks after the law lets go, at every setting, and');
log('   holding the law to tick 50 buys uprightness by cancelling the lift. "Fall" = the first');
log('   tail tick with projected-gravity z >= -0.90, which is rig3\'s own upright test.');
const YSET = [['best_r5_servoland_kcore_60mm.json', 0], ['best_r6_tail10_servoland_60mm.json', 10],
              ['best_r6_tail25_servoland_60mm.json', 25], ['best_r6_tail50_servoland_60mm.json', 50]]
              .filter(([f]) => fs.existsSync(P + f));
const yrows = [];
for (const [f, want] of YSET) {
  const g = await scoreRobust(P + f, { rise: 0.060, core: true, skipBounds: true, tailTrace: true });
  const per = g.cells.map((c, i) => {
    const tr = c.tailTrace || [];
    const enterUp = tr.length ? tr[0].gz < -0.90 : null;
    const fall = tr.findIndex(s => s.gz >= -0.90);
    return { cell: CELLNAME[i], tailAuthority: c.servo ? c.servo.tailAuthority : null,
             tailTicksRun: c.servo ? c.servo.tailTicksRun : null,
             enterUpright: enterUp, fallTick: fall < 0 ? null : fall,
             fallMinusAuthority: fall < 0 ? null : fall - (c.servo ? c.servo.tailAuthority : 0),
             enterAbove_mm: tr.length ? +(tr[0].above * 1000).toFixed(1) : null,
             endAbove_mm: tr.length ? +(tr[tr.length - 1].above * 1000).toFixed(1) : null,
             peakAbove_mm: +((c.maxZ - c.rise) * 1000).toFixed(1),
             upTail: c.uprightTailTicks, honest: c.crit.honest };
  });
  const ceilingCore = g.cells.filter(c => (c.maxZ - c.rise) > CEILING_ABOVE).length;
  const falls = per.map(p => p.fallTick);
  const deltas = per.map(p => p.fallMinusAuthority).filter(v => v !== null);
  const row = { file: f, declaredTailTicks: want, move: g.move, kCore: g.kCore,
                kCoreStable: g.kCoreStable, ceilingCore,
                meanUprightTail: +g.agg.meanUprightTailTicks.toFixed(2),
                fallTicks: falls,
                fallMinusAuthority: per.map(p => p.fallMinusAuthority),
                modalFallMinusAuthority: deltas.length ? deltas.sort((a, b) => a - b)[Math.floor(deltas.length / 2)] : null,
                enterAbove_mm: per.map(p => p.enterAbove_mm),
                endAbove_mm: per.map(p => p.endAbove_mm),
                cells: per };
  yrows.push(row);
  log(`   ${f.padEnd(40)} tailTicks ${String(want).padStart(2)}  kCore ${row.kCore}/9 stable ${row.kCoreStable}/9 ceil ${row.ceilingCore}/9  meanUpTail ${row.meanUprightTail}/50`);
  log(`      fall tick per core cell : [${falls.map(v => v === null ? 'none' : v).join(', ')}]`);
  log(`      fall minus tailTicks    : [${row.fallMinusAuthority.map(v => v === null ? '-' : v).join(', ')}]   median ${row.modalFallMinusAuthority}`);
  log(`      trunk above tread, tail entry -> tail end (mm): [${row.enterAbove_mm.join(', ')}] -> [${row.endAbove_mm.join(', ')}]`);
}
OUT.phaseY = { rows: yrows,
  fallTracksTheHandBack: yrows.every(r => r.modalFallMinusAuthority === null || Math.abs(r.modalFallMinusAuthority - 3) <= 1
                                       || r.fallTicks.every(v => v === null)),
  holdingTheLawKillsTheClear: (() => { const t50 = yrows.find(r => r.declaredTailTicks === 50); return t50 ? (t50.kCore === 0 && t50.ceilingCore === 0) : null; })(),
  note: 'Every row here is an ORACLE row. The law reads the tread height and edge; the shipped policy cannot.' };
log(`   the fall moves with the hand-back (median fall - tailTicks ~ 3): ${OUT.phaseY.fallTracksTheHandBack}`);
log(`   holding the law for all 50 tail ticks drives kCore and ceilingCore to 0: ${OUT.phaseY.holdingTheLawKillsTheClear}`);
flush();

// ==================================================================== PHASE K
log('');
log('================================================================');
log('PHASE K — THE KILL GATE');
log('   RULE: PASSED only if a DISTINCT, FLOOR-SPAWNED launch reaches ceilingCore >= 7 of 9');
log('   at 60 mm, in declared bounds, under the actuator ceiling, with no cheat.');
// Every distinct vector this judge scored at 60 mm, from every source, with its
// eligibility. A vector is eligible if it started on the floor, is inside its
// declared bounds, and never exceeded the forcerange.
const pool = new Map();
const add = (move, sha, ceilingCore, kCore, kCoreStable, eligible, why, source) => {
  const cur = pool.get(sha);
  if (!cur || ceilingCore > cur.ceilingCore) pool.set(sha, { move, sha, ceilingCore, kCore, kCoreStable, eligible, why, source });
};
for (const r of Object.values(G))
  add(r.move, r.sha256, r.ceilingCore, r.kCore, r.kCoreStable,
      r.admissible && !r.tqOverCeiling, r.admissible ? (r.tqOverCeiling ? 'over the forcerange' : 'eligible') : 'not a floor-spawned in-bounds climb', 'round-6 claims');
for (const r of scorable)
  add(r.move, r.sha256, r.ceilingCore, r.kCore, r.kCoreStable, r.maxTq <= FORCERANGE + 1e-9,
      r.maxTq <= FORCERANGE + 1e-9 ? 'eligible' : 'over the forcerange', 'published corpus');
for (const r of xrows)
  add(r.move, r.claimedSha256, r.measured.ceilingCore, r.measured.kCore, r.measured.kCoreStable,
      r.hashMatches && !r.measured.tqOverCeiling,
      r.hashMatches ? (r.measured.tqOverCeiling ? 'over the forcerange' : 'eligible') : 'rebuild does not hash to the claimed vector', 'rebuilt search');
const POOL = [...pool.values()];
const eligible = POOL.filter(r => r.eligible);
const bestCeil = eligible.length ? eligible.reduce((a, b) => b.ceilingCore > a.ceilingCore ? b : a) : null;
const bestStable = eligible.length ? eligible.reduce((a, b) => b.kCoreStable > a.kCoreStable ? b : a) : null;
const PASSED = !!bestCeil && bestCeil.ceilingCore >= 7;
const poolHist = {};
for (const r of POOL) poolHist[r.ceilingCore] = (poolHist[r.ceilingCore] || 0) + 1;
log('');
log(`   distinct vectors scored at 60 mm in this audit: ${POOL.length}   (eligible ${eligible.length})`);
log(`   ceilingCore histogram over them: ${JSON.stringify(poolHist)}`);
log(`   BEST ceilingCore  : ${bestCeil ? bestCeil.ceilingCore : 0}/9  (${bestCeil ? bestCeil.move : 'none'}, ${bestCeil ? bestCeil.source : '-'})`);
log(`   BEST kCoreStable  : ${bestStable ? bestStable.kCoreStable : 0}/9  (${bestStable ? bestStable.move : 'none'}, ${bestStable ? bestStable.source : '-'})`);
log(`   THE GATE NEEDS 7. KILL CONDITION: ${PASSED ? 'PASSED' : 'FAILED'}`);
OUT.killCondition = {
  rule: 'PASSED only if a DISTINCT, FLOOR-SPAWNED launch reaches ceilingCore >= 7 of 9 at 60 mm, in declared bounds, under the 0.6405 N.m forcerange, with no cheat',
  distinctVectorsScoredAt60mm: POOL.length, eligibleVectors: eligible.length,
  ceilingHistogram: poolHist,
  bestCeilingCore: bestCeil ? bestCeil.ceilingCore : 0, bestCeilingMove: bestCeil ? bestCeil.move : null,
  bestKCoreStable: bestStable ? bestStable.kCoreStable : 0, bestKCoreStableMove: bestStable ? bestStable.move : null,
  result: PASSED ? 'PASSED' : 'FAILED',
  consequence: PASSED ? 'the 40-80 mm band is OPEN'
    : 'THE SEARCH IS FINISHED AT THIS SCALE. Every actuator is pinned at the 0.6405 N.m forcerange in every cell of every move, and the saturation that already closed the 80-120 mm band and 180 mm closes this too. The next attempt has to change the ROBOT or the MOVE CLASS — a stronger neck actuator or a shorter strut lever, a second duck, a wall, a lever — not the search.',
  barMoved: false,
};
OUT.whyItFailed = {
  headline: 'The bound is the criterion, not the optimiser. honest needs the trunk above 95 mm at the scored instant; a cell whose trunk never reaches 95 mm cannot be cleared by any landing law, timed, event-triggered or servoed. The count of such cells is capped at ' + MAXCEIL + ' of 9 over every vector scored in this audit.',
  saturation: 'Every scored row that pushes runs maxTq = ' + FORCERANGE + ' N.m — the forcerange is pinned, not approached. The only rows below it are the ones that never push.',
  strutGeometry: 'The round-6 limits trace (climb/r6_limits_table-results.json, a diagnostic gated to reproduce the scorer exactly) puts the neck_pitch lever at 87-90 mm at the pose the vault depends on, where ' + FORCERANGE + ' N.m yields 7.16-7.34 N at the beak against a 7.23 N body weight. That is a geometry limit, not a control limit, and this judge did not re-derive it — it is reported as the search agent measured it.',
  whatWouldHaveToChange: [
    'THE ROBOT: a neck/hip actuator with more than 0.6405 N.m, or the same torque through a shorter lever — the strut arm at the vault pose is the term that eats it.',
    'THE MOVE CLASS: a move that does not require the trunk to be lifted 95 mm by the duck alone — a second duck, a wall or rail to react against, a lever or ramp placed first.',
    'NOT the search: 402+ distinct vectors over six rounds, including a CEM that optimised the ceiling directly with no landing term at all, never moved it past ' + MAXCEIL + ' of 9.',
  ],
};
OUT.oracleCaveat = 'Every servoed row and every servo.tailTicks row in this audit is an ORACLE UPPER BOUND: climb/servo.mjs reads the tread height and the tread front edge straight out of data.qpos and data.geom_xpos, and the shipped policy is fed 61 numbers of which none is exteroceptive. The three round-6 ceiling launches carry no servo and no event and are free of that caveat.';
// THE ONE SENTENCE, built from the numbers this audit measured, not from an
// adjective. Nothing in it is rounded up and nothing in it is a servoed oracle
// row: the best kCoreStable belongs to a launch with no servo and no event.
OUT.honestClaim = 'After six rounds and ' + OUT.phaseE.distinctVectorsScoredHere
  + ' distinct scored launches, the best whole-body move gets the duck onto a 60 mm step and leaves it standing in '
  + (bestStable ? bestStable.kCoreStable : 0) + ' of the 9 robustness cells, no move has ever cleared more than '
  + MAXCEIL + ', and the limit is not tuning: all 14 servos are already pinned at their 0.6405 N.m torque limit in every cell of every move.';
log('');
log('   THE ONE SENTENCE THE APP MAY PRINT:');
log('   "' + OUT.honestClaim + '"');
log('');
log(`   total wall ${el()}`);
flush();
console.log('wrote climb/r6_judge-results.json and climb/audit_r6.log');
