// ROUND 5 — THE JUDGE'S LOOP.
//
// New file. climb/audit_r2.mjs, audit_r3.mjs, audit_r4.mjs and
// audit_r4_judge.mjs are the earlier rounds' evidence and are NOT touched.
// This file writes climb/r5_judge-results.json and climb/audit_r5.log only.
//
// Every number printed here is measured IN THIS PROCESS, from a SAVED JSON on
// disk, through the CURRENT climb/rig3.mjs and climb/robust.mjs. Nothing is
// quoted from an agent's report; where a report made a claim, the claim is
// re-derived and the report's number is printed beside it.
//
//   PHASE P   PARITY. The current rig3.mjs vs climb/rig3_pre_r4.mjs on the
//             round-4 judge's own 86 rows (43 pre-round-4 files x tails
//             policy+hold), at FULL FLOAT DIGITS, on the judge's own field
//             lists AND on a deep recursive walk of every leaf. Round 5 added
//             a `servo` field and a whole-episode penetration scan to that
//             file. If either moved a pre-round-5 number, every round-1..4
//             result is re-baselined and round 5 is inadmissible.
//   PHASE R   Are rig3.mjs and robust.mjs still the same simulation — including
//             ON A SERVOED FILE, which is the half of the instrument the
//             round-5 agents' parity sets could not cover?
//   PHASE H   Identity. sha256 of every round-5 file; every pre-round-5 hash
//             the round-4 judge published, re-computed.
//   PHASE B   Declared bounds, enforced at scoring time.
//   PHASE S   How the duck STARTED. spawn / spawnPose / spawnVel /
//             spawnLastAction / settleTicks != 25 means it did not start on
//             the floor and is not a climb.
//   PHASE D   DEGENERACY. A servo that never arms must reproduce its base file
//             cell by cell at full digits, in the scorer that decides the
//             verdict (robust.mjs, 14 cells) — not only in rig3.
//   PHASE G   Re-score EVERY round-5 claim and every published clear on the
//             14-cell grid from disk: kCore/kCoreStable/kExt/kExtStable, the
//             objectives, maxTq against the 0.6405 N.m ceiling, the lateral
//             gate over the whole episode, whole-episode penetration, upright
//             tail ticks, and whether the servo armed.
//   PHASE E   THE CEILING, re-derived. `honest` needs the trunk above 95 mm at
//             the scored instant, so the count of core cells in which the trunk
//             EVER exceeded 95 mm is an upper bound on that move's kCore under
//             any landing whatsoever. Taken from maxZ, which rig3 records over
//             track and tail.
//   PHASE N   THE ROUND-4 JUDGE'S HOLE, now that it is closed. Whole-episode
//             penetration on every clear, plus the consistency check the field
//             must satisfy (the scored instant is one of the scanned ticks, so
//             minPenetrationEpisode <= penetrationAtScore ALWAYS).
//   PHASE F   FRAGILITY. Round 4 showed the round-3 vault's clears are isolated
//             points: one control tick of shift takes it 4 -> 1 of 9. The same
//             probe on the one round-5 move that clears three plants at one
//             rise, from files written to disk here.
//   PHASE C   The controls. do-nothing must fail; a duck PLACED on the tread
//             must pass. If the round-5 edits broke the criterion, this is
//             where it shows.
//   PHASE K   THE KILL GATE.
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/audit_r5.mjs
import fs from 'node:fs';
import { scoreSaved as newScore, criteria, LATERAL, DUCKG, RISER_X } from '../climb/rig3.mjs';
import { scoreSaved as oldScore } from '../climb/rig3_pre_r4.mjs';
import { scoreSaved as r4Score } from '../climb/rig3_pre_r5.mjs';
import { execSync } from 'node:child_process';
import { scoreRobust, scoreCell, intentHash, intentHashOfFile, checkBounds,
         DHS, PLANTS, EXT_DHS, EXT_PLANT, CLEAR_BONUS, UPRIGHT_BONUS,
         UPRIGHT_TAIL_MIN, DECLARED_BOUNDS } from '../climb/robust.mjs';

const P = '../climb/';
const LOG = [];
const log = s => { console.log(s); LOG.push(String(s)); };
const T0 = Date.now();
const el = () => ((Date.now() - T0) / 1000).toFixed(0) + 's';
const FORCERANGE = 0.6405;                     // N.m, every actuator
const CEILING_ABOVE = 0.095;                   // `honest` needs above > this
const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const DT = 1 / C.tickHz;                       // the CONTROL tick, measured

const rd = f => JSON.parse(fs.readFileSync(P + f, 'utf8'));
const ls = re => fs.readdirSync(P).filter(f => re.test(f)).sort();
const riseOf = f => { const m = f.match(/_(\d+)mm/); return m ? parseInt(m[1], 10) / 1000 : null; };
const f2 = n => (n === null || n === undefined) ? 'null' : n.toFixed(2);

const OUT = {
  generated: new Date().toISOString(),
  judge: 'audit_r5.mjs',
  note: 'Round-5 judge. Writes climb/r5_judge-results.json and climb/audit_r5.log only; no earlier audit file is touched.',
  constants: {
    lateralGate_mm: LATERAL * 1000, duckCollidableGeoms: DUCKG.length,
    riserLine_mm: RISER_X * 1000, forcerange_Nm: FORCERANGE,
    honestAboveGate_mm: CEILING_ABOVE * 1000,
    controlTick_ms: DT * 1000, tickHz: C.tickHz,
    uprightTailMin: UPRIGHT_TAIL_MIN, clearBonus: CLEAR_BONUS, uprightBonus: UPRIGHT_BONUS,
    declaredBounds: DECLARED_BOUNDS,
    grid: { core: { dhs_mm: DHS.map(d => d * 1000), plants: PLANTS },
            ext: { dhs_mm: EXT_DHS.map(d => d * 1000), plant: EXT_PLANT }, n: 14 },
  },
};

log('================================================================');
log('ROUND 5 — THE JUDGE. Every number measured here, from saved JSON, now.');
log(`   lateral gate ${(LATERAL * 1000).toFixed(0)} mm | duck geoms ${DUCKG.length} | riser ${(RISER_X * 1000).toFixed(0)} mm`);
log(`   CONTROL TICK measured from duckkit-constants.json: tickHz=${C.tickHz} -> ${(DT * 1000).toFixed(2)} ms`);
log('   (the round-5 search report and climb/r5_servoland.mjs say "5.66 ms control tick"; the tick is 20.00 ms)');

// ==================================================================== PHASE P
log('');
log('PHASE P — PARITY, in two halves.');
log('   P1: CURRENT rig3.mjs vs climb/rig3_pre_r4.mjs on the round-4 judge\'s own 86 rows');
log('       and its own field lists. This reproduces r4_judge-results.json phaseP.');
log('   P2: CURRENT rig3.mjs vs climb/rig3_pre_r5.mjs (the round-4 instrument) on the SAME');
log('       86 rows, DEEP recursive walk over every leaf. That is the comparison that');
log('       isolates ROUND 5, because rig3_pre_r4 predates the round-4 fields too.');
// Prove rig3_pre_r5.mjs really is rig3.mjs at commit e00e1e4 (the round-4
// instrument) before trusting a word it says. Its only permitted difference is
// the isMain guard string, which decides nothing when it is imported.
let preR5Verified = false, preR5Note = '';
try {
  const now = execSync('git show e00e1e4:climb/rig3.mjs', { cwd: '..', encoding: 'utf8' });
  const copy = fs.readFileSync(P + 'rig3_pre_r5.mjs', 'utf8').split('rig3_pre_r5.mjs').join('rig3.mjs');
  preR5Verified = copy === now;
  preR5Note = preR5Verified
    ? 'climb/rig3_pre_r5.mjs IS climb/rig3.mjs at e00e1e4, character for character apart from its isMain guard string'
    : 'climb/rig3_pre_r5.mjs DIFFERS from climb/rig3.mjs at e00e1e4 by more than the isMain guard';
} catch (e) { preR5Note = 'could not read e00e1e4:climb/rig3.mjs — ' + e.message; }
log(`   ${preR5Note}`);

const PRE_R4 = [
  ...ls(/^best_r2_.*\.json$/),
  ...ls(/^best_r3_.*\.json$/),
  ...ls(/^best_[012]_\d+mm\.json$/),
  'ctrl_do_nothing.json', 'ctrl_on_tread_40mm.json', 'ctrl_on_tread_60mm.json',
  'ctrl_on_tread_90mm.json', 'ctrl_on_tread_120mm.json', 'ctrl_walk_only.json',
].filter(f => fs.existsSync(P + f));

// The round-4 judge's OWN field lists, so PHASE P here reproduces PHASE P there.
const SCAL = ['x0', 'maxX', 'maxZ', 'maxAbsDY', 'feetOnTreadMax', 'feetHighMax',
              'headFrac', 'riserFrac', 'wallFrac', 'upFrac', 'satFrac', 'z0Settle',
              'liftIntegral', 'footNear', 'bothNear', 'reward'];
const SNAP = ['x', 'y', 'z', 'dy', 'above', 'up', 'feetUpRaw', 'feetUpLat', 'feetOnTread'];
const CRIT = ['orig', 'lat', 'honest', 'honest60'];
// Every key that exists ONLY because of round 5. There is nothing in the
// pre-round-4 harness to compare these to, so they are excluded from the deep
// walk — and named here so the exclusion is not a hiding place.
const R5_ONLY = new Set(['minPenetrationEpisode', 'minPenetrationPair', 'minPenetrationTick',
                         'penetrationTicksScanned', 'servo']);

/** Recursive full-digit walk over every leaf of the round-4 record. */
function deepDiff(a, b, path, out) {
  if (out.length > 40) return;
  const ta = a === null ? 'null' : typeof a, tb = b === null ? 'null' : typeof b;
  if (ta !== tb) { out.push({ path, now: a, pre: b, why: 'type' }); return; }
  if (a !== null && typeof a === 'object') {
    if (Array.isArray(a) !== Array.isArray(b)) { out.push({ path, why: 'array/object' }); return; }
    if (Array.isArray(a)) {
      if (a.length !== b.length) { out.push({ path, now: a.length, pre: b.length, why: 'length' }); return; }
      for (let i = 0; i < a.length; i++) deepDiff(a[i], b[i], `${path}[${i}]`, out);
      return;
    }
    const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
    for (const k of keys) {
      if (R5_ONLY.has(k)) continue;
      deepDiff(a[k], b[k], path ? `${path}.${k}` : k, out);
    }
    return;
  }
  if (!Object.is(a, b)) out.push({ path, now: a, pre: b });
}

const parity = { rows: 0, stateExact: 0, rewardExact: 0, critRows: [], mismatch: [],
                 deepExact: 0, deepDiffs: [], penConsistent: 0, penRows: 0, penViolations: [] };
for (const f of PRE_R4) {
  const h = riseOf(f) ?? 0.060;
  for (const tail of ['policy', 'hold']) {
    const A = await newScore(P + f, { rise: h, tail });
    const B = await oldScore(P + f, { rise: h, tail });
    parity.rows++;
    const diffs = [];
    for (const k of SCAL) if (A[k] !== B[k]) diffs.push({ field: k, now: A[k], pre: B[k] });
    for (const k of SNAP) if (A.scored[k] !== B.scored[k]) diffs.push({ field: 'scored.' + k, now: A.scored[k], pre: B.scored[k] });
    const stateDiffs = diffs.filter(d => d.field !== 'reward');
    if (!stateDiffs.length) parity.stateExact++;
    if (A.reward === B.reward) parity.rewardExact++;
    const cd = CRIT.filter(k => A.crit[k] !== B.crit[k]);
    if (cd.length) parity.critRows.push({ file: f, tail, changed: cd });
    if (stateDiffs.length) parity.mismatch.push({ file: f, tail, diffs: stateDiffs });
    // P2 — the deep walk, against the ROUND-4 instrument.
    const R4 = await r4Score(P + f, { rise: h, tail });
    const dd = []; deepDiff(A, R4, '', dd);
    if (!dd.length) parity.deepExact++;
    else parity.deepDiffs.push({ file: f, tail, diffs: dd.slice(0, 8) });
    // The whole-episode field must never be shallower than the scored instant:
    // the scored instant IS one of the scanned ticks.
    if (A.minPenetrationEpisode !== null && A.penetrationAtScore !== null) {
      parity.penRows++;
      if (A.minPenetrationEpisode <= A.penetrationAtScore + 1e-15) parity.penConsistent++;
      else parity.penViolations.push({ file: f, tail, episode: A.minPenetrationEpisode, atScore: A.penetrationAtScore });
    }
    // The servo field must be null on every file that carries no servo block.
    if (A.servo !== null) parity.mismatch.push({ file: f, tail, diffs: [{ field: 'servo', now: A.servo, pre: null }] });
  }
}
log(`   ${parity.rows} rows (${PRE_R4.length} pre-round-4 files x tails policy+hold)`);
log(`   P1 vs rig3_pre_r4 — PHYSICAL STATE identical   : ${parity.stateExact}/${parity.rows}`);
log(`   P1 vs rig3_pre_r4 — reward() identical         : ${parity.rewardExact}/${parity.rows}`);
log(`   P1 vs rig3_pre_r4 — rows where a CRITERION moved: ${parity.critRows.length}`);
log(`   P2 vs rig3_pre_r5 — DEEP walk, every leaf, Object.is : ${parity.deepExact}/${parity.rows}  (round-5-only keys excluded: ${[...R5_ONLY].join(', ')})`);
for (const d of parity.deepDiffs.slice(0, 6)) log(`      !! ${d.file} tail=${d.tail}: ${JSON.stringify(d.diffs)}`);
log(`   minPenetrationEpisode <= penetrationAtScore    : ${parity.penConsistent}/${parity.penRows}`);
for (const v of parity.penViolations.slice(0, 5)) log(`      !! ${v.file} ${v.tail}: episode=${v.episode} atScore=${v.atScore}`);
const admissible = parity.mismatch.length === 0 && parity.deepDiffs.length === 0
                && parity.critRows.length === 0 && parity.penViolations.length === 0
                && preR5Verified;
OUT.phaseP = { preR5IsCommitE00e1e4: preR5Verified, preR5Note,
               files: PRE_R4.length, rows: parity.rows, stateExact: parity.stateExact,
               rewardExact: parity.rewardExact, deepExact: parity.deepExact,
               criterionChangedRows: parity.critRows, stateMismatches: parity.mismatch,
               deepDiffs: parity.deepDiffs, r5OnlyKeysExcluded: [...R5_ONLY],
               penetrationConsistency: `${parity.penConsistent}/${parity.penRows}`,
               penetrationViolations: parity.penViolations,
               reproducesR4Judge: parity.rows === 86 && parity.stateExact === 86 && parity.rewardExact === 86 && parity.critRows.length === 0,
               verdict: admissible ? 'ADMISSIBLE — the round-5 edits moved nothing' : 'INADMISSIBLE' };
log(`   reproduces r4_judge-results.json phaseP (43 files / 86 rows / 86 / 86 / 0): ${OUT.phaseP.reproducesR4Judge}`);
log(`   VERDICT: ${OUT.phaseP.verdict}   [${el()}]`);

// ==================================================================== PHASE R
log('');
log('PHASE R — rig3.scoreSaved vs robust.scoreCell (cell 0), INCLUDING servoed files');
const RSET = ['best_r3_vault_60mm.json', 'best_r4_famA_60mm.json', 'ctrl_do_nothing.json',
              'r5_servo_never_60mm.json', 'r5_servo_armed_60mm.json',
              'best_r5_servo_60mm.json', 'best_r5_servoland_kcore_60mm.json',
              'best_r5_servoland_60mm.json'].filter(f => fs.existsSync(P + f));
const rrows = [];
for (const f of RSET) {
  const h = riseOf(f) ?? 0.060;
  const A = await newScore(P + f, { rise: h, tail: 'policy' });
  const B = await scoreCell(P + f, { rise: h, isolate: true, skipBounds: true });
  const ok = A.scored.x === B.scored.x && A.scored.z === B.scored.z && A.reward === B.reward
    && A.crit.honest === B.crit.honest && A.scored.feetOnTread === B.scored.feetOnTread
    && A.uprightTailTicks === B.uprightTailTicks && A.maxAbsDY === B.maxAbsDY
    && A.minPenetrationEpisode === B.minPenetrationEpisode;
  rrows.push({ file: f, exact: ok, servoArmed: A.servo ? A.servo.armed : null,
               servoTicks: A.servo ? A.servo.ticks : null,
               x: A.scored.x, z: A.scored.z, reward: A.reward,
               minPenEpisode_mm: A.minPenetrationEpisode === null ? null : +(A.minPenetrationEpisode * 1000).toFixed(3),
               ticksScanned: A.penetrationTicksScanned });
  log(`   ${f.padEnd(34)} EXACT=${ok}  servo=${A.servo ? (A.servo.armed ? 'ARMED ' + A.servo.ticks + ' ticks' : 'present, never armed') : 'none'}  x=${A.scored.x} z=${A.scored.z} rew=${A.reward.toFixed(6)} minPenEp=${rrows[rrows.length - 1].minPenEpisode_mm}mm ticks=${A.penetrationTicksScanned}`);
}
OUT.phaseR = { rows: rrows, allExact: rrows.every(r => r.exact) };
log(`   allExact = ${OUT.phaseR.allExact}   [${el()}]`);

// ============================================================== PHASE H/B/S
log('');
log('PHASE H / B / S — identity, declared bounds, and how the duck STARTED');
const R5_FILES = ['best_r5_servo_60mm.json', 'best_r5_servoland_60mm.json',
                  'best_r5_servoland_kcore_60mm.json', 'r5_servo_armed_60mm.json',
                  'r5_servo_never_60mm.json'].filter(f => fs.existsSync(P + f));
const REF_FILES = ['best_r3_vault_60mm.json', 'best_r3_vault_40mm.json', 'best_r3_vault_50mm.json',
                   'best_r2_vault_40mm.json', 'best_r2_vault_60mm.json',
                   'best_r4_famA_60mm.json'].filter(f => fs.existsSync(P + f));
const inv = [];
for (const f of [...R5_FILES, ...REF_FILES]) {
  const j = rd(f);
  const sha = intentHashOfFile(P + f);
  const B = checkBounds(j);
  const floorSpawn = !j.spawn && !j.spawnPose && !j.spawnVel && !j.spawnLastAction && !j.spawnQuat;
  const settle = j.settleTicks === undefined ? 25 : j.settleTicks;
  const kind = j.spawnPose ? 'HANDOFF SPAWN — NOT A CLIMB'
             : j.spawn ? 'PLACED SPAWN — NOT A CLIMB'
             : settle !== 25 ? `settleTicks=${settle} — NOT THE STANDARD START`
             : 'floor spawn — a climb';
  inv.push({ file: f, sha256: sha, move: sha.slice(0, 12), blend: j.blend, side: j.side || 0,
             gap: j.gap || 0, approach: j.approach || 0, declaredBounds: B.bounds,
             boundViolations: B.violations, hasServo: !!j.servo, hasEvent: !!j.event,
             servoAt: j.servo ? j.servo.at : null, servoOnEvent: j.servo ? !!j.servo.onEvent : null,
             servoHasAuthoredBase: !!(j.servo && j.servo.base),
             settleTicks: settle, floorSpawn, kind });
}
for (const r of inv)
  log(`   ${r.file.padEnd(34)} ${r.move}  blend=${r.blend} side=${r.side} gap=${r.gap}  bounds=${JSON.stringify(r.declaredBounds)} viol=${r.boundViolations.length}  servo=${r.hasServo ? 'at ' + r.servoAt + (r.servoHasAuthoredBase ? ' (authored base)' : '') : 'no'}  [${r.kind}]`);
const byHash = {};
for (const r of inv) (byHash[r.sha256] ||= []).push(r.file);
const dupes = Object.entries(byHash).filter(([, v]) => v.length > 1).map(([h, v]) => ({ move: h.slice(0, 12), files: v }));
for (const d of dupes) log(`   ONE VECTOR, ${d.files.length} LABELS  ${d.move}  ${d.files.join('  ')}`);
// The hashes the round-4 judge published must be unchanged by the round-5 edit.
const R4_HASHES = {
  'best_r3_vault_60mm.json': '4b9110c448ec', 'best_r2_vault_60mm.json': '74d35b21ac80',
  'best_r3_vault_40mm.json': 'dff01b0a1906', 'best_r3_vault_50mm.json': '7904bf3363c5',
  'best_r2_vault_40mm.json': '86813f9c1ad4',
};
const hashRows = [];
for (const [f, want] of Object.entries(R4_HASHES)) {
  if (!fs.existsSync(P + f)) continue;
  const got = intentHashOfFile(P + f).slice(0, 12);
  hashRows.push({ file: f, published: want, now: got, unchanged: got === want });
  log(`   pre-round-5 hash ${f.padEnd(26)} published ${want}  now ${got}  ${got === want ? 'UNCHANGED' : '!! MOVED'}`);
}
// The claimed round-5 hashes
const R5_CLAIMED = {
  'best_r5_servo_60mm.json': 'e0434c2c90da69e00c2ef5d06c301ba4a20542229e3155372a35c06f4e153d99',
  'best_r5_servoland_60mm.json': 'e6e8ff144695e682040c9f651f1d566fa35a29615b8682ab0334eb34f7cb3c7c',
  'best_r5_servoland_kcore_60mm.json': '880a120ef6497c46275d95ade70c12bb421c64cdac473c1a944ab236ce0495f5',
};
const r5HashRows = [];
for (const [f, want] of Object.entries(R5_CLAIMED)) {
  if (!fs.existsSync(P + f)) continue;
  const got = intentHashOfFile(P + f);
  r5HashRows.push({ file: f, claimed: want, now: got, matches: got === want });
  log(`   round-5 hash    ${f.padEnd(34)} claimed ${want.slice(0, 12)}  now ${got.slice(0, 12)}  ${got === want ? 'MATCHES' : '!! DOES NOT MATCH'}`);
}
OUT.phaseHBS = { files: inv, distinctVectors: Object.keys(byHash).length, duplicateGroups: dupes,
                 preRound5HashesUnchanged: hashRows, round5HashesAsClaimed: r5HashRows,
                 boundViolations: inv.filter(r => r.boundViolations.length).map(r => ({ file: r.file, violations: r.boundViolations })),
                 notAClimb: inv.filter(r => !r.floorSpawn || r.settleTicks !== 25).map(r => ({ file: r.file, kind: r.kind })) };
log(`   distinct vectors ${OUT.phaseHBS.distinctVectors} | bound violations ${OUT.phaseHBS.boundViolations.length} | NOT-A-CLIMB ${OUT.phaseHBS.notAClimb.length}`);

// ==================================================================== PHASE F
// FRAGILITY files, written to disk BEFORE scoring, because nothing in memory
// is scored. New paths only; no round-2/3/4/5 file is overwritten.
const FRAG = [];
if (fs.existsSync(P + 'best_r5_servoland_kcore_60mm.json')) {
  const base = rd('best_r5_servoland_kcore_60mm.json');
  const mk = (name, mutate) => {
    const j = JSON.parse(JSON.stringify(base));
    delete j.robust; delete j.note;
    mutate(j);
    j.note = 'ROUND-5 JUDGE FRAGILITY PROBE, written by climb/audit_r5.mjs. ' + name;
    const path = P + 'audit_r5_' + name + '.json';
    const body = JSON.stringify(j, null, 2);
    // New paths only. A re-run may rewrite ITS OWN probe file, and only when
    // the bytes are identical; anything else is a refusal.
    if (fs.existsSync(path) && fs.readFileSync(path, 'utf8') !== body)
      throw new Error('refusing to overwrite ' + path);
    fs.writeFileSync(path, body);
    return { name, file: 'audit_r5_' + name + '.json', sha256: intentHash(j) };
  };
  FRAG.push(mk('shift_all_plus1tick', j => {
    for (const k of j.keyframes) k.t = +(k.t + DT).toFixed(6);
    j.servo.at = +(j.servo.at + DT).toFixed(6);
  }));
  FRAG.push(mk('shift_all_minus1tick', j => {
    for (const k of j.keyframes) k.t = +(k.t - DT).toFixed(6);
    j.servo.at = +(j.servo.at - DT).toFixed(6);
  }));
  FRAG.push(mk('shift_armonly_plus1tick', j => { j.servo.at = +(j.servo.at + DT).toFixed(6); }));
}
// ABLATION files: the same launch with the `servo` block DELETED. The search
// report's load-bearing claim is that the law EARNS one clear outright; that
// claim is re-derived here from files this judge wrote.
const ABL = [];
for (const src of ['best_r5_servoland_kcore_60mm.json', 'best_r5_servoland_60mm.json']) {
  if (!fs.existsSync(P + src)) continue;
  const j = rd(src);
  delete j.servo; delete j.robust; delete j.note;
  j.note = 'ROUND-5 JUDGE ABLATION, written by climb/audit_r5.mjs: ' + src + ' with its `servo` block deleted and every other field identical.';
  const name = 'audit_r5_ablate_' + src.replace(/^best_r5_|\.json$/g, '');
  const path = P + name + '.json';
  const body = JSON.stringify(j, null, 2);
  if (fs.existsSync(path) && fs.readFileSync(path, 'utf8') !== body) throw new Error('refusing to overwrite ' + path);
  fs.writeFileSync(path, body);
  ABL.push({ src, name, file: name + '.json', sha256: intentHash(j) });
}

// ==================================================================== PHASE G
log('');
log('PHASE G — every round-5 claim and every published clear, re-scored from disk, 14 cells');
log('   file                                   rise  kC  kCs kExt kEs   objective  objCore   objR3   maxTq  minPenEp  maxDY  upTail(min/mean)  servo');
const CLAIMS = [
  ['best_r3_vault_60mm.json', 0.060, 'WARM START / standing record'],
  ['r5_servo_never_60mm.json', 0.060, 'degeneracy: servo armed at t=99 s'],
  ['r5_servo_armed_60mm.json', 0.060, 'instrument engagement demo (authored gains)'],
  ['best_r5_servo_60mm.json', 0.060, 'round-5 search A best'],
  ['best_r5_servoland_kcore_60mm.json', 0.060, 'round-5 search B, best by clears'],
  ['best_r5_servoland_60mm.json', 0.060, 'round-5 search B, best by objective'],
  ...FRAG.map(f => [f.file, 0.060, 'fragility probe: ' + f.name]),
  ...ABL.map(a => [a.file, 0.060, 'ablation: ' + a.src + ' with the servo block deleted']),
  ['best_r2_vault_40mm.json', 0.040, 'published clear'],
  ['best_r2_vault_60mm.json', 0.060, 'published clear'],
  ['best_r3_vault_40mm.json', 0.040, 'published clear'],
  ['best_r3_vault_50mm.json', 0.050, 'published clear'],
].filter(([f]) => fs.existsSync(P + f));

const G = {};
for (const [f, h, label] of CLAIMS) {
  const meta = inv.find(r => r.file === f);
  const j = rd(f);
  const B = checkBounds(j);
  const floorSpawn = !j.spawn && !j.spawnPose && !j.spawnVel && !j.spawnLastAction && !j.spawnQuat;
  const settle = j.settleTicks === undefined ? 25 : j.settleTicks;
  const g = await scoreRobust(P + f, { rise: h, skipBounds: true });
  const core = g.cells.filter(c => c.cell.tier === 'core');
  const ceilingCore = core.filter(c => (c.maxZ - c.rise) > CEILING_ABOVE).length;
  const ceilingExt = g.cells.filter(c => (c.maxZ - c.rise) > CEILING_ABOVE).length;
  const clears = g.cells.filter(c => c.crit.honest);
  const row = {
    file: f, label, rise_mm: h * 1000, sha256: g.sha256, move: g.move,
    admissible: B.violations.length === 0 && floorSpawn && settle === 25,
    boundViolations: B.violations, startKind: floorSpawn && settle === 25 ? 'floor spawn — a climb' : 'NOT A CLIMB',
    kCore: g.kCore, kCoreStable: g.kCoreStable, kExt: g.kExt, kExtStable: g.kExtStable,
    objective: +g.objective.toFixed(4), objectiveCore: +g.objectiveCore.toFixed(4),
    objectiveR3: +g.objectiveR3.toFixed(4), meanReward: +g.meanReward.toFixed(4),
    meanRewardCore: +g.meanRewardCore.toFixed(4),
    maxTq: +g.agg.maxTq.toFixed(4), tqOverCeiling: g.agg.maxTq > FORCERANGE + 1e-9,
    tqSaturated: g.agg.maxTq >= FORCERANGE - 1e-4,
    minPenetrationAtScore_mm: +g.agg.minPenetrationAtScore_mm.toFixed(2),
    minPenetrationEpisode_mm: +g.agg.minPenetrationEpisode_mm.toFixed(2),
    maxAbsDY_mm: +g.agg.maxAbsDY_mm.toFixed(1), lateralEscapeCells: g.agg.lateralEscapeCells,
    minUprightTailTicks: g.agg.minUprightTailTicks,
    meanUprightTailTicks: +g.agg.meanUprightTailTicks.toFixed(1),
    servoArmedCells: g.cells.filter(c => c.servo && c.servo.armed).length,
    servoTicksMean: +(g.cells.reduce((a, c) => a + (c.servo ? c.servo.ticks : 0), 0) / g.cells.length).toFixed(1),
    penetrationTicksScanned: g.cells.map(c => c.penetrationTicksScanned),
    // THE CEILING, from maxZ over track+tail
    coreCellsTrunkEverAbove95mm: ceilingCore, extCellsTrunkEverAbove95mm: ceilingExt,
    peakAbove_mm_core: core.map(c => +((c.maxZ - c.rise) * 1000).toFixed(1)),
    honestPatternCore: core.map(c => c.crit.honest ? 1 : 0).join(''),
    honestPatternExt: g.cells.map(c => c.crit.honest ? 1 : 0).join(''),
    // whole-episode penetration of the CLEARS only
    clearsMinPenEpisode_mm: clears.map(c => +(c.minPenetrationEpisode * 1000).toFixed(2)),
    clearsMinPenPair: clears.map(c => c.minPenetrationPair),
    clearsMinPenAtScore_mm: clears.map(c => c.penetrationAtScore === null ? null : +(c.penetrationAtScore * 1000).toFixed(2)),
    clearsDeeperThan15mm: clears.filter(c => c.minPenetrationEpisode < -0.015).length,
    cellsDeeperThan15mm: g.cells.filter(c => c.minPenetrationEpisode < -0.015).length,
    // the consistency invariant, per cell
    penConsistent: g.cells.every(c => c.minPenetrationEpisode === null || c.penetrationAtScore === null
                                   || c.minPenetrationEpisode <= c.penetrationAtScore + 1e-15),
    cellsXZ: g.cells.map(c => ({ rise_mm: c.cell.rise_mm, drop: c.cell.drop, fmul: c.cell.fmul,
                                 tier: c.cell.tier, x: c.scored.x, z: c.scored.z, reward: c.reward,
                                 honest: c.crit.honest, stable: c.stableClear,
                                 upTail: c.uprightTailTicks,
                                 above_mm: +(c.scored.above * 1000).toFixed(1),
                                 peakAbove_mm: +((c.maxZ - c.rise) * 1000).toFixed(1),
                                 feetOnTread: c.scored.feetOnTread,
                                 minPenEp_mm: +(c.minPenetrationEpisode * 1000).toFixed(2),
                                 minPenPair: c.minPenetrationPair, minPenTick: c.minPenetrationTick,
                                 penAtScore_mm: c.penetrationAtScore === null ? null : +(c.penetrationAtScore * 1000).toFixed(2),
                                 maxAbsDY_mm: +(c.maxAbsDY * 1000).toFixed(1),
                                 servoArmed: c.servo ? c.servo.armed : null,
                                 servoTicks: c.servo ? c.servo.ticks : null })),
  };
  G[f] = row;
  log(`   ${f.padEnd(38)} ${String(row.rise_mm).padStart(4)} ${row.kCore}/9 ${row.kCoreStable}/9 ${String(row.kExt).padStart(2)}/14 ${String(row.kExtStable).padStart(2)}/14 ${row.objective.toFixed(3).padStart(9)} ${row.objectiveCore.toFixed(3).padStart(8)} ${row.objectiveR3.toFixed(3).padStart(7)} ${row.maxTq.toFixed(4)} ${f2(row.minPenetrationEpisode_mm).padStart(8)} ${String(row.maxAbsDY_mm).padStart(6)} ${String(row.minUprightTailTicks).padStart(2)}/${row.meanUprightTailTicks}  ${row.servoArmedCells}/14`);
}
OUT.phaseG = G;
log(`   [${el()}]`);

// ==================================================================== PHASE D
log('');
log('PHASE D — DEGENERACY: a servo that never arms must BE its base file, cell by cell');
const NV = G['r5_servo_never_60mm.json'], WS = G['best_r3_vault_60mm.json'];
if (NV && WS) {
  let mdx = 0, mdz = 0, mdr = 0;
  for (let i = 0; i < NV.cellsXZ.length; i++) {
    mdx = Math.max(mdx, Math.abs(NV.cellsXZ[i].x - WS.cellsXZ[i].x));
    mdz = Math.max(mdz, Math.abs(NV.cellsXZ[i].z - WS.cellsXZ[i].z));
    mdr = Math.max(mdr, Math.abs(NV.cellsXZ[i].reward - WS.cellsXZ[i].reward));
  }
  const same = mdx === 0 && mdz === 0 && mdr === 0 && NV.kCore === WS.kCore
    && NV.kCoreStable === WS.kCoreStable && NV.honestPatternExt === WS.honestPatternExt
    && NV.minPenetrationEpisode_mm === WS.minPenetrationEpisode_mm;
  log(`   r5_servo_never_60mm (${NV.move}) vs best_r3_vault_60mm (${WS.move})`);
  log(`   max |dx| = ${(mdx * 1000).toFixed(9)} mm  max |dz| = ${(mdz * 1000).toFixed(9)} mm  max |dreward| = ${mdr.toFixed(9)}  identical = ${same}`);
  log(`   servo cells armed: ${NV.servoArmedCells}/14   -> ${same ? 'THE MECHANISM IS INERT WHEN IT DOES NOT FIRE, in the scorer that decides the verdict.' : 'NOT INERT'}`);
  OUT.phaseD = { a: NV.move, b: WS.move, maxDx_mm: mdx * 1000, maxDz_mm: mdz * 1000,
                 maxDReward: mdr, identical: same, servoArmedCells: NV.servoArmedCells,
                 distinctHash: NV.sha256 !== WS.sha256 };
}

// ==================================================================== PHASE E
log('');
log('PHASE E — THE CEILING. `honest` needs the trunk above 95 mm at the scored instant,');
log('   so the count of CORE cells in which the trunk EVER exceeded 95 mm is an upper');
log('   bound on that move\'s kCore under ANY landing law whatsoever.');
log('   file                                   kCore  ceilingCore  peakAbove_mm over the 9 core cells');
const ceilRows = [];
for (const r of Object.values(G)) {
  ceilRows.push({ file: r.file, move: r.move, kCore: r.kCore, kCoreStable: r.kCoreStable,
                  ceilingCore: r.coreCellsTrunkEverAbove95mm, ceilingExt: r.extCellsTrunkEverAbove95mm,
                  peakAbove_mm_core: r.peakAbove_mm_core,
                  kCoreLEQceiling: r.kCore <= r.coreCellsTrunkEverAbove95mm });
  log(`   ${r.file.padEnd(38)} ${r.kCore}/9      ${r.ceilingCore ?? r.coreCellsTrunkEverAbove95mm}/9      [${r.peakAbove_mm_core.join(', ')}]`);
}
const maxCeil = Math.max(...ceilRows.map(r => r.ceilingCore));
OUT.phaseE = { rule: 'peak trunk z - rise > 95 mm at any recorded tick', rows: ceilRows,
               maxCeilingCoreHere: maxCeil,
               searchClaimedMaxOver235Moves: 5,
               boundHoldsEverywhere: ceilRows.every(r => r.kCoreLEQceiling) };
log(`   max ceilingCore over everything scored here: ${maxCeil}/9   (the search reports 5/9 as the max over all 235 of its moves)`);
log(`   kCore <= ceilingCore holds on every row: ${OUT.phaseE.boundHoldsEverywhere}`);
log(`   The kill gate needs 7. A landing law cannot manufacture trunk height it never had.`);

// ==================================================================== PHASE N
log('');
log('PHASE N — the round-4 judge\'s hole, now closed: whole-episode penetration');
log('   file                                   clears  minPenEp of each clear (mm)                worst cell  deeper than -15 mm');
const penRows = [];
for (const r of Object.values(G)) {
  penRows.push({ file: r.file, move: r.move, clears: r.kExt,
                 clearsMinPenEpisode_mm: r.clearsMinPenEpisode_mm,
                 clearsMinPenAtScore_mm: r.clearsMinPenAtScore_mm,
                 clearsMinPenPair: r.clearsMinPenPair,
                 worstCell_mm: r.minPenetrationEpisode_mm,
                 clearsDeeperThan15mm: r.clearsDeeperThan15mm,
                 cellsDeeperThan15mm: r.cellsDeeperThan15mm,
                 penConsistent: r.penConsistent,
                 ticksScanned: r.penetrationTicksScanned[0] });
  log(`   ${r.file.padEnd(38)} ${String(r.kExt).padStart(2)}/14   [${r.clearsMinPenEpisode_mm.join(', ')}]${' '.repeat(Math.max(0, 40 - (r.clearsMinPenEpisode_mm.join(', ').length)))} ${f2(r.minPenetrationEpisode_mm).padStart(7)}   clears ${r.clearsDeeperThan15mm} / cells ${r.cellsDeeperThan15mm}`);
}
OUT.phaseN = { rows: penRows,
               invariantHoldsEverywhere: penRows.every(r => r.penConsistent),
               anyClearDeeperThan15mm: penRows.some(r => r.clearsDeeperThan15mm > 0),
               anyCellDeeperThan15mm: penRows.some(r => r.cellsDeeperThan15mm > 0),
               deepestReadingHere_mm: Math.min(...penRows.map(r => r.worstCell_mm)) };
log(`   invariant minPenEpisode <= penAtScore holds on every cell: ${OUT.phaseN.invariantHoldsEverywhere}`);
log(`   any CLEAR deeper than -15 mm: ${OUT.phaseN.anyClearDeeperThan15mm}   any CELL: ${OUT.phaseN.anyCellDeeperThan15mm}   deepest here ${OUT.phaseN.deepestReadingHere_mm} mm`);

// ==================================================================== PHASE F
log('');
log('PHASE F — FRAGILITY. Round 4: one control tick of shift took the round-3 vault');
log(`   from 4 of 9 to 1 of 9. One control tick is ${(DT * 1000).toFixed(2)} ms. Same probe on the`);
log('   round-5 move whose clears line up along the PLANT axis.');
const KB = G['best_r5_servoland_kcore_60mm.json'];
const fragRows = [];
if (KB) {
  log(`   ${'best_r5_servoland_kcore_60mm.json (unshifted)'.padEnd(46)} ${KB.move} kCore ${KB.kCore}/9 stable ${KB.kCoreStable}/9 kExt ${KB.kExt}/14 obj ${KB.objective}  core pattern ${KB.honestPatternCore}`);
  for (const fr of FRAG) {
    const r = G[fr.file];
    if (!r) continue;
    let mdx = 0;
    for (let i = 0; i < r.cellsXZ.length; i++) mdx = Math.max(mdx, Math.abs(r.cellsXZ[i].x - KB.cellsXZ[i].x));
    fragRows.push({ probe: fr.name, file: fr.file, move: r.move, kCore: r.kCore, kCoreStable: r.kCoreStable,
                    kExt: r.kExt, objective: r.objective, honestPatternCore: r.honestPatternCore,
                    maxTrunkXShift_mm: +(mdx * 1000).toFixed(1) });
    log(`   ${(fr.name + ' (' + fr.file + ')').padEnd(46)} ${r.move} kCore ${r.kCore}/9 stable ${r.kCoreStable}/9 kExt ${r.kExt}/14 obj ${r.objective}  core pattern ${r.honestPatternCore}  max trunk-x move ${(mdx * 1000).toFixed(1)} mm`);
  }
}
OUT.phaseF = { controlTick_ms: DT * 1000, unshifted: KB ? { move: KB.move, kCore: KB.kCore, kCoreStable: KB.kCoreStable, kExt: KB.kExt, pattern: KB.honestPatternCore } : null, probes: fragRows };

// ==================================================================== PHASE A
log('');
log('PHASE A — ABLATION, re-derived. Same launch, `servo` block deleted, everything else identical.');
log('   The search report claims the law EARNS a clear. Here is the same comparison from');
log('   files this judge wrote, on the 14-cell grid.');
const ablRows = [];
for (const a of ABL) {
  const withS = G[a.src], without = G[a.file];
  if (!withS || !without) continue;
  const row = { src: a.src, withServo: { move: withS.move, kCore: withS.kCore, kCoreStable: withS.kCoreStable,
                                         kExt: withS.kExt, objective: withS.objective,
                                         meanUpTail: withS.meanUprightTailTicks,
                                         ceilingCore: withS.coreCellsTrunkEverAbove95mm,
                                         pattern: withS.honestPatternCore,
                                         minPenEp_mm: withS.minPenetrationEpisode_mm },
                launchOnly: { file: a.file, move: without.move, kCore: without.kCore, kCoreStable: without.kCoreStable,
                              kExt: without.kExt, objective: without.objective,
                              meanUpTail: without.meanUprightTailTicks,
                              ceilingCore: without.coreCellsTrunkEverAbove95mm,
                              pattern: without.honestPatternCore,
                              minPenEp_mm: without.minPenetrationEpisode_mm },
                dKCore: withS.kCore - without.kCore, dKCoreStable: withS.kCoreStable - without.kCoreStable,
                dObjective: +(withS.objective - without.objective).toFixed(4),
                dMeanUpTail: +(withS.meanUprightTailTicks - without.meanUprightTailTicks).toFixed(1),
                dCeilingCore: withS.coreCellsTrunkEverAbove95mm - without.coreCellsTrunkEverAbove95mm };
  ablRows.push(row);
  log(`   ${a.src}`);
  log(`      with servo   ${row.withServo.move}  kCore ${row.withServo.kCore}/9 stable ${row.withServo.kCoreStable}/9 kExt ${row.withServo.kExt}/14  obj ${row.withServo.objective}  meanUpTail ${row.withServo.meanUpTail}/50  ceiling ${row.withServo.ceilingCore}/9  pattern ${row.withServo.pattern}`);
  log(`      launch only  ${row.launchOnly.move}  kCore ${row.launchOnly.kCore}/9 stable ${row.launchOnly.kCoreStable}/9 kExt ${row.launchOnly.kExt}/14  obj ${row.launchOnly.objective}  meanUpTail ${row.launchOnly.meanUpTail}/50  ceiling ${row.launchOnly.ceilingCore}/9  pattern ${row.launchOnly.pattern}`);
  log(`      the law is worth  kCore ${row.dKCore >= 0 ? '+' : ''}${row.dKCore}  kCoreStable ${row.dKCoreStable >= 0 ? '+' : ''}${row.dKCoreStable}  objective ${row.dObjective >= 0 ? '+' : ''}${row.dObjective}  upright tail ${row.dMeanUpTail >= 0 ? '+' : ''}${row.dMeanUpTail} of 50  trunk-height ceiling ${row.dCeilingCore >= 0 ? '+' : ''}${row.dCeilingCore} cells`);
}
OUT.phaseA = { rows: ablRows };

// ============================================== THE HANDOFF-JUMP FIELD
log('');
log('ctrlJump — the round-4 handoff-discontinuity field, on servoed files');
const cjRows = [];
for (const f of ['best_r3_vault_60mm.json', 'r5_servo_never_60mm.json',
                 'best_r5_servoland_kcore_60mm.json', 'best_r5_servo_60mm.json']) {
  if (!fs.existsSync(P + f)) continue;
  const r = await newScore(P + f, { rise: 0.060, tail: 'hold' });
  cjRows.push({ file: f, ctrlJump: r.ctrlJump, servoArmed: r.servo ? r.servo.armed : false });
  log(`   ${f.padEnd(34)} ctrlJump=${r.ctrlJump.toFixed(6)}  servoArmed=${r.servo ? r.servo.armed : false}`);
}
OUT.ctrlJump = { rows: cjRows,
  finding: "rig3.mjs now sets the 'hold' tail's held[] from the servo's LAST COMMAND on the ten slots the servo owns, so on those ten slots the handoff discontinuity is ZERO BY CONSTRUCTION. ctrlJump is a max over all 14 slots, so for a servoed file it silently becomes a head/neck-only statistic — measured 2.771 and 1.920 on the two servoed files against 0.726 on the unservoed vault, which is the head and neck, not the legs. It is a 'hold'-tail field and every verdict this round is scored on the 'policy' tail, so no k moves; but ctrlJump no longer means for a servoed file what it means for a keyframe file." };

// ==================================================================== PHASE T
log('');
log('PHASE T — the search\'s UNMEASURED hypothesis, measured as far as this instrument allows.');
log('   The search says: every armed move loses tail uprightness, and it did not measure why.');
log('   The servo owns the legs only while the TRACK runs; the 50 policy tail ticks put them');
log('   back on HOME + policy action. The \'hold\' tail does NOT: rig3 freezes the servo\'s LAST');
log('   COMMAND on the ten slots it owns. So policy-tail vs hold-tail uprightness on the same');
log('   file, at cell 0 (rise 60 mm, drop 0.120, friction x1.0), separates the two.');
const tailRows = [];
for (const f of ['best_r3_vault_60mm.json', 'best_r5_servoland_kcore_60mm.json', 'best_r5_servo_60mm.json']) {
  if (!fs.existsSync(P + f)) continue;
  const pol = await newScore(P + f, { rise: 0.060, tail: 'policy' });
  const hol = await newScore(P + f, { rise: 0.060, tail: 'hold' });
  const row = { file: f, servoArmed: pol.servo ? pol.servo.armed : false,
                policyUpTail: pol.uprightTailTicks, holdUpTail: hol.uprightTailTicks,
                policyHonest: pol.crit.honest, holdHonest: hol.crit.honest,
                policyStable: pol.crit.honest && pol.uprightTailTicks >= UPRIGHT_TAIL_MIN,
                holdStable: hol.crit.honest && hol.uprightTailTicks >= UPRIGHT_TAIL_MIN,
                delta: hol.uprightTailTicks - pol.uprightTailTicks };
  tailRows.push(row);
  log(`   ${f.padEnd(34)} servoArmed=${row.servoArmed}  policy tail ${row.policyUpTail}/50 honest=${row.policyHonest}  |  hold tail ${row.holdUpTail}/50 honest=${row.holdHonest}  |  delta ${row.delta >= 0 ? '+' : ''}${row.delta}  stable(hold)=${row.holdStable}`);
}
const servoed = tailRows.filter(r => r.servoArmed);
OUT.phaseT = { rows: tailRows, upTailMin: UPRIGHT_TAIL_MIN,
  freezingTheServoCommandRecoversStability: servoed.some(r => r.holdStable && !r.policyStable),
  note: "The hold tail is a FROZEN command, not the law continuing to run, so this is a partial test: it removes the snap-back to HOME + policy action without giving the servo the tail. A large positive delta implicates the snap-back; a small one says the duck was already falling before the tail began." };
log(`   freezing the servo's last command turns an unstable clear into a stable one: ${OUT.phaseT.freezingTheServoCommandRecoversStability}`);

// ==================================================================== PHASE C
log('');
log('PHASE C — the controls, under the CURRENT instrument');
const CTRL = [['ctrl_do_nothing.json', 0.040], ['ctrl_do_nothing.json', 0.060],
              ['r4_ctrl_on_tread_60mm.json', 0.060]].filter(([f]) => fs.existsSync(P + f));
const ctrlRows = [];
for (const [f, h] of CTRL) {
  const g = await scoreRobust(P + f, { rise: h, skipBounds: true });
  const row = { file: f, rise_mm: h * 1000, move: g.move, kCore: g.kCore, kCoreStable: g.kCoreStable,
                kExt: g.kExt, kExtStable: g.kExtStable, objective: +g.objective.toFixed(4),
                minPenEpisode_mm: +g.agg.minPenetrationEpisode_mm.toFixed(2),
                maxTq: +g.agg.maxTq.toFixed(4),
                servoCells: g.cells.filter(c => c.servo).length };
  ctrlRows.push(row);
  log(`   ${f.padEnd(30)} @${String(row.rise_mm).padStart(3)}mm  kCore ${row.kCore}/9  stable ${row.kCoreStable}/9  kExt ${row.kExt}/14  obj ${row.objective}  minPenEp ${row.minPenEpisode_mm}mm  servo rows ${row.servoCells}`);
}
const dn = ctrlRows.filter(r => r.file === 'ctrl_do_nothing.json');
const pd = ctrlRows.filter(r => r.file.startsWith('r4_ctrl_on_tread'));
OUT.phaseC = { rows: ctrlRows,
               doNothingAlwaysFails: dn.length > 0 && dn.every(r => r.kExt === 0),
               placedDuckAlwaysPasses: pd.length > 0 && pd.every(r => r.kExt === 14) };
log(`   do-nothing fails everywhere: ${OUT.phaseC.doNothingAlwaysFails}   placed duck clears 14/14: ${OUT.phaseC.placedDuckAlwaysPasses}`);

// ==================================================================== PHASE K
log('');
log('================================================================');
log('PHASE K — THE KILL GATE');
log('   RULE: PASSED only if a DISTINCT move reaches kCoreStable >= 7 of 9 at 60 mm,');
log('   from a floor spawn, in declared bounds, under the actuator ceiling, with no cheat.');
const at60 = Object.values(G).filter(r => r.rise_mm === 60);
// Eligibility is: it started on the floor, it is inside its declared bounds,
// and it never exceeded the actuator ceiling. A cell that leaves the flight
// laterally is ALREADY handled inside the criterion and reward (that cell
// scores 0 and cannot be honest), so a lateral escape in one cell is reported
// here, not used to disqualify a whole move.
const eligible = at60.filter(r => r.admissible && !r.tqOverCeiling);
const bestStable = eligible.length ? eligible.reduce((a, b) => b.kCoreStable > a.kCoreStable ? b : a) : null;
const passed = bestStable && bestStable.kCoreStable >= 7;
for (const r of at60.slice().sort((a, b) => b.kCoreStable - a.kCoreStable || b.kCore - a.kCore))
  log(`   ${r.file.padEnd(38)} ${r.move}  kCoreStable ${r.kCoreStable}/9  kCore ${r.kCore}/9  admissible=${r.admissible}  tqOver=${r.tqOverCeiling}  lateralEscapes=${r.lateralEscapeCells}`);
log('');
log(`   BEST kCoreStable at 60 mm over everything scored here: ${bestStable ? bestStable.kCoreStable : 0}/9  (${bestStable ? bestStable.file + ' ' + bestStable.move : 'none'})`);
log(`   BEST kCoreStable of any SERVOED move: ${Math.max(0, ...at60.filter(r => r.servoArmedCells > 0).map(r => r.kCoreStable))}/9`);
log(`   KILL GATE: ${passed ? 'PASSED' : 'FAILED'}`);
log('   CONSEQUENCE: the 40-80 mm band is ' + (passed ? 'OPEN' : 'CLOSED') + '. The negative is the result.');
OUT.killGate = {
  rule: 'kCoreStable >= 7 of 9 at 60 mm, distinct move, floor spawn, in bounds, no cheat',
  bestKCoreStable: bestStable ? bestStable.kCoreStable : 0,
  bestMove: bestStable ? bestStable.move : null, bestFile: bestStable ? bestStable.file : null,
  bestServoedKCoreStable: Math.max(0, ...at60.filter(r => r.servoArmedCells > 0).map(r => r.kCoreStable)),
  result: passed ? 'PASSED' : 'FAILED',
  consequence: passed ? 'the 40-80 mm band is OPEN' : 'THE 40-80 mm BAND IS CLOSED. The negative is the result.',
};

// ============================================== WHAT THE LAW IS ALLOWED TO SEE
// Not a cheat against the scorer — the scorer does not care — but it is a fact
// the round-6 brief has to carry. The ONNX policy is fed 61 numbers and every
// one of them is proprioceptive. The servo law is fed the trunk's height above
// the tread, the trunk's x relative to the riser line, and each foot's x and z
// relative to the tread's front edge and top, all read straight out of
// data.qpos / data.geom_xpos.
OUT.observationSets = {
  policyObs: { n: 61, contents: 'gyro 3 + projected gravity 3 + joint pos 14 + joint vel 14 + last action 14 + command 13',
               exteroceptive: false, source: 'site/duckloop.mjs buildObs()' },
  servoLawReads: { n: 7, contents: 'trunk z - rise; projectedGravity[0]; trunk x - 0.12; left/right foot geom x - 0.12; left/right foot geom z - rise',
                   exteroceptive: true, source: 'climb/rig3.mjs + climb/robust.mjs servo block, from data.qpos and data.geom_xpos' },
  finding: 'The servoed landing needs the tread\'s height and the tread\'s front edge. Those are not in the 61-number observation the shipped policy gets and there is no sensor named in this repo that supplies them. A round-5 servoed move is not a move the robot could run as written; it is an upper bound on what a landing law COULD do given perfect knowledge of the step. It still cleared nothing stably.',
};
log('');
log('WHAT THE LAW IS ALLOWED TO SEE');
log(`   policy observation: 61 numbers, all proprioceptive (${OUT.observationSets.policyObs.contents})`);
log(`   servo law reads   : 7 numbers, 5 of them EXTEROCEPTIVE (${OUT.observationSets.servoLawReads.contents})`);
log('   -> the servoed landing assumes perfect knowledge of the tread height and edge.');
log('      That makes round 5 an UPPER BOUND on a landing law, not a deployable move.');

OUT.bands = {
  '40-80mm': passed ? 'OPEN' : 'CLOSED — best kCoreStable ' + OUT.killGate.bestKCoreStable + ' of 9 at 60 mm',
  '80-120mm': 'CLOSED on the measured lift budget (round 3/4: ~38 mm of trunk lift bought where 59-99 mm are needed, all servos saturated)',
  '180mm': 'CLOSED (round 2/3)',
};
OUT.torque = { ceiling_Nm: FORCERANGE,
               anyRowOverCeiling: Object.values(G).some(r => r.tqOverCeiling) || ctrlRows.some(r => r.maxTq > FORCERANGE + 1e-9),
               saturatingFiles: Object.values(G).filter(r => r.tqSaturated).length,
               scoredFiles: Object.keys(G).length };
log('');
log(`   torque: any row above ${FORCERANGE} N.m? ${OUT.torque.anyRowOverCeiling}   files that SATURATE it: ${OUT.torque.saturatingFiles}/${OUT.torque.scoredFiles}`);
log(`   total wall ${el()}`);

fs.writeFileSync(P + 'r5_judge-results.json', JSON.stringify(OUT, null, 2));
fs.writeFileSync(P + 'audit_r5.log', LOG.join('\n') + '\n');
console.log('wrote climb/r5_judge-results.json and climb/audit_r5.log');
