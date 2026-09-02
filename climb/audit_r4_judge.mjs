// ROUND 4 — THE JUDGE'S LOOP.
//
// FILENAME NOTE, READ THIS FIRST. The brief said "write climb/audit_r4.mjs".
// That file ALREADY EXISTED when this judge started (mtime 09-02 02:59): it is
// the instrument agent's hole-closing proof and it is the cited source of
// climb/r4_audit-results.json, climb/audit_r4.log and climb/audit_r4.stdout.
// Overwriting it would have destroyed the evidence chain for three published
// artifacts. This judge therefore writes climb/audit_r4_judge.mjs instead and
// says so in its own output. Nothing in climb/audit_r4.mjs was touched.
//
// WHAT THIS FILE DOES, IN ORDER. Every number it prints is measured HERE, in
// this process, from a SAVED JSON file on disk. It re-derives every claim both
// families made rather than quoting them.
//
//   PHASE P   PARITY to the CURRENT rig3.mjs. climb/rig3_pre_r4.mjs is the
//             byte copy of rig3.mjs as it stood before round 4. Since that copy
//             was taken, THREE agents have edited rig3.mjs (the instrument
//             agent's three holes, family A's `event`, family B's handoff
//             spawn). Every pre-round-4 saved file is scored through BOTH and
//             compared at FULL FLOAT DIGITS. The only field permitted to differ
//             is a criterion that the whole-episode lateral gate now denies,
//             and every such row is printed. This is the check that makes the
//             rest of the round admissible.
//   PHASE R   Is robust.mjs cell 0 still rig3.mjs? Same files, full digits.
//   PHASE H   Intent hashes of every best_* and control. One vector under two
//             labels is ONE move; the duplicate groups are printed by hash.
//   PHASE B   Declared bounds, enforced at scoring time, on every file.
//   PHASE S   SPAWN CLASSIFICATION. A file carrying spawn / spawnPose /
//             spawnVel / spawnLastAction / settleTicks!=25 did not start on the
//             floor. It is labelled NOT-A-CLIMB and its k is reported inside
//             that label, never beside a floor-spawned move's k.
//   PHASE G   Re-score EVERY claimed clear and every round-4 best on the
//             14-cell extended grid, from disk, recording kCore / kCoreStable /
//             kExt / kExtStable, the three objectives, maxTq against the
//             0.6405 N.m ceiling, penetrationAtScore, whole-episode maxAbsDY
//             and uprightTailTicks.
//   PHASE D   The duplicate-behaviour claim: family A says its 60 mm file is
//             the round-3 vector re-expressed. Checked cell by cell at full
//             digits on trunk x and z.
//   PHASE C   The controls, on the same plant: do-nothing must fail, a duck
//             PLACED on the tread must pass.
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/audit_r4_judge.mjs
import fs from 'node:fs';
import { scoreSaved as newScore, criteria, reward, LATERAL, DUCKG, RISER_X } from '../climb/rig3.mjs';
import { scoreSaved as oldScore } from '../climb/rig3_pre_r4.mjs';
import { scoreRobust, scoreCell, intentHashOfFile, checkBounds,
         DHS, PLANTS, EXT_DHS, EXT_PLANT, CLEAR_BONUS, UPRIGHT_BONUS,
         UPRIGHT_TAIL_MIN, DECLARED_BOUNDS } from '../climb/robust.mjs';

const P = '../climb/';
const LOG = [];
const log = s => { console.log(s); LOG.push(String(s)); };
const T0 = Date.now();
const el = () => ((Date.now() - T0) / 1000).toFixed(0) + 's';
const FORCERANGE = 0.6405;                       // N.m, every actuator
const OUT = {
  generated: new Date().toISOString(),
  judge: 'audit_r4_judge.mjs',
  filenameNote: 'climb/audit_r4.mjs already existed (the instrument agent proof, source of r4_audit-results.json); it was NOT overwritten. This judge is climb/audit_r4_judge.mjs.',
  lateralGate_mm: LATERAL * 1000, duckGeoms: DUCKG.length, riserLine_mm: RISER_X * 1000,
  forcerange_Nm: FORCERANGE, declaredBounds: DECLARED_BOUNDS,
  uprightTailMin: UPRIGHT_TAIL_MIN, clearBonus: CLEAR_BONUS, uprightBonus: UPRIGHT_BONUS,
  grid: { core: { dhs_mm: DHS.map(d => d * 1000), plants: PLANTS },
          ext: { dhs_mm: EXT_DHS.map(d => d * 1000), plant: EXT_PLANT, n: 14 } },
};

const riseOf = f => { const m = f.match(/_(\d+)mm/); return m ? parseInt(m[1], 10) / 1000 : null; };
const ls = re => fs.readdirSync(P).filter(f => re.test(f)).sort();
const rd = f => JSON.parse(fs.readFileSync(P + f, 'utf8'));

// ==================================================================== PHASE P
log('================================================================');
log('PHASE P — CURRENT rig3.mjs vs climb/rig3_pre_r4.mjs, full float digits');
log(`   duck collidable geoms ${DUCKG.length} | lateral gate ${(LATERAL * 1000).toFixed(0)} mm | riser line ${(RISER_X * 1000).toFixed(0)} mm`);
log('   Three agents have edited rig3.mjs since that byte copy was taken.');

// Every file written BEFORE round 4 — none of them carries event/spawnPose, so
// the old harness can replay all of them.
const PRE_R4 = [
  ...ls(/^best_r2_.*\.json$/),
  ...ls(/^best_r3_.*\.json$/),
  ...ls(/^best_[012]_\d+mm\.json$/),
  'ctrl_do_nothing.json', 'ctrl_on_tread_40mm.json', 'ctrl_on_tread_60mm.json',
  'ctrl_on_tread_90mm.json', 'ctrl_on_tread_120mm.json', 'ctrl_walk_only.json',
].filter(f => fs.existsSync(P + f));

const SCAL = ['x0', 'maxX', 'maxZ', 'maxAbsDY', 'feetOnTreadMax', 'feetHighMax',
              'headFrac', 'riserFrac', 'wallFrac', 'upFrac', 'satFrac', 'z0Settle',
              'liftIntegral', 'footNear', 'bothNear', 'reward'];
const SNAP = ['x', 'y', 'z', 'dy', 'above', 'up', 'feetUpRaw', 'feetUpLat', 'feetOnTread'];
const CRIT = ['orig', 'lat', 'honest', 'honest60'];

const parity = { rows: 0, stateExact: 0, rewardExact: 0, critRows: [], mismatch: [] };
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
    if (cd.length) parity.critRows.push({ file: f, rise_mm: h * 1000, tail, changed: cd,
      now: Object.fromEntries(cd.map(k => [k, A.crit[k]])),
      pre: Object.fromEntries(cd.map(k => [k, B.crit[k]])),
      maxAbsDY_mm: +(A.maxAbsDY * 1000).toFixed(1),
      lateralAtScore: A.crit.lateralAtScore, lateralEpisode: A.crit.lateralEpisode,
      lateralSource: A.crit.lateralSource });
    if (stateDiffs.length) parity.mismatch.push({ file: f, tail, diffs: stateDiffs });
  }
}
log(`   ${parity.rows} rows (${PRE_R4.length} pre-round-4 files x tails policy+hold)`);
log(`   PHYSICAL STATE identical at full float digits: ${parity.stateExact}/${parity.rows}`);
log(`   reward() identical                           : ${parity.rewardExact}/${parity.rows}`);
log(`   rows where a CRITERION changed               : ${parity.critRows.length}`);
for (const r of parity.critRows)
  log(`      ${r.file} @${r.rise_mm}mm tail=${r.tail}  ${r.changed.join(',')}  pre=${JSON.stringify(r.pre)} now=${JSON.stringify(r.now)}  maxAbsDY=${r.maxAbsDY_mm}mm (gate ${(LATERAL * 1000).toFixed(0)}mm) atScore=${r.lateralAtScore} episode=${r.lateralEpisode}`);
for (const m of parity.mismatch) log(`   !! STATE MISMATCH ${m.file} tail=${m.tail}: ${JSON.stringify(m.diffs)}`);
OUT.phaseP = { files: PRE_R4.length, rows: parity.rows, stateExact: parity.stateExact,
               rewardExact: parity.rewardExact, criterionChangedRows: parity.critRows,
               stateMismatches: parity.mismatch,
               verdict: parity.mismatch.length === 0 ? 'ADMISSIBLE — no physical state changed' : 'INADMISSIBLE' };
log(`   VERDICT: ${OUT.phaseP.verdict}   [${el()}]`);

// ==================================================================== PHASE R
log('');
log('PHASE R — robust.mjs cell 0 (drop 0.120, x1.0, isolate) vs rig3.scoreSaved');
const RSET = ['best_r3_vault_60mm.json', 'best_r3_vault_40mm.json', 'best_r2_vault_90mm.json',
              'best_r3_landvault_90mm.json', 'ctrl_do_nothing.json', 'best_r4_famA_60mm.json',
              'best_r4_famB_concat_90mm.json', 'r4_ctrl_on_tread_60mm.json'];
const rrows = [];
for (const f of RSET) {
  if (!fs.existsSync(P + f)) continue;
  const h = riseOf(f) ?? 0.060;
  const A = await newScore(P + f, { rise: h, tail: 'policy' });
  const B = await scoreCell(P + f, { rise: h, isolate: true, skipBounds: true });
  const ok = A.scored.x === B.scored.x && A.scored.z === B.scored.z && A.reward === B.reward
    && A.crit.honest === B.crit.honest && A.scored.feetOnTread === B.scored.feetOnTread
    && A.uprightTailTicks === B.uprightTailTicks && A.maxAbsDY === B.maxAbsDY;
  rrows.push({ file: f, rise_mm: h * 1000, exact: ok, rig3_x: A.scored.x, robust_x: B.scored.x,
               rig3_z: A.scored.z, robust_z: B.scored.z, rig3_reward: A.reward, robust_reward: B.reward });
  log(`   ${f.padEnd(32)} EXACT=${ok}  x=${A.scored.x} z=${A.scored.z} rew=${A.reward.toFixed(6)} upTail=${A.uprightTailTicks}`);
}
OUT.phaseR = { rows: rrows, allExact: rrows.every(r => r.exact) };
log(`   allExact = ${OUT.phaseR.allExact}   [${el()}]`);

// ==================================================================== PHASE H+B+S
log('');
log('PHASE H / B / S — identity, bounds, and how the duck STARTED');
const ALLF = [...ls(/^best_r[234]_.*\.json$/), ...ls(/^r4_ctrl_on_tread_\d+mm\.json$/), 'ctrl_do_nothing.json'];
const inv = [];
for (const f of ALLF) {
  const j = rd(f);
  const sha = intentHashOfFile(P + f);
  const B = checkBounds(j);
  const floorSpawn = !j.spawn && !j.spawnPose && !j.spawnVel && !j.spawnLastAction && !j.spawnQuat;
  const settle = j.settleTicks === undefined ? 25 : j.settleTicks;
  const kind = j.spawnPose ? 'HANDOFF SPAWN — NOT A CLIMB'
             : j.spawn ? 'PLACED SPAWN — NOT A CLIMB'
             : settle !== 25 ? `settleTicks=${settle} — NOT THE STANDARD START`
             : 'floor spawn — a climb';
  inv.push({ file: f, sha256: sha, move: sha.slice(0, 12), rise_mm: riseOf(f),
             blend: j.blend, side: j.side || 0, gap: j.gap || 0, approach: j.approach || 0,
             boundsDeclared: B.bounds, boundViolations: B.violations,
             hasEvent: !!j.event, settleTicks: settle, floorSpawn, kind });
}
// duplicate vectors
const byHash = {};
for (const r of inv) (byHash[r.sha256] ||= []).push(r.file);
const dupes = Object.entries(byHash).filter(([, v]) => v.length > 1)
  .map(([h, v]) => ({ sha256: h, move: h.slice(0, 12), files: v }));
log(`   ${inv.length} files, ${Object.keys(byHash).length} DISTINCT vectors`);
for (const d of dupes) log(`   ONE VECTOR, ${d.files.length} LABELS  ${d.move}  ${d.files.join('  ')}`);
const viol = inv.filter(r => r.boundViolations.length);
log(`   bound violations: ${viol.length} file(s)`);
for (const v of viol) log(`      ${v.file}  ${v.boundViolations.map(b => `${b.param}=${b.value} outside [${b.lo},${b.hi}]`).join('; ')}`);
const notClimb = inv.filter(r => !r.floorSpawn || r.settleTicks !== 25);
log(`   NOT-A-CLIMB files (did not start standing on the floor): ${notClimb.length}`);
for (const n of notClimb) log(`      ${n.file.padEnd(34)} ${n.kind}`);
OUT.phaseHBS = { files: inv, distinctVectors: Object.keys(byHash).length, duplicateGroups: dupes,
                 boundViolations: viol.map(v => ({ file: v.file, violations: v.boundViolations })),
                 notAClimb: notClimb.map(n => ({ file: n.file, kind: n.kind })) };

// ==================================================================== PHASE G
log('');
log('PHASE G — every claimed clear + every round-4 best, re-scored from disk on the 14-cell grid');
log('   file                                 rise  kCore kCoreStable kExt kExtStable  objective objCore  objR3  maxTq  minPen  maxDY  upTail(min/mean)  start');
const CLAIMS = [
  // --- round-3 / round-2 claimed clears
  ['best_r3_vault_40mm.json', 0.040], ['best_r3_vault_50mm.json', 0.050],
  ['best_r3_vault_60mm.json', 0.060], ['best_r3_vault_70mm.json', 0.070],
  ['best_r3_vault_80mm.json', 0.080],
  ['best_r2_vault_40mm.json', 0.040], ['best_r2_vault_60mm.json', 0.060],
  ['best_r2_vault_90mm.json', 0.090],
  ['best_r3_landvault_80mm.json', 0.080], ['best_r3_landvault_90mm.json', 0.090],
  // --- round-3 family C, expected out of bounds; scored anyway, for the record
  ['best_r3_cornerclimb_120mm.json', 0.120], ['best_r3_cornerclimb2_120mm.json', 0.120],
  ['best_r3_cornerclimb_180mm.json', 0.180],
  // --- round 4
  ['best_r4_famA_60mm.json', 0.060],
  ['best_r4_famB_beat1_80mm.json', 0.080], ['best_r4_famB_beat1_90mm.json', 0.090],
  ['best_r4_famB_beat1_120mm.json', 0.120],
  ['best_r4_famB_beat2_80mm.json', 0.080], ['best_r4_famB_beat2_90mm.json', 0.090],
  ['best_r4_famB_beat2_120mm.json', 0.120],
  ['best_r4_famB_concat_80mm.json', 0.080], ['best_r4_famB_concat_90mm.json', 0.090],
  ['best_r4_famB_concat_120mm.json', 0.120],
].filter(([f]) => fs.existsSync(P + f));

const G = {};
for (const [f, h] of CLAIMS) {
  const meta = inv.find(r => r.file === f);
  const badBounds = meta && meta.boundViolations.length > 0;
  const g = await scoreRobust(P + f, { rise: h, skipBounds: true });   // score it, then judge it
  const row = {
    file: f, rise_mm: h * 1000, sha256: g.sha256, move: g.move,
    boundViolations: meta ? meta.boundViolations : [],
    admissible: !badBounds && meta && meta.floorSpawn && meta.settleTicks === 25,
    startKind: meta ? meta.kind : '?',
    kCore: g.kCore, kCoreStable: g.kCoreStable, kExt: g.kExt, kExtStable: g.kExtStable,
    objective: +g.objective.toFixed(4), objectiveCore: +g.objectiveCore.toFixed(4),
    objectiveR3: +g.objectiveR3.toFixed(4), meanReward: +g.meanReward.toFixed(4),
    maxTq: +g.agg.maxTq.toFixed(4), tqOverCeiling: g.agg.maxTq > FORCERANGE + 1e-9,
    tqSaturated: g.agg.maxTq >= FORCERANGE - 1e-4,
    minPenetrationAtScore_mm: +g.agg.minPenetrationAtScore_mm.toFixed(2),
    maxAbsDY_mm: +g.agg.maxAbsDY_mm.toFixed(1), lateralEscapeCells: g.agg.lateralEscapeCells,
    minUprightTailTicks: g.agg.minUprightTailTicks,
    meanUprightTailTicks: +g.agg.meanUprightTailTicks.toFixed(1),
    feetOnTreadMax: g.agg.feetOnTreadMax, maxZ_mm: +(g.agg.maxZ * 1000).toFixed(1),
    meanAbove_mm: +g.agg.meanAbove_mm.toFixed(1), meanX_mm: +g.agg.meanX_mm.toFixed(1),
    verdicts: g.verdicts,
    cellsXZ: g.cells.map(c => ({ rise_mm: c.cell.rise_mm, drop: c.cell.drop, fmul: c.cell.fmul,
                                 x: c.scored.x, z: c.scored.z })),
  };
  G[f] = row;
  const flag = badBounds ? ' OUT-OF-BOUNDS' : (row.admissible ? '' : ' NOT-A-CLIMB');
  log(`   ${f.padEnd(36)} ${String(row.rise_mm).padStart(4)}  ${row.kCore}/9   ${row.kCoreStable}/9      ${row.kExt}/14  ${row.kExtStable}/14     ${row.objective.toFixed(3).padStart(8)} ${row.objectiveCore.toFixed(3).padStart(8)} ${row.objectiveR3.toFixed(3).padStart(7)}  ${row.maxTq.toFixed(4)} ${String(row.minPenetrationAtScore_mm).padStart(7)} ${String(row.maxAbsDY_mm).padStart(6)}  ${String(row.minUprightTailTicks).padStart(2)}/${row.meanUprightTailTicks}${flag}`);
}
OUT.phaseG = G;
log(`   [${el()}]`);

// ==================================================================== PHASE D
log('');
log('PHASE D — family A\'s "same behaviour, second hash" claim, cell by cell');
const A60 = G['best_r4_famA_60mm.json'], R60 = G['best_r3_vault_60mm.json'];
if (A60 && R60) {
  let mdx = 0, mdz = 0;
  for (let i = 0; i < A60.cellsXZ.length; i++) {
    mdx = Math.max(mdx, Math.abs(A60.cellsXZ[i].x - R60.cellsXZ[i].x));
    mdz = Math.max(mdz, Math.abs(A60.cellsXZ[i].z - R60.cellsXZ[i].z));
  }
  const same = mdx === 0 && mdz === 0 && A60.kExt === R60.kExt
    && JSON.stringify(A60.verdicts.map(v => v.honest)) === JSON.stringify(R60.verdicts.map(v => v.honest));
  log(`   best_r4_famA_60mm (${A60.move}) vs best_r3_vault_60mm (${R60.move})`);
  log(`   max |dx| over 14 cells = ${(mdx * 1000).toFixed(6)} mm   max |dz| = ${(mdz * 1000).toFixed(6)} mm   identical honest pattern = ${same}`);
  log(`   -> ${same ? 'ONE MOVE, TWO HASHES. It is not a new behaviour and is not counted as one.' : 'DIFFERENT BEHAVIOUR — count as a distinct move.'}`);
  OUT.phaseD = { a: A60.move, b: R60.move, maxDx_mm: mdx * 1000, maxDz_mm: mdz * 1000, sameBehaviour: same };
}
// round-3's own three-label vector
const v60 = intentHashOfFile(P + 'best_r3_vault_60mm.json');
const v70 = intentHashOfFile(P + 'best_r3_vault_70mm.json');
const v80 = intentHashOfFile(P + 'best_r3_vault_80mm.json');
log(`   round-3 relabelling, re-checked: vault_60=${v60.slice(0, 8)} vault_70=${v70.slice(0, 8)} vault_80=${v80.slice(0, 8)}  -> ${v60 === v70 && v70 === v80 ? 'ONE vector under THREE labels' : 'distinct'}`);
OUT.phaseD_r3relabel = { v60, v70, v80, oneVector: v60 === v70 && v70 === v80 };

// ==================================================================== PHASE C
log('');
log('PHASE C — the controls, on the same plant');
const CTRL = [['ctrl_do_nothing.json', 0.040], ['ctrl_do_nothing.json', 0.060],
              ['ctrl_do_nothing.json', 0.090],
              ['r4_ctrl_on_tread_40mm.json', 0.040], ['r4_ctrl_on_tread_50mm.json', 0.050],
              ['r4_ctrl_on_tread_60mm.json', 0.060], ['r4_ctrl_on_tread_80mm.json', 0.080],
              ['r4_ctrl_on_tread_90mm.json', 0.090]].filter(([f]) => fs.existsSync(P + f));
const ctrlRows = [];
for (const [f, h] of CTRL) {
  const g = await scoreRobust(P + f, { rise: h, skipBounds: true });
  const meta = inv.find(r => r.file === f);
  const row = { file: f, rise_mm: h * 1000, move: g.move, startKind: meta ? meta.kind : '?',
                kCore: g.kCore, kCoreStable: g.kCoreStable, kExt: g.kExt, kExtStable: g.kExtStable,
                objective: +g.objective.toFixed(4), objectiveR3: +g.objectiveR3.toFixed(4),
                minPen_mm: +g.agg.minPenetrationAtScore_mm.toFixed(2),
                minUpTail: g.agg.minUprightTailTicks, maxTq: +g.agg.maxTq.toFixed(4) };
  ctrlRows.push(row);
  log(`   ${f.padEnd(30)} @${String(row.rise_mm).padStart(3)}mm  kCore ${row.kCore}/9  kCoreStable ${row.kCoreStable}/9  kExt ${row.kExt}/14  obj ${row.objective.toFixed(3)}  objR3 ${row.objectiveR3.toFixed(3)}  pen ${row.minPen_mm}mm  upTail>=${row.minUpTail}  [${row.startKind}]`);
}
const dn = ctrlRows.filter(r => r.file === 'ctrl_do_nothing.json');
const pd = ctrlRows.filter(r => r.file.startsWith('r4_ctrl_on_tread'));
OUT.phaseC = { rows: ctrlRows,
  doNothingAlwaysFails: dn.every(r => r.kCore === 0 && r.kExt === 0),
  placedDuckAlwaysPasses: pd.every(r => r.kCore === 9) };
log(`   do-nothing fails everywhere: ${OUT.phaseC.doNothingAlwaysFails}   placed duck clears 9/9 core everywhere: ${OUT.phaseC.placedDuckAlwaysPasses}`);

// ==================================================================== SUMMARY
log('');
log('================================================================');
log('THE LADDER — best ADMISSIBLE (floor-spawned, in bounds) DISTINCT move at each rise');
const admissible = Object.values(G).filter(r => r.admissible);
const byRise = {};
for (const r of admissible) (byRise[r.rise_mm] ||= []).push(r);
const ladder = [];
for (const rise of Object.keys(byRise).map(Number).sort((a, b) => a - b)) {
  const rows = byRise[rise].slice().sort((a, b) => b.kCore - a.kCore || b.objectiveCore - a.objectiveCore);
  const b = rows[0];
  ladder.push({ rise_mm: rise, file: b.file, move: b.move, kCore: b.kCore, kCoreStable: b.kCoreStable,
                kExt: b.kExt, kExtStable: b.kExtStable, objective: b.objective,
                objectiveCore: b.objectiveCore, objectiveR3: b.objectiveR3,
                alsoScoredFiles: rows.map(r => r.file) });
  log(`   ${String(rise).padStart(4)} mm  ${b.file.padEnd(34)} ${b.move}  kCore ${b.kCore}/9  stable ${b.kCoreStable}/9  kExt ${b.kExt}/14  obj ${b.objective.toFixed(3)}`);
}
OUT.ladder = ladder;
OUT.torque = { ceiling_Nm: FORCERANGE,
  anyRowOverCeiling: Object.values(G).some(r => r.tqOverCeiling) || ctrlRows.some(r => r.maxTq > FORCERANGE + 1e-9),
  saturatedFiles: Object.values(G).filter(r => r.tqSaturated).map(r => r.file) };
log(`   torque: any row above ${FORCERANGE} N.m? ${OUT.torque.anyRowOverCeiling}   files that SATURATE it: ${OUT.torque.saturatedFiles.length}/${Object.keys(G).length}`);
log(`   total wall ${el()}`);

fs.writeFileSync(P + 'r4_judge-results.json', JSON.stringify(OUT, null, 2));
fs.writeFileSync(P + 'audit_r4_judge.log', LOG.join('\n') + '\n');
log('wrote climb/r4_judge-results.json and climb/audit_r4_judge.log');
