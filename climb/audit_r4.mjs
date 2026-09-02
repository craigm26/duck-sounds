// ROUND 4 — CLOSE THE FOUR HOLES, THEN RE-BASELINE.
//
// The round-3 judge named four holes in the instrument. This file is the proof
// that they are closed and the re-baseline of every round-3 claim under the
// closed instrument.
//
//   HOLE 1  rig3.criteria() applied the lateral gate only at the scored
//           snapshot, while rig3.reward() applied it over the whole episode.
//           A move could swing off the 340 mm flight and come back.
//           CLOSED: snapshot() now carries maxAbsDY and criteria() gates on it.
//   HOLE 2  the shared objective had no "upright for the last N ticks" term,
//           so a candidate that reached the tread and toppled could outrank one
//           that stood. CLOSED: uprightTailTicks is a first-class field, a
//           clear earns its +4 only at >= 45 of 50 tail ticks upright, and the
//           mean upright-tail fraction is worth a further 4.
//   HOLE 3  duck-into-step penetration at the scored instant was not a
//           first-class field. CLOSED: penetrationAtScore (+ the geom pair) is
//           on every scored row of rig3 AND robust.
//   HOLE 4  one vector was published under three rise labels. CLOSED: robust
//           hashes the intent (sha256 over keyframes, blend, gap, side,
//           approach, spawn, isolate, stepCount) into every result row.
//   PLUS    family C's files breached the declared side bound [-0.02, 0.09].
//           CLOSED: robust.checkBounds() refuses to score an out-of-bounds file
//           and says so on stdout and stderr.
//
// PHASE P is the parity proof: climb/rig3_pre_r4.mjs is a byte copy of rig3.mjs
// as it stood before this round; every round-2 and round-3 best is scored
// through BOTH at three rise offsets and three tails and compared at full float
// digits on every field they share. Everything must match except where the
// whole-episode gate now bites, and every such row is listed.
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/audit_r4.mjs
import fs from 'node:fs';
import { scoreSaved as newScore, criteria as newCriteria, LATERAL, DUCKG } from '../climb/rig3.mjs';
import { scoreSaved as oldScore } from '../climb/rig3_pre_r4.mjs';
import { scoreRobust, intentHashOfFile, checkBounds, DHS, PLANTS, EXT_DHS, EXT_PLANT,
         CLEAR_BONUS, UPRIGHT_BONUS, UPRIGHT_TAIL_MIN, DECLARED_BOUNDS, STAIR_Y, saveIntent } from '../climb/robust.mjs';

const P = '../climb/';
const LOG = [];
const log = s => { console.log(s); LOG.push(s); };
const mm = v => (v === null || v === undefined) ? null : +(v * 1000).toFixed(4);
const f1 = v => (v === null || v === undefined) ? '   n/a' : v.toFixed(1);
const OUT = { generated: new Date().toISOString(), plant: 'scene.mjb', policy: 'BEST_alpha_stand.onnx',
              lateralGate_mm: LATERAL * 1000, duckGeoms: DUCKG.length,
              declaredBounds: DECLARED_BOUNDS, upright_tail_min: UPRIGHT_TAIL_MIN,
              clearBonus: CLEAR_BONUS, uprightBonus: UPRIGHT_BONUS };
const t00 = Date.now();

// ============================================================ PHASE P — parity
log('=== PHASE P — new rig3 vs climb/rig3_pre_r4.mjs (byte copy of the old one) ===');
log(`    duck collidable geoms ${DUCKG.length} | lateral gate ${(LATERAL * 1000).toFixed(0)} mm`);

const R2 = fs.readdirSync(P).filter(f => /^best_r2_.*\.json$/.test(f)).sort();
const R3 = fs.readdirSync(P).filter(f => /^best_r3_.*\.json$/.test(f)).sort();
const R1 = fs.readdirSync(P).filter(f => /^best_[012]_\d+mm\.json$/.test(f)).sort();
const riseOf = f => { const m = f.match(/_(\d+)mm/); return m ? parseInt(m[1], 10) / 1000 : 0.060; };

// every numeric / boolean field the two harnesses share
const SCALARS = ['x0', 'ctrlJump', 'maxX', 'maxZ', 'maxAbsDY', 'feetOnTreadMax', 'feetUpRawMax',
  'maxTreadSag_mm', 'maxTreadDriftX_mm', 'minStepGap_mm', 'headFrac', 'riserFrac', 'upFrac',
  'satFrac', 'z0Settle', 'bothFrac', 'maxGainBoth', 'sustainFrac', 'liftIntegral', 'reward'];
const SNAPF = ['x', 'y', 'z', 'dy', 'above', 'up', 'feetUpRaw', 'feetUpLat', 'feetOnTread'];
const CRITF = ['orig', 'lat', 'honest', 'honest60', 'lateral'];

function comparePair(a, b) {
  const diffs = [];
  for (const k of SCALARS) if (a[k] !== b[k]) diffs.push({ field: k, old: a[k], new: b[k] });
  for (const k of SNAPF) if (a.scored[k] !== b.scored[k]) diffs.push({ field: 'scored.' + k, old: a.scored[k], new: b.scored[k] });
  for (const k of ['x', 'y', 'z']) {
    if (a.scored.lfoot[k] !== b.scored.lfoot[k]) diffs.push({ field: 'lfoot.' + k, old: a.scored.lfoot[k], new: b.scored.lfoot[k] });
    if (a.scored.rfoot[k] !== b.scored.rfoot[k]) diffs.push({ field: 'rfoot.' + k, old: a.scored.rfoot[k], new: b.scored.rfoot[k] });
  }
  const critDiffs = [];
  for (const k of CRITF) if (a.crit[k] !== b.crit[k]) critDiffs.push({ field: 'crit.' + k, old: a.crit[k], new: b.crit[k] });
  return { diffs, critDiffs };
}

async function parityFile(f, rise, offsets, tails) {
  const rows = [];
  for (const d of offsets) for (const t of tails) {
    const A = await oldScore(P + f, { rise, tail: t, gapOffset: d });
    const B = await newScore(P + f, { rise, tail: t, gapOffset: d });
    const { diffs, critDiffs } = comparePair(A, B);
    rows.push({ file: f, rise_mm: Math.round(rise * 1000), gapOffset_mm: Math.round(d * 1000), tail: t,
      physicsExact: diffs.length === 0, critExact: critDiffs.length === 0,
      diffs, critDiffs, maxAbsDY_mm: mm(B.maxAbsDY),
      oldHonest: A.crit.honest, newHonest: B.crit.honest,
      penetrationAtScore_mm: mm(B.penetrationAtScore), penetrationPair: B.penetrationPair,
      uprightTailTicks: B.uprightTailTicks, tailTicks: B.tailTicks });
  }
  return rows;
}

OUT.parity = [];
const OFF = [-0.010, 0, 0.010], TAILS = ['policy', 'hold', 'none'];
for (const f of [...R2, ...R3]) OUT.parity.push(...await parityFile(f, riseOf(f), OFF, TAILS));
// round-1 bests too: the lateral confound (excursions up to 426 mm) lives here,
// so this is where the whole-episode gate has any chance of biting.
for (const f of R1) OUT.parity.push(...await parityFile(f, riseOf(f), [0], TAILS));

const physFail = OUT.parity.filter(r => !r.physicsExact);
const critMoved = OUT.parity.filter(r => !r.critExact);
OUT.parityRows = OUT.parity.length;
OUT.parityPhysicsExact = physFail.length === 0;
log(`    ${OUT.parity.length} scored rows through both harnesses ` +
    `(${R2.length} round-2 + ${R3.length} round-3 bests x 3 offsets x 3 tails, ${R1.length} round-1 bests x 1 offset x 3 tails)`);
log(`    physics/geometry/reward EXACT on ${OUT.parity.length - physFail.length}/${OUT.parity.length} rows ` +
    `(${OUT.parityPhysicsExact ? 'PARITY PASS' : 'PARITY FAIL'})`);
if (physFail.length) for (const r of physFail.slice(0, 20)) log(`      MISMATCH ${r.file} off=${r.gapOffset_mm} tail=${r.tail}: ${JSON.stringify(r.diffs.slice(0, 4))}`);
log(`    rows whose VERDICT the whole-episode lateral gate changes: ${critMoved.length}`);
if (critMoved.length === 0) {
  log('      none. Every round-1/2/3 best that scored `honest` stayed inside the 340 mm');
  log('      flight for the whole episode, so closing hole 1 costs no published claim.');
} else {
  log('      file                          rise off tail    maxAbsDY_mm  changed');
  for (const r of critMoved) log(`      ${r.file.padEnd(30)} ${String(r.rise_mm).padStart(4)} ${String(r.gapOffset_mm).padStart(3)} ${r.tail.padEnd(6)} ` +
    `${f1(r.maxAbsDY_mm).padStart(11)}  ${r.critDiffs.map(d => `${d.field} ${d.old}->${d.new}`).join(', ')}`);
}
const worstDY = Math.max(...OUT.parity.map(r => r.maxAbsDY_mm));
OUT.worstExcursion_mm = worstDY;
OUT.lateralEscapeRows = OUT.parity.filter(r => r.maxAbsDY_mm > LATERAL * 1000).length;
log(`    worst whole-episode excursion over all ${OUT.parity.length} rows: ${worstDY.toFixed(1)} mm ` +
    `(gate ${(LATERAL * 1000).toFixed(0)} mm); rows that leave the flight at some tick: ${OUT.lateralEscapeRows}`);
log('');

// ------------------------------------------------- the three new fields exist
log('=== PHASE F — the three new first-class fields, on every scored row ===');
const haveP = OUT.parity.filter(r => r.penetrationAtScore_mm !== null).length;
const haveU = OUT.parity.filter(r => r.uprightTailTicks !== undefined && r.tailTicks === 50).length;
OUT.fieldCoverage = { rows: OUT.parity.length, penetrationAtScore: haveP, uprightTailTicks: haveU,
  maxAbsDY: OUT.parity.filter(r => r.maxAbsDY_mm !== null).length };
log(`    penetrationAtScore present on ${haveP}/${OUT.parity.length} rows; uprightTailTicks (of 50) on ${haveU}/${OUT.parity.length}; maxAbsDY on ${OUT.fieldCoverage.maxAbsDY}/${OUT.parity.length}`);
const penVals = OUT.parity.map(r => r.penetrationAtScore_mm).filter(v => v !== null);
log(`    penetrationAtScore range over those rows: ${Math.min(...penVals).toFixed(2)} mm to ${Math.max(...penVals).toFixed(2)} mm`);
const upVals = OUT.parity.map(r => r.uprightTailTicks);
log(`    uprightTailTicks range: ${Math.min(...upVals)} to ${Math.max(...upVals)} of 50`);
log('');

// ============================================== PHASE H — hashes and bounds
log('=== PHASE H — hole 4: intent hashes. One vector, one move, however it is labelled. ===');
OUT.hashes = [];
const byHash = new Map();
for (const f of [...R2, ...R3]) {
  const h = intentHashOfFile(P + f);
  OUT.hashes.push({ file: f, sha256: h });
  if (!byHash.has(h)) byHash.set(h, []);
  byHash.get(h).push(f);
}
const dupes = [...byHash.entries()].filter(([, v]) => v.length > 1);
OUT.distinctMoves = byHash.size;
OUT.duplicatePublications = dupes.map(([h, v]) => ({ sha256: h, files: v }));
log(`    ${OUT.hashes.length} published round-2/round-3 files -> ${byHash.size} DISTINCT moves`);
for (const [h, v] of dupes) log(`      ONE VECTOR, ${v.length} LABELS  sha256 ${h.slice(0, 16)}...  ${v.join(', ')}`);
log('');

log('=== PHASE B — bounds enforced AT SCORING TIME, not declared in a comment ===');
log(`    declared bounds: ${JSON.stringify(DECLARED_BOUNDS)}`);
OUT.boundsCheck = [];
for (const f of [...R2, ...R3, 'ctrl_do_nothing.json']) {
  const j = JSON.parse(fs.readFileSync(P + f, 'utf8'));
  const c = checkBounds(j);
  OUT.boundsCheck.push({ file: f, blend: j.blend, side: j.side || 0, violations: c.violations });
  if (c.violations.length) log(`    OUT OF BOUNDS  ${f.padEnd(34)} ${c.violations.map(v => `${v.param}=${v.value} not in [${v.lo},${v.hi}]`).join('; ')}`);
}
const nBad = OUT.boundsCheck.filter(r => r.violations.length).length;
log(`    ${nBad} of ${OUT.boundsCheck.length} published files are outside the declared box.`);
// prove the refusal actually fires
const badFile = OUT.boundsCheck.find(r => r.violations.length);
if (badFile) {
  log(`    scoring ${badFile.file} through scoreRobust — expect a refusal, not a number:`);
  const bad = await scoreRobust(P + badFile.file, { rise: riseOf(badFile.file) });
  OUT.boundsRefusal = { file: badFile.file, invalid: bad.invalid, objective: bad.objective,
                        cellsRun: bad.cells.length, violations: bad.boundViolations };
  log(`    -> invalid=${bad.invalid} objective=${bad.objective} cells simulated=${bad.cells.length}`);
}
log('');

// ============================================ PHASE G — the extended grid
const N_EXT = DHS.length * PLANTS.length + EXT_DHS.length + DHS.length;
log('=== PHASE G — THE GRID every family and the judge now use ===');
log(`    core 9 : rise {h-10, h, h+10} x plant {(0.120, x1.0), (0.130, x0.7), (0.125, x1.3)}   [round 3, unchanged]`);
log(`    +2     : rise {h-5, h+5} at the nominal plant (0.120, x1.0)`);
log(`    +3     : rise {h-10, h, h+10} at the slippery plant (drop ${EXT_PLANT.drop}, friction x${EXT_PLANT.fmul})`);
log(`    N = ${N_EXT}. objective = meanReward(N) + ${CLEAR_BONUS}*kStable(N) + ${UPRIGHT_BONUS}*meanUprightCredit(N)`);
log(`    a clear counts toward kStable only at >= ${UPRIGHT_TAIL_MIN} of 50 tail ticks upright.`);
log(`    uprightCredit is uprightTailFrac in cells where the duck reached the flight (trunk past x=0.12`);
log(`    at some tick, or a foot on a tread at some tick) and 0 otherwise, so standing still pays nothing.`);
log('');

// ---------------------------------------------------------------- controls
// the placed-duck recipe from climb/ctrl_on_tread_60mm.json: x 0.25, y STAIR_Y,
// z = rise + 0.12. Written to r4_-prefixed files; no round-2/3 file is touched.
const CLAIM_RISES = [40, 50, 60, 80, 90];
const placed = {};
for (const rm of CLAIM_RISES) {
  const src = JSON.parse(fs.readFileSync(P + 'ctrl_on_tread_60mm.json', 'utf8'));
  const p = P + `r4_ctrl_on_tread_${rm}mm.json`;
  saveIntent({ name: `r4_control_on_tread_${rm}mm`, keyframes: src.keyframes, blend: 1, gap: 0.05,
    side: 0, approach: 0, spawn: { x: 0.25, y: STAIR_Y, z: rm / 1000 + 0.12 },
    note: `PLACED-DUCK CONTROL, rise ${rm} mm. Recipe copied from climb/ctrl_on_tread_60mm.json `
        + `(x 0.25 mid-tread, y STAIR_Y, z rise+0.12). A criterion this row fails cannot express success.` }, p);
  placed[rm] = p;
}

// ======================================== PHASE R — the re-baseline
const CLAIMS = [
  ['best_r3_vault_40mm.json', 0.040, 'A beak-strut vault r3'],
  ['best_r3_vault_50mm.json', 0.050, 'A beak-strut vault r3'],
  ['best_r3_vault_60mm.json', 0.060, 'A beak-strut vault r3'],
  ['best_r2_vault_40mm.json', 0.040, 'C beak-strut vault r2'],
  ['best_r2_vault_60mm.json', 0.060, 'C beak-strut vault r2'],
  ['best_r3_landvault_80mm.json', 0.080, 'B land-the-vault'],
  ['best_r3_landvault_90mm.json', 0.090, 'B land-the-vault'],
];
const CONTROLS = [];
for (const rm of CLAIM_RISES) CONTROLS.push(['ctrl_do_nothing.json', rm / 1000, `do-nothing @${rm}`]);
for (const rm of CLAIM_RISES) CONTROLS.push([`r4_ctrl_on_tread_${rm}mm.json`, rm / 1000, `placed duck @${rm}`]);

log('=== PHASE R — the seven round-3 claims and the controls, re-scored ===');
OUT.rebaseline = [];
async function rescore(file, rise, label) {
  const g = await scoreRobust(P + file, { rise });
  const core = g.cells.filter(c => c.cell.tier === 'core');
  const row = {
    label, file, rise_mm: Math.round(rise * 1000), sha256: g.sha256, move: g.move,
    kCore: g.kCore, nCore: g.nCore, kCoreStable: g.kCoreStable,
    kExt: g.kExt, nExt: g.nExt, kExtStable: g.kExtStable,
    meanRewardCore: +g.meanRewardCore.toFixed(4), meanRewardExt: +g.meanReward.toFixed(4),
    objectiveR3: +g.objectiveR3.toFixed(4), objectiveCore: +g.objectiveCore.toFixed(4),
    objective: +g.objective.toFixed(4),
    uprightTailFracExt: +g.uprightTailFrac.toFixed(4),
    uprightTailFracRaw: +g.uprightTailFracRaw.toFixed(4),
    reachedFlightCells: g.reachedFlightCells,
    objectiveUngatedUpright: +g.objectiveUngatedUpright.toFixed(4),
    minUprightTailTicks: g.agg.minUprightTailTicks,
    meanUprightTailTicks: +g.agg.meanUprightTailTicks.toFixed(1),
    minPenetrationAtScore_mm: +g.agg.minPenetrationAtScore_mm.toFixed(2),
    lateralEscapeCells: g.agg.lateralEscapeCells,
    maxAbsDY_mm: +g.agg.maxAbsDY_mm.toFixed(1),
    clearsCore: core.filter(c => c.crit.honest).map(c => `${c.cell.rise_mm}/${c.cell.drop}/x${c.cell.fmul}`),
    clearsExtOnly: g.cells.filter(c => c.cell.tier === 'ext' && c.crit.honest).map(c => `${c.cell.rise_mm}/${c.cell.drop}/x${c.cell.fmul}`),
    unstableClears: g.cells.filter(c => c.crit.honest && !c.stableClear)
      .map(c => `${c.cell.rise_mm}/${c.cell.drop}/x${c.cell.fmul} up${c.uprightTailTicks}/50`),
    verdicts: g.verdicts,
  };
  OUT.rebaseline.push(row);
  return row;
}
for (const [f, r, l] of CLAIMS) { const row = await rescore(f, r, l); log(`    scored ${f.padEnd(30)} kCore ${row.kCore}/9  kExt ${row.kExt}/${row.nExt}`); }
for (const [f, r, l] of CONTROLS) { const row = await rescore(f, r, l); log(`    scored ${l.padEnd(30)} kCore ${row.kCore}/9  kExt ${row.kExt}/${row.nExt}`); }

// the round-3 numbers, from the files themselves, for the delta column
const R3CLAIM = { 'best_r3_vault_40mm.json': 2, 'best_r3_vault_50mm.json': 2, 'best_r3_vault_60mm.json': 4,
                  'best_r2_vault_40mm.json': null, 'best_r2_vault_60mm.json': null,
                  'best_r3_landvault_80mm.json': null, 'best_r3_landvault_90mm.json': null };
// round-3's own audit measured these; fill from audit_r3-results.json where present
try {
  const a3 = JSON.parse(fs.readFileSync(P + 'audit_r3-results.json', 'utf8'));
  for (const g of (a3.grids || [])) if (g.file in R3CLAIM) R3CLAIM[g.file] = g.k;
} catch (e) { /* leave nulls */ }
OUT.round3_k = R3CLAIM;

log('');
log('  === THE RE-BASELINE TABLE ===');
log('  move (sha256[0:8])  file                          rise  r3 k/9  k/9(core)  kStable/9  k/14(ext)  kStable/14   objR3    objCore   objective   upTail(min/mean of 50)  penAtScore_mm  maxDY_mm');
const fmt = (row) => {
  const r3k = R3CLAIM[row.file];
  return `  ${row.sha256.slice(0, 8)}            ${row.file.padEnd(28)} ${String(row.rise_mm).padStart(4)}  ` +
    `${(r3k === null || r3k === undefined ? '  -' : String(r3k) + '/9').padStart(6)}  ` +
    `${(row.kCore + '/9').padStart(9)}  ${(row.kCoreStable + '/9').padStart(9)}  ` +
    `${(row.kExt + '/' + row.nExt).padStart(9)}  ${(row.kExtStable + '/' + row.nExt).padStart(10)}  ` +
    `${row.objectiveR3.toFixed(3).padStart(7)}  ${row.objectiveCore.toFixed(3).padStart(8)}  ${row.objective.toFixed(3).padStart(9)}   ` +
    `${(row.minUprightTailTicks + '/' + row.meanUprightTailTicks).padStart(12)}         ` +
    `${row.minPenetrationAtScore_mm.toFixed(2).padStart(9)}  ${row.maxAbsDY_mm.toFixed(1).padStart(8)}`;
};
for (const row of OUT.rebaseline.filter(r => CLAIMS.some(c => c[0] === r.file && c[1] * 1000 === r.rise_mm))) log(fmt(row));
log('  --- controls ---');
for (const row of OUT.rebaseline.filter(r => /ctrl_/.test(r.file))) log(fmt(row));

log('');
log('  per-claim detail: which cells clear, and which clears topple inside the tail');
for (const row of OUT.rebaseline) {
  if (/ctrl_/.test(row.file)) continue;
  log(`   ${row.file} @${row.rise_mm} mm  [${row.label}]  move ${row.move}`);
  log(`      core clears  : ${row.clearsCore.length ? row.clearsCore.join('  ') : 'none'}`);
  log(`      ext-only     : ${row.clearsExtOnly.length ? row.clearsExtOnly.join('  ') : 'none'}`);
  log(`      toppling     : ${row.unstableClears.length ? row.unstableClears.join('  ') : 'none (every clear held >= ' + UPRIGHT_TAIL_MIN + '/50)'}`);
  log(`      upright term : credited ${row.uprightTailFracExt} (raw ${row.uprightTailFracRaw}); reached the flight in ${row.reachedFlightCells}/${row.nExt} cells; ungated objective would be ${row.objectiveUngatedUpright}`);
}

// ---------------------------------------------------------------- verdict
const claimRows = OUT.rebaseline.filter(r => CLAIMS.some(c => c[0] === r.file && c[1] * 1000 === r.rise_mm));
const dn = OUT.rebaseline.filter(r => /ctrl_do_nothing/.test(r.file));
const pl = OUT.rebaseline.filter(r => /r4_ctrl_on_tread/.test(r.file));
OUT.verdict = {
  parityRows: OUT.parity.length,
  parityPhysicsExact: OUT.parityPhysicsExact,
  verdictRowsMovedByWholeEpisodeGate: critMoved.length,
  distinctMovesAmongPublished: byHash.size,
  publishedFiles: OUT.hashes.length,
  filesOutOfDeclaredBounds: nBad,
  bestClaim_kCore: Math.max(...claimRows.map(r => r.kCore)),
  bestClaim_kExt: Math.max(...claimRows.map(r => r.kExt)),
  bestClaim_kExtStable: Math.max(...claimRows.map(r => r.kExtStable)),
  doNothing_kExt: dn.map(r => `${r.rise_mm}mm:${r.kExt}/${r.nExt}`),
  doNothing_objective: dn.map(r => `${r.rise_mm}mm:${r.objective}`),
  doNothing_objectiveIfUprightUngated: dn.map(r => `${r.rise_mm}mm:${r.objectiveUngatedUpright}`),
  doNothing_reachedFlightCells: dn.map(r => `${r.rise_mm}mm:${r.reachedFlightCells}/${r.nExt}`),
  placedDuck_kExt: pl.map(r => `${r.rise_mm}mm:${r.kExt}/${r.nExt}`),
  placedDuck_kExtStable: pl.map(r => `${r.rise_mm}mm:${r.kExtStable}/${r.nExt}`),
  claim_minPenetrationAtScore_mm: Math.min(...claimRows.map(r => r.minPenetrationAtScore_mm)),
  placed_minPenetrationAtScore_mm: Math.min(...pl.map(r => r.minPenetrationAtScore_mm)),
};
log('');
log('=== VERDICT ===');
log(`  parity: ${OUT.parity.length} rows through both harnesses, physics EXACT on all but ${physFail.length}; ` +
    `${critMoved.length} verdict rows moved by the whole-episode gate`);
log(`  published round-2/3 files ${OUT.hashes.length} -> distinct moves ${byHash.size}; out of declared bounds ${nBad}`);
log(`  best claim: kCore ${OUT.verdict.bestClaim_kCore}/9, kExt ${OUT.verdict.bestClaim_kExt}/${N_EXT}, kExtStable ${OUT.verdict.bestClaim_kExtStable}/${N_EXT}`);
log(`  do-nothing control: ${OUT.verdict.doNothing_kExt.join(' ')}`);
log(`    do-nothing objective ${OUT.verdict.doNothing_objective.join(' ')}`);
log(`    do-nothing objective IF the upright term were ungated: ${OUT.verdict.doNothing_objectiveIfUprightUngated.join(' ')}  <- why it is gated`);
log(`    do-nothing cells that reached the flight: ${OUT.verdict.doNothing_reachedFlightCells.join(' ')}`);
log(`  placed-duck control: ${OUT.verdict.placedDuck_kExt.join(' ')}  (stable: ${OUT.verdict.placedDuck_kExtStable.join(' ')})`);
log(`  penetration at the scored instant: claims worst ${OUT.verdict.claim_minPenetrationAtScore_mm.toFixed(2)} mm, ` +
    `placed duck worst ${OUT.verdict.placed_minPenetrationAtScore_mm.toFixed(2)} mm`);

OUT.elapsed_s = +((Date.now() - t00) / 1000).toFixed(1);
fs.writeFileSync(P + 'r4_audit-results.json', JSON.stringify(OUT, null, 1));
fs.writeFileSync(P + 'audit_r4.log', LOG.join('\n') + '\n');
log(`\nWROTE ${P}r4_audit-results.json and ${P}audit_r4.log  (${OUT.elapsed_s}s)`);
