#!/usr/bin/env node
// check_numbers.mjs — re-derive every number stated in README.md from the files
// in results/, and print PASS or FAIL for each one.
//
// This script reads NOTHING but results/ (and, when it happens to be sitting in
// the repository, ../sim/scene.mjb for the plant digest). It runs no simulation:
// every assertion is a value that was already written to a results file by the
// round that measured it. If a number in the card cannot be found here, the card
// is wrong.
//
//   cd challenge && node check_numbers.mjs
//
// Exit status 0 when every check passes, 1 otherwise.

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const R = f => JSON.parse(fs.readFileSync(path.join(HERE, 'results', f), 'utf8'));
const T = f => fs.readFileSync(path.join(HERE, 'results', f), 'utf8');

let pass = 0, fail = 0;
const rows = [];

/** Assert `got` deep-equals `want`; `src` names the results file it came from. */
function check(label, got, want, src) {
  const g = JSON.stringify(got), w = JSON.stringify(want);
  const ok = g === w;
  ok ? pass++ : fail++;
  rows.push({ ok, label, got: g, want: w, src });
}
function checkNear(label, got, want, tol, src) {
  const ok = typeof got === 'number' && Math.abs(got - want) <= tol;
  ok ? pass++ : fail++;
  rows.push({ ok, label, got: String(got), want: `${want} ±${tol}`, src });
}

// ---------------------------------------------------------------- the files
const r4j = R('r4_judge-results.json');
const r5j = R('r5_judge-results.json');
const r6j = R('r6_judge-results.json');
const r6c = R('r6_ceiling-results.json');
const r6s = R('r6_screen-results.json');
const r6lt = R('r6_limits_table-results.json');
const famB = R('r4_famB-results.json');
const corner = R('r3_cornerclimb-results.json');

const G6 = r6j.phaseG, G4 = r4j.phaseG;
const E5 = Object.fromEntries(r5j.phaseE.rows.map(r => [r.file, r]));
const ladder = Object.fromEntries(r4j.ladder.map(r => [r.rise_mm, r]));
const HBS6 = Object.fromEntries(r6j.phaseHBS.files.map(r => [r.file, r]));

// =================================================== 1. the constants
const C = r6j.constants;
check('forcerange 0.6405 N.m', C.forcerange_Nm, 0.6405, 'r6_judge-results.json constants');
check('honest height gate 95 mm', C.honestAboveGate_mm, 95, 'r6_judge-results.json constants');
check('lateral gate 170 mm (340 mm flight)', C.lateralGate_mm, 170, 'r6_judge-results.json constants');
check('riser line 120 mm', C.riserLine_mm, 120, 'r6_judge-results.json constants');
check('control tick 20 ms', C.controlTick_ms, 20, 'r6_judge-results.json constants');
check('tickHz 50', C.tickHz, 50, 'r6_judge-results.json constants');
check('stable = 45 of 50 tail ticks', C.uprightTailMin, 45, 'r6_judge-results.json constants');
check('soft-contact line -15 mm', C.softContactLine_mm, -15, 'r6_judge-results.json constants');
check('declared bounds blend [0.7, 2.4]', C.declaredBounds.blend, [0.7, 2.4], 'r6_judge-results.json constants');
check('declared bounds side [-0.02, 0.09]', C.declaredBounds.side, [-0.02, 0.09], 'r6_judge-results.json constants');
check('core grid rises {-10, 0, +10} mm', C.grid.core.dhs_mm, [-10, 0, 10], 'r6_judge-results.json constants.grid');
check('core grid plants', C.grid.core.plants,
      [{ drop: 0.12, fmul: 1 }, { drop: 0.13, fmul: 0.7 }, { drop: 0.125, fmul: 1.3 }],
      'r6_judge-results.json constants.grid');
check('extended rises {-5, +5} mm', C.grid.ext.dhs_mm, [-5, 5], 'r6_judge-results.json constants.grid');
check('extended plant (0.140, x0.5)', C.grid.ext.plant, { drop: 0.14, fmul: 0.5 }, 'r6_judge-results.json constants.grid');
check('grid n = 14', C.grid.n, 14, 'r6_judge-results.json constants.grid');

// =================================================== 2. the leaderboard
// rank 1 — best_r6_ceilvaultC_60mm.json, sha a56d459fb649
const c1 = G6['best_r6_ceilvaultC_60mm.json'];
check('rank 1 sha256 prefix', c1.sha256.slice(0, 12), 'a56d459fb649', 'r6_judge-results.json phaseG');
check('rank 1 kCoreStable 5 of 9', c1.kCoreStable, 5, 'r6_judge-results.json phaseG');
check('rank 1 kCore 5', c1.kCore, 5, 'r6_judge-results.json phaseG');
check('rank 1 kExt 5 / kExtStable 5', [c1.kExt, c1.kExtStable], [5, 5], 'r6_judge-results.json phaseG');
check('rank 1 ceilingCore 5 of 9', c1.ceilingCore, 5, 'r6_judge-results.json phaseG');
check('rank 1 admissible + floor spawn', [c1.admissible, c1.floorSpawn], [true, true], 'r6_judge-results.json phaseG');
check('rank 1 in declared bounds', c1.boundViolations, [], 'r6_judge-results.json phaseG');
check('rank 1 carries no servo and no event',
      [HBS6['best_r6_ceilvaultC_60mm.json'].hasServo, HBS6['best_r6_ceilvaultC_60mm.json'].hasEvent],
      [false, false], 'r6_judge-results.json phaseHBS');
check('rank 1 maxTq pinned at the ceiling', c1.maxTq, 0.6405, 'r6_judge-results.json phaseG');

// rank 2 — the beak-strut vault, sha 4b9110c448ec
const v60 = G6['best_r3_vault_60mm.json'];
check('rank 2 sha256 prefix', v60.sha256.slice(0, 12), '4b9110c448ec', 'r6_judge-results.json phaseG');
check('the round-3 vault: 4 of 9 stable at 60 mm (the record through round 5)', v60.kCoreStable, 4, 'r6_judge-results.json phaseG');
check('rank 2 kCore 4', v60.kCore, 4, 'r6_judge-results.json phaseG');
check('rank 2 kExt 4 / kExtStable 4', [v60.kExt, v60.kExtStable], [4, 4], 'r6_judge-results.json phaseG');
check('rank 2 ceilingCore 5 of 9', v60.ceilingCore, 5, 'r6_judge-results.json phaseG');
check('rank 2 maxTq pinned at the ceiling', v60.maxTq, 0.6405, 'r6_judge-results.json phaseG');
check('rank 2 also scored by round 4 as 4/9', [ladder[60].kCore, ladder[60].kCoreStable], [4, 4],
      'r4_judge-results.json ladder');

// one vector under three rise labels
check('60/70/80 mm are ONE vector', r4j.phaseD_r3relabel.oneVector, true, 'r4_judge-results.json phaseD_r3relabel');
check('60/70/80 mm share sha256 4b9110c448ec',
      [r4j.phaseD_r3relabel.v60.slice(0, 12), r4j.phaseD_r3relabel.v70.slice(0, 12), r4j.phaseD_r3relabel.v80.slice(0, 12)],
      ['4b9110c448ec', '4b9110c448ec', '4b9110c448ec'], 'r4_judge-results.json phaseD_r3relabel');
check('the same vector at 70 mm clears 2 of 9', [ladder[70].kCore, ladder[70].kCoreStable], [2, 2],
      'r4_judge-results.json ladder');
check('the same vector at 80 mm clears 1 of 9', [ladder[80].kCore, ladder[80].kCoreStable], [1, 1],
      'r4_judge-results.json ladder');
{
  // the 80 mm file's single clear is its 70 mm cell
  const v80 = G4['best_r3_vault_80mm.json'];
  const clears = (v80.verdicts || []).filter(v => v.honest).map(v => v.rise_mm);
  check("the 80 mm 'clear' is its 70 mm cell", clears, [70], 'r4_judge-results.json phaseG.verdicts');
}

// rank 3 — the round-4 event landing, behaviourally identical to rank 2
const famA = G6['best_r4_famA_60mm.json'];
check('rank 3 sha256 prefix', famA.sha256.slice(0, 12), '7b790070b010', 'r6_judge-results.json phaseG');
check('rank 3 kCore 4 / kCoreStable 4', [famA.kCore, famA.kCoreStable], [4, 4], 'r6_judge-results.json phaseG');
check('rank 3 ceilingCore 5', famA.ceilingCore, 5, 'r6_judge-results.json phaseG');
check('rank 3 is behaviourally identical to rank 2 (dx, dz = 0 mm)',
      [r4j.phaseD.maxDx_mm, r4j.phaseD.maxDz_mm, r4j.phaseD.sameBehaviour], [0, 0, true],
      'r4_judge-results.json phaseD');

// rank 4 / 7 — the other two round-6 ceiling launches
const cB = G6['best_r6_ceilvaultB_60mm.json'], cA = G6['best_r6_ceilvault_60mm.json'];
check('rank 4 sha / kCore 3 / stable 2 / ceiling 5',
      [cB.sha256.slice(0, 12), cB.kCore, cB.kCoreStable, cB.kExt, cB.kExtStable, cB.ceilingCore],
      ['29c97398fe13', 3, 2, 3, 2, 5], 'r6_judge-results.json phaseG');
check('rank 7 sha / kCore 1 / stable 1 / ceiling 5',
      [cA.sha256.slice(0, 12), cA.kCore, cA.kCoreStable, cA.kExt, cA.kExtStable, cA.ceilingCore],
      ['8c57838ee9d0', 1, 1, 1, 1, 5], 'r6_judge-results.json phaseG');

// ranks 5, 6, 8, 9 — the 40 and 50 mm rows and the round-2 vaults
check('50 mm: kCore 2, stable 2', [ladder[50].kCore, ladder[50].kCoreStable], [2, 2], 'r4_judge-results.json ladder');
check('50 mm ceilingCore 2', E5['best_r3_vault_50mm.json'].ceilingCore, 2, 'r5_judge-results.json phaseE');
check('50 mm sha 7904bf3363c5', ladder[50].move, '7904bf3363c5', 'r4_judge-results.json ladder');
check('40 mm: kCore 2, stable 1', [ladder[40].kCore, ladder[40].kCoreStable], [2, 1], 'r4_judge-results.json ladder');
check('40 mm ceilingCore 3', E5['best_r3_vault_40mm.json'].ceilingCore, 3, 'r5_judge-results.json phaseE');
check('40 mm sha dff01b0a1906', ladder[40].move, 'dff01b0a1906', 'r4_judge-results.json ladder');
check('r2 vault 60 mm: kCore 2, stable 1, ceiling 3',
      [G4['best_r2_vault_60mm.json'].kCore, G4['best_r2_vault_60mm.json'].kCoreStable,
       E5['best_r2_vault_60mm.json'].ceilingCore], [2, 1, 3],
      'r4_judge-results.json phaseG + r5_judge-results.json phaseE');
check('r2 vault 40 mm: kCore 1, stable 1, ceiling 4',
      [G4['best_r2_vault_40mm.json'].kCore, G4['best_r2_vault_40mm.json'].kCoreStable,
       E5['best_r2_vault_40mm.json'].ceilingCore], [1, 1, 4],
      'r4_judge-results.json phaseG + r5_judge-results.json phaseE');

// the zero rows
check('90 mm: 0 of 9 (best in corpus, sha 2524a35672b4)',
      [ladder[90].kCore, ladder[90].kCoreStable, ladder[90].move], [0, 0, '2524a35672b4'],
      'r4_judge-results.json ladder');
check('120 mm: 0 of 9 (best in corpus, sha 7c52acef4acf)',
      [ladder[120].kCore, ladder[120].kCoreStable, ladder[120].move], [0, 0, '7c52acef4acf'],
      'r4_judge-results.json ladder');
check('180 mm: 0 of 9 (best in corpus, sha 725674c1b517)',
      [ladder[180].kCore, ladder[180].kCoreStable, ladder[180].move], [0, 0, '725674c1b517'],
      'r4_judge-results.json ladder');

// the oracle rows
const sv = G6['best_r5_servo_60mm.json'], sl = G6['best_r5_servoland_kcore_60mm.json'];
check('oracle r5 servo: kCore 4, stable 0, kExt 5, ceiling 4',
      [sv.sha256.slice(0, 12), sv.kCore, sv.kCoreStable, sv.kExt, sv.kExtStable, sv.ceilingCore],
      ['e0434c2c90da', 4, 0, 5, 0, 4], 'r6_judge-results.json phaseG');
check('oracle r5 servoland: kCore 3, stable 0, kExt 4, ceiling 3',
      [sl.sha256.slice(0, 12), sl.kCore, sl.kCoreStable, sl.kExt, sl.kExtStable, sl.ceilingCore],
      ['880a120ef649', 3, 0, 4, 0, 3], 'r6_judge-results.json phaseG');
check('a servoed entry is an oracle: the servo law is exteroceptive, the policy is not',
      [r5j.observationSets.policyObs.n, r5j.observationSets.policyObs.exteroceptive,
       r5j.observationSets.servoLawReads.n, r5j.observationSets.servoLawReads.exteroceptive],
      [61, false, 7, true], 'r5_judge-results.json observationSets');

// =================================================== 3. the controls
{
  const C4 = Object.fromEntries(r4j.phaseC.rows.map(r => [r.file + '@' + r.rise_mm, r]));
  const C6 = Object.fromEntries(r6j.phaseC.rows.map(r => [r.file + '@' + r.rise_mm, r]));
  const onTread = C4['r4_ctrl_on_tread_60mm.json@60'];
  check('placed duck passes 9/9 core and 14/14 extended',
        [onTread.kCore, onTread.kCoreStable, onTread.kExt, onTread.kExtStable], [9, 9, 14, 14],
        'r4_judge-results.json phaseC');
  check('placed duck sha d99589396fcb, ceilingCore 9',
        [C6['r4_ctrl_on_tread_60mm.json@60'].move, C6['r4_ctrl_on_tread_60mm.json@60'].ceilingCore],
        ['d99589396fcb', 9], 'r6_judge-results.json phaseC');
  check('placed duck at 90 mm passes 9/9 and 14/14',
        [C4['r4_ctrl_on_tread_90mm.json@90'].kCore, C4['r4_ctrl_on_tread_90mm.json@90'].kExtStable,
         C4['r4_ctrl_on_tread_90mm.json@90'].move],
        [9, 14, 'f5bb2f0476c1'], 'r4_judge-results.json phaseC');
  check('placed spawn is labelled NOT A CLIMB', C4['r4_ctrl_on_tread_60mm.json@60'].startKind,
        'PLACED SPAWN — NOT A CLIMB', 'r4_judge-results.json phaseC');
  const dn = C4['ctrl_do_nothing.json@60'];
  check('do-nothing duck fails 0/9 core and 0/14 extended',
        [dn.kCore, dn.kCoreStable, dn.kExt, dn.kExtStable], [0, 0, 0, 0], 'r4_judge-results.json phaseC');
  check('do-nothing sha c703ee6f5a14, ceilingCore 0',
        [C6['ctrl_do_nothing.json@60'].move, C6['ctrl_do_nothing.json@60'].ceilingCore],
        ['c703ee6f5a14', 0], 'r6_judge-results.json phaseC');
  check('do-nothing fails at 40 and 90 mm too',
        [C4['ctrl_do_nothing.json@40'].kCore, C4['ctrl_do_nothing.json@90'].kCore], [0, 0],
        'r4_judge-results.json phaseC');
  check('all three judges: do-nothing always fails, placed duck always passes',
        [r4j.phaseC.doNothingAlwaysFails, r4j.phaseC.placedDuckAlwaysPasses,
         r5j.phaseC.doNothingAlwaysFails, r5j.phaseC.placedDuckAlwaysPasses,
         r6j.phaseC.doNothingAlwaysFails, r6j.phaseC.placedDuckAlwaysPasses],
        [true, true, true, true, true, true],
        'r4/r5/r6_judge-results.json phaseC');
}

// =================================================== 4. the ceiling finding
const kc = r6j.killCondition;
check('394 distinct floor-spawned launches scored at 60 mm', kc.distinctVectorsScoredAt60mm, 394,
      'r6_judge-results.json killCondition');
check('ceiling histogram over all 394', kc.ceilingHistogram,
      { 0: 64, 1: 136, 2: 106, 3: 65, 4: 18, 5: 5 }, 'r6_judge-results.json killCondition');
check('the histogram sums to 394', Object.values(kc.ceilingHistogram).reduce((a, b) => a + b, 0), 394,
      'r6_judge-results.json killCondition');
check('best ceilingCore anywhere is 5 of 9', kc.bestCeilingCore, 5, 'r6_judge-results.json killCondition');
check('best kCoreStable anywhere is 5, held by a56d459fb649',
      [kc.bestKCoreStable, kc.bestKCoreStableMove], [5, 'a56d459fb649'], 'r6_judge-results.json killCondition');
check('the 7-of-9 kill condition FAILED', [kc.result, kc.barMoved], ['FAILED', false],
      'r6_judge-results.json killCondition');
check('round 5 kill gate also FAILED', r5j.killGate.result, 'FAILED', 'r5_judge-results.json killGate');
check('through round 5 the record was 4b9110c448ec at 4 of 9',
      [r5j.killGate.bestKCoreStable, r5j.killGate.bestMove, r5j.killGate.bestFile],
      [4, '4b9110c448ec', 'best_r3_vault_60mm.json'], 'r5_judge-results.json killGate');
check('kCore <= ceilingCore is an identity, and it holds everywhere',
      [r6j.phaseW.kCoreLEQceilingEverywhere, r6j.phaseX.kCoreLEQceilingEverywhere,
       r6j.phaseE.rows.every(r => r.kCoreLEQceiling)], [true, true, true],
      'r6_judge-results.json phaseW/phaseX/phaseE');

// the two halves of the 394
check('64 published intent files screened at 60 mm', r6s.files, 64, 'r6_screen-results.json');
check('48 distinct vectors among them', r6s.distinctVectors, 48, 'r6_screen-results.json');
check('2 of them refused as out of declared bounds',
      r6s.rows.filter(r => r.invalid).map(r => r.file),
      ['best_r3_cornerclimb2_120mm.json', 'best_r3_cornerclimb_120mm.json'], 'r6_screen-results.json');
check('the published screen ran at a 60 mm rise', r6s.rise_mm, 60, 'r6_screen-results.json');
check('the published screen maxes out at ceilingCore 5', r6s.maxCeilingCore, 5, 'r6_screen-results.json');
check('340 CEM moves rebuilt and re-scored, all agreeing',
      [r6j.phaseX.claimedDistinctMoves, r6j.phaseX.rebuilt, r6j.phaseX.hashMatches,
       r6j.phaseX.ceilingMatches, r6j.phaseX.kMatches, r6j.phaseX.maxMeasuredCeilingCore],
      [340, 340, 340, 340, 340, 5], 'r6_judge-results.json phaseX');
check('the CEM optimised the CEILING, with no landing term, no servo, no event',
      [r6c.landingTerm, r6c.servo, r6c.event, r6c.bar_m], ['NONE', 'NONE', 'NONE', 0.095],
      'r6_ceiling-results.json');
check('none of the 340 is out of declared bounds or out of the search box',
      [r6j.phaseX.outOfDeclaredBounds, r6j.phaseX.outOfTheSearchsOwnBox], [[], []],
      'r6_judge-results.json phaseX');

// every cell is individually reachable, and they never accumulate
{
  const peaks = r6j.phaseX.rows.map(r => (r.measured && r.measured.peakAboveTread_mm) || r.claimed.peakAboveTread_mm);
  const percell = [...Array(9)].map((_, i) => +Math.max(...peaks.map(p => p[i])).toFixed(1));
  check('per-cell best peak over the 340-move corpus', percell,
        [124.1, 141.6, 133.1, 143.8, 131.7, 129.8, 127.5, 131, 124.9], 'r6_judge-results.json phaseX');
  check('all 9 cells individually reachable, peaks 124-144 mm',
        [percell.every(v => v > 95), Math.min(...percell), Math.max(...percell)], [true, 124.1, 143.8],
        'r6_judge-results.json phaseX');
  check('no single vector holds more than 5 of them', Math.max(...r6j.phaseX.rows.map(r => r.measured.ceilingCore)), 5,
        'r6_judge-results.json phaseX');
}

// =================================================== 5. the strut lever
{
  const levers = r6lt.perFile.map(f => f.neckLeverH_mm[1]);
  const forces = r6lt.perFile.map(f => f.neckMaxForceAtLongestLever_N);
  check('longest neck_pitch levers at the vault pose (mm)', levers, [89.3, 89.3, 89.5, 87.3],
        'r6_limits_table-results.json perFile');
  check('strut lever range 87.3 - 89.5 mm', [Math.min(...levers), Math.max(...levers)], [87.3, 89.5],
        'r6_limits_table-results.json perFile');
  check('force at the beak through those levers (N)', forces, [7.172, 7.172, 7.156, 7.337],
        'r6_limits_table-results.json perFile');
  check('beak force range 7.16 - 7.34 N',
        [+Math.min(...forces).toFixed(2), +Math.max(...forces).toFixed(2)], [7.16, 7.34],
        'r6_limits_table-results.json perFile');
  check('body weight 7.23 N', r6lt.bodyWeight_N, 7.23, 'r6_limits_table-results.json');
  check('neck stall 7.66 N', r6lt.neckStall_N, 7.66, 'r6_limits_table-results.json');
  checkNear('neck stall lever = 0.6405 / 7.66 = 0.0836 m', r6lt.forcerange_Nm / r6lt.neckStall_N, 0.0836, 5e-5,
            'r6_limits_table-results.json');
  check('every parity check in the limits trace is 9/9 EXACT', r6lt.perFile.map(f => f.parity),
        ['9/9 EXACT', '9/9 EXACT', '9/9 EXACT', '9/9 EXACT'], 'r6_limits_table-results.json perFile');
  check('hip saturation correlates POSITIVELY with peak height, r = +0.254',
        r6lt.correlationsOverAll36Cells.hipPitchSaturation_vs_peakHeight, 0.254,
        'r6_limits_table-results.json correlationsOverAll36Cells');
}
check('in a clearing cell some actuator is at the ceiling 73.6% of a 1,399 ms push-off',
      [r6lt.pooled.over95.anyOf14AtForceCeiling, r6lt.pooled.over95.pushOffMs, r6lt.pooled.over95.n],
      [0.7358, 1399, 20], 'r6_limits_table-results.json pooled.over95');
check('saturation: every pushing row is pinned at 0.6405 N.m, none over it',
      [r6j.phaseT.claimsSaturating, r6j.phaseT.claims,
       r6j.phaseT.publishedCorpusSaturating, r6j.phaseT.publishedCorpusScored,
       r6j.phaseT.rebuiltSearchSaturating, r6j.phaseT.rebuiltSearchScored,
       r6j.phaseT.anyRowOverCeilingAnywhere],
      [21, 21, 60, 62, 340, 340, false], 'r6_judge-results.json phaseT');
check('the two non-saturating published files never push',
      r6j.phaseT.publishedCorpusNotSaturating.map(r => r.file),
      ['best_r2_blockdrag.json', 'best_r2_blockpush.json'], 'r6_judge-results.json phaseT');

// =================================================== 6. the lift budget
{
  const stand = famB.instrument.parity.find(p => p.file === 'ctrl_do_nothing.json');
  check('a standing duck\'s trunk sits at 116.2 mm', stand.terminal_z_mm, 116.2,
        'r4_famB-results.json instrument.parity');
  const peaks = [];
  (function walk(o) {
    if (Array.isArray(o)) o.forEach(walk);
    else if (o && typeof o === 'object') {
      if (typeof o.peakZ_mm === 'number') peaks.push(o.peakZ_mm);
      Object.values(o).forEach(walk);
    }
  })(famB);
  const best = Math.max(...peaks);
  check('family B recorded 114 peakZ_mm values', peaks.length, 114, 'r4_famB-results.json');
  check('family B peaked at 153.8 mm of trunk height', best, 153.8, 'r4_famB-results.json (every peakZ_mm)');
  checkNear('so it bought about 38 mm of lift (153.8 - 116.2 = 37.6)', +(best - stand.terminal_z_mm).toFixed(1),
            37.6, 0.05, 'r4_famB-results.json');
  check('lift NEEDED at 80 / 90 / 120 mm = 59 / 69 / 99 mm',
        [80, 90, 120].map(h => +(h + 95 - stand.terminal_z_mm).toFixed(1)), [58.8, 68.8, 98.8],
        'derived: rise + 95 mm gate - 116.2 mm standing trunk (r4_famB-results.json)');
  check('every family-B file cleared 0 of 9 at its own rise',
        Object.keys(G4).filter(f => f.startsWith('best_r4_famB_')).map(f => G4[f].kCore),
        [0, 0, 0, 0, 0, 0, 0, 0, 0], 'r4_judge-results.json phaseG');
  check('and the ladder\'s best at 90 and 120 mm is 0 of 9',
        [ladder[90].kCore, ladder[120].kCore], [0, 0], 'r4_judge-results.json ladder');
  check('one cell of the grid costs 0.712 s', famB.secPerCell, 0.712, 'r4_famB-results.json');
}

// =================================================== 7. 180 mm
{
  const arm180 = corner.arms.find(a => a.riseMM === 180);
  check('180 mm: 2,829 episodes', arm180.episodes, 2829, 'r3_cornerclimb-results.json arms');
  check('180 mm: ZERO tread contact in all of them',
        [arm180.agg.feetOnTreadMax, arm180.agg.meanFeetOnTreadMax, arm180.agg.meanFeetOnTreadFinal,
         arm180.agg.sustainFrac], [0, 0, 0, 0], 'r3_cornerclimb-results.json arms.agg');
  checkNear('180 mm: mean trunk height ends 80.6 mm BELOW the tread', arm180.agg.meanAbove_mm, -80.6, 0.05,
            'r3_cornerclimb-results.json arms.agg');
  check('180 mm: k = 0', arm180.k, 0, 'r3_cornerclimb-results.json arms');
}

// =================================================== 8. penetration
{
  const N6 = Object.fromEntries(r6j.phaseN.rows.map(r => [r.file, r]));
  check('the record\'s four clears penetrate -8.52, -9.27, -8.57, -8.88 mm',
        N6['best_r3_vault_60mm.json'].clearsMinPenEpisode_mm, [-8.52, -9.27, -8.57, -8.88],
        'r6_judge-results.json phaseN');
  check('its worst single cell anywhere is -13.67 mm', N6['best_r3_vault_60mm.json'].worstCell_mm, -13.67,
        'r6_judge-results.json phaseN');
  check('no clear, in any file, is deeper than the -15 mm soft-contact line',
        [r6j.phaseN.rows.every(r => r.clearsDeeperThanSoft === 0),
         r6j.phaseN.rows.every(r => r.cellsDeeperThanSoft === 0),
         r6j.phaseN.rows.every(r => r.penConsistent)], [true, true, true], 'r6_judge-results.json phaseN');
  check('the round-5 judge measured the same four penetrations',
        r5j.phaseN.rows.find(r => r.file === 'best_r3_vault_60mm.json').clearsMinPenEpisode_mm,
        [-8.52, -9.27, -8.57, -8.88], 'r5_judge-results.json phaseN');
}

// =================================================== 9. the 20 ms tick matters
check('unshifted round-5 servoland launch clears 3 of 9', r5j.phaseF.unshifted.kCore, 3,
      'r5_judge-results.json phaseF');
check('shifting every keyframe by +1 control tick takes it to 1 of 9',
      r5j.phaseF.probes.find(p => p.probe === 'shift_all_plus1tick').kCore, 1, 'r5_judge-results.json phaseF');
check('one tick moves the trunk up to 234.2 mm',
      Math.max(...r5j.phaseF.probes.map(p => p.maxTrunkXShift_mm)), 234.2, 'r5_judge-results.json phaseF');
check('the control tick in that measurement is 20 ms', r5j.phaseF.controlTick_ms, 20, 'r5_judge-results.json phaseF');

// =================================================== 10. the broken flight
{
  const log = T('rig3.r2.log');
  const phaseE = log.slice(log.indexOf('PHASE E'), log.indexOf('threshold sweep'));
  for (const [rise, gap] of [['20', '-45.57'], ['40', '-52.57'], ['60', '-52.57'], ['90', '-52.57'],
                             ['120', '-51.80'], ['180', '-16.62']]) {
    const line = phaseE.split('\n').find(l => new RegExp(`^\\s+${rise}\\s+4\\s`).test(l));
    check(`4-step flight at ${rise} mm: step blocks interpenetrate by ${gap} mm`,
          line && line.trim().split(/\s+/)[2], gap, 'rig3.r2.log PHASE E');
  }
  check('tread drift up to 19.63 mm and tread sag up to 16.26 mm under the broken flight',
        [/19\.63/.test(phaseE), /16\.26/.test(phaseE)], [true, true], 'rig3.r2.log PHASE D/E');
  const sweep = log.slice(log.indexOf('threshold sweep'));
  const rowsSweep = sweep.split('\n').filter(l => /^\s+\d+\s+[14]\s+\d+/.test(l));
  const four = rowsSweep.filter(l => /^\s+\d+\s+4\s/.test(l) && +l.trim().split(/\s+/)[0] <= 140);
  const one = rowsSweep.filter(l => /^\s+\d+\s+1\s/.test(l) && +l.trim().split(/\s+/)[0] <= 140
                                    && +l.trim().split(/\s+/)[0] >= 40);
  check('a duck PLACED on the tread fails honest at every rise 20-140 mm with the 4-step flight',
        [four.length > 0, four.every(l => !/PASS/.test(l))], [true, true], 'rig3.r2.log threshold sweep');
  check('...and passes at every rise 40-140 mm with a single step',
        [one.length > 0, one.every(l => /PASS/.test(l))], [true, true], 'rig3.r2.log threshold sweep');
}

// =================================================== 11. round-6 admissibility
check('round 6 moved nothing on the round-4 floor: 86/86 rows reproduce exactly',
      [r6j.phaseP1.rows, r6j.phaseP1.deepExact, r6j.phaseP1.reproducesR4Judge86, r6j.phaseP1.deepDiffs.length],
      [86, 86, true, 0], 'r6_judge-results.json phaseP1');
check('the verdict scorer is unchanged (45/45 cells exact)',
      [r6j.phaseP3.cells, r6j.phaseP3.deepExact, r6j.phaseP3.verdict],
      [45, 45, 'THE VERDICT SCORER IS UNCHANGED'], 'r6_judge-results.json phaseP3');
check('round 5 moved nothing either: 86/86', [r5j.phaseP.rows, r5j.phaseP.deepExact, r5j.phaseP.reproducesR4Judge],
      [86, 86, true], 'r5_judge-results.json phaseP');
check('the round-6 judge declares the corpus admissible', r6j.admissible, true, 'r6_judge-results.json');

// =================================================== 12. search effort
{
  const s0 = R('search_0-results.json'), s1 = R('search_1-results.json'), s2 = R('search_2-results.json');
  const s3 = R('search_3-results.json'), blk = R('search_block-results.json'), vlt = R('vault-results.json');
  const r3v = R('r3_vault-results.json'), r3lv = R('r3_landvault-results.json');
  const cc2 = R('r3_cornerclimb2-results.json'), fA = R('r4_famA-results.json');
  const r5ss = R('r5_servo_search-results.json'), r5sl = R('r5_servoland-results.json');
  check('round 1 search_0: 1,822 evals', s0.runs.reduce((a, r) => a + r.evals, 0), 1822, 'search_0-results.json');
  check('round 1 search_1: 2,072 evals', Object.values(s1).reduce((a, r) => a + r.evals, 0), 2072,
        'search_1-results.json');
  check('round 2 search_2: 5,202 evals',
        ['leg1_baseline', 'leg2_single_start', 'leg2_robust']
          .reduce((a, k) => a + Object.values(s2[k]).reduce((b, r) => b + r.evals, 0), 0), 5202,
        'search_2-results.json');
  check('round 2 search_3: 4,974 evals', s3.totalEvals, 4974, 'search_3-results.json');
  check('round 2 block search: 4,410 episodes',
        ['B', 'C_90mm', 'C_180mm'].reduce((a, k) => a + blk.phases[k].episodes, 0), 4410,
        'search_block-results.json');
  check('round 2 vault: 5,265 episodes', vlt.totals.episodes, 5265, 'vault-results.json');
  check('round 3 vault: 3,411 evals', r3v.evals, 3411, 'r3_vault-results.json');
  check('round 3 landvault: 762 evals', r3lv.totals.evals, 762, 'r3_landvault-results.json');
  check('round 3 corner climb: 2,753 + 2,829 = 5,582 episodes',
        corner.arms.reduce((a, x) => a + x.episodes, 0), 5582, 'r3_cornerclimb-results.json');
  check('round 3 corner climb 2: 2,220 episodes', cc2.episodes, 2220, 'r3_cornerclimb2-results.json');
  check('round 4 family A: 416 evals', fA.search.evals, 416, 'r4_famA-results.json');
  check('round 4 family B: 3,186 evals',
        ['beat1', 'beat2', 'concat'].reduce((a, k) => a + famB[k].reduce((b, r) => b + (r.evals || 0), 0), 0),
        3186, 'r4_famB-results.json');
  check('round 5 servo search: 786 stage-1 candidates, 24 stage-2',
        [r5ss.stage1Candidates, r5ss.stage2Evaluated], [786, 24], 'r5_servo_search-results.json');
  check('round 5 servoland: 234 full-grid evaluations, 317 screened',
        [r5sl.budget.fullGridEvaluations, r5sl.budget.screened], [234, 317], 'r5_servoland-results.json');
  check('round 6 ceiling: 1,614 evals, 341 full, 340 distinct moves',
        [r6c.evals, r6c.fullEvals, r6c.distinct], [1614, 341, 340], 'r6_ceiling-results.json');
}

// =================================================== 13. the vault's parameter count
{
  const p = path.join(HERE, 'intents', 'best_r3_vault_60mm.json');
  const j = JSON.parse(fs.readFileSync(p, 'utf8'));
  check('the beak-strut vault has 29 search parameters', Object.keys(j.params).length, 29,
        'intents/best_r3_vault_60mm.json params');
  check('the rank-2 vault intent has 6 keyframes of 14 joint targets',
        [j.keyframes.length, [...new Set(j.keyframes.map(k => k.pose.length))]], [6, [14]],
        'intents/best_r3_vault_60mm.json keyframes');
  check('the rank-2 vault intent is in declared bounds',
        [j.blend >= 0.7 && j.blend <= 2.4, j.side >= -0.02 && j.side <= 0.09], [true, true],
        'intents/best_r3_vault_60mm.json + r6_judge-results.json constants.declaredBounds');
  check('the rank-2 full sha256, as the leaderboard abbreviates it', v60.sha256,
        '4b9110c448ec45b7e9aa9e25e8720ab6e149562dbba2c00dda73f5c86aee8f15', 'r6_judge-results.json phaseG');
  check('the rank-1 full sha256', c1.sha256,
        'a56d459fb6493855d635021dce569cc8b06b325b32b3c19e8593cf430ca442d1', 'r6_judge-results.json phaseG');
}

// =================================================== 14. the plant digest
{
  const mjb = path.join(HERE, '..', 'sim', 'scene.mjb');
  if (fs.existsSync(mjb)) {
    const d = crypto.createHash('sha256').update(fs.readFileSync(mjb)).digest('hex');
    check('plant sim/scene.mjb sha256', d,
          '3f8c9ab9b409ba74c73c30179d5f7c12b025f631693f9eec78d80dca242547be', 'sim/scene.mjb (hashed here)');
  } else {
    rows.push({ ok: true, label: 'plant sim/scene.mjb sha256 (SKIPPED — not in this package)',
                got: 'n/a', want: '3f8c9ab9b409…', src: 'sim/scene.mjb' });
    pass++;
  }
}

// ---------------------------------------------------------------- report
const W = Math.max(...rows.map(r => r.label.length));
for (const r of rows) {
  console.log(`${r.ok ? 'PASS' : 'FAIL'}  ${r.label.padEnd(W)}  ${r.ok ? '' : `got ${r.got} want ${r.want}  `}[${r.src}]`);
}
console.log(`\n${pass} PASS, ${fail} FAIL, ${rows.length} checks.`);
process.exit(fail ? 1 : 0);
