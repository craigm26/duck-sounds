// ROUND 6, PHASE 2 — DOES KEEPING THE LAW THROUGH THE TAIL MOVE THE FALL?
//
// Phase 1 (climb/r6_tailmeasure-results.json) measured WHEN the servoed moves
// lose uprightness: within the first 3 tail ticks, at a hand-back pose ~46-53 mm
// above the tread with 13 or 14 of the 14 slots already clamped at a limit.
// The obvious question is whether the law holding the legs longer prevents it.
//
// This writes SIX new intents — the two round-5 servoed bests, each with
// servo.tailTicks 10, 25 and 50 and nothing else changed — and scores every one
// of them, from the SAVED file, over robust.mjs's 9 core cells, with the same
// per-tail-tick trace phase 1 used. The two parents (tailTicks 0) are re-scored
// in the same run so the comparison is apples to apples.
//
// It is a MEASUREMENT: tailTicks is the only field that moves, no gain is
// tuned, no search is run.
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/r6_tail_experiment.mjs
import fs from 'node:fs';
import { scoreRobust, intentHash, intentHashOfFile } from '../climb/robust.mjs';

const P = '../climb/';
const LOG = [];
const log = s => { console.log(s); LOG.push(String(s)); };
const T0 = Date.now();
const el = () => ((Date.now() - T0) / 1000).toFixed(0) + 's';
const RISE = 0.060;
const mm = v => +(v * 1000).toFixed(1);

const PARENTS = [
  ['best_r5_servo_60mm.json', 'servo', 'round-5 servoed best by objective'],
  ['best_r5_servoland_kcore_60mm.json', 'servoland', 'round-5 servoed best by kCore'],
];
const TICKS = [10, 25, 50];

// ------------------------------------------------------------------ the files
const PLAN = [];
for (const [pf, fam, label] of PARENTS) {
  PLAN.push({ file: pf, parent: pf, family: fam, tailTicks: 0, label: label + ', tailTicks 0 (as published)' });
  for (const n of TICKS) {
    const j = JSON.parse(fs.readFileSync(P + pf, 'utf8'));
    j.servo = { ...j.servo, tailTicks: n };
    j.name = `r6_tail${n}_${fam}_60mm`;
    j.family = `r6 tail authority ${n} ticks`;
    j.note = `ROUND 6, THE TAIL. ${pf} verbatim — every keyframe, blend, gap, side, approach, ` +
      `set-point and gain unchanged — with servo.tailTicks = ${n}: the same servo law keeps the ` +
      `leg slots for the first ${n} of the 50 policy-tail ticks. It is a MEASUREMENT of where the ` +
      `tail's topple comes from, not a tuned move. The law is EXTEROCEPTIVE (it reads the tread's ` +
      `height and edge straight from the plant), so every number it produces is an ORACLE UPPER ` +
      `BOUND: the robot's 61 proprioceptive policy inputs cannot see those readings.`;
    delete j.robust;
    const out = `best_r6_tail${n}_${fam}_60mm.json`;
    fs.writeFileSync(P + out, JSON.stringify(j, null, 2));
    PLAN.push({ file: out, parent: pf, family: fam, tailTicks: n,
                label: `${fam} + ${n} tail ticks of law` });
  }
}

const OUT = { generated: new Date().toISOString(), script: 'r6_tail_experiment.mjs',
              rise_mm: 60, grid: 'the 9 core cells of climb/robust.mjs (core:true)',
              oracle: 'the servo law reads tread height and edge from the plant; the policy cannot. Every servoed number is an upper bound.',
              rows: [] };

log('================================================================');
log('ROUND 6, PHASE 2 — servo.tailTicks 0 / 10 / 25 / 50 on the two round-5 servoed bests');
log('  9 core cells each, scored from the SAVED file through robust.mjs scoreRobust(core:true)');
log('');
log('file                                    tail  sha256        kCore kStable meanUpTail  ceilingCore  medianFall  meanSatFrac  objectiveCore');

for (const p of PLAN) {
  const sha = intentHashOfFile(P + p.file);
  const traces = [];
  const g = await scoreRobust(P + p.file, { rise: RISE, core: true, tailTrace: true,
    onCell: c => traces.push(c) });
  const cells = traces.map(c => {
    const tr = c.tailTrace;
    let fall = null;
    for (const s of tr) if (!s.up) { fall = s.t; break; }
    let recovered = null;
    if (fall !== null) { recovered = false; for (const s of tr) if (s.t > fall && s.up) { recovered = true; break; } }
    const satSum = tr.reduce((a, s) => a + s.sat, 0);
    return {
      cell: c.cell, honest: c.crit.honest, stableClear: c.stableClear,
      uprightTailTicks: c.uprightTailTicks,
      enteredTailUpright: c.atTrackEnd.up,
      fallTick: fall, recoveredAfterFall: recovered,
      tailSatFrac: +(satSum / (tr.length * 14)).toFixed(4),
      servoTailTicksRun: c.servo ? c.servo.tailTicksRun : null,
      peakZ_mm: mm(c.maxZ), ceiling_mm: mm(c.maxZ - c.cell.rise_mm / 1000),
      above_mm: mm(c.scored.above), x_mm: mm(c.scored.x), up: c.scored.up,
      feetOnTread: c.scored.feetOnTread,
      lastTick: { gz: +tr[49].gz.toFixed(4), pitch: +tr[49].pitch.toFixed(4),
                  above_mm: mm(tr[49].above), sat: tr[49].sat },
      tailEvery5: tr.filter(s => s.t % 5 === 0).map(s => ({ t: s.t, gz: +s.gz.toFixed(3),
        up: s.up, pitch: +s.pitch.toFixed(3), above_mm: mm(s.above), sat: s.sat })),
    };
  });
  const fell = cells.filter(c => c.fallTick !== null);
  const med = (() => { const v = fell.map(c => c.fallTick).sort((a, b) => a - b); return v.length ? v[Math.floor(v.length / 2)] : null; })();
  const row = {
    file: p.file, parent: p.parent, family: p.family, tailTicks: p.tailTicks, label: p.label,
    sha256: sha, move: sha.slice(0, 12),
    kCore: g.kCore, kCoreStable: g.kCoreStable,
    ceilingCore: cells.filter(c => c.ceiling_mm > 95).length,
    meanUprightTailTicks: +(cells.reduce((a, c) => a + c.uprightTailTicks, 0) / cells.length).toFixed(2),
    cellsUprightAll50: cells.filter(c => c.fallTick === null).length,
    cellsEnteringTailUpright: cells.filter(c => c.enteredTailUpright).length,
    fallTicks: cells.map(c => c.fallTick),
    medianFallTick: med,
    meanTailSatFrac: +(cells.reduce((a, c) => a + c.tailSatFrac, 0) / cells.length).toFixed(4),
    servoTailTicksRun: cells.map(c => c.servoTailTicksRun),
    objectiveCore: +g.objectiveCore.toFixed(4), objectiveR3: +g.objectiveR3.toFixed(4),
    meanReward: +g.meanReward.toFixed(4),
    minPenetrationEpisode_mm: mm(Math.min(...traces.map(c => c.minPenetrationEpisode))),
    cells,
  };
  OUT.rows.push(row);
  log(`${p.file.padEnd(40)}${String(p.tailTicks).padStart(4)}  ${row.move}  ${String(row.kCore).padStart(3)}/9 ${String(row.kCoreStable).padStart(5)}/9 ` +
      `${String(row.meanUprightTailTicks).padStart(9)}/50  ${String(row.ceilingCore).padStart(9)}/9  ${String(row.medianFallTick).padStart(9)}  ` +
      `${row.meanTailSatFrac.toFixed(4).padStart(10)}  ${row.objectiveCore.toFixed(4).padStart(12)}   [${el()}]`);
}

// ------------------------------------------------------------------- the table
log('');
log('PER-CELL FALL TICK (null = upright for all 50 tail ticks)');
const cellName = c => `${c.cell.rise_mm}/${c.cell.drop}/${c.cell.fmul}`;
log('file                                    tail  ' + OUT.rows[0].cells.map(c => cellName(c).padStart(14)).join(''));
for (const r of OUT.rows)
  log(`${r.file.padEnd(40)}${String(r.tailTicks).padStart(4)}  ` +
      r.cells.map(c => String(c.fallTick === null ? 'none' : c.fallTick).padStart(14)).join(''));
log('');
log('PER-CELL UPRIGHT TAIL TICKS (of 50)');
for (const r of OUT.rows)
  log(`${r.file.padEnd(40)}${String(r.tailTicks).padStart(4)}  ` +
      r.cells.map(c => String(c.uprightTailTicks).padStart(14)).join(''));

// ------------------------------------------------------------------- verdict
const byFam = {};
for (const r of OUT.rows) (byFam[r.family] ||= []).push(r);
OUT.verdict = {};
for (const [fam, rows] of Object.entries(byFam)) {
  const base = rows.find(r => r.tailTicks === 0);
  OUT.verdict[fam] = rows.map(r => ({
    tailTicks: r.tailTicks, move: r.move, kCore: r.kCore, kCoreStable: r.kCoreStable,
    meanUprightTailTicks: r.meanUprightTailTicks,
    dMeanUprightTailTicks: +(r.meanUprightTailTicks - base.meanUprightTailTicks).toFixed(2),
    medianFallTick: r.medianFallTick, meanTailSatFrac: r.meanTailSatFrac,
    objectiveCore: r.objectiveCore,
  }));
}
OUT.killCondition = 'If no launch reaches ceilingCore >= 7 of 9 at 60 mm, the search is FINISHED AT THIS SCALE: the shortfall is the 0.6405 N.m saturation that already closed the 80-120 mm band and 180 mm, and the answer is a different actuator or a different move class (a second duck, a wall, a lever), not more optimisation.';
OUT.maxCeilingCore = Math.max(...OUT.rows.map(r => r.ceilingCore));

log('');
log('VERDICT');
for (const [fam, rows] of Object.entries(OUT.verdict)) {
  log(`  ${fam}:`);
  for (const r of rows) log(`     tailTicks ${String(r.tailTicks).padStart(2)}  kCore ${r.kCore}/9  kCoreStable ${r.kCoreStable}/9  meanUpTail ${r.meanUprightTailTicks}/50 (${r.dMeanUprightTailTicks >= 0 ? '+' : ''}${r.dMeanUprightTailTicks})  medianFall ${r.medianFallTick}  satFrac ${r.meanTailSatFrac}  objCore ${r.objectiveCore}`);
}
log(`  max ceilingCore over every row here: ${OUT.maxCeilingCore}/9 (the bar is 7)`);

fs.writeFileSync(P + 'r6_tail-results.json', JSON.stringify(OUT, null, 2));
fs.writeFileSync(P + 'r6_tail_experiment.log', LOG.join('\n') + '\n');
log('');
log(`written climb/r6_tail-results.json  [${el()}]`);
