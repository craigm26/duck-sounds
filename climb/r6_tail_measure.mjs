// ROUND 6, PHASE 1 — WHEN, INSIDE THE TAIL, DOES THE DUCK FALL?
//
// Round 5 reported one number about the tail: the servoed moves are upright a
// mean 17 of 50 ticks against the record's 47 of 50. Nobody measured WHEN
// inside those 50 ticks uprightness goes, or what the duck was doing when it
// went. This does, per core cell, for three saved files:
//
//   best_r3_vault_60mm.json            the record (no servo block at all)
//   best_r5_servo_60mm.json            round 5's best servoed move by objective
//   best_r5_servoland_kcore_60mm.json  round 5's best servoed move by kCore
//
// The 9 core cells are robust.mjs's own: rise {h-10, h, h+10} x plant
// {(0.120, x1.0), (0.130, x0.7), (0.125, x1.3)}, scored through scoreCell so
// every number comes out of the ONE shared scorer, from the SAVED file.
//
// Per cell it reports:
//   * the tail tick at which projected gravity z crosses -0.90 (the fall),
//     or that the duck entered the tail already down, or never fell
//   * trunk pitch / height above the tread / x, and both feet's positions,
//     at that tick and 5 ticks before it
//   * the saturation fraction over the 50 tail ticks
//   * the commanded-minus-measured joint error on both hips, knees and ankles
//     at the fall tick
//
// It is a MEASUREMENT: nothing is tuned, nothing is searched, no file is
// written except the results.
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/r6_tail_measure.mjs
import fs from 'node:fs';
import { scoreCell, PLANTS, DHS, intentHashOfFile } from '../climb/robust.mjs';

const P = '../climb/';
const LOG = [];
const log = s => { console.log(s); LOG.push(String(s)); };
const T0 = Date.now();
const el = () => ((Date.now() - T0) / 1000).toFixed(0) + 's';

const FILES = [
  ['best_r3_vault_60mm.json', 'record (round-3 beak-strut vault, NO servo)'],
  ['best_r5_servo_60mm.json', 'round-5 servoed best by objective'],
  ['best_r5_servoland_kcore_60mm.json', 'round-5 servoed best by kCore'],
];
const RISE = 0.060;

// slot -> name, for the commanded-vs-measured error
const LEG = [[2, 'L_hip_pitch'], [3, 'L_knee'], [4, 'L_ankle'],
             [11, 'R_hip_pitch'], [12, 'R_knee'], [13, 'R_ankle']];

const mm = v => +(v * 1000).toFixed(1);
const f4 = v => +v.toFixed(4);

/** Everything worth knowing about one tail sample. */
const shot = s => s === null ? null : ({
  t: s.t, servoed: s.servoed, gz: f4(s.gz), up: s.up, pitch: f4(s.pitch), roll: f4(s.roll),
  trunkX_mm: mm(s.x), trunkAbove_mm: mm(s.above), trunkZ_mm: mm(s.z), dy_mm: mm(s.y - 0),
  lfoot_mm: s.lfoot.map(mm), rfoot_mm: s.rfoot.map(mm),
  sat: s.sat,
});

const OUT = { generated: new Date().toISOString(), script: 'r6_tail_measure.mjs',
              rise_mm: 60, cells: 'the 9 core cells of climb/robust.mjs',
              upright: 'projectedGravity(trunkQuat)[2] < -0.90', files: [] };

for (const [f, label] of FILES) {
  const sha = intentHashOfFile(P + f);
  const j = JSON.parse(fs.readFileSync(P + f, 'utf8'));
  log('');
  log('================================================================');
  log(`${f}   ${sha.slice(0, 12)}   ${label}`);
  log(`   servo: ${j.servo ? `armed at t=${j.servo.at}s, tailTicks=${j.servo.tailTicks || 0}` : 'NONE'}`);
  const rec = { file: f, sha256: sha, move: sha.slice(0, 12), label,
                servo: j.servo ? { at: j.servo.at, tailTicks: j.servo.tailTicks || 0 } : null, cells: [] };
  for (const dh of DHS) for (const p of PLANTS) {
    const r = await scoreCell(P + f, { rise: RISE, dh, drop: p.drop, fmul: p.fmul, tailTrace: true });
    const tr = r.tailTrace;
    const enteredUp = r.atTrackEnd.up;
    // THE FALL: the first tail tick at which the duck is no longer upright.
    let fall = -1;
    for (const s of tr) if (!s.up) { fall = s.t; break; }
    const neverFell = fall === -1;
    const alreadyDown = !enteredUp && fall === 0;
    const at = fall >= 0 ? tr[fall] : null;
    const before = fall >= 5 ? tr[fall - 5] : null;
    // did it come back?
    let recovered = null;
    if (fall >= 0) { recovered = false; for (const s of tr) if (s.t > fall && s.up) { recovered = true; break; } }
    const satSum = tr.reduce((a, s) => a + s.sat, 0);
    const err = at ? LEG.map(([k, n]) => ({ joint: n, slot: k,
      cmd: f4(at ? 0 : 0) })) : null;
    // commanded minus measured, at the fall tick
    const jerr = fall >= 0 ? LEG.map(([k, n]) => {
      const s = tr[fall];
      return { joint: n, cmd: f4(s.cmd[k]), q: f4(s.qpos[k]), err: f4(s.cmd[k] - s.qpos[k]) };
    }) : null;
    const cell = {
      cell: { rise_mm: Math.round((RISE + dh) * 1000), drop: p.drop, fmul: p.fmul },
      honest: r.crit.honest, stableClear: r.crit.honest && r.uprightTailTicks >= 45,
      uprightTailTicks: r.uprightTailTicks,
      enteredTailUpright: enteredUp,
      trunkAboveAtTrackEnd_mm: mm(r.atTrackEnd.above),
      peakZ_mm: mm(r.maxZ), ceiling_mm: mm(r.maxZ - (RISE + dh)),
      fallTick: neverFell ? null : fall,
      fallStatus: neverFell ? 'upright for all 50 tail ticks'
        : alreadyDown ? 'entered the tail already down'
        : `fell at tail tick ${fall} (${(fall * 20)} ms into the tail)`,
      recoveredAfterFall: recovered,
      tailSatFrac: +(satSum / (tr.length * 14)).toFixed(4),
      servoTailTicksRun: r.servo ? r.servo.tailTicksRun : null,
      atFall: shot(at), fiveBefore: shot(before),
      jointErrorAtFall: jerr,
      scored: { x_mm: mm(r.scored.x), above_mm: mm(r.scored.above), up: r.scored.up,
                feetOnTread: r.scored.feetOnTread },
      // the whole tail, thinned to every 5th tick, so the shape is on the record
      tailEvery5: tr.filter(s => s.t % 5 === 0).map(s => ({ t: s.t, gz: f4(s.gz), up: s.up,
        pitch: f4(s.pitch), above_mm: mm(s.above), x_mm: mm(s.x),
        lz_mm: mm(s.lfoot[2]), rz_mm: mm(s.rfoot[2]), sat: s.sat })),
    };
    rec.cells.push(cell);
    log(`   rise ${cell.cell.rise_mm} drop ${p.drop} f${p.fmul}  honest=${String(r.crit.honest).padEnd(5)} upTail=${String(r.uprightTailTicks).padStart(2)}/50  enteredUp=${String(enteredUp).padEnd(5)} ` +
        `fall=${cell.fallTick === null ? 'none' : String(cell.fallTick).padStart(2)}  satFrac=${cell.tailSatFrac.toFixed(3)}` +
        (at ? `  @fall pitch=${shot(at).pitch} above=${shot(at).trunkAbove_mm}mm x=${shot(at).trunkX_mm}mm lz=${shot(at).lfoot_mm[2]} rz=${shot(at).rfoot_mm[2]}` : ''));
  }
  // per-file summary
  const cs = rec.cells;
  const fell = cs.filter(c => c.fallTick !== null && c.enteredTailUpright);
  rec.summary = {
    kCore: cs.filter(c => c.honest).length,
    kCoreStable: cs.filter(c => c.stableClear).length,
    meanUprightTailTicks: +(cs.reduce((a, c) => a + c.uprightTailTicks, 0) / cs.length).toFixed(2),
    cellsEnteringTailUpright: cs.filter(c => c.enteredTailUpright).length,
    cellsUprightAll50: cs.filter(c => c.fallTick === null).length,
    cellsEnteringDown: cs.filter(c => !c.enteredTailUpright).length,
    fallTicks: cs.map(c => c.fallTick),
    meanFallTickOfThoseThatFellFromUpright: fell.length ? +(fell.reduce((a, c) => a + c.fallTick, 0) / fell.length).toFixed(2) : null,
    medianFallTick: (() => { const v = fell.map(c => c.fallTick).sort((a, b) => a - b); return v.length ? v[Math.floor(v.length / 2)] : null; })(),
    meanTailSatFrac: +(cs.reduce((a, c) => a + c.tailSatFrac, 0) / cs.length).toFixed(4),
  };
  log(`   SUMMARY kCore=${rec.summary.kCore}/9 kCoreStable=${rec.summary.kCoreStable}/9 meanUpTail=${rec.summary.meanUprightTailTicks}/50 ` +
      `enteredUp=${rec.summary.cellsEnteringTailUpright}/9 upAll50=${rec.summary.cellsUprightAll50}/9 ` +
      `meanFallTick=${rec.summary.meanFallTickOfThoseThatFellFromUpright} medianFallTick=${rec.summary.medianFallTick} meanSatFrac=${rec.summary.meanTailSatFrac}  [${el()}]`);
  OUT.files.push(rec);
}

fs.writeFileSync(P + 'r6_tailmeasure-results.json', JSON.stringify(OUT, null, 2));
fs.writeFileSync(P + 'r6_tail_measure.log', LOG.join('\n') + '\n');
log('');
log(`written climb/r6_tailmeasure-results.json  [${el()}]`);
