// ROUND 5 — (a) the servoed landing ENGAGES, (b) the whole-episode penetration
// field on every published clear.
//
// Run from sim/:  cd ~/projects/duck-sounds/sim && node ../climb/_r5_servo_demo.mjs
import fs from 'node:fs';
import { scoreSaved } from '../climb/rig3.mjs';
import { scoreRobust, intentHashOfFile } from '../climb/robust.mjs';

const P = '../climb/';
const LOG = [];
const log = s => { console.log(s); LOG.push(String(s)); };
const T0 = Date.now();
const el = () => ((Date.now() - T0) / 1000).toFixed(0) + 's';
const mm = v => (v * 1000).toFixed(1);
const OUT = { generated: new Date().toISOString(), script: '_r5_servo_demo.mjs' };

const BASE = P + 'best_r3_vault_60mm.json';
const base = JSON.parse(fs.readFileSync(BASE, 'utf8'));
// The vault's TUCK: keyframe 4 of 6, t = 1.1484 s — the deep knee tuck the
// round-4 judge shifted by one control tick to take the move from 4 of 9 to
// 1 of 9. Everything after it is the landing, which is what the servo replaces.
const TUCK = base.keyframes[3].t;

// ==================================================================== PHASE S0
// A servo block that NEVER ARMS must reproduce the timed move exactly — the
// same degeneracy round 4 required of the event mechanism.
log('================================================================');
log(`PHASE S0 — a servo that never arms is the timed move (base ${BASE}, tuck t=${TUCK}s)`);
const neverPath = P + 'r5_servo_never_60mm.json';
fs.writeFileSync(neverPath, JSON.stringify({ ...base,
  name: 'r5_servo_never', servo: { at: 99, kHipZ: 3, kKneeZ: -3, kAnkFz: 2 },
  note: 'DEGENERACY CHECK. A servo armed at t=99 s never fires inside a 3.2 s track, so this file must replay as best_r3_vault_60mm.json does, at full float digits.' }, null, 2));
const A0 = await scoreSaved(BASE, { rise: 0.060, tail: 'policy' });
const B0 = await scoreSaved(neverPath, { rise: 0.060, tail: 'policy' });
const same = ['x', 'y', 'z', 'above', 'up', 'feetOnTread', 'feetUpRaw'].every(k => Object.is(A0.scored[k], B0.scored[k]))
  && Object.is(A0.reward, B0.reward) && Object.is(A0.maxZ, B0.maxZ) && Object.is(A0.maxX, B0.maxX)
  && Object.is(A0.minPenetrationEpisode, B0.minPenetrationEpisode);
log(`   base      x=${A0.scored.x} z=${A0.scored.z} reward=${A0.reward} minPenEpisode=${A0.minPenetrationEpisode}`);
log(`   servo@99s x=${B0.scored.x} z=${B0.scored.z} reward=${B0.reward} minPenEpisode=${B0.minPenetrationEpisode}  armed=${B0.servo.armed}`);
log(`   DEGENERATE MATCH: ${same}`);
OUT.phaseS0 = { file: neverPath, armed: B0.servo.armed, exact: same,
  base: { x: A0.scored.x, z: A0.scored.z, reward: A0.reward },
  servoNever: { x: B0.scored.x, z: B0.scored.z, reward: B0.reward } };

// ==================================================================== PHASE S1
// ARMED at the tuck. The legs stop following keyframes and are commanded from
// measured trunk/foot state every tick.
const SERVO = {
  at: TUCK, yawRoll: 'hold',
  zTarget: 0.115, xTrunk: 0.16, xFoot: 0.10, fz: 0.015, pitchRef: 0,
  kHipZ: 1.5, kHipPitch: 0.6, kHipX: 1.2, kHipTrunkX: 0.0,
  kKneeZ: -2.0, kKneeFz: -1.5, kKneeX: 0.0,
  kAnkPitch: 0.4, kAnkFz: 1.0,
  rate: 0.15, span: 1.2, sign: [1, -1],
};
const servoPath = P + 'r5_servo_armed_60mm.json';
fs.writeFileSync(servoPath, JSON.stringify({ ...base,
  name: 'r5_servo_armed', family: 'r5_servo',
  servo: SERVO,
  note: `ENGAGEMENT DEMONSTRATION, not a search result. best_r3_vault_60mm.json with a servo block armed at the vault's own tuck time (${TUCK} s). From that tick the ten LEG slots are commanded per tick from measured trunk height/pitch/x and foot x/z; neck and head keep following the keyframes. Gains are authored, NOT searched.` }, null, 2));

log('');
log(`PHASE S1 — servo ARMED at the tuck (t=${TUCK}s). Same file, same keyframes, same blend/gap/side.`);
const A = await scoreSaved(BASE, { rise: 0.060, tail: 'policy', overrides: { trace: true } });
const B = await scoreSaved(servoPath, { rise: 0.060, tail: 'policy', overrides: { trace: true, servoTrace: true } });
log(`   timed  : x=${mm(A.scored.x)}mm z=${mm(A.scored.z)}mm above=${mm(A.scored.above)}mm feetOnTread=${A.scored.feetOnTread} up=${A.scored.up} honest=${A.crit.honest} reward=${A.reward.toFixed(3)} upTail=${A.uprightTailTicks}/${A.tailTicks}`);
log(`   servoed: x=${mm(B.scored.x)}mm z=${mm(B.scored.z)}mm above=${mm(B.scored.above)}mm feetOnTread=${B.scored.feetOnTread} up=${B.scored.up} honest=${B.crit.honest} reward=${B.reward.toFixed(3)} upTail=${B.uprightTailTicks}/${B.tailTicks}`);
log(`   servo armed=${B.servo.armed} tArm=${B.servo.tArm}s ticks=${B.servo.ticks}`);
log(`   trunk moved ${mm(B.scored.x - A.scored.x)} mm in x, ${mm(B.scored.z - A.scored.z)} mm in z at the scored instant`);
// tick-by-tick divergence of the two trajectories
const tA = A.trace, tB = B.trace;
log('   tick   phase   timed x/z/lfootX/lfootZ (mm)          servoed x/z/lfootX/lfootZ (mm)        |dx| |dz| |dfootZ|');
let maxDX = 0, maxDZ = 0, maxDF = 0, firstDiverge = null;
for (let i = 0; i < Math.min(tA.length, tB.length); i++) {
  const a = tA[i], b = tB[i];
  const dx = Math.abs(b.x_mm - a.x_mm), dz = Math.abs(b.z_mm - a.z_mm), df = Math.abs(b.lfootZ_mm - a.lfootZ_mm);
  if (dx > maxDX) maxDX = dx; if (dz > maxDZ) maxDZ = dz; if (df > maxDF) maxDF = df;
  if (firstDiverge === null && (dx > 0.05 || dz > 0.05 || df > 0.05)) firstDiverge = a.tick;
  if (i % 3 === 0 || dx > 5)
    log(`   ${String(a.tick).padStart(4)}  ${a.phase.padEnd(6)} ${String(a.x_mm).padStart(7)} ${String(a.z_mm).padStart(6)} ${String(a.lfootX_mm).padStart(7)} ${String(a.lfootZ_mm).padStart(6)}   ` +
        `${String(b.x_mm).padStart(7)} ${String(b.z_mm).padStart(6)} ${String(b.lfootX_mm).padStart(7)} ${String(b.lfootZ_mm).padStart(6)}   ${dx.toFixed(1).padStart(5)} ${dz.toFixed(1).padStart(5)} ${df.toFixed(1).padStart(6)}`);
}
log(`   first tick the two trajectories differ: ${firstDiverge} (servo armed at track t=${B.servo.tArm}s = tick ${Math.round(B.servo.tArm * 50) + 25})`);
log(`   peak divergence over the episode: trunk x ${maxDX.toFixed(1)} mm, trunk z ${maxDZ.toFixed(1)} mm, left foot z ${maxDF.toFixed(1)} mm`);
log(`   scoreSaved returned a FULL row: fields=${Object.keys(B).length} crit=${JSON.stringify(B.crit.honest)} penAtScore=${mm(B.penetrationAtScore)}mm minPenEpisode=${mm(B.minPenetrationEpisode)}mm uprightTail=${B.uprightTailTicks}/50 sha-able=${!!B.source}`);
log('   servo readings and commands (every 5th servoed tick):');
for (const s of B.servo.trace.slice(0, 14))
  log(`      t=${String(s.t).padStart(6)}  above=${String(s.above_mm).padStart(7)}mm pitch=${String(s.pitch).padStart(8)} trunkX=${String(s.trunkX_mm).padStart(7)}mm  Lfoot dx=${String(s.lfoot.dx_mm).padStart(7)} dz=${String(s.lfoot.dz_mm).padStart(7)}  ->  hipP=[${s.cmd[2]}, ${s.cmd[11]}] knee=[${s.cmd[3]}, ${s.cmd[12]}] ank=[${s.cmd[4]}, ${s.cmd[13]}]`);
OUT.phaseS1 = {
  file: servoPath, servo: SERVO, tuck_s: TUCK,
  timed: { x_mm: +mm(A.scored.x), z_mm: +mm(A.scored.z), above_mm: +mm(A.scored.above), feetOnTread: A.scored.feetOnTread, up: A.scored.up, honest: A.crit.honest, reward: A.reward, uprightTailTicks: A.uprightTailTicks, minPenetrationEpisode_mm: +mm(A.minPenetrationEpisode) },
  servoed: { x_mm: +mm(B.scored.x), z_mm: +mm(B.scored.z), above_mm: +mm(B.scored.above), feetOnTread: B.scored.feetOnTread, up: B.scored.up, honest: B.crit.honest, reward: B.reward, uprightTailTicks: B.uprightTailTicks, minPenetrationEpisode_mm: +mm(B.minPenetrationEpisode) },
  armed: B.servo.armed, tArm: B.servo.tArm, servoTicks: B.servo.ticks,
  firstDivergingTick: firstDiverge, peakDivergence_mm: { trunkX: +maxDX.toFixed(1), trunkZ: +maxDZ.toFixed(1), lfootZ: +maxDF.toFixed(1) },
  servoTrace: B.servo.trace,
};

// ==================================================================== PHASE S2
// THE POINT OF THE LEVER. A clock cannot see the rise. The servo's very first
// reading is the trunk's height above the tread, so the SAME file commands
// DIFFERENT leg targets on a 50, 60 and 70 mm flight.
log('');
log('PHASE S2 — the same servo file on 50 / 60 / 70 mm: does the command change with the rise?');
OUT.phaseS2 = [];
for (const rmm of [50, 60, 70]) {
  const r = await scoreSaved(servoPath, { rise: rmm / 1000, tail: 'policy', overrides: { servoTrace: true } });
  const t = await scoreSaved(BASE, { rise: rmm / 1000, tail: 'policy' });
  const first = r.servo.trace[0], last = r.servo.trace[r.servo.trace.length - 1];
  log(`   rise ${rmm}mm  servoed x=${mm(r.scored.x)} z=${mm(r.scored.z)} honest=${r.crit.honest} | timed x=${mm(t.scored.x)} z=${mm(t.scored.z)} honest=${t.crit.honest}`);
  log(`      first servo tick above=${first.above_mm}mm -> hipPitch=[${first.cmd[2]}, ${first.cmd[11]}] knee=[${first.cmd[3]}, ${first.cmd[12]}]`);
  log(`      last  servo tick above=${last.above_mm}mm -> hipPitch=[${last.cmd[2]}, ${last.cmd[11]}] knee=[${last.cmd[3]}, ${last.cmd[12]}]`);
  OUT.phaseS2.push({ rise_mm: rmm,
    servoed: { x_mm: +mm(r.scored.x), z_mm: +mm(r.scored.z), honest: r.crit.honest, feetOnTread: r.scored.feetOnTread, reward: r.reward },
    timed: { x_mm: +mm(t.scored.x), z_mm: +mm(t.scored.z), honest: t.crit.honest, feetOnTread: t.scored.feetOnTread, reward: t.reward },
    firstCmd: { above_mm: first.above_mm, hipPitch: [first.cmd[2], first.cmd[11]], knee: [first.cmd[3], first.cmd[12]], ankle: [first.cmd[4], first.cmd[13]] },
    lastCmd: { above_mm: last.above_mm, hipPitch: [last.cmd[2], last.cmd[11]], knee: [last.cmd[3], last.cmd[12]], ankle: [last.cmd[4], last.cmd[13]] } });
}

// ==================================================================== PHASE N
// THE NEW HOLE, CLOSED. minPenetrationEpisode for every published clear, over
// the whole 14-cell grid. -15 mm is the line the brief named.
log('');
log('PHASE N — whole-episode penetration on every published clear (14-cell grid, every tick of every cell)');
const PUBLISHED = [
  ['best_r2_vault_40mm.json', 0.040], ['best_r2_vault_60mm.json', 0.060],
  ['best_r3_vault_40mm.json', 0.040], ['best_r3_vault_50mm.json', 0.050],
  ['best_r3_vault_60mm.json', 0.060],
];
OUT.phaseN = [];
for (const [f, h] of PUBLISHED) {
  const g = await scoreRobust(P + f, { rise: h });
  const cells = g.verdicts.map(v => ({ ...v }));
  const clears = cells.filter(c => c.honest);
  const worst = Math.min(...cells.map(c => c.minPenetrationEpisode_mm));
  const worstClear = clears.length ? Math.min(...clears.map(c => c.minPenetrationEpisode_mm)) : null;
  const deep = cells.filter(c => c.minPenetrationEpisode_mm < -15);
  const deepClears = clears.filter(c => c.minPenetrationEpisode_mm < -15);
  log(`   ${f.padEnd(28)} sha ${g.move}  kCore=${g.kCore}/9 kCoreStable=${g.kCoreStable}/9 kExt=${g.kExt}/14`);
  log(`      worst minPenetrationEpisode over all 14 cells : ${worst.toFixed(2)} mm   (at-score worst ${g.agg.minPenetrationAtScore_mm.toFixed(2)} mm)`);
  log(`      worst over the CLEARED cells                  : ${worstClear === null ? 'n/a (no clear)' : worstClear.toFixed(2) + ' mm'}`);
  log(`      cells deeper than -15 mm: ${deep.length}/14   of which CLEARS: ${deepClears.length}`);
  for (const c of clears)
    log(`      CLEAR rise=${c.rise_mm} drop=${c.drop} f=${c.fmul} stable=${c.stableClear} atScore=${c.penetrationAtScore_mm}mm EPISODE=${c.minPenetrationEpisode_mm}mm (tick ${c.minPenetrationTick}, ${c.minPenetrationPair})`);
  OUT.phaseN.push({ file: f, rise_mm: h * 1000, sha256: intentHashOfFile(P + f), move: g.move,
    kCore: g.kCore, kCoreStable: g.kCoreStable, kExt: g.kExt,
    worstEpisode_mm: +worst.toFixed(2), worstAtScore_mm: +g.agg.minPenetrationAtScore_mm.toFixed(2),
    worstClearEpisode_mm: worstClear === null ? null : +worstClear.toFixed(2),
    cellsDeeperThanMinus15: deep.length, clearsDeeperThanMinus15: deepClears.length,
    clears: clears.map(c => ({ rise_mm: c.rise_mm, drop: c.drop, fmul: c.fmul, stable: c.stableClear,
      penetrationAtScore_mm: c.penetrationAtScore_mm, minPenetrationEpisode_mm: c.minPenetrationEpisode_mm,
      minPenetrationTick: c.minPenetrationTick, minPenetrationPair: c.minPenetrationPair })),
    allCells: cells.map(c => ({ rise_mm: c.rise_mm, drop: c.drop, fmul: c.fmul, honest: c.honest,
      penetrationAtScore_mm: c.penetrationAtScore_mm, minPenetrationEpisode_mm: c.minPenetrationEpisode_mm,
      minPenetrationPair: c.minPenetrationPair })) });
  log(`      [${el()}]`);
}

OUT.killGate = 'KILL GATE: if no move reaches kCoreStable >= 7 of 9 at 60 mm, the 40-80 mm band is CLOSED and the negative is the result. Round 5 added an instrument; it published no move that reaches it.';
fs.writeFileSync(P + 'r5_servo-results.json', JSON.stringify(OUT, null, 2));
fs.writeFileSync(P + '_r5_servo_demo.log', LOG.join('\n') + '\n');
log('');
log(`wrote climb/r5_servo-results.json   [${el()}]`);
