// SPIKE 1 — does turning robotd's two knobs move the duck further than the
// bench's own jitter?
//
// THE QUESTION, PRECISELY. `action_scale` ∈ {0.7, 0.8, 0.9, 1.0} crossed with
// the low-pass pair (head α, legs α) ∈ {(0.5, 0.7), (1, 1)}, running
// alpha_walking at vx = 0.5 m/s for six seconds, measuring how far the duck
// gets. A knob is worth searching over only if its effect is larger than the
// spread the bench produces when NOTHING changes.
//
// WHAT THE JITTER IS. Nothing in this rollout is random — MuJoCo is
// deterministic and this policy runs in float32 through the same code every
// time — so repeating one setting reproduces one number exactly, and a
// repeat-based error bar would be a row of zeroes and a lie. The bench's own
// answer to that is `/measure`'s randomised drop height, 0.12 m to 0.13 m,
// which is Pollen's range from `measure_success.mjs`. That is the jitter here
// too: every cell is sixteen episodes at sixteen drop heights evenly spaced
// across that range, reported as median and full spread. A knob effect smaller
// than that spread is not an effect this bench can see.
//
// THE CANON COLUMN IS NOT OPTIONAL. Scale 1.0 with no filter is what the bench
// itself runs and what every clip in duckkit was recorded from, so it is the
// row every other row has to be read against — and it is a real cell in the
// sweep, not a footnote: (1.0, no filter) IS the canon path.
import { makeRig, checkAgainstDuckloop, median } from './rollout_robotd.mjs';

const DROPS = 16;
const dropAt = i => 0.12 + (0.01 * i) / (DROPS - 1);

const fidelity = checkAgainstDuckloop();
console.log(`pipeline vs duckloop gaitTargets at robotd's own constants: worst |Δ| = ${fidelity}`);
if (fidelity !== 0) {
  console.error('REFUSING: this file models a different robot than the rest of the repo.');
  process.exit(1);
}

const rig = await makeRig();
console.log(`plant ${rig.plant}, ${rig.timestep} s timestep, ${rig.substeps} substeps `
          + `per ${rig.tickHz} Hz tick`);

const scales = [0.7, 0.8, 0.9, 1.0];
const filters = [
  { name: 'robotd 0.5/0.7', alphas: [0.5, 0.7], filter: true },
  { name: 'none (1.0/1.0)', alphas: [1, 1], filter: false },
];

const timing = {};
const rows = [];
const t0 = Date.now();
for (const f of filters) {
  for (const scale of scales) {
    const travels = [], forwards = [];
    let standing = 0;
    for (let i = 0; i < DROPS; i++) {
      const run = await rig.episode({ seconds: 6, drop: dropAt(i), vx: 0.5,
                                      scale, alphas: f.alphas, filter: f.filter, timing });
      travels.push(run.travelled);
      forwards.push(run.forwardTravel);
      if (run.standing) standing++;
    }
    rows.push({ filter: f.name, scale,
                median: median(travels), lo: Math.min(...travels), hi: Math.max(...travels),
                spread: Math.max(...travels) - Math.min(...travels),
                forward: median(forwards), standing });
    process.stderr.write('.');
  }
}
process.stderr.write('\n');
const seconds = (Date.now() - t0) / 1000;

const mm = v => (v * 1000).toFixed(0).padStart(6);
console.log('');
console.log(`travelled over 6 s at vx = 0.5 m/s, alpha_walking, `
          + `${DROPS} drop heights 0.120-0.130 m`);
console.log('');
console.log('filter          scale   median    min    max  spread  forward  stands');
for (const r of rows) {
  console.log(`${r.filter.padEnd(15)} ${r.scale.toFixed(1)} ${mm(r.median)} ${mm(r.lo)} `
            + `${mm(r.hi)} ${mm(r.spread)} ${mm(r.forward)}    ${r.standing}/${DROPS}`);
}
console.log('                        (mm)   (mm)   (mm)    (mm)     (mm)');
console.log('');

// The verdict, stated in the terms the question was asked in: is the biggest
// knob effect larger than the biggest jitter spread?
const worstSpread = Math.max(...rows.map(r => r.spread));
const medians = rows.map(r => r.median);
const knobRange = Math.max(...medians) - Math.min(...medians);
console.log(`largest drop-height spread within one setting: ${(worstSpread * 1000).toFixed(0)} mm`);
console.log(`range of medians across all eight settings:    ${(knobRange * 1000).toFixed(0)} mm`);
const spreads = rows.map(r => r.spread).sort((a, b) => a - b);
const typicalSpread = spreads[spreads.length >> 1];
console.log(`typical (median) drop-height spread within one setting: ${(typicalSpread * 1000).toFixed(0)} mm`);
console.log(knobRange > worstSpread
  ? 'VERDICT: the knob effect clears even the WORST cell\'s jitter.'
  : knobRange > typicalSpread
    ? 'VERDICT: the knob effect clears the TYPICAL cell\'s jitter but not the worst cell\'s — '
      + 'the response surface is not smooth in the knob.'
    : 'VERDICT: the knobs do NOT clear the bench\'s own jitter; any tuning signal here is noise.');
console.log('');

// HOW SENSITIVE IS ONE CELL TO NOTHING AT ALL? Two drop heights a tenth of a
// millimetre apart, one setting. If six seconds of walking separates on that,
// then a single episode is not a measurement of a policy, it is a measurement
// of a coin, and every number above has to be read as a median over drops
// rather than as a distance.
const probe = [];
for (const z of [0.12300, 0.12310, 0.12320]) {
  const run = await rig.episode({ seconds: 6, drop: z, vx: 0.5, scale: 0.9,
                                  alphas: [0.5, 0.7], filter: true });
  probe.push([z, run.travelled]);
}
console.log('one setting (robotd 0.5/0.7, scale 0.9), three drop heights 0.1 mm apart:');
for (const [z, d] of probe) console.log(`  drop ${z.toFixed(5)} m -> ${(d * 1000).toFixed(0)} mm`);
const probeSpread = Math.max(...probe.map(p => p[1])) - Math.min(...probe.map(p => p[1]));
console.log(`  spread ${(probeSpread * 1000).toFixed(0)} mm from a 0.2 mm change in where it was dropped`);
console.log('');
const perTick = ms => (ms / timing.ticks).toFixed(3);
console.log(`${timing.ticks} control ticks: physics ${perTick(timing.physicsNs / 1e6)} ms/tick, `
          + `policy ${perTick(timing.policyNs / 1e6)} ms/tick (this Pi, Node ${process.versions.node})`);
console.log(`wall time ${seconds.toFixed(1)} s for ${rows.length * DROPS} episodes of 6 s`);
