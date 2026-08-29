// TAKE_A_BOW candidate — see zz_celebration_lib.mjs for the harness.
import { verify } from './zz_celebration_lib.mjs';
const FRAMES = [
  [0.0, {}],
  [0.7, { 5: 0.55, 6: 0.45 }],   // neck_pitch +0.55, head_pitch +0.45
  [1.5, { 5: 0.55, 6: 0.45 }],   // hold the bow
  [2.2, {}],
];
const r = await verify('TAKE_A_BOW', FRAMES, 2.4, 1);
if (r.pass < 14 || r.peakRate >= 13) await verify('TAKE_A_BOW', FRAMES, 2.4, 0.5);
