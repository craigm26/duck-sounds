// HEAD_SHAKE candidate — see zz_celebration_lib.mjs for the harness.
import { verify } from './zz_celebration_lib.mjs';
const FRAMES = [
  [0.0, {}],
  [0.4, { 7: 0.8 }],
  [0.9, { 7: -0.8 }],
  [1.4, { 7: 0.5 }],
  [1.8, {}],
];
const r = await verify('HEAD_SHAKE', FRAMES, 2.0, 1);
if (r.pass < 14 || r.peakRate >= 13) await verify('HEAD_SHAKE', FRAMES, 2.0, 0.5);
