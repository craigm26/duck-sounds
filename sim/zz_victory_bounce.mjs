// VICTORY_BOUNCE candidate — see zz_celebration_lib.mjs for the harness.
import { verify } from './zz_celebration_lib.mjs';
const CROUCH = { 2: -0.25, 11: 0.25, 3: -0.2, 12: 0.2, 4: 0.15, 13: -0.15 };
const FRAMES = [
  [0.0, {}],
  [0.5, CROUCH],
  [0.9, {}],
  [1.3, CROUCH],
  [1.7, {}],
  [2.3, {}],
];
const r = await verify('VICTORY_BOUNCE', FRAMES, 2.6, 1);
if (r.pass < 14 || r.peakRate >= 13) await verify('VICTORY_BOUNCE', FRAMES, 2.6, 0.5);
