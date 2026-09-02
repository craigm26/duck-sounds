// The stair bank lives in site/stairs.js. This file is the name the bench core
// and the shared climb scorer need it under.
//
// WHY IT IS A RE-EXPORT AND NOT A COPY — the same reason duckloop.mjs is one.
// `climb_score.mjs` must import the staircase by a path that resolves BOTH in
// Node, next to the core in sim/, and in a browser, next to the core in
// /assets/, where there is no `../site` to reach into. `./stairs.js` is that
// path, and in Node it lands here. The phone bundle copies site/stairs.js in
// under this name (scripts/make_phonebench.sh, duck-studio's
// scripts/make_phone_bench.sh), so both shells run the SAME flight: the same
// STAIR_Y, the same 340 mm width, the same step-step isolation. A second copy
// of those constants is a second staircase, and the one that goes stale
// silently is the one nobody reads.
export * from '../site/stairs.js';
