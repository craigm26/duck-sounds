// The control loop lives in site/duckloop.mjs. This file is the name the
// browser bundle needs it under.
//
// WHY IT IS A RE-EXPORT AND NOT A COPY. It WAS a copy — a second, older
// implementation of the same observation layout, action scaling and low-pass,
// carrying its own `HOME`, its own `buildObs` and, crucially, NO `reference`
// argument, so a policy that declared its own neutral pose would have been fed
// a deviation measured from the wrong one. Nothing imported it (checked across
// this repo on 2026-09-01: all forty-four importers say `../site/duckloop.mjs`),
// which is exactly how a stale near-duplicate survives — it is never wrong
// because it is never run, right up until somebody imports the wrong one.
//
// It is kept rather than deleted because `duckbench-core.mjs` must import the
// loop by a path that resolves BOTH in Node, next to the core in sim/, and in a
// browser, next to the core in /assets/. `./duckloop.mjs` is that path, and in
// Node it lands here.
export * from '../site/duckloop.mjs';
