# dumpparams — a policy's canonical parameter bytes, out of duckkit

WHY IT IS SWIFT AND NOT JAVASCRIPT. `sim/policyforward.mjs` runs a policy in a
browser by reading `DuckPolicy.canonicalParameterBytes` — normalizer mean, then
standard deviation, then for each layer outermost-first its weights and then its
biases, every value a little-endian float32. That layout is duckkit's
definition of what a policy IS, and it is what `DuckEvidence` fingerprints. A
second implementation of it — a JavaScript ONNX reader guessing at initializer
order — would be a second opinion about the contract, and the first time the two
disagreed the phone would run a different network from the one the app attested
to. This is the contract, executed.

    export PATH=$HOME/swift-6.3.3/usr/bin:$PATH SWIFT_BACKTRACE=enable=no
    cd sim/dumpparams && swift build -c release
    .build/release/dumpparams ../ ../params

It writes `sim/params/<name>.bin`, 791,584 bytes each (197,896 float32), and
prints the layer widths it found so a file with a different architecture is
visible rather than silently reshaped. `policy_parity.mjs` then proves each one
against onnxruntime.

The package depends on duckkit by absolute path, which is the one thing here
that has to be edited on another machine.
