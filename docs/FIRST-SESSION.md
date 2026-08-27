# The first build session

One sitting. What to write, and what proves it worked.

Write DuckSound and DuckVoice in DuckKit, and listen to the result. Nothing else.

WHAT TO WRITE. `Sources/DuckKit/DuckSound.swift`: the seven tags as an enum with the robot.sound tag string, one-shot vs held, nominal duration, the 100 ms hold-resend cadence against the ~0.5 s deadman, and a one-line character description per tag. Then `Sources/DuckKit/DuckVoice.swift`: a procedural duck-call synthesizer in pure Swift — a pulse oscillator with a breakpoint pitch contour, a two-pole formant bandpass, a seeded-noise breath bed, and a per-tag amplitude envelope, rendering mono [Float] at 48 kHz. Seven parameter sets, one synth. Held tags (coo, wheee) render as three buffers: start, loop, end. The noise generator takes an explicit seed so the output is byte-identical for a fixed one. Expose the amplitude envelope alongside the samples, because the haptics will be built from exactly that curve. Then `Tests/DuckKitTests/DuckVoiceTests.swift` in the house style: testEverySoundRendersItsNominalDurationAtFortyEightKilohertz, testTheRenderedBufferNeverGoesNonFiniteOrClipsFullScale, testTheSameSeedRendersByteIdenticalSamples, testHeldSoundsRenderAStartALoopAndAnEndSegment, testTheLoopSegmentJoinsToItselfWithoutADiscontinuity. Finally `tools/render_voices.py`, which shells `swift test` with an env var that dumps all seven to WAV in `tools/out/`.

WHAT PROVES IT. Two gates, and the second one is the real one.

The machine gate: `/home/craigm26/swift-6.3.3/usr/bin/swift test` green in /home/craigm26/projects/duckkit, all 848 existing tests still passing, new tests passing, on the Pi, with no Mac and no phone involved.

The ear gate: seven WAV files, played by a human, who then answers one question — "is that a duck?" The tests can prove duration, determinism, no NaN and no clipping. They cannot prove charm, and charm is the entire product. There is no measured spectrum of Pollen's bank in this repo because we have never heard it, so the parameters are chosen by ear and the acceptance is an ear.

WHY THIS SESSION AND NOT THE AR VIEW. If the voice is not charming there is no app — an AR duck that opens its beak in silence, or in a sound people wince at, is a tech demo with a beak. Every other unknown in this build (does the standing policy track head commands, do fifteen primitives read as a duck, does socat bridge robotd) can be discovered later and worked around. This one cannot: it is upstream of the reason anyone downloads the thing, and it costs one session to find out. If the answer is no, the recovery is cheap and known — M0-1 (check whether Pollen's WAVs are Apache-2.0 and redistributable) becomes the whole plan, and the app ships silent-until-vendored rather than shipping something that sounds bad. Finding that out on Saturday morning is a weekend; finding it out after the AR view is built is a wasted one.
