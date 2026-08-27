# What this app needs from DuckKit

[DuckKit](https://github.com/craigm26/duckkit) is the shared pure-Swift core:
the joint tables, the 61-float observation contract, a hand-written ONNX reader
and MLP, the gait pipeline, forward kinematics and the 50 Hz loop. It has zero
package dependencies so it tests on Linux, and that claim is load-bearing.

This app needs the following. Each item is an API to add or a type to move --
resolve them in duckkit first, then depend on a tag.

1. DuckSound (new) — the seven tags as a type: robot.sound tag string, one-shot vs held, nominal duration, the ~0.5 s deadman and the 100 ms hold-resend cadence, and the one-line character description. Tiny, but it is the vocabulary every duck app shares and it must not be re-typed per app.

2. DuckVoice (new) — the procedural duck-call synthesizer. Pure Swift, no AVFoundation, renders mono [Float] at 48 kHz plus an amplitude envelope for haptics; held tags render as (start, loop, end). Deterministic given an explicit seed so tests can hash the buffer. Belongs in DuckKit because it is arithmetic, because the Pi is the only place it can be tested, and because it is the answer to a gap (Pollen's bank lives on the robot) that every duck app hits.

3. DuckPerformance (new) — tag + elapsed time -> DuckCommand + mouth fraction, as a keyframe timeline, with the robot's own hold-and-decay semantics implemented once. This is the file that makes the AR ghost and the real robot the same animal, and it is the single most valuable thing to have Linux-tested: every timeline can be asserted to stay inside joint travel and to cross (or not cross) the 0.05 twist threshold, with no phone in the room.

4. DuckClock (new) — a fixed 50 Hz accumulator with a catch-up clamp. Four lines of logic guarding a real bug: stepping DuckSimulation once per display frame runs the gait 20% fast on a 60 Hz phone and 140% fast on a 120 Hz one, and after a background/foreground stall it fires hundreds of ticks at once.

5. DuckBeak (new) — the beak's pivot frame, derived from the mouth_tip and head_camera sites on bottom_head_shell, rotated by DuckModel.mouthTarget(open:). This is an INVENTION and must be labelled as one in the file: robot_walk.xml has 14 hinges and no mouth joint, so unlike everything else in DuckKinematics there is no upstream number to check it against. Every duck app that draws a duck needs it; none of them should hardcode it privately.

6. DuckRPC (new, M3) — NDJSON JSON-RPC 2.0 codec: request/notification encoding, id management, line framing, robot.state decoding. Transport-free (Data in, Data out) so NWConnection stays in the app and the codec tests on the Pi against a recorded stream.

7. A shipping path for the policy weights. DuckKit currently carries alpha_walking.onnx only as a Tests fixture, which an app cannot reach — but DuckSimulation cannot be constructed without a DuckPolicy. For M1 the app vendors its own copies of alpha_walking.onnx and alpha_stand.onnx (1.6 MB, Apache-2.0, plus NOTICE). When a second duck app needs the same two files, add a DuckPolicyBundle resource target to DuckKit and switch both; do not do it for one consumer.

8. Nothing from CastorKit needs moving for this app, and that is a deliberate answer rather than an oversight: NOT CanonicalJSON, NOT the Journal, NOT Ed25519/swift-crypto, NOT SigningKeyStore, NOT PhoneSight, NOT DuckSoccerMatch. A soundboard has nothing to attest, and linking BoringSSL to hash a recording of button taps would make it a worse app. Keep DuckKit dependency-free; if signing is ever needed for another duck app, put it in a separate DuckAttest target so soundboard apps do not inherit it.

9. Observation, not a need: DuckSoccerMatch is listed as part of DuckKit but is not in Sources/DuckKit/ yet — it is still only in CastorKit. Not blocking here, but the extraction is incomplete and someone will trip over it.
