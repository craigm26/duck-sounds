# Duck Sounds — plan

## The shape of the thing

One screen. A duck on your floor, seven buttons across the bottom, and nothing
else on screen unless you go looking for it. Every tap is a performance; two of
the seven are held.

The unusual bit is that the duck is driven by the same pipeline the real robot
runs, at the same rate, from the same weights — so the app is a genuine preview
of a $399 robot rather than a cartoon of one. That is the reason it is worth a
weekend and the reason it is worth a download.

## Architecture

### The line between the two repos

**If it can be tested on the Pi, it does not live in this repo.** DuckKit is pure
Swift and `swift test`s on Linux aarch64; this repo is the parts that need a phone.
The practical effect is that `duck-sounds` has almost no unit tests of its own,
because it has almost no logic of its own, and the things that would be hardest to
debug on a device (the synth, the choreography, the timestep) are already green
before the Mac is ever touched.

```
DuckKit (../duckkit, public, Apache-2.0)
  DuckModel, DuckObservation, DuckPolicy, DuckGait,
  DuckKinematics, DuckSimulation            already written, 848 tests green
  + DuckSound          the seven tags: robot.sound name, one-shot vs held,
                       nominal duration, hold cadence, decay timeout
  + DuckVoice          procedural synth -> [Float] @ 48 kHz, deterministic per
                       seed, plus the amplitude envelope for haptics
  + DuckPerformance    tag + elapsed -> DuckCommand + mouth fraction, as a
                       keyframe timeline with the robot's hold/decay rules
  + DuckClock          fixed 50 Hz accumulator with a catch-up clamp
  + DuckBeak           the beak pivot frame — OURS, because upstream has none
  + DuckRPC            (M3) NDJSON JSON-RPC 2.0 codec, transport-free

duck-sounds (private)
  DuckStage.swift      ARView, plane raycast, 15 ModelEntities + beak, the
                       50 Hz step and the render-frame interpolation
  DuckBody.swift       the fifteen primitives and their materials
  VoiceEngine.swift    AVAudioEngine, preloaded PCM buffers, start/loop/end
  Haptics.swift        CHHapticEngine, patterns built from DuckVoice envelopes
  SoundboardView.swift the seven buttons, tap and hold gestures, drag steering
  Mood.swift           lastSeenAt, time of day, what the duck does on launch
  Calls.swift          (M2) record / name / replay a sequence of taps
  DuckLink.swift       (M3) NWConnection to host:7788, DuckRPC over it
  QuackIntent.swift    (M2) App Intent + widget
```

### The pipeline, one tap

```
tap "inquire"
  -> DuckPerformance.timeline(.inquire)          keyframes, 0.7 s
  -> VoiceEngine plays DuckVoice.render(.inquire, seed: random)
  -> Haptics plays a pattern built from that render's envelope
  -> each 20 ms tick:
       command  = timeline.command(at: t)        head roll to +0.35, neck to -0.2
       tick     = simulation.step(command:)      alpha_stand, 61 floats, ~40 us
       simulation.setMouth(open: timeline.mouth(at: t))
  -> each display frame:
       angles = lerp(previousTick, latestTick, alpha)
       poses  = DuckKinematics.bodyPoses(jointAngles: angles)
       fifteen entity transforms, plus DuckBeak for the beak
```

Two details in there matter more than they look.

**The display runs at 60 or 120 Hz and the control loop runs at 50 Hz.** Stepping
the sim once per display frame runs the gait 20% fast on a 60 Hz phone and 140%
fast on a ProMotion one. `DuckClock` accumulates real time and emits whole 20 ms
steps, with a clamp so a backgrounded app does not return and fire two hundred
ticks at once. Display frames interpolate between the last two ticks.

**The head goes in the command block, not on top of the output.** DuckObservation's
own doc calls this out: head targets ride in the 13-value command and are *not*
added to the policy's output afterwards, because doing both bends the head twice.
So `DuckPerformance` sets `DuckCommand.head` and lets the policy produce the head
joints, exactly as `robot.head` does on the hardware.

That is the correct thing and it carries a real unknown: **we do not know how much
head excursion the standing policy actually produces for a commanded step.** If it
damps the command heavily, `inquire`'s tilt will not read and the performances
will be limp. So M1-3 is a measurement, not a guess: step `alpha_stand` for 0.5 s
with `head = (-0.5, 0, 0, 0)` and record the peak `neck_pitch` joint angle. If the
answer is small, we add a direct-drive head layer that composes policy legs with
performance-driven head joints, filtered at the trained α=0.5 and clamped to the
model's travel — and a comment saying which measurement forced it. Either way the
number goes in the test name.

### Rendering with no meshes

`robot_walk.xml` contains 14 hinges, named sites, inertial frames, and **zero
`<geom>` elements**. There is no duck geometry to load. Two consequences:

- The body is fifteen RealityKit primitives, one per MJCF body, each parented to
  the world and transformed every frame from `DuckKinematics.bodyPoses`. Positions
  are exact; sizes come from each body's `<inertial pos=... mass=...>` — the mass
  gives a plausible extent, the `pos` gives where the mass actually sits. No
  skinning, no rig file, no artist.
- The beak has no hinge upstream, so `DuckBeak` in DuckKit derives one from the
  `mouth_tip` and `head_camera` sites on `bottom_head_shell` and rotates it by
  `DuckModel.mouthTarget(open:)` over the −5°..+30° travel. It is labelled as ours
  in the file, because it is.

Default scale is 1.0 = actual size, 25 cm. Pinch scales 0.5×–3× with "actual size"
called out at 1.0, because a real-size duck three metres away is genuinely small
and people will want to make it big.

### Audio

`DuckVoice` renders mono `[Float]` at 48 kHz. Held tags render as three buffers
(start, loop, end). All seven are rendered at launch — 7 × ~1 s × 48 000 × 4 B is
about 1.3 MB and a few milliseconds of arithmetic — so a tap is a `scheduleBuffer`,
not a synthesis.

Each utterance takes a random seed and detunes ±4% with a little amplitude jitter,
so tapping `chirp` eight times does not sound like eight copies of one file. Tests
pass a fixed seed and hash the buffer.

Session category `.playback` with `.mixWithOthers`: a soundboard that goes silent
because the ringer switch is flipped reads as a broken app, and one that stops the
user's podcast reads as a rude one. This picks the first complaint over the second,
deliberately.

Haptics come off the same envelope the audio was rendered from, so the honk's
transient and the buzz in your hand are literally the same curve. `alarm` is one
sharp transient at full intensity; `coo` is a continuous low rumble modulated at
the breathing rate for as long as you hold it.

### When the camera is refused

AR is a way to show the duck, not the app. Decline camera access, or run in the
Simulator, and the same RealityKit scene renders in an `ARView` with
`cameraMode = .nonAR` and a perspective camera on a plain backdrop — duck on a
table, all seven sounds, everything works. This also means the MacInCloud
simulator build exercises the real render path, which matters a lot when the
operator has no local Mac.

## Milestones

### M0 — an hour, before any Swift

1. Check whether Pollen's sound WAVs are in `pollen-robotics/microduck` and under
   Apache-2.0. If yes, vendoring them replaces most of `DuckVoice`'s reason to
   exist as the *default* voice (keep it as the offline fallback). If no, open the
   issue asking, and carry on.
2. Register `com.ducksounds.ios` on the WYGG3JXWMG team; reserve the App Store
   name.
3. Stand up `duck.craigm26.com` as a Cloudflare Pages project with Web Analytics
   on — one page, what a Microduck is, a link to Pollen, a link to DuckKit. It is
   the awareness landing page and the GATES denominator. A bought domain waits
   until the 60-day gate passes.
4. Add `ducksounds` to the existing golinks worker so the in-app "what is a
   Microduck?" link is counted server-side, counts only, no cookies, no IPs.

### M1 — the weekend. Target: TestFlight by Sunday night.

DuckKit:
- `DuckSound`, the seven tags with hold semantics.
- `DuckVoice` + `DuckVoiceTests` — determinism, duration, no NaN, peak < 1.0,
  golden hash per tag at seed 1.
- `DuckPerformance` + tests — every timeline stays inside joint travel, held tags
  decay to rest within their end segment, `wheee` crosses the 0.05 twist threshold
  and `chirp` never does.
- `DuckClock` + tests — 3 ticks in 61 ms, 0 in 19 ms, clamp at 10 after a 5 s stall.
- ~~`DuckBeak` + tests~~ — **not built, and not needed as a type.** The mouth is
  joint 9 (`DuckModel.mouthIndex`), `DuckKinematics` already returns the
  `mouth_tip` site, and `DuckPerformance` already decides how far open the beak
  is at each keyframe. The assertion this line wanted — closed and open beak
  tips differ by the expected arc — is writable against those three today; a
  fourth type would only wrap them.
- ~~Tag `0.2.0`~~ — shipped as **`v1.0.0`**. duckkit went out as a public,
  two-product package rather than a 0.x sibling, so the version reflects a
  published API rather than a private one. Depend on it with
  `.package(url: "https://github.com/craigm26/duckkit.git", from: "1.0.0")`.

App:
- `project.yml`, `PrivacyInfo.xcprivacy`, icon, launch screen.
- Vendor `alpha_walking.onnx` and `alpha_stand.onnx` into `Resources/policies/`
  with NOTICE.
- `DuckBody` — fifteen primitives, materials, the beak.
- `DuckStage` — non-AR scene first (it runs in the Simulator), then AR on top.
- `VoiceEngine`, `Haptics`.
- `SoundboardView` — seven buttons, tap and press-and-hold, drag steering on `wheee`.
- Petting: touch and hold the duck itself → `coo`, head leans toward the touch.
- About screen: the honest voice paragraph, the not-affiliated line, the golinks
  link out.

Done when: `swift test` green on the Pi, `mac_build.py` green, and the operator
can hold a phone, tap seven buttons, and want to do it again.

### M2 — the second-launch layer. Week 2.

The product question this app exists to answer is whether a soundboard earns a
second launch. M2 is the set of answers that cost nothing to run.

- **Mood.** `lastSeenAt` in UserDefaults. Away more than 12 hours → the duck
  greets you with the double wak-wak. Local time 21:00–06:00 → it starts drowsy,
  head low, and coos instead. Backgrounding → the `peck` goodbye tock. No
  notifications, no server, no permission. This is the mechanism most likely to
  actually work, because it turns the app from a menu into a pet.
- **Widget + App Intents.** An interactive WidgetKit widget with the seven
  buttons (iOS 17 `Button(intent:)`), a `QuackIntent` exposed to Shortcuts and the
  Action Button, and a Siri phrase. v1 uses `openAppWhenRun: true` — playing audio
  from a non-opening intent is a thing we have not verified, and a widget button
  that opens the app is still a launch.
- **Calls.** Record up to 20 taps over 30 seconds, name it, replay it. Stored as a
  small JSON file. Share sheet posts the name plus a `duck.craigm26.com/?from=share`
  link. Encoding the call into the link so the web page can play it back is M3.
- **Rate prompt.** `SKStoreReviewController` once, after the fifth distinct
  session, never again.

### M3 — the duck arrives. December 2026.

- `DuckRPC` in DuckKit — NDJSON JSON-RPC 2.0 framing, request ids, notification
  encoding, `robot.state` decoding. Transport-free so it tests on the Pi against
  a recorded stream.
- `DuckLink` — `NWConnection` to `host:7788`, the deadman cadence (held sounds at
  10 Hz, head/move at 25 Hz — half the control rate, still five deadman periods of
  margin), reconnect, and a battery readout that says "unknown" rather than a
  number when the bus does not answer.
- Pairing UI, the bridge instructions, and the unauthenticated-LAN warning.
- The `wheee`-does-not-walk rule and the explicit override toggle.
- "Teach me your voice" — record the real bank through the mic, on-device only.
  `NSMicrophoneUsageDescription` and `NSLocalNetworkUsageDescription` ship **in
  this milestone and not before**: a purpose string for an API the binary does not
  call is how Heat Compass drew ITMS-90683.

## Task list

```
M0-1  check Pollen sound-asset licence; issue if unclear                   [blocker for scope]
M0-2  register com.ducksounds.ios, reserve App Store name
M0-3  duck.craigm26.com Pages project + Web Analytics
M0-4  golinks: add app=ducksounds

M1-1  DuckKit: DuckSound + tests
M1-2  DuckKit: DuckVoice + DuckVoiceTests + tools/render_voices.py        [ear test = go/no-go]
M1-3  DuckKit: measure alpha_stand head excursion for a commanded step    [decides M1-4]
M1-4  DuckKit: DuckPerformance, seven timelines + tests
M1-5  DuckKit: DuckClock + tests
M1-6  DuckKit: DuckBeak + tests
M1-7  DuckKit: tag 0.2.0
M1-8  app: project.yml, PrivacyInfo.xcprivacy, Info.plist, icon
M1-9  app: vendor alpha_walking.onnx + alpha_stand.onnx + NOTICE
M1-10 app: DuckBody, fifteen primitives from the MJCF inertial frames
M1-11 app: DuckStage non-AR path; green mac_build.py
M1-12 app: DuckStage AR path, plane raycast, tap to place, pinch to scale
M1-13 app: VoiceEngine, preloaded buffers, start/loop/end
M1-14 app: Haptics from DuckVoice envelopes
M1-15 app: SoundboardView, tap + hold + drag steering
M1-16 app: petting gesture on the duck
M1-17 app: About screen, honest-voice paragraph, not-affiliated line
M1-18 device smoke on a real iPhone; TestFlight build 1
M1-19 screenshots, ASC metadata, App Privacy = Data Not Collected, submit

M2-1  Mood: lastSeenAt, time of day, greet/coo/peck
M2-2  QuackIntent + Shortcuts + Action Button
M2-3  interactive widget, seven buttons
M2-4  Calls: record, name, replay, share
M2-5  rate prompt after 5th session

M3-1  DuckKit: DuckRPC + tests against a recorded NDJSON stream
M3-2  DuckLink: NWConnection, cadence, reconnect, honest battery
M3-3  pairing UI + bridge docs + LAN warning
M3-4  route sounds to robot.sound; ghost mirrors
M3-5  wheee safety rule + explicit override
M3-6  teach-me-your-voice, on-device only
```

## Decisions already made, so they are not re-argued

- **No crypto.** No swift-crypto, no Journal, no CanonicalJSON, no signing key
  store. This app has nothing to attest and a soundboard that links BoringSSL to
  hash a recording of taps is a worse app. DuckKit stays dependency-free; if
  signing is ever needed for another duck app it goes in a separate `DuckAttest`
  target so soundboards do not inherit it.
- **No LiDAR gate.** `PhoneSight` stays where it is. Plane detection works on
  every iOS 17 iPhone and the point of this app is that everyone can run it.
- **No physics.** `DuckSimulation` is honest about having none and that is the
  correct amount for a ghost on a carpet.
- **No server, ever.** Nothing in this app makes an outbound request except the
  user tapping a link in Safari.
- **The ONNX files ship in the app, not the library.** DuckKit currently carries
  the policy weights as a *test fixture*, which the app cannot reach. The app
  vendors its own copies (1.6 MB, Apache-2.0). When a second duck app needs the
  same two files, DuckKit gains a `DuckPolicyBundle` resource target and both
  switch to it — not before.
