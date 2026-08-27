# Duck Sounds

Seven duck calls. Tap one and a 25 cm robot duck standing on your floor performs
it — the honk, the tilt, the peck, the held joy ride. Free, no ads, no accounts,
no data collected. When a real [Microduck](https://github.com/pollen-robotics/microduck)
turns up on your Wi-Fi, the sound comes out of the duck instead of the phone.

The duck is not an animation. Its legs are driven by Pollen's own trained
`alpha_walking.onnx` and `alpha_stand.onnx` running at 50 Hz on the phone,
through the same observation vector, the same action scale, and the same
trained-in low-pass filter the real robot's control loop uses. Its joints are
placed by forward kinematics over the robot's own MuJoCo model. It is 25 cm tall
on your floor because the model says the head camera sits 0.244 m up, not because
someone eyeballed it. All of that comes from [DuckKit](https://github.com/craigm26/duckkit),
which is tested on a Raspberry Pi with no phone in the room.

Nobody has the hardware yet — first deliveries are around Christmas 2026. This
app has to be worth opening before then, and that is the whole design brief.

## The seven

| Tag | What it is | What the duck does |
|---|---|---|
| `alarm` | a sharp honk | snaps upright, head thrown back, beak open 120 ms, one yaw twitch after |
| `greet` | the wake-up quack, sometimes a double wak-wak | finds your camera, two beak pulses, then a nod |
| `inquire` | a rising question | head **tilts** to 0.35 rad of roll and holds there, beak open at the top of the rise |
| `peck` | a low tock, the goodbye before power-off | drives the head down to the floor, beak snaps on the tock, head comes halfway back |
| `chirp` | its quack, the plain one | a small beak pop and a body bob, and deliberately nothing else |
| `coo` | drowsy and content — the **petting** response | held. Head sinks, leans toward your finger, breathes at ~0.6 Hz |
| `wheee` | the held joy ride: start → loop → end | held. Winds up, then **actually runs** across your floor; drag to steer |

`coo` and `wheee` are the two that make this an instrument rather than a menu.
Press and hold `wheee` and the walking policy engages — twist crosses the 0.05
threshold, `alpha_walking` takes over from `alpha_stand`, and the duck runs. Let
go and it decays through an end segment, exactly the way the robot's own `wheee`
decays when the holds stop arriving. That decay rule is not decoration; it is the
robot's hold semantics, implemented once in DuckKit so the ghost and the hardware
behave like the same animal.

## About the voice: this is our impression, not the duck's

**Pollen's sound bank lives on the robot, not in this repo.** `robot.sound
{tag, hold}` plays a file that exists on the duck's own filesystem. We have not
verified whether those WAVs ship in the Apache-2.0 repo and are redistributable,
and we are not going to pretend otherwise.

So the phone renders its own. `DuckVoice` is a procedural duck-call synthesizer in
pure Swift: a pulse oscillator with a pitch contour, a formant bandpass, a noise
bed for the breath, and a per-tag amplitude envelope, at 48 kHz. Seven parameter
sets, one synth. It has no assets, works offline, and is deterministic given a
seed — the tests hash the rendered buffer, so a change to the voice shows up in
review as a changed hash rather than as a surprise on someone's phone.

What the tests can prove: right duration, no NaN, peak below full scale, byte-identical
for a fixed seed. What they cannot prove is that it sounds like a duck. That is an
ear test, run by a human, and it is the gate on the whole app — see PLAN.md §First
session. Parameters were chosen by ear. There is no measured spectrum of Pollen's
bank here because we have never heard it.

Three ways to close the gap, in the order we will try them:

1. **Check the licence.** If the WAVs are in `pollen-robotics/microduck` under
   Apache-2.0, vendor them, ship them as the default voice, keep the synth as the
   fallback. This is a twenty-minute check and it is task M0-1.
2. **Ask.** If they are not in the repo, open an issue asking whether the bank may
   be redistributed. Cost: one polite paragraph.
3. **Record it.** When hardware arrives, "Teach me your voice" records the real
   duck through the phone's microphone and stores it **on the phone only**. It is
   never uploaded — that would be data collection and this app does not do that.
   Any public voice dataset comes from the operator's own duck, recorded by the
   operator, published by hand.

Until then the app says so, in plain words, on the About screen: *this is the
phone's impression of the duck; the real voice lives on the robot.*

## What is real and what is not

Real: the policies, the 61-float observation, the action scale, the low-pass
coefficients, the joint travel limits, the kinematic chain, the 25 cm scale, the
50 Hz tick, the hold-and-decay semantics.

Not real: physics. `DuckSimulation` has none — `isGrounded` is a constant `true`
and joint velocity is estimated by differencing targets. The ghost walks the way
the network says to and cannot tip over. That is the right amount of simulation
for a duck on your carpet and the wrong amount for predicting whether the real
robot clears a step.

Also not real: the duck's *appearance*. `robot_walk.xml` is a kinematics-only
model — 14 hinges, named sites, inertial frames, and **zero `<geom>` elements**.
There are no meshes to render. So the body you see is fifteen primitives, one per
MJCF body, placed at the exact frames the model declares and sized from each
body's inertial mass and centre offset. The *pose* is upstream's; the *shape* is
ours. If Pollen's meshes turn out to be usable, swapping them in changes the
render and not one line of the pose pipeline.

And the beak: `robot_walk.xml` has no mouth hinge at all (the mouth is joint index
9 of 15 on the wire, excluded from every policy, and absent from the kinematic
model). Its pivot frame is derived by us from the `mouth_tip` and `head_camera`
sites on `bottom_head_shell` and lives in `DuckBeak`, clearly marked as ours.

## Pairing with a real duck

`robotd` listens on a Unix socket, `/run/robotd.sock`, and a phone cannot open a
Unix socket on another machine. One line on the duck bridges it:

```bash
socat TCP-LISTEN:7788,fork,reuseaddr UNIX-CONNECT:/run/robotd.sock
```

**That bridge is unauthenticated. Anyone on your Wi-Fi can drive your duck.** The
app says this before it will connect, the docs say it, and pairing is off by
default. A persistent systemd unit is in `docs/bridge.md` for people who accept
that trade on a home network.

When paired, taps route to `robot.sound {tag}` on the robot and the phone stays
quiet, while the AR ghost mirrors the same performance so you can see what you
just asked for. Held sounds resend `hold` every 100 ms, comfortably inside the
~0.5 s deadman, so one dropped Wi-Fi frame does not cut a `wheee` short.

One safety rule, and it is not configurable by accident: **`wheee` does not walk a
real duck.** An 800 g biped that starts running on your desk during a joy ride
walks off the edge of it. Paired `wheee` plays the sound and moves the head. There
is a separate toggle, off by default, behind an explicit warning, that lets the
twist through — for people standing over a clear floor who meant it.

## Privacy

Data Not Collected. Not "anonymised", not "aggregated" — none. There is no
analytics SDK, no first-party endpoint, no event stream, no crash reporter, no
per-user anything. `PrivacyInfo.xcprivacy` declares `NSPrivacyTracking=false`, an
empty `NSPrivacyTrackingDomains`, and an empty `NSPrivacyCollectedDataTypes`. The
only required-reason API declared is `UserDefaults` (`CA92.1`), for remembering
when the duck last saw you.

Camera frames are used to draw the duck on your floor and are never recorded and
never leave the device. Nothing needs the network. The app works in aeroplane mode.

How we know whether it worked, then: we count doors, never people. App Store
Connect's aggregate downloads and sessions, cookieless Cloudflare Web Analytics on
the docs page, a server-side click counter on the "what is a Microduck?" link, and
GitHub stars on DuckKit. All measured outside the app. See GATES.md.

## Building

There is no GitHub Actions workflow and there will not be one.

```bash
# logic tests — on the Pi, no Mac, no phone
cd ../duckkit && /home/craigm26/swift-6.3.3/usr/bin/swift test

# render the seven voices to WAV and listen to them
python3 tools/render_voices.py            # writes tools/out/*.wav

# compile the app — MacInCloud, simulator SDK
MAC_PASS=... python3 scripts/mac_build.py

# archive and upload to TestFlight
MAC_PASS=... python3 scripts/mac_archive.py && python3 scripts/mac_upload.py
```

The `.xcodeproj` is generated by xcodegen from `project.yml` and is gitignored.
The simulator build is a real gate here, not a formality: the non-AR fallback
scene renders in the Simulator, so a Mac with no phone attached still proves the
duck draws.

## Where code lives

**If it can be tested on the Pi, it does not live in this repo.** The synthesizer,
the seven performances, the fixed-timestep clock, the beak frame, and the JSON-RPC
codec are all in DuckKit, where `swift test` reaches them on Linux. This repo holds
ARKit, RealityKit, AVFoundation, CoreHaptics, SwiftUI and WidgetKit — the parts
that need a phone — and as little decision-making as we can manage.

## Licence and provenance

The app is proprietary; DuckKit is Apache-2.0, matching upstream. `Resources/policies/`
vendors `alpha_walking.onnx` and `alpha_stand.onnx` verbatim from
[pollen-robotics/microduck](https://github.com/pollen-robotics/microduck) (Apache-2.0),
with the NOTICE file alongside.

Duck Sounds is not affiliated with, endorsed by, or sponsored by Pollen Robotics.
"Microduck" is used to name the robot this app talks to and for nothing else.
