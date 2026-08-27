> Pre-registered decision gates for Duck Sounds. Written before launch, on
> purpose, so the decision later is a lookup rather than an argument. Clocks
> start at **App Store approval**; record the date here on the day it happens:
>
> **Approval date: ________________**

Duck Sounds is the awareness play. It is free, it took a weekend, and its job is
that people who did not know a $399 open-source robot duck exists now do. That is
a real job, but it is also the kind of job that is easy to declare successful
without evidence, which is why the numbers below are fixed now.

**Hard rule: Duck Sounds ships no in-app analytics.** No SDK, no first-party
endpoint, no event stream, no crash reporter, no per-user telemetry. The App
Privacy label is Data Not Collected and `PrivacyInfo.xcprivacy` declares an empty
`NSPrivacyCollectedDataTypes`. Every number below is a **$0 side-channel proxy**
measured outside the app. We count doors, never people.

---

## PASS — keep investing (evaluated at day 60 post-approval)

**All three** must hold.

- **≥ 1,000 first-time downloads.** App Store Connect → App Analytics, cumulative
  over the 60 days. Higher than OpenCastor's 300 on purpose: OpenCastor's audience
  is people who own a robot, and this one's audience is everybody. A general-audience
  free toy that cannot reach four figures did not do the awareness job, and
  awareness is the entire justification for the slot it took.
- **≥ 3.0 sessions per active device** over the 60-day window. App Store Connect
  App Analytics reports Sessions and Active Devices as Apple-provided aggregates —
  no SDK, no instrumentation, and it is the only honest answer available to the
  one product question this app was built to ask: *does a soundboard earn a second
  launch?* Three sessions per device means the median person came back more than
  once. Below that, it is a novelty.
- **≥ 120 counted taps through to "what is a Microduck?"** — the golinks D1 rollup
  for `app=ducksounds` (counts per app/src/day, no cookies, no IPs), cross-checked
  against Cloudflare Web Analytics on `duck.craigm26.com`. 120 against 1,000
  downloads is a 12% curiosity rate. This is the awareness conversion and the only
  number here that measures the *point* rather than the app.

A PASS means the weekend bought a real audience that was curious about the robot,
and M2/M3 are worth building.

## KILL — stop and archive (evaluated at day 90 post-approval)

**Either** triggers a kill.

- **< 300 downloads** over the 90 days, **or**
- **< 1.6 sessions per active device** over the 90 days — the median person opened
  it once and never came back.

A KILL means archive the repo, keep DuckKit (which is the durable asset and is
tested and public either way), keep the write-up, move on. Do not add analytics to
find out why. The answer to "why didn't they come back" is not worth the privacy
promise.

## The middle, which is the likely case

**Downloads good, sessions bad** is a *pivot* signal, not an automatic anything.
It means the app is findable and charming for ten seconds and has no reason to
exist on day two. The response is exactly one intervention — ship M2 (mood, widget,
App Intents, Calls) — and one re-measurement at **day 120**, against the same
3.0 sessions-per-device bar. If M2 does not move it, kill it at 120. One rescue
attempt, pre-registered, with a date.

**Sessions good, downloads bad** means it works and nobody found it. That is an
App Store listing problem, not a product problem: re-shoot screenshots, rewrite the
subtitle away from the crowded "duck sounds" keyword and toward "robot duck" /
"AR duck", re-measure at day 120 against 1,000 cumulative.

## Deferred gate — the hardware gate

Clock starts on **the day the operator's Microduck is delivered**, not on approval,
because the whole M3 half of the app cannot be evaluated before there is a duck.
Record it here: **delivery date: ________________**

At **delivery + 60 days**, PASS requires both:

- **The app drives a real duck.** Bench-proven, yes or no: paired over the socat
  bridge, all seven sounds play on the robot, the ghost mirrors, the deadman holds
  a `wheee` for ten seconds without cutting out. One line in a runbook, no metric.
- **≥ 3 pieces of third-party evidence that a duck owner used it** — GitHub issues
  or stars on DuckKit from identifiable duck owners, a Pollen Discord/forum mention,
  or a review that describes real hardware. Three is a low bar and that is
  intentional: the population of duck owners in early 2027 is small, so this gate
  tests *reachability*, not scale.

Failing the hardware gate does not kill the app if the day-60 consumer gates
passed. It kills M3 and leaves the soundboard shipped.

## Immediate kill, independent of any clock

If **Pollen Robotics asks us to stop** using the Microduck name, the policies, or
the model — pull the app from sale within 7 days, no negotiation, no argument.
Pre-registering the response is the point: the app exists to send people toward
their robot, so a version of it they do not want is worse than no version. (The
policies and the MJCF are Apache-2.0 and the name is used nominatively, so this is
unlikely. It is written down anyway because the cost of being wrong about it is
the whole relationship.)

## The four $0 proxy sources

1. **App Store Connect App Analytics** — first-time downloads, Sessions, Active
   Devices. Apple-provided aggregates, no SDK, nothing installed. Downloads are
   the numerator; sessions-per-device is the retention proxy.
2. **golinks D1 rollup**, `app=ducksounds` — the in-app "what is a Microduck?"
   link 302s through `golinks.craigm26.workers.dev/ducksounds/<src>`, which
   increments a `(app, src, day)` count. Counts only. No cookies, no IPs, no
   per-user rows. Already deployed and already used by OpenCastor.
   Read: `curl -H "Authorization: Bearer <token>" https://golinks.craigm26.workers.dev/stats`
3. **Cloudflare Web Analytics** on `duck.craigm26.com` — cookieless, no
   fingerprinting. Cross-check for the golinks number and the place shared Calls
   land.
4. **GitHub** — stars, forks and clone counts on the public `craigm26/duckkit`.
   The durable asset is the package, and a soundboard that drives engineers to read
   a pure-Swift ONNX runtime has done a second job worth counting.

None of these instruments the app or its users. That is not a constraint we are
working around; it is the only way to run a Data-Not-Collected app honestly, and
it happens to be free.

---

### Provenance
- Downloads, Sessions, Active Devices: App Store Connect → App Analytics, from the
  approval date.
- Link taps: golinks D1 `golinks_stats.clicks`, `app=ducksounds`.
- Docs visits: Cloudflare Web Analytics, `duck.craigm26.com`.
- DuckKit stars/clones: `gh api repos/craigm26/duckkit/traffic/clones` and the repo
  page.
- **Evaluate PASS at day 60, KILL at day 90, the single rescue re-measure at day
  120, all from the App Store approval date. Hardware gate at delivery + 60.**
