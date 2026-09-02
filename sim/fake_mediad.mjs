#!/usr/bin/env node
//
// fake_mediad — a daemon-shaped thing to point a phone's transport code at.
//
// WHY THIS EXISTS. `LinePeer` in StudioKit is the app's side of a link that
// nobody can test against the real thing: `robotd` speaks JSON-RPC over a UNIX
// socket that no iPhone can open, mediad's WebRTC path has five unanswered
// questions in front of it (`DuckWebRTC.fiveThingsNobodyHereKnows`), and the
// robots ship around Christmas 2026. What CAN be tested today is the half of
// the problem that is pure protocol — correlating a reply to a request while
// telemetry is arriving in between, reassembling lines out of whatever chunks
// the OS hands over, and handing one state to three readers at once — and
// testing that needs something on the other end that behaves like a daemon
// rather than like a fixture.
//
// SO THIS IS DELIBERATELY NOT A SIMULATION OF A DUCK. It runs no physics, holds
// no policy and does not pretend the numbers in its state mean anything: the
// pose it reports is canned. What it reproduces faithfully is the SHAPE of the
// traffic — one JSON object per line, replies carrying the id they answer, and
// `robot.state` notifications arriving unbidden at the loop rate, including
// while a request is in flight. That last one is the case that breaks naive
// clients, because a client that reads "the next line" after writing a request
// will read a state notification and report it as a corrupt answer.
//
// STDIO RATHER THAN A SOCKET, so a test can spawn it with a pipe and no port,
// no permission and nothing left listening if the test crashes.
//
// IT IS A SPIKE AND ITS NUMBERS ARE NOT EVIDENCE. Nothing here has been checked
// against a robot. The method names come from the app's own transcription of
// `duck-ipc-proto`; the state fields come from `DuckState` in duckkit. If the
// real daemon disagrees, this file is wrong and the app is wrong in the same
// way, which is exactly what a fake cannot tell you.
//
// Usage:
//   node fake_mediad.mjs [--rate HZ] [--limit N] [--state-before-reply]
//                        [--standing] [--battery] [--quiet]
//
//   --rate HZ             state notifications per second (default 50, 0 = none)
//   --limit N             stop after N state notifications (default 0 = forever)
//   --state-before-reply  emit one extra state immediately before each reply,
//                         so a reply is never the first thing a request sees
//   --standing            report safety.fallen false (default is true, which is
//                         the canned state the Swift test asserts on)
//   --battery             include a battery block (default: absent, so a reader
//                         gets a nil rather than a plausible percentage)
//   --quiet               do not emit unsolicited states at all; --limit and
//                         --state-before-reply still apply
//
// It exits when stdin ends. That is the only shutdown path on purpose: a fake
// that exited on its own would race the test that is still reading its output.

const argv = process.argv.slice(2);

function flag(name) {
  return argv.includes(`--${name}`);
}

function value(name, fallback) {
  const at = argv.indexOf(`--${name}`);
  if (at === -1 || at === argv.length - 1) return fallback;
  const parsed = Number(argv[at + 1]);
  return Number.isFinite(parsed) ? parsed : fallback;
}

const rateHz = flag('quiet') ? 0 : value('rate', 50);
const limit = value('limit', 0);
const fallen = !flag('standing');
const withBattery = flag('battery');
const stateBeforeReply = flag('state-before-reply');

// THE API VERSION IS THE APP'S OWN NUMBER, ECHOED BACK. `DuckLink.apiVersion`
// is 16 and `DuckLink.verdict(for:)` has three different sentences for equal,
// newer and older — so a fake that answered a different number would send the
// app down a path about a version difference that does not exist.
const apiVersion = 16;

let statesSent = 0;
let ticking = null;

// ── one canned state ────────────────────────────────────────────────────────
//
// EVERY FIELD HERE IS ONE `DuckState` KNOWS, AND THE ABSENCES ARE THE POINT.
// There is no battery block unless it is asked for, because a client that fills
// a missing battery with a zero is the exact failure duckkit's all-optional
// design exists to prevent — "a zero is a lie that looks exactly like data" —
// and a test can only prove the client leaves it nil if the fake leaves it out.

function cannedState() {
  const state = {
    policy: 'alpha_walking',
    safety: { fallen, limp: false },
    loop: { hz: 50, missed: 0 },
    odom: { position: [0.12, -0.03], yaw: 0.4 },
    move: { requested: [0.3, 0, 0], applied: [0.3, 0, 0], limited_by: [] },
  };
  if (withBattery) state.battery = { volts: 11.4, percent: 62 };
  return state;
}

function write(object) {
  // ONE OBJECT, ONE LINE, AND `JSON.stringify` NEVER EMITS A RAW NEWLINE — the
  // same fact `framing.rs` relies on: a newline inside a string is escaped, so
  // the terminator is unambiguous.
  process.stdout.write(`${JSON.stringify(object)}\n`);
}

function sendState() {
  if (limit > 0 && statesSent >= limit) {
    if (ticking) {
      clearInterval(ticking);
      ticking = null;
    }
    return false;
  }
  statesSent += 1;
  write({ jsonrpc: '2.0', method: 'robot.state', params: cannedState() });
  return true;
}

// ── answering ───────────────────────────────────────────────────────────────

function resultFor(method) {
  switch (method) {
    case 'hello':
      // A robot answers hello with the version it speaks. The other members are
      // this fake naming itself, so a reply read by a human is obviously not a
      // robot's.
      return { api_version: apiVersion, name: 'fake-mediad', real: false };
    case 'robot.stop':
    case 'robot.enable':
    case 'robot.relax':
    case 'robot.init':
      return { ok: true };
    case 'robot.look':
      return { ok: true, settled: true };
    default:
      return null;
  }
}

function handle(message) {
  // A NOTIFICATION IS NEVER ANSWERED. JSON-RPC 2.0: a method with no id owes no
  // reply, and a fake that answered one would let a client pass a test it would
  // fail against a robot — the client would learn to expect a reply per twist
  // at 50 Hz, which is the queue the contract's notification shape exists to
  // avoid.
  if (message.id === undefined || message.id === null) return;

  if (stateBeforeReply) sendState();

  const result = resultFor(message.method);
  if (result === null) {
    write({
      jsonrpc: '2.0',
      id: message.id,
      error: { code: -32601, message: `no such method: ${message.method}` },
    });
    return;
  }
  write({ jsonrpc: '2.0', id: message.id, result });
}

// ── reading, across whatever chunks arrive ──────────────────────────────────
//
// THE FRAMING IS THE HARD PART ON THIS SIDE TOO. A pipe hands over whatever it
// feels like, so a line can be split across two reads and four can arrive in
// one. This holds a single partial line, exactly as `DuckRPC.StreamDecoder`
// does, and refuses a line that grows past a ceiling rather than buffering an
// unbounded write.

const maxLineBytes = 256 * 1024;
let pending = '';

process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => {
  pending += chunk;
  if (pending.length > maxLineBytes) {
    // Drop it and resynchronise at the next newline, rather than holding a
    // wedged sender's bytes until this process is killed.
    const at = pending.indexOf('\n');
    pending = at === -1 ? '' : pending.slice(at + 1);
    return;
  }
  let at = pending.indexOf('\n');
  while (at !== -1) {
    const line = pending.slice(0, at).replace(/\r$/, '');
    pending = pending.slice(at + 1);
    if (line.trim().length > 0) {
      try {
        handle(JSON.parse(line));
      } catch {
        // A line that is not JSON is not this fake's problem to diagnose; the
        // client is the thing under test and it will find out by not getting an
        // answer. Counting it silently is what a daemon does.
      }
    }
    at = pending.indexOf('\n');
  }
});

process.stdin.on('end', () => {
  if (ticking) clearInterval(ticking);
  process.exit(0);
});

if (rateHz > 0) {
  // `setInterval` is not a control loop and this is not claiming to be one: at
  // 50 Hz Node's timer resolution puts the real spacing somewhere near 20 ms
  // and nothing here measures it. What the test depends on is the ORDER of the
  // lines and the COUNT of them, never their timing.
  ticking = setInterval(sendState, Math.max(1, Math.round(1000 / rateHz)));
}
