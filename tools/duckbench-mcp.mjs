#!/usr/bin/env node
// The duck bench, as MCP tools — so Claude can run a policy in physics.
//
// WHAT THIS IS FOR. duckbench has the physics an iPhone does not: load a
// trained ONNX policy, run it, and say how often it ends standing and how far
// it actually got. Until now the only ways to ask were the Duck Studio app and
// curl. This makes the same verbs available to a model with a terminal, which
// is the shape most of this work actually takes: "blend these two, run it
// forward, tell me whether it still travels" is three curl calls and some
// arithmetic, done by hand, every time.
//
// IT WRAPS THE HTTP API AND EMBEDS NOTHING. duckbench is already a server with
// a physics lane and a world loaded; a second process with its own MuJoCo would
// be a second world, disagreeing. So this is a thin client, it can point at any
// bench — the Pi, a desktop, a machine on a tailnet — through DUCKBENCH_URL,
// and starting it costs nothing because the bench is already running.
//
// NO DEPENDENCIES, DELIBERATELY. MCP is JSON-RPC 2.0 over stdio and the part
// needed here is four methods. duckkit has no dependencies and duckbench has
// two; a `npm install` between a person and their robot is exactly the kind of
// friction this family exists to remove.
//
// THE HONESTY RULES ARE IN THE TOOL DESCRIPTIONS, WHICH IS THE ONLY PLACE THEY
// CAN BE. A model reading `{"achieves": 16, "rollouts": 16}` will report a
// success, and on this bench that number is reachable by a duck standing
// perfectly still — measured: alpha_walking averaged 75/25 with
// BEST_alpha_stand scores 16 of 16 on "ends standing, trunk at least 100 mm up"
// while travelling two millimetres, where the walking policy it came from
// covers 1.207 m. So `bench_measure` returns the distance beside the count by
// default, and says in its own description why the count alone is not an
// answer. See sim/BLEND-MEASUREMENT.md.

import fs from 'node:fs';

const BASE = (process.env.DUCKBENCH_URL || 'http://127.0.0.1:8770').replace(/\/+$/, '');
const TOKEN = process.env.DUCKBENCH_TOKEN || null;
const PROTOCOL = '2025-06-18';
const FALLBACK_PROTOCOL = '2024-11-05';

// ── talking to the bench ────────────────────────────────────────────────

async function bench(path, body, timeoutMs = 900_000) {
  const headers = { 'content-type': 'application/json' };
  if (TOKEN) headers.authorization = `Bearer ${TOKEN}`;
  const control = new AbortController();
  const timer = setTimeout(() => control.abort(), timeoutMs);
  try {
    const response = await fetch(BASE + path, {
      method: body === undefined ? 'GET' : 'POST',
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: control.signal,
    });
    const text = await response.text();
    let value;
    try { value = JSON.parse(text); }
    catch { throw new Error(`${path} did not answer with JSON: ${text.slice(0, 200)}`); }
    if (response.status === 401) {
      throw new Error('That bench wants a token. Set DUCKBENCH_TOKEN to the same string '
                    + 'DUCKBENCH_TOKEN was set to when it started.');
    }
    // A BENCH REFUSAL IS AN ANSWER, NOT A CRASH. duckbench replies 200 with an
    // `error` key when it will not do something, and the reason is the useful
    // part — "that file did not load as a policy: Missing opset" is what tells
    // somebody their writer is wrong.
    if (value && value.error) throw new Error(`The bench said: ${value.error}`);
    return value;
  } catch (e) {
    if (e.name === 'AbortError') {
      throw new Error(`${path} did not answer within ${Math.round(timeoutMs / 1000)}s. `
                    + 'Physics on a small board is slow; raise the seconds or rollouts you asked '
                    + 'for, or check the bench is still running.');
    }
    if (e.cause?.code === 'ECONNREFUSED') {
      throw new Error(`Nothing is listening at ${BASE}. Start duckbench, or set DUCKBENCH_URL `
                    + 'to the address its start script prints.');
    }
    throw e;
  }
}

/** Planar distance start-to-finish, and the summed step-to-step path.
 *
 * THE PAIR TELLS THEM APART. A duck marching on the spot has a long path and
 * no travel; a duck that topples has some of both. Either number alone calls
 * those the same thing.
 */
function travelOf(roots) {
  if (!Array.isArray(roots) || roots.length < 2) return { travelled: 0, path: 0 };
  const hyp = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  let path = 0;
  for (let i = 1; i < roots.length; i++) path += hyp(roots[i], roots[i - 1]);
  return {
    travelled: +hyp(roots[roots.length - 1], roots[0]).toFixed(4),
    path: +path.toFixed(4),
  };
}

/** The command schedule that actually produces a walk.
 *
 * MEASURED, NOT CHOSEN. alpha_walking travels 7 mm in six seconds at vx 0.15 on
 * this plant — below the gait's threshold, so it just stands, which looks
 * exactly like a broken policy and is not one. At 0.3 it covers 0.681 m and at
 * 0.5, 1.207 m. The half second of nothing at the start is the drop settling.
 */
const WALK = [[0, { vx: 0, vy: 0, vyaw: 0 }], [0.5, { vx: 0.5, vy: 0, vyaw: 0 }]];

function scheduleFrom(input) {
  if (!input) return WALK;
  if (input === 'walk') return WALK;
  if (input === 'still') return [[0, { vx: 0, vy: 0, vyaw: 0 }]];
  return input;
}

// ── the tools ───────────────────────────────────────────────────────────

const TOOLS = [
  {
    name: 'bench_health',
    description:
      'What this bench is: which physics world (plant) is loaded, its sha256, the tick rate, '
      + 'and every policy it can run. Answers without running any physics, so it is the cheapest '
      + 'way to check a bench is reachable. The plant digest matters: a number measured against a '
      + 'different world is not comparable to one measured here, however alike the filenames look.',
    inputSchema: { type: 'object', properties: {} },
    run: async () => bench('/health', undefined, 30_000),
  },
  {
    name: 'bench_record',
    description:
      'Run one policy in physics and report what happened to the duck: how far it travelled, how '
      + 'far it stepped, its final trunk height, and whether it ended upright. Use this to find '
      + 'out what a policy DOES. The default command drives it forward at vx 0.5, which is what '
      + 'it takes to get a gait — below about 0.3 the walking policy simply stands still, and a '
      + 'comparison made at vx 0 is a comparison of two ducks standing still.',
    inputSchema: {
      type: 'object',
      properties: {
        policy: { type: 'string', description: 'A name from bench_health.policies, or one returned by bench_upload_policy.' },
        seconds: { type: 'number', description: 'Rollout length, 0.2 to 30. Default 6.' },
        schedule: {
          description: '"walk" (forward at vx 0.5 after a 0.5 s settle, the default), "still", '
                     + 'or a raw list of [atSeconds, {vx, vy, vyaw}] steps.',
        },
      },
      required: ['policy'],
    },
    run: async ({ policy, seconds = 6, schedule }) => {
      const run = await bench('/record', { policy, seconds, schedule: scheduleFrom(schedule) });
      return {
        policy: run.policy,
        ...travelOf(run.roots),
        endHeight: run.endHeight,
        endsUpright: run.endsUpright,
        seconds,
        plantName: run.plantName,
        plantDigest: run.plantDigest,
        note: run.endsUpright
          ? 'Ended upright. Upright is not the same as working — check the distance.'
          : 'Did not end upright: it fell.',
        // The frame count, not the frames: 300 poses of 14 floats is a lot of
        // context for a model that asked how far the duck got.
        frameCount: run.frames?.length ?? 0,
      };
    },
  },
  {
    name: 'bench_measure',
    description:
      'Run one policy many times over randomised drop heights and report how often it met the '
      + "bench's criterion — plus, by default, how far it got. READ BOTH NUMBERS. The criterion is "
      + '"ends standing, trunk at least 100 mm up", which a duck standing perfectly still passes '
      + 'every time: measured on this bench, alpha_walking averaged 75/25 with BEST_alpha_stand '
      + 'scores 16 of 16 while travelling 2 mm, where the walking policy it was made from covers '
      + '1.207 m. Reporting the rate without the distance turns a total loss of behaviour into a '
      + 'perfect score. Always quote the criterion with the rate.',
    inputSchema: {
      type: 'object',
      properties: {
        policy: { type: 'string' },
        rollouts: { type: 'number', description: '1 to 32. Default 8.' },
        seconds: { type: 'number', description: 'Default 6.' },
        schedule: { description: 'As bench_record. Default "walk".' },
        include_travel: {
          type: 'boolean',
          description: 'Also run one recording to measure distance. Default true. Setting this '
                     + 'false costs one rollout and leaves you with a number that cannot tell a '
                     + 'walk from a duck standing still.',
        },
      },
      required: ['policy'],
    },
    run: async ({ policy, rollouts = 8, seconds = 6, schedule, include_travel = true }) => {
      const plan = scheduleFrom(schedule);
      const rate = await bench('/measure', { policy, rollouts, seconds, schedule: plan });
      const out = {
        policy,
        achieves: rate.achieves,
        rollouts: rate.rollouts,
        criterion: rate.criterion,
        randomised: rate.randomised,
        medianHeight: rate.medianHeight,
        worstHeight: rate.worstHeight,
      };
      if (include_travel) {
        const run = await bench('/record', { policy, seconds, schedule: plan });
        Object.assign(out, travelOf(run.roots), {
          plantName: run.plantName, plantDigest: run.plantDigest,
        });
        out.note = out.travelled < 0.05
          ? 'It barely moved. If this policy is meant to walk, the rate above is measuring it '
          + 'standing still — that passes the criterion and is not the behaviour.'
          : 'It travelled, so the rate above is about a duck that was actually moving.';
      } else {
        out.note = 'No distance measured. This rate alone cannot distinguish a walk from a duck '
                 + 'standing still, because standing passes the criterion.';
      }
      return out;
    },
  },
  {
    name: 'bench_perform',
    description:
      'Run an AUTHORED motion — keyframes somebody wrote, not a trained policy — in real physics, '
      + 'several times. This is the only way to find out what an authored motion actually does: a '
      + 'phone can preview one with kinematics, which shows what you asked for rather than what '
      + 'happens. A single rollout that stays up proves very little; the four authored stair '
      + 'motions in this corpus get up their flight 0 times in 16.',
    inputSchema: {
      type: 'object',
      properties: {
        track: {
          type: 'array',
          description: 'Keyframes: [{at: seconds, pose: [14 joint angles, mouth excluded]}]. '
                     + 'Poses are absolute joint angles in radians.',
        },
        seconds: { type: 'number' },
        rollouts: { type: 'number', description: 'Default 8.' },
        policy: { type: 'string', description: 'The policy the track rides on. Defaults to the standing one.' },
        blend: { type: 'number', description: '0 to 1: how much of the authored pose to apply over the policy. Default 1.' },
      },
      required: ['track'],
    },
    run: async ({ track, seconds, rollouts = 8, policy, blend = 1 }) => {
      const body = { track, rollouts, blend, seconds: seconds ?? (Math.max(...track.map(k => +k.at)) + 0.5) };
      if (policy) body.policy = policy;
      const run = await bench('/perform', body);
      return {
        authored: true,
        achieves: run.achieves, rollouts: run.rollouts, criterion: run.criterion,
        ...travelOf(run.roots),
        endsUpright: run.endsUpright, endHeight: run.endHeight,
        peakJointRate: run.peakJointRate,
        plantName: run.plantName, plantDigest: run.plantDigest,
        policy: run.policy,
      };
    },
  },
  {
    name: 'bench_upload_policy',
    description:
      'Put an ONNX policy file from this machine onto the bench so it can be run. Returns the '
      + 'name the bench filed it under — pass that to bench_record or bench_measure. The bench '
      + 'refuses a file that will not load, and the reason it gives is the useful part: it is a '
      + 'real onnxruntime, so it catches things a hand-written parser does not (a missing opset, '
      + 'nodes that declare no outputs, an attribute with no type).',
    inputSchema: {
      type: 'object',
      properties: { path: { type: 'string', description: 'Absolute path to a .onnx file on this machine.' } },
      required: ['path'],
    },
    run: async ({ path }) => {
      if (!fs.existsSync(path)) throw new Error(`There is no file at ${path}.`);
      const onnx = fs.readFileSync(path).toString('base64');
      const out = await bench('/upload', { onnx });
      return { ...out, note: 'It loaded. That says nothing about whether it works — run it.' };
    },
  },
  {
    name: 'bench_state',
    description:
      'Where the duck in the LIVE world is right now — the one that keeps standing between '
      + 'requests, as opposed to the fresh drop each recording starts from. Use with bench_steer '
      + 'to drive it a step at a time.',
    inputSchema: { type: 'object', properties: {} },
    run: async () => bench('/state', {}, 60_000),
  },
  {
    name: 'bench_steer',
    description:
      'Hold a twist on the live duck for a short window and advance physics, then report where it '
      + 'ended up. This is the steering loop: send an intent, read the state it caused, send the '
      + 'next one. Note that vx below about 0.3 does not start a gait.',
    inputSchema: {
      type: 'object',
      properties: {
        vx: { type: 'number', description: 'Forward m/s.' },
        vy: { type: 'number', description: 'Left m/s.' },
        vyaw: { type: 'number', description: 'Yaw rad/s.' },
        hold: { type: 'number', description: 'Seconds to hold it, 0.02 to 2. Default 0.1.' },
      },
    },
    run: async (args) => bench('/intent', args, 120_000),
  },
  {
    name: 'bench_stop',
    description:
      'Zero the live duck\'s command and let it settle under it. NOT a reset: stopping is '
      + 'something the policy does, and a duck teleported upright to stop would be hiding a fall.',
    inputSchema: {
      type: 'object',
      properties: { settle: { type: 'number', description: 'Seconds, up to 5. Default 0.5.' } },
    },
    run: async (args) => bench('/stop', args, 120_000),
  },
];

const BY_NAME = new Map(TOOLS.map(t => [t.name, t]));

// ── MCP over stdio ──────────────────────────────────────────────────────

function send(message) {
  process.stdout.write(JSON.stringify(message) + '\n');
}

function reply(id, result) { send({ jsonrpc: '2.0', id, result }); }
function fail(id, code, message) { send({ jsonrpc: '2.0', id, error: { code, message } }); }

async function handle(request) {
  const { id, method, params } = request;
  // A NOTIFICATION HAS NO ID AND MUST GET NO ANSWER. Replying to one is a
  // protocol error that some clients treat as fatal.
  const isNotification = id === undefined || id === null;

  switch (method) {
    case 'initialize': {
      const asked = params?.protocolVersion;
      return reply(id, {
        protocolVersion: asked === FALLBACK_PROTOCOL ? FALLBACK_PROTOCOL : PROTOCOL,
        capabilities: { tools: {} },
        serverInfo: { name: 'duckbench', version: '1.0.0' },
        instructions:
          `Physics for a Pollen Microduck, served by duckbench at ${BASE}.\n\n`
          + 'Two rules that are easy to get wrong here:\n'
          + '1. A success rate is not an answer on its own. The criterion is "ends standing", '
          + 'which a duck standing perfectly still passes every time — so always report the '
          + 'distance travelled beside the count, and say what the criterion was.\n'
          + '2. Name the plant. Every result carries plantName and plantDigest; a measurement '
          + 'against a different world is not comparable to one made here.',
      });
    }
    case 'notifications/initialized':
      return;
    case 'ping':
      return isNotification ? undefined : reply(id, {});
    case 'tools/list':
      return reply(id, {
        tools: TOOLS.map(({ name, description, inputSchema }) => ({ name, description, inputSchema })),
      });
    case 'tools/call': {
      const tool = BY_NAME.get(params?.name);
      if (!tool) return fail(id, -32602, `There is no tool called ${params?.name}.`);
      try {
        const value = await tool.run(params.arguments || {});
        return reply(id, { content: [{ type: 'text', text: JSON.stringify(value, null, 2) }] });
      } catch (e) {
        // A FAILED RUN IS A RESULT, NOT A PROTOCOL ERROR. isError lets the
        // model read the reason and act on it — "the bench said: Missing
        // opset" is the whole point — instead of seeing a transport fault.
        return reply(id, { content: [{ type: 'text', text: String(e.message || e) }], isError: true });
      }
    }
    default:
      if (isNotification) return;
      return fail(id, -32601, `This server does not implement ${method}.`);
  }
}

let buffer = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => {
  buffer += chunk;
  let cut;
  // Line-delimited JSON, which is what every current MCP stdio client speaks.
  while ((cut = buffer.indexOf('\n')) !== -1) {
    const line = buffer.slice(0, cut).trim();
    buffer = buffer.slice(cut + 1);
    if (!line) continue;
    let request;
    try { request = JSON.parse(line); }
    catch { fail(null, -32700, 'That was not JSON.'); continue; }
    handle(request).catch(e => {
      if (request.id !== undefined && request.id !== null) {
        fail(request.id, -32603, String(e.message || e));
      }
    });
  }
});
process.stdin.on('end', () => process.exit(0));
