// This machine, described to the bench core.
//
// WHY IT IS ITS OWN FILE AND NOT PART OF duckbench.mjs. Importing duckbench.mjs
// starts an HTTP server on a port, which makes the bench impossible to test
// without a socket and a subprocess — and the parity gate that licensed the
// split needs to call `handle` directly, in one process, with no network
// between the request and the answer. So the ENV lives here, importable, and
// duckbench.mjs is the door.
//
// EVERYTHING IN HERE IS A NODE FACT. fs for the assets, os for the core count,
// node:crypto for the digests, onnxruntime-node for the forward pass. The core
// beside it knows none of those words, which is what lets the same physics run
// in a browser on a phone.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import { makeBench } from './duckbench-core.mjs';
import { makeLoop } from './duckloop.mjs';
import { declaredDefaultPoseOf } from './onnx_meta.mjs';
import { makeForwardSession, FLOAT_COUNT } from './policyforward.mjs';

// ASSETS ARE FOUND BESIDE THIS FILE, NOT BESIDE THE SHELL THAT LAUNCHED IT.
// The bench used to read `duckkit-constants.json` and scan `.` — the process's
// working directory — so `node sim/duckbench.mjs` from the repo root started
// and died on ENOENT, and a systemd unit without a WorkingDirectory did the
// same. The files it needs sit next to it; that is where it looks.
const HERE = path.dirname(fileURLToPath(import.meta.url));
const at = name => path.join(HERE, name);

/**
 * Every .onnx this bench will run, by bare name — the whole allow-list,
 * rescanned on every call so that an upload appears without a restart.
 *
 * NAME AND PATH ARE DIFFERENT THINGS and this is the only place that knows
 * both: the core is handed names and hands them back to `readAsset`, so a
 * community policy is `flamingo-cycle/policy.onnx` to everyone above this line
 * and `community/flamingo-cycle/policy.onnx` to the disk.
 */
function scan() {
  const out = new Map();
  for (const file of fs.readdirSync(HERE)) {
    if (file.endsWith('.onnx')) out.set(file, file);
  }
  const community = at('community');
  if (fs.existsSync(community)) {
    for (const dir of fs.readdirSync(community)) {
      const candidate = path.join('community', dir, 'policy.onnx');
      if (fs.existsSync(at(candidate))) out.set(`${dir}/policy.onnx`, candidate);
    }
  }
  return out;
}

/** The MuJoCo this bench runs, said out loud: every clip in duckkit is stamped
 *  with a plant digest, and the engine that produced it is the other half. */
function mujocoVersion() {
  try {
    return JSON.parse(fs.readFileSync(at('node_modules/mujoco/package.json'), 'utf8')).version;
  } catch { return 'unknown'; }
}

/** The CPU this is, as the kernel reports it — for a saved measurement to carry. */
function device() {
  const cpus = os.cpus();
  const model = cpus[0]?.model?.trim() || os.arch();
  let board = null;
  try { board = fs.readFileSync('/proc/device-tree/model', 'utf8').replace(/\0/g, '').trim(); }
  catch { /* not a device-tree machine */ }
  return `${board || `${os.type()} ${os.arch()}`} — ${cpus.length}× ${model}`;
}

/**
 * A bench on this machine.
 *
 * `engine` PICKS THE FORWARD PASS, AND IT EXISTS FOR ONE REASON: to take the
 * inference out of the phone-versus-desk comparison. onnxruntime and
 * `policyforward.mjs` agree to 3.5e-6 per action (policy_parity.mjs), which is
 * nothing in one tick and is not nothing after 250 closed-loop ticks. So
 * `physics_parity.mjs` runs THIS engine — the same arithmetic the browser runs
 * — and what is left over is the physics, which is the thing being compared.
 */
export async function nodeBench({ engine = 'onnxruntime' } = {}) {
  const constants = JSON.parse(fs.readFileSync(at('duckkit-constants.json'), 'utf8'));
  const { HOME } = makeLoop(constants);
  let catalogue = scan();

  return makeBench({
    sceneName: process.env.DUCKBENCH_SCENE || 'scene.mjb',
    mujoco: await load(),
    readAsset(name) {
      // A POLICY NAME GOES THROUGH THE ALLOW-LIST, EVERYTHING ELSE IS A FIXED
      // ASSET. Nothing here joins a caller's string onto a path: the core only
      // ever asks for a name it was handed by `listPolicies`, plus the two
      // files this bench is built out of.
      const file = catalogue.get(name) ?? name;
      return fs.readFileSync(at(file));
    },
    listPolicies() {
      catalogue = scan();
      return [...catalogue.keys()];
    },
    /**
     * THE CANONICAL PARAMETER BYTES, WHICH ON THIS MACHINE ARE A DIFFERENT FILE
     * FROM THE POLICY.
     *
     * A browser is served duckkit's canonical bytes UNDER THE POLICY'S OWN
     * NAME, so there the two are one file and the core needs no help. Here
     * `readAsset` hands back an .onnx and the canonical dump sits beside it in
     * `params/`, produced by the `dumpparams` tool from duckkit's own writer.
     * /tune folds a gain into the last layer, which means holding parameters —
     * and where they are is exactly the kind of fact the core is not allowed to
     * know.
     *
     * A MISSING DUMP IS AN ERROR AND NOT A FALLBACK. Quietly scoring through
     * onnxruntime instead would score the BASE network and call it the folded
     * one, which is a search that cannot fail and cannot succeed.
     */
    readParameters(file, name) {
      const dump = `params/${name.replace('/policy.onnx', '-policy').replace(/\.onnx$/, '')}.bin`;
      if (!fs.existsSync(at(dump))) {
        throw new Error(`${name} has no canonical parameter bytes at ${dump}: run dumpparams`);
      }
      return fs.readFileSync(at(dump));
    },
    sha256: bytes => createHash('sha256').update(bytes).digest('hex'),
    cores: os.cpus().length,
    async makeSession(bytes, name) {
      if (engine === 'policyforward') {
        // The canonical parameter bytes beside the .onnx, which is what the
        // browser is served. A policy without them is refused rather than
        // silently falling back to onnxruntime — a fallback here would make
        // the physics gate compare two different networks and call it a pass.
        const file = `params/${name.replace('/policy.onnx', '-policy').replace(/\.onnx$/, '')}.bin`;
        if (!fs.existsSync(at(file))) {
          throw new Error(`${name} has no canonical parameter bytes at ${file}: run dumpparams`);
        }
        const params = fs.readFileSync(at(file));
        if (params.byteLength !== FLOAT_COUNT * 4) throw new Error(`${file} is the wrong size`);
        return { ...makeForwardSession(params, name),
                 reference: declaredDefaultPoseOf(bytes, HOME) ?? undefined };
      }
      // FROM BYTES, NOT FROM A PATH. onnxruntime-node takes either; bytes is
      // the one that also works for an upload that has not been written yet,
      // and it is the same call for both, so a shipped policy and an uploaded
      // one cannot diverge in how they were loaded.
      const net = await ort.InferenceSession.create(bytes);
      const output = net.outputNames[0];
      return {
        name,
        // The neutral pose the policy DECLARES, which is the shell's to read
        // because the shell is the half holding the file. `undefined` when it
        // declares this project's own, so the core keeps HOME by identity and
        // /policy can still say which of the two it got.
        reference: declaredDefaultPoseOf(bytes, HOME) ?? undefined,
        async run(obs) {
          const out = await net.run({ obs: new ort.Tensor('float32', obs, [1, 61]) });
          return out[output].data;
        },
      };
    },
    scratch: {
      has: name => fs.existsSync(at(name)),
      get: name => fs.readFileSync(at(name)),
      set: (name, bytes) => fs.writeFileSync(at(name), bytes),
      delete: name => fs.unlinkSync(at(name)),
    },
    host: {
      kind: 'desk',
      device: device(),
      engine: engine === 'policyforward'
        ? `Node ${process.version}, MuJoCo ${mujocoVersion()} (WASM), policyforward.mjs`
        : `Node ${process.version}, MuJoCo ${mujocoVersion()} (WASM), onnxruntime-node`,
    },
  });
}
