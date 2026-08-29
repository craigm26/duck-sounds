// Read an ONNX file's metadata_props without a protobuf library.
//
// WHY BOTHER. Every policy states the neutral pose it was trained against in
// `metadata_props.default_joint_pos`, and the harness ignored it — computing
// the observation's joint deviation from this project's HOME instead, and
// adding the action to HOME as well. Pollen's ten files all declare a pose
// equal to HOME, so nothing noticed; the community `headspin.onnx` declares
// neck_pitch 0.220 and head_pitch 0.680 where HOME has 0.349 and 0.349, so it
// was being lied to by 7 degrees and 19 degrees on the head — in a policy whose
// whole job is balancing on that head.
//
// onnxruntime-node loads the graph and does not expose metadata_props, and
// pulling in protobufjs to read six strings is a dependency for nothing. The
// wire format is four field types and this reads two of them.
import fs from 'node:fs';

function varint(b, i) {
  let v = 0, s = 0;
  for (;;) {
    const x = b[i++];
    v += (x & 0x7f) * 2 ** s;
    if (!(x & 0x80)) return [v, i];
    s += 7;
  }
}

/** Top-level fields of one message: [fieldNumber, wireType, payload]. */
function* fields(b, start = 0, end = b.length) {
  let i = start;
  while (i < end) {
    let key; [key, i] = varint(b, i);
    const f = key >>> 3, wt = key & 7;
    if (wt === 0) { let v; [v, i] = varint(b, i); yield [f, wt, v]; }
    else if (wt === 2) {
      let n; [n, i] = varint(b, i);
      yield [f, wt, b.subarray(i, i + n)]; i += n;
    } else if (wt === 5) { yield [f, wt, b.subarray(i, i + 4)]; i += 4; }
    else if (wt === 1) { yield [f, wt, b.subarray(i, i + 8)]; i += 8; }
    else return;
  }
}

/** ModelProto.metadata_props (field 14) as a plain object. */
export function metadata(path) {
  const b = fs.readFileSync(path);
  const out = {};
  for (const [f, wt, p] of fields(b)) {
    if (f !== 14 || wt !== 2) continue;
    let k = null, v = null;
    for (const [sf, , sp] of fields(p)) {
      if (sf === 1) k = Buffer.from(sp).toString('utf8');
      if (sf === 2) v = Buffer.from(sp).toString('utf8');
    }
    if (k !== null) out[k] = v;
  }
  return out;
}

/**
 * The neutral pose a policy was trained against, or null when it declares none
 * or declares this project's own.
 *
 * DELIBERATELY RETURNS NULL WHEN IT MATCHES. The metadata is serialised to
 * three decimals, so adopting it verbatim would replace exact HOME values with
 * rounded ones on ten files that do not need it — a corpus-wide change to fix
 * one policy. Only a real difference is honoured.
 */
export function declaredDefaultPose(path, home, tolerance = 1e-3) {
  const raw = metadata(path).default_joint_pos;
  if (!raw) return null;
  const pose = raw.split(',').map(Number);
  if (pose.length !== home.length || pose.some(v => !Number.isFinite(v))) return null;
  const differs = pose.some((v, k) => Math.abs(v - home[k]) > tolerance);
  return differs ? pose : null;
}
