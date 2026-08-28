// A small WebGL renderer for the duck.
//
// The geometry, the per-part offsets and the colours all come out of MuJoCo's
// own compiled model (see sim/export_visual.mjs) — 70 visual meshes, 100,881
// triangles — so what is on screen is the robot's actual shape rather than
// boxes standing in for it. Canvas 2D could not carry this: a hundred thousand
// depth-sorted triangles a frame is a WebGL job.
//
// No three.js. It is 70 draw calls of static geometry with one directional
// light; the whole renderer is smaller than the loader would have been.

// ── tiny matrix helpers ────────────────────────────────────────────────────
export function mat4() { const m = new Float32Array(16); m[0]=m[5]=m[10]=m[15]=1; return m; }
export function multiply(out, a, b) {
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    out[c*4+r] = a[r]*b[c*4] + a[4+r]*b[c*4+1] + a[8+r]*b[c*4+2] + a[12+r]*b[c*4+3];
  }
  return out;
}
/** MuJoCo quaternions are [w, x, y, z]. */
export function fromQuatPos(out, q, p) {
  const [w, x, y, z] = q;
  const x2=x+x, y2=y+y, z2=z+z;
  const xx=x*x2, xy=x*y2, xz=x*z2, yy=y*y2, yz=y*z2, zz=z*z2;
  const wx=w*x2, wy=w*y2, wz=w*z2;
  out[0]=1-(yy+zz); out[1]=xy+wz;     out[2]=xz-wy;     out[3]=0;
  out[4]=xy-wz;     out[5]=1-(xx+zz); out[6]=yz+wx;     out[7]=0;
  out[8]=xz+wy;     out[9]=yz-wx;     out[10]=1-(xx+yy);out[11]=0;
  out[12]=p[0];     out[13]=p[1];     out[14]=p[2];     out[15]=1;
  return out;
}
export function perspective(out, fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  out.fill(0);
  out[0]=f/aspect; out[5]=f; out[10]=(far+near)*nf; out[11]=-1; out[14]=2*far*near*nf;
  return out;
}
export function lookAt(out, eye, center, up) {
  let z0=eye[0]-center[0], z1=eye[1]-center[1], z2=eye[2]-center[2];
  let l = 1/Math.hypot(z0,z1,z2); z0*=l; z1*=l; z2*=l;
  let x0=up[1]*z2-up[2]*z1, x1=up[2]*z0-up[0]*z2, x2=up[0]*z1-up[1]*z0;
  l = Math.hypot(x0,x1,x2) || 1; x0/=l; x1/=l; x2/=l;
  const y0=z1*x2-z2*x1, y1=z2*x0-z0*x2, y2=z0*x1-z1*x0;
  out[0]=x0; out[1]=y0; out[2]=z0; out[3]=0;
  out[4]=x1; out[5]=y1; out[6]=z1; out[7]=0;
  out[8]=x2; out[9]=y2; out[10]=z2; out[11]=0;
  out[12]=-(x0*eye[0]+x1*eye[1]+x2*eye[2]);
  out[13]=-(y0*eye[0]+y1*eye[1]+y2*eye[2]);
  out[14]=-(z0*eye[0]+z1*eye[1]+z2*eye[2]);
  out[15]=1;
  return out;
}

const VERT = `
attribute vec3 aPos; attribute vec3 aNormal;
uniform mat4 uModel, uView, uProj;
varying vec3 vNormal; varying float vDepth;
void main() {
  vec4 world = uModel * vec4(aPos, 1.0);
  vNormal = mat3(uModel) * aNormal;
  vec4 eye = uView * world;
  vDepth = -eye.z;
  gl_Position = uProj * eye;
}`;

const FRAG = `
precision mediump float;
varying vec3 vNormal; varying float vDepth;
uniform vec3 uColor; uniform vec3 uFog;
void main() {
  vec3 n = normalize(vNormal);
  // One key light high and to the side, plus sky/ground ambient, which is
  // what stops a white shell reading as a flat silhouette.
  float key = max(dot(n, normalize(vec3(0.45, 0.6, 0.85))), 0.0);
  float sky = 0.5 + 0.5 * n.z;
  vec3 lit = uColor * (0.34 + 0.20 * sky + 0.62 * key);
  lit += vec3(0.06) * pow(max(dot(n, normalize(vec3(-0.5, -0.3, 0.4))), 0.0), 2.0);
  float fog = clamp((vDepth - 0.55) / 1.6, 0.0, 1.0);
  gl_FragColor = vec4(mix(lit, uFog, fog * 0.55), 1.0);
}`;

const LINE_VERT = `
attribute vec3 aPos; uniform mat4 uView, uProj; varying float vDepth;
void main(){ vec4 e = uView * vec4(aPos,1.0); vDepth = -e.z; gl_Position = uProj * e; }`;
const LINE_FRAG = `
precision mediump float; varying float vDepth; uniform vec3 uColor; uniform vec3 uFog;
void main(){ float f = clamp((vDepth - 0.4)/2.0, 0.0, 1.0); gl_FragColor = vec4(mix(uColor, uFog, f), 1.0); }`;

function compile(gl, vs, fs) {
  const p = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    gl.attachShader(p, s);
  }
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}

/** A unit cube: 12 triangles, positions and flat normals. */
function unitBox() {
  const f = [
    [[ 1,0,0],[1,-1,-1],[1,1,-1],[1,1,1],[1,-1,1]],
    [[-1,0,0],[-1,-1,-1],[-1,-1,1],[-1,1,1],[-1,1,-1]],
    [[0, 1,0],[-1,1,-1],[-1,1,1],[1,1,1],[1,1,-1]],
    [[0,-1,0],[-1,-1,-1],[1,-1,-1],[1,-1,1],[-1,-1,1]],
    [[0,0, 1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
    [[0,0,-1],[-1,-1,-1],[-1,1,-1],[1,1,-1],[1,-1,-1]],
  ];
  const pos = [], nrm = [];
  for (const [n, a, b, c, d] of f) {
    for (const v of [a, b, c, a, c, d]) { pos.push(...v); nrm.push(...n); }
  }
  return { pos, nrm };
}

/** A unit sphere by latitude/longitude. Coarse on purpose — it is a prop. */
function unitSphere(seg = 16, rings = 12) {
  const pos = [], nrm = [];
  const at = (i, j) => {
    const t = Math.PI * j / rings, p = 2 * Math.PI * i / seg;
    return [Math.sin(t) * Math.cos(p), Math.sin(t) * Math.sin(p), Math.cos(t)];
  };
  for (let j = 0; j < rings; j++) for (let i = 0; i < seg; i++) {
    const a = at(i, j), b = at(i + 1, j), c = at(i + 1, j + 1), d = at(i, j + 1);
    for (const v of [a, b, c, a, c, d]) { pos.push(...v); nrm.push(...v); }
  }
  return { pos, nrm };
}

export async function createRenderer(canvas, url) {
  const gl = canvas.getContext('webgl', { antialias: true, alpha: true });
  if (!gl) throw new Error('this browser has no WebGL');

  const raw = await (await fetch(url)).arrayBuffer();
  const headerLen = new DataView(raw).getUint32(0, true);
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(raw, 4, headerLen)));
  let off = 4 + headerLen;
  const positions = new Float32Array(raw, off, meta.nvert * 3); off += meta.nvert * 12;
  const normals = new Float32Array(raw, off, meta.nvert * 3); off += meta.nvert * 12;
  const indices = new Uint32Array(raw, off, meta.nface * 3);

  const ext = gl.getExtension('OES_element_index_uint');
  if (!ext) throw new Error('this browser cannot index past 65k vertices');

  const posBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf); gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
  const nrmBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, nrmBuf); gl.bufferData(gl.ARRAY_BUFFER, normals, gl.STATIC_DRAW);
  const idxBuf = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

  const prog = compile(gl, VERT, FRAG);
  const u = n => gl.getUniformLocation(prog, n);
  const loc = { model: u('uModel'), view: u('uView'), proj: u('uProj'), color: u('uColor'), fog: u('uFog') };
  const aPos = gl.getAttribLocation(prog, 'aPos'), aNrm = gl.getAttribLocation(prog, 'aNormal');

  // floor grid
  const lineProg = compile(gl, LINE_VERT, LINE_FRAG);
  const lLoc = { view: gl.getUniformLocation(lineProg, 'uView'), proj: gl.getUniformLocation(lineProg, 'uProj'),
                 color: gl.getUniformLocation(lineProg, 'uColor'), fog: gl.getUniformLocation(lineProg, 'uFog') };
  const lAPos = gl.getAttribLocation(lineProg, 'aPos');
  const gridVerts = [];
  const N = 10, S = 0.1;
  for (let i = -N; i <= N; i++) {
    gridVerts.push(i*S, -N*S, 0, i*S, N*S, 0, -N*S, i*S, 0, N*S, i*S, 0);
  }
  const gridBuf = gl.createBuffer();
  const gridMoved = new Float32Array(gridVerts.length);
  gl.bindBuffer(gl.ARRAY_BUFFER, gridBuf);
  gl.bufferData(gl.ARRAY_BUFFER, gridMoved, gl.DYNAMIC_DRAW);

  // Primitive geometry, so anything in the model that is not a mesh — the
  // stairs, the ball, the blocks, the cones — gets drawn too. Before this the
  // renderer only knew about the duck's 70 meshes, so the stairs were solid to
  // walk on and completely invisible.
  const box = unitBox(), sph = unitSphere();
  const primPos = new Float32Array([...box.pos, ...sph.pos]);
  const primNrm = new Float32Array([...box.nrm, ...sph.nrm]);
  const BOX_AT = 0, BOX_N = box.pos.length / 3;
  const SPH_AT = BOX_N, SPH_N = sph.pos.length / 3;
  const pPosBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, pPosBuf); gl.bufferData(gl.ARRAY_BUFFER, primPos, gl.STATIC_DRAW);
  const pNrmBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, pNrmBuf); gl.bufferData(gl.ARRAY_BUFFER, primNrm, gl.STATIC_DRAW);

  const proj = mat4(), view = mat4(), modelM = mat4(), localM = mat4(), bodyM = mat4(), scaleM = mat4();
  const placeM = mat4(), tmpM = mat4();

  // Resolve every draw's body by NAME against the live model. The pack is
  // exported from the robot-only scene; the scene the page runs has stairs and
  // props whose bodies come first, so the indices in the pack are not the
  // indices here.
  let resolved = false;
  function resolve(model) {
    if (resolved || !model) return;
    const byName = new Map();
    for (let b = 0; b < model.nbody; b++) byName.set(model.body(b).name, b);
    for (const d of meta.draws) {
      const b = byName.get(d.bodyName);
      if (b !== undefined) d.body = b;
    }
    resolved = true;
  }

  return {
    gl,
    triangles: meta.nface,
    draws: meta.draws.length,
    render(data, opts) {
      resolve(opts.model);
      const w = canvas.clientWidth, h = canvas.clientHeight;
      const dpr = Math.min(devicePixelRatio || 1, 2);
      if (canvas.width !== Math.round(w * dpr)) { canvas.width = Math.round(w*dpr); canvas.height = Math.round(h*dpr); }
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.enable(gl.DEPTH_TEST);
      gl.clearColor(opts.bg[0], opts.bg[1], opts.bg[2], 0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      // The camera trails the duck at a fixed offset, looking at its chest.
      // NOT qpos[0]: the stair bodies sit ahead of the duck in the model, so
      // the trunk's free joint starts wherever opts.root says it does.
      const cx = data.qpos[opts.root], cy = data.qpos[opts.root + 1];
      // Zoom moves the camera in and out along its own offset rather than
      // changing the field of view: a wider lens on a 25 cm robot distorts it.
      const zoom = Math.min(Math.max(opts.zoom || 1, 0.45), 3.2);
      // Close enough that a 25 cm robot reads as the subject: the duck is
      // about 0.25 m tall, so a 0.42 m standoff at 40 degrees fills the frame.
      const eye = [cx - 0.34 / zoom, cy - 0.55 / zoom, 0.30 / zoom];

      // In an immersive session the headset supplies both matrices and the duck
      // stands wherever the hit test landed. `placeM` moves the whole scene
      // there, and turns MuJoCo's z-up into WebXR's y-up on the way.
      const xr = opts.xr || null;
      if (xr) {
        const [ox, oy, oz] = xr.origin;
        placeM[0]=1; placeM[1]=0; placeM[2]=0;  placeM[3]=0;
        placeM[4]=0; placeM[5]=0; placeM[6]=-1; placeM[7]=0;
        placeM[8]=0; placeM[9]=1; placeM[10]=0; placeM[11]=0;
        placeM[12]=ox; placeM[13]=oy; placeM[14]=oz; placeM[15]=1;
      }
      if (xr) {
        proj.set(xr.proj);
        view.set(xr.view);
      } else {
        perspective(proj, 0.70, w / h, 0.01, 8);
        lookAt(view, eye, [cx, cy, 0.13], [0, 0, 1]);
      }

      // The grid stands in for a floor. In AR the room already has one, so
      // drawing ours would put a glowing lattice over the carpet.
      if (!xr) {
      gl.useProgram(lineProg);
      gl.uniformMatrix4fv(lLoc.view, false, view);
      gl.uniformMatrix4fv(lLoc.proj, false, proj);
      gl.uniform3fv(lLoc.color, opts.grid);
      gl.uniform3fv(lLoc.fog, opts.bg);
      // The grid follows the duck, snapped to its own spacing, so the floor
      // reads as ground moving past rather than sliding out of frame.
      const snapX = Math.round(cx / S) * S, snapY = Math.round(cy / S) * S;
      for (let i = 0; i < gridVerts.length; i += 3) {
        gridMoved[i] = gridVerts[i] + snapX;
        gridMoved[i+1] = gridVerts[i+1] + snapY;
        gridMoved[i+2] = 0;
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, gridBuf);
      gl.bufferData(gl.ARRAY_BUFFER, gridMoved, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(lAPos);
      gl.vertexAttribPointer(lAPos, 3, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.LINES, 0, gridMoved.length / 3);
      }

      // the duck
      gl.useProgram(prog);
      gl.uniformMatrix4fv(loc.view, false, view);
      gl.uniformMatrix4fv(loc.proj, false, proj);
      gl.uniform3fv(loc.fog, opts.bg);
      gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
      gl.enableVertexAttribArray(aPos); gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, nrmBuf);
      gl.enableVertexAttribArray(aNrm); gl.vertexAttribPointer(aNrm, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);

      for (const d of meta.draws) {
        const b = d.body;
        fromQuatPos(bodyM,
          [data.xquat[b*4], data.xquat[b*4+1], data.xquat[b*4+2], data.xquat[b*4+3]],
          [data.xpos[b*3], data.xpos[b*3+1], data.xpos[b*3+2]]);
        fromQuatPos(localM, d.quat, d.pos);
        multiply(modelM, bodyM, localM);
        if (xr) { multiply(tmpM, placeM, modelM); modelM.set(tmpM); }
        gl.uniformMatrix4fv(loc.model, false, modelM);
        gl.uniform3f(loc.color, d.rgba[0], d.rgba[1], d.rgba[2]);
        const m = meta.meshes[d.mesh];
        gl.drawElements(gl.TRIANGLES, m.fn * 3, gl.UNSIGNED_INT, (m.f * 3) * 4);
      }

      // Everything in the model that is NOT a mesh: the stairs, the ball, the
      // blocks, the cones. Before this pass the renderer only knew about the
      // duck's own 70 meshes, so a staircase was solid to walk on and entirely
      // invisible — which is exactly how it was reported.
      if (opts.model && opts.geomTypes) {
        const m = opts.model, T = opts.geomTypes;
        gl.bindBuffer(gl.ARRAY_BUFFER, pPosBuf);
        gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, pNrmBuf);
        gl.vertexAttribPointer(aNrm, 3, gl.FLOAT, false, 0, 0);
        for (let g = 0; g < m.ngeom; g++) {
          const t = m.geom_type[g];
          const isBox = t === T.box, isSphere = t === T.sphere, isCap = t === T.capsule;
          if (!isBox && !isSphere && !isCap) continue;
          const z = data.geom_xpos[g * 3 + 2];
          if (z < -1) continue;                       // parked below the floor
          const sx = m.geom_size[g*3], sy = m.geom_size[g*3+1], sz = m.geom_size[g*3+2];
          // MuJoCo sizes: box is half-extents, sphere uses size[0], a capsule is
          // radius plus half-length. A capsule is drawn as a stretched sphere —
          // at prop scale the difference is not visible.
          const hx = isBox ? sx : sx, hy = isBox ? sy : sx, hz = isBox ? sz : (isCap ? sx + sy : sx);
          // Orientation from MuJoCo's own world matrix. geom_xmat is row-major
          // 3x3; the shader wants column-major 4x4.
          const M = data.geom_xmat, o = g * 9;
          bodyM[0]=M[o]*hx;   bodyM[1]=M[o+3]*hx; bodyM[2]=M[o+6]*hx; bodyM[3]=0;
          bodyM[4]=M[o+1]*hy; bodyM[5]=M[o+4]*hy; bodyM[6]=M[o+7]*hy; bodyM[7]=0;
          bodyM[8]=M[o+2]*hz; bodyM[9]=M[o+5]*hz; bodyM[10]=M[o+8]*hz; bodyM[11]=0;
          bodyM[12]=data.geom_xpos[g*3]; bodyM[13]=data.geom_xpos[g*3+1]; bodyM[14]=z; bodyM[15]=1;
          if (xr) { multiply(tmpM, placeM, bodyM); bodyM.set(tmpM); }
          gl.uniformMatrix4fv(loc.model, false, bodyM);
          gl.uniform3f(loc.color, m.geom_rgba[g*4], m.geom_rgba[g*4+1], m.geom_rgba[g*4+2]);
          if (isBox) gl.drawArrays(gl.TRIANGLES, BOX_AT, BOX_N);
          else gl.drawArrays(gl.TRIANGLES, SPH_AT, SPH_N);
        }
      }
    },
  };
}
