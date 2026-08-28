// Pull the visual geometry straight out of the compiled model, so the shapes
// and their placements are MuJoCo's own rather than anything re-derived.
import load from 'mujoco';
import fs from 'node:fs';
const mj = await load();
try { mj.FS.mkdir('/assets'); } catch {}
for (const f of fs.readdirSync('assets')) mj.FS.writeFile('/assets/' + f, fs.readFileSync('assets/' + f));
mj.FS.writeFile('/scene.xml', fs.readFileSync('scene_pollen.xml', 'utf8'));
const m = mj.MjModel.mj_loadXML('/scene.xml');

const vert = m.mesh_vert, face = m.mesh_face;
const positions = new Float32Array(m.nmeshvert * 3);
for (let i = 0; i < m.nmeshvert * 3; i++) positions[i] = vert[i];
// MuJoCo's mesh_face indices are LOCAL to each mesh (0-based within it). This
// packs every mesh into one vertex buffer, so each index has to be shifted by
// that mesh's vertex offset — without it the draws pull vertices belonging to
// other parts and the duck renders as an exploded diagram of itself.
const indices = new Uint32Array(m.nmeshface * 3);
for (let mi = 0; mi < m.nmesh; mi++) {
  const fa = m.mesh_faceadr[mi], fn = m.mesh_facenum[mi], va = m.mesh_vertadr[mi];
  for (let f = fa; f < fa + fn; f++) {
    indices[f*3]   = face[f*3]   + va;
    indices[f*3+1] = face[f*3+1] + va;
    indices[f*3+2] = face[f*3+2] + va;
  }
}

// Smooth normals, accumulated per vertex from face normals.
const normals = new Float32Array(m.nmeshvert * 3);
for (let f = 0; f < m.nmeshface; f++) {
  const a = indices[f*3], b = indices[f*3+1], c = indices[f*3+2];
  const ax=positions[a*3],ay=positions[a*3+1],az=positions[a*3+2];
  const bx=positions[b*3],by=positions[b*3+1],bz=positions[b*3+2];
  const cx=positions[c*3],cy=positions[c*3+1],cz=positions[c*3+2];
  const ux=bx-ax, uy=by-ay, uz=bz-az, vx=cx-ax, vy=cy-ay, vz=cz-az;
  const nx=uy*vz-uz*vy, ny=uz*vx-ux*vz, nz=ux*vy-uy*vx;
  for (const k of [a,b,c]) { normals[k*3]+=nx; normals[k*3+1]+=ny; normals[k*3+2]+=nz; }
}
for (let i = 0; i < m.nmeshvert; i++) {
  const l = Math.hypot(normals[i*3], normals[i*3+1], normals[i*3+2]) || 1;
  normals[i*3]/=l; normals[i*3+1]/=l; normals[i*3+2]/=l;
}

// Mesh table, and the draw list: every visual geom with its body and offset.
const meshes = [];
for (let i = 0; i < m.nmesh; i++) {
  meshes.push({ v: m.mesh_vertadr[i], vn: m.mesh_vertnum[i], f: m.mesh_faceadr[i], fn: m.mesh_facenum[i] });
}
// Colours live in <material>, which the compiled model does not attach to
// geom_rgba — so they are read from the XML. Every material here is named
// "<mesh>_material", and every visual geom names one, so mesh -> colour is a
// direct lookup and stays the robot's own palette rather than my guess at it.
const xml = fs.readFileSync('scene_pollen.xml', 'utf8');
const matRgba = new Map();
for (const mm of xml.matchAll(/<material\s+name="([^"]+)"\s+rgba="([^"]+)"/g)) {
  matRgba.set(mm[1], mm[2].trim().split(/\s+/).map(Number));
}
// mesh id follows <mesh file="..."> document order
const meshNames = [...xml.matchAll(/<mesh\s+file="([^"]+)\.stl"/g)].map(x => x[1]);
console.log('materials:', matRgba.size, ' mesh names:', meshNames.length);

const draws = [];
for (let g = 0; g < m.ngeom; g++) {
  if (m.geom_type[g] !== mj.mjtGeom.mjGEOM_MESH.value) continue;
  if (m.geom_contype[g] !== 0) continue;              // visual geoms only
  draws.push({
    // Store the body's NAME, not its index. Indices are a property of the
    // scene, and the scene grew stairs and props whose bodies sit ahead of the
    // duck's — three separate bugs in this project have come from an index
    // that was correct when it was written down.
    mesh: m.geom_dataid[g], body: m.geom_bodyid[g],
    bodyName: m.body(m.geom_bodyid[g]).name,
    pos: [m.geom_pos[g*3], m.geom_pos[g*3+1], m.geom_pos[g*3+2]],
    quat: [m.geom_quat[g*4], m.geom_quat[g*4+1], m.geom_quat[g*4+2], m.geom_quat[g*4+3]],
    rgba: matRgba.get((meshNames[m.geom_dataid[g]] || '') + '_material')
          || [m.geom_rgba[g*4], m.geom_rgba[g*4+1], m.geom_rgba[g*4+2], m.geom_rgba[g*4+3]],
  });
}
console.log('visual mesh geoms:', draws.length, 'of', m.ngeom);

let header = Buffer.from(JSON.stringify({ nvert: m.nmeshvert, nface: m.nmeshface, meshes, draws }), 'utf8');
// Pad so the float arrays that follow start on a 4-byte boundary: a typed
// array view over a misaligned offset throws outright.
while ((4 + header.length) % 4 !== 0) header = Buffer.concat([header, Buffer.from(' ')]);
const len = Buffer.alloc(4); len.writeUInt32LE(header.length, 0);
fs.writeFileSync('duck-visual.bin', Buffer.concat([
  len, header, Buffer.from(positions.buffer), Buffer.from(normals.buffer), Buffer.from(indices.buffer),
]));
console.log('SAVED duck-visual.bin', (fs.statSync('duck-visual.bin').size/1048576).toFixed(2), 'MB');
