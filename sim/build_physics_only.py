import io, re, json
src = io.open("pollen_robot.xml", encoding="utf-8").read()

# Pull out the visual geoms BEFORE stripping them — the renderer needs to know
# which mesh hangs off which body, and at what local offset.
visual = []
body_re = re.compile(r'<body\s+name="([^"]+)"')
pos = 0
current = None
for m in re.finditer(r'<body\s+name="([^"]+)"|<geom\b[^>]*/>', src):
    tok = m.group(0)
    if tok.startswith('<body'):
        current = m.group(1)
    elif 'class="visual"' in tok:
        g = dict(re.findall(r'(\w+)="([^"]*)"', tok))
        if 'mesh' in g:
            visual.append({
                "body": current,
                "mesh": g["mesh"],
                "pos": g.get("pos", "0 0 0"),
                "quat": g.get("quat", "1 0 0 0"),
            })
io.open("visual_geoms.json", "w").write(json.dumps(visual))
print("visual geoms extracted:", len(visual))

# Physics scene: drop every visual geom, then drop mesh assets nobody references.
phys = re.sub(r'\s*<geom\b[^>]*class="visual"[^>]*/>', '', src)
kept = set(re.findall(r'mesh="([^"]+)"', phys))
def drop_mesh(m):
    # These <mesh> elements carry no name attribute — MuJoCo derives the mesh
    # name from the filename, so that is what has to be matched against.
    tag = m.group(0)
    name = re.search(r'name="([^"]+)"', tag)
    if not name:
        f = re.search(r'file="([^"]+)\.stl"', tag)
        return tag if (f and f.group(1) in kept) else ''
    return tag if name.group(1) in kept else ''
phys = re.sub(r'\s*<mesh\b[^>]*/>', drop_mesh, phys)
# ...and materials, which reference textures we are not loading either
phys = re.sub(r'\s*<material\b[^>]*/>', '', phys)
phys = re.sub(r'\s+material="[^"]*"', '', phys)

add = '''  <option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast" />
  <worldbody>
    <geom name="floor" type="plane" size="6 6 0.05" pos="0 0 0" rgba="0.16 0.18 0.16 1"
          contype="1" conaffinity="1" condim="3" friction="1.0 0.02 0.001" />
  </worldbody>
'''
phys = phys.replace("<worldbody>", add + "  <worldbody>", 1)
io.open("scene_physics.xml", "w", encoding="utf-8").write(phys)
print("physics meshes kept:", len(kept), "->", sorted(kept))
print("scene_physics.xml", len(phys), "bytes")
