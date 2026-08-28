import io, re, json
src = io.open("pollen_rollers.xml", encoding="utf-8").read()

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
io.open("visual_geoms_rollers.json", "w").write(json.dumps(visual))
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

# A fixed bank of stair boxes, each on its own x and z slide joints.
#
# Two things forced this shape. Geometry cannot be created at runtime — a meshed
# MJCF will not compile in the browser, it dies in MuJoCo's threaded convex-hull
# pass. And moving a STATIC geom by writing model.geom_pos does not work either:
# measured, the duck walks straight through a platform placed that way, even
# with geom_rbound corrected. Position that comes from qpos IS live, which is
# why Pollen drive their own terrain the same way. Two joints per step means the
# rise AND the run are both adjustable; parking a step far below removes it.
STAIRS = 14
# These MUST match site/stairs.js — STEP_HALF_DEPTH, STAIR_HALF_WIDTH,
# STEP_HALF_HEIGHT and STAIR_Y. The rollers scene positions its steps with the
# same layoutStairs(), so a step sized differently here lands somewhere the
# shared code does not expect. It already had: 60x160x25 mm blocks on the
# centreline while the walking scene used 170x170x100 mm flush to the wall.
STAIR_Y = 1.5 - 0.025 - 0.17

steps = "".join(
    f'''
    <body name="step{i}" pos="0 {STAIR_Y} 0">
      <joint name="step{i}_x" type="slide" axis="1 0 0" limited="false" damping="0" armature="0" frictionloss="0"/>
      <joint name="step{i}_z" type="slide" axis="0 0 1" limited="false" damping="0" armature="0" frictionloss="0"/>
      <geom name="step{i}_geom" type="box" size="0.17 0.17 0.10" pos="0 0 0"
            contype="4" conaffinity="4" condim="3" friction="1.0 0.02 0.001"
            rgba="0.62 0.65 0.61 1" mass="200"/>
    </body>'''
    for i in range(STAIRS))

# Things to knock about and pick up. All free bodies, so the duck can actually
# shove them: a ball, some blocks, a couple of cones. MuJoCo has no cone
# primitive, so a cone is a short capsule — close enough at this scale and it
# rolls the way a skittle does.
PROPS = """
    <body name="ball" pos="0.55 0.10 0.036">
      <freejoint name="ball_free"/>
      <geom name="ball_geom" type="sphere" size="0.035" mass="0.045"
            contype="1" conaffinity="5" condim="4" friction="0.7 0.02 0.002"
            rgba="0.96 0.96 0.94 1"/>
    </body>
    <body name="block_a" pos="0.30 0.40 0.021">
      <freejoint name="block_a_free"/>
      <geom name="block_a_geom" type="box" size="0.02 0.02 0.02" mass="0.030"
            contype="1" conaffinity="5" condim="4" friction="0.9 0.02 0.002"
            rgba="0.85 0.30 0.24 1"/>
    </body>
    <body name="block_b" pos="0.42 -0.36 0.019">
      <freejoint name="block_b_free"/>
      <geom name="block_b_geom" type="box" size="0.018 0.018 0.018" mass="0.025"
            contype="1" conaffinity="5" condim="4" friction="0.9 0.02 0.002"
            rgba="0.25 0.45 0.75 1"/>
    </body>
    <body name="block_c" pos="0.62 -0.14 0.016">
      <freejoint name="block_c_free"/>
      <geom name="block_c_geom" type="box" size="0.015 0.015 0.015" mass="0.018"
            contype="1" conaffinity="5" condim="4" friction="0.9 0.02 0.002"
            rgba="0.92 0.72 0.20 1"/>
    </body>
    <body name="cone_a" pos="0.20 -0.44 0.038">
      <freejoint name="cone_a_free"/>
      <geom name="cone_a_geom" type="capsule" size="0.016 0.022" mass="0.020"
            contype="1" conaffinity="5" condim="4" friction="0.8 0.02 0.002"
            rgba="0.94 0.45 0.13 1"/>
    </body>
    <body name="cone_b" pos="0.20 0.46 0.038">
      <freejoint name="cone_b_free"/>
      <geom name="cone_b_geom" type="capsule" size="0.016 0.022" mass="0.020"
            contype="1" conaffinity="5" condim="4" friction="0.8 0.02 0.002"
            rgba="0.94 0.45 0.13 1"/>
    </body>
"""

add = '''  <option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast" />
  <worldbody>
    <geom name="floor" type="plane" size="6 6 0.05" pos="0 0 0" rgba="0.16 0.18 0.16 1"
          contype="1" conaffinity="1" condim="3" friction="1.0 0.02 0.001" />%s%s
  </worldbody>
''' % (steps, PROPS if __import__('os').environ.get('PROPS','1')=='1' else '')
phys = phys.replace("<worldbody>", add + "  <worldbody>", 1)
# The step blocks are tall and sit buried in the floor plane, which as a
# collision pair generates enormous forces and destabilises the whole solve —
# measured, a 1 mm step toppled a duck that walks 3 m on the flat. So steps live
# on collision bit 2 and the floor on bit 1, which never meet, while the duck's
# own collision geoms answer to both.
# Bit 4 for the stairs, bit 1 for the floor and props, and the duck answers to
# both (1|4 = 5). NOT bit 2: Pollen reserve that for their `self_collision_only`
# geoms, and giving the duck's main collision geoms mask 3 made those two sets
# start colliding — the duck fought itself and collapsed at spawn.
phys = re.sub(r'(<default class="collision">\s*<geom group="3")(/>)',
              r'\1 contype="5" conaffinity="5"\2', phys)

io.open("scene_physics_rollers.xml", "w", encoding="utf-8").write(phys)
print("physics meshes kept:", len(kept), "->", sorted(kept))
print("scene_physics_rollers.xml", len(phys), "bytes")
