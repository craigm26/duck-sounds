import io, re, json
src = io.open("robot_walk.xml", encoding="utf-8").read()
C = json.load(open("duckkit-constants.json"))
policy_joints = [n for n in C["jointNames"] if n != "mouth"]

# 1 — simulation options + contact defaults. timestep 0.005 with decimation 4
#     gives the policy its 50 Hz, which is the cadence upstream states.
opts = '''  <option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast" />
  <default>
    <geom contype="1" conaffinity="1" condim="3" friction="1.0 0.02 0.001" rgba="0.85 0.72 0.25 1" />
    <joint damping="0.06" armature="0.008" frictionloss="0.002" />
  </default>
'''
src = src.replace("<worldbody>", opts + "  <worldbody>", 1)

# 2 — a floor to walk on
floor = '''
    <geom name="floor" type="plane" size="4 4 0.05" pos="0 0 0" rgba="0.16 0.18 0.16 1"
          contype="1" conaffinity="1" condim="3" friction="1.0 0.02 0.001" />
'''
src = src.replace("<worldbody>", "<worldbody>" + floor, 1)

# 3 — feet. The site positions are the model's own; the box around each is OURS.
for side, site in (("left", "left_foot"), ("right", "right_foot")):
    m = re.search(r'(<site[^>]*name="%s"[^>]*pos="([^"]+)"[^>]*/>)' % site, src)
    if not m:
        raise SystemExit("foot site not found: " + site)
    pos = m.group(2)
    geom = (m.group(1) +
            '\n                <geom name="%s_sole" type="box" size="0.026 0.017 0.006" pos="%s" '
            'rgba="0.75 0.62 0.20 1" />' % (side, pos))
    src = src.replace(m.group(1), geom, 1)

# 4 — position actuators, one per policy joint, in DuckKit's order.
acts = ['  <actuator>']
for j in policy_joints:
    acts.append('    <position name="%s_act" joint="%s" kp="9.0" kv="0.25" forcerange="-2.5 2.5" />' % (j, j))
acts.append('  </actuator>')
src = src.replace("</mujoco>", "\n".join(acts) + "\n</mujoco>", 1)

io.open("scene.xml", "w", encoding="utf-8").write(src)
print("scene.xml written; actuators:", len(policy_joints))
