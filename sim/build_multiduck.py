# build_multiduck.py — N ducks in ONE MuJoCo model.
#
# WHY THIS FILE EXISTS AT ALL, AND WHY IT IS THE PREREQUISITE FOR THE REST.
# A hardware Microduck cannot perceive another Microduck. The observation
# robotd builds is 61 values — 48 of proprioception, 3 of commanded twist, 4 of
# head pose, 6 of body pose (Pollen's `duck-ipc-proto` and the training env,
# read 2026-09-01) — and not one of those slots holds anything about a second
# robot. Two real ducks in a room are two blind agents that happen to share a
# floor; whatever coordination they show has to be smuggled in over a network,
# and the network is where it falls apart: `intents.rs` is last-writer-wins on
# one slot, so two writers at 50 Hz interleave into one slot and produce a robot
# that obeys neither, and the deadman is age-based, so a partition simply stops
# a duck mid-stride.
#
# Ducks in ONE MuJoCo model have none of those problems, and they do perceive
# each other — not through a sensor slot, but through the physics: contact
# forces when they touch, the shared floor they both push against, the
# dynamics of an object one of them lifts while the other holds it. One
# integrator, one clock, no link jitter, no last-writer-wins. So the simulator
# is not a lesser swarm than a room full of hardware. It is the only place a
# swarm can exist right now, and it is the only place a genuinely multi-duck
# policy could ever be trained, because training needs exactly the shared-state
# rollout that the 61-wide hardware observation cannot supply.
#
# HOW. There is one duck in `scene_physics.xml` and every name in it — joint,
# body, site, geom, camera, actuator, sensor — is a global in MJCF's namespace,
# so a second copy collides on all of them. This walks the duck's worldbody
# subtree and prefixes every `name` it holds, then rebuilds the <actuator> and
# <sensor> blocks per duck against the prefixed names. Nothing else is
# duplicated: the <default> classes, the <asset> meshes, the floor, the walls,
# the stair bank, the ball and the graspables are one shared world, which is
# the entire point.
#
# WHAT IS NOT TOUCHED. `scene_physics.xml` itself, and therefore `scene.mjb`.
# Every recorded clip in duckkit claims to have come from that plant and a duck
# added to it would change the physics those recordings claim — the same reason
# `compile_grasp.mjs` builds a separate scene rather than editing the canon one.
#
#   python3 build_multiduck.py                 -> huey, dewey
#   python3 build_multiduck.py huey dewey louie
#   node compile_multiduck.mjs                 -> scene_multiduck.mjb
#   DUCKBENCH_SCENE=scene_multiduck.mjb node duckbench.mjs
import copy
import sys
import xml.etree.ElementTree as ET

SOURCE = "scene_physics.xml"
OUT = "scene_multiduck.xml"

# The three ducks anybody names when asked to name more than one duck. Past
# three the caller has to say the names, because inventing `duck4` here would
# be a name nobody chose showing up in an answer from /health.
DEFAULT_NAMES = ["huey", "dewey", "louie"]

names = sys.argv[1:] or DEFAULT_NAMES[:2]
if len(names) < 2:
    sys.exit("a multi-duck scene needs at least two ducks; scene.mjb is the one-duck plant")
if len(set(names)) != len(names):
    sys.exit("two ducks cannot share a name: " + " ".join(names))
for n in names:
    if not n.replace("_", "").isalnum():
        sys.exit(f"duck name {n!r} must be letters, digits and underscores: it becomes an MJCF prefix")

tree = ET.parse(SOURCE)
root = tree.getroot()

# The duck lives in its own <worldbody>, separate from the one holding the
# floor and the props. Found by the body it contains rather than by being the
# second element, because "the second worldbody" stops being true the first
# time somebody adds a third.
worldbodies = root.findall("worldbody")
duck_world = None
for wb in worldbodies:
    if wb.find('body[@name="trunk_base"]') is not None:
        duck_world = wb
        break
if duck_world is None:
    sys.exit(f"{SOURCE} has no worldbody containing a body named trunk_base")
duck_body = duck_world.find('body[@name="trunk_base"]')

sensors = root.find("sensor")
actuators = root.find("actuator")
if sensors is None or actuators is None:
    sys.exit(f"{SOURCE} is missing its <sensor> or <actuator> block")

# Every attribute that holds a NAME rather than a value, by element. A `mesh`
# attribute names an asset and a `class`/`childclass` names a default, and both
# are shared across all the ducks, so neither is in here — prefixing them would
# ask MuJoCo for meshes and defaults that do not exist.
SENSOR_REFS = ("name", "site", "body", "objname")
ACTUATOR_REFS = ("name", "joint")


def prefix_subtree(element, prefix):
    """Prefix every `name` in this element and everything under it."""
    for node in element.iter():
        if "name" in node.attrib:
            node.set("name", prefix + node.get("name"))


def spawn(index, count):
    """Where duck `index` starts.

    ABREAST ALONG y, ALL FACING +x, half a metre apart. The duck is about
    120 mm tall and stands on soles a few centimetres across, so half a metre
    is far enough that they settle from the drop without touching and near
    enough that a walk of a second or two brings them into contact — which is
    the interaction this whole file is for. The props in the shared world sit
    at |y| >= 0.36 (block_a, block_b, cone_a, cone_b in scene_physics.xml), so
    three ducks abreast still spawn clear of all of them.
    """
    y = (index - (count - 1) / 2) * 0.5
    return f"0 {y:.4f} 0.12"


merged = ET.Element("worldbody")
for i, name in enumerate(names):
    body = copy.deepcopy(duck_body)
    prefix_subtree(body, name + "_")
    body.set("pos", spawn(i, len(names)))
    merged.append(body)

new_sensors = ET.Element("sensor")
for name in names:
    for sensor in sensors:
        s = copy.deepcopy(sensor)
        for attr in SENSOR_REFS:
            if attr in s.attrib:
                s.set(attr, name + "_" + s.get(attr))
        new_sensors.append(s)

new_actuators = ET.Element("actuator")
for name in names:
    for act in actuators:
        a = copy.deepcopy(act)
        for attr in ACTUATOR_REFS:
            if attr in a.attrib:
                a.set(attr, name + "_" + a.get(attr))
        new_actuators.append(a)

# Swap the rebuilt blocks in where the originals were, so the document keeps
# its section order — MJCF tolerates any order, but a diff against
# scene_physics.xml is worth being able to read.
root[list(root).index(duck_world)] = merged
root[list(root).index(sensors)] = new_sensors
root[list(root).index(actuators)] = new_actuators

ET.indent(tree, space="  ")
header = (
    "<!-- GENERATED by build_multiduck.py from %s: %d ducks (%s) in one model. -->\n"
    "<!-- Edit the source, not this file. -->\n" % (SOURCE, len(names), ", ".join(names))
)
xml = ET.tostring(root, encoding="unicode")
with open(OUT, "w", encoding="utf-8") as f:
    f.write('<?xml version="1.0" ?>\n' + header + xml + "\n")

print(f"wrote {OUT}: {len(names)} ducks — {', '.join(names)}")
print(f"  actuators {len(new_actuators)}  sensors {len(new_sensors)}")
print("  next: node compile_multiduck.mjs")
