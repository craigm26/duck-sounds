"""The flamingo where a flamingo belongs: Miami Beach at dusk.

The motion is real — RemiFabre/microduck-flamingo-cycle, recorded on the canon
plant, both sides, 15 seconds. Only the scenery is invented, and it is invented
in the RENDER scene alone: the physics that produced the motion never saw a
beach, a palm or the sea.

The scene is laid out around the camera move. The orbit runs azimuth 150 -> 215
once per clip, which always looks toward +X, so the ocean, the surf and the low
sun all live out that way and the palms flank the frame at +/-Y.
"""
import json, math, os, subprocess, sys
import mujoco
from PIL import Image

SIM = os.path.dirname(os.path.abspath(__file__))
# The backdrop plate, cut from the reference photograph at its sand line and
# widened to a letterbox — a ground plane under a low camera always takes the
# lower half of the frame, so only a wide band of the photo is ever visible.
PLATE_SOURCE = f"{SIM}/plates/oceandrive.jpg"
PLATE_PNG = "/tmp/oceandrive_plate.png"
PLATE_CROP = (0, 132, 588, 380)


def make_plate() -> None:
    if os.path.exists(PLATE_PNG):
        return
    plate = Image.open(PLATE_SOURCE).convert("RGB").crop(PLATE_CROP)
    plate = plate.resize((plate.width * 3, plate.height * 3), Image.LANCZOS)
    plate.save(PLATE_PNG)

W, H, FPS = 1280, 720, 50

ASSETS = """
    <texture name="platetex" type="2d" file="PLATE_PNG"/>
    <material name="platemat" texture="platetex" texuniform="false"
              texrepeat="1 1" rgba="1 1 1 1" emission="1" specular="0" shininess="0"/>
    <texture name="sandtex" type="2d" builtin="flat" rgb1="0.378 0.418 0.633"
             rgb2="0.378 0.418 0.633" mark="random" markrgb="0.32 0.36 0.56"
             random="0.28" width="512" height="512"/>
    <material name="sandmat" texture="sandtex" texrepeat="90 90"
              specular="0.05" shininess="0.04"/>
"""

# The sun sits low over the water; the sky fills from above, and a weak bounce
# keeps the duck's shaded side from going black.
LIGHTS = """
    <light name="ambientsky" pos="0 0 6" dir="0 0 -1" directional="true"
           castshadow="false" diffuse="0.42 0.44 0.58" specular="0.05 0.05 0.08"/>
    <light name="stripglow" pos="-4 0.6 0.9" dir="0.94 -0.14 -0.31" directional="true"
           diffuse="0.55 0.42 0.40" specular="0.30 0.24 0.26"/>
    <light name="rim" pos="1.6 -1.4 0.7" dir="-0.7 0.62 -0.35" directional="true"
           castshadow="false" diffuse="0.20 0.22 0.34"/>
"""

# THE CAMERA LOOKS TOWARD -X. MuJoCo's azimuth put the lens on the +X side at
# 150-215 degrees, so the first version of this scene built an entire sea
# behind the camera and rendered a desert; only the palms' shadows gave it
# away. Everything the shot needs lives at -X.
# Close to the axis of view AND behind the waterline, or they simply are not
# in the picture: the lens is about 78 degrees wide, so a palm 1.7 m off to
# the side of a duck 0.8 m away sits well outside the frame.
# Far enough back that a whole palm fits the lens. At 1 m they were diagonal
# poles crossing the frame with their fronds above the top edge.
# THE BACKDROP IS THE PHOTOGRAPH. MuJoCo's renderer was never going to beat a
# real picture of Ocean Drive at dusk, so it does not try: the strip, the palms
# and the sky are the reference plate standing at PLATE_DISTANCE with its foot
# on the horizon, and only the sand and the robot are rendered. The camera is
# fixed for the same reason — a moving lens would show the plate is flat.
PLATE_DISTANCE = 26.0
PLATE_ASPECT = 588 / 248.0


def backdrop() -> str:
    """A billboard wide enough to fill the frame, its base on the ground."""
    half_width = PLATE_DISTANCE * math.tan(math.radians(39.0)) * 1.06
    half_height = half_width / PLATE_ASPECT
    return (f'<body name="plate" pos="{-PLATE_DISTANCE} 0 {half_height:.3f}">'
            f'<geom type="box" size="0.02 {half_width:.3f} {half_height:.3f}"'
            f' material="platemat" contype="0" conaffinity="0"'
            f' euler="0 0 0"/></body>')


def build(path: str) -> str:
    import re
    make_plate()
    src = open(f"{SIM}/scene_pollen.xml").read().replace(
        'meshdir="assets"', f'meshdir="{SIM}/assets"', 1)
    src = src.replace("<asset>", "<asset>" + ASSETS.replace("PLATE_PNG", PLATE_PNG), 1)
    # The sand runs from under the robot all the way to the strip: in the
    # reference the beach fills the foreground and the hotels sit on it.
    src = re.sub(r'(<geom name="floor"[^>]*?)rgba="[^"]*"',
                 r'\1material="sandmat"', src, count=1)
    src = re.sub(r'(<geom name="floor"[^>]*?)size="[^"]*"',
                 r'\1size="60 60 0.05"', src, count=1)
    src = re.sub(r'(<geom name="floor"[^>]*?)pos="[^"]*"',
                 r'\1pos="0 0 0"', src, count=1)
    scenery = LIGHTS + "\n    " + backdrop()
    src = src.replace("<worldbody>", "<worldbody>\n    " + scenery, 1)
    statistic = '<statistic extent="0.45" center="0 0 0.11"/>'
    import re as _re
    if "<statistic" in src:
        src = _re.sub(r"<statistic[^>]*/>", statistic, src, count=1)
    else:
        src = src.replace("<worldbody>", statistic + "\n  <worldbody>", 1)
    # Haze gives the distance its depth; without it the skyline sits on the duck.
    #
    # AND THE CLIPPING PLANES ARE PINNED. znear and zfar are multipliers of the
    # model's extent, and putting an ocean 40 m away made that extent enormous —
    # which pushed the near plane out to about 0.4 m and clipped away a 25 cm
    # duck filmed from 0.62 m. The duck vanished and left only its shadow. So
    # the statistics are stated for the SUBJECT, and the far plane is opened up
    # by hand to keep the sea.
    visual = (f'<global offwidth="{W}" offheight="{H}"/>'
              '<rgba haze="0.93 0.78 0.66 1"/><map haze="0.006" znear="0.004" zfar="400"/>')
    if "<visual>" in src:
        src = src.replace("<visual>", "<visual>" + visual, 1)
    else:
        src = src.replace("<worldbody>", f"<visual>{visual}</visual>\n  <worldbody>", 1)
    open(path, "w").write(src)
    return path


def camera_for(phase: float, root) -> mujoco.MjvCamera:
    """Fixed, low and level — a photograph's camera, not a fly-around.

    The backdrop is flat, so any orbit would betray it. Sitting low puts the
    robot big in the lower half with the strip behind, which is the shot.
    """
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(camera_for.model, camera)
    camera.azimuth = 180.0
    camera.elevation = 1.5
    camera.distance = 0.55
    camera.lookat[:] = [root[0], root[1], 0.115]
    return camera


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/flamingo_miami.mp4"
    model = mujoco.MjModel.from_xml_path(build("/tmp/flamingo_beach.xml"))
    data = mujoco.MjData(model)
    camera_for.model = model
    clips = json.load(open(f"{SIM}/duck-intent-clips.json"))
    order = clips["joints"]
    duck = model.jnt_qposadr[model.joint("trunk_base_freejoint").id]
    renderer = mujoco.Renderer(model, H, W)

    frames_dir = "/tmp/flamingo_frames"
    os.makedirs(frames_dir, exist_ok=True)
    for stale in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, stale))

    index = 0
    for name in ("flamingo_left", "flamingo_right"):
        clip = clips["clips"][name]
        span = max(len(clip["frames"]) - 1, 1)
        for within, (frame, root) in enumerate(zip(clip["frames"], clip["roots"])):
            data.qpos[:] = 0
            data.qpos[duck:duck + 7] = root[:7]
            for joint, value in zip(order, frame):
                data.qpos[model.jnt_qposadr[model.joint(joint).id]] = value
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera_for(within / span, root))
            Image.fromarray(renderer.render()).save(f"{frames_dir}/f{index:05d}.png")
            index += 1
    renderer.close()
    print(f"rendered {index} frames")

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", f"{frames_dir}/f%05d.png", "-c:v", "libx264", "-preset", "slow",
        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out,
    ], check=True)
    print(f"wrote {out} — {os.path.getsize(out)/1e6:.2f} MB, {index/FPS:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
