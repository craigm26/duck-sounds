"""The duck's own eye, rendered — and a Hailo-8 looking through it.

WHY THIS EXISTS. Every intent in the corpus is BLIND: a policy plus a command
schedule, recorded open-loop. The robot has a camera, Pollen ship a detector
for it, and this machine has a Hailo-8 — so this is the bench for the first
PERCEPTION-CONDITIONED intents, where what the duck does depends on what it
sees.

WHAT IS POLLEN'S AND WHAT IS OURS
  * The optical frame is theirs. `microduck/kinematics/src/head.rs` carries
    `SITE_TO_CV2 = (0.5, -0.5, 0.5, -0.5)`: the MJCF `head_camera` SITE, not
    the `<camera>` element, composed with that constant, is the frame their
    vision stack speaks (cv2 axes: +x right, +y down, +z forward). The
    `<camera>` element that onshape-to-robot's post-import injects
    (`quat="0 0 -1 0"`, see config_mjcf_allcollisions*.json) points BACKWARD
    and rolled 90 degrees — measured here, and it is not what their code uses.
    This module rebuilds the camera from the site and their constant.
  * The sensor is theirs: an IMX219 at 720x1280 PORTRAIT, and
    `microduck/docs/project/media-bringup.md` says plainly that "nothing in
    the pipeline rotates" — the mount is reported, not corrected. So frames
    are rendered at that shape and letterboxed square exactly as
    `duck-detect/src/lib.rs::letterbox_rgb` does (PAD=114, nearest-neighbour).
  * The DETECTOR here is a stand-in. Pollen's own `duck-detect` is a
    single-class 320x320 model shipped as .onnx and .rknn (their Radxa NPU);
    it cannot be compiled for a Hailo on this machine, because the Hailo
    Dataflow Compiler is x86-64 only and is not installed. So this runs the
    stock COCO yolov8s HEF that ships with hailo-models. Its `bearing` is the
    same primitive `duck-detect`'s `Detection::bearing(frame_width)` gives.
    (`pet-detect`, despite the name, is AUDIO — a petting classifier over
    log-mel windows — and is nothing to do with this.)

MEASURED ON THIS MACHINE, 2026-08-29 (Pi 5 + Hailo-8, hailort 4.23):
  * yolov8s on the Hailo: 12.3 ms per frame, 82 FPS. The policy MLP is not
    worth accelerating for comparison: at 0.098 ms against 1.66 ms of physics
    it is 5.6% of a sim tick, so the accelerator earns its place on
    perception, not on the control loop.
  * The duck's horizontal field of view is 26 degrees. MuJoCo's fovy is
    VERTICAL (45 degrees) and the sensor is portrait, so horizontally it is
    2*atan(tan(22.5) * 720/1280) = 26.2. A ball 20 degrees off the nose is
    invisible: the duck must turn to search.
  * Head level, the nearest visible ground is 0.61 m away (camera at 0.251 m,
    22.5 degrees down to the bottom edge). A ball at the duck's feet can only
    be seen by pitching the head down.
  * neck_pitch and head_pitch are OPPOSED: positive neck_pitch looks up,
    positive head_pitch looks down, and the home pose (both +0.349) is a level
    gaze. head_pitch 0.6 / 0.9 / 1.2 looks down 14.4 / 31.6 / 48.8 degrees.
  * Detection of the 36 mm ball: 0.86 confidence at 0.8 m, 0.26 at 0.45 m,
    nothing at 1.5 m. COCO calls it class 49, "orange" — a smooth orange
    sphere has none of a sports ball's panel seams. The intent wants the BOX,
    not the label, so nothing keys on the class id.

WHAT THIS DOES NOT DO. It does not step physics. Rendering runs on MuJoCo
3.12 (pip) while the canon recorders run MuJoCo 3.5.1 (WASM) against
scene.mjb, and a clip recorded on a different engine would not be canon.
Drawing is not physics: this loads the same plant purely to place and
photograph it. A closed-loop intent therefore drives physics from the Node
harness and asks this for perception.
"""
from __future__ import annotations
import json, os, re, sys
import numpy as np

SIM = os.path.dirname(os.path.abspath(__file__))
# Pollen's translation layer, verbatim (microduck kinematics/src/head.rs).
SITE_TO_CV2 = (0.5, -0.5, 0.5, -0.5)
# cv2 axes -> MuJoCo camera axes (+x right, +y up, looks down -z): 180 about x.
RX180 = (0.0, 1.0, 0.0, 0.0)
SENSOR_W, SENSOR_H = 720, 1280          # IMX219 as the duck runs it, portrait
LETTERBOX_PAD = 114                     # duck-detect/src/lib.rs PAD
HEF_PATH = "/usr/share/hailo-models/yolov8s_h8.hef"
# THE LENS IS THE SENSOR'S, NOT MUJOCO'S DEFAULT. Pollen's MJCF declares the
# head_camera with a pose and no `fovy`, so a render inherits MuJoCo's default
# 45 degrees vertical — which on a portrait frame is 26.2 degrees across, and
# is nothing to do with the robot. The module is an IMX219 (Pi Cam v2,
# `microduck/docs/project/media-bringup.md`), nominally 62.2 x 48.8 degrees;
# rokbenko/quackd independently uses fov_deg=62 for a real one. The frame is
# portrait, so the sensor's long axis runs up the image and FOVY is 62.2.
#
# UNCONFIRMED: whether the mount is rotated 90 or 180 degrees. media-bringup
# records `rotation: 180` on the alpha ("the IMX219 is mounted upside down")
# while duck-detect letterboxes a 720x1280 portrait frame, which implies a
# quarter turn somewhere. Both readings give the same VERTICAL angle; they
# differ in which way the image is up. Check against hardware before trusting
# a bearing's sign.
SENSOR_FOVY_DEGREES = 62.2


def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return (w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2)


def build_render_scene(out_path: str, ball_rgba="0.93 0.42 0.09 1") -> str:
    """A RENDER-ONLY scene: the visual plant, lights, and a ball.

    scene_pollen.xml carries the meshes; scene_physics.xml is the stripped
    physics build and draws no duck at all. Lights are added because the
    physics scene has none (nlight 0) and an unlit sphere is invisible to the
    detector — measured: a flat-shaded ball scores nothing, a lit one 0.80.
    """
    src = open(f"{SIM}/scene_pollen.xml").read().replace(
        'meshdir="assets"', f'meshdir="{SIM}/assets"', 1)
    extra = f'''
    <light name="key" pos="0.6 0.5 1.2" dir="-0.4 -0.35 -1" directional="true"
           diffuse="0.85 0.85 0.82" specular="0.2 0.2 0.2"/>
    <light name="fill" pos="-0.8 -0.6 0.9" dir="0.5 0.4 -1" directional="true"
           diffuse="0.35 0.38 0.45"/>
    <body name="seek_ball" pos="0.45 0 0.036">
      <freejoint/>
      <geom name="seek_ball_geom" type="sphere" size="0.036" rgba="{ball_rgba}"/>
    </body>
'''
    src = src.replace("<worldbody>", "<worldbody>" + extra, 1)
    src = src.replace("<visual>",
                      f'<visual>\n    <global offwidth="{SENSOR_W}" offheight="{SENSOR_H}"/>', 1) \
        if "<visual>" in src else src.replace(
            "<worldbody>",
            f'<visual><global offwidth="{SENSOR_W}" offheight="{SENSOR_H}"/></visual>\n  <worldbody>', 1)
    # The camera, rebuilt from the SITE and Pollen's constant.
    site = re.search(r'<site[^>]*name="head_camera"[^>]*quat="([^"]*)"', src)
    sq = tuple(float(v) for v in site.group(1).split()) if site else (0.707107, 0, 0.707107, 0)
    q = qmul(qmul(sq, SITE_TO_CV2), RX180)
    n = sum(c * c for c in q) ** 0.5
    q = " ".join(f"{c/n:.6f}" for c in q)
    src = re.sub(r'(<camera name="head_camera" pos="[^"]*" quat=")[^"]*(")',
                 lambda mm: mm.group(1) + q + mm.group(2), src)
    src = src.replace('<camera name="head_camera"',
                      f'<camera fovy="{SENSOR_FOVY_DEGREES}" name="head_camera"', 1)
    open(out_path, "w").write(src)
    return out_path


class DuckEye:
    """Places the duck and photographs what it sees."""

    def __init__(self, scene_path: str):
        import mujoco
        self.mujoco = mujoco
        self.m = mujoco.MjModel.from_xml_path(scene_path)
        self.d = mujoco.MjData(self.m)
        self.C = json.load(open(f"{SIM}/duckkit-constants.json"))
        self.cam = self.m.camera("head_camera").id
        self.duck = self.m.jnt_qposadr[self.m.joint("trunk_base_freejoint").id]
        ball = [i for i in range(self.m.njnt)
                if self.m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
                and self.m.body(self.m.jnt_bodyid[i]).name == "seek_ball"][0]
        self.ball = self.m.jnt_qposadr[ball]
        self.renderer = mujoco.Renderer(self.m, SENSOR_H, SENSOR_W)

    def place(self, joints=None, root=(0, 0, 0.1231), quat=(1, 0, 0, 0), ball=(0.45, 0.0)):
        """Home pose unless given all 15 joint angles in the wire order."""
        self.d.qpos[:] = 0
        self.d.qpos[self.duck:self.duck + 7] = [*root, *quat]
        angles = joints if joints is not None else self.C["homePose"]
        for name, value in zip(self.C["jointNames"], angles):
            if name == "mouth":
                continue
            self.d.qpos[self.m.jnt_qposadr[self.m.joint(name).id]] = value
        self.d.qpos[self.ball:self.ball + 7] = [ball[0], ball[1], 0.036, 1, 0, 0, 0]
        self.mujoco.mj_forward(self.m, self.d)

    def gaze(self):
        """Where the camera looks, world frame, and how far below the horizon."""
        v = -self.d.cam_xmat[self.cam].reshape(3, 3)[:, 2]
        return v, float(np.degrees(np.arcsin(-v[2])))

    def frame(self) -> np.ndarray:
        self.renderer.update_scene(self.d, camera="head_camera")
        return self.renderer.render()


def letterbox(px: np.ndarray, size: int = 640):
    """Pollen's letterbox: nearest-neighbour, centred, PAD grey.

    Returns the square AND the mapping back to sensor coordinates. The
    mapping is not optional bookkeeping: a 720x1280 portrait frame occupies
    just 360 of the 640 columns, so a box's centre in the square is
    compressed by 0.5625 against the real lens. Reading bearing straight off
    the square under-reports it — measured, a ball truly 10.0 degrees off the
    nose came back as 6.0. Pollen keep a `Letterbox` struct for the same
    reason (duck-detect/src/lib.rs).
    """
    from PIL import Image
    h, w, _ = px.shape
    s = min(size / w, size / h)
    nw, nh = int(w * s), int(h * s)
    im = Image.fromarray(px).resize((nw, nh), Image.NEAREST)
    out = Image.new("RGB", (size, size), (LETTERBOX_PAD,) * 3)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    out.paste(im, (pad_x, pad_y))
    meta = {"size": size, "pad_x": pad_x, "pad_y": pad_y, "fitted_w": nw, "fitted_h": nh}
    # A writeable, C-contiguous buffer: HailoRT refuses a read-only array.
    return np.array(out, dtype=np.uint8, order="C"), meta


class Detector:
    """The Hailo-8, as a context manager. 12.3 ms a frame, measured."""

    def __init__(self, hef_path: str = HEF_PATH):
        self.hef_path = hef_path

    def __enter__(self):
        from contextlib import ExitStack
        from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams,
                                    ConfigureParams, InputVStreamParams,
                                    OutputVStreamParams, FormatType)
        # AN ExitStack, NOT hand-chained __enter__ calls. Entering a context
        # and keeping only its RESULT lets the manager itself be collected,
        # which deactivates the network group under you: the first version of
        # this raised HailoRTNetworkGroupNotActivatedException at infer time.
        self._stack = ExitStack()
        self.hef = HEF(self.hef_path)
        self.target = self._stack.enter_context(VDevice())
        cfg = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.ng = self.target.configure(self.hef, cfg)[0]
        self.iname = self.hef.get_input_vstream_infos()[0].name
        self._stack.enter_context(self.ng.activate(self.ng.create_params()))
        self.pipe = self._stack.enter_context(InferVStreams(
            self.ng,
            InputVStreamParams.make(self.ng, format_type=FormatType.UINT8),
            OutputVStreamParams.make(self.ng, format_type=FormatType.FLOAT32)))
        return self

    def __exit__(self, *exc):
        self._stack.close()
        return False

    hfov_degrees: float = 2 * np.degrees(np.arctan(
        np.tan(np.radians(SENSOR_FOVY_DEGREES / 2)) * SENSOR_W / SENSOR_H))

    def best(self, square: np.ndarray, meta: dict | None = None, floor: float = 0.15):
        """The strongest box in the frame, class-agnostic.

        CLASS-AGNOSTIC ON PURPOSE. The stock COCO model calls our orange
        sphere class 49, "orange"; a real ball with panel seams would be 32,
        "sports ball"; Pollen's own duck-detect has ONE class. An intent that
        keyed on a class id would break on all three.
        """
        out = self.pipe.infer({self.iname: square[None]})
        dets = list(out.values())[0]
        if isinstance(dets, list):
            dets = dets[0]
        best = None
        for cls in range(len(dets)):
            for b in dets[cls]:
                if best is None or b[4] > best["conf"]:
                    ymin, xmin, ymax, xmax = (float(v) for v in b[:4])
                    best = {"conf": float(b[4]), "cls": cls,
                            "cx": (xmin + xmax) / 2, "cy": (ymin + ymax) / 2,
                            "w": xmax - xmin, "h": ymax - ymin}
        if best is None or best["conf"] < floor:
            return None
        # Bearing, as duck-detect's Detection::bearing does it: the horizontal
        # offset from the frame centre, in radians, through the REAL lens —
        # after undoing the letterbox, in sensor coordinates.
        if meta:
            x_px = best["cx"] * meta["size"] - meta["pad_x"]
            best["cx_sensor"] = float(np.clip(x_px / meta["fitted_w"], 0.0, 1.0))
        else:
            best["cx_sensor"] = best["cx"]
        half_h = np.radians(self.hfov_degrees / 2)
        best["bearing"] = float(np.arctan(np.tan(half_h) * (best["cx_sensor"] - 0.5) * 2))
        return best


def main() -> int:
    scene = build_render_scene(os.environ.get("DUCK_RENDER_SCENE", "/tmp/duck_render.xml"))
    eye = DuckEye(scene)
    print(f"scene: {scene}")
    with Detector() as det:
        for label, ball, head in [("0.45 m ahead", (0.45, 0.0), 1.00),
                                  ("0.8 m ahead", (0.80, 0.0), 0.75),
                                  ("0.8 m, 10 deg left", (0.79, 0.14), 0.75),
                                  ("0.8 m, 20 deg left", (0.78, 0.28), 0.75),
                                  ("head level, 0.45 m", (0.45, 0.0), 0.349)]:
            joints = list(eye.C["homePose"])
            joints[eye.C["jointNames"].index("head_pitch")] = head
            eye.place(joints=joints, ball=ball)
            _, below = eye.gaze()
            square, meta = letterbox(eye.frame())
            hit = det.best(square, meta)
            if hit:
                print(f"  {label:22s} head {below:4.1f} deg down -> conf {hit['conf']:.2f} "
                      f"class {hit['cls']} bearing {np.degrees(hit['bearing']):+5.1f} deg")
            else:
                print(f"  {label:22s} head {below:4.1f} deg down -> nothing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
