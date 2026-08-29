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
  * Detection of Pollen's ball (radius 0.05 m, the bench's own): 0.73
    confidence at 0.75 m and 0.45 at 0.44 m, nothing at 21.8 degrees off the
    nose. Bearing is exact to 0.1 degree against ground truth read out of the
    plant; distance from apparent size runs about 7% under, which is the
    "crude but monotonic" quackd warns about and all a steering loop needs.
    COCO calls it class 49, "orange" — a smooth sphere has no panel seams —
    so nothing keys on the class id.
  * quackd's own fields, against ground truth read out of the plant: a ball
    11.1 degrees to the duck's LEFT reports bearing_deg +10.9, 0.2 degrees
    off. est_distance_m reads 0.399 / 0.729 / 0.710 m where the lens is truly
    0.406 / 0.746 / 0.749 m away — 2 to 5% short — and 0.387 against 0.442 m
    for the one case where the ball sits on the bottom edge of the frame
    (cy 0.939) and the box is at its least trustworthy. That cy is itself a
    check on the geometry: a ball 0.45 m out and 0.25 m below a level camera
    is 29 degrees down, and tan(29)/tan(31.1) puts it 0.96 of the way to the
    bottom edge.

SPEAKING QUACKD. rokbenko/quackd drives a Microduck with composed verbs
(walk_to, follow, fetch) that ask a detector for boxes. Its
`quackd/perception/base.py` declares the protocol — an object with a `name`
and a `.detect(image) -> list[Detection]` — and a `Detection` carrying label,
cx, cy, area, confidence, bearing_deg, est_distance_m. `HailoDetector` below
satisfies that STRUCTURALLY: no import, no dependency, no shared base class,
because this file has to keep running on the Pi with numpy, PIL, mujoco and
hailo_platform and nothing else. Where their contract differs from ours it is
THEIRS that wins inside `detect`, and each difference is called out there:
their bearing is in DEGREES and POSITIVE TO THE LEFT, their cx/cy are
normalised against the image the caller handed in (not the letterboxed
square), and their focal length spans the image's WIDTH.

WHAT THIS DOES NOT DO. It does not step physics. Rendering runs on MuJoCo
3.12 (pip) while the canon recorders run MuJoCo 3.5.1 (WASM) against
scene.mjb, and a clip recorded on a different engine would not be canon.
Drawing is not physics: this loads the same plant purely to place and
photograph it. A closed-loop intent therefore drives physics from the Node
harness and asks this for perception.
"""
from __future__ import annotations
import json, os, re, sys
from dataclasses import dataclass
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
# The render scene's ball, so a monocular range estimate has a real size to
# divide by. The geom below is a sphere of BALL_RADIUS_M, and a sphere's
# MJCF size IS its radius, so the thing in front of the lens is 72 mm across.
# This is the SIM ball's diameter, measured off the scene this file builds —
# it is not a claim about whatever ball Pollen ship with the hardware.
# Pollen's own ball, as the bench's world declares it: BALL_RADIUS 0.05,
# mass 0.03, condim 6 (scene_physics.xml). The render scene matches it so a
# distance estimated from apparent size means the same thing in both.
BALL_RADIUS_M = 0.05
BALL_DIAMETER_M = 2 * BALL_RADIUS_M

# The 80 COCO classes in the order the stock hailo-models yolov8s was trained
# in. Confirmed on this machine, not copied on faith: our orange sphere comes
# back as class 49, and 49 in this list is "orange".
COCO_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush")

# quackd's whole vocabulary is "ball" | "person" | "pet": those three strings
# are what its composite verbs match on, and this is the COCO map it uses.
COCO_TO_QUACKD = {"sports ball": "ball", "person": "person",
                  "cat": "pet", "dog": "pet"}


@dataclass
class Detection:
    """quackd's `Detection`, field for field (quackd/perception/base.py).

    A PLAIN DATACLASS, DELIBERATELY. Importing quackd's own model would drag
    pydantic onto the Pi to gain nothing: the verbs read attributes, so
    matching the names IS the contract. Same reason there is no base class.
    """

    label: str            # "ball" | "person" | "pet", or a COCO name (see detect)
    cx: float             # normalised [0,1] across the image handed to detect()
    cy: float             # normalised [0,1] down it, origin TOP-LEFT
    area: float           # fraction of that image's pixels, [0,1]
    confidence: float     # [0,1]
    bearing_deg: float | None    # degrees, POSITIVE = LEFT of the duck's heading
    est_distance_m: float | None  # metres, or None when the object's size is unknown


def width_fov_degrees(width_px: int, height_px: int,
                      fovy_deg: float = SENSOR_FOVY_DEGREES) -> float:
    """How wide the lens sees ACROSS an image of this shape.

    THE SENSOR'S 62.2 DEGREES IS THE PORTRAIT FRAME'S TALL AXIS, NOT ITS WIDE
    ONE. Every angle downstream is computed from a width, so the number that
    goes into them has to be the width's.
    """
    return float(2 * np.degrees(np.arctan(
        np.tan(np.radians(fovy_deg / 2)) * width_px / height_px)))


def focal_px(width_px: int, hfov_deg: float) -> float:
    """quackd's own focal length: f = (w/2) / tan(fov/2), fov spanning the WIDTH."""
    return (width_px / 2.0) / float(np.tan(np.radians(hfov_deg) / 2.0))


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
    <body name="seek_ball" pos="0.45 0 {BALL_RADIUS_M}">
      <freejoint/>
      <geom name="seek_ball_geom" type="sphere" size="{BALL_RADIUS_M}" rgba="{ball_rgba}"/>
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
        self.d.qpos[self.ball:self.ball + 7] = [ball[0], ball[1], BALL_RADIUS_M, 1, 0, 0, 0]
        self.mujoco.mj_forward(self.m, self.d)

    def gaze(self):
        """Where the camera looks, world frame, and how far below the horizon."""
        v = -self.d.cam_xmat[self.cam].reshape(3, 3)[:, 2]
        return v, float(np.degrees(np.arcsin(-v[2])))

    def frame(self) -> np.ndarray:
        self.renderer.update_scene(self.d, camera="head_camera")
        return self.renderer.render()

    def truth(self):
        """Where the ball really is from the lens: (bearing degrees, metres).

        The ground truth a bearing has to be checked against, taken from the
        plant rather than from the picture. `place` puts the duck at the origin
        with quat (1,0,0,0) and head_yaw at 0, so the camera's azimuth IS the
        duck's heading: +x is straight ahead and +y is the duck's LEFT, which
        makes this positive-left exactly as quackd wants it.
        """
        cam = np.array(self.d.cam_xpos[self.cam], dtype=float)
        ball = np.array(self.d.qpos[self.ball:self.ball + 3], dtype=float)
        v = ball - cam
        return float(np.degrees(np.arctan2(v[1], v[0]))), float(np.linalg.norm(v))


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


class HailoDetector:
    """The Hailo-8, as a context manager. 12.3 ms a frame, measured.

    NAMED `HailoDetector`, NOT `Detector`. quackd's protocol is itself called
    `Detector` (quackd/perception/base.py, alongside their ColorBlobDetector
    and YoloDetector, which both carry a `name`), so a class of ours by that
    name would shadow theirs the moment the two files met in one process.

    `known_width_m` maps an EMITTED label to how wide that object really is,
    in metres, and is the only reason `est_distance_m` can be anything but
    None: a monocular camera measures angles, and turning an angle into a
    range needs a size from outside the picture. Nothing is assumed — the
    caller who built the scene, or measured the ball on the bench, is the one
    who knows.
    """

    name = "hailo-yolov8s"

    def __init__(self, hef_path: str = HEF_PATH,
                 known_width_m: dict[str, float] | None = None):
        self.hef_path = hef_path
        self.known_width_m = dict(known_width_m or {})

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

    hfov_degrees: float = width_fov_degrees(SENSOR_W, SENSOR_H)

    def _boxes(self, square: np.ndarray):
        """One inference, unpacked: (class id, confidence, box) in SQUARE units.

        The HEF's NMS output is a list indexed by class, each entry a row of
        [ymin, xmin, ymax, xmax, score] normalised to the 640 square. Both
        readers below start here so there is one place that knows that shape.
        """
        out = self.pipe.infer({self.iname: square[None]})
        dets = list(out.values())[0]
        if isinstance(dets, list):
            dets = dets[0]
        rows = []
        for cls in range(len(dets)):
            for b in dets[cls]:
                ymin, xmin, ymax, xmax = (float(v) for v in b[:4])
                rows.append((cls, float(b[4]), xmin, ymin, xmax, ymax))
        return rows

    def detect(self, image: np.ndarray, floor: float = 0.15) -> list[Detection]:
        """quackd's `detect`: every box in the frame, in quackd's own units.

        `image` IS THE FRAME AS THE SENSOR GIVES IT — 720x1280 portrait here —
        not a letterboxed square. That is not a convenience: quackd normalises
        cx/cy against "the image handed to detect()", so the letterbox has to
        happen inside this call for those numbers to mean anything to them.

        Strongest first. Callers who want our own class-agnostic reading of
        the same frame want `best`, which speaks duck-detect's convention and
        is deliberately NOT this one.
        """
        h, w, _ = image.shape
        square, meta = letterbox(image)
        # DEFECT 4, FIXED: THE FOV AXIS. quackd's f = (w/2)/tan(fov/2) — the
        # fov it wants SPANS THE WIDTH. Our frame is portrait, so the IMX219's
        # 62.2 degrees is its TALL axis and only 37.49 degrees run across it.
        # Hand a width formula 62.2 and f drops from 1060.9 px to 596.8, which
        # on the measured 0.8 m ball turns +10.9 deg and 0.71 m into +18.9 deg
        # and 0.40 m: wrong by enough to overshoot every turn and stop short of
        # every ball, and plausible enough that neither would look like a bug.
        hfov = width_fov_degrees(w, h)
        f = focal_px(w, hfov)
        found = []
        for cls, conf, xmin, ymin, xmax, ymax in self._boxes(square):
            if conf < floor:
                continue
            # DEFECT 3, FIXED: cy IS DE-LETTERBOXED TOO. The old code mapped cx
            # back through the letterbox and left cy in square coordinates. On
            # THIS frame that happens to be harmless — 720x1280 portrait pads
            # 140 columns each side and pad_y is 0, measured — so the bug is
            # latent, not absent: the padding moves to the top and bottom the
            # moment the frame is wider than it is tall, and the mount's
            # rotation is UNCONFIRMED (see the note by SENSOR_FOVY_DEGREES). On
            # a 1280x720 frame a box at cy 0.75 in the square is really at
            # 0.944, which is the difference between the duck's feet and the
            # horizon. Both axes are mapped back, or neither should be.
            cx = ((xmin + xmax) / 2 * meta["size"] - meta["pad_x"]) / meta["fitted_w"]
            cy = ((ymin + ymax) / 2 * meta["size"] - meta["pad_y"]) / meta["fitted_h"]
            bw = (xmax - xmin) * meta["size"] / meta["fitted_w"]
            bh = (ymax - ymin) * meta["size"] / meta["fitted_h"]
            cx, cy = float(np.clip(cx, 0.0, 1.0)), float(np.clip(cy, 0.0, 1.0))
            bw, bh = float(np.clip(bw, 0.0, 1.0)), float(np.clip(bh, 0.0, 1.0))
            # DEFECTS 1 AND 2, FIXED: SIGN AND UNITS. Ours used to be positive
            # when the box sat RIGHT of centre, in radians. quackd's is
            # positive to the LEFT, in degrees — the robotics +yaw — and their
            # walk_to steers with wz = clamp(bearing * 0.05), a POSITIVE gain.
            # Feed that a right-positive bearing and the loop is positive
            # feedback: the duck turns away from the ball and keeps turning
            # until the ball leaves a 37-degree frame. Radians would have been
            # the milder bug of the two, an unexplained 57x-slow turn.
            bearing = float(np.degrees(np.arctan((0.5 - cx) * w / f)))
            name = COCO_NAMES[cls] if cls < len(COCO_NAMES) else f"class{cls}"
            # PASS ANYTHING ELSE THROUGH UNDER ITS COCO NAME. Their four-entry
            # map covers what their verbs act on; our stand-in ball is a smooth
            # sphere the stock net calls "orange", and quietly relabelling
            # oranges as balls would teach a real duck to chase fruit across a
            # kitchen. The sim's ball being the wrong class is the SIM's
            # problem — fix it with a seamed ball (class 32) or with duck-detect's
            # single-class model, not with a lie in the vocabulary. A caller who
            # knows better widens COCO_TO_QUACKD; nothing here decides for them.
            label = COCO_TO_QUACKD.get(name, name)
            size_m = self.known_width_m.get(label)
            # Pinhole range: an object of known width w_m spanning bw*w pixels
            # sits w_m * f / pixels away. None when we do not know the width —
            # quackd's field is Optional for exactly this case, and a guessed
            # metre is worse than no metre.
            distance = None
            if size_m and bw * w > 0:
                distance = round(size_m * f / (bw * w), 3)
            found.append(Detection(
                label=label, cx=cx, cy=cy, area=float(np.clip(bw * bh, 0.0, 1.0)),
                confidence=conf, bearing_deg=round(bearing, 1),
                est_distance_m=distance))
        found.sort(key=lambda d: d.confidence, reverse=True)
        return found

    def best(self, square: np.ndarray, meta: dict | None = None, floor: float = 0.15):
        """The strongest box in the frame, class-agnostic.

        CLASS-AGNOSTIC ON PURPOSE. The stock COCO model calls our orange
        sphere class 49, "orange"; a real ball with panel seams would be 32,
        "sports ball"; Pollen's own duck-detect has ONE class. An intent that
        keyed on a class id would break on all three.

        THIS IS NOT quackd's BEARING and is not meant to be: it is radians,
        positive to the RIGHT, off the SQUARE the caller already letterboxed.
        Our own callers were written against it and it is left alone. Anything
        speaking to quackd wants `detect`.
        """
        best = None
        for cls, conf, xmin, ymin, xmax, ymax in self._boxes(square):
            if best is None or conf > best["conf"]:
                best = {"conf": conf, "cls": cls,
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
    print(f"frame {SENSOR_W}x{SENSOR_H} portrait, {SENSOR_FOVY_DEGREES} deg tall -> "
          f"{width_fov_degrees(SENSOR_W, SENSOR_H):.2f} deg across, "
          f"f = {focal_px(SENSOR_W, width_fov_degrees(SENSOR_W, SENSOR_H)):.1f} px")
    left_case = None
    # The ball is the only object in this scene whose size we know, so it is
    # the only one that can carry a range — and it is keyed by "orange",
    # because that is the label the stock net gives it. See `detect`.
    with HailoDetector(known_width_m={"orange": BALL_DIAMETER_M}) as det:
        print(f"detector: {det.name}")
        for label, ball, head in [("0.45 m ahead", (0.45, 0.0), 1.00),
                                  ("0.8 m ahead", (0.80, 0.0), 0.75),
                                  ("0.8 m, 10 deg left", (0.79, 0.14), 0.75),
                                  ("0.8 m, 20 deg left", (0.78, 0.28), 0.75),
                                  ("head level, 0.45 m", (0.45, 0.0), 0.349)]:
            joints = list(eye.C["homePose"])
            joints[eye.C["jointNames"].index("head_pitch")] = head
            eye.place(joints=joints, ball=ball)
            _, below = eye.gaze()
            bearing_true, range_true = eye.truth()
            hits = det.detect(eye.frame())
            print(f"  {label:22s} head {below:4.1f} deg down | "
                  f"truth {bearing_true:+5.1f} deg, {range_true:.3f} m")
            if not hits:
                print("      nothing")
            for hit in hits:
                print(f"      {hit.label:12s} conf {hit.confidence:.2f} "
                      f"cx {hit.cx:.3f} cy {hit.cy:.3f} area {hit.area:.4f} "
                      f"bearing {hit.bearing_deg:+5.1f} deg "
                      f"dist {hit.est_distance_m if hit.est_distance_m is not None else '—'}")
            if label.endswith("10 deg left"):
                left_case = (bearing_true, hits[0] if hits else None)
    # THE CHECK THIS FILE EXISTS TO PASS: a ball 10 degrees to the duck's LEFT
    # reports a POSITIVE bearing. Not a style point — quackd's walk_to steers
    # on this sign, so getting it wrong makes the duck run away from the ball.
    bearing_true, hit = left_case if left_case else (0.0, None)
    if hit is None or hit.bearing_deg is None or hit.bearing_deg <= 0:
        print(f"FAIL: a ball {bearing_true:+.1f} deg to the duck's left reported "
              f"{hit.bearing_deg if hit else 'nothing'}")
        return 1
    print(f"PASS: a ball {bearing_true:+.1f} deg to the duck's left reports "
          f"{hit.bearing_deg:+.1f} deg, positive, {abs(hit.bearing_deg - bearing_true):.1f} deg off truth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
