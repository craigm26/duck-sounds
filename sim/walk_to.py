"""walk_to: the first duck in this project that steers on what it can see.

Every intent in the corpus is blind — a policy plus a command schedule, decided
before the run and never revised. This one closes the loop: render the duck's
own camera, put the frame through the Hailo, take the bearing off the detection,
and steer. Nothing tells it where the ball is.

THE THREE LOOPS, RUNNING AT ONCE (duckkit/docs/architecture.md, after quackd's
ADR-0003). Reflexes are the walking policy at 50 Hz inside the bench; steering
is this file at 10 Hz; deliberation is absent on purpose — a composite verb
never calls a model, it just runs.

WHY IT IS SPLIT ACROSS TWO PROCESSES. The bench owns physics (MuJoCo 3.5.1 in
WASM, the plant every canon clip was recorded on) and this owns rendering
(MuJoCo 3.12 through Python, the only one here that can draw). Drawing is not
physics, so the split costs nothing real: the bench is asked for the duck's
pose, the pose is drawn, the picture is read, and a command goes back. The
bench owns the clock too, which is why a trial is reproducible.

SCORING IS GROUND TRUTH; STEERING IS NOT. `/state` reports where the ball
actually is so a trial can be marked, exactly as measure_success.mjs does. The
controller never reads it — it only ever sees `bearing_deg` from the detector.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time, urllib.request

SIM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM)
import duckvision as dv                                    # noqa: E402

BENCH = os.environ.get("DUCKBENCH", "http://127.0.0.1:8770")
_C = json.load(open(f"{SIM}/duckkit-constants.json"))
HOME, MOUTH_INDEX = _C["homePose"], _C["mouthIndex"]
# head_yaw sits at 7 in both the 15-wide wire order and the 14-wide policy
# list, because the mouth it drops comes after it.
HEAD_YAW = _C["jointNames"].index("head_yaw")
STEER_HZ = 10.0            # quackd's composite rate; the bench still ticks at 50
ARRIVE_M = 0.25            # "stop about 0.25 m away", their walk_to's own figure


def wire_pose(policy_joints: list[float]) -> list[float]:
    """The bench reports the 14 joints a policy commands; a pose is 15 wide.

    The mouth is the one the policies skip, so it is put back at its home
    value rather than left out — a 14-wide array handed to a 15-wide door is
    the silent-narrowing bug ADR-0002 warns about.
    """
    pose = list(policy_joints)
    pose.insert(MOUTH_INDEX, HOME[MOUTH_INDEX])
    return pose


def call(path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f"{BENCH}{path}", data=data,
        headers={"content-type": "application/json"} if data else {})
    with urllib.request.urlopen(request, timeout=60) as answer:
        return json.load(answer)


def command_for(bearing_deg: float | None, distance_m: float | None,
                search_sign: float = 1.0) -> tuple[float, float]:
    """quackd's steering law, and the one place our envelope replaces theirs.

    `wz = bearing * 0.05` with a POSITIVE gain is theirs verbatim — which is
    exactly why the detector's sign had to be fixed before this could work at
    all: with the sign we shipped yesterday this is positive feedback and the
    duck turns away from the ball.

    The forward speeds are OURS, because theirs are a generic robot's. This
    duck walks 0.106 m/s at a 0.25 command and has a dead band below about
    0.25 (duck-trajectories.json, measured), so 0.05 would be a command to
    stand still while believing it is approaching.
    """
    if bearing_deg is None:
        # A DUCK CANNOT TURN ON THE SPOT, and this is the measurement that says
        # so. Commanded vyaw -1.0 with vx 0, alpha_walking yaws 14.4 degrees in
        # the first two seconds and then STOPS: at 4 s it is at -14.3, at 20 s
        # still -14.4. It is not a slow turn rate, it is a saturation — which is
        # why a stationary sweep never found a ball 40 degrees off the nose in
        # 200 steps, and why reading the first two seconds as "0.128 rad/s" was
        # wrong. Searching therefore ARCS: enough forward speed to clear the
        # gait's dead band (below ~0.25 alpha_walking marches in place) with
        # full yaw, which sweeps the heading continuously and covers ground.
        return 0.25, search_sign
    wz = max(-1.0, min(1.0, bearing_deg * 0.05))
    if abs(bearing_deg) > 25:
        return 0.0, wz                       # square up first
    near = distance_m is not None and distance_m < 0.45
    return (0.25 if near else 0.35), wz


def trial(eye: dv.DuckEye, detector, bearing_deg: float, range_m: float,
          seconds: float = 20.0, verbose: bool = False) -> dict:
    # A KNOWN START, EVERY TIME. Without this the second trial of a batch
    # inherits the first one's heading and half-finished stride, and opens
    # already spinning — which reads exactly like a controller that cannot see.
    call("/reset")
    call("/policy", {"policy": "alpha_walking.onnx"})
    state = call("/ball", {"bearing": bearing_deg, "range": range_m})
    start = math.dist(state["position"][:2], state["ball"][:2])

    deadline = state["t"] + seconds
    search_sign = 1.0 if bearing_deg >= 0 else -1.0     # start by looking where it was put
    seen = 0
    steps = 0
    while state["t"] < deadline:
        # DRAW WHAT THE DUCK CAN SEE, from the pose the bench reports.
        eye.place(joints=wire_pose(state["joints"]),
                  root=state["position"], quat=state["quaternion"],
                  ball=(state["ball"][0], state["ball"][1]))
        # detect() takes the RAW sensor frame and letterboxes internally, which
        # is the only way cx/cy can mean "of the image handed to detect()".
        # best() is our older reader and returns radians positive-RIGHT; using
        # it here would invert the steering.
        found = detector.detect(eye.frame())
        hit = max(found, key=lambda d: d.confidence) if found else None
        # THE CAMERA RIDES A HEAD THE GAIT SWINGS. Measured on the first
        # working run: a ball dead ahead read anywhere from -8 to +11 degrees
        # within a single stride, because `head_yaw` is a joint the walking
        # policy drives and the lens is bolted to it. Steering on the raw
        # camera bearing therefore fights the gait — the duck still arrives,
        # but the turn command swings +/-0.5 rad/s the whole way. The bearing
        # the BODY needs is the camera's plus wherever the head is pointing.
        head_yaw_deg = math.degrees(state["joints"][HEAD_YAW])
        bearing = (hit.bearing_deg + head_yaw_deg) if hit else None
        distance = hit.est_distance_m if hit else None
        if hit:
            seen += 1
        if bearing is not None:
            # Remember which way it went, so a ball that leaves the frame is
            # looked for on the side it left by.
            search_sign = 1.0 if bearing >= 0 else -1.0
        vx, wz = command_for(bearing, distance, search_sign)
        state = call("/intent", {"vx": vx, "vyaw": wz, "hold": 1.0 / STEER_HZ})
        steps += 1
        gap = math.dist(state["position"][:2], state["ball"][:2])
        if verbose:
            saw = (f"{bearing:+5.1f} deg (cam {hit.bearing_deg:+.1f}, head {head_yaw_deg:+.1f})"
                   if hit and distance
                   else f"{bearing:+5.1f} deg" if hit else "nothing")
            print(f"    t {state['t']:5.2f}  saw {saw:>18}  vx {vx:.2f} wz {wz:+.2f}"
                  f"  gap {gap:.3f}")
        if gap <= ARRIVE_M:
            return {"arrived": True, "gap": round(gap, 3), "start": round(start, 3),
                    "t": state["t"], "steps": steps, "seen": seen,
                    "upright": state["upright"]}
        if not state["upright"]:
            return {"arrived": False, "why": "fell", "gap": round(gap, 3),
                    "start": round(start, 3), "t": state["t"], "steps": steps,
                    "seen": seen, "upright": False}
    gap = math.dist(state["position"][:2], state["ball"][:2])
    return {"arrived": False, "why": "ran out of time", "gap": round(gap, 3),
            "start": round(start, 3), "t": state["t"], "steps": steps,
            "seen": seen, "upright": state["upright"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    health = call("/health")
    print(f"bench {health['bench']} — steering at {STEER_HZ:.0f} Hz, "
          f"arriving at {ARRIVE_M} m")
    eye = dv.DuckEye(dv.build_render_scene("/tmp/walk_to_scene.xml"))
    # A spread that includes bearings outside the lens: the duck has to turn to
    # find those, which is the behaviour worth having.
    placements = [(0, 0.8), (12, 0.8), (-12, 0.8), (25, 0.9),
                  (-25, 0.9), (40, 1.0), (-40, 1.0), (0, 1.2)]
    with dv.HailoDetector() as detector:
        results = []
        for i in range(args.trials):
            bearing, distance = placements[i % len(placements)]
            print(f"  trial {i + 1}: ball {bearing:+d} deg, {distance} m")
            outcome = trial(eye, detector, bearing, distance,
                            seconds=args.seconds, verbose=args.verbose)
            results.append(outcome)
            print(f"    -> {'ARRIVED' if outcome['arrived'] else 'no: ' + outcome['why']}"
                  f"  gap {outcome['gap']} m (from {outcome['start']})"
                  f"  {outcome['steps']} steps, saw the ball on {outcome['seen']}")
    arrived = sum(1 for r in results if r["arrived"])
    print(f"\narrived {arrived}/{len(results)}; "
          f"median final gap {sorted(r['gap'] for r in results)[len(results) // 2]} m")
    return 0 if arrived else 1


if __name__ == "__main__":
    sys.exit(main())
