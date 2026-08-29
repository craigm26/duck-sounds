#!/usr/bin/env python3
"""Drive the flamingo-cycle policy on a physical Microduck through robotd's IPC socket.

One file, standard library only. A TEST AID, not a product: no gamepad, no supervision beyond
what robotd reports. NOT yet run on hardware — keep a hand near the robot for the first lift.

Robot config (deploy/robotd.toml, then `sudo systemctl restart robotd`). robotd writes
`robot.move {vx, vy, vyaw}` verbatim into the network's twist slots (EMA + 500 ms deadman only),
so the policy runs under the `walk` role with the standing switch disabled:

    [control]
    cmd_alpha = 1.0                # pass the flag through unsmoothed (default 0.2 = a 0.4 s ramp)

    [policy]
    walk = "/home/radxa/policies/flamingo/policy.onnx"
    stand = "none"                 # no "twist <= 0.05 -> standing network": this policy stands by itself at [0,0,0]
    sitstand = "none"              # no skill may take the command block over during the test
    ground_pick = "none"
    kick_left = "none"
    kick_right = "none"
    roulade = "none"

    [safety]
    limp_fall_tilt_z = -0.80       # fall predictor's "already tilted" gate is -0.90 (~26 deg); the hold leans ~24 deg

Falls stay the daemon's: the fall predictor makes the robot go limp, waits for the gyro to settle,
ramps to the default pose and hands back with the twist at zero, which for this policy is "stand".
For a permanent integration the daemon wants a `flamingo` role (a Net variant writing
twist = [flag, side, 0], like sit writes [1, 0, 0]) so the walk/stand policies stay loaded.

The policy reads the twist slots as [flag, side, 0]:
    flag  0 = stand on two feet, 1 = stand on one foot
    side +1 = right foot down / left leg lifted, -1 = left foot down

    # on the laptop, after:  ssh -L /tmp/robotd.sock:/run/robotd.sock radxa@<robot>
    python3 control.py --socket /tmp/robotd.sock                       # interactive: f = lift/lower, c = side, q = quit
    python3 control.py --socket /tmp/robotd.sock --timeline "0:0,1;3.6:1,1;11.6:0,1;18.3:1,-1;26:0,-1"   # scripted t:flag,side
    python3 control.py --socket /tmp/robotd.sock --no-enable           # don't send robot.enable (already driving)
    python3 control.py --socket /tmp/robotd.sock --init                # power the joints first (robot.init) if the robot is relaxed

If it exits immediately it tells you why (no socket / forward, permission, daemon refusal). Please
report the exact message when asking for help.

What it does: opens the socket, `robot.enable {on: true}` (a request, answered), then sends
`robot.move {vx: flag, vy: side, vyaw: 0}` notifications at 10 Hz (robotd zeroes a twist older
than its 500 ms deadman) while printing robotd's state (policy label, fallen/limp, gravity,
applied twist) at 2 Hz. On exit it sends flag 0 for 3 s so the robot is back on two feet before the
deadman takes over, then `robot.stop`. It never sends `robot.relax` (that cuts motor power).
"""

import argparse
import json
import select
import socket
import sys
import termios
import time
import tty

RESEND_HZ = 10.0
LOWER_S = 3.0          # time given to the policy to come back down before exiting


class Robotd:
    def __init__(self, path):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.sock.connect(path)
        except FileNotFoundError:
            sys.exit(f"no socket at {path}.\n  on the robot: is robotd running?  (systemctl status robotd)\n"
                     f"  from a laptop: forward it first — ssh -L {path}:/run/robotd.sock radxa@<robot>\n"
                     f"  (ssh refuses to bind if {path} already exists: rm it, then reconnect)")
        except PermissionError:
            sys.exit(f"no permission on {path} — on the robot run as the user that owns it (or with sudo)")
        except ConnectionRefusedError:
            sys.exit(f"{path} exists but nothing answers — a stale socket file or a dead ssh forward; remove it and reconnect")
        self.sock.setblocking(False)
        self.buf = b""
        self.next_id = 1
        self.state = None

    def notify(self, method, params=None):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self.sock.sendall((json.dumps(msg) + "\n").encode())

    def request(self, method, params=None, timeout=2.0):
        rid = self.next_id
        self.next_id += 1
        msg = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            msg["params"] = params
        self.sock.sendall((json.dumps(msg) + "\n").encode())
        t0 = time.time()
        while time.time() - t0 < timeout:
            for m in self.poll():
                if m.get("id") == rid:
                    if "error" in m:
                        raise RuntimeError(f"{method}: {m['error']}")
                    return m.get("result")
            time.sleep(0.01)
        raise TimeoutError(f"{method}: no answer in {timeout}s")

    def poll(self):
        """Read whatever arrived; keep the latest robot.state; return all messages."""
        out = []
        try:
            while True:
                chunk = self.sock.recv(65536)
                if not chunk:
                    raise ConnectionError("robotd closed the socket")
                self.buf += chunk
        except BlockingIOError:
            pass
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            if not line.strip():
                continue
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            if m.get("method") == "robot.state":
                self.state = m.get("params", {})
            out.append(m)
        return out

    def move(self, flag, side):
        self.notify("robot.move", {"vx": float(flag), "vy": float(side), "vyaw": 0.0})

    def status(self):
        s = self.state
        if not s:
            return "no state yet"
        saf = s.get("safety", {})
        mv = s.get("movement", {})
        g = saf.get("gravity", [0, 0, 0])
        return (f"policy={s.get('policy')} fallen={saf.get('fallen')} limp={saf.get('limp')} "
                f"gz={g[2]:+.2f} applied_twist={[round(x, 2) for x in mv.get('applied', [])]} "
                f"limited_by={mv.get('limited_by')}")


def parse_timeline(spec):
    steps = []
    for part in spec.split(";"):
        t, v = part.split(":")
        flag, side = (float(x) for x in v.split(","))
        steps.append((float(t), flag, side))
    return sorted(steps)


class Keys:
    """Raw single-key reads from the terminal (interactive mode)."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def get(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1).lower()
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--socket", default="/run/robotd.sock", help="robotd IPC socket (forwarded: /tmp/robotd.sock)")
    ap.add_argument("--timeline", default=None, help="scripted 't:flag,side;...' instead of the keyboard")
    ap.add_argument("--no-enable", action="store_true", help="skip robot.enable (the robot is already driving)")
    ap.add_argument("--state-hz", type=float, default=2.0, help="robot.state subscription rate for the status line")
    ap.add_argument("--init", action="store_true", help="send robot.init first (powers the joints and ramps to the home pose)")
    a = ap.parse_args()
    if a.timeline is None and not sys.stdin.isatty():
        sys.exit("interactive mode needs a terminal (keys are read from stdin); use --timeline \"t:flag,side;...\" instead")

    rd = Robotd(a.socket)
    try:
        print("robot.subscribe ->", rd.request("robot.subscribe", {"hz": int(a.state_hz)}))
    except Exception as e:  # the status line is a convenience, not a requirement
        print("subscribe failed (continuing without the status line):", e)
    if a.init:
        try:
            print("robot.init ->", rd.request("robot.init", timeout=10.0))
        except Exception as e:
            sys.exit(f"robot.init refused: {e}")
    if not a.no_enable:
        try:
            print("robot.enable ->", rd.request("robot.enable", {"on": True}))
        except Exception as e:
            sys.exit(f"robot.enable refused: {e}\n  (is the policy loaded? robotd reports 'policy unavailable: ...' in robot.health; "
                     f"is the robot initialised? try --init)")

    flag, side = 0.0, 1.0
    period = 1.0 / RESEND_HZ
    t0 = time.time()
    last_print = 0.0
    timeline = parse_timeline(a.timeline) if a.timeline else None
    keys = Keys() if timeline is None else None
    if keys:
        keys.__enter__()
        print("f = lift / lower the foot   c = choose the side (while standing)   q = quit (lowers first)")
    try:
        while True:
            now = time.time() - t0
            if timeline is not None:
                for t, f, s_ in timeline:
                    if now >= t:
                        flag, side = f, s_
                if now > timeline[-1][0] + LOWER_S and flag == 0.0:
                    break
            else:
                k = keys.get()
                if k == "q":
                    break
                if k == "f":
                    flag = 0.0 if flag else 1.0
                    print(f"\nflag -> {flag:.0f} ({'lifting the ' + ('LEFT' if side > 0 else 'RIGHT') + ' foot' if flag else 'coming down'})")
                if k == "c":
                    if flag:
                        print("\ncome down first (f) before changing the side")
                    else:
                        side = -side
                        print(f"\nside -> {side:+.0f} ({'right foot down' if side > 0 else 'left foot down'})")
            rd.move(flag, side)
            rd.poll()
            if now - last_print >= 1.0 / a.state_hz:
                last_print = now
                print(f"\r{now:6.1f}s  cmd=[{flag:.0f},{side:+.0f},0]  {rd.status()}", end="", flush=True)
            time.sleep(period)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if keys:
            keys.__exit__(None, None, None)
        print("\nlowering (flag 0) for %.0f s ..." % LOWER_S)
        t1 = time.time()
        while time.time() - t1 < LOWER_S:
            rd.move(0.0, side)
            rd.poll()
            time.sleep(period)
        try:
            rd.request("robot.stop")
        except Exception as e:
            print("robot.stop:", e)
        print("done:", rd.status())


if __name__ == "__main__":
    main()
