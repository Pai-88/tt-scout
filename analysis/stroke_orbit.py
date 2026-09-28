"""A player's typical stroke in 3D, turning a full circle while it replays in slow motion, with the pros' typical stroke of the
same side as a grey ghost standing where the player stood: a video drawn like the report's 3D plates (render3d), from the same
data as the report's 3D viewer (the a3d-data block of report.html) and with the viewer's timing, ball path and ghost placement.

    python analysis/stroke_orbit.py out_own_product/<match>/report.html --player Alex --seconds 20 --out orbit.mp4

The camera circles the player at constant speed, starting and ending where the phone stood (side on), so the video loops cleanly.
While the table is between the camera and the player it rises a little and the table fades to half see-through, both smoothly, so
the table top never hides the body.
"""
import argparse, json, math, pathlib, subprocess, sys, tempfile
from multiprocessing import Pool
import numpy as np, cv2
ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from tt_scout.render3d import Scene, mix
from tt_scout import body3d
from tt_scout.body3d import I

FLOOR = -0.76
SLOW = 10.0            # replayed at a tenth of real speed
BACK = 0.7             # seconds to ease back from the follow-through to the start of the backswing between replays
W_, H_ = 1920, 1080


def _clean(st):
    """The report's strokes as they are: their frames already have Vision's flipped frames left out and the feet planted
    (analysis3d._stroke_json, body3d.strokes_from), so nothing is re-applied here (re-planting planted frames and re-checking flips
    on smoothed ones moved joints a second time until 2026-09-28)."""
    return None if st is None else dict(st, frames=[np.asarray(F, float) for F in st["frames"]])


def load(report, player):
    D = json.loads(pathlib.Path(report).read_text().split('id="a3d-data">')[1].split("</script>")[0].replace("<\\/", "</"))
    p = next(q for q in D["players"] if q["name"].lower() == player.lower())
    s = _clean(p["strokes"][p["typ"]])
    g = _clean(next((q for q in D["pro"]["strokes"] if q["side"] == s["side"]), None))
    return p, s, g


def body_at(s, t):
    """The viewer's bodyAt: Catmull-Rom through the pose samples at their times (s['dts'], seconds from the contact)."""
    d, F = s["dts"], np.asarray(s["frames"], float); n = len(d)
    if t <= d[0]:
        return F[0]
    if t >= d[-1]:
        return F[-1]
    i = 0
    while i < n - 2 and d[i + 1] < t:
        i += 1
    a = (t - d[i]) / ((d[i + 1] - d[i]) or 1.0)
    A, B, C, E = F[max(0, i - 1)], F[i], F[i + 1], F[min(n - 1, i + 2)]
    return 0.5 * (2 * B + (-A + C) * a + (2 * A - 5 * B + 4 * C - E) * a * a + (-A + 3 * B - 3 * C + E) * a ** 3)


def along(P, u):
    P = np.asarray(P, float); u = min(max(u, 0.0), 1.0) * (len(P) - 1); i = min(len(P) - 2, int(u))
    return P[i] + (P[i + 1] - P[i]) * (u - i)


def ball_at(s, t):
    """The viewer's ballAt: on the rising arc before the contact, on the fitted flight after it."""
    if t < 0:
        return None if (not s.get("arc") or not s.get("dti") or t < -s["dti"]) else along(s["arc"], 1 + t / s["dti"])
    if not s.get("path") or not s.get("ft"):
        return np.asarray(s["ball"], float) if t < 0.03 else None
    return None if t > s["ft"] else along(s["path"], t / s["ft"])


def stroke_time(tau, s):
    """Video seconds -> (stroke seconds, w): the replay at 1/SLOW, then an eased return to the start (w from 0 to 1 = how far from
    the follow-through back to the backswing's first pose; None while replaying), so the loop never jumps."""
    t0, t1 = s["dts"][0], s["dts"][-1]
    run = (t1 - t0) * SLOW
    u = tau % (run + BACK)
    if u < run:
        return t0 + u / SLOW, None
    x = (u - run) / BACK
    return t1, x * x * (3 - 2 * x)


def render(job):
    k, n, report, player, out_dir, ss, clean = job
    p, s, g = load(report, player)
    t, w = stroke_time(k / 59.94, s)
    ci = np.asarray(s["frames"][s["ci"]], float); root = ci[I["root"]]
    target = np.array([root[0] + 0.2, root[1], root[2] + 0.05])
    yaw = -math.pi / 2 + 2 * math.pi * k / n                           # from the phone's side of the table, one full turn at constant speed
    pitch = 0.17 + 0.22 * ((1 + math.cos(yaw)) / 2) ** 2               # rises over the table (the table is towards +x)
    c = math.cos(yaw); u = min(max((c - 0.1) / 0.55, 0.0), 1.0)
    table_alpha = 1.0 - 0.5 * u * u * (3 - 2 * u)                      # smoothstep: opaque at the sides, half see-through behind the table
    dist = 4.9
    eye = target + dist * np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
    sc = Scene(eye=eye, target=target, fov=42, size=(W_, H_), theme="dark", ss=ss)
    T = sc.T; col = T[p["key"]]; grey = mix(T["paper"], T["ink"], 0.8)
    pose = lambda st: body_at(st, t) if w is None else (1 - w) * body_at(st, t) + w * body_at(st, st["dts"][0] if st is s else s["dts"][0])
    J = pose(s); ball = ball_at(s, t) if w is None else None
    G = None
    if g is not None:                                                   # the pros' stroke, standing where he stood, same contact time
        G = pose(g); gc = np.asarray(g["frames"][g["ci"]], float)
        G = G + np.array([ci[I["root"]][0] - gc[I["root"]][0], ci[I["root"]][1] - gc[I["root"]][1], 0.0])

    def set_():
        with sc.group(table_alpha):
            sc.table(); sc.net()

    def people():
        with sc.shadows(0.3, 10):
            for nm in ("left_ankle", "right_ankle"):
                sc.shadow_disc([J[I[nm]][0] + 0.06, J[I[nm]][1], FLOOR], 0.13, FLOOR)
            sc.shadow_disc([J[I["root"]][0], J[I["root"]][1], FLOOR], 0.22, FLOOR)
        if G is not None:
            with sc.group(0.5):
                sc.body(G, grey, racket=dict(hand=g.get("hand") or "right", towards=None))
        sc.body(J, col, racket=dict(hand=s.get("hand") or "right", towards=np.asarray(s["ball"], float) if abs(t) < 0.02 else None))

    def ball_():
        if s.get("arc"):
            sc.line(s["arc"], T["ink3"], 1.4, dash=(5, 5))
        if s.get("path"):
            P = np.asarray(s["path"], float); sc.line(P[: max(4, len(P) // 2)], T["ink3"], 1.4, dash=(5, 5))
        if ball is not None:
            sc.sphere(ball, 0.022, mix(T["paper"], (255, 255, 255), 0.75), edge=T["ink"], width=0.8)

    sc.floor(bands=False)
    if eye[0] > 0.0:                                                    # beyond the end line: the table is between us and him
        people(); set_(); ball_()
    elif eye[0] > root[0]:                                              # to his side
        set_(); people(); ball_()
    else:                                                               # behind him: the ball is further away than he is
        set_(); ball_(); people()
    m, pm = p["body"]["med"], (g or {}).get("med") or {}
    name = p["name"]
    if clean:                                                           # a still with no lettering: the picture alone
        cv2.imwrite(str(pathlib.Path(out_dir) / f"f{k:05d}.png"), sc.finish())
        return k
    sc.text((64, 78), "HOW TO HIT  ·  IN 3D", "demi", 24, col, spacing=4.0)
    sc.text((64, 146), f"{name}’s forehand, next to the pros’", "serif", 64, T["ink"], anchor="ls")
    sc.text((64, 190), "rebuilt in 3D from one phone video, replayed at a tenth of real speed", "regular", 26, T["ink3"])
    y0 = H_ - 96
    for i in range(40):                                                 # a soft dark band under the text at the bottom
        sc.rect((0, H_ - 220 + 5 * i, W_, H_ - 215 + 5 * i), T["paper"], alpha=0.85 * ((i + 1) / 40) ** 1.3)
    sc.rect((0, H_ - 20, W_, H_), T["paper"], alpha=0.85)
    for x, y, c_, txt in [(64, y0, col, f"{name}: typical of {p['body']['n']} forehands"), (64, y0 + 44, grey, "the pros: typical forehand, standing where he stood")]:
        sc.rect((x, y - 17, x + 20, y + 3), c_)
        sc.text((x + 34, y + 2), txt, "medium", 26, T["ink2"], anchor="ls")
    if pm:
        sc.text((W_ - 64, y0 + 2), "AT CONTACT: " + name.upper() + " (PROS)", "demi", 20, T["ink3"], anchor="rs", spacing=2.5)
        sc.text((W_ - 64, y0 + 46), f"trunk forward {m['lean']:.0f}° ({pm['lean']:.0f}°)  ·  shoulder turn {m['turn']:.0f}° ({pm['turn']:.0f}°)",
                "medium", 26, T["ink"], anchor="rs")                # no knee: it is measured in the picture, not on the 3D body (pose.py)
    img = sc.finish()
    cv2.imwrite(str(pathlib.Path(out_dir) / f"f{k:05d}.png"), img)
    return k


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("report"); ap.add_argument("--player", required=True)
    ap.add_argument("--seconds", type=float, default=20.0); ap.add_argument("--out", required=True)
    ap.add_argument("--ss", type=int, default=2, help="supersampling (3 = the plates' quality, slower)")
    ap.add_argument("--frames", help="only these frame numbers, for checking (e.g. 0,300,600)")
    ap.add_argument("--clean", action="store_true", help="no titles, legend or numbers on the frames (stills for a post)")
    a = ap.parse_args()
    n = int(round(a.seconds * 59.94))
    with tempfile.TemporaryDirectory() as td:
        ks = [int(x) for x in a.frames.split(",")] if a.frames else list(range(n))
        with Pool() as pool:
            for i, _ in enumerate(pool.imap_unordered(render, [(k, n, a.report, a.player, td, a.ss, a.clean) for k in ks], chunksize=4)):
                if i % 120 == 0:
                    print(f"{i}/{len(ks)} frames", flush=True)
        if a.frames:
            for k in ks:
                pathlib.Path(td, f"f{k:05d}.png").rename(pathlib.Path(a.out).with_name(f"{pathlib.Path(a.out).stem}_{k:05d}.png"))
            return
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "60000/1001", "-i", str(pathlib.Path(td) / "f%05d.png"),
                        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", a.out], check=True)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
