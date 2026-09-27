"""The players' bodies in 3D through every stroke: Apple Vision's 3D body pose (tools/pose3d, 17 joints in metres), anchored in the
table's frame through the calibrated camera with the feet on the floor.

How a body is placed: Vision gives the joints relative to the hips and where each falls in the picture; solving for the pose that puts
those joints where they are seen, through the camera found from the table's corners (camera.py), places the body in the room. Vision
assumes a 1.8 m person, so the whole body is then scaled about the camera until the lower ankle stands 8 cm above the floor (the
picture does not change under that scaling; the size and the distance do). Everything is estimated from one camera: good for posture
and position to a few centimetres, not a motion-capture lab.

    reqs, meta = requests(shots, assigned, fps)                 # which frames, and a box round the hitter in each
    raw = run(video, reqs, work_dir, focal)                     # tools/pose3d over those frames
    strokes = strokes_from(raw, meta, cam)                      # per rally shot: the body frame by frame, in the table's frame
"""
import json, math, pathlib, subprocess
import numpy as np, cv2
from .config import TABLE_LENGTH as L, TABLE_WIDTH as W

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "pose3d" / "pose3d"
NAMES = ["top_head", "center_head", "center_shoulder", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
         "right_wrist", "spine", "root", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
I = {n: i for i, n in enumerate(NAMES)}
BONES = [("top_head", "center_head"), ("center_head", "center_shoulder"), ("center_shoulder", "left_shoulder"),
         ("center_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
         ("right_elbow", "right_wrist"), ("center_shoulder", "spine"), ("spine", "root"), ("root", "left_hip"), ("root", "right_hip"),
         ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("right_hip", "right_knee"), ("right_knee", "right_ankle")]
FLOOR, ANKLE_Z = -0.76, -0.68
WINDOW = (-0.30, 0.15)          # s round each contact: the backswing, the contact, the start of the follow-through
FLIP_DEG = 60.0                 # a frame whose hips face this far from the frames around it was put the wrong way round (flipped())
PLANT_M = 0.03                  # for drawing, a foot is kept within this height of standing on the floor (planted())


def _orient(a, end):
    a = np.asarray(a, float).copy()
    if end == "far":
        a[..., 0] = L - a[..., 0]; a[..., 1] = W - a[..., 1]
    return a


def requests(shots, assigned, fps, step=2, window=WINDOW):
    """The frames to analyse (every step-th through each rally shot's stroke) and a box round the hitter's 2D skeleton in each.
    Returns (lines for tools/pose3d, {request id: what it is})."""
    from . import pose as P
    lines, meta = [], {}
    for s in shots:
        if s.get("serve") or s.get("contact") is None or s.get("contact_src") == "late":
            continue
        tc = s["t"]; f0 = int(round(tc * fps))
        for f in range(f0 + int(round(window[0] * fps)), f0 + int(round(window[1] * fps)) + 1, step):
            a = P.skeleton_near(assigned, f / fps, s["end"], fps, reach=3) if assigned else None
            if a is None:
                continue
            ok = a[:, 2] >= P.MIN_C
            if ok.sum() < 6:
                continue
            x0, y0 = a[ok, 0].min(), a[ok, 1].min(); x1, y1 = a[ok, 0].max(), a[ok, 1].max()
            h = max(y1 - y0, 120.0)
            cx = (x0 + x1) / 2
            half = max((x1 - x0) / 2 + 0.45 * h, 0.55 * h)                 # room for the arm and racket either side
            box = (cx - half, y0 - 0.25 * h, cx + half, y1 + 0.3 * h)
            rid = f"{s['point']}_{s['shot']}_{f}"
            lines.append(f"{f} {box[0]:.0f} {box[1]:.0f} {box[2]:.0f} {box[3]:.0f} {rid}")
            meta[rid] = dict(point=s["point"], shot=s["shot"], name=s["name"], end=s["end"], frame=f, dt=round(f / fps - tc, 4),
                             contact=s.get("contact"))
    return lines, meta


def run(video, lines, work_dir, focal):
    """tools/pose3d over the requested frames; returns the raw answers (list of dicts)."""
    work = pathlib.Path(work_dir); work.mkdir(parents=True, exist_ok=True)
    rq, out = work / "pose3d_requests.txt", work / "pose3d.jsonl"
    rq.write_text("\n".join(sorted(lines, key=lambda l: int(l.split()[0]))) + "\n")
    r = subprocess.run([str(TOOL), str(video), str(rq), str(out), "--f", f"{focal:.2f}"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("pose3d failed: " + r.stderr[-400:])
    print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "pose3d: no output")
    return [json.loads(l) for l in out.read_text().splitlines() if l.strip()]


def place(rec, cam):
    """One Vision body in the table's frame: (17 x 3 joints in metres, dict of checks) or (None, why)."""
    if not all(n in rec["joints"] and n in rec["img"] for n in NAMES):
        return None, "joints missing"
    X = np.array([rec["joints"][n] for n in NAMES], np.float64)
    uv = np.array([rec["img"][n] for n in NAMES], np.float64)
    ok, rvec, tvec = cv2.solvePnP(X, uv, cam.K, None, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return None, "no pose"
    rvec, tvec = cv2.solvePnPRefineLM(X, uv, cam.K, None, rvec, tvec)
    Rb, _ = cv2.Rodrigues(rvec)
    Xc = X @ Rb.T + tvec.ravel()                                     # camera coordinates (OpenCV: x right, y down, z forward)
    if np.any(Xc[:, 2] <= 0.3):
        return None, "behind the camera"
    rms = float(np.sqrt(np.mean(np.sum((cam.project((Xc - cam.t) @ cam.R) - uv) ** 2, axis=1))))
    Xw = (Xc - cam.t) @ cam.R                                        # world = R^T (camera - t)
    za = min(Xw[I["left_ankle"], 2], Xw[I["right_ankle"], 2])
    s = (ANKLE_Z - cam.C[2]) / (za - cam.C[2]) if abs(za - cam.C[2]) > 1e-6 else 1.0
    Xw = cam.C + s * (Xw - cam.C)                                    # the feet down to the floor: size and distance, same picture
    return Xw, dict(rms_px=round(rms, 2), scale=round(float(s), 3), height=round(float(rec.get("height", 1.8)) * float(s), 3))


REACH = 0.17                    # m from the wrist to the middle of the racket's blade, where the ball meets it


def _anchor(frames, k, ball, cam, max_shift=0.3):
    """Move a whole stroke along the camera's line of sight so that at contact the hitting wrist is a racket's reach from the ball, then
    stand each frame's lower ankle back at its height above the floor. Depth along the line of sight is what one camera measures worst;
    the picture itself hardly changes. Returns (frames, shift in m)."""
    J = frames[k]
    w = min((J[I["left_wrist"]], J[I["right_wrist"]]), key=lambda q: np.linalg.norm(q - ball))
    ray = J[I["root"]] - cam.C; ray /= np.linalg.norm(ray)
    d = ball - w; along = float(d @ ray); across = float(np.linalg.norm(d - along * ray))
    keep = math.copysign(math.sqrt(max(REACH ** 2 - across ** 2, 0.0)), along)
    shift = float(np.clip(along - keep, -max_shift, max_shift))
    out = []
    for F in frames:
        G = F + shift * ray
        G[:, 2] += ANKLE_Z - min(G[I["left_ankle"], 2], G[I["right_ankle"], 2])
        out.append(G)
    return out, shift


def side_of(J, ball, hand=None):
    """Which hand met the ball (the nearer wrist, unless given) and on which side of the body it was: ("right", "forehand"), ..."""
    wl, wr = J[I["left_wrist"]], J[I["right_wrist"]]
    hand = hand or ("left" if np.linalg.norm(wl - ball) < np.linalg.norm(wr - ball) else "right")
    lr = (J[I["left_shoulder"]] - J[I["right_shoulder"]])[:2]; lr /= max(np.linalg.norm(lr), 1e-9)
    off = float((np.asarray(ball)[:2] - J[I["root"], :2]) @ lr)               # + = on his left
    return hand, ("forehand" if (off < 0) == (hand == "right") else "backhand")


def _facing(J):
    h = J[I["left_hip"]] - J[I["right_hip"]]
    return math.atan2(h[1], h[0])


def flipped(frames):
    """Frames Vision put the wrong way round. From one side camera the front and the back of a body look much alike, and for a frame
    Vision sometimes swaps them: the hips' direction jumps by about 150 deg and back (on our 2026-09-25 recording, 16 of one player's 32
    strokes had such a frame). A real body turns 30 to 50 deg through a whole stroke, so a frame whose hips face more than FLIP_DEG
    away from the frames round it (the median over up to two either side) is one of these."""
    a = [_facing(np.asarray(J)) for J in frames]
    out = []
    for k in range(len(a)):
        d = [abs((a[k] - a[j] + math.pi) % (2 * math.pi) - math.pi) for j in (k - 2, k - 1, k + 1, k + 2) if 0 <= j < len(a)]
        if d and float(np.median(d)) > math.radians(FLIP_DEG):
            out.append(k)
    return out


def without_flips(st):
    """A stroke with its flipped frames left out (dts and the contact index kept in step), or None if the contact frame itself is
    one: the posture at contact would then be the wrong way round, so the stroke is not used at all."""
    bad = set(flipped(st["frames"]))
    if not bad:
        return st
    ci = st["contact_index"]
    if ci in bad:
        return None
    keep = [k for k in range(len(st["frames"])) if k not in bad]
    out = dict(st, frames=[st["frames"][k] for k in keep], dts=[st["dts"][k] for k in keep], contact_index=keep.index(ci))
    if st.get("checks") is not None:
        out["checks"] = [st["checks"][k] for k in keep]
    return out


def _knee_for(hip, knee, ankle, l1, l2):
    """Two-bone leg: with the hip and the ankle fixed and the thigh and shin lengths kept, the knee nearest where it was. An ankle out
    of reach is pulled in along the leg to a straight knee. Returns (knee, ankle)."""
    d = ankle - hip; n = float(np.linalg.norm(d))
    if n >= l1 + l2 - 1e-6:
        ankle = hip + d / n * (l1 + l2 - 1e-6); n = l1 + l2 - 1e-6
    u = d / max(n, 1e-9)
    a = (l1 * l1 - l2 * l2 + n * n) / (2 * n)                          # along the hip-ankle line, from the hip to the knee's circle
    c = hip + a * u; r = math.sqrt(max(l1 * l1 - a * a, 0.0))
    v = knee - c; v = v - (v @ u) * u
    if np.linalg.norm(v) < 1e-9:
        v = np.cross(u, [0.0, 0.0, 1.0])
    return c + r * v / max(np.linalg.norm(v), 1e-9), ankle


def planted(frames):
    """For drawing only: each foot kept on the floor. One leg hides the other from a side camera, and Vision's guess for the hidden
    leg often lifts its foot 10 to 35 cm off the floor where the video shows it down. Each ankle is brought down to at most PLANT_M
    above standing height (a smooth squeeze, so small real movements stay and nothing jumps between frames), its x and y kept, and
    the knee re-solved so the thigh and shin keep their lengths. The measures use the frames as estimated."""
    out = []
    for F in frames:
        G = np.array(F, float)
        for sd in ("left", "right"):
            h, k, a = G[I[f"{sd}_hip"]], G[I[f"{sd}_knee"]], G[I[f"{sd}_ankle"]]
            lift = a[2] - ANKLE_Z
            if lift <= 0:
                continue
            l1, l2 = float(np.linalg.norm(k - h)), float(np.linalg.norm(a - k))
            target = a.copy(); target[2] = ANKLE_Z + PLANT_M * math.tanh(lift / PLANT_M)
            off = target[:2] - h[:2]; dz = h[2] - target[2]; reach = l1 + l2 - 0.01
            if dz < reach and off @ off + dz * dz > reach * reach:       # too far out for the leg to reach the floor there: the foot
                target[:2] = h[:2] + off * math.sqrt(reach * reach - dz * dz) / max(np.linalg.norm(off), 1e-9)   # comes in under the hip
            G[I[f"{sd}_knee"]], G[I[f"{sd}_ankle"]] = _knee_for(h, k, target, l1, l2)
        out.append(G)
    return out


def strokes_from(raw, meta, cam, max_rms=6.0):
    """Per rally shot, the hitter's body through the stroke in the table's frame turned to the left end: {(point, shot): stroke}."""
    by = {}
    for rec in raw:
        m = meta.get(rec["id"])
        if m is None:
            continue
        Xw, chk = place(rec, cam)
        if Xw is None or chk["rms_px"] > max_rms or not 0.75 < chk["scale"] < 1.25:
            continue
        by.setdefault((m["point"], m["shot"]), []).append((m, Xw, chk))
    out = {}
    for key, fr in by.items():
        fr.sort(key=lambda x: x[0]["dt"])
        m0 = fr[0][0]
        k = int(np.argmin([abs(x[0]["dt"]) for x in fr]))
        if abs(fr[k][0]["dt"]) > 0.05 or m0.get("contact") is None:
            continue                                                  # no body near the contact itself
        bad = set(flipped([x[1] for x in fr]))                       # frames Vision put the wrong way round
        if k in bad:
            continue
        fr = [x for j, x in enumerate(fr) if j not in bad]
        k = int(np.argmin([abs(x[0]["dt"]) for x in fr]))
        ball = np.asarray(m0["contact"], float)
        frames, shift = _anchor([x[1] for x in fr], k, ball, cam)
        hand, side = side_of(frames[k], ball)                        # redone with each player's usual hand below
        out[key] = dict(point=key[0], shot=key[1], name=m0["name"], end=m0["end"], contact_index=k, hand=hand, side=side, shift=round(shift, 3),
                        dts=[x[0]["dt"] for x in fr], frames=[_orient(F, m0["end"]) for F in frames], ball=_orient(ball, m0["end"]),
                        checks=[x[2] for x in fr])
    for n in {v["name"] for v in out.values()}:                    # his racket hand is the one that meets most balls
        mine = [v for v in out.values() if v["name"] == n]
        hand = max(("right", "left"), key=lambda h: sum(v["hand"] == h for v in mine))
        for v in mine:
            v["hand"] = hand; v["side"] = side_of(v["frames"][v["contact_index"]], v["ball"], hand)[1]
    return out


def typical(strokes, side=None):
    """The real stroke nearest this player's medians of knee bend, trunk lean and shoulder turn (each scaled by its spread), among
    those where the ball is within reach of the racket at contact (so the picture of the contact is a true one)."""
    cand = [(s, measures(s)) for s in strokes if side is None or s.get("side") == side]
    good = [c for c in cand if c[1].get("wrist_ball", 0) <= 0.3]
    cand = good if len(good) >= 3 else cand
    if not cand:
        return None
    X = np.array([[m["knee"], m["lean"], m["turn"]] for _, m in cand])
    sc = np.maximum(np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0), 1.0)
    return cand[int(np.argmin(np.sum(((X - np.median(X, axis=0)) / sc) ** 2, axis=1)))][0]


def pack(st, d=3):
    """A stroke as plain lists for JSON (joints rounded to the millimetre)."""
    return dict(point=st["point"], shot=st["shot"], name=st["name"], end=st["end"], contact_index=st["contact_index"],
                hand=st.get("hand"), side=st.get("side"),
                dts=[round(float(x), 4) for x in st["dts"]], frames=[np.round(F, d).tolist() for F in st["frames"]],
                ball=None if st.get("ball") is None else np.round(st["ball"], d).tolist())


# ------------------------------------------------------------------------------------------------ measures in 3D
def _ang(a, b, c):
    u, v = a - b, c - b
    return float(np.degrees(np.arccos(np.clip(u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-9), -1, 1))))


def _line_angle(J):
    """The shoulder line's direction in the floor's plane, radians, as an undirected line (a left/right swap does not change it)."""
    d = J[I["left_shoulder"]] - J[I["right_shoulder"]]
    return math.atan2(d[1], d[0]) % math.pi


def measures(st):
    """At the contact and through the stroke: knee bend, trunk lean towards the table, hip height, stance width, shoulder turn."""
    J = st["frames"][st["contact_index"]]
    knee = float(np.mean([_ang(J[I[f"{s}_hip"]], J[I[f"{s}_knee"]], J[I[f"{s}_ankle"]]) for s in ("left", "right")]))
    sp = J[I["center_shoulder"]] - J[I["root"]]
    lean = float(np.degrees(math.atan2(sp[0], sp[2])))               # + = leaning forward, over the table's end (+x after turning)
    th = np.unwrap(2 * np.array([_line_angle(F) for F in st["frames"]])) / 2
    turn = float(np.degrees(th.max() - th.min()))                   # how far the shoulders rotate through the stroke window
    hip = float(J[I["root"], 2] - FLOOR)
    width = float(np.linalg.norm(J[I["left_ankle"], :2] - J[I["right_ankle"], :2]))
    out = dict(knee=round(knee, 1), lean=round(lean, 1), turn=round(turn, 1), hip=round(hip, 3), width=round(width, 3))
    if st.get("ball") is not None:                                   # which hand met the ball, and how far from the wrist it was
        b = np.asarray(st["ball"])
        dl, dr = (float(np.linalg.norm(J[I[f"{s}_wrist"]] - b)) for s in ("left", "right"))
        out.update(hand="left" if dl < dr else "right", wrist_ball=round(min(dl, dr), 3))
    return out
