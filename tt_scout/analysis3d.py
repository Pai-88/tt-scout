"""After the match, in 3D: where each player stood when he hit, where and when he met the ball, and where his shots went, against the
same measurements on the professional OpenTTGames matches.

Nothing here is invented or drawn by a model. Every arc is a shot's flight fitted in 3D through the calibrated camera (flight.py);
every ball is the fitted position at the racket contact; every footprint is the hitter's ankles taken down to the floor
(technique.py); the rising arc into the racket joins the fitted bounce to the fitted contact under gravity. The pros' reference is the
same measurements on OpenTTGames test_2/3/4 (pro_shots.json; CC BY-NC-SA 4.0, a reference in a non-commercial tool). The advice is
rules over those numbers. Each player is turned round to play from the left end, so both can be laid over the pros.

    a3 = build(shots, ["Sam", "Robin"], out_dir)    # plates written to out_dir; a3["players"][i]["advice"], a3["scene"] for the viewer
"""
import json, math, pathlib
import numpy as np, cv2
from .config import TABLE_LENGTH as L, TABLE_WIDTH as W, NET_X
from .render3d import Scene, FLOOR, mix, ellipse_pts, bgr, THEMES as THEMES_

G, R_BALL, NET_H = 9.81, 0.02, 0.1525
TOP = 0.03                     # s: a contact this close to the top of the bounce counts as "at the top" (under two frames at 60 fps)
MIN_ROWS = 6                   # fewer measured rally shots than this and a player gets no plates


def orient(a, end):
    """Table coordinates as seen by a player at the left end: the other end's points turned half a circle about the table's centre."""
    if a is None:
        return None
    a = np.asarray(a, float).copy()
    if end == "far":
        a[..., 0] = L - a[..., 0]
        a[..., 1] = W - a[..., 1]
    return a


def incoming_arc(inc, contact, n=16):
    """The ball rising off its bounce into the racket: the parabola under gravity through the fitted bounce and the fitted contact.
    Returns the points and the time of the top of the bounce after the bounce (s)."""
    b = np.array([inc["bounce"][0], inc["bounce"][1], R_BALL]); c = np.asarray(contact, float); dt = float(inc["dt"])
    v0 = (c - b) / dt; v0[2] += 0.5 * G * dt
    t = np.linspace(0, dt, n)
    P = b + np.outer(t, v0); P[:, 2] -= 0.5 * G * t ** 2
    return P, float(v0[2] / G)


def over_net(path):
    """cm between the ball's underside and the top of the net as the flight crossed the net's plane."""
    p = np.asarray(path)
    k = np.where(np.diff(np.sign(p[:, 0] - NET_X)) != 0)[0]
    if not len(k):
        return None
    i = k[0]; a = (NET_X - p[i, 0]) / (p[i + 1, 0] - p[i, 0])
    return 100 * (p[i, 2] + a * (p[i + 1, 2] - p[i, 2]) - R_BALL - NET_H)


def rows_of(shots):
    """Rally shots with a fitted contact, turned to the left end."""
    out = []
    for s in shots:
        if s.get("serve") or not s.get("contact"):
            continue
        e = s.get("end") or "near"
        r = dict(shot=s.get("shot"), point=s.get("point"), contact=orient(s["contact"], e), timing=s.get("timing"), speed=s.get("speed"),
                 path=orient(s["path"], e) if s.get("path") else None, feet=orient(s["feet"], e) if s.get("feet") else None,
                 late=s.get("contact_src") == "late",                  # the ball was hidden at the racket and the contact could not be pinned down
                 flight_t=s.get("flight_t"), free=bool(s.get("free")), net=bool(s.get("net")))   # free: no bounce seen; net: into the net
        if s.get("incoming") and not r["late"]:
            P, t_top = incoming_arc(s["incoming"], s["contact"])
            r.update(arc=orient(P, e), t_top=t_top, dt=float(s["incoming"]["dt"]))
        out.append(r)
    return out


def _med(v):
    v = [float(x) for x in v if x is not None and np.isfinite(x)]
    return (float(np.median(v)), len(v)) if v else (None, 0)


def metrics(rows):
    paths = [r["path"] for r in rows if r["path"] is not None and not r.get("free") and not r.get("net")]   # flights that landed
    rows = [r for r in rows if not r.get("late")]
    feet = [r for r in rows if r["feet"] is not None and len(r["feet"])]
    pairs = [r["feet"] for r in feet if len(r["feet"]) == 2]
    tm = [r["timing"] for r in rows if r["timing"] is not None]
    m = dict(feet_back=_med([-float(np.mean(r["feet"][:, 0])) for r in feet]),            # m behind the end line, feet at contact
             width=_med([float(np.linalg.norm(p[0] - p[1])) for p in pairs]),                # m between the ankles
             reach=_med([float(r["contact"][0] - np.mean(r["feet"][:, 0])) for r in feet]),  # m the ball was in front of the feet
             height=_med([100 * r["contact"][2] for r in rows]),                             # cm, ball centre above the playing surface
             contact_back=_med([-r["contact"][0] for r in rows]),                            # m behind the end line (negative: over the table)
             timing=_med(tm),
             top_share=(float(np.mean(np.array(tm) <= TOP)), len(tm)) if tm else (None, 0),
             net=_med([over_net(p) for p in paths]),
             depth=_med([float(p[-1][0] - NET_X) for p in paths]),
             apex=_med([100 * float(np.max(p[:, 2])) for p in paths]))
    if pairs:                                                        # the typical stance: each foot's median, the pair ordered across the table
        srt = np.array([p[np.argsort(p[:, 1])] for p in pairs])
        m["stance"] = [np.median(srt[:, 0], axis=0), np.median(srt[:, 1], axis=0)]
    mids = np.array([np.mean(r["feet"], axis=0) for r in feet]) if feet else np.zeros((0, 2))
    m["mid"] = np.median(mids, axis=0) if len(mids) else None
    m["mid_cov"] = np.cov(mids.T) if len(mids) >= 5 else None
    lands = np.array([p[-1][:2] for p in paths]) if paths else np.zeros((0, 2))
    m["land"] = np.median(lands, axis=0) if len(lands) else None
    m["land_cov"] = np.cov(lands.T) if len(lands) >= 5 else None
    return m


def _medoid(rows, keys):
    """The real shot closest to the player's medians on these measures (each scaled by its spread): 'a typical shot', not an average."""
    cand = [r for r in rows if all(k(r) is not None for k in keys)]
    if not cand:
        return None
    X = np.array([[k(r) for k in keys] for r in cand], float)
    s = np.maximum(np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0), 1e-6)
    return cand[int(np.argmin(np.sum(((X - np.median(X, axis=0)) / s) ** 2, axis=1)))]


def typical(rows):
    fl = _medoid([r for r in rows if r["path"] is not None and not r.get("free") and not r.get("net")],
                 [lambda r: over_net(r["path"]), lambda r: float(r["path"][-1][0]), lambda r: float(np.max(r["path"][:, 2]))])
    st = _medoid([r for r in rows if r.get("arc") is not None and r["timing"] is not None and not r.get("late")],
                 [lambda r: r["timing"], lambda r: r["timing"], lambda r: float(r["contact"][2]), lambda r: float(r["contact"][0])])
    return dict(flight=fl, strike=st)


# ------------------------------------------------------------------------------------------------ advice
def _f(v, d=2):
    return f"{v:.{d}f}"


def advice(name, m, pm):
    """What to change, in a coach's words, with the numbers behind each point. Returns {topic: dict(head, body, fix: bool)}."""
    out = {}
    fb, pb = m["feet_back"][0], pm["feet_back"][0]
    wd, pw = m["width"][0], pm["width"][0]
    rc, pr = m["reach"][0], pm["reach"][0]
    stand = []
    if fb is not None and pb is not None:
        if fb < pb - 0.15:
            stand.append(("Stand further back", f"His feet were {_f(fb)} m behind the end line when he hit; the pros' {_f(pb)} m. Start each rally about "
                          f"{_f(pb - fb, 1)} m further back so the ball comes to him instead of at him."))
        elif fb > pb + 0.25:
            stand.append(("Move in closer", f"His feet were {_f(fb)} m behind the end line when he hit; the pros' {_f(pb)} m. At this pace, closer lets "
                          "him take the ball earlier."))
    if wd is not None and pw is not None and wd < pw - 0.12:
        stand.append(("Widen the stance", f"His ankles were {_f(wd)} m apart at contact; the pros' {_f(pw)} m. A wider base keeps the body low and "
                      "turning, not standing up through the stroke."))
    if rc is not None and pr is not None:
        if rc < pr - 0.12:
            stand.append(("Give the ball room", f"He met it {_f(rc)} m in front of his feet; the pros {_f(pr)} m. Step back or aside so the stroke meets "
                          "the ball in front, not against the body."))
        elif rc > pr + 0.25:
            stand.append(("Move the feet, not the arm", f"He met it {_f(rc)} m in front of his feet; the pros {_f(pr)} m. That is reaching: step to "
                          "the ball first."))
    out["stand"] = dict(items=stand, ok=not stand,
                        summary=(f"Feet {_f(fb)} m behind the end line (pros {_f(pb)} m), {_f(wd)} m apart (pros {_f(pw)} m)."
                                 if fb is not None and wd is not None and pb is not None and pw is not None else ""))
    ts, pts_ = m["top_share"][0], pm["top_share"][0]; tm = m["timing"][0]; ht, ph = m["height"][0], pm["height"][0]
    strike = []
    if ts is not None and pts_ is not None and ts < pts_ - 0.12:
        strike.append(("Take the ball at the top of the bounce", f"{ts:.0%} of his shots were hit at the top or on the rise; the pros {pts_:.0%}. "
                       f"The rest he let drop, typically {_f(tm)} s past the top. Move in as it rises and hit it at its highest point."))
    cb, pcb = m["contact_back"][0], pm["contact_back"][0]
    if cb is not None and pcb is not None and cb < -0.05 and pcb > 0.1:
        strike.append(("Hit from behind the end line", f"His typical contact was {abs(cb) * 100:.0f} cm in over the table; the pros' {pcb * 100:.0f} cm "
                       "behind the end line. Over the table the stroke has no room to swing."))
    out["strike"] = dict(items=strike, ok=not strike,
                         summary=(f"{ts:.0%} at the top or rising (pros {pts_:.0%}); contact {ht:.0f} cm above the table (pros {ph:.0f} cm)."
                                  if ts is not None and pts_ is not None and ht is not None and ph is not None else ""))
    on, pon = m["net"][0], pm["net"][0]; dp, pdp = m["depth"][0], pm["depth"][0]
    flight = []
    if on is not None and pon is not None and on > pon + 5:
        flight.append(("Lower over the net", f"His shots crossed {on:.0f} cm over the net; the pros' {pon:.0f} cm. Close the racket a little and "
                       "brush up the back of the ball: topspin brings a lower ball down onto the table."))
    if dp is not None and pdp is not None and dp < pdp - 0.12:
        flight.append(("Aim deeper", f"They landed {_f(dp)} m past the net; the pros' {_f(pdp)} m. Aim for the last third of the table: a deep ball "
                       "pushes the opponent back and gives him less angle."))
    out["flight"] = dict(items=flight, ok=not flight,
                         summary=(f"{on:.0f} cm over the net (pros {pon:.0f} cm), landing {_f(dp)} m past it (pros {_f(pdp)} m)."
                                  if on is not None and pon is not None and dp is not None and pdp is not None else ""))
    return out


# ------------------------------------------------------------------------------------------------ plates
def _legend(sc, items, x, y):
    """items: (kind, colour, text); kind 'ball', 'ring', 'line', 'dash', 'zone'. Drawn along a baseline from (x, y), final pixels."""
    from .render3d import font
    k, ss = 16, sc.ss
    f = font("medium", 17)
    for kind, col, text in items:
        cx, cy = int((x + 12) * ss * k), int((y - 6) * ss * k)
        if kind == "ball":
            cv2.circle(sc.img, (cx, cy), 7 * ss * k, bgr(col), -1, cv2.LINE_AA, 4)
        elif kind == "ring":
            cv2.circle(sc.img, (cx, cy), 7 * ss * k, bgr(col), int(1.8 * ss), cv2.LINE_AA, 4)
        elif kind == "line":
            cv2.line(sc.img, (int(x * ss * k), cy), (int((x + 24) * ss * k), cy), bgr(col), int(3.2 * ss), cv2.LINE_AA, 4)
        elif kind == "dash":
            for i in range(3):
                x0 = x + i * 9
                cv2.line(sc.img, (int(x0 * ss * k), cy), (int((x0 + 5) * ss * k), cy), bgr(col), int(3 * ss), cv2.LINE_AA, 4)
        elif kind == "zone":
            cv2.ellipse(sc.img, (cx, cy), (12 * ss * k, 6 * ss * k), 0, 0, 360, bgr(col), int(1.5 * ss), cv2.LINE_AA, 4)
        sc.text((x + 38, y), text, "medium", 17, sc.T["ink2"], anchor="ls")
        x += 38 + f.getlength(text) + 36


def _title(sc, kicker, title, sub):
    sc.text((56, 66), kicker.upper(), "demi", 16, sc.T["ink3"], spacing=3.2)
    sc.text((56, 116), title, "serif", 44, sc.T["ink"], anchor="ls")
    if sub:
        sc.text((56, 152), sub, "regular", 21, sc.T["ink2"], anchor="ls")


def _dim(sc, a, b, text, col, sub=None, off=(0, -16), size=25, tick=0.07):
    """A dimension line between two floor points, with end ticks and its measure at the middle (engineering-drawing style)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    sc.line([a, b], col, 1.5)
    d = (b - a)[:2]; n = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-6)
    for q in (a, b):
        sc.line([[q[0] - n[0] * tick, q[1] - n[1] * tick, q[2]], [q[0] + n[0] * tick, q[1] + n[1] * tick, q[2]]], col, 1.5)
    sc.label((a + b) / 2, text, sub, dx=off[0], dy=off[1], col=col, align="center", size=size, leader=False, dot=False)


def _view(eye, target, fov, size):
    """Scene's camera (render3d.Scene), without drawing: (right, down, forward, focal px) for projecting."""
    f = target - eye; f = f / np.linalg.norm(f)
    up = np.array([0, 0, 1.0]) if abs(f[2]) < 0.97 else np.array([1.0, 0, 0])
    r = np.cross(f, up); r /= np.linalg.norm(r); u = np.cross(r, f)
    return r, -u, f, (size[0] / 2) / math.tan(math.radians(fov) / 2)


def _fit(eye, target, fov, size, pts, pad=(60, 90, 60, 110), max_scale=3.0):
    """Eye and target for a picture that shows every point in pts: the same viewing angle, the view centred on them and the camera
    backed away along its line of sight just far enough that they all fall inside the frame with pad pixels to spare (left, top,
    right, bottom). Never closer than the given eye (a picture that already fits is unchanged). Returns (eye, target)."""
    eye, target, P = np.asarray(eye, float), np.asarray(target, float), np.asarray(pts, float)
    w, h = size

    def project(e, t):
        r, d, f, fp = _view(e, t, fov, size); X = P - e
        z = X @ f
        return np.c_[fp * (X @ r) / np.maximum(z, 1e-3) + w / 2, fp * (X @ d) / np.maximum(z, 1e-3) + h / 2], z

    def fits(e, t):
        uv, z = project(e, t)
        return (z > 0.1).all() and uv[:, 0].min() >= pad[0] and uv[:, 1].min() >= pad[1] and uv[:, 0].max() <= w - pad[2] and uv[:, 1].max() <= h - pad[3]

    if fits(eye, target):
        return eye, target
    back, s = eye - target, 1.0                                        # the camera moves with its target: the angle never changes
    for _ in range(3):                                                  # centre on the points, then back away until they fit
        e = target + s * back
        uv, _ = project(e, target)
        r, d, f, fp = _view(e, target, fov, size)
        mpp = s * float(np.linalg.norm(back)) / fp                      # metres per pixel at the target's distance
        cx = (uv[:, 0].min() + uv[:, 0].max()) / 2 - (w + pad[0] - pad[2]) / 2
        cy = (uv[:, 1].min() + uv[:, 1].max()) / 2 - (h + pad[1] - pad[3]) / 2
        target = target + r * cx * mpp + d * cy * mpp
        s = 1.0
        while s < max_scale and not fits(target + s * back, target):
            s += 0.02
    return target + s * back, target


def plate_stand(p, pro, theme="light", size=(1600, 900), titled=False):
    """Where he stood when he hit, from behind and above his end: every contact's ankles, his typical stance, the pros' zone and stance,
    the move between them, and where the ball was when he met it."""
    m, pm = p["m"], pro["m"]
    keep = [[0, 0, 0], [0, W, 0], [0, -0.9, FLOOR], [0, W + 0.9, FLOOR]]              # every foot, the table's end and the end line
    keep += [[f[0], f[1], FLOOR] for r in p["rows"] if r["feet"] is not None and not r.get("late") for f in r["feet"]]
    keep += [[f[0], f[1], FLOOR] for f in (m.get("stance") or []) + (pm.get("stance") or [])]
    if pm.get("mid_cov") is not None:
        keep += ellipse_pts(pm["mid"], pm["mid_cov"], k=1.5, z=FLOOR).tolist()
    eye, target = _fit((-4.9, W / 2 - 1.35, 3.95), (-0.5, W / 2 + 0.02, FLOOR + 0.3), 39, size, keep)
    sc = Scene(eye=eye, target=target, fov=39, size=size, theme=theme)
    T = sc.T; col = T[p["key"]]; dark = mix(col, (0, 0, 0), 0.45)
    sc.floor(); sc.table()
    sc.line([[0, -0.9, FLOOR + 0.002], [0, W + 0.9, FLOOR + 0.002]], T["ink3"], 1.3, dash=(10, 6))          # the end line, on the floor
    sc.label([0, W + 0.9, FLOOR], "end line", None, dx=10, dy=4, col=T["ink3"], size=16, leader=False, dot=False, face="medium")
    for xb in (0.5, 1.0, 1.5, 2.0):
        sc.label([-xb, W + 0.9, FLOOR], f"{xb:.1f} m", None, dx=10, dy=4, col=T["ink3"], size=16, leader=False, dot=False, face="medium")
    if pm.get("mid_cov") is not None:                                                    # the pros' zone: 68% of their stances
        E = ellipse_pts(pm["mid"], pm["mid_cov"], k=1.5, z=FLOOR + 0.002)
        with sc.group(0.09):
            sc.poly(E, T["pro"])
        sc.line(E, T["pro"], 1.5, dash=(9, 7), closed=True)
    with sc.group(0.4):                                                                  # his ankles at every contact
        for r in p["rows"]:
            if r["feet"] is not None and not r.get("late"):
                for f in r["feet"]:
                    sc.disc(f, 0.03, FLOOR + 0.004, col)
    if m.get("stance") is not None:
        for f in m["stance"]:
            sc.disc(f, 0.13, FLOOR + 0.006, col, sx=1.0, sy=0.42, edge=dark, width=1.6)
    if pm.get("stance") is not None:
        for f in pm["stance"]:
            E = [[f[0] + 0.13 * math.cos(t), f[1] + 0.055 * math.sin(t), FLOOR + 0.008] for t in np.linspace(0, 2 * np.pi, 40, endpoint=False)]
            sc.line(E, T["pro"], 2.2, closed=True)
    if m.get("mid") is not None and pm.get("mid") is not None:
        a_, b_ = np.asarray(m["mid"]), np.asarray(pm["mid"])
        if np.linalg.norm(a_ - b_) > 0.12:
            sc.arrow_flat(a_ + (b_ - a_) * 0.2, a_ + (b_ - a_) * 0.85, FLOOR + 0.01, T["rust"], width=0.035, head=0.12)
        ys = [f[1] for f in (m.get("stance") or [])] + [f[1] for f in (pm.get("stance") or [])] or [W / 2]
        y1, y2 = min(ys) - 0.32, max(ys) + 0.32
        _dim(sc, [0, y1, FLOOR + 0.004], [a_[0], y1, FLOOR + 0.004], f"{-a_[0]:.2f} m", col, sub=f"{p['name']}'s feet")
        _dim(sc, [0, y2, FLOOR + 0.004], [b_[0], y2, FLOOR + 0.004], f"{-b_[0]:.2f} m", T["pro"], sub="pros' feet")
    rc = m["reach"][0]
    if rc is not None and m.get("mid") is not None and m["height"][0] is not None:       # where the ball was, relative to the feet
        ball = np.array([m["mid"][0] + rc, m["mid"][1], m["height"][0] / 100])
        base = [ball[0], ball[1], sc.under(ball) + 0.003]
        with sc.shadows(0.35, 4):
            sc.shadow_disc(ball, 0.03, sc.under(ball))
        sc.line([[m["mid"][0], m["mid"][1], FLOOR + 0.012], [ball[0], ball[1], FLOOR + 0.012]], col, 1.6, dash=(7, 5))
        sc.line([ball, base], col, 1.2, dash=(4, 4))
        sc.sphere(ball, 0.03, col)
        sc.label(ball, f"{rc:.2f} m in front", f"where he met the ball\npros {pm['reach'][0]:.2f} m in front of their feet", dx=-330, dy=-40, col=col, size=25,
                 align="right")
    if titled:
        _title(sc, f"After the match · {p['name']} · where to stand", "Where he stood when he hit", "feet at the moment of contact, every rally shot")
    _legend(sc, [("ball", col, f"{p['name']}'s ankles at each contact ({m['feet_back'][1]} shots)"), ("zone", T["pro"], "where pros stand (68%)"),
                 ("line", T["rust"], "the move to make")], 56, size[1] - 38)
    return sc.finish()


def plate_strike(p, pro, theme="light", size=(1600, 900), titled=False):
    """Where and when he met the ball, from the side: the ball rising off its bounce into the racket; filled if he took it at the top
    or on the rise, a ring if he let it drop first."""
    m, pm = p["m"], pro["m"]
    sc = Scene(eye=(-0.3, -2.95, 0.62), target=(-0.18, W / 2, 0.02), fov=50, size=size, theme=theme)
    T = sc.T; col = T[p["key"]]
    rows = [r for r in p["rows"] if not r.get("late")]
    sc.floor(); sc.table(); sc.net()
    with sc.shadows(0.25, 4):
        for r in rows:
            sc.shadow_disc(r["contact"], 0.022, sc.under(r["contact"]))
    with sc.group(0.22):
        for r in rows:
            if r.get("arc") is not None:
                sc.line(r["arc"], col, 1.3)
            sc.line([r["contact"], [r["contact"][0], r["contact"][1], sc.under(r["contact"])]], col, 0.8)
    for r in sorted(rows, key=lambda r: -float(sc.cam([r["contact"]])[0, 2])):
        sc.sphere(r["contact"], 0.021, col, hollow=not (r["timing"] is not None and r["timing"] <= TOP), width=1.9)
    t_pro, t = pro["typ"]["strike"], p["typ"]["strike"]
    up = sc.project([t["contact"]])[0][0] if t is not None else None
    ur = sc.project([t_pro["contact"]])[0][0] if t_pro is not None else None
    side = 1 if (up is None or ur is None or ur[0] >= up[0]) else -1                    # pros' label on their side, his on the other
    if t_pro is not None:
        sc.line(t_pro["arc"], T["pro"], 2.2, dash=(10, 7)); sc.sphere(t_pro["contact"], 0.03, T["pro"], hollow=True, width=2.4)
        sc.label(t_pro["contact"], "pros: at the top" if t_pro["timing"] <= TOP else f"pros: {t_pro['timing']:.2f} s after the top",
                 f"{100 * t_pro['contact'][2]:.0f} cm above the table", dx=70 * side, dy=-130, col=T["pro"], size=24, align="left" if side > 0 else "right")
    if t is not None:
        sc.tube(t["arc"], col, 5)
        P = t["arc"]; k = int(np.argmax(P[:, 2]))
        if 0 < k < len(P) - 1:
            sc.line([P[k] + [-0.05, 0, 0.014], P[k] + [0.05, 0, 0.014]], T["ink"], 1.8)
            sc.label(P[k] + [0, 0, 0.02], "top of the bounce", None, dx=0, dy=-12, col=T["ink2"], size=17, leader=False, dot=False, face="medium", align="center")
        sc.sphere(t["contact"], 0.032, col)
        sc.label(t["contact"], f"{t['timing']:.2f} s after the top" if t["timing"] > TOP else "at the top",
                 f"{p['name']}'s typical contact\n{100 * t['contact'][2]:.0f} cm above the table", dx=-70 * side, dy=-150, col=col, size=26,
                 align="right" if side > 0 else "left")
    ts, pts_ = m["top_share"], pm["top_share"]
    if ts[0] is not None and pts_[0] is not None:                                       # the share hit at the top, him against the pros
        x0, y0, wbar = size[0] - 500, size[1] - 176, 330
        sc.rect((x0 - 26, y0 - 44, size[0] - 30, y0 + 104), T["paper"], 0.9)
        sc.text((x0, y0 - 14), "TAKEN AT THE TOP OR RISING", "demi", 15, T["ink3"], spacing=2.2)
        for i, (nm, v, n, c_) in enumerate(((p["name"], ts[0], ts[1], col), ("pros", pts_[0], pts_[1], T["pro"]))):
            y = y0 + 22 + i * 44
            sc.text((x0, y - 6), f"{nm}  ({n} shots)", "medium", 15, T["ink2"], anchor="ls")
            sc.rect((x0, y, x0 + wbar, y + 12), mix(T["paper"], T["ink"], 0.12))
            sc.rect((x0, y, x0 + wbar * v, y + 12), c_)
            sc.text((x0 + wbar + 16, y + 14), f"{v:.0%}", "serif", 26, c_, anchor="ls")
    if titled:
        _title(sc, f"After the match · {p['name']} · how to strike", "Where and when he met the ball", "the ball rising off the bounce into his racket")
    _legend(sc, [("ball", col, "taken at the top or rising"), ("ring", col, "let drop first"), ("dash", T["pro"], "pros, a typical contact")],
            56, size[1] - 38)
    return sc.finish()


def plate_flight(p, pro, theme="light", size=(1600, 900), titled=False):
    """Where his shots went: every fitted flight from the racket to the bounce, a typical one of his against a typical one of the pros'."""
    m, pm = p["m"], pro["m"]
    p = dict(p, rows=[r for r in p["rows"] if not r.get("free") and not r.get("net")])     # the flights that landed
    keep = [[0, 0, 0], [0, W, 0], [L, W, 0], [L, 0, 0]]                                # the whole table and every flight, end to end
    keep += [q for r in p["rows"] if r["path"] is not None for q in np.asarray(r["path"])[::2].tolist()]
    for t_ in (pro["typ"]["flight"], p["typ"]["flight"]):
        if t_ is not None:
            keep += np.asarray(t_["path"]).tolist()
    eye, target = _fit((0.45, -3.85, 1.6), (1.6, W / 2, -0.06), 44, size, keep, pad=(60, 150, 60, 110))
    sc = Scene(eye=eye, target=target, fov=44, size=size, theme=theme)
    T = sc.T; col = T[p["key"]]
    sc.floor(); sc.table()
    if pm.get("land_cov") is not None:                                                   # where the pros' shots land
        E = ellipse_pts(pm["land"], pm["land_cov"], k=1.5, z=0.001)
        E[:, 0] = np.clip(E[:, 0], NET_X + 0.02, L - 0.02); E[:, 1] = np.clip(E[:, 1], 0.02, W - 0.02)
        with sc.group(0.14):
            sc.poly(E, T["line"])
        sc.line(E, T["line"], 1.6, dash=(9, 7), closed=True)
    with sc.shadows(0.26, 5):
        for r in p["rows"]:
            if r["path"] is not None:
                sc.shadow_path(r["path"], 2.0, table_only=True)
    with sc.group(0.8):
        for r in p["rows"]:
            if r["path"] is not None:
                q = r["path"][-1]
                sc.line([[q[0] + 0.035 * math.cos(a), q[1] + 0.035 * math.sin(a), 0.002] for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)],
                        col, 1.6, closed=True)
    sc.net()
    with sc.group(0.26):
        for r in p["rows"]:
            if r["path"] is not None:
                sc.line(r["path"], col, 1.4)
    tp, t = pro["typ"]["flight"], p["typ"]["flight"]
    if tp is not None:
        sc.line(tp["path"], T["pro"], 2.2, dash=(10, 7))
    if t is not None:
        with sc.shadows(0.4, 4):
            sc.shadow_path(t["path"], 4, table_only=True)
        sc.tube(t["path"], col, 5.5)
        sc.sphere(t["path"][0], 0.026, col)
        P = t["path"]; k = np.where(np.diff(np.sign(P[:, 0] - NET_X)) != 0)[0]
        if len(k) and m["net"][0] is not None:
            i = k[0]; a = (NET_X - P[i, 0]) / (P[i + 1, 0] - P[i, 0]); q = P[i] + a * (P[i + 1] - P[i])
            sc.line([[NET_X, q[1], NET_H], [NET_X, q[1], q[2] - R_BALL]], T["ink"], 1.6)
            sc.label([NET_X, q[1], q[2]], f"{m['net'][0]:.0f} cm", f"over the net, typically\npros {pm['net'][0]:.0f} cm", dx=-120, dy=-110,
                     col=col, size=28, align="right")
        if m["depth"][0] is not None:
            sc.label(P[-1], f"{m['depth'][0]:.2f} m", f"past the net, typically\npros {pm['depth'][0]:.2f} m", dx=90, dy=-120, col=col, size=28)
    if titled:
        _title(sc, f"After the match · {p['name']} · where it went", "What the ball did after he hit it", "every fitted flight, racket to bounce")
    _legend(sc, [("line", col, f"{p['name']}, a typical shot"), ("ring", col, "where each landed"), ("dash", T["pro"], "pros, a typical shot"),
                 ("zone", T["ink3"], "where pros land (68%)")], 56, size[1] - 38)
    return sc.finish()


def _arc3(sc, centre, u, v, r, deg_from, deg_to, col, width=1.6, n=40):
    """An arc of radius r round centre in the plane spanned by unit vectors u, v (angles from u towards v)."""
    a = np.radians(np.linspace(deg_from, deg_to, n))
    sc.line([centre + r * (math.cos(t) * u + math.sin(t) * v) for t in a], col, width)


def _angle_marks(sc, J, T, lean_lbl, turn_lbl, turn_prev=None, col=None, left=True):
    """Trunk lean against the vertical, the shoulder line now and at the backswing. No knee: the 3D knee is a template's guess
    whenever the leg is edge-on to the camera (pose.py, top), so the knee bend is measured in the picture and reported elsewhere."""
    from .body3d import I
    ink = T["ink"]; col = col or ink
    root, cs = J[I["root"]], J[I["center_shoulder"]]
    sc.line([root, root + [0, 0, 0.62]], T["ink3"], 1.3, dash=(6, 5))
    sp = (cs - root) / np.linalg.norm(cs - root); up = np.array([0, 0, 1.0])
    v2 = sp - (sp @ up) * up
    if np.linalg.norm(v2) > 1e-6:
        v2 /= np.linalg.norm(v2)
        _arc3(sc, root, up, v2, 0.36, 0, math.degrees(math.acos(float(np.clip(sp @ up, -1, 1)))), ink, 1.8)
    sc.label(root + [0, 0, 0.4], lean_lbl[0], lean_lbl[1], dx=-110 if left else 110, dy=-20, col=col, size=26, align="right" if left else "left")
    if turn_prev is not None:
        sc.line([turn_prev[I["left_shoulder"]], turn_prev[I["right_shoulder"]]], T["ink3"], 1.6, dash=(6, 5))
    sc.label(cs + [0, 0, 0.1], turn_lbl[0], turn_lbl[1], dx=-70 if left else 70, dy=-95, col=col, size=26, align="right" if left else "left")


def _pose_panel(st, m, T_name, label, col_key, theme, size, pro=False):
    """One body at the contact of a typical stroke, seen from in front and to the side of the hitting arm."""
    from .body3d import I
    J = np.asarray(st["frames"][st["contact_index"]]); root = J[I["root"]]
    sc = Scene(eye=(root[0] + 2.35, root[1] - 4.1, root[2] + 0.42), target=(root[0] + 0.12, root[1] - 0.05, root[2] - 0.06), fov=24,
               size=size, theme=theme)
    T = sc.T; col = T["pro"] if pro else T[col_key]
    body_col = mix(T["paper"], T["ink"], 0.55) if pro else col
    sc.floor(bands=False); sc.table(); sc.net()
    with sc.shadows(0.3, 10):
        for n in ("left_ankle", "right_ankle"):
            sc.shadow_disc([J[I[n]][0] + 0.06, J[I[n]][1], FLOOR], 0.13, FLOOR)
        sc.shadow_disc([root[0], root[1], FLOOR], 0.22, FLOOR)
    ball = np.asarray(st["ball"]) if st.get("ball") is not None else None
    if st.get("arc") is not None:
        sc.line(st["arc"], T["ink3"], 1.4, dash=(5, 5))
    if st.get("path") is not None:
        P = np.asarray(st["path"]); sc.line(P[: max(4, len(P) // 3)], T["ink3"], 1.4, dash=(5, 5))
    ths = [math.atan2(*(F[I["left_shoulder"]] - F[I["right_shoulder"]])[1::-1]) % math.pi for F in st["frames"]]
    th = np.unwrap(2 * np.array(ths)) / 2
    k0 = int(np.argmax(np.abs(th - th[st["contact_index"]])))
    sc.body(J, body_col, racket=dict(hand=st.get("hand") or "right", towards=ball))
    if ball is not None:
        sc.sphere(ball, 0.02, mix(T["paper"], (255, 255, 255), 0.6), edge=T["ink"], width=0.8)
    lab = m
    _angle_marks(sc, J, T, (f"{lab['lean']:.0f}°", "trunk forward\nfrom upright"), (f"{lab['turn']:.0f}°", "shoulder turn,\nbackswing to contact"),
                 turn_prev=np.asarray(st["frames"][k0]), col=col)
    sc.text((40, 60), label.upper(), "demi", 16, col, spacing=3.0)
    sc.text((40, 92), T_name, "serif", 30, T["ink"], anchor="ls")
    return sc.finish()


def plate_pose(p, pro, theme="light", size=(1600, 900), titled=False):
    """How he stood into the ball: his typical forehand at the moment of contact beside the pros' typical forehand, as 3D bodies, with
    the trunk's lean and the shoulder turn marked on each (medians over all the forehands). The knee bend is not marked: it is
    measured in the picture (pose.py) and shown in the posture section."""
    b, pb = p.get("body"), pro.get("body")
    if not b or not pb:
        return None
    wl = size[0] // 2
    left = _pose_panel(b["typ"], b["m"], p["name"], f"forehand at contact · {b['n']} strokes", p["key"], theme, (wl, size[1]))
    right = _pose_panel(pb["typ"], pb["m"], "The pros", f"forehand at contact · {pb['n']} strokes", "pro", theme, (size[0] - wl, size[1]), pro=True)
    img = np.hstack([left, right])
    T = THEMES_[theme] if isinstance(theme, str) else theme
    cv2.line(img, (wl, 30), (wl, size[1] - 30), bgr(mix(T["paper"], T["ink"], 0.25)), 1, cv2.LINE_AA)
    return img


def body_summary(strokes, side="forehand"):
    """The measures of one player's strokes of one side: medians, count, and a typical stroke to draw."""
    from . import body3d
    mine = [s for s in strokes if s.get("side") == side]
    if len(mine) < 3:
        return None
    ms = [body3d.measures(s) for s in mine]
    med = {k: float(np.median([m[k] for m in ms if m.get(k) is not None])) for k in ("lean", "turn", "hip", "width")}
    return dict(n=len(mine), m=med, typ=body3d.typical(mine, side))


def pose_advice(name, bm, pm):
    """Posture at contact from the 3D bodies, forehands, against the pros': trunk lean and shoulder turn. The knee is judged from the
    picture in critique.py (the 3D knee is unreliable on an edge-on leg: pose.py, top), so the old "+10 deg on the mean of both 3D
    knees" rule is gone from here."""
    if not bm or not pm:
        return None
    m, q = bm["m"], pm["m"]
    items = []
    if m["lean"] < q["lean"] - 7:
        items.append(("Lean into the ball", f"His trunk was {m['lean']:.0f}° forward of upright at contact; the pros' {q['lean']:.0f}°. Bend from the hips so "
                      "the head is over the ball."))
    if m["turn"] < q["turn"] - 12:
        items.append(("Turn the shoulders", f"His shoulders turned {m['turn']:.0f}° from the backswing to the contact; the pros' {q['turn']:.0f}°. Turn back from "
                      "the waist before the stroke and through the ball: that turn is where the pace comes from."))
    return dict(items=items, ok=not items, summary=f"Trunk {m['lean']:.0f}° forward (pros {q['lean']:.0f}°), shoulders turn {m['turn']:.0f}° "
                f"(pros {q['turn']:.0f}°); {bm['n']} of his forehands. Knee bend is read from the picture, in the posture section.")


def _load_pro_bodies():
    """The pros' strokes in 3D (analysis/pro_bodies.py), as body3d strokes."""
    from . import body3d
    try:
        d = json.loads(pathlib.Path(__file__).with_name("pro_bodies.json").read_text())
    except Exception:
        return []
    out = []
    for s in d.get("strokes", []):
        s = dict(s); s["frames"] = [np.asarray(F, float) for F in s["frames"]]
        s["ball"] = None if s.get("ball") is None else np.asarray(s["ball"], float)
        s = body3d.without_flips(s)                                    # a guard only: since 2026-09-28 the file is written with the flipped
        if s is not None:                                              # frames already out (body3d.strokes_from), so critique.PRO3D sees the same strokes
            out.append(s)
    return out


PLATES = dict(stand=plate_stand, strike=plate_strike, flight=plate_flight, pose=plate_pose)


# ------------------------------------------------------------------------------------------------ assembly
def _load_pro():
    try:
        return json.loads(pathlib.Path(__file__).with_name("pro_shots.json").read_text())
    except Exception:
        return None


def _join(strokes, rows):
    """Give each 3D stroke its ball's rising arc and flight from the matching shot."""
    by = {(r["point"], r["shot"]): r for r in rows}
    for s in strokes:
        r = by.get((s["point"], s["shot"]))
        if r is not None:
            s["arc"] = r.get("arc"); s["path"] = r.get("path"); s["flight_t"] = r.get("flight_t"); s["dt_in"] = r.get("dt")
    return strokes


def analyse(shots, names, strokes=None):
    """Per player (turned to the left end) and for the pros: rows, metrics, typical shots, advice; with strokes (body3d) the bodies too."""
    pj = _load_pro()
    prows = rows_of(pj["shots"]) if pj else []
    pro = dict(rows=prows, m=metrics(prows) if prows else None, typ=typical(prows) if prows else dict(flight=None, strike=None),
               source=(pj or {}).get("source"), licence=(pj or {}).get("licence"))
    pstrokes = _load_pro_bodies() if strokes else []
    pro["strokes"] = pstrokes
    pro["body"] = body_summary(pstrokes) if pstrokes else None
    out = []
    for i, n in enumerate(names):
        rows = rows_of([s for s in shots if s.get("name") == n])
        if len(rows) < MIN_ROWS or pro["m"] is None:
            continue
        m = metrics(rows)
        mine = _join([s for s in (strokes or []) if s["name"] == n], rows)
        body = body_summary(mine) if mine else None
        adv = advice(n, m, pro["m"])
        pa = pose_advice(n, body, pro["body"])
        if pa:
            adv["pose"] = pa
        out.append(dict(name=n, key="p1" if i == 0 else "p2", rows=rows, m=m, typ=typical(rows), advice=adv, strokes=mine, body=body))
    return out, pro


def _r(a, d=2):
    return None if a is None else np.round(np.asarray(a, float), d).tolist()


def _smooth(frames):
    """The pose samples lightly smoothed through time (1-4-1: a third of Vision's frame-to-frame jitter out, the swing kept), for the
    viewer; its numbers are measured on these same frames (_stroke_json). The plates' medians (body_summary) use the raw poses."""
    F = np.asarray(frames, float)
    if len(F) < 3:
        return F
    G = F.copy()
    G[1:-1] = (F[:-2] + 4 * F[1:-1] + F[2:]) / 6
    return G


def _stroke_json(s):
    """A stroke for the viewer: the frames it draws (feet planted, lightly smoothed, rounded to the cm) and its measures taken on
    THOSE frames, so the numbers under the picture are the picture's (until 2026-09-28 m was measured on the raw frames, 20 to 40 deg
    off what was drawn on some strokes)."""
    from . import body3d
    frames = [_r(F) for F in _smooth(body3d.planted(s["frames"]))]
    m = body3d.measures(dict(s, frames=[np.asarray(F, float) for F in frames]))
    return dict(id=f"{s['point']}_{s['shot']}", point=s["point"], shot=s["shot"], side=s.get("side"), hand=s.get("hand"),
                dts=[round(float(x), 3) for x in s["dts"]], ci=s["contact_index"], frames=frames,
                ball=_r(s.get("ball")), arc=_r(s.get("arc")), path=_r(np.asarray(s["path"])[::2]) if s.get("path") is not None else None,
                ft=s.get("flight_t"), dti=s.get("dt_in"), m={k: v for k, v in m.items() if k in ("lean", "turn")})


def viewer_scene(players, pro, max_paths=160):
    """Compact scene data for the report's 3D viewer: every shot (flights, contacts, feet) and every stroke's body."""
    def pack(rows):
        seen = [r for r in rows if not r.get("late")]
        return dict(contacts=[_r(r["contact"]) for r in seen],
                    top=[bool(r["timing"] is not None and r["timing"] <= TOP) for r in seen],
                    paths=[_r(r["path"][::2]) for r in rows if r["path"] is not None][:max_paths],
                    arcs=[_r(r["arc"][::3]) for r in seen if r.get("arc") is not None][:max_paths],
                    feet=[_r(f) for r in seen if r["feet"] is not None for f in r["feet"]])
    pro_strokes = []
    for side in ("forehand", "backhand"):                              # the pros' typical stroke of each side, for the ghost
        b = body_summary(pro.get("strokes") or [], side)
        if b and b["typ"] is not None:
            pro_strokes.append(dict(_stroke_json(b["typ"]), med=b["m"], n=b["n"]))
    def strokes_of(p):
        ss = sorted(p.get("strokes") or [], key=lambda s: (s["point"], s["shot"]))
        typ = (p.get("body") or {}).get("typ")
        k = next((i for i, s in enumerate(ss) if typ is not None and (s["point"], s["shot"]) == (typ["point"], typ["shot"])), 0)
        return [_stroke_json(s) for s in ss], k
    out = []
    for p in players:
        ss, k = strokes_of(p)
        out.append(dict(name=p["name"], key=p["key"], strokes=ss, typ=k, body=None if not p.get("body") else dict(med=p["body"]["m"], n=p["body"]["n"]),
                        **pack(p["rows"])))
    return dict(players=out,
                pro=dict(pack(pro["rows"]), strokes=pro_strokes), table=dict(L=L, W=W, net_x=NET_X, net_h=NET_H, floor=FLOOR))


def _plain(m):
    """Metrics for JSON: tuples to lists, arrays dropped (they are only for drawing)."""
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in m.items() if not isinstance(v, (np.ndarray, list)) or k in ()}


def build(shots, names, out_dir, themes=("light", "dark"), size=(1600, 900), strokes=None):
    """Plates for each player (one file per view and theme) in out_dir, and what the report needs to show them."""
    out_dir = pathlib.Path(out_dir)
    players, pro = analyse(shots, names, strokes)
    res = dict(players=[], pro=dict(m=_plain(pro["m"]) if pro["m"] else None, source=pro["source"], licence=pro["licence"],
                                    n=len(pro["rows"])))
    for i, p in enumerate(players):
        plates = {}
        for view, fn in PLATES.items():
            for th in themes:
                img = fn(p, pro, theme=th, size=size)
                if img is None:
                    break
                f = f"a3d_{i + 1}_{view}_{th}.jpg"
                cv2.imwrite(str(out_dir / f), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                plates.setdefault(view, {})[th] = f
        res["players"].append(dict(name=p["name"], key=p["key"], n=len(p["rows"]), m=_plain(p["m"]), advice=p["advice"], plates=plates))
    res["scene"] = viewer_scene(players, pro)
    (out_dir / "after_match.json").write_text(json.dumps(dict(players=[{k: v for k, v in q.items() if k != "plates"} for q in res["players"]],
                                                              pro=res["pro"]), indent=1, default=float))
    return res, players, pro
