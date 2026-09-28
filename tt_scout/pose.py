"""The two players' skeletons and the posture read off them. ; product layer, not part of the evaluated method.

Skeletons come from tools/pose (Apple's Vision body pose: 19 joints in 2D, on-device, nothing downloaded), run on the left and right
halves of each frame. Every person found is given to a player only if their lowest joint reads as standing at THIS table (the same
zone the name balloons use, so the people at a table further back are never measured) and by which side of the net line they are on;
per side the tallest such person is the player.

Measured at each of a player's hits (the racket contact the event layer finds, nearest skeleton within 0.1 s):
  knee  = the more bent knee, hip-knee-ankle angle in degrees (180 = straight leg);
  lean  = how far the trunk (hip centre to neck) leans from upright, in degrees, positive = towards the table.
Both are angles IN THE PICTURE: from a side camera a player at the end of the table is seen roughly side-on, the view knee bend and
forward lean are usually judged from, but they are not 3D joint angles, and a player turned towards the camera reads straighter.
"""
import csv, math
import numpy as np
from .config import TABLE_LENGTH as L, TABLE_WIDTH as W, NET_X

JOINTS = ["nose", "leye", "reye", "lear", "rear", "neck", "lsho", "rsho", "lelb", "relb", "lwri", "rwri", "root", "lhip", "rhip",
          "lkne", "rkne", "lank", "rank"]
J = {n: i for i, n in enumerate(JOINTS)}
BONES = [("neck", "nose"), ("neck", "lsho"), ("neck", "rsho"), ("lsho", "lelb"), ("lelb", "lwri"), ("rsho", "relb"), ("relb", "rwri"),
         ("neck", "root"), ("root", "lhip"), ("root", "rhip"), ("lhip", "lkne"), ("lkne", "lank"), ("rhip", "rkne"), ("rkne", "rank")]
FACE = {"nose", "leye", "reye", "lear", "rear"}
DRAW_BONES = [b for b in BONES if not (set(b) & FACE)]              # what is drawn on the picture: nothing on the face (2026-09-26)
MIN_C = 0.25            # joints Vision is less sure of than this are not drawn or measured
FILL_GAP = 8            # frames: a player lost for at most this long is interpolated between the two skeletons around the gap
HIT_WINDOW_S = 0.10


def load(path):
    """{frame: [array (19, 3) of x, y, confidence], ...} from tools/pose's CSV."""
    out = {}
    for r in csv.DictReader(open(path)):
        a = np.zeros((len(JOINTS), 3))
        for i, n in enumerate(JOINTS):
            c = float(r[f"{n}_c"] or 0)
            if c > 0:
                a[i] = (float(r[f"{n}_x"]), float(r[f"{n}_y"]), c)
        out.setdefault(int(r["frame"]), []).append(a)
    return out


def _side_sign(table):
    """Signed distance to the net line in the picture, and the sign the near (x = 0) end has."""
    n1, n2 = table.to_px([[NET_X, 0.0], [NET_X, W]])
    def d(x, y):
        return ((x - n1[0]) * (n2[1] - n1[1]) - (y - n1[1]) * (n2[0] - n1[0])) / float(np.hypot(*(n2 - n1)))
    near = table.to_px([[0.0, W / 2]])[0]
    return d, np.sign(d(*near))


SIZE_GATE = 0.72            # a person whose torso is under this share of the player's usual torso length is someone further back
                            # (our 2026-09-25 recording: a second game behind the table put a player's skeleton on a stranger whose feet read
                            # right at the table's far edge; the players' torsos run 120 to 145 px, the people behind 60 to 80)


CONT_FRAMES = 30          # a player seen within this many frames is followed rather than chosen afresh
JUMP_PX, JUMP_PER_FRAME = 160.0, 12.0   # how far (px) a player may be from where he was: a lunge is ~12 px a frame at 60 fps; a
                          # passer-by picked because he is bigger (nearer the camera) is hundreds of px away (our 2026-09-26 recording,
                          # point 72: the near skeleton jumped to someone walking along the left edge in 7% of the rally's frames)


def _centre(a):
    ok = a[:, 2] >= MIN_C
    return a[ok, :2].mean(axis=0) if ok.any() else a[:, :2].mean(axis=0)


def assign(poses, table, standing):
    """{frame: {"near": array or None, "far": array or None}} with short gaps filled. standing = heads.standing_zone(table, 1.0).
    Per side the player is the biggest person standing at this table (torso length: the nearest to the camera), and nobody much
    smaller than that player usually is."""
    d, near_sign = _side_sign(table)
    cands = {}
    for f, people in poses.items():
        for a in people:
            ok = a[:, 2] >= MIN_C
            if ok.sum() < 6:
                continue
            legs = [J[n] for n in ("lank", "rank", "lkne", "rkne", "lhip", "rhip") if a[J[n], 2] >= MIN_C]
            if not legs:
                continue
            k = max(legs, key=lambda i: a[i, 1])                     # the lowest leg joint stands in for the feet
            if not standing(a[k, 0], a[k, 1]):
                continue
            side = "near" if np.sign(d(a[k, 0], a[k, 1])) == near_sign else "far"
            size = _torso(a) or 0.6 * (a[ok, 1].max() - a[ok, 1].min()) / 2.4     # no neck or hips: guess from the height
            cands.setdefault(f, []).append((side, size, a))
    usual = {}
    for side in ("near", "far"):                                     # each player's usual size: the median of the biggest person per frame
        per = [max(sz for sd, sz, _ in cs if sd == side) for cs in cands.values() if any(sd == side for sd, _, _ in cs)]
        usual[side] = float(np.median(per)) if per else 0.0
    raw = {}
    last = {"near": None, "far": None}                               # (frame, centre) where each side's player just was
    for f in sorted(cands):
        best = {}
        for side in ("near", "far"):
            opts = [(size, a) for sd, size, a in cands[f] if sd == side and size >= SIZE_GATE * usual[side]]
            if not opts:
                continue
            pick = max(opts, key=lambda o: o[0])                     # the biggest person standing at this end ...
            lf = last[side]
            if lf is not None and f - lf[0] <= CONT_FRAMES:          # ... unless that means a jump across the picture: players move
                reach = JUMP_PX + JUMP_PER_FRAME * (f - lf[0])           # a few pixels a frame, a passer-by nearer the camera is elsewhere
                nearest = min(opts, key=lambda o: float(np.hypot(*(_centre(o[1]) - lf[1]))))
                if float(np.hypot(*(_centre(nearest[1]) - lf[1]))) <= reach:
                    pick = nearest
                else:
                    continue                                          # nobody where the player was: this frame is left empty
            best[side] = pick
            last[side] = (f, _centre(pick[1]))
        raw[f] = {s: v[1] for s, v in best.items()}
    out = {f: {"near": v.get("near"), "far": v.get("far")} for f, v in raw.items()}
    frames = sorted(raw)
    for side in ("near", "far"):                                     # bridge short gaps joint by joint
        have = [f for f in frames if raw[f].get(side) is not None]
        for fa, fb in zip(have, have[1:]):
            if 1 < fb - fa <= FILL_GAP + 1:
                A, B = raw[fa][side], raw[fb][side]
                for f in range(fa + 1, fb):
                    u = (f - fa) / (fb - fa)
                    m = np.zeros_like(A); both = (A[:, 2] >= MIN_C) & (B[:, 2] >= MIN_C)
                    m[both, :2] = A[both, :2] * (1 - u) + B[both, :2] * u
                    m[both, 2] = np.minimum(A[both, 2], B[both, 2])
                    out.setdefault(f, {"near": None, "far": None})[side] = m
    return out


def _pt(a, n):
    return a[J[n], :2] if a[J[n], 2] >= MIN_C else None


def angle(a, b, c):
    """Angle at b in degrees."""
    v1, v2 = a - b, c - b
    den = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    return None if den < 1e-6 else math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / den, -1, 1))))


def knees(a):
    """{'l': angle, 'r': angle} for the knees whose hip, knee and ankle are all found."""
    out = {}
    for s in ("l", "r"):
        h, k, an = _pt(a, s + "hip"), _pt(a, s + "kne"), _pt(a, s + "ank")
        if h is not None and k is not None and an is not None:
            out[s] = angle(h, k, an)
    return out


def lean(a, towards_x):
    """Trunk lean from upright in degrees, positive towards towards_x (+1 = right in the picture). None without hips and neck."""
    neck = _pt(a, "neck"); hips = [p for p in (_pt(a, "lhip"), _pt(a, "rhip"), _pt(a, "root")) if p is not None]
    if neck is None or not hips:
        return None
    base = np.mean(hips, axis=0); v = neck - base
    if v[1] >= 0:                                                    # neck below the hips: not a standing trunk
        return None
    return math.degrees(math.atan2(v[0] * towards_x, -v[1]))


def measures(a, side):
    kn = knees(a)
    return dict(knee=min(kn.values()) if kn else None, lean=lean(a, +1 if side == "near" else -1))


def at_hits(events, assigned, fps):
    """[{t, side, knee, lean}] for each racket hit the event layer found, from the hitter's skeleton nearest in time (within 0.1 s)."""
    out = []
    w = int(round(HIT_WINDOW_S * fps))
    for e in events:
        if e["kind"] != "hit":
            continue
        f0 = int(round(e["t"] * fps)); side = e["side"]
        for df in sorted(range(-w, w + 1), key=abs):
            a = (assigned.get(f0 + df) or {}).get(side)
            if a is None:
                continue
            m = measures(a, side)
            if m["knee"] is not None or m["lean"] is not None:
                out.append(dict(t=e["t"], side=side, **m)); break
    return out


def hitter_name(h, points, names):
    """The name of the player who made hit h: the point's own naming (it follows changes of ends), else the fixed names by end."""
    p = next((p for p in points if p["start_t"] - 0.5 <= h["t"] <= p["end_t"] + 0.5), None)
    if p and p.get("near_player") and p.get("far_player"):
        return p["near_player"] if h["side"] == "near" else p["far_player"]
    return names[h["side"]]


def summary(hits, points, names):
    """Per player: hits measured, median knee bend and lean at contact, and the same on points they won and lost."""
    res = {}
    for name in dict.fromkeys([names["near"], names["far"]]):
        hs = [h for h in hits if hitter_name(h, points, names) == name]
        def med(xs):
            xs = [x for x in xs if x is not None]
            return (float(np.median(xs)), len(xs)) if xs else (None, 0)
        won, lost = [], []
        for h in hs:
            p = next((p for p in points if p["start_t"] <= h["t"] <= p["end_t"]), None)
            if p and p.get("winner_name"):
                (won if p["winner_name"] == name else lost).append(h)
        res[name] = dict(n=len(hs), knee=med([h["knee"] for h in hs]), lean=med([h["lean"] for h in hs]),
                         knee_won=med([h["knee"] for h in won]), knee_lost=med([h["knee"] for h in lost]))
    return res


def skeleton_near(assigned, t, side, fps, reach=6):
    """The player's skeleton in the frame nearest time t (within reach frames), or None."""
    f0 = int(round(t * fps))
    for d in sorted(range(-reach, reach + 1), key=abs):
        a = (assigned.get(f0 + d) or {}).get(side)
        if a is not None:
            return a
    return None


def typical_stance(hits, assigned, fps, side, min_n=5):
    """The player's typical body shape at contact: each skeleton at a hit moved to the hip centre, scaled to a torso (hips to neck) of
    1 and turned to face the table (+x), then the median of every joint. {joint: (x, y)} with y pointing down, or None with fewer than
    min_n skeletons. Forehands and backhands are mixed, so the arms are an average and mean little; legs and trunk are the point."""
    arr = []
    for h in hits:
        if h["side"] != side:
            continue
        a = skeleton_near(assigned, h["t"], side, fps)
        if a is None:
            continue
        hips = [a[J[n], :2] for n in ("lhip", "rhip") if a[J[n], 2] >= MIN_C] or ([a[J["root"], :2]] if a[J["root"], 2] >= MIN_C else [])
        if not hips or a[J["neck"], 2] < MIN_C:
            continue
        hc = np.mean(hips, axis=0); scale = float(np.linalg.norm(a[J["neck"], :2] - hc))
        if scale < 10:
            continue
        facing = 1.0 if side == "near" else -1.0
        n = np.full((len(JOINTS), 2), np.nan); ok = a[:, 2] >= MIN_C
        n[ok, 0] = (a[ok, 0] - hc[0]) / scale * facing
        n[ok, 1] = (a[ok, 1] - hc[1]) / scale
        arr.append(n)
    if len(arr) < min_n:
        return None
    with np.errstate(all="ignore"):
        med = np.nanmedian(np.stack(arr), axis=0)
    return {JOINTS[i]: (round(float(med[i, 0]), 3), round(float(med[i, 1]), 3)) for i in range(len(JOINTS)) if not np.isnan(med[i, 0])}


# ---- leg clean-up (2026-09-26: the skeletons were inaccurate when the legs were covered by the table or one leg
# hid the other). From one side camera a hidden leg is not seen, but Vision still returns joints for it: they land on the
# table's legs, swap left for right between frames, or fold onto the other leg. These checks leave such a joint out (confidence 0), so
# it is neither drawn nor measured: a missing shin is better than a wrong one.
LEG_MIN_C = 0.35            # knees and ankles need more confidence than other joints
TABLE_MIN_C = 0.6           # ... and over the table's outline, more again: joints there have median confidence 0.36 to 0.56 against
                            # 0.72 to 0.77 elsewhere (our two recordings); a leg in front of the table's end is seen clearly and passes,
                            # one behind it (Vision's guess) does not. A flat "over the table = hidden" rule removed visible legs.
BONE_TOL = (0.6, 1.6)       # a thigh or shin this far off the player's own median length (in torso lengths) is not his leg
JUMP_TORSO = 0.8            # a knee or ankle that moves more than this many torso lengths in one frame jumped to something else


def table_footprint(table, height_m=0.76):
    """Where the table covers the picture: its top and the frame and legs under it down to the floor (the top's corners dropped by the
    table's height at the local scale). A leg joint in here is behind or under the table, or Vision put it on the table's own legs."""
    import cv2
    top = [(0.0, 0.0), (0.0, W), (L, W), (L, 0.0)]
    pts = [tuple(c) for c in table.corners]
    pts += [(x, y + height_m * table.ppm_at(tp) * 0.9) for (x, y), tp in zip(table.corners, top)]
    return cv2.convexHull(np.float32(pts))


def _torso(a):
    hips = [a[J[n], :2] for n in ("lhip", "rhip") if a[J[n], 2] >= MIN_C] or ([a[J["root"], :2]] if a[J["root"], 2] >= MIN_C else [])
    if not hips or a[J["neck"], 2] < MIN_C:
        return None
    v = float(np.linalg.norm(a[J["neck"], :2] - np.mean(hips, axis=0)))
    return v if v > 8 else None


def _thighs_cross(a):
    """Do the segments left hip to left knee and right hip to right knee cross in the picture?"""
    idx = [J[n] for n in ("lhip", "lkne", "rhip", "rkne")]
    if (a[idx, 2] < MIN_C).any():
        return False
    p1, p2, p3, p4 = (a[i, :2] for i in idx)
    def orient(p, q, r):
        return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    return orient(p1, p2, p3) != orient(p1, p2, p4) and orient(p3, p4, p1) != orient(p3, p4, p2)


def clean_legs(assigned, table, log=None):
    """Knees and ankles the picture cannot back up are set to confidence 0 in place; left and right legs swapped by Vision between
    frames are swapped back. Returns (assigned, dict of how many joints each check removed)."""
    import cv2
    hull = table_footprint(table)
    inside = lambda p: cv2.pointPolygonTest(hull, (float(p[0]), float(p[1])), False) >= 0
    d = lambda p, q: float(np.linalg.norm(p[:2] - q[:2]))
    counts = dict(low_confidence=0, behind_table=0, bone_length=0, jump=0, swapped=0)
    frames = sorted(assigned)
    for side in ("near", "far"):
        ratio = {"thigh": [], "shin": []}                                # the player's own leg proportions, from his clear frames
        for f in frames:
            a = (assigned.get(f) or {}).get(side); t = None if a is None else _torso(a)
            if not t:
                continue
            for s in ("l", "r"):
                h, k, an = a[J[s + "hip"]], a[J[s + "kne"]], a[J[s + "ank"]]
                if h[2] >= MIN_C and k[2] >= LEG_MIN_C and not inside(k):
                    ratio["thigh"].append(d(h, k) / t)
                    if an[2] >= LEG_MIN_C and not inside(an):
                        ratio["shin"].append(d(k, an) / t)
        med = {k: (float(np.median(v)) if len(v) >= 20 else None) for k, v in ratio.items()}
        prev, prev_f = None, None
        for f in frames:
            a = (assigned.get(f) or {}).get(side)
            if a is None:
                prev = None; continue
            a = a.copy(); t = _torso(a) or (_torso(prev) if prev is not None else None)
            if _thighs_cross(a):                                         # the thighs form an X: Vision gave each hip the other leg
                for j in ("kne", "ank"):
                    a[[J["l" + j], J["r" + j]]] = a[[J["r" + j], J["l" + j]]]
                counts["swapped"] += 1
                if log is not None: log.append((f, side, "legs", "left/right swapped back"))
            for s in ("l", "r"):
                for j in ("kne", "ank"):
                    p = a[J[s + j]]
                    if p[2] <= 0:
                        continue
                    if p[2] < LEG_MIN_C:
                        a[J[s + j], 2] = 0; counts["low_confidence"] += 1
                        if log is not None: log.append((f, side, s + j, "low confidence"))
                    elif p[2] < TABLE_MIN_C and inside(p):
                        a[J[s + j], 2] = 0; counts["behind_table"] += 1
                        if log is not None: log.append((f, side, s + j, "over the table, unsure"))
                h, k, an = a[J[s + "hip"]], a[J[s + "kne"]], a[J[s + "ank"]]
                if t and med["thigh"] and h[2] >= MIN_C and k[2] > 0 and not (BONE_TOL[0] <= d(h, k) / t / med["thigh"] <= BONE_TOL[1]):
                    a[J[s + "kne"], 2] = 0; counts["bone_length"] += 1
                    if log is not None: log.append((f, side, s + "kne", "thigh length"))
                if t and med["shin"] and a[J[s + "kne"], 2] > 0 and an[2] > 0 and not (BONE_TOL[0] <= d(k, an) / t / med["shin"] <= BONE_TOL[1]):
                    a[J[s + "ank"], 2] = 0; counts["bone_length"] += 1
                    if log is not None: log.append((f, side, s + "ank", "shin length"))
                if prev is not None and t and f - prev_f <= 2:
                    for j in ("kne", "ank"):
                        p, q = a[J[s + j]], prev[J[s + j]]
                        if p[2] > 0 and q[2] > 0 and d(p, q) > JUMP_TORSO * t:
                            a[J[s + j], 2] = 0; counts["jump"] += 1
                            if log is not None: log.append((f, side, s + j, "jumped"))
                if a[J[s + "kne"], 2] <= 0:                              # no knee: its ankle has nothing to hang from
                    a[J[s + "ank"], 2] = 0
            assigned[f][side] = a
            prev, prev_f = a, f
    return assigned, counts


class Others:
    """What annotate.blur_people needs for one frame: .get(frame) -> dict(hide=[box], zone=[polygon], keep=[skeleton or box]) or None;
    see others()."""

    def __init__(self, seen, keep, k, zone=None, balls=None, fixed_keep=None, k_keep=60):
        self.seen, self.keep, self.k, self.zone, self.balls = seen, keep, k, zone, balls
        self.fixed_keep, self.k_keep = fixed_keep or [], k_keep
        self.frames = sorted(seen)
        self.kframes = {sd: sorted(f for f, v in keep.items() if v.get(sd) is not None) for sd in ("near", "far")}

    def __bool__(self):
        return bool(self.frames) or bool(self.zone)

    def __len__(self):
        return len(self.frames)

    def _held(self, sd, f):
        import bisect
        fr = self.kframes[sd]
        i = bisect.bisect_left(fr, f)
        near = [g for g in fr[max(0, i - 1):i + 1] if abs(g - f) <= self.k_keep]
        return self.keep[min(near, key=lambda g: abs(g - f))][sd] if near else None

    def get(self, f):
        import bisect
        lo, hi = bisect.bisect_left(self.frames, f - self.k), bisect.bisect_right(self.frames, f + self.k)
        hide = [b for g in self.frames[lo:hi] for b in self.seen[g]]
        if not hide and not self.zone:
            return None
        keep = [q for q in (self._held("near", f), self._held("far", f)) if q is not None] + list(self.fixed_keep)
        if self.balls is not None and 0 <= f < len(self.balls) and self.balls[f] is not None:
            x, y = self.balls[f]
            keep.append((x - 24, y - 24, x + 24, y + 24))
        return dict(hide=hide, zone=self.zone or [], keep=keep)


def others(raw, assigned, fps, hold_s=3.0, min_c=0.2, zone=None, balls=None, fixed_keep=None):
    """For blurring people who never agreed to be filmed. Everyone Vision found who is not one of the two players (raw = load(),
    assigned = assign()) gives a box padded for the head, arms and racket; a frame hides every such box seen within hold_s either side
    of it, because Vision loses people at the back for a second or more at a time (on our 2026-09-26 recording, at 943 s it found only
    the two players with two others plainly in view). zone = polygons blurred in every frame whatever Vision finds (the back of the
    hall: see back_zone()). The two players stay sharp along their own outline (their skeletons, thickened), each held for up to a
    second where Vision loses them, so a stranger right beside or behind one of them still blurs and the player never does; balls =
    the tracked ball per frame ((x, y) or None) and fixed_keep = boxes (the net) stay sharp too. Returns an Others."""
    def box(a, px=0.35, top=0.25, bottom=0.10, extra=15):
        ok = a[:, 2] >= min_c; xs, ys = a[ok, 0], a[ok, 1]
        w, h = max(xs.max() - xs.min(), 30.0), max(ys.max() - ys.min(), 60.0)
        return (xs.min() - px * w - extra, ys.min() - top * h - extra, xs.max() + px * w + extra, ys.max() + bottom * h + extra)

    def same(a, b):
        ok = (a[:, 2] >= min_c) & (b[:, 2] >= min_c)
        return ok.sum() >= 3 and float(np.median(np.hypot(*(a[ok, :2] - b[ok, :2]).T))) < 6.0
    seen, keep = {}, {}
    for f, people in raw.items():
        sides = {}
        for sd in ("near", "far"):
            q = (assigned.get(f) or {}).get(sd)
            if q is not None and (q[:, 2] >= min_c).sum() >= 3:
                sides[sd] = q
        keep[f] = sides
        for a in people:
            if (a[:, 2] >= min_c).sum() >= 3 and not any(same(a, q) for q in sides.values()):
                seen.setdefault(f, []).append(box(a))
    return Others(seen, keep, int(round(hold_s * fps)), zone=zone, balls=balls, fixed_keep=fixed_keep, k_keep=int(round(1.0 * fps)))


def back_zone(table, width, height, margin_px=6):
    """The back of the hall in the picture: everything above the table's far long edge, extended to both sides of the frame (where
    other games and people waiting stand). Returns [polygon] in the recording's pixels."""
    (x0, y0), (x1, y1) = table.to_px([[0.0, W], [L, W]])
    slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.0
    yl, yr = y0 + slope * (0 - x0) - margin_px, y0 + slope * (width - x0) - margin_px
    return [np.array([[0, 0], [width, 0], [width, yr], [0, yl]], np.int32)]
