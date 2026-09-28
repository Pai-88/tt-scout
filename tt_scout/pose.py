"""The two players' skeletons and the posture read off them. ; product layer, not part of the evaluated method.

Skeletons come from tools/pose (Apple's Vision body pose: 19 joints in 2D, on-device, nothing downloaded), run on the left and right
halves of each frame. Every person found is given to a player only if their lowest joint reads as standing at THIS table (the same
zone the name balloons use, so the people at a table further back are never measured) and by which side of the net line they are on;
per side the tallest such person is the player.

Measured at each of a player's hits (the racket contact the event layer finds, nearest skeleton within 0.1 s):
  knee  = hip-knee-ankle angle in degrees (180 = straight leg) of the leg nearer the camera, see near_leg();
  lean  = how far the trunk (hip centre to neck) leans from upright, in degrees, positive = towards the table.
Both are angles IN THE PICTURE: from a side camera a player at the end of the table is seen roughly side-on, the view knee bend and
forward lean are usually judged from, but they are not 3D joint angles, and a player turned towards the camera reads straighter.

THE KNEE BEND AT CONTACT IS ONE QUANTITY EVERYWHERE (2026-09-28): the picture angle above, on rally forehands, from the hitter's own
end, with BOTH legs found, on the camera-near leg, and only at contacts whose legs the camera sees in profile (the gate in
technique.gate_knees: PROFILE_MIN). It is reported as a median with its count and the words KNEE_LABEL, or NOT_MEASURABLE under
KNEE_MIN_N contacts. Why not the 3D knee: Apple's 3D body pose returns a fixed 1.8 m template and only chooses joint rotations, so a leg
whose flexion plane is edge-on to the camera is explained by folding the knee and lifting the foot along the viewing ray: on our
recordings the 3D knee read 28 deg more bent than the picture on edge-on legs (n=213) and 4 deg straighter on legs seen in profile
(n=272). The pros carry the same bias, so their reference (benchmarks.json) is measured with the identical gate and leg rule.
"""
import bisect, csv, math
import numpy as np
from .config import TABLE_LENGTH as L, TABLE_WIDTH as W, NET_X

PROFILE_MIN = 60.0          # deg: a contact counts when the pelvis line is within 30 deg of the viewing ray (legs seen in profile).
                            # Calibrated on 338 forehands with a 3D body (8 recordings, 2026-09-28): at 60 the 3D and the 2D angle of the
                            # camera-near leg agree to a median -2 deg (n=133, IQR -9..5); ungated they differ by -11 (n=338)
HIP_RATIO_FACING = 0.46     # 2D hip width / torso length when the player squarely faces the camera: the profile from the picture alone
                            # (profile_2d) is arccos(ratio / this); fitted on the same 338 forehands (ratio = 0.46 cos(3D profile), agrees
                            # with the 3D gate on 82% of them)
KNEE_MIN_N = 3              # fewer gated contacts than this and the knee is NOT_MEASURABLE (the critique and the contact tiles use the same)
LEG_RATIO_MIN = 1.1         # without a 3D body the two legs' picture spans must differ by this factor for the longer one to be called the
                            # camera-near leg (with a body the placed knees decide, technique.gate_knees). Calibrated on the same 338
                            # forehands: the longer leg is the leg nearer the camera in 3D on 53% under a ratio of 1.05 (a coin flip), 66%
                            # at 1.05 to 1.1, and 88 to 91% from 1.1 up
KNEE_LABEL = "as the camera sees it"
NOT_MEASURABLE = "not measurable from this camera"
PAIR_S = 0.25               # s: a racket hit belongs to the rally shot nearest in time within this

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


def leg_spans(a):
    """{'l': px, 'r': px}: each leg's hip-to-ankle span in the picture, or None unless BOTH legs are fully found (knees())."""
    if len(knees(a)) < 2:
        return None
    return {s: float(np.hypot(*(a[J[s + "hip"], :2] - a[J[s + "ank"], :2]))) for s in ("l", "r")}


def near_leg(a):
    """(leg 'l' or 'r', its hip-knee-ankle angle) for the leg nearer the camera, or (None, None) unless BOTH legs are fully found.
    The camera-near leg is the one whose hip-to-ankle span in the picture is the longer: it is nearer (so larger) and it is the leg the
    camera actually sees, where the far leg is partly hidden behind it and Vision guesses its joints (the guessed leg is what read 20 to
    35 deg wrong on our recordings). On 338 forehands with a 3D body this picked the leg whose knee is nearer the camera in 3D 83% of
    the time (a much straighter far leg projects longer and wins), and on the gated contacts the 20% where it did not were where the 3D
    and the 2D angle disagreed most (9 of 27 by over 20 deg, against 7 of 106), so technique.gate_knees refuses a contact whose placed 3D
    body says the other leg is nearer, and without a body one whose spans differ by less than LEG_RATIO_MIN. Never the min of whatever
    legs were found: a lone guessed leg was 21 deg more bent than the seen one (n=119)."""
    span = leg_spans(a)
    if span is None:
        return None, None
    leg = max(span, key=span.get)
    return leg, knees(a)[leg]


def profile_2d(a):
    """How far the pelvis line is from the viewing ray, from the picture alone: degrees, 90 = seen side-on (legs in profile), 0 = the
    player squarely faces the camera. The hips read HIP_RATIO_FACING torso lengths wide when facing and narrow towards side-on, so
    the angle is arccos(width / torso / HIP_RATIO_FACING). None without both hips and the neck. Used only for a contact with no 3D
    body (technique.gate_knees); with one, body3d.pelvis_profile measures the same angle properly."""
    lh, rh, nk = _pt(a, "lhip"), _pt(a, "rhip"), _pt(a, "neck")
    if lh is None or rh is None or nk is None:
        return None
    t = float(np.linalg.norm(nk - (lh + rh) / 2))
    if t < 8:
        return None
    return math.degrees(math.acos(min(1.0, float(np.linalg.norm(lh - rh)) / t / HIP_RATIO_FACING)))


def measures(a, side):
    """knee = the camera-near leg's angle (None unless both legs are found) and which leg; leg_ratio = the longer span over the shorter
    (how clearly that leg is the nearer one, see LEG_RATIO_MIN); lean; profile2d (see profile_2d)."""
    leg, knee = near_leg(a); span = leg_spans(a)
    ratio = None if span is None else round(max(span.values()) / max(min(span.values()), 1e-6), 3)
    return dict(knee=knee, leg=leg, leg_ratio=ratio, lean=lean(a, +1 if side == "near" else -1), profile2d=profile_2d(a))


def stroke_end(p, t, shots=None):
    """The end the stroke at time t in point p was hit from: the point's serve order, as technique.measure counts it (stroke k = 1 +
    the net crossings, implied ones too, before t; k odd is the server's). From the measured shot (p, k) when it is among shots, else
    from the point's own naming; None when the point is unnamed."""
    k = 1 + sum(1 for c in p.get("events", []) if c.get("kind") == "net" and c["t"] < t)
    sh = next((s for s in (shots or []) if s["point"] == p["id"] and s["shot"] == k), None)
    if sh is not None:
        return sh["end"]
    if not (p.get("server") and p.get("near_player") and p.get("far_player")):
        return None
    serve_end = "near" if p.get("server") == p.get("near_player") else "far"
    return serve_end if k % 2 == 1 else ("far" if serve_end == "near" else "near")


def at_hits(events, assigned, fps, shots=None, points=None, pair_s=PAIR_S):
    """[{t, side, knee, leg, lean, point, shot}] for each racket hit the event layer found, from the hitter's skeleton nearest in time
    (within 0.1 s). With shots (technique.measure after gate_knees) a hit belongs to the rally shot nearest in time within pair_s: its
    side is that shot's hitter end, and its knee is that shot's gated knee (None where not measurable). The event layer sets a hit's
    side from where the ball was in the table's frame at the kink, which put a hit at the wrong end for 15 of 33 hits on our first
    recording (2026-09-28), so the skeleton measured was the other player's; the shot's hitter end comes from the point's serve order.
    A hit with no shot near it (a stroke whose flight could not be fitted has no contact time, 7 of 17 hits on that recording) takes
    its end from the same serve order through the points (stroke_end) and has no knee; with no points to place it in, it is not
    measured at all (left out), never read from the ball's side. Without shots (no video, so nothing was measured at contact) a hit
    keeps the ball's side, only its lean is read, and there is no knee: forehands cannot be told from backhands there."""
    out = []
    w = int(round(HIT_WINDOW_S * fps))
    ts = sorted(shots or [], key=lambda s: s["t"]); tt = [s["t"] for s in ts]
    for e in events:
        if e["kind"] != "hit":
            continue
        side, sh = e["side"], None
        if shots is not None:
            i = bisect.bisect_left(tt, e["t"])
            near = [ts[j] for j in (i - 1, i) if 0 <= j < len(ts) and abs(tt[j] - e["t"]) <= pair_s]
            sh = min(near, key=lambda s: abs(s["t"] - e["t"])) if near else None
            if sh is not None:
                side = sh["end"]
            else:
                p = next((p for p in (points or []) if p["start_t"] - 0.5 <= e["t"] <= p["end_t"] + 0.5), None)
                side = stroke_end(p, e["t"], shots) if p is not None else None
                if side is None:
                    continue
        f0 = int(round(e["t"] * fps))
        for df in sorted(range(-w, w + 1), key=abs):
            a = (assigned.get(f0 + df) or {}).get(side)
            if a is None:
                continue
            m = measures(a, side)
            m["knee"] = None if sh is None else sh.get("knee"); m["leg"] = None if sh is None else sh.get("leg")   # the shot's gated knee, never the hit's own
            m.pop("profile2d", None); m.pop("leg_ratio", None)
            if m["knee"] is not None or m["lean"] is not None:
                out.append(dict(t=e["t"], side=side, point=None if sh is None else sh["point"], shot=None if sh is None else sh["shot"], **m)); break
    return out


def knee_median(vals):
    """(median, n) of the gated knee angles; the median is None under KNEE_MIN_N contacts (then it is NOT_MEASURABLE)."""
    vals = [v for v in vals if v is not None]
    return (float(np.median(vals)) if len(vals) >= KNEE_MIN_N else None, len(vals))


def knee_text(med_n, what="contacts", why=None):
    """The knee as it is printed everywhere: '137 deg, median of 9 forehand contacts, as the camera sees it' or NOT_MEASURABLE, with
    the count that fell short or the reason (why) in brackets."""
    med, n = med_n
    if med is None:
        return NOT_MEASURABLE + (f" ({why})" if why else (f" ({n} forehand {what} seen in profile; {KNEE_MIN_N} are needed)" if n else ""))
    return f"{med:.0f}°, median of {n} forehand {what}, {KNEE_LABEL}"


def knee_contacts(shots, points, names):
    """{name: [dict(t, point, shot, side, knee, leg, src, won)]}: every gated contact (a rally forehand seen in profile, both legs found:
    technique.gate_knees leaves knee None on the rest), by the hitter's name, in time order, src = what judged the profile ('3d' the
    placed body, '2d' the picture's hip width). What the posture section, the contact tiles and the profiles all read, so they show
    one number."""
    out = {n: [] for n in dict.fromkeys([names["near"], names["far"]])}
    for s in sorted(shots, key=lambda s: s["t"]):
        if s.get("serve") or s.get("knee") is None or s.get("name") not in out:
            continue
        p = next((p for p in points if p["id"] == s["point"]), None)
        won = (p.get("winner_name") == s["name"]) if p and p.get("winner_name") else None
        out[s["name"]].append(dict(t=round(s["t"], 4), point=s["point"], shot=s["shot"], side=s["end"], knee=round(s["knee"], 2), leg=s.get("leg"),
                                   src=s.get("profile_src"), won=won))
    return out


def hitter_name(h, points, names):
    """The name of the player who made hit h: the point's own naming (it follows changes of ends), else the fixed names by end."""
    p = next((p for p in points if p["start_t"] - 0.5 <= h["t"] <= p["end_t"] + 0.5), None)
    if p and p.get("near_player") and p.get("far_player"):
        return p["near_player"] if h["side"] == "near" else p["far_player"]
    return names[h["side"]]


NO_SHOTS = "no video, so nothing was measured at contact"


def summary(hits, points, names, shots=None):
    """Per player: hits measured, median lean at contact from the hits (as the camera sees it), and the knee bend at contact as the ONE
    gated quantity (knee_contacts over the shots): (median, n) with the median None under KNEE_MIN_N contacts, the same on points they
    won and lost, and knee_src = how many of the gated contacts the 3D body judged and how many the picture alone. Without shots there
    is no knee at all: knee = (None, 0) and knee_why = NO_SHOTS, so every surface prints NOT_MEASURABLE with that reason."""
    res = {}
    kc = knee_contacts(shots, points, names) if shots is not None else None
    for name in dict.fromkeys([names["near"], names["far"]]):
        hs = [h for h in hits if hitter_name(h, points, names) == name]
        def med(xs):
            xs = [x for x in xs if x is not None]
            return (float(np.median(xs)), len(xs)) if xs else (None, 0)
        cs = kc.get(name, []) if kc is not None else []
        res[name] = dict(n=len(hs), knee=knee_median([c["knee"] for c in cs]), lean=med([h["lean"] for h in hs]),
                         knee_won=knee_median([c["knee"] for c in cs if c["won"] is True]), knee_lost=knee_median([c["knee"] for c in cs if c["won"] is False]),
                         knee_src={k: sum(1 for c in cs if c.get("src") == k) for k in ("3d", "2d")}, knee_label=KNEE_LABEL,
                         knee_why=None if kc is not None else NO_SHOTS)
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
