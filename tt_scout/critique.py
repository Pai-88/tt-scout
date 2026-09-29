"""A coach's criticism of each player, from how he actually hit the ball, for casual players trying to improve their technique.

So: the same checks for both players, mostly about technique rather than about who won; each one needs only a handful of shots (an
11-point game gives each player 10 serves and 20 to 40 shots); every criticism states its counts, compares with the professional
OpenTTGames test matches measured the same way (benchmarks.json, analysis/pro_benchmarks.py) and gives the usual fix. Not a language
model: rules over what tt_scout measured (technique.py), so it can only say what the camera saw.

    notes = critique(shots, points, "Sam")            # [Critique], worst first
    by_point = critiques_by_point(shots, points)       # {point id: {name: [Critique]}} from the shots BEFORE each point
"""
import json, math, pathlib
from dataclasses import dataclass
import numpy as np
from .match_stats import players
from .pose import KNEE_MIN_N, KNEE_LABEL

BENCH, BENCH_CLIPS = {}, {}
try:
    _bj = json.loads((pathlib.Path(__file__).with_name("benchmarks.json")).read_text())
    BENCH, BENCH_CLIPS = _bj.get("pooled", {}), _bj.get("clips", {})
except Exception:
    pass


def _b(key, default):
    v = BENCH.get(key)
    return v[0] if isinstance(v, list) and v and v[0] is not None else default


PRO = dict(speed=_b("rally_speed", 10.8), dip=_b("dip_share", 0.75), over_net=_b("over_net", 11.2), timing=_b("timing", 0.0),
           knee=_b("knee", 145.0), lean=_b("lean", 15.0), stance=_b("stance", 0.9), depth=_b("depth", 0.94))
PRO_N = {k: (v[1] if isinstance(v, list) and len(v) > 1 else 0) for k, v in BENCH.items()}   # how many measurements each reference rests on
# THE KNEE REFERENCE IS THIN (2026-09-28): with the gate the pros' knee comes from a handful of forehand contacts (the pros turn their
# hips 40 to 60 deg into a forehand, which is exactly what the gate refuses; benchmarks.json knee_note says how many). So the margin
# a player's median must sit above the pros' to be called upright is the larger of 8 deg and half the width of the 90% bootstrap
# interval of the pros' pooled median (knee_ci in benchmarks.json, pro_benchmarks.py), the reference's own uncertainty, instead of
# the flat +8 that served the old ungated min-leg quantity. With 9 contacts that interval is about 20 deg wide, so the margin is 10.
_ci = BENCH.get("knee_ci") or [None, None]
KNEE_MARGIN = max(8.0, math.ceil((_ci[1] - _ci[0]) / 2)) if _ci[0] is not None and _ci[1] is not None else 8.0

# posture in 3D (body3d.py): trunk lean and shoulder turn measured the same way on the pros' strokes (pro_bodies.json). The lean in 2D
# depends on where the camera stands, so when a recording has 3D bodies the 3D lean replaces it; the shoulder turn exists only in 3D.
# The knee is NOT taken from the 3D bodies (pose.py, top): it is the picture angle, gated, against the identically gated PRO["knee"].
PRO3D = {}
try:
    _pb = json.loads((pathlib.Path(__file__).with_name("pro_bodies.json")).read_text())["strokes"]
    PRO3D = {k: float(np.median([s_["m"][k] for s_ in _pb if s_["m"].get(k) is not None])) for k in ("lean", "turn")}
except Exception:
    pass


@dataclass
class Critique:
    player: str
    head: str        # the fault, as a coach would say it
    evidence: str    # the counts behind it, and the professional reference
    fix: str         # the usual correction
    weight: float    # how far off, times how sure: the order they are shown in
    key: str = ""    # which check (for rotating through them without repeats)


def _conf(n, full=12):
    """How sure a check is on n cases: rises from 0.5 at 4 cases to 1 at `full`."""
    return min(1.0, 0.5 + 0.5 * max(0, n - 4) / max(1, full - 4))


def critique(shots, points, name, min_n=4, so_far=False):
    """Every criticism the shots support for one player, worst first. so_far: the shots are only those before a point (a clip's
    note), so a number like the fastest shot says "so far": a shot in the point being played can beat it."""
    mine = [s for s in shots if s.get("name") == name]
    rally = [s for s in mine if not s["serve"]]
    out = []

    def vals(key, ss=rally):
        return [s[key] for s in ss if s.get(key) is not None]

    # 1. topspin
    spins = vals("spin")
    if len(spins) >= min_n:
        dip = float(np.mean(np.array(spins) < -3)); k = int(round(dip * len(spins)))
        if dip < 0.5:
            out.append(Critique(name, "Hitting flat: not enough topspin", f"{k} of {len(spins)} rally shots dipped like topspin (pros {PRO['dip']:.0%})",
                                "Brush up the back of the ball, low to high",
                                (PRO["dip"] - dip) * 1.3 * _conf(len(spins)), "spin"))
    # 2. timing
    tm = vals("timing")
    if len(tm) >= min_n:
        med = float(np.median(tm)); late = sum(t > 0.08 for t in tm)
        if med >= 0.06:
            out.append(Critique(name, "Taking the ball late, as it drops", f"hits {med:.2f} s after the top of the bounce, {late} of {len(tm)} well past it (pros: at the top)",
                                "Move in early, meet the ball at the top of the bounce",
                                min(1.0, med / 0.15) * _conf(len(tm)), "timing"))
    # 3. height over the net
    on = vals("over_net")
    if len(on) >= min_n:
        med = float(np.median(on)); high = sum(v > 25 for v in on)
        if med >= PRO["over_net"] + 4 or high / len(on) >= 0.3:
            out.append(Critique(name, "Balls crossing high over the net", f"typically {med:.0f} cm over the net, {high} of {len(on)} over 25 cm (pros {PRO['over_net']:.0f} cm)",
                                "Close the bat angle, aim just over the net", min(1.0, (med - PRO["over_net"]) / 12 + high / len(on)) * _conf(len(on)), "high"))
    # 4. pace
    sp = vals("speed")
    if len(sp) >= min_n:
        med = float(np.median(sp)); top = max(sp)
        if med < 0.8 * PRO["speed"]:
            # km/h throughout, as the clips' speed gauge shows it (a note mixing m/s and km/h beside the gauge read as a contradiction)
            out.append(Critique(name, "Slow rally pace", f"typically {3.6 * med:.0f} km/h off the racket, fastest {3.6 * top:.0f}{' so far' if so_far else ''} (pros {3.6 * PRO['speed']:.0f})",
                                "Turn hips and shoulders, accelerate through the ball",
                                min(1.0, (PRO["speed"] - med) / PRO["speed"] * 1.4) * _conf(len(sp)), "pace"))
    # 5. knees: the ONE gated quantity (technique.gate_knees: the camera-near leg's picture angle on rally forehands seen in profile;
    #    None on every other shot), against the pros measured with the identical gate and leg rule; judged from pose.KNEE_MIN_N
    #    contacts, the same threshold that makes it measurable in the report and the profile
    kn = vals("knee")
    if len(kn) >= KNEE_MIN_N:
        med = float(np.median(kn)); pk = PRO["knee"]
        if med >= pk + KNEE_MARGIN:
            out.append(Critique(name, "Standing too upright at contact", f"knees {med:.0f} deg at contact, {KNEE_LABEL} (straight = 180; pros {pk:.0f} "
                                f"measured the same way, from {PRO_N.get('knee', 0)} contacts), {len(kn)} forehands seen in profile",
                                "Wider stance, knees bent, weight on the balls of the feet",
                                min(1.0, (med - pk) / 30) * _conf(len(kn)), "knees"))
    # 6. trunk (in 3D where the bodies were measured, else as the camera sees it; the evidence says which)
    three = PRO3D and len(vals("lean3d")) >= min_n
    lk, pl = ("lean3d", PRO3D.get("lean")) if three else ("lean", PRO["lean"])
    ln = vals(lk)
    if len(ln) >= min_n:
        med = float(np.median(ln))
        if med < pl - 7:
            what = "leaning back" if med < 0 else "upright"
            out.append(Critique(name, f"Trunk {what} at contact", f"trunk {abs(med):.0f} deg {'back' if med < 0 else 'forward'} (pros {pl:.0f} forward), "
                                f"{len(ln)} shots, {'in 3D' if lk == 'lean3d' else KNEE_LABEL}",
                                "Lean forward from the hips, head over the ball",
                                min(1.0, (pl - med) / 25) * _conf(len(ln)), "lean"))
    # 6b. shoulder turn through the stroke (3D only)
    tn = vals("turn3d")
    if PRO3D and len(tn) >= min_n:
        med = float(np.median(tn))
        if med < PRO3D["turn"] - 12:
            out.append(Critique(name, "Not turning the shoulders", f"shoulders turned {med:.0f} deg from backswing to contact (pros {PRO3D['turn']:.0f}), "
                                f"{len(tn)} strokes in 3D", "Turn back from the waist, then rotate through the ball",
                                min(1.0, (PRO3D["turn"] - med) / 30) * _conf(len(tn)), "turn"))
    # 7. depth
    dp = vals("depth")
    if len(dp) >= min_n:
        short = sum(d < 0.6 for d in dp); med = float(np.median(dp))
        if short / len(dp) >= 0.3 or med < PRO["depth"] - 0.15:
            out.append(Critique(name, "Landing short, mid-table", f"{short} of {len(dp)} landed within 60 cm of the net (pros land {PRO['depth']:.2f} m deep)",
                                "Aim for the last third of the table",
                                min(1.0, short / len(dp) + (PRO["depth"] - med)) * _conf(len(dp)), "depth"))
    # 8. errors
    if len(mine) >= 6:
        miss = sum(s["outcome"] == "missed" for s in mine)
        if miss >= 2 and miss / len(mine) >= 0.06:
            out.append(Critique(name, "Missing the table", f"{miss} of his {len(mine)} shots went long or wide ({100 * miss / len(mine):.0f}%)",
                                "Less flat power, more topspin, aim inside the lines", min(1.0, miss / len(mine) * 6) * _conf(len(mine), 30), "errors"))
    # 9. serve
    serves = [s for s in mine if s["serve"] and s.get("depth") is not None]
    if len(serves) >= min_n:
        cats = ["short" if s["depth"] < 0.5 else ("half-long" if s["depth"] < 1.0 else "long") for s in serves]
        top = max(set(cats), key=cats.count); k = cats.count(top)
        if k / len(cats) >= 0.75:
            out.append(Critique(name, "Predictable serve", f"{k} of {len(cats)} serves landed {top}",
                                "Mix short serves with fast long ones, vary the spin",
                                (k / len(cats) - 0.5) * 1.6 * _conf(len(cats)), "serve"))
    sv = [p for p in points if p.get("server") == name and p.get("winner_name")]
    if len(sv) >= min_n:
        won = sum(p["winner_name"] == name for p in sv)
        if won / len(sv) < 0.5:
            out.append(Critique(name, "Not winning on his own serve", f"won {won} of {len(sv)} points he served",
                                "Serve to set up the third ball, then attack a corner", (0.55 - won / len(sv)) * 1.5 * _conf(len(sv)), "servewin"))
    rc = [p for p in points if p.get("server") not in (None, name) and p.get("winner_name") and name in (p.get("near_player"), p.get("far_player"))]
    if len(rc) >= min_n:
        early = sum(p["winner_name"] != name and (p.get("n_crossings") or 9) <= 3 for p in rc)
        if early / len(rc) >= 0.35:
            out.append(Critique(name, "Losing points straight off the serve", f"{early} of {len(rc)} receive points lost within three shots",
                                "Watch the racket at contact, return short and low", (early / len(rc) - 0.2) * 1.5 * _conf(len(rc)), "receive"))
    return sorted(out, key=lambda c: -c.weight)


def critiques_by_point(shots, points):
    """{point id: {name: [Critique]}}: each player's criticisms from the shots and points BEFORE that point (what a coach could have
    said between points)."""
    names = players(points); out = {}
    for i, p in enumerate(points):
        t0 = p["start_t"]
        before = [s for s in shots if s["t"] < t0]
        notes = {n: c for n in names if (c := critique(before, points[:i], n, so_far=True))}
        out[p["id"]] = notes
    return out


def critiques_after(shots, points):
    """Each player's criticisms after the whole recording."""
    return {n: c for n in players(points) if (c := critique(shots, points, n))}


def pick(notes, name, k):
    """The k-th criticism of a player, cycling through his list (so a whole game shows each of them in turn)."""
    lst = (notes or {}).get(name) or []
    return lst[k % len(lst)] if lst else None
