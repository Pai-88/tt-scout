"""The drop test: the measured speeds checked against a speed known from physics alone.

Hold the ball still with its bottom a measured height above the table (1.00 m on a tape measure), let go without pushing it, and let
it bounce. Do it about 10 times, away from the net, filmed from the tripod exactly as a match is. A ball let go from rest reaches the
table at a speed set only by the height, gravity and air drag: 4.15 m/s (15.0 km/h) from 1.00 m. Each drop is measured by the same
camera model and the same flight fit (flight.fit) that measure the match speeds, so the difference is the real error of the chain
table corners -> camera -> ball tracking -> 3D fit, on a real ball.

What it checks: the camera's scale (a wrong focal length or table corner shows up here), the frame timing, the tracking and the fit
where it is best pinned down, at the bounce. What it cannot check: a match shot's spin, or a racket contact the camera did not see
(the speed off the racket is carried back from the bounce along the fitted flight).

    tt-scout droptest drops.MOV --height 1.00
"""
import math
import numpy as np
from .flight import C0, CD0, CD_SD, G, R_BALL, fit, integrate

REBOUND_S = 1.2         # a bounce this soon after the one before is the ball coming down again, lower than the hand let it go
MIN_FRAMES = 8          # tracked frames of the fall needed to fit it
RISE_PX = 1.0           # going back in time over a fall the ball only climbs in the picture; lower than this = it was rising


def predicted_speed(h, cd=CD0):
    """Speed (m/s) at which a ball let go from rest, its bottom h metres above the table, reaches it: gravity and air drag growing
    with the square of the speed, solved exactly. With no drag this is sqrt(2 g h)."""
    vt2 = G / (C0 * cd)                                             # terminal speed squared, where drag equals the ball's weight
    return math.sqrt(vt2 * (1.0 - math.exp(-2.0 * G * h / vt2)))


def find_drops(track, events):
    """The first bounce after each drop and the tracked frames of the fall before it.
    track: rows [frame, t, x, y, ...], x and y NaN where the ball was not seen; events: tt_scout.events.detect_events.
    Returns (drops, skipped): drops = [dict(bounce=event, idx=track rows of the fall, in time order)], skipped = [(t, reason)]."""
    t, y = track[:, 1], track[:, 3]
    seen = ~np.isnan(track[:, 2])
    drops, skipped, prev = [], [], -1e9
    for b in sorted((e for e in events if e["kind"] == "bounce"), key=lambda e: e["t"]):
        since, prev = b["t"] - prev, b["t"]
        if since < REBOUND_S:
            skipped.append((b["t"], "the ball coming down again after a bounce"))
            continue
        idx, k, gap, rising = [], int(np.searchsorted(t, b["t"])) - 1, 0, False
        while k >= 0 and gap <= 2 and b["t"] - t[k] < 1.0:          # back from the bounce over the fall
            if seen[k]:
                if idx and y[k] > y[idx[-1]] + RISE_PX:
                    rising = True                                   # lower before: it was going up, so it bounced before this fall
                    break
                idx.append(k); gap = 0
            else:
                gap += 1
            k -= 1
        if rising:
            skipped.append((b["t"], "the ball was rising before it fell (a bounce before it was missed)"))
        elif len(idx) < MIN_FRAMES:
            skipped.append((b["t"], f"only {len(idx)} frames of the fall were tracked"))
        else:
            drops.append(dict(bounce=b, idx=idx[::-1]))
    return drops, skipped


def fitted_height(f):
    """How high the fitted flight started from rest: where, going back from the bounce, the ball stopped rising. None if it never did."""
    a = f["anchor"]; ts = np.linspace(a["t"] - 0.8, a["t"] - 1e-4, 800)
    pos, vel = integrate(np.array(a["p"]), np.array(a["v"]), (f["cl_top"], f["cl_side"], f["cd"]), a["t"], ts)
    up = np.where(vel[:, 2] >= 0)[0]
    return None if not len(up) else float(pos[up[-1], 2] - R_BALL)


def measure(track, events, fps, cam, height):
    """Every drop in a recording, fitted. Returns dict(expected, rows=[per drop], skipped, summary)."""
    exp = predicted_speed(height)
    drops, skipped = find_drops(track, events)
    rows = []
    for d in drops:
        b, idx = d["bounce"], d["idx"]
        times, uv = track[idx, 1], track[idx, 2:4]
        f = fit(cam, times, uv, b["t"], (b["x_px"], b["y_px"]), fps, float(times[0]), v_guess=(0.0, 0.0, -exp), speed_range=(0.0, 40.0))
        if not f.get("ok"):
            skipped.append((b["t"], f"the fit was not believable ({f.get('why')})"))
            continue
        v, sd = f["speed_bounce"], f["speed_bounce_sd"]
        rows.append(dict(t=round(b["t"], 2), speed=round(v, 3), sd=None if sd is None else round(sd, 3), err_pct=round(100 * (v / exp - 1), 2),
                         height_fit=None if (h := fitted_height(f)) is None else round(h, 3), n=f["n"], rms_px=round(f["rms_px"], 2),
                         bounce_m=[round(c, 3) for c in f["bounce"]]))
    return dict(height=height, expected=round(exp, 3), expected_lo=round(predicted_speed(height, CD0 + CD_SD), 3),
                expected_hi=round(predicted_speed(height, CD0 - CD_SD), 3), rows=rows, skipped=sorted(skipped),
                summary=summarise(rows, exp))


def summarise(rows, exp):
    """Bias (mean error, with its standard error), scatter (sd of the errors), and how many drops came within their own error bar:
    about 2 in 3 should, if the fit's error bars are honest."""
    if not rows:
        return None
    e = np.array([r["err_pct"] for r in rows])
    inside = [abs(r["speed"] - exp) <= r["sd"] for r in rows if r["sd"]]
    return dict(n=len(rows), bias_pct=round(float(e.mean()), 2),
                bias_se_pct=round(float(e.std(ddof=1) / math.sqrt(len(e))), 2) if len(e) > 1 else None,
                scatter_pct=round(float(e.std(ddof=1)), 2) if len(e) > 1 else None, worst_pct=round(float(np.abs(e).max()), 2),
                within_own_bar=int(sum(inside)), with_bar=len(inside))


def lines(res):
    """The result as text for the terminal."""
    out = [f"the physics: let go from {res['height']:.2f} m, a ball reaches the table at {res['expected']:.2f} m/s "
           f"({3.6 * res['expected']:.1f} km/h); {res['expected_lo']:.2f} to {res['expected_hi']:.2f} for any drag coefficient "
           f"{CD0 - CD_SD:.2f} to {CD0 + CD_SD:.2f}"]
    if res["rows"]:
        out.append("  drop   time     measured             error   fitted height")
        for i, r in enumerate(res["rows"], 1):
            bar = f"± {r['sd']:.2f}" if r["sd"] is not None else "      "
            hf = f"{r['height_fit']:.2f} m" if r["height_fit"] is not None else "-"
            out.append(f"  {i:>4}  {r['t']:6.1f} s  {r['speed']:.2f} {bar} m/s   {r['err_pct']:+5.1f}%   {hf}")
    rebounds = [t for t, why in res["skipped"] if why.startswith("the ball coming down again")]
    if rebounds:
        out.append(f"  {len(rebounds)} bounces skipped: the ball coming down again after a bounce, lower than it was let go from")
    for t, why in res["skipped"]:
        if not why.startswith("the ball coming down again"):
            out.append(f"  skipped the bounce at {t:.1f} s: {why}")
    s = res["summary"]
    if s is None:
        out.append("no drop could be measured")
    else:
        se = f" (± {s['bias_se_pct']:.1f}%)" if s["bias_se_pct"] is not None else ""
        sc = f", scatter {s['scatter_pct']:.1f}%" if s["scatter_pct"] is not None else ""
        out.append(f"{s['n']} drops: bias {s['bias_pct']:+.1f}%{se}{sc}, worst {s['worst_pct']:.1f}%; {s['within_own_bar']} of "
                   f"{s['with_bar']} within their own error bar (about 2 in 3 if the error bars are honest)")
    return out


def figure(res, path):
    """Each drop's measured speed with its error bar against the speed physics gives (the band: any drag coefficient in range)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = res["rows"]
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    ax.axhspan(res["expected_lo"], res["expected_hi"], color="0.85", lw=0)
    ax.axhline(res["expected"], color="0.3", lw=1)
    x = np.arange(1, len(rows) + 1)
    ax.errorbar(x, [r["speed"] for r in rows], yerr=[r["sd"] or 0 for r in rows], fmt="o", color="#b5651d", capsize=3, ms=5)
    ax.set_xlabel("drop"); ax.set_ylabel("speed at the table (m/s)")
    ax.set_title(f"Drops from {res['height']:.2f} m: measured by the camera vs physics ({res['expected']:.2f} m/s)", fontsize=10)
    ax.set_xticks(x)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
