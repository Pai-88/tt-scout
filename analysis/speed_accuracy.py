"""How accurate is the 3D shot speed? Simulated shots with known speeds, seen through a calibrated camera from our recordings, fitted by flight.py.

The simulation is deliberately harder than the fitter's model: air drag with Cd drawn from 0.40 to 0.50 (the fitter assumes 0.42),
Magnus force growing with speed from topspin, backspin and sidespin (the fitter can only approximate it), pixel noise, dropped
frames, wrong detections, a bounce time known only to the nearest frame, and a racket contact found only to within a frame or, for
half the shots, hidden for 3 to 12 frames (the ball behind the body or racket, or the tracker on something else) and found as the
real pipeline finds it (flight._bridge).
    python analysis/speed_accuracy.py [--n 400] [--fps 59.94] [--noise 1.0]
"""
import argparse, json, math, pathlib, sys, time
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from tt_scout.table import Table
from tt_scout.camera import Camera
from tt_scout import flight
from tt_scout.config import NET_X

RHO, AREA, MASS, R = 1.2, math.pi * 0.02 ** 2, 0.0027, 0.02
USE_Y_PRIOR = "--no-feet" not in sys.argv
C0 = RHO * AREA / (2 * MASS)                      # 0.279: drag and lift accelerations are C0 * coefficient * |v| * v


def simulate(p0, v0, cd, cl_top, cl_side, dt=1 / 1500, t_max=1.5):
    """Positions and velocities until the ball's centre comes down to R above the table (or t_max). Plain floats: fast."""
    hx, hy = v0[0], v0[1]; hn = math.hypot(hx, hy) or 1.0
    sx, sy = -hy / hn, hx / hn                                          # topspin axis: z cross (horizontal heading)
    kd, kt, ks = C0 * cd, C0 * cl_top, C0 * cl_side
    px, py, pz = map(float, p0); vx, vy, vz = map(float, v0); t = 0.0
    ts, ps, vs = [0.0], [(px, py, pz)], [(vx, vy, vz)]

    def acc(vx, vy, vz):
        sp = math.sqrt(vx * vx + vy * vy + vz * vz)
        # topspin: s x v with s = (sx, sy, 0); sidespin: z x v
        tx, ty, tz = sy * vz, -sx * vz, sx * vy - sy * vx
        zx, zy = -vy, vx
        return (-kd * sp * vx + sp * (kt * tx + ks * zx), -kd * sp * vy + sp * (kt * ty + ks * zy), -9.81 - kd * sp * vz + sp * kt * tz)
    while t < t_max:
        a1 = acc(vx, vy, vz)
        a2 = acc(vx + 0.5 * dt * a1[0], vy + 0.5 * dt * a1[1], vz + 0.5 * dt * a1[2])
        a3 = acc(vx + 0.5 * dt * a2[0], vy + 0.5 * dt * a2[1], vz + 0.5 * dt * a2[2])
        a4 = acc(vx + dt * a3[0], vy + dt * a3[1], vz + dt * a3[2])
        px += dt * (vx + dt / 6 * (a1[0] + a2[0] + a3[0])); py += dt * (vy + dt / 6 * (a1[1] + a2[1] + a3[1])); pz += dt * (vz + dt / 6 * (a1[2] + a2[2] + a3[2]))
        vx += dt / 6 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0]); vy += dt / 6 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1]); vz += dt / 6 * (a1[2] + 2 * a2[2] + 2 * a3[2] + a4[2])
        t += dt
        ts.append(t); ps.append((px, py, pz)); vs.append((vx, vy, vz))
        if pz <= R and vz < 0:
            break
    return np.array(ts), np.array(ps), np.array(vs)


def make_shot(rng, near=True):
    for _ in range(200):
        sgn = 1 if near else -1
        x0 = -rng.uniform(0.2, 1.4) if near else 2.74 + rng.uniform(0.2, 1.4)
        p0 = np.array([x0, rng.uniform(-0.3, 1.8), rng.uniform(0.05, 0.45)])
        target = np.array([NET_X + sgn * rng.uniform(0.25, 1.30), rng.uniform(0.1, 1.42)])
        speed = math.exp(rng.uniform(math.log(3), math.log(22)))
        kind = rng.choice(["top", "back", "flat"], p=[0.6, 0.25, 0.15])
        cl_top = rng.uniform(0.05, 0.35) if kind == "top" else (-rng.uniform(0.05, 0.25) if kind == "back" else rng.uniform(-0.02, 0.02))
        cl_side = rng.uniform(-0.12, 0.12); cd = rng.uniform(0.40, 0.55)
        az0 = math.atan2(target[1] - p0[1], target[0] - p0[0])
        best = None
        for it in range(2):                                           # aim: correct the heading for sidespin drift
            lo, hi = -0.6, 1.2
            for _ in range(22):                                      # bisection on the launch elevation to land at the target's range
                el = 0.5 * (lo + hi)
                v0 = speed * np.array([math.cos(el) * math.cos(az0), math.cos(el) * math.sin(az0), math.sin(el)])
                ts, ps, vs = simulate(p0, v0, cd, cl_top, cl_side)
                rng_land = abs(ps[-1][0] - p0[0])
                if ps[-1][2] > R + 1e-3 or rng_land > abs(target[0] - p0[0]):
                    hi = el
                else:
                    lo = el
            miss_y = target[1] - ps[-1][1]; az0 += 0.8 * miss_y / max(0.5, abs(target[0] - p0[0]))
            best = (ts, ps, vs, v0)
        ts, ps, vs, v0 = best
        land = ps[-1]
        if not (min(abs(land[0] - NET_X), 1.4) and (1.37 < land[0] < 2.74 if near else 0 < land[0] < 1.37) and 0 <= land[1] <= 1.525):
            continue
        k = np.where(np.diff(np.sign(ps[:, 0] - NET_X)) != 0)[0]
        if not len(k) or ps[k[0], 2] - R < 0.1525 + 0.005:
            continue                                                  # into the net
        if ts[-1] < 0.12:
            continue
        return dict(ts=ts, ps=ps, vs=vs, speed=float(np.linalg.norm(v0)), cd=cd, cl_top=cl_top, cl_side=cl_side)
    return None


HIDE_P = 0.5                 # share of shots whose racket contact is hidden (on our recordings about half: body, racket, a switch to another object)


def hidden_trial(rng, cam, fps, noise, shot, near, drop=0.08, outl=0.04):
    """The contact hidden: the ball rises off its bounce on the hitter's half into the racket, then 1 to 4 frames before and 2 to 9 after
    the contact are lost or taken by a far-off object; the contact is found as flight.shots finds it (flight._bridge), then fitted."""
    ts, ps = shot["ts"], shot["ps"]
    p0 = ps[0]
    xb = rng.uniform(0.25, 1.1) if near else 2.74 - rng.uniform(0.25, 1.1)
    b = np.array([xb, np.clip(p0[1] + rng.normal(0, 0.3), 0.1, 1.4), R]); dti = rng.uniform(0.18, 0.5)
    v_in = (p0 - b) / dti; v_in[2] += 0.5 * 9.81 * dti                   # the rising ball under gravity from the bounce to the contact
    t0 = rng.uniform(0, 1 / fps)
    ft = np.arange(t0 - math.ceil((dti + 0.1) * fps) / fps, ts[-1], 1 / fps)
    ft = ft[(ft > -dti + 0.5 / fps) & (ft < ts[-1] - 0.5 / fps)]
    P = np.empty((len(ft), 3))
    inc = ft < 0
    tt = ft[inc] + dti
    P[inc] = b + np.outer(tt, v_in); P[inc, 2] -= 0.5 * 9.81 * tt ** 2
    P[~inc] = np.stack([np.interp(ft[~inc], ts, ps[:, i]) for i in range(3)], axis=1)
    uv = cam.project(P) + rng.normal(0, noise, (len(ft), 2))
    h1, h2 = rng.integers(1, 5), rng.integers(2, 10)
    k0 = int(np.searchsorted(ft, 0.0))
    hid = np.zeros(len(ft), bool); hid[max(0, k0 - h1):k0 + h2] = True
    if rng.random() < 0.5:                                               # the tracker holds something else meanwhile
        uv[hid] = np.array([rng.uniform(1300, 1700), rng.uniform(600, 760)]) + rng.normal(0, 3, (hid.sum(), 2))
    else:
        uv[hid] = np.nan
    keep = (rng.random(len(ft)) > drop) | hid
    keep[k0 + h2:k0 + h2 + 3] = True                                     # the first outgoing frames are seen (they end the gap)
    bad = (rng.random(len(ft)) < outl) & ~hid
    uv[bad] += rng.uniform(10, 40, (bad.sum(), 2)) * rng.choice([-1, 1], (bad.sum(), 2))
    uv[~keep] = np.nan
    track = np.c_[np.arange(len(ft)), ft, uv]
    vis_out = np.where(~np.isnan(uv[:, 0]) & ~hid & (ft > 0))[0]
    if len(vis_out) < 6:
        return None
    t_out = ft[vis_out[0]]
    d_out = np.sign(uv[vis_out[1], 0] - uv[vis_out[0], 0])
    tb_ev = t0 + round((-dti - t0) * fps) / fps                          # the incoming bounce, to the nearest frame
    tc = flight._bridge(track, t_out, tb_ev, d_out, fps)
    tc_est = tc if tc is not None else t_out
    sel = (ft >= t_out - 1e-6) & ~np.isnan(uv[:, 0])
    fts, fuv = ft[sel], uv[sel]
    tb = ts[-1]; tb_frame = t0 + round((tb - t0) * fps) / fps
    buv = cam.project([ps[-1]])[0] + rng.normal(0, 1.5, 2)
    yp = (float(p0[1] + rng.normal(0, 0.25)), 0.35) if USE_Y_PRIOR else None
    f = flight.fit(cam, fts, fuv, tb_frame, buv, fps, tc_est, y_prior=yp)
    base = dict(hidden=True, bridged=tc is not None, tc_err=float(tc_est) * fps, true=shot["speed"])
    if not f.get("ok"):
        return dict(ok=False, why=f.get("why"), **base)
    return dict(ok=True, est=f["speed"], err=(f["speed"] - shot["speed"]) / shot["speed"], n=len(fts), spin=shot["cl_top"], side=shot["cl_side"],
                cd=shot["cd"], sd=f.get("speed_sd"), p0_err=[float(a - b_) for a, b_ in zip(f["p0"], p0)], **base)


FREE = "--free" in sys.argv       # bench the no-bounce fit (flight.fit_free): the bounce is hidden and the flight is cut short
PRIOR_NOISE = 0.45                # m: how far the true contact is from where the pipeline assumes it across the table (measured 0.4-0.47)


def free_trial(rng, cam, fps, noise, shot, drop=0.08, outl=0.04):
    """A shot whose bounce is never seen (it went long or wide, or the landing was missed): frames from the contact to a random point
    after the net crossing, no bounce event, and the crossing's time known to the nearest frame."""
    ts, ps = shot["ts"], shot["ps"]
    k = np.where(np.diff(np.sign(ps[:, 0] - NET_X)) != 0)[0]
    if not len(k):
        return None
    t_cross = float(ts[k[0]])
    t0 = rng.uniform(0, 1 / fps)
    t_end = t_cross + rng.uniform(0.25, 0.95) * (ts[-1] - t_cross)
    ft = np.arange(t0, t_end, 1 / fps)
    ft = ft[ft > 0.5 / fps]
    if len(ft) < 7:
        return None
    P = np.stack([np.interp(ft, ts, ps[:, i]) for i in range(3)], axis=1)
    uv = cam.project(P) + rng.normal(0, noise, (len(ft), 2))
    keep = rng.random(len(ft)) > drop
    bad = rng.random(len(ft)) < outl
    uv[bad] += rng.uniform(10, 40, (bad.sum(), 2)) * rng.choice([-1, 1], (bad.sum(), 2))
    ft, uv = ft[keep], uv[keep]
    if len(ft) < 6:
        return None
    tc_frame = t0 + round((t_cross - t0) * fps) / fps
    tc_est = round(rng.uniform(-1, 1)) / fps
    # the hitter's position as the pipeline knows it: his feet plus his own forehand offset, off by his scatter (0.4 to 0.47 m measured)
    yp = (float(ps[0][1] + rng.normal(0, PRIOR_NOISE)), max(0.35, PRIOR_NOISE)) if USE_Y_PRIOR else None
    # the crossing event: the first frame past the net's base line in the picture (the plane through the camera and that line)
    Qn = np.array([NET_X, 0.0, 0.0]); nrm = np.cross(cam.C - Qn, [0.0, 1.0, 0.0]); nrm /= np.linalg.norm(nrm)
    sd_ = (ps - Qn) @ nrm; kk = np.where(np.diff(np.sign(sd_)) != 0)[0]
    if not len(kk):
        return None
    t_pl = float(ts[kk[0]]); tc_frame = t0 + math.ceil((t_pl - t0) * fps) / fps
    f = flight.fit_free(cam, ft, uv, fps, tc_est, t_cross=tc_frame, y_prior=yp)
    base = dict(free=True, true=shot["speed"], frac=float((t_end - t_cross) / max(1e-6, ts[-1] - t_cross)))
    if not f.get("ok"):
        return dict(ok=False, why=f.get("why"), **base)
    return dict(ok=True, est=f["speed"], err=(f["speed"] - shot["speed"]) / shot["speed"], n=len(ft), spin=shot["cl_top"], side=shot["cl_side"],
                cd=shot["cd"], sd=f.get("speed_sd"), p0_err=[float(a - b) for a, b in zip(f["p0"], ps[0])], **base)


def trial(rng, cam, fps, noise, drop=0.08, outl=0.04):
    near = rng.random() < 0.5
    shot = make_shot(rng, near=near)
    if shot is None:
        return None
    if FREE:
        return free_trial(rng, cam, fps, noise, shot, drop, outl)
    if HIDE_P and rng.random() < HIDE_P:
        return hidden_trial(rng, cam, fps, noise, shot, near, drop, outl)
    ts, ps = shot["ts"], shot["ps"]
    t0 = rng.uniform(0, 1 / fps)                                     # the camera's frame clock against the contact
    ft = np.arange(t0, ts[-1], 1 / fps)
    ft = ft[(ft > 0.5 / fps) & (ft < ts[-1] - 0.5 / fps)]
    if len(ft) < 6:
        return None
    P = np.stack([np.interp(ft, ts, ps[:, i]) for i in range(3)], axis=1)
    uv = cam.project(P) + rng.normal(0, noise, (len(ft), 2))
    keep = rng.random(len(ft)) > drop
    bad = rng.random(len(ft)) < outl
    uv[bad] += rng.uniform(10, 40, (bad.sum(), 2)) * rng.choice([-1, 1], (bad.sum(), 2))
    ft, uv = ft[keep], uv[keep]
    if len(ft) < 5:
        return None
    tb = ts[-1]; tb_frame = t0 + round((tb - t0) * fps) / fps        # the bounce event: the nearest frame
    buv = cam.project([ps[-1]])[0] + rng.normal(0, 1.5, 2)
    tc_est = round(rng.uniform(-1, 1)) / fps                          # the contact found by the direction reversal: within a frame
    yp = (float(ps[0][1] + rng.normal(0, 0.25)), 0.35) if USE_Y_PRIOR else None   # the hitter's feet: within arm's reach of the ball
    f = flight.fit(cam, ft, uv, tb_frame, buv, fps, tc_est, y_prior=yp)
    if not f.get("ok"):
        return dict(ok=False, why=f.get("why"), true=shot["speed"])
    return dict(ok=True, true=shot["speed"], est=f["speed"], err=(f["speed"] - shot["speed"]) / shot["speed"], n=len(ft),
                spin=shot["cl_top"], side=shot["cl_side"], cd=shot["cd"], sd=f.get("speed_sd"),
                p0_err=[float(a - b) for a, b in zip(f["p0"], ps[0])])            # the fitted contact point against the true one (m)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=300); ap.add_argument("--fps", type=float, default=59.94)
    ap.add_argument("--noise", type=float, default=1.0); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--out")
    ap.add_argument("--no-feet", action="store_true", help="fit without the hitter's-feet prior on where across the table the ball was hit")
    ap.add_argument("--free", action="store_true", help="bench the no-bounce fit: the bounce hidden and the flight cut short")
    ap.add_argument("--table", help="the calibration whose camera sees the shots (default: our side-on recording, else OpenTTGames test_2)")
    a = ap.parse_args()
    tp = pathlib.Path(a.table) if a.table else next(p for p in (ROOT / "data/own/IMG_3144_table.json", ROOT / "data/test_2_table.json") if p.exists())
    cam = Camera.from_table_pnp(Table.load(tp), 1920, 1080)
    rng = np.random.default_rng(a.seed); res = []; t0 = time.time()
    while len(res) < a.n:
        r = trial(rng, cam, a.fps, a.noise)
        if r is not None:
            res.append(r)
    ok = [r for r in res if r["ok"]]
    if FREE:
        rej = [r for r in res if not r["ok"]]
        print("no-bounce fits: rejected", len(rej), "of", len(res), dict(__import__("collections").Counter(r["why"].split(";")[0] for r in rej).most_common(4)))
    e = np.array([r["err"] for r in ok]) * 100
    print(f"{len(ok)}/{len(res)} fits accepted ({time.time() - t0:.0f} s); speed error %: median |e| {np.median(np.abs(e)):.2f}, "
          f"mean e (bias) {e.mean():+.2f}, p90 |e| {np.percentile(np.abs(e), 90):.2f}, p95 |e| {np.percentile(np.abs(e), 95):.2f}, max |e| {np.abs(e).max():.1f}")
    for lo, hi in ((3, 6), (6, 10), (10, 15), (15, 22)):
        sel = [abs(r["err"]) * 100 for r in ok if lo <= r["true"] < hi]
        if sel:
            print(f"   true {lo:2d}-{hi:2d} m/s: n {len(sel):3d}, median |e| {np.median(sel):.2f}%, p95 {np.percentile(sel, 95):.2f}%")
    for name, sel in (("topspin", [r for r in ok if r["spin"] > 0.05]), ("backspin", [r for r in ok if r["spin"] < -0.05]), ("flat", [r for r in ok if abs(r["spin"]) <= 0.05])):
        if sel:
            ee = np.array([r["err"] for r in sel]) * 100
            print(f"   {name:8s}: n {len(sel):3d}, bias {ee.mean():+.2f}%, median |e| {np.median(np.abs(ee)):.2f}%")
    for name, sel in (("contact seen", [r for r in ok if not r.get("hidden")]), ("contact hidden, bridged", [r for r in ok if r.get("hidden") and r["bridged"]]),
                      ("contact hidden, not bridged", [r for r in ok if r.get("hidden") and not r["bridged"]]),
                      ("seen or bridged (what the report shows)", [r for r in ok if not r.get("hidden") or r["bridged"]])):
        if not sel:
            continue
        ee = np.array([r["err"] for r in sel]) * 100; pe = np.abs(np.array([r["p0_err"] for r in sel])) * 100
        print(f"   {name}: n {len(sel)}, speed bias {ee.mean():+.2f}%, median |e| {np.median(np.abs(ee)):.2f}%, p95 {np.percentile(np.abs(ee), 95):.2f}%; "
              "contact point cm (along, across, height) median " + ", ".join(f"{v:.1f}" for v in np.median(pe, axis=0))
              + ", p90 " + ", ".join(f"{v:.1f}" for v in np.percentile(pe, 90, axis=0)))
    hid = [r for r in res if r.get("hidden")]
    if hid:
        te = np.abs([r["tc_err"] for r in hid if r["bridged"]])
        print(f"   hidden contacts: {sum(r['bridged'] for r in hid)}/{len(hid)} bridged, contact time error median {np.median(te):.2f} frames, "
              f"p90 {np.percentile(te, 90):.2f}")
    if a.out:
        json.dump(res, open(a.out, "w"), default=float)
