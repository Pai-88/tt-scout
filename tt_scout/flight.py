"""Shot speed in 3D: each shot's flight from the racket to its bounce, fitted with real ball physics through the calibrated camera.
It replaces the old number (the along-table distance from the net to the bounce over the time between), which
read 3 to 7 m/s: it measured the ball late in its flight, only along the table, and left out its drop.

For one shot:
  * where it starts: the racket contact, found as the last reversal of the ball's direction along the table before it crossed the
    net (or, for a serve, its bounce on the server's side);
  * where it ends: its bounce on the other side, a known point on the table (the ball's centre 2 cm above the surface);
  * the flight between is fitted: gravity, air drag (k = rho Cd A / 2m = 0.117 per metre for a 40 mm, 2.7 g ball, Cd 0.42) and a
    constant extra up-or-down acceleration standing in for spin (topspin dips the ball, backspin holds it up). Unknowns: the bounce
    point and time, the velocity just before the bounce, and the spin term; they are chosen so the fitted flight, seen through the
    camera, lands on the tracked ball in every frame (Levenberg-Marquardt, robust to the odd wrong detection).
  * the speed reported is the fitted speed at the racket contact.
A fit is only used when it is believable: the fitted flight must go over the net (15.25 cm) where the shot crossed it, match the
tracked ball within a few pixels, and give a speed a table tennis ball can have.

Checked 2026-09-26: on our first two recordings the fitted flight lands on the tracked ball within 0.7 px (median) and every accepted
flight clears the net; on the professional OpenTTGames test matches the same method gives rally shots of 11 to 13 m/s (median, 2 of
3 matches; 7 m/s in the slower third), in line with published ball speeds for competitive play.
"""
import bisect
import math
import numpy as np
from .config import NET_X, TABLE_LENGTH as L, TABLE_WIDTH as W

G, R_BALL, NET_H = 9.81, 0.02, 0.1525
C0 = 1.2 * math.pi * 0.02 ** 2 / (2 * 0.0027)     # rho A / 2m = 0.279: drag and lift accelerations are C0 * coefficient * |v| * v
CD0, CD_SD = 0.47, 0.05                           # drag coefficient: a smooth 40 mm sphere at these speeds (Re ~ 3e4) is ~0.47
K_DRAG = C0 * CD0                                 # kept for callers


def integrate(p_b, v_b, spin, t_b, times, h=1 / 600):
    """Positions and velocities at `times` (all before t_b), integrating backwards from the state (p_b, v_b) at t_b with RK4.
    spin = (cl_top, cl_side, cd): lift from topspin (about the horizontal axis across the flight; positive dips the ball), from
    sidespin (about the vertical axis) and the drag coefficient. Plain floats for speed."""
    cl_t, cl_s, cd = spin
    hx, hy = v_b[0], v_b[1]; hn = math.hypot(hx, hy) or 1.0
    sx, sy = -hy / hn, hx / hn                                          # z cross heading: topspin makes s x v point down
    kd, kt, ks = C0 * cd, C0 * cl_t, C0 * cl_s

    def acc(vx, vy, vz):
        sp = math.sqrt(vx * vx + vy * vy + vz * vz)
        tx, ty, tz = sy * vz, -sx * vz, sx * vy - sy * vx
        return (-kd * sp * vx + sp * (kt * tx - ks * vy), -kd * sp * vy + sp * (kt * ty + ks * vx), -G - kd * sp * vz + sp * kt * tz)
    order = np.argsort(times)[::-1]
    out_p = np.zeros((len(times), 3)); out_v = np.zeros((len(times), 3))
    px, py, pz = map(float, p_b); vx, vy, vz = map(float, v_b); t = float(t_b)
    for i in order:
        target = float(times[i])
        while t - target > 1e-9:
            dt = -min(h, t - target)
            a1 = acc(vx, vy, vz)
            a2 = acc(vx + 0.5 * dt * a1[0], vy + 0.5 * dt * a1[1], vz + 0.5 * dt * a1[2])
            a3 = acc(vx + 0.5 * dt * a2[0], vy + 0.5 * dt * a2[1], vz + 0.5 * dt * a2[2])
            a4 = acc(vx + dt * a3[0], vy + dt * a3[1], vz + dt * a3[2])
            px += dt * (vx + dt / 6 * (a1[0] + a2[0] + a3[0])); py += dt * (vy + dt / 6 * (a1[1] + a2[1] + a3[1]))
            pz += dt * (vz + dt / 6 * (a1[2] + a2[2] + a3[2]))
            vx += dt / 6 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0]); vy += dt / 6 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1])
            vz += dt / 6 * (a1[2] + 2 * a2[2] + 2 * a3[2] + a4[2])
            t += dt
        out_p[i] = (px, py, pz); out_v[i] = (vx, vy, vz)
    return out_p, out_v


def fit(cam, times, uv, t_b0, bounce_uv, fps, t_contact, sigma_px=2.5, iters=40, y_prior=None, v_guess=None, speed_range=(1.0, 40.0)):
    """Fit one flight. times, uv: tracked ball (seconds, pixels) between the contact and the bounce; t_b0, bounce_uv: the bounce event.
    Unknowns: bounce point (x, y) and time, velocity just before the bounce, topspin and sidespin lift coefficients, drag coefficient.
    y_prior = (mu, sd): where across the table (m) the ball was at the contact, from the hitter's feet. From one side camera the ball's
    motion toward or away from the lens is the least observed part of the flight; this pins it.
    v_guess: the velocity just before the bounce to start from, instead of a shot along the table (the drop test's straight fall).
    speed_range: the speeds at t_contact (m/s) a believable flight may have (a dropped ball starts at rest).
    Returns a dict: speed (m/s at t_contact) and speed_sd (its standard error from the fit), speed_bounce and speed_bounce_sd (the same
    just before the bounce), v0, rms_px, n, net_z, ok, why, ..."""
    times = np.asarray(times, float); uv = np.asarray(uv, float)
    if len(times) < 5:
        return dict(ok=False, why="too few tracked frames")
    pb0 = cam.on_plane_z([bounce_uv], R_BALL)[0]
    P = np.array([_on_plane_y(cam, q, pb0[1]) for q in uv])             # starting guess: the flight in the vertical plane along the table
    good = np.isfinite(P).all(axis=1)
    if good.sum() >= 4:
        A = np.c_[times[good] - t_b0, np.ones(good.sum())]
        vx = float(np.linalg.lstsq(A, P[good, 0], rcond=None)[0][0])
        cz = np.polyfit(times[good] - t_b0, P[good, 2], 2)
        vz = float(cz[1]); vx = vx if abs(vx) > 0.5 else 5.0 * np.sign(pb0[0] - L / 2)
    else:
        vx, vz = 6.0 * np.sign(pb0[0] - L / 2), -2.0
    NP = 9
    theta = np.array([pb0[0], pb0[1], t_b0, vx, 0.0, min(vz, -0.5), 0.05, 0.0, CD0])
    if v_guess is not None:
        theta[3:6] = v_guess
    prior_mu = np.array([pb0[0], pb0[1], t_b0, np.nan, np.nan, np.nan, 0.05, 0.0, CD0])
    prior_sd = np.array([0.04, 0.06, 0.8 / fps, np.nan, np.nan, np.nan, 0.25, 0.12, CD_SD])
    PRI = [0, 1, 2, 6, 7, 8]

    t_eval = np.r_[times, t_contact] if y_prior else times

    def resid(th):
        if th[2] <= times.max() or th[8] <= 0.05:
            return None
        pos, _ = integrate(np.array([th[0], th[1], R_BALL]), th[3:6], (th[6], th[7], th[8]), th[2], t_eval)
        r = ((cam.project(pos[:len(times)]) - uv) / sigma_px).ravel()
        pri = [(th[j] - prior_mu[j]) / prior_sd[j] for j in PRI]
        if y_prior:
            pri.append((pos[-1, 1] - y_prior[0]) / y_prior[1])
        return np.r_[r, pri]

    NPRI = len(PRI) + (1 if y_prior else 0)
    lam = 1e-2; r = resid(theta)
    if r is None:
        return dict(ok=False, why="bounce before the tracked flight")
    w = _cauchy_w(r[:-NPRI]); cost = _cost(r, w, NPRI)
    steps = np.array([1e-3, 1e-3, 1e-4, 1e-3, 1e-3, 1e-3, 2e-3, 2e-3, 2e-3])
    J = None; A = None
    for _ in range(iters):
        J = np.zeros((len(r), NP))
        for j in range(NP):
            d = np.zeros(NP); d[j] = steps[j]
            rp, rm = resid(theta + d), resid(theta - d)
            if rp is None or rm is None:
                return dict(ok=False, why="fit left the flight window")
            J[:, j] = (rp - rm) / (2 * steps[j])
        Wv = np.r_[w, np.ones(NPRI)]
        JW = J * Wv[:, None]
        A = J.T @ JW; g = JW.T @ r
        improved = False
        for _ in range(8):
            try:
                delta = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
            except np.linalg.LinAlgError:
                break
            cand = theta + delta; rc = resid(cand)
            if rc is not None:
                wc = _cauchy_w(rc[:-NPRI]); cc = _cost(rc, wc, NPRI)
                if cc < cost:
                    theta, r, w, cost = cand, rc, wc, cc; lam = max(lam / 3, 1e-7); improved = True
                    break
            lam *= 4
        if not improved or np.abs(delta).max() < 1e-6:
            break
    p_b = np.array([theta[0], theta[1], R_BALL]); v_b = theta[3:6]; spin = (theta[6], theta[7], theta[8])
    pos, vel = integrate(p_b, v_b, spin, theta[2], np.r_[times, t_contact])
    v0 = vel[-1]; speed = float(np.linalg.norm(v0)); p0 = pos[-1]
    # the whole fitted flight, contact to bounce, for drawing it in 3D (analysis3d.py)
    tt_ = np.linspace(t_contact, theta[2] - 1e-4, 24)
    path, _ = integrate(np.array([theta[0], theta[1], R_BALL]), theta[3:6], (theta[6], theta[7], theta[8]), theta[2], tt_)
    # standard error of the speed: the fit's covariance pushed through the speed at the contact (numerical gradient)
    speed_sd = speed_bounce_sd = None
    try:
        dof = max(1, len(r) - NP); s2 = max(1.0, cost / dof)
        cov = np.linalg.inv(A + 1e-9 * np.eye(NP)) * s2
        gr = np.zeros(NP)
        for j in range(NP):
            d = np.zeros(NP); d[j] = steps[j]
            thp = theta + d
            _, vv = integrate(np.array([thp[0], thp[1], R_BALL]), thp[3:6], (thp[6], thp[7], thp[8]), thp[2], np.array([t_contact]))
            gr[j] = (np.linalg.norm(vv[0]) - speed) / steps[j]
        speed_sd = float(np.sqrt(max(0.0, gr @ cov @ gr)))
        gb = np.zeros(NP); gb[3:6] = theta[3:6] / max(1e-9, float(np.linalg.norm(theta[3:6])))   # |v| just before the bounce: exact gradient
        speed_bounce_sd = float(np.sqrt(max(0.0, gb @ cov @ gb)))
    except np.linalg.LinAlgError:
        pass
    profile = sorted([(float(t_contact), speed)] + [(float(ti), float(np.linalg.norm(vi))) for ti, vi in zip(times, vel[:-1])])
    err = np.linalg.norm(cam.project(pos[:-1]) - uv, axis=1)
    inl = err < 4 * sigma_px
    rms = float(np.sqrt(np.mean(err[inl] ** 2))) if inl.any() else float("inf")
    net_z = None
    tt = np.linspace(t_contact, theta[2], 60)
    pp, _ = integrate(p_b, v_b, spin, theta[2], tt[:-1])
    xs = pp[:, 0]
    k = np.where(np.diff(np.sign(xs - NET_X)) != 0)[0]
    if len(k):
        i = k[0]; a_ = (NET_X - xs[i]) / (xs[i + 1] - xs[i]); net_z = float(pp[i, 2] + a_ * (pp[i + 1, 2] - pp[i, 2]))
    vmean = float(np.mean([v for _, v in profile]))
    az = float(-C0 * theta[6] * vmean ** 2)                              # the topspin dip as a vertical acceleration at the mean speed
    why = []
    if inl.sum() < max(5, 0.6 * len(err)):
        why.append("too many frames off the fitted flight")
    if rms > 3.5 * sigma_px:
        why.append(f"fits the ball badly ({rms:.1f} px)")
    if net_z is not None and not (NET_H - 0.03 <= net_z - R_BALL <= 1.2):
        why.append(f"passes the net line at {100 * (net_z - R_BALL):.0f} cm")
    if not speed_range[0] <= speed <= speed_range[1]:
        why.append(f"speed {speed:.1f} m/s")
    if not (-0.6 <= theta[6] <= 0.8 and abs(theta[7]) <= 0.6 and 0.2 <= theta[8] <= 0.9):
        why.append("implausible spin or drag")
    return dict(ok=not why, why="; ".join(why), speed=speed, speed_sd=speed_sd, v0=[float(x) for x in v0], rms_px=rms, n=int(len(times)),
                n_in=int(inl.sum()), net_z=net_z, az=az, cl_top=float(theta[6]), cl_side=float(theta[7]), cd=float(theta[8]),
                t_bounce=float(theta[2]), bounce=[float(theta[0]), float(theta[1])], v_bounce=[float(x) for x in v_b],
                speed_bounce=float(np.linalg.norm(v_b)), speed_bounce_sd=speed_bounce_sd, p0=[float(x) for x in p0], path=[[round(float(c), 3) for c in q] for q in path],
                over_net_cm=None if net_z is None else round(100 * (net_z - R_BALL - NET_H), 1), profile=profile,
                anchor=dict(t=float(theta[2]), p=[float(theta[0]), float(theta[1]), R_BALL], v=[float(x) for x in v_b]))


def fit_free(cam, times, uv, fps, t_contact, t_cross=None, sigma_px=2.5, iters=60, y_prior=None, x_end=None, max_sd_frac=0.2):
    """Fit a flight that never bounced on the table (long, wide, into the net, or its landing was not seen), so there is no bounce to
    hang it on. The state (position and velocity) at the last tracked frame is free and integrated back through every tracked frame
    to the contact, with the same physics as fit(). Without a bounce the flight's depth is pinned by: the moment it crossed the net
    (t_cross: the ball is in the net's plane then), gravity against the picture's scale, and the hitter's feet (y_prior). For a ball
    into the net, x_end: the net's plane at the last frame. Accepted only when the fit is tight, the speed's standard error is under
    max_sd_frac of the speed, and the contact is within the hitter's reach. Same result dict as fit(), with free=True."""
    times = np.asarray(times, float); uv = np.asarray(uv, float)
    if len(times) < 6:
        return dict(ok=False, why="too few tracked frames", free=True)
    t_a = float(times.max())
    y0 = y_prior[0] if y_prior else W / 2
    P = np.array([_on_plane_y(cam, q, y0) for q in uv])
    good = np.isfinite(P).all(axis=1)
    if good.sum() < 4:
        return dict(ok=False, why="flight off the table's plane", free=True)
    A = np.c_[times[good] - t_a, np.ones(good.sum())]
    cx = np.linalg.lstsq(A, P[good, 0], rcond=None)[0]
    cz = np.polyfit(times[good] - t_a, P[good, 2], 2)
    theta = np.array([cx[1], y0, cz[2], cx[0], 0.0, cz[1], 0.05, 0.0, CD0])
    NP = 9
    prior_mu = np.array([np.nan] * 6 + [0.05, 0.0, CD0]); prior_sd = np.array([np.nan] * 6 + [0.25, 0.12, CD_SD]); PRI = [6, 7, 8]
    t_cr = None if t_cross is None else t_cross - 0.5 / fps          # the crossing event is the first frame past the net line
    extra = [t for t in (t_cr, t_contact) if t is not None]
    t_eval = np.r_[times, extra]
    n_t = len(times); i_last = int(np.argmax(times))
    Qn = np.array([NET_X, 0.0, 0.0])                                    # the net's base line on the table, x = NET_X, z = 0, along y:
    nrm = np.cross(cam.C - Qn, [0.0, 1.0, 0.0]); nrm /= np.linalg.norm(nrm)   # its plane with the camera = where the picture crosses it

    def resid(th):
        if th[8] <= 0.05:
            return None
        pos, _ = integrate(th[0:3], th[3:6], (th[6], th[7], th[8]), t_a + 1e-9, t_eval)
        if not np.all(np.isfinite(pos)) or np.abs(th[3:6]).max() > 60:
            return None                                                   # a step that blew the flight up: not a candidate
        r = ((cam.project(pos[:n_t]) - uv) / sigma_px).ravel()
        pri = [(th[j] - prior_mu[j]) / prior_sd[j] for j in PRI]
        k = n_t
        if t_cr is not None:
            pri.append(float(nrm @ (pos[k] - Qn)) / 0.06)                    # crossing the net line in the picture then
            below = (NET_H + R_BALL - 0.02) - pos[k, 2]
            pri.append(max(0.0, below) / 0.02)                              # and over the net, not through it
            k += 1
        if y_prior:
            pri.append((pos[k, 1] - y_prior[0]) / y_prior[1])
        if x_end is not None:
            pri.append((pos[i_last, 0] - x_end) / 0.05)                      # a ball into the net ends in the net's plane
        return np.r_[r, pri]

    r = resid(theta)
    if r is None:
        return dict(ok=False, why="bad start", free=True)
    NPRI = len(r) - 2 * n_t
    lam = 1e-2; w = _cauchy_w(r[:2 * n_t]); cost = _cost(r, w, NPRI)
    steps = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 2e-3, 2e-3, 2e-3])
    A_ = None
    for _ in range(iters):
        J = np.zeros((len(r), NP))
        for j in range(NP):
            d = np.zeros(NP); d[j] = steps[j]
            rp, rm = resid(theta + d), resid(theta - d)
            if rp is None or rm is None:
                return dict(ok=False, why="fit left the plausible range", free=True)
            J[:, j] = (rp - rm) / (2 * steps[j])
        Wv = np.r_[w, np.ones(NPRI)]
        JW = J * Wv[:, None]; A_ = J.T @ JW; g = JW.T @ r
        improved = False; delta = np.zeros(NP)
        for _ in range(8):
            try:
                delta = -np.linalg.solve(A_ + lam * np.diag(np.diag(A_) + 1e-9), g)
            except np.linalg.LinAlgError:
                break
            cand = theta + delta; rc = resid(cand)
            if rc is not None:
                wc = _cauchy_w(rc[:2 * n_t]); cc = _cost(rc, wc, NPRI)
                if cc < cost:
                    theta, r, w, cost = cand, rc, wc, cc; lam = max(lam / 3, 1e-7); improved = True
                    break
            lam *= 4
        if not improved or np.abs(delta).max() < 1e-7:
            break
    spin = (theta[6], theta[7], theta[8])
    pos, vel = integrate(theta[0:3], theta[3:6], spin, t_a + 1e-9, np.r_[times, t_contact])
    v0 = vel[-1]; speed = float(np.linalg.norm(v0)); p0 = pos[-1]
    tt_ = np.linspace(t_contact, t_a, 24)
    path, _ = integrate(theta[0:3], theta[3:6], spin, t_a + 1e-9, tt_)
    speed_sd = None
    try:
        dof = max(1, len(r) - NP); s2 = max(1.0, cost / dof)
        cov = np.linalg.inv(A_ + 1e-9 * np.eye(NP)) * s2
        gr = np.zeros(NP)
        for j in range(NP):
            d = np.zeros(NP); d[j] = steps[j]; thp = theta + d
            _, vv = integrate(thp[0:3], thp[3:6], (thp[6], thp[7], thp[8]), t_a + 1e-9, np.array([t_contact]))
            gr[j] = (np.linalg.norm(vv[0]) - speed) / steps[j]
        speed_sd = float(np.sqrt(max(0.0, gr @ cov @ gr)))
    except np.linalg.LinAlgError:
        pass
    profile = sorted([(float(t_contact), speed)] + [(float(ti), float(np.linalg.norm(vi))) for ti, vi in zip(times, vel[:-1])])
    err = np.linalg.norm(cam.project(pos[:-1]) - uv, axis=1); inl = err < 4 * sigma_px
    rms = float(np.sqrt(np.mean(err[inl] ** 2))) if inl.any() else float("inf")
    net_z = None
    if t_cr is not None:
        pc, _ = integrate(theta[0:3], theta[3:6], spin, t_a + 1e-9, np.array([t_cr])); net_z = float(pc[0, 2])
    vmean = float(np.mean([v for _, v in profile])); az = float(-C0 * theta[6] * vmean ** 2)
    why = []
    if inl.sum() < max(5, 0.6 * len(err)):
        why.append("too many frames off the fitted flight")
    if rms > 3.5 * sigma_px:
        why.append(f"fits the ball badly ({rms:.1f} px)")
    if not 1.0 <= speed <= 40.0:
        why.append(f"speed {speed:.1f} m/s")
    if not (-0.6 <= theta[6] <= 0.8 and abs(theta[7]) <= 0.6 and 0.2 <= theta[8] <= 0.9):
        why.append("implausible spin or drag")
    if speed_sd is None or speed_sd > max_sd_frac * speed:
        why.append("speed not pinned down without a bounce")
    if y_prior and abs(p0[1] - y_prior[0]) > 2.5 * y_prior[1]:
        why.append("contact out of the hitter's reach")
    return dict(ok=not why, why="; ".join(why), free=True, speed=speed, speed_sd=speed_sd, v0=[float(x) for x in v0], rms_px=rms, n=int(len(times)),
                n_in=int(inl.sum()), net_z=net_z, az=az, cl_top=float(theta[6]), cl_side=float(theta[7]), cd=float(theta[8]),
                t_bounce=t_a, bounce=None, v_bounce=None, speed_bounce=float(np.linalg.norm(theta[3:6])), p0=[float(x) for x in p0],
                path=[[round(float(c), 3) for c in q] for q in path],
                over_net_cm=None if net_z is None else round(100 * (net_z - R_BALL - NET_H), 1), profile=profile,
                anchor=dict(t=t_a + 1e-9, p=[float(x) for x in theta[0:3]], v=[float(x) for x in theta[3:6]]))


def state_at(f, ts):
    """Positions and velocities of a fitted flight (fit or fit_free result) at times ts, run back from its anchor."""
    a = f["anchor"]
    return integrate(np.array(a["p"]), np.array(a["v"]), (f["cl_top"], f["cl_side"], f["cd"]), a["t"], np.asarray(ts, float))


def _cauchy_w(r, c=2.0):
    """Cauchy weights on the pixel residuals (x and y of one frame share their weight)."""
    e = np.hypot(r[0::2], r[1::2])
    w = 1.0 / (1.0 + (e / c) ** 2)
    return np.repeat(w, 2)


def _cost(r, w, npri=4):
    return float(np.sum(w * r[:-npri] ** 2) + np.sum(r[-npri:] ** 2))


def _on_plane_y(cam, uv, y):
    C, d = cam.ray([uv])
    d = d[0]
    if abs(d[1]) < 1e-9:
        return np.array([np.nan] * 3)
    s = (y - C[1]) / d[1]
    return C + d * s


def _meet(track, t_rev, fps, k=4):
    """Sub-frame racket contact: lines through the last k tracked positions before the reversal and the first k after it (picture
    coordinates against time); the contact is the time the two lines come closest, within 1.5 frames of the reversal frame."""
    t_all = track[:, 1]; ok = ~np.isnan(track[:, 2])
    before = np.where(ok & (t_all <= t_rev + 1e-6) & (t_all > t_rev - 6 / fps))[0][-k:]
    after = np.where(ok & (t_all > t_rev + 1e-6) & (t_all < t_rev + 6 / fps))[0][:k]
    if len(before) < 3 or len(after) < 3:
        return t_rev
    def line(idx):
        return [np.polyfit(t_all[idx], track[idx, c], 1) for c in (2, 3)]
    A, B = line(before), line(after)
    ts = np.linspace(t_rev - 1.5 / fps, t_rev + 1.5 / fps, 61)
    d = np.hypot(np.polyval(A[0], ts) - np.polyval(B[0], ts), np.polyval(A[1], ts) - np.polyval(B[1], ts))
    return float(ts[int(np.argmin(d))])


JUMP_PX = 70.0            # px per frame: faster than any ball crosses the picture here, so a step this big is the tracker holding something else


def _bridge(track, t_out, t_bounce_in, out_dir, fps, k=4, max_gap=0.35):
    """The racket contact when the ball was hidden around it (behind the player's body or the racket, or the tracker held something else
    for a few frames): where the incoming path after its bounce on the hitter's side and the outgoing path, each extended across the gap
    as a straight line along the picture's x, meet. None when there is no clean incoming path to extend."""
    t_all, u_all, v_all = track[:, 1], track[:, 2], track[:, 3]
    ok = ~np.isnan(u_all)
    after = np.where(ok & (t_all >= t_out - 1e-6) & (t_all < t_out + 8 / fps))[0][:k]
    if len(after) < 3 or t_bounce_in is None:
        return None
    seq = []
    for i in np.where(ok & (t_all > t_bounce_in) & (t_all < t_out - 0.5 / fps))[0]:
        if seq:
            j = seq[-1]; step = max(1, i - j); du = u_all[i] - u_all[j]
            if abs(du) > JUMP_PX * step or abs(v_all[i] - v_all[j]) > JUMP_PX * step:
                break                                                     # the track left the ball
            if np.sign(du) == out_dir and abs(du) > 1.0:
                break                                                     # already going back: not the incoming ball any more
        seq.append(i)
    if len(seq) < 3:
        return None
    before = np.array(seq[-k:])
    if t_all[after[0]] - t_all[before[-1]] > max_gap:
        return None
    A = np.polyfit(t_all[before], u_all[before], 1); B = np.polyfit(t_all[after], u_all[after], 1)
    if abs(A[0] - B[0]) < 1e-6 or np.sign(A[0]) == np.sign(B[0]):
        return None
    t = (B[1] - A[1]) / (A[0] - B[0])
    return float(np.clip(t, t_all[before[-1]], t_all[after[0]]))


def shots(points, events, track, fps, table):
    """Every shot that can be measured: (point id, shot number, hitter end, crossing, contact time, bounce event, frames). A shot runs
    from the racket contact (the ball's last reversal along the table before its crossing, or the serve's bounce on the server's side)
    to its bounce on the other side."""
    out = []
    t_all = track[:, 1]; x_all = track[:, 2]
    bounces = sorted([e for e in events if e["kind"] == "bounce"], key=lambda e: e["t"])
    for p in points:
        evs = sorted([e for e in p.get("events", []) if e.get("kind") in ("net", "bounce")], key=lambda e: e["t"])
        crossings = [e for e in evs if e["kind"] == "net"]
        for k, c in enumerate(crossings, start=1):
            if c.get("implied") or "x_px" not in c:
                continue
            land = next((b for b in bounces if 0.02 < b["t"] - c["t"] <= 1.0 and b.get("side") == c.get("side")), None)
            t_end = None
            if land is None:                                              # long, wide, or its landing not seen: measured without a bounce
                t_end = _track_end(track, c["t"], fps)
                if t_end is None or t_end - c["t"] < 3.0 / fps:
                    continue
            # walk back from the crossing while the ball keeps moving the same way along the picture's x
            fc = int(round(c["t"] * fps)); f = fc; direction = None; last_seen = fc; contact = None
            own = [b for b in bounces if c["t"] - 0.9 < b["t"] < c["t"] and b.get("side") != c.get("side")]
            t_floor = own[-1]["t"] if own else c["t"] - 0.9               # a serve starts at its own-side bounce
            prev = None; broke = False
            while f > 0 and (fc - f) < int(0.9 * fps):
                if f < len(track) and not np.isnan(x_all[f]):
                    if prev is not None:
                        dx = x_all[prev] - x_all[f]
                        if abs(dx) > JUMP_PX * (prev - f) or abs(track[prev, 3] - track[f, 3]) > JUMP_PX * (prev - f):
                            broke = True; break                           # the track jumped to something else: the ball is hidden here
                        if abs(dx) > 1.0:
                            sgn = np.sign(dx)
                            if direction is None:
                                direction = sgn
                            elif sgn != direction:
                                contact = t_all[f]; break
                    if last_seen - f > 6:                                 # a long gap: stop here
                        broke = True; break
                    prev = f; last_seen = f
                if t_all[f] <= t_floor:
                    contact = t_floor; break
                f -= 1
            t_first = None                                               # the first frame the outgoing ball is seen, when the contact was hidden
            if contact is None and k == 1 and own:                       # a serve lost for a few frames after its own bounce: it starts there
                t_first = t_all[last_seen]; contact = t_floor; src = "bounce"
            elif contact is None:
                t_first = t_all[last_seen]
                # the incoming ball from its bounce on the hitter's side, or, when that bounce was not seen, from the net crossing that
                # brought it (it runs on towards the hitter along the picture either way)
                prev_cross = crossings[k - 2]["t"] if k >= 2 else None
                tb_in = own[-1]["t"] if own else prev_cross
                bridged = _bridge(track, t_first, tb_in, direction or 0.0, fps) if broke and direction else None
                contact = bridged if bridged is not None else t_first
                src = "bridged" if bridged is not None else ("late" if broke else "seen")
            elif contact > t_floor + 1e-6:                               # a racket contact: refine it to where the two paths meet
                contact = _meet(track, contact, fps); src = "seen"
            else:
                src = "bounce"                                            # a serve: its own-side bounce
            t_stop = (land["t"] - 0.5 / fps) if land is not None else t_end + 1e-6
            if t_first is not None:
                sel = (t_all >= t_first - 1e-6) & (t_all < t_stop) & ~np.isnan(x_all)
            else:
                sel = (t_all > contact + 0.5 / fps) & (t_all < t_stop) & ~np.isnan(x_all)
            idx = np.where(sel)[0]
            if len(idx) < 5:
                continue
            out.append(dict(point=p["id"], shot=k, crossing=c, bounce=land, contact=float(contact), frames=idx, serve=k == 1 and bool(own),
                            contact_src=src, hidden=None if t_first is None else round(float(t_first - contact), 4), free=land is None,
                            hitter_end="near" if c.get("side") == "far" else "far"))
    return out


def _track_end(track, t0, fps, max_s=1.0, max_gap=4):
    """The last frame of the ball's clean track after t0 (no jumps to another object, no gap longer than max_gap frames)."""
    t_all, u_all, v_all = track[:, 1], track[:, 2], track[:, 3]
    f = int(round(t0 * fps)); last = None; prev = None
    while f < len(track) and t_all[f] <= t0 + max_s:
        if not np.isnan(u_all[f]):
            if prev is not None:
                step = f - prev
                if step > max_gap or abs(u_all[f] - u_all[prev]) > JUMP_PX * step or abs(v_all[f] - v_all[prev]) > JUMP_PX * step:
                    break
            prev = f; last = f
        elif prev is not None and f - prev > max_gap:
            break
        f += 1
    return None if last is None else float(t_all[last])


def net_strikes(points, events, track, fps):
    """Returns that went into the net. In a point nobody returned, after the ball's last bounce on the loser's half, his racket sends it
    back towards the net (the track turns round and heads for the net line) but it never crosses. Same dicts as shots(), crossing None."""
    out = []
    t_all, u_all, v_all = track[:, 1], track[:, 2], track[:, 3]
    for p in points:
        if p.get("ending") != "not_returned":
            continue
        evs = sorted([e for e in p.get("events", []) if e.get("kind") in ("net", "bounce")], key=lambda e: e["t"])
        crossings = [e for e in evs if e["kind"] == "net" and "x_px" in e and not e.get("implied")]
        if not crossings:
            continue
        c = crossings[-1]
        n_all = sum(e["kind"] == "net" for e in evs)                     # shots are numbered over every crossing, inferred ones too
        b = next((e for e in evs if e["kind"] == "bounce" and e["t"] > c["t"] and e.get("side") == c.get("side")), None)
        if b is None:
            continue
        net_u = float(c["x_px"])                                         # where the ball crossed: the net line in the picture
        f = int(round(b["t"] * fps)); seq = []
        while f < len(track) and t_all[f] <= b["t"] + 1.2:
            if not np.isnan(u_all[f]):
                if seq and (f - seq[-1] > 4 or abs(u_all[f] - u_all[seq[-1]]) > JUMP_PX * (f - seq[-1])):
                    break
                seq.append(f)
            f += 1
        if len(seq) < 8:
            continue
        u = u_all[seq]; toward = np.sign(net_u - u[0])                   # after the bounce it moves away from the net ...
        k_rev = None
        for i in range(2, len(seq) - 5):
            steps = np.diff(u[i:i + 6])
            if np.all(np.sign(steps) == toward) and np.all(np.abs(steps) > 1.0) and np.sign(u[i] - u[i - 2]) != toward:
                k_rev = i; break                                          # ... until his racket turns it round towards the net
        if k_rev is None:
            continue
        contact = _meet(track, float(t_all[seq[k_rev]]), fps)
        run = [seq[k_rev + 1]]
        for j in seq[k_rev + 2:]:
            if np.sign(u_all[j] - u_all[run[-1]]) != toward or np.sign(net_u - u_all[j]) != toward:
                break                                                     # turned again (hit the net) or reached the net line
            run.append(j)
        if len(run) < 6:
            continue
        at_net = abs(net_u - u_all[run[-1]]) < 30.0
        out.append(dict(point=p["id"], shot=n_all + 1, crossing=None, bounce=None, contact=float(contact), frames=np.array(run),
                        serve=False, contact_src="seen", hidden=None, free=True, net=True, net_end=bool(at_net), hitter_end=b["side"]))
    return out


def measure(points, events, track, fps, cam, y_prior_fn=None, only_free=False):
    """Fit every measurable shot. y_prior_fn(t_contact, hitter_end) -> (mu, sd) or None: where across the table the hitter stood.
    Returns a list of dicts (shot info + fit), in time order."""
    res = []
    for s in shots(points, events, track, fps, None) + net_strikes(points, events, track, fps):
        if only_free and not s.get("free"):
            continue
        idx = s["frames"]
        hitter_end = s["hitter_end"]
        yp = y_prior_fn(s["contact"], hitter_end) if (y_prior_fn and not s["serve"]) else None
        if s.get("free"):                                                # no bounce on the table to hang the flight on
            f = fit_free(cam, track[idx, 1], track[idx, 2:4], fps, s["contact"], t_cross=s["crossing"]["t"] if s["crossing"] else None,
                         y_prior=yp, x_end=NET_X if s.get("net_end") else None)
        else:
            f = fit(cam, track[idx, 1], track[idx, 2:4], s["bounce"]["t"], (s["bounce"]["x_px"], s["bounce"]["y_px"]), fps, s["contact"], y_prior=yp)
            f["free"] = False
            if not f["ok"]:                                              # the bounce did not fit (a wrong bounce event, say): try without it
                f2 = fit_free(cam, track[idx, 1], track[idx, 2:4], fps, s["contact"], t_cross=s["crossing"]["t"], y_prior=yp)
                if f2["ok"]:
                    f = f2
        res.append(dict(point=s["point"], shot=s["shot"], serve=s["serve"], t_contact=s["contact"],
                        t_cross=s["crossing"]["t"] if s["crossing"] else None, contact_src=s.get("contact_src"), hidden=s.get("hidden"),
                        t_bounce_event=s["bounce"]["t"] if s["bounce"] else None, hitter_end=hitter_end, net=bool(s.get("net")),
                        to_side=s["crossing"].get("side") if s["crossing"] else ("far" if hitter_end == "near" else "near"), **f))
    return sorted(res, key=lambda r: r["t_contact"])


class Approx(float):
    """A speed measured without a bounce to hang its flight on (typically within 8%, not 3%): the gauge shows it as about (~)."""
    approx = True


DASH = "-"                  # the gauge's reading for a shot that was played but not measured
READ_AFTER_CONTACT_S = 0.1  # a shot's number goes up this long after its racket contact, while the ball is on its way
CONTACT_BEFORE_CROSS_S = 0.3  # an unmeasured shot's contact, when only its net crossing is known (contact to net: 0.2 to 0.4 s)


class Speeds:
    """The measured shots as the clips need them: the speed of the shot being played (reading), the last shot's speed off the racket
    once it has bounced (last), and the ball's fitted speed at any moment inside a fitted flight (the gauge's needle).
    points = the points the clip shows; reading() needs them, because every shot played is a crossing of the net there."""

    def __init__(self, shots, points=None):
        self.shots = sorted([s for s in shots if s.get("ok")], key=lambda s: s["t_bounce"])
        self._bt = [s["t_bounce"] for s in self.shots]
        self.slots = self.readings(points) if points is not None else []
        self._st = [a for a, _ in self.slots]

    @staticmethod
    def value(sh):
        """What the gauge prints for one fitted shot: its speed (Approx when it is an estimate), or DASH when it has none to trust."""
        if sh is None or not sh.get("ok") or sh.get("speed") is None:
            return DASH
        if sh.get("contact_src") == "late" and not sh.get("approx"):     # hidden and not pinned down: reads low
            return DASH
        return Approx(sh["speed"]) if (sh.get("free") or sh.get("approx")) else sh["speed"]

    def readings(self, points):
        """(time, value) each time the gauge's number changes: one per shot played (every crossing of the net, inferred ones
        included, and every return into the net), READ_AFTER_CONTACT_S after its racket contact. So a hard hit shows its own number
        while it is in the air, and a shot with no measured flight reads DASH instead of leaving the slower shot before it up
        (2026-09-28: an unmeasured smash read as the previous block's speed)."""
        by_key = {(s.get("point"), s.get("shot")): s for s in self.shots}
        out = []
        for p in points:
            cross = sorted((e for e in p.get("events", []) if e.get("kind") == "net"), key=lambda e: e["t"])
            mine = []
            for k, c in enumerate(cross, start=1):
                sh = by_key.get((p["id"], k))
                t_c = sh["t_contact"] if sh is not None else c["t"] - CONTACT_BEFORE_CROSS_S
                mine.append([t_c + READ_AFTER_CONTACT_S, self.value(sh)])
            for (pid, k), sh in sorted(by_key.items(), key=lambda kv: kv[1]["t_contact"]):
                if pid == p["id"] and sh.get("net") and k > len(cross):  # a return into the net: a stroke with no crossing
                    mine.append([sh["t_contact"] + READ_AFTER_CONTACT_S, self.value(sh)])
            for i in range(1, len(mine)):                                # an inferred crossing's time is a guess: keep the order
                mine[i][0] = max(mine[i][0], mine[i - 1][0] + 0.15)
            out += [tuple(m) for m in mine]
        return sorted(out, key=lambda r: r[0])

    def reading(self, t):
        """The gauge's number at t: the speed of the shot being played (a float, Approx for an estimate), DASH when that shot was not
        measured, None before the first shot."""
        i = bisect.bisect_right(self._st, t) - 1
        return self.slots[i][1] if i >= 0 else None

    def last(self, t):
        i = bisect.bisect_right(self._bt, t) - 1
        sh = self.shots[i] if i >= 0 else None
        if sh is None or (sh.get("contact_src") == "late" and not sh.get("approx")):   # hidden and not pinned down: reads low
            return None
        return Approx(sh["speed"]) if (sh.get("free") or sh.get("approx")) else sh["speed"]

    def live(self, t):
        i = bisect.bisect_right(self._bt, t)
        for s in self.shots[i:i + 2]:
            prof = s.get("profile") or []
            if prof and prof[0][0] <= t <= s["t_bounce"]:
                ts = [a for a, _ in prof]; j = min(bisect.bisect_left(ts, t), len(prof) - 1)
                return prof[j][1]
        return None
