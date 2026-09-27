"""Kinks in the ball track -> bounce / hit events; runs of ball presence -> rallies.

A ball in flight follows a smooth arc. Anything that is not smooth is a contact: the table or a racket.
The second difference of the track (pixel acceleration) spikes at a contact and is tiny in between, so
peaks of it are the events. Which kind: a kink located on the table surface is a bounce; anywhere else
it is a racket (or the floor). From a side camera a bounce is additionally the lowest point of the ball
in the image, which is used as a second check when the view is "side".
"""
import numpy as np
from .config import TABLE_LENGTH as L, TABLE_WIDTH as W, NET_X


def _runs(seen, max_gap):
    """Index ranges [s, e] of frames where the ball is seen with no gap longer than max_gap frames."""
    idx = np.where(seen)[0]
    if idx.size == 0:
        return []
    cuts = np.where(np.diff(idx) > max_gap)[0]
    starts = np.r_[idx[0], idx[cuts + 1]]
    ends = np.r_[idx[cuts], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def _vertex(xs, ys, li, K):
    """Contact point as the intersection of the straight paths before and after frame li.
    The centroid path through a bounce is a V; the interpolated centre frame cuts the corner."""
    ia = np.arange(li - K, li); ib = np.arange(li + 1, li + K + 1)
    A = np.polyfit(ia, xs[ia], 1), np.polyfit(ia, ys[ia], 1)
    B = np.polyfit(ib, xs[ib], 1), np.polyfit(ib, ys[ib], 1)
    # both paths parameterised by frame index; solve for the frame where they meet in x (or y if x is flat)
    dx, dy = A[0][0] - B[0][0], A[1][0] - B[1][0]
    if abs(dy) >= abs(dx):
        if abs(dy) < 1e-6:
            return float(xs[li]), float(ys[li])
        f = (B[1][1] - A[1][1]) / dy
    else:
        f = (B[0][1] - A[0][1]) / dx
    f = float(np.clip(f, li - 1.5, li + 1.5))
    return float(np.polyval(A[0], f)), float(np.polyval(A[1], f))


def _front_game(frame, track_id, track, table, cfg, fps):
    """True when, within cfg.cross_front_s of a crossing, the ball is seen in at least cfg.cross_front_min frames either down at this
    table's surface or lower (below its far edge line in the picture) or out past its ends (beyond the far corners by
    cfg.cross_front_margin_px), which is where this table's players hit it; also True when too few frames are tracked to tell. A game
    on a table further back crosses this net's line in the picture, but its ball stays above this table's far edge and between its ends
    (our 2026-09-25 recording, first ten minutes)."""
    w = int(round(cfg.cross_front_s * fps))
    # the crossing's own track only: when both games are live the tracker holds one ball at a time, and the other game's ball
    # being near this table in the same second says nothing about the ball that crossed
    sel = (track[:, 0] >= frame - w) & (track[:, 0] <= frame + w) & ~np.isnan(track[:, 2]) & (track[:, 4] == track_id)
    if sel.sum() < cfg.cross_min_seen:
        return True
    px, py = track[sel, 2], track[sel, 3]
    c = table.corners                                                     # side lines: corners 1 -> 4 (y = 0) and 2 -> 3 (y = W)
    a, b = (c[1], c[2]) if (c[1][1] + c[2][1]) < (c[0][1] + c[3][1]) else (c[0], c[3])     # the far one is higher in the picture
    (x2, y2), (x3, y3) = sorted([tuple(a), tuple(b)])
    far_y = y2 + (px - x2) * (y3 - y2) / (x3 - x2)
    m = cfg.cross_front_margin_px
    evidence = (py > far_y - cfg.cross_front_tol_px) | (px < x2 - m) | (px > x3 + m)
    return int(evidence.sum()) >= cfg.cross_front_min


def _real_crossing(frame, track_id, d_sign, track, n1, n2, ppm_net, cfg, fps):
    """False when the ball on the crossing's own track never gets cfg.net_touch_min_m clear of the net line on the side it came from
    (in the cfg.net_touch_s before) or on the side it went to (in the cfg.net_touch_s after): it touched the net and dropped back, or
    it sat on the net line jittering a pixel either side (our 2026-09-25 recording, points at 1:40 and 4:17). A side with fewer than
    cfg.net_touch_min_seen tracked frames is not judged. Other tracks are ignored: the tracker jumping to a player's shoe 900 px away
    is not the ball getting clear of the net."""
    w = int(round(cfg.net_touch_s * fps))
    same = (track[:, 4] == track_id) & ~np.isnan(track[:, 2])
    lim = cfg.net_touch_min_m * ppm_net
    for lo, hi, sgn in ((frame - w, frame - 1, -d_sign), (frame + 1, frame + w, d_sign)):
        sel = same & (track[:, 0] >= lo) & (track[:, 0] <= hi)
        if sel.sum() < cfg.net_touch_min_seen:
            continue
        x, y = track[sel, 2], track[sel, 3]
        d = ((x - n1[0]) * (n2[1] - n1[1]) - (y - n1[1]) * (n2[0] - n1[0])) / float(np.hypot(*(n2 - n1)))
        if float(np.max(d * sgn)) < lim:
            return False
    return True


def _on_table(table, pt, cfg):
    """Is a kink at image point pt on the table? Judged at the ball's centre and, with cfg.contact_shift, at the contact point one
    ball radius lower in the picture."""
    if table.inside(pt, cfg.table_margin_x, cfg.table_margin_y):
        return True
    if not getattr(cfg, "contact_shift", False):
        return False
    xm, ym = table.to_table([pt])[0]
    r = table.ball_radius_px((float(np.clip(xm, 0, L)), float(np.clip(ym, 0, W))))
    return table.inside((pt[0], pt[1] + r), cfg.table_margin_x, cfg.table_margin_y)


def detect_events(track, table, cfg, fps):
    """track: array of rows [frame, t, x, y, track_id]; x, y NaN when the ball was not seen.
    Track ids are ignored: there is one ball, so the series is one series, split only at long gaps."""
    frames, t, x, y = (track[:, i] for i in range(4))
    seen_all = ~np.isnan(x)
    K = max(2, int(round(cfg.kink_window_s * fps)))
    gap = max(2, int(round(cfg.event_min_gap_s * fps)))
    events = []
    for s, e in _runs(seen_all, int(cfg.max_gap_s * fps)):
        n = e - s + 1
        if n < 2 * K + 3:
            continue
        xs, ys, seen = x[s:e + 1].copy(), y[s:e + 1].copy(), seen_all[s:e + 1]
        ii = np.arange(n)
        xs[~seen] = np.interp(ii[~seen], ii[seen], xs[seen])   # bridge short gaps
        ys[~seen] = np.interp(ii[~seen], ii[seen], ys[seen])
        # velocity averaged over K frames before and after each frame; a contact is a jump between them
        vbx = (xs[K:n - K] - xs[:n - 2 * K]) / K;  vax = (xs[2 * K:] - xs[K:n - K]) / K
        vby = (ys[K:n - K] - ys[:n - 2 * K]) / K;  vay = (ys[2 * K:] - ys[K:n - K]) / K
        mag = np.hypot(vax - vbx, vay - vby)                       # mag[j] belongs to local frame j+K
        # window ends must be real detections; the centre may be bridged by its neighbours (the ball is
        # often lost for exactly the contact frame, when it merges with its own shadow on the table)
        c_ok = seen[K:n - K] | (seen[K - 1:n - K - 1] & seen[K + 1:n - K + 1])
        cs = np.r_[0, np.cumsum(seen)]
        before = cs[K:n - K] - cs[:n - 2 * K]                 # real detections in the K frames before
        after = cs[2 * K + 1:n + 1] - cs[K + 1:n - K + 1]      # ... and in the K frames after
        real = c_ok & (before >= max(1, K - 2)) & (after >= max(1, K - 2))   # at least one real frame per side
        if real.sum() < 4:
            continue
        base = float(np.median(mag[real]))
        last = -10 ** 9
        for j in range(1, mag.size - 1):
            if not real[j] or mag[j] < cfg.event_abs_thresh_px or mag[j] < mag[j - 1] or mag[j] < mag[j + 1]:
                continue
            li = j + K
            pt = (float(xs[li]), float(ys[li]))
            xm, ym = table.to_table([pt])[0]
            ppm = table.ppm_at((float(np.clip(xm, 0, L)), float(np.clip(ym, 0, W))))
            # threshold: a multiple of the segment's own noise, capped at a physically small velocity change
            if mag[j] < max(cfg.event_abs_thresh_px, min(cfg.event_rel_thresh * base, cfg.event_cap_m_s * ppm / fps)):
                continue
            if mag[j] > cfg.kink_max_m_s * ppm / fps:
                continue                                           # faster than any racket: the tracker jumped
            if j - last < gap:
                continue
            inside = _on_table(table, pt, cfg)
            kind = "bounce" if inside else "hit"
            if inside and table.view == "side":
                # a bounce is the lowest point: moving down the image before, up after
                if not (vby[j] > cfg.bounce_min_vy and vay[j] < -cfg.bounce_min_vy):
                    kind = "hit"
            if kind == "bounce" and mag[j] > cfg.bounce_max_m_s * ppm / fps:
                kind = "hit"                                       # too violent for the table: a racket at the end line
            if kind == "hit" and mag[j] < cfg.hit_min_m_s * ppm / fps:
                continue                                           # too gentle for a racket: noise
            last = j
            if kind == "bounce":
                vx = _vertex(xs, ys, li, K)                        # refine to the V's vertex ...
                step = max(1.0, float(np.hypot(vbx[j], vby[j])), float(np.hypot(vax[j], vay[j])))
                if np.hypot(vx[0] - pt[0], vx[1] - pt[1]) <= 1.5 * step and _on_table(table, vx, cfg):
                    pt = vx                                        # ... unless the refinement ran away
                xm, ym = table.to_table([pt])[0]
                if abs(xm - NET_X) < cfg.net_margin:               # the ball does not bounce on the net line
                    kind = "nethit"
            events.append(dict(frame=int(frames[s + li]), t=float(t[s + li]), kind=kind,
                               side="near" if xm < L / 2 else "far", x_px=pt[0], y_px=pt[1],
                               x_m=float(xm), y_m=float(ym), strength=float(mag[j]), track=int(track[s + li, 4])))
        # net crossings: the track passes the net line in the image (valid at any height, unlike the homography)
        n1, n2 = table.to_px([[NET_X, 0.0], [NET_X, 1.0]])
        d = ((xs - n1[0]) * (n2[1] - n1[1]) - (ys - n1[1]) * (n2[0] - n1[0])) / float(np.hypot(*(n2 - n1)))  # signed px distance to the net line
        crossed = []
        for li in range(1, n):
            if seen[li] and seen[li - 1] and d[li] * d[li - 1] < 0 and max(abs(d[li]), abs(d[li - 1])) <= cfg.cross_max_px:
                (xa, ya), (xb, yb) = table.to_table([(float(xs[li]), float(ys[li])), (float(xs[li - 1]), float(ys[li - 1]))])
                events.append(dict(frame=int(frames[s + li]), t=float(t[s + li]), kind="net",
                                   side="far" if xa > xb else "near",            # the side the ball crossed TO
                                   x_px=float(xs[li]), y_px=float(ys[li]),
                                   x_m=float(NET_X), y_m=float(ya), strength=0.0, track=int(track[s + li, 4])))
    n1, n2 = table.to_px([[NET_X, 0.0], [NET_X, 1.0]])
    ppm_net = table.ppm_at((NET_X, W / 2))
    kept = []
    for ev in events:
        if ev["kind"] == "net":
            # the side the ball crossed TO, as a sign of the distance to the net line (+1 = the side the formula calls positive)
            probe = table.to_px([[NET_X + (0.3 if ev["side"] == "far" else -0.3), W / 2]])[0]
            sgn = np.sign(((probe[0] - n1[0]) * (n2[1] - n1[1]) - (probe[1] - n1[1]) * (n2[0] - n1[0])))
            if cfg.net_touch_s > 0 and not _real_crossing(ev["frame"], ev["track"], sgn, track, n1, n2, ppm_net, cfg, fps):
                continue
            nb = next((b for b in events if b["kind"] == "bounce" and ev["t"] < b["t"] <= ev["t"] + cfg.net_touch_bounce_s), None)
            if nb is not None and nb["side"] != ev["side"] and abs(nb["x_m"] - NET_X) < cfg.net_touch_bounce_m:
                continue                       # the ball's next landing is on the side it came from, by the net: it hit the net and fell back
            if cfg.cross_front_s > 0 and not _front_game(ev["frame"], ev["track"], track, table, cfg, fps):
                continue
        kept.append(ev)
    events = kept
    # no racket is ever at the net: a "hit" right at a crossing is jitter from the ball passing the
    # umpire's desk and scoreboard, so drop it
    cross_t = sorted(ev["t"] for ev in events if ev["kind"] == "net")
    events = [ev for ev in events if not (ev["kind"] == "hit" and any(abs(ev["t"] - c) < 0.05 for c in cross_t))]
    # between two crossings the ball is hit exactly once (the return): keep only the strongest kink there.
    # Hits are over-detected (tracker jumps near players); this makes stroke counts mean something.
    bounds = [-1e9] + cross_t + [1e9]
    kept = [ev for ev in events if ev["kind"] != "hit"]
    for a, b in zip(bounds, bounds[1:]):
        hs = [ev for ev in events if ev["kind"] == "hit" and a <= ev["t"] < b]
        if hs:
            kept.append(max(hs, key=lambda ev: ev["strength"]))
    events = sorted(kept, key=lambda ev: ev["t"])
    return events


def segment_rallies(track, events, cfg, fps):
    t, x = track[:, 1], track[:, 2]
    seen = np.where(~np.isnan(x))[0]
    if seen.size == 0:
        return []
    gap = int(cfg.rally_gap_s * fps)
    breaks = np.where(np.diff(seen) > gap)[0]
    starts = np.r_[seen[0], seen[breaks + 1]]
    ends = np.r_[seen[breaks], seen[-1]]
    rallies = []
    for s, e in zip(starts, ends):
        t0, t1 = float(t[s]), float(t[e])
        if t1 - t0 < cfg.rally_min_s:
            continue
        evs = [ev for ev in events if t0 <= ev["t"] <= t1]
        bounces = [ev for ev in evs if ev["kind"] == "bounce"]
        hits = [ev for ev in evs if ev["kind"] == "hit"]
        first = bounces[0] if bounces else (hits[0] if hits else None)
        rallies.append(dict(id=len(rallies) + 1, start_t=t0, end_t=t1, duration=t1 - t0,
                            serve_side=first["side"] if first else None,
                            n_bounces=len(bounces), n_hits=len(hits),
                            coverage=float(np.mean(~np.isnan(x[s:e + 1]))), events=evs, winner=None))
    return rallies


def segment_points(events, cfg, track=None):
    """Points from the sequence of bounces and net crossings, as a small state machine.

    A point starts at the first net crossing after the previous point's dead time (that crossing is the
    serve going over; the server is the side it came FROM). After every crossing the ball is on one side;
    the point ends at the first of:
      * a second bounce on that side before the ball crosses back        -> double bounce
      * one bounce and no crossing back within cfg.return_timeout_s      -> landed, never returned
      * no bounce and no crossing back within cfg.long_timeout_s         -> never landed, went long
    otherwise the next crossing continues the rally. Crossings within cfg.dead_time_s after an ending are
    the loser knocking the ball back to the server, and are ignored.
    """
    evs = [e for e in events if e["kind"] in ("bounce", "net")]
    evs.sort(key=lambda e: e["t"])
    cross_idx = [i for i, e in enumerate(evs) if e["kind"] == "net"]
    points, dead_until = [], -1e9
    ci = 0
    while ci < len(cross_idx):
        i = cross_idx[ci]
        if evs[i]["t"] < dead_until:
            ci += 1
            continue
        t_first, k = evs[i]["t"], ci
        end_t = None
        while end_t is None:
            last = evs[cross_idx[k]]
            nxt_i = cross_idx[k + 1] if k + 1 < len(cross_idx) else len(evs)
            between = [e for e in evs[cross_idx[k] + 1:nxt_i] if e["kind"] == "bounce"]
            nxt_t = evs[nxt_i]["t"] if nxt_i < len(evs) else 1e9
            # a landing follows its crossing within long_timeout_s; anything later is the next point's business
            b1 = between[0] if between and between[0]["t"] - last["t"] <= cfg.long_timeout_s else None
            b2 = between[1] if b1 is not None and len(between) > 1 else None
            if b1 is not None and b2 is not None and b2["t"] - b1["t"] <= cfg.return_timeout_s:
                end_t, ending = b2["t"] + 0.2, "double_bounce"     # the two bounces are < 1.5 s apart
            elif b1 is not None and nxt_t - b1["t"] > cfg.return_timeout_s:
                end_t, ending = b1["t"] + cfg.return_timeout_s, "not_returned"   # landed, never came back
            elif b1 is None and nxt_t - last["t"] > cfg.long_timeout_s:
                end_t, ending = last["t"] + cfg.long_timeout_s, "long"           # never landed: long or wide
            else:
                k += 1                                             # the rally continues with the next crossing
        pre = [e for e in evs if e["kind"] == "bounce" and max(dead_until, t_first - cfg.serve_lead_s) <= e["t"] < t_first]
        t0 = pre[0]["t"] if pre else t_first - 0.3
        pevs = [e for e in events if t0 <= e["t"] <= end_t]
        # landing of shot k = first bounce after the k-th crossing (shot 1 = serve, 2 = receive, 3 = third ball)
        pcross = [e for e in pevs if e["kind"] == "net"]
        landings = []
        for si, c in enumerate(pcross, start=1):
            nxt = pcross[si]["t"] if si < len(pcross) else end_t
            land = next((e for e in pevs if e["kind"] == "bounce" and c["t"] < e["t"] <= nxt
                         and e["t"] - c["t"] <= cfg.long_timeout_s), None)
            if land:
                landings.append(dict(shot=si, x_m=land["x_m"], y_m=land["y_m"], side=land["side"], t=land["t"]))
        cov = 1.0
        if track is not None:
            sel = (track[:, 1] >= t0) & (track[:, 1] <= end_t)
            cov = float(np.mean(~np.isnan(track[sel, 2]))) if sel.any() else 0.0
        points.append(dict(id=len(points) + 1, start_t=float(t0), end_t=float(end_t), duration=float(end_t - t0),
                           serve_side="near" if evs[i]["side"] == "far" else "far",
                           n_bounces=sum(e["kind"] == "bounce" for e in pevs), n_hits=sum(e["kind"] == "hit" for e in pevs),
                           n_crossings=k - ci + 1, coverage=cov, events=pevs, winner=None,
                           ending=ending, landings=landings))
        dead_until = end_t + cfg.dead_time_s
        ci = k + 1
        while ci < len(cross_idx) and evs[cross_idx[ci]]["t"] < dead_until:
            ci += 1
    return points
