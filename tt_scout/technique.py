"""Every shot's technique, measured: what a coach watching casual players would look at, as numbers, for players trying to improve.

Per shot, where the recording allows it:
  speed      m/s as the ball left the racket (flight.py: the flight fitted in 3D through the calibrated camera)
  spin       the flight's extra up-or-down acceleration (m/s2): clearly negative = the ball dipped, as topspin makes it
  over_net   cm between the ball and the top of the net as it crossed
  timing     s from the top of the incoming ball's bounce to the racket contact (0 = at the top; positive = later, as it drops)
  knee, lean the hitter's knee angle and forward trunk lean at contact (pose.py; angles as the camera sees them). knee is the ONE
             gated quantity once gate_knees() has run: the camera-near leg on a rally forehand seen in profile, else None; knee_raw
             keeps the ungated angle of the camera-near leg, and leg says which leg it was
  stance     m the hitter's feet stood behind his end line at contact (ankles taken down to the floor through the camera)
  depth      m from the net to where the shot landed (1.37 = on the end line)
  outcome    "winner" / "missed" for the stroke that decided the point, else "" (see match_stats.decisive)

    shots = measure(points, events, track, fps, cam, assigned)    # assigned = pose.assign(...) after clean_legs, or None
    gate_knees(shots, strokes, cam)                               # strokes = body3d.strokes_from(...), after body3d.attach
"""
import numpy as np
from .config import NET_X, TABLE_LENGTH as L
from . import flight
from .match_stats import hitter, decisive

ANKLE_Z = -0.76 + 0.08          # an ankle joint is about 8 cm above the floor; the floor is 76 cm below the table's playing surface
SD_MAX = 0.10                   # a speed counts as measured only when its fit's own uncertainty is within 10% of it; else it is an estimate (~)
HIDDEN_MAX_F = 4                # a serve lost for more than 4 frames after its own bounce: its speed is taken where it is seen again (~)


def contact_timing(track, bounces, t_contact, hitter_side, fps):
    """Seconds from the top of the incoming ball's bounce (on the hitter's half) to the racket contact, or None when the ball was not
    seen well enough between that bounce and the contact."""
    inc = [b for b in bounces if t_contact - 0.8 < b["t"] < t_contact - 0.03 and b.get("side") == hitter_side]
    if not inc:
        return None
    tb = inc[-1]["t"]
    sel = (track[:, 1] > tb + 0.5 / fps) & (track[:, 1] < t_contact + 0.5 / fps) & ~np.isnan(track[:, 3])
    if sel.sum() < 4:
        return None
    yy, tt = track[sel, 3], track[sel, 1]
    tpk = tt[np.argmin(yy)]                                             # highest point in the picture = the top of the bounce
    if tpk >= tt[-1] - 0.5 / fps and tt[-1] < t_contact - 1.5 / fps:
        return None                                                     # it was lost while still rising: the top is unknown
    return float(t_contact - tpk)


def stance(cam, skel, end):
    """Metres the player's feet are behind his end line at contact (negative = level with the table), from the ankles."""
    from .pose import J, MIN_C
    if skel is None:
        return None
    ank = [skel[J[n], :2] for n in ("lank", "rank") if skel[J[n], 2] >= MIN_C]
    if not ank:
        return None
    P = cam.on_plane_z(np.array(ank), ANKLE_Z)
    x = float(np.mean(P[:, 0]))
    return -x if end == "near" else x - L


def _feet(cam, skel):
    """The hitter's two ankles on the floor at contact, table coordinates (m), for drawing where he stood."""
    from .pose import J, MIN_C
    if skel is None:
        return None
    ank = [skel[J[n], :2] for n in ("lank", "rank") if skel[J[n], 2] >= MIN_C]
    if not ank:
        return None
    P = cam.on_plane_z(np.array(ank), ANKLE_Z)
    return [[round(float(x), 3), round(float(y), 3)] for x, y, _ in P]


def wrist_contact(fit, assigned, end, fps, cam, t_first, window=0.35):
    """When the hit was hidden and the two paths could not be joined: the moment the fitted flight, run back from its first seen
    frame, passes nearest the hitter's wrist in the picture (2D skeleton), accepted only within a racket's reach (0.9 of his
    torso's length in the picture). Returns (t, px) or None."""
    from . import pose as P
    if not assigned or not fit or not fit.get("ok") or not fit.get("anchor"):
        return None
    ts = np.arange(t_first - window, t_first + 0.25 / fps, 1 / fps)
    pos, _ = flight.state_at(fit, ts)
    uv = cam.project(pos)
    best = None
    for t, q in zip(ts, uv):
        a = P.skeleton_near(assigned, t, end, fps, reach=1)
        if a is None:
            continue
        torso = [a[P.J[n], :2] for n in ("neck", "root") if a[P.J[n], 2] >= P.MIN_C]
        tor = float(np.linalg.norm(torso[0] - torso[1])) if len(torso) == 2 else 120.0
        for w in ("lwri", "rwri"):
            if a[P.J[w], 2] < P.MIN_C:
                continue
            d = float(np.hypot(*(q - a[P.J[w], :2])))
            if best is None or d < best[1]:
                best = (float(t), d, tor)
    if best is None or best[1] > 0.9 * best[2]:
        return None
    return best[0], best[1]


def _incoming(prev, r):
    """Where and when the ball bounced on the hitter's half before this contact (the previous shot's fitted bounce), for drawing the
    ball rising into the racket: dict(bounce=[x, y], t=seconds before contact), or None."""
    if not (prev and prev.get("ok") and prev.get("bounce") and r and r.get("t_contact") is not None):
        return None                                                     # the shot before had no bounce on this side (it went long)
    dt = float(r["t_contact"]) - float(prev["t_bounce"])
    if not 0.05 < dt < 0.9:
        return None
    return dict(bounce=[round(float(v), 3) for v in prev["bounce"]], dt=round(dt, 4))


def measure(points, events, track, fps, cam, assigned=None):
    from . import pose as pose_mod
    def y_prior(t_c, end):                                              # the ball at contact is within arm's reach of the hitter's feet
        from .pose import J, MIN_C
        if not assigned:
            return None
        sk = pose_mod.skeleton_near(assigned, t_c, end, fps, reach=4)
        if sk is None:
            return None
        ank = [sk[J[n], :2] for n in ("lank", "rank") if sk[J[n], 2] >= MIN_C]
        if not ank:
            return None
        return (float(np.mean(cam.on_plane_z(np.array(ank), ANKLE_Z)[:, 1])), 0.35)
    fits = {(r["point"], r["shot"]): r for r in flight.measure(points, events, track, fps, cam, y_prior_fn=y_prior)}
    # without a bounce the flight's depth leans on the hitter's position, and the ball is met to his forehand side, not over his feet:
    # measure each end's offset (contact across the table minus the feet) on the flights that did bounce, then refit the others with it
    offs = {}
    for r in fits.values():
        if r.get("ok") and not r.get("free") and not r.get("serve") and r.get("contact_src") in ("seen", "bridged"):
            yp = y_prior(r["t_contact"], r["hitter_end"])
            if yp is not None:
                offs.setdefault(r["hitter_end"], []).append(r["p0"][1] - yp[0])
    offs = {e: (float(np.median(v)), max(0.3, 1.4826 * float(np.median(np.abs(np.array(v) - np.median(v)))))) for e, v in offs.items() if len(v) >= 8}
    if offs and any(r.get("free") for r in fits.values()):
        def y_prior2(t_c, end):
            yp = y_prior(t_c, end)
            if yp is None or end not in offs:
                return yp
            return (yp[0] + offs[end][0], offs[end][1])
        for r in flight.measure(points, events, track, fps, cam, y_prior_fn=y_prior2, only_free=True):
            fits[(r["point"], r["shot"])] = r
    for r in fits.values():                                             # a hit hidden and not bridged: time it by the hitter's wrist
        if r.get("ok") and r.get("contact_src") == "late":
            got = wrist_contact(r, assigned, r["hitter_end"], fps, cam, r["t_contact"])
            if got is None:
                continue
            t_c = got[0]; pos, vel = flight.state_at(r, [t_c]); sp = float(np.linalg.norm(vel[0]))
            if r.get("speed_sd"):
                r["speed_sd"] = r["speed_sd"] * sp / max(r["speed"], 1e-6)
            pp, _ = flight.state_at(r, np.linspace(t_c, r["anchor"]["t"] - 1e-4, 24))
            r.update(t_contact=t_c, speed=sp, p0=[float(x) for x in pos[0]], contact_src="wrist",
                     path=[[round(float(c), 3) for c in q] for q in pp],
                     profile=sorted([(t_c, sp)] + [x for x in (r.get("profile") or []) if x[0] > t_c]))
    for r in fits.values():                                             # which speeds are measurements and which are estimates
        if not r.get("ok") or r.get("speed") is None:
            continue
        hid = r.get("hidden") or 0.0
        if r.get("contact_src") == "bounce" and hid > HIDDEN_MAX_F / fps:
            # a serve lost after its own bounce: run back over the whole gap, air drag inflates the speed (a 0.5 s gap read 103 km/h
            # on a casual serve), so take it where the ball is seen again, a few % under the true value, and call it an estimate
            _, vel = flight.state_at(r, [r["t_contact"] + hid]); sp = float(np.linalg.norm(vel[0]))
            if r.get("speed_sd"):
                r["speed_sd"] = r["speed_sd"] * sp / max(r["speed"], 1e-6)
            r["speed"] = sp; r["approx"] = True
        elif r.get("speed_sd") and r["speed_sd"] > SD_MAX * r["speed"]:
            r["approx"] = True                                          # the fit itself cannot pin it down
    bounces = sorted([e for e in events if e["kind"] == "bounce"], key=lambda e: e["t"])
    out = []
    for p in points:
        evs = sorted([e for e in p.get("events", []) if e.get("kind") in ("net", "bounce")], key=lambda e: e["t"])
        crossings = [e for e in evs if e["kind"] == "net"]
        d = decisive(p)
        strokes = [(k, c) for k, c in enumerate(crossings, start=1)]
        net_k = [r["shot"] for r in fits.values() if r["point"] == p["id"] and r.get("net")]
        strokes += [(k, None) for k in net_k]                            # a return that went into the net: a stroke with no crossing
        for k, c in strokes:
            r = fits.get((p["id"], k))
            name = hitter(p, k) if c is not None else (p.get("near_player") if r["hitter_end"] == "near" else p.get("far_player"))
            if not name:
                continue
            end = "near" if name == p.get("near_player") else "far"
            ok = bool(r and r.get("ok"))
            land = None if c is None else next((b for b in evs if b["kind"] == "bounce" and c["t"] < b["t"] <= c["t"] + 1.0 and b.get("side") == c.get("side")), None)
            t_c = r["t_contact"] if r else None
            skel = pose_mod.skeleton_near(assigned, t_c, end, fps, reach=6) if (assigned and t_c is not None) else None
            m = pose_mod.measures(skel, end) if skel is not None else {}
            outcome = ""
            if c is None:
                outcome = "net"                                         # into the net
            elif d and d["stroke"] == k:
                outcome = "winner" if d["won"] else ("missed" if p.get("ending") == "long" else "")
            late = bool(r and r.get("contact_src") == "late")          # hidden at the racket, not bridged, no wrist near it: no speed off it
            if late and k == 1 and ok:                                  # ... except a serve whose own bounce went unseen: its speed where the
                late = False; r["approx"] = True                         # ball is first seen, a few % under, marked as an estimate
            free = bool(r and r.get("free"))
            out.append(dict(point=p["id"], shot=k, name=name, end=end, serve=k == 1, t=float(t_c if t_c is not None else c["t"]),
                            speed=round(r["speed"], 2) if ok and not late else None, spin=round(r["az"], 2) if ok else None,
                            over_net=r.get("over_net_cm") if ok and c is not None else None,
                            timing=contact_timing(track, bounces, t_c, end, fps) if (t_c is not None and k > 1) else None,
                            knee=m.get("knee"), leg=m.get("leg"), leg_ratio=m.get("leg_ratio"), profile2d=m.get("profile2d"), lean=m.get("lean"), stance=stance(cam, skel, end),
                            depth=round(abs(land["x_m"] - NET_X), 3) if land else None, outcome=outcome,
                            speed_sd=round(r["speed_sd"], 3) if ok and r.get("speed_sd") and not late else None,
                            contact=r.get("p0") if ok else None, path=r.get("path") if ok else None,
                            feet=_feet(cam, skel), incoming=_incoming(fits.get((p["id"], k - 1)), r if ok else None),
                            contact_src=r.get("contact_src") if r else None, free=free, net=c is None, approx=bool(r and r.get("approx")),
                            flight_t=round(float(r["t_bounce"]) - float(r["t_contact"]), 4) if ok else None))
    return out, list(fits.values())


def side_2d(s, hand="right"):
    """Forehand or backhand for a shot with no 3D body: from where the ball was met across the table relative to the feet (both in
    the table's frame). A player at the near end faces +x, so his right is -y; at the far end his right is +y. None without a fitted
    contact and the feet."""
    if not s.get("contact") or not s.get("feet"):
        return None
    off = float(s["contact"][1]) - float(np.mean([f[1] for f in s["feet"]]))
    right = off < 0 if s.get("end") == "near" else off > 0
    return "forehand" if right == (hand == "right") else "backhand"


def gate_knees(shots, strokes=None, cam=None):
    """The knee bend at contact as ONE quantity (pose.py, top). On every shot sets: side ('forehand' or 'backhand': from the 3D body
    when one was placed, else side_2d), profile (degrees between the pelvis line and the viewing ray), profile_src ('3d' the placed
    body, '2d' the picture's own pose.profile_2d, None = nothing to judge it by), knee_raw (the camera-near leg's picture angle whenever
    both legs were found), leg_ok (the leg rule checked) and knee = knee_raw on a rally forehand whose profile >= pose.PROFILE_MIN with
    the leg check passed, else None. The rules, in order:
      * strokes given (the 3D pass ran on this recording): a shot with no placed body is REFUSED, not judged by the picture. A body is
        missing exactly where the placement failed (rms over 6 px, scale off, a flipped or late contact: strokes_from), i.e. on the shot
        with the worse skeleton, and the picture's hip-width proxy let 26% of such contacts through that the pelvis refused, 14 deg off
        in the median (2026-09-28). With a body: profile from its pelvis (body3d.profile_of) with the camera, else the picture's.
      * no strokes at all (no 3D pass, e.g. a script without pose3d): the picture's profile_2d judges the profile.
      * the leg: with a body and a camera the 2D camera-near leg must be the leg whose knee is nearer the camera in the placed body
        (body3d.nearer_leg_of; they disagree on 20% of gated contacts, where the 3D and 2D angles differed most); without one the two
        legs' spans must differ by pose.LEG_RATIO_MIN.
    The player's racket hand for side_2d is the one his 3D strokes say (body3d.strokes_from), else right. Returns the shots."""
    from . import body3d, pose as P
    by = {(b["point"], b["shot"]): b for b in (strokes or [])}
    hand = {}
    for b in (strokes or []):
        hand.setdefault(b.get("name"), b.get("hand") or "right")
    for s in shots:
        b = by.get((s["point"], s["shot"]))
        if "knee_raw" not in s:
            s["knee_raw"] = s.get("knee")
        p2 = None if s.get("profile2d") is None else round(s["profile2d"], 1)
        if b is not None:
            s["side"] = b.get("side")
            if cam is not None:
                s["profile"] = round(body3d.profile_of(b, cam), 1); s["profile_src"] = "3d"
                s["leg_ok"] = None if s.get("leg") is None else (s["leg"] == body3d.nearer_leg_of(b, cam)[0])
            else:                                                       # a body but no camera to see it by: the picture judges both
                s["profile"] = p2; s["profile_src"] = None if p2 is None else "2d"
                s["leg_ok"] = None if s.get("leg_ratio") is None else (s["leg_ratio"] >= P.LEG_RATIO_MIN)
        elif strokes:                                                   # the 3D pass ran and could not place this one: refused
            s["side"] = side_2d(s, hand.get(s.get("name"), "right")); s["profile"] = None; s["profile_src"] = None; s["leg_ok"] = None
        else:
            s["side"] = side_2d(s, hand.get(s.get("name"), "right")); s["profile"] = p2; s["profile_src"] = None if p2 is None else "2d"
            s["leg_ok"] = None if s.get("leg_ratio") is None else (s["leg_ratio"] >= P.LEG_RATIO_MIN)
        ok = (not s.get("serve") and s["side"] == "forehand" and s["knee_raw"] is not None and s["profile"] is not None
              and s["profile"] >= P.PROFILE_MIN and s["leg_ok"] is True)
        s["knee"] = s["knee_raw"] if ok else None
    return shots
