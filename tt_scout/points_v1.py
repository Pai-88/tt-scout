"""Point segmentation and winner logic, version 1: the event sequence is REPAIRED before it is read.

# ------------------------------------------------------------------------------------------------------------
# REWRITE TARGET (research track, item 6), written 2026-09-17 from the error analysis in analysis/oracle.py and
# analysis/probe_v1*.py on the five OpenTTGames TRAINING games only. This logic is to be re-derived and
# re-implemented; tests/test_points_v1.py pins the behaviour. v0 stays in events.py /
# scoring.py (git tag v0-scouting) as the baseline for the ablation tables.
# ------------------------------------------------------------------------------------------------------------

What the v0 error analysis showed (development games, 223 points):
  * 95 % of true points contain a detected net crossing, so the events suffice; the logic loses the points.
  * 79 % of consecutive bounce pairs with no detected crossing between them lie on OPPOSITE sides of the net: the
    crossing was missed. v0 read such a pair as a double bounce (false ending) and, in the final stretch, credited the
    wrong player. Only 5 % of bounces are within 10 cm of the net, so a bounce's side is reliable evidence.
  * Spurious "points" between real ones are the ball being knocked back to the server: 94 % have exactly one crossing,
    the shot is slow (median 2.8 m/s along the table vs 4.3 for serves) and it comes soon after the previous point
    (median 5.5 s vs 10.4 s for the next real serve).
  * When no bounce follows the last crossing, "went long" is right 79 % of the time, so that rule is kept.

Rules:
  repair      walk bounces and crossings in time order keeping the side the ball is on; a bounce on the other side with
              no crossing since the last evidence inserts an IMPLIED crossing; a crossing to the side the ball is already
              on is a duplicate if it follows a crossing within dup_s, otherwise it implies a missed crossing back.
  points      start at the first crossing after the dead time; after each crossing TO side X: two bounces on X before the
              next crossing -> double bounce (other(X) wins); one bounce and no crossing within return_timeout_s -> not
              returned (other(X) wins); no bounce and no crossing within long_timeout_s -> long (X wins).
  knock-back  a candidate with exactly one crossing is dropped when that shot never lands on the table or is slower than
              serve_min_m_s; if the crossing was only implied (no crossing time, so no speed) it is dropped when it starts
              sooner than knock_gap_s after the previous point ended. A dropped candidate does not start a dead time.
  double      two bounces on X count as a double bounce only if no crossing follows within return_timeout_s; otherwise the
              rally went on and the second "bounce" was a false detection.
"""
from dataclasses import dataclass
import numpy as np
from .config import NET_X


def other(side):
    return "far" if side == "near" else "near"


@dataclass
class V1Config:
    net_margin_m: float = 0.10      # a bounce this close to the net line does not vote on the side (grid on the dev games: 0.10 beats 0.20)
    dup_s: float = 0.15             # two crossings to the same side closer than this are one crossing seen twice
    return_timeout_s: float = 1.5   # landed and no crossing back within this -> never returned (complete annotated rallies would allow 1.1 s, but detected sequences have gaps: grid on the dev games picks 1.5)
    long_timeout_s: float = 1.5     # crossed and neither landed nor crossed back within this -> long (grid on the dev games: 1.2 and 1.5 tie)
    dead_time_s: float = 2.5        # crossings this soon after an ending are ball retrieval
    serve_lead_s: float = 1.0       # the serve's own-side bounce is at most this long before the first crossing
    start_before_bounce_s: float = 0.3   # serve contact precedes its own-side bounce by about this
    start_before_cross_s: float = 0.8    # ... and the first crossing by about this when that bounce was not detected
    serve_min_m_s: float = 3.8      # along-table speed from the net to the first landing (dev games: real serves p10 3.9, knock-backs p90 4.0, median 3.6)
    knock_gap_s: float = 6.5        # real serves come later than this after the previous point (p10 of the dev games is 6.9 s)
    min_coverage: float = 0.3
    implied_max_gap_s: float = 1.3  # a bounce on the other side this long after the last evidence is NOT a missed crossing: in a rally the
                                    # two bounces of an exchange are 0.3 to 1.2 s apart (830 detected pairs with a crossing between them on the
                                    # 12 OpenTTGames videos, 99 % under 1.2 s). Later than that it is a dead ball (our 2026-09-25 recording:
                                    # a stray bounce by the table 2 s after the point gave two points to the wrong player).
    min_bounces: int = 1            # a "point" in which the ball never lands on this table is not one: another table's rally crossing this
                                    # net's line in the picture, or tracker junk (our 2026-09-25 recording: 6 such, all while a game ran
                                    # on the table behind; none in the recording without it)
    implied_cross_max_gap_s: float = 2.0  # the same for a crossing to the side the ball is already on: from a bounce on one side to the next
                                    # crossing back to it is a whole exchange, 0.6 to 2.0 s in rallies (1,402 on the 12 videos, a dip at
                                    # 1.8 to 2.0 s before the between-point gaps begin)


def repair(events, cfg=None):
    """Bounces and crossings in time order with implied crossings inserted. Every returned item has
    t, kind ('bounce' | 'net'), side (bounce: the half it landed on; net: the half crossed TO) and implied (bool)."""
    cfg = cfg or V1Config()
    evs = sorted((dict(e) for e in events if e["kind"] in ("bounce", "net")), key=lambda e: e["t"])
    out, side, last_cross_t = [], None, -1e9
    for e in evs:
        if e["kind"] == "net":
            if side == e["side"]:
                if e["t"] - last_cross_t <= cfg.dup_s:
                    continue                                        # the same crossing detected twice
            if side == e["side"] and (not out or e["t"] - out[-1]["t"] <= cfg.implied_cross_max_gap_s):
                # a missed crossing back, unless the last evidence is so old that the side it left is stale (between points the ball
                # goes back to the server untracked; our 2026-09-25 recording: a 5 s gap made a point out of nothing at 1:07)
                prev_t = out[-1]["t"] if out else e["t"] - 0.2
                mid = 0.5 * (prev_t + e["t"])
                out.append(dict(t=mid, kind="net", side=other(e["side"]), implied=True, x_m=NET_X, y_m=e.get("y_m", 0.0)))
            e["implied"] = False
            out.append(e); side, last_cross_t = e["side"], e["t"]
        else:
            ambiguous = abs(e.get("x_m", 0.0) - NET_X) < cfg.net_margin_m
            if ambiguous and side is not None:
                e["side"] = side                                    # too close to the net to vote: keep the ball where it was
            elif side is not None and e["side"] != side and (not out or e["t"] - out[-1]["t"] <= cfg.implied_max_gap_s):
                prev_t = out[-1]["t"] if out else e["t"] - 0.2
                mid = 0.5 * (prev_t + e["t"])
                out.append(dict(t=mid, kind="net", side=e["side"], implied=True, x_m=NET_X, y_m=e.get("y_m", 0.0)))
                last_cross_t = mid
            e["implied"] = False
            out.append(e); side = e["side"]
    return out


def _first_shot_speed(seq, i_cross, cfg):
    """Along-table speed (m/s) from the net to the first landing after crossing i_cross; None if it cannot be measured."""
    c = seq[i_cross]
    if c.get("implied"):
        return None
    for e in seq[i_cross + 1:]:
        if e["kind"] == "net" or e["t"] - c["t"] > 1.0:
            return None
        if e["kind"] == "bounce" and e["t"] > c["t"]:
            return abs(e["x_m"] - NET_X) / (e["t"] - c["t"])
    return None


def segment_points_v1(events, cfg=None, track=None):
    cfg = cfg or V1Config()
    seq = repair(events, cfg)
    cross_idx = [i for i, e in enumerate(seq) if e["kind"] == "net"]
    points, dead_until, last_end, ci = [], -1e9, -1e9, 0
    while ci < len(cross_idx):
        i = cross_idx[ci]
        if seq[i]["t"] < dead_until:
            ci += 1
            continue
        t_first, k = seq[i]["t"], ci
        end_t = ending = winner = None
        while end_t is None:
            c = seq[cross_idx[k]]
            X = c["side"]
            nxt_i = cross_idx[k + 1] if k + 1 < len(cross_idx) else len(seq)
            nxt_t = seq[nxt_i]["t"] if nxt_i < len(seq) else 1e9
            between = [e for e in seq[cross_idx[k] + 1:nxt_i] if e["kind"] == "bounce"]
            b1 = between[0] if between and between[0]["t"] - c["t"] <= cfg.long_timeout_s else None
            b2 = between[1] if b1 is not None and len(between) > 1 else None
            if (b1 is not None and b2 is not None and b2["t"] - b1["t"] <= cfg.return_timeout_s
                    and nxt_t - b2["t"] > cfg.return_timeout_s):
                # two bounces on X and the ball does NOT come back: a double bounce. If a crossing follows shortly after
                # the "second bounce" the rally went on, so that bounce was a false detection (a racket near the table)
                end_t, ending, winner = b2["t"] + 0.2, "double_bounce", other(X)
            elif b1 is not None and nxt_t - b1["t"] > cfg.return_timeout_s:
                end_t, ending, winner = b1["t"] + cfg.return_timeout_s, "not_returned", other(X)
            elif b1 is None and nxt_t - c["t"] > cfg.long_timeout_s:
                end_t, ending, winner = c["t"] + cfg.long_timeout_s, "long", X
            else:
                k += 1
        n_cross = k - ci + 1
        # knock-back to the server: one slow crossing soon after the previous point
        if n_cross == 1:
            v = _first_shot_speed(seq, i, cfg)
            if seq[i].get("implied"):
                knock = t_first - last_end < cfg.knock_gap_s       # crossing time unknown, so no speed: fall back on timing
            else:
                # a serve has to land on the receiver's half; a ball knocked back to the server is slow, or is caught and
                # never lands (dev games: 64 % of spurious single crossings have no landing, 12 % of real ones)
                knock = v is None or v < cfg.serve_min_m_s
            if knock:
                ci = k + 1
                continue
        if sum(e["kind"] == "bounce" and t_first - cfg.serve_lead_s <= e["t"] <= seq[cross_idx[k]]["t"] + cfg.long_timeout_s
               for e in seq) < cfg.min_bounces:
            ci = k + 1                          # never landed on this table: not a point, and it starts no dead time
            continue
        server = other(seq[i]["side"])
        pre = [e for e in seq if e["kind"] == "bounce" and e["side"] == server
               and max(dead_until, t_first - cfg.serve_lead_s) <= e["t"] < t_first]
        t0 = pre[0]["t"] - cfg.start_before_bounce_s if pre else t_first - cfg.start_before_cross_s
        pevs = [e for e in seq if t0 <= e["t"] <= end_t]
        pcross = [e for e in pevs if e["kind"] == "net"]
        landings = []
        for si, c in enumerate(pcross, start=1):
            nxt = pcross[si]["t"] if si < len(pcross) else end_t
            land = next((e for e in pevs if e["kind"] == "bounce" and c["t"] < e["t"] <= nxt and e["t"] - c["t"] <= cfg.long_timeout_s), None)
            if land:
                landings.append(dict(shot=si, x_m=land["x_m"], y_m=land["y_m"], side=land["side"], t=land["t"]))
        cov = 1.0
        if track is not None:
            sel = (track[:, 1] >= t0) & (track[:, 1] <= end_t)
            cov = float(np.mean(~np.isnan(track[sel, 2]))) if sel.any() else 0.0
        points.append(dict(id=len(points) + 1, start_t=float(t0), end_t=float(end_t), duration=float(end_t - t0), serve_side=server,
                           n_bounces=sum(e["kind"] == "bounce" for e in pevs), n_crossings=n_cross,
                           n_implied_crossings=sum(bool(e.get("implied")) for e in pcross), coverage=cov, events=pevs,
                           winner=winner if cov >= cfg.min_coverage else None, ending=ending, landings=landings))
        last_end = end_t
        dead_until = end_t + cfg.dead_time_s
        ci = k + 1
        while ci < len(cross_idx) and seq[cross_idx[ci]]["t"] < dead_until:
            ci += 1
    return points
