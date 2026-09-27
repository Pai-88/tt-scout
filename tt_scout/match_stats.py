"""Match statistics as a table tennis coach or a broadcaster quotes them, from the points tt_scout found. Arithmetic on rallies.json (and stats.json / posture.json
when they exist); nothing here looks at the video.

Rules used: a game is won by the first player to 11 points with a lead of 2 (ITTF), and a change of ends also closes a game.
The count is tt_scout's own: a point it is unsure of is not counted, and a wrong call can move where a game ends.

Every point is put down to the shot that decided it (the "decisive stroke"): the shot that won it, or the shot that missed.
  * the last shot over the net went long or wide (ending "long")      -> that shot missed: its hitter lost the point with it;
  * the last shot over the net landed and did not come back           -> that shot won it: its hitter won the point with it.
Shot 1 is the serve. The three phases are the split table tennis coaching uses: serve and third ball (the server's shots 1 and 3),
receive and fourth ball (the receiver's shots 2 and 4), rally (shot 5 onwards). A ball returned into the net counts as won by the
shot that forced it: the camera cannot tell a forced miss from an unforced one.
"""
import numpy as np

GAME_TO, WIN_BY = 11, 2
PHASES = [("serve", "Serve and third ball"), ("receive", "Receive and fourth ball"), ("rally", "Rally, fifth shot on")]
DEPTH_NAMES = {"short": "short", "mid": "half-long", "deep": "long"}


def players(points):
    names = []
    for p in points:
        for k in ("near_player", "far_player"):
            if p.get(k) and p[k] not in names:
                names.append(p[k])
    return (names + ["near", "far"])[:2]


def games_of(points, to=GAME_TO, by=WIN_BY):
    """(per_point, games). per_point[id] = dict(game, before, after, games, games_after): scores as sparse {name: points} (a player
    who has not scored in the game is absent) and games won before and after the point. games = [dict(n, score, winner, finished,
    first, last)]. A game ends at `to` points with a lead of `by`, or at a change of ends (players swapped) with points played."""
    per, games = {}, []
    tally, won, prev_swap, first = {}, {}, None, None

    def close(last_id, finished):
        nonlocal tally, first
        if not tally:
            return
        lead = sorted(tally.items(), key=lambda kv: -kv[1])
        winner = lead[0][0] if finished and (len(lead) == 1 or lead[0][1] > lead[1][1]) else None
        if winner:
            won[winner] = won.get(winner, 0) + 1
        games.append(dict(n=len(games) + 1, score=dict(tally), winner=winner, finished=finished, first=first, last=last_id))
        tally, first = {}, None

    last_id = None
    for p in points:
        swap = p.get("ends_swapped")
        if prev_swap is not None and swap != prev_swap and sum(tally.values()) > 0:
            close(last_id, True)                                     # the players changed ends: whatever the count says, a game ended
        prev_swap = swap
        if first is None:
            first = p["id"]
        before, gb = dict(tally), dict(won)
        w = p.get("winner_name")
        if w:
            tally[w] = tally.get(w, 0) + 1
        after = dict(tally)
        game_no = len(games) + 1
        vals = sorted(tally.values(), reverse=True) + [0]
        if vals[0] >= to and vals[0] - vals[1] >= by:
            close(p["id"], True)
        per[p["id"]] = dict(game=game_no, before=before, after=after, games=gb, games_after=dict(won))
        last_id = p["id"]
    close(last_id, False)
    return per, games


def decisive(p):
    """dict(stroke, by, won, phase) for the shot that decided point p, or None when it cannot be read (no winner, no server, no shot)."""
    n, w, s = p.get("n_crossings") or 0, p.get("winner_name"), p.get("server")
    near, far = p.get("near_player"), p.get("far_player")
    if not n or not w or not s or s not in (near, far):
        return None
    receiver = far if s == near else near
    by = s if n % 2 == 1 else receiver
    phase = "serve" if n in (1, 3) else ("receive" if n in (2, 4) else "rally")
    return dict(stroke=n, by=by, won=by == w, phase=phase)


def longest_run(points, name):
    best = cur = 0
    for p in points:
        w = p.get("winner_name")
        if not w:
            continue
        cur = cur + 1 if w == name else 0
        best = max(best, cur)
    return best


def compute(points, stats=None, posture=None):
    """Everything the report's match-stats, momentum and rallies sections show. stats = stats.json content (speeds, serve depth);
    posture = pose.summary() output. Returns a plain dict (JSON-safe)."""
    names = players(points)
    per, games = games_of(points)
    decided = [(p, decisive(p)) for p in points]
    pl = (stats or {}).get("players", {})
    out = {"names": names, "games": games, "players": {}, "match": {}}
    for n in names:
        other = names[1] if n == names[0] else names[0]
        served = [p for p in points if p.get("server") == n and p.get("winner_name")]
        received = [p for p in points if p.get("server") == other and p.get("winner_name")]
        ph = {k: [0, 0] for k, _ in PHASES}
        for p, d in decided:
            if d and d["by"] == n:
                ph[d["phase"]][0 if d["won"] else 1] += 1
        missed = sum(1 for p, d in decided if d and d["by"] == n and not d["won"] and p.get("ending") == "long")
        forced = sum(1 for p, d in decided if d and d["by"] == n and d["won"])
        sp = pl.get(n, {}).get("speed_m_s", {})
        kmh = lambda v: None if v is None else round(v * 3.6)
        depth = pl.get(n, {}).get("serves", {}).get("depth", {})
        out["players"][n] = dict(
            points=sum(1 for p in points if p.get("winner_name") == n),
            games=sum(1 for g in games if g["winner"] == n),
            serve=[sum(1 for p in served if p["winner_name"] == n), len(served)],
            receive=[sum(1 for p in received if p["winner_name"] == n), len(received)],
            phases=ph, run=longest_run(points, n), missed_table=missed, not_returned=forced,
            fastest_kmh=kmh(sp.get("fastest")), fastest_est=bool(sp.get("fastest_est")), rally_kmh=kmh(sp.get("rally_median")),
            serve_kmh=kmh(sp.get("serve_median")),
            serve_length={DEPTH_NAMES[k]: int(v) for k, v in depth.items()},
            posture=(posture or {}).get(n))
    shots = [p.get("n_crossings") or 0 for p in points if p.get("n_crossings")]
    longest = max(points, key=lambda p: p.get("n_crossings") or 0) if points else None
    out["match"] = dict(
        points=len(points), decided=sum(1 for p in points if p.get("winner_name")),
        mean_shots=round(float(np.mean(shots)), 1) if shots else None,
        within_four=round(100 * float(np.mean([s <= 4 for s in shots]))) if shots else None,
        longest=dict(id=longest["id"], shots=longest.get("n_crossings"), t=longest["start_t"]) if longest else None,
        tempo_s=((stats or {}).get("match", {}).get("tempo_s") or {}).get("overall_median"))
    # momentum: the lead inside each game after every counted point (names[0] minus names[1])
    mom = []
    for p in points:
        s = per[p["id"]]
        if p.get("winner_name"):
            mom.append(dict(id=p["id"], game=s["game"], lead=s["after"].get(names[0], 0) - s["after"].get(names[1], 0),
                            winner=p["winner_name"]))
    out["momentum"] = mom
    return out


class Live:
    """What each player's analysis panel shows at any moment of the recording: points won on serve, winning shots and shots that missed
    the table (all up to the last point decided), the fastest shot so far and the knee bend at his last hit (both live, shot by shot).
        live = Live(points, events, fps, hits);  live.at(t) -> {name: dict(serve=(won, played), winners, misses, fastest, knee)}
    hits = [dict(t, name, knee)] (pose.at_hits, named); fastest is km/h over the table, net to bounce, 9 to 126 km/h kept."""

    def __init__(self, points, events, fps, hits=(), settle=0.8):
        from bisect import bisect_right
        from .config import NET_X
        self._bisect = bisect_right
        self.names = players(points)
        run = {n: dict(serve=[0, 0], winners=0, misses=0) for n in self.names}
        self.decided = []                                                # (time the point is visibly over, counts after it)
        self.shots = {n: [] for n in self.names}                         # (time of the landing, km/h), in time order
        for p in sorted(points, key=lambda p: p["start_t"]):
            evs = sorted(p.get("events", []), key=lambda e: e["t"])
            crossings = [e for e in evs if e["kind"] == "net" and not e.get("implied")]
            k_all = [e for e in evs if e["kind"] == "net"]
            for c in crossings:                                          # each real crossing's landing: the shot's speed over the table
                k = k_all.index(c) + 1
                land = next((b for b in evs if b["kind"] == "bounce" and 0.03 < b["t"] - c["t"] <= 1.0 and b.get("side") == c.get("side")), None)
                n = hitter(p, k)
                if land is not None and n in self.shots:
                    v = abs(land["x_m"] - NET_X) / (land["t"] - c["t"])
                    if 2.5 <= v <= 35.0:
                        self.shots[n].append((land["t"], v * 3.6))
            w, s = p.get("winner_name"), p.get("server")
            if s in run and w:
                run[s]["serve"][1] += 1; run[s]["serve"][0] += w == s
            d = decisive(p)
            if d and d["by"] in run:
                if d["won"]:
                    run[d["by"]]["winners"] += 1
                elif p.get("ending") == "long":
                    run[d["by"]]["misses"] += 1
            last = max([e["t"] for e in evs] or [p["end_t"]])
            self.decided.append((last + settle, {n: dict(serve=tuple(v["serve"]), winners=v["winners"], misses=v["misses"]) for n, v in run.items()}))
        self.decided.sort(key=lambda x: x[0])
        self._dt = [t for t, _ in self.decided]
        self.hits = {n: sorted((h["t"], h["knee"]) for h in hits if h.get("name") == n and h.get("knee") is not None) for n in self.names}
        for n in self.shots:
            self.shots[n].sort()

    def at(self, t):
        i = self._bisect(self._dt, t) - 1
        base = self.decided[i][1] if i >= 0 else {n: dict(serve=(0, 0), winners=0, misses=0) for n in self.names}
        out = {}
        for n in self.names:
            sh = [v for tt, v in self.shots[n] if tt <= t]
            kn = [k for tt, k in self.hits.get(n, []) if tt <= t]
            out[n] = dict(base[n], fastest=round(max(sh)) if sh else None, knee=round(kn[-1]) if kn else None)
        return out


def hitter(p, k):
    """Who hit shot k of point p (1 = the serve)."""
    s, near, far = p.get("server"), p.get("near_player"), p.get("far_player")
    return s if k % 2 == 1 else (far if s == near else near)


PALETTES = {                      # (outer plate, inner plate, dots) in RGB, for the celebration burst
    "classic": ((228, 38, 38), (255, 221, 51), (228, 38, 38)),
    "gold": ((240, 150, 20), (255, 236, 130), (225, 110, 10)),
    "blue": ((40, 110, 225), (255, 221, 51), (30, 90, 200)),
    "fire": ((220, 40, 30), (255, 150, 40), (255, 221, 51)),
    "muted": ((95, 115, 145), (222, 230, 238), (70, 85, 110)),
}


def moments(points, shots=None):
    """{point id: dict(word, line2, palette, intensity)}: the celebration each point earns, from what actually happened. intensity 0 = an
    error (muted), 1 = an ordinary point, 2 = a big shot or run, 3 = a game won or a comeback. Priority: game won, comeback, a run of
    5+, an ace, a fast winner, a long rally, a return winner, a third-ball winner, a run of 3 or 4, ending the other's run, a miss."""
    names = players(points)
    per, _ = games_of(points)
    sp = {(s["point"], s["shot"]): s.get("speed") for s in (shots or [])}
    streak = {n: 0 for n in names}; low = {}; out = {}
    for p in points:
        w = p.get("winner_name"); pid = p["id"]
        if not w or w not in names:
            continue
        other = next(n for n in names if n != w)
        g = per[pid]; W_, O_ = w.upper()[:12], other.upper()[:12]
        before = g["before"].get(w, 0) - g["before"].get(other, 0)
        low[(g["game"], w)] = min(low.get((g["game"], w), 0), before)
        lead_after = g["after"].get(w, 0) - g["after"].get(other, 0)
        streak[w] += 1; broken = streak[other]; streak[other] = 0
        game_won = g["games_after"].get(w, 0) > g["games"].get(w, 0)
        d = decisive(p); n = p.get("n_crossings") or 0
        speed = sp.get((pid, d["stroke"])) if d else None
        kmh = round(speed * 3.6) if speed else None
        won_by_shot = bool(d and d["won"] and d["by"] == w)
        sc = f"{g['after'].get(w, 0)}–{g['after'].get(other, 0)}"
        if game_won:
            m = ("GAME!", f"GAME TO {W_}  ·  {sc}", "gold", 3)
        elif low[(g["game"], w)] <= -4 and before < 0 <= lead_after:
            m = ("COMEBACK!", f"{W_} WAS {-low[(g['game'], w)]} DOWN  ·  NOW {sc}", "gold", 3)
        elif streak[w] >= 5:
            m = ("UNSTOPPABLE!", f"{streak[w]} POINTS IN A ROW FOR {W_}", "fire", 2)
        elif won_by_shot and d["stroke"] == 1:
            m = ("ACE!", "SERVE WINNER" + (f"  ·  {kmh} KM/H" if kmh else ""), "blue", 2)
        elif won_by_shot and speed and speed >= 9.0:
            m = ("ROCKET!" if speed >= 12 else "SMASH!", f"{kmh} KM/H WINNER", "classic", 2)
        elif n >= 12:
            m = ("MARATHON!", f"{n} SHOTS OVER THE NET", "classic", 2)
        elif n >= 8:
            m = ("EPIC RALLY!", f"{n} SHOTS OVER THE NET", "classic", 2)
        elif won_by_shot and d["stroke"] == 2:
            m = ("RETURN WINNER!", "STRAIGHT OFF THE SERVE" + (f"  ·  {kmh} KM/H" if kmh else ""), "blue", 2)
        elif won_by_shot and d["stroke"] == 3:
            m = ("THIRD BALL!", "SERVE, THEN THE KILL" + (f"  ·  {kmh} KM/H" if kmh else ""), "classic", 1)
        elif streak[w] >= 3:
            m = ("ON FIRE!", f"{streak[w]} IN A ROW FOR {W_}", "fire", 2)
        elif broken >= 3:
            m = ("STREAK OVER!", f"ENDS {O_}'S RUN OF {broken}", "classic", 1)
        elif p.get("ending") == "long":
            m = ("OUT!", f"{O_}'S SHOT WENT LONG", "muted", 0)
        elif p.get("ending") == "double_bounce":
            m = ("TOO GOOD!", f"{O_} COULD NOT GET IT BACK", "classic", 1)
        else:
            m = ("WINNER!" if won_by_shot else "POINT!", (f"{kmh} KM/H" if (won_by_shot and kmh) else f"POINT TO {W_}"), "classic", 1)
        out[pid] = dict(word=m[0], line2=m[1], palette=m[2], intensity=m[3], score=sc, winner=w)
    return out


def confirm(point, events, track, table, fps):
    """(confirmed, t, how): whether the camera saw the point end, when, and how. A celebration is only shown for a confirmed point.
      * a second bounce on the same side                                         -> confirmed at that bounce ("double bounce")
      * after the last shot crossed the net, before any bounce on that side, the ball is seen dropping below the table's height
        beyond an end or beside the table (it missed the table and is falling to the floor) -> confirmed then ("missed the table")
      * after the last shot bounced, the ball is seen dropping below the table's height the same way (it got past the player)
                                                                                  -> confirmed then ("winner")
      * after the last shot bounced, the ball comes back toward the net and drops there without crossing (the return hit the net)
                                                                                  -> confirmed then ("return into the net")
    Nothing seen: not confirmed (the camera lost the ball), and the point is only counted."""
    evs = sorted([e for e in point.get("events", []) if e.get("kind") in ("net", "bounce") and "x_px" in e], key=lambda e: e["t"])
    crossings = [e for e in evs if e["kind"] == "net"]
    if not crossings:
        return False, None, ""
    c = crossings[-1]; side = c.get("side")
    after_b = [e for e in evs if e["kind"] == "bounce" and e["t"] > c["t"] and e.get("side") == side]
    if len(after_b) >= 2 and after_b[1]["t"] - after_b[0]["t"] < 1.2:
        return True, after_b[1]["t"] + 0.1, "double bounce"
    corners = table.corners
    xs = corners[:, 0]; x_lo, x_hi = float(xs.min()), float(xs.max())
    n1, n2 = table.to_px([[1.37, 0.0], [1.37, 1.0]])
    def dnet(x, y):
        return ((x - n1[0]) * (n2[1] - n1[1]) - (y - n1[1]) * (n2[0] - n1[0])) / float(np.hypot(*(n2 - n1)))
    toward = 1 if table.to_px([[2.74, 0.76]])[0][0] > n1[0] else -1          # picture direction of the far end
    far_side = side == "far"
    end_x = x_hi if (far_side == (toward > 0)) else x_lo                      # the picture x of the end the shot went toward
    near_edge_y = float(max(corners[0, 1], corners[3, 1]))
    t0 = after_b[0]["t"] if after_b else c["t"]
    sel = (track[:, 1] > t0 + 0.5 / fps) & (track[:, 1] < t0 + 1.6) & ~np.isnan(track[:, 2])
    idx = np.where(sel)[0]
    past = (lambda x: x > end_x + 12) if end_x == x_hi else (lambda x: x < end_x - 12)
    for i in idx:
        x, y, t = track[i, 2], track[i, 3], track[i, 1]
        # dead: dropping below the table's height toward the floor (the ball passes the end line in every rally, so that alone proves
        # nothing: the other player hits it from behind the line). Beyond the end or beside the table, below the near edge's line.
        if y > near_edge_y + 40 and (past(x) or not (x_lo - 12 <= x <= x_hi + 12) or y > near_edge_y + 120):
            return True, float(t) + 0.1, ("winner" if after_b else "missed the table")
    if after_b:                                                                 # the return came back and died at the net
        seen = track[idx]
        if len(seen) >= 4:
            d = np.array([dnet(x, y) for x, y in seen[:, 2:4]])
            s_side = np.sign(d[0]) if d[0] != 0 else 1
            if np.any(np.abs(d) < 60) and not np.any(np.sign(d) == -s_side) and (np.abs(d).min() < 0.5 * np.abs(d).max()):
                j = int(np.argmin(np.abs(d)))
                return True, float(seen[j, 1]) + 0.25, "return into the net"
    return False, None, ""
