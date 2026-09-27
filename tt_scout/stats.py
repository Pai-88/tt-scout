"""Numbers from the points: serve placement by score situation, which shot ends points, tempo, depth/width
buckets, shot speed. Everything here is arithmetic on rallies.json; nothing looks at the video again.

Conventions: "oriented" coordinates put the player at x=0 and their opponent's half at x > L/2.
Depth is measured from the net into the opponent's half: short < 0.5 m, mid 0.5-1.0 m, deep > 1.0 m (the end line is
1.37 m from the net). Width is left / middle / right thirds as the player looks down the table.
"""
import json, pathlib, sys
import numpy as np
from .config import TABLE_LENGTH as L, TABLE_WIDTH as W, NET_X

DEPTH = [("short", 0.0, 0.5), ("mid", 0.5, 1.0), ("deep", 1.0, 9.9)]
# oriented y = W is the player's LEFT (near player: y = W is the far side of the table as he looks down it; far player: the oriented
# frame mirrors y). Until 2026-09-26 these labels were the wrong way round (flagged 2026-09-17); the placement charts were always right.
WIDTH = [("right", 0.0, W / 3), ("middle", W / 3, 2 * W / 3), ("left", 2 * W / 3, 9.9)]


def oriented(x, y, end):
    return (x, y) if end == "near" else (L - x, W - y)


def player_end(pt, name):
    return "near" if pt.get("near_player") == name else "far"


def bucket(v, table):
    return next((n for n, a, b in table if a <= v < b), table[-1][0])


def situation(score, names):
    a, b = score[names[0]], score[names[1]]
    hi, diff = max(a, b), abs(a - b)
    if min(a, b) >= 10 and diff <= 1:
        return "deuce"
    if hi >= 8:
        return "late"
    if hi < 5:
        return "early"
    return "mid"


def compute_stats(points, shots=None):
    names = []
    for pt in points:
        for k in ("near_player", "far_player"):
            if pt.get(k) and pt[k] not in names:
                names.append(pt[k])
    names = (names + ["near", "far"])[:2]
    out = {"players": {n: {} for n in names}, "match": {}}
    if not points:
        return out

    # ---- running score in games to 11 won by 2 (a change of ends also closes a game): match_stats.games_of
    from .match_stats import games_of
    per, games = games_of(points)
    for pt in points:
        g = per[pt["id"]]
        score = {n: g["before"].get(n, 0) for n in names}
        pt["game"] = g["game"]; pt["score_before"] = score; pt["situation"] = situation(score, names)
    game = max([pt["game"] for pt in points] or [1])
    out["match"]["games_seen"] = game

    for name in names:
        P = out["players"][name]
        served = [pt for pt in points if pt.get("server") == name]
        mine = [pt for pt in points if name in (pt.get("near_player"), pt.get("far_player"))]

        # ---- 1 + 4: serve placement, buckets, by situation
        serves = []
        for pt in served:
            for l in pt["landings"]:
                if l["shot"] == 1:
                    x, y = oriented(l["x_m"], l["y_m"], player_end(pt, name))
                    serves.append(dict(depth=bucket(x - NET_X, DEPTH), width=bucket(y, WIDTH), situation=pt["situation"],
                                       x=round(x - NET_X, 2), y=round(y, 2), won=pt.get("winner_name") == name))
        P["serves"] = {"n": len(serves),
                       "depth": {d: sum(s["depth"] == d for s in serves) for d, _, _ in DEPTH},
                       "width": {w: sum(s["width"] == w for s in serves) for w, _, _ in WIDTH},
                       "by_situation": {sit: {"n": sum(s["situation"] == sit for s in serves),
                                              "short": sum(s["situation"] == sit and s["depth"] == "short" for s in serves),
                                              "won": sum(s["situation"] == sit and s["won"] for s in serves)}
                                        for sit in ("early", "mid", "late", "deuce") if any(s["situation"] == sit for s in serves)}}
        thirds = []
        for pt in served:
            for l in pt["landings"]:
                if l["shot"] == 3:
                    x, y = oriented(l["x_m"], l["y_m"], player_end(pt, name))
                    thirds.append(dict(depth=bucket(x - NET_X, DEPTH), width=bucket(y, WIDTH)))
        P["third_ball"] = {"n": len(thirds),
                           "depth": {d: sum(s["depth"] == d for s in thirds) for d, _, _ in DEPTH},
                           "width": {w: sum(s["width"] == w for s in thirds) for w, _, _ in WIDTH}}

        # ---- 2: which shot ends the point, and win rate by rally length
        ends, by_len = [], {"short (<=3 shots)": [0, 0], "medium (4-7)": [0, 0], "long (8+)": [0, 0]}
        for pt in mine:
            n = pt["n_crossings"]; won = pt.get("winner_name") == name
            key = "short (<=3 shots)" if n <= 3 else ("medium (4-7)" if n <= 7 else "long (8+)")
            by_len[key][1] += 1; by_len[key][0] += won
            # the shot that crossed last was hit by the server if n is odd, by the receiver if even
            last_hitter = pt.get("server") if n % 2 == 1 else (pt["far_player"] if pt.get("server") == pt["near_player"] else pt["near_player"])
            if last_hitter == name:
                ends.append(dict(shot=n, won=won, ending=pt.get("ending")))
        P["points"] = {"played": len(mine), "won": sum(1 for pt in mine if pt.get("winner_name") == name),
                       "win_rate_by_length": {k: f"{v[0]}/{v[1]}" for k, v in by_len.items() if v[1]},
                       "my_last_shot_ended_point": {"n": len(ends), "won": sum(e["won"] for e in ends),
                                                   "lost_by": {k: sum(1 for e in ends if not e["won"] and e["ending"] == k) for k in ("long", "not_returned", "double_bounce")},
                                                   "shot_numbers": sorted(e["shot"] for e in ends)}}

        # ---- 5: shot speed along the table: net crossing to landing
        speeds = []
        for pt in served + [pt for pt in mine if pt not in served]:
            cross = [e for e in pt["events"] if e["kind"] == "net"]
            for l in pt["landings"]:
                k = l["shot"]
                if k - 1 < len(cross):
                    hitter = pt.get("server") if k % 2 == 1 else (pt["far_player"] if pt.get("server") == pt["near_player"] else pt["near_player"])
                    dt = l["t"] - cross[k - 1]["t"]
                    if hitter == name and dt > 0.03:
                        speeds.append(dict(shot=k, v=abs(l["x_m"] - NET_X) / dt))
        if shots is not None:                                   # the 3D-fitted speed off the racket (technique.py), replacing net-to-bounce
            speeds = [dict(shot=sh["shot"], v=sh["speed"], est=bool(sh.get("free") or sh.get("approx"))) for sh in shots   # est: about
                      if sh.get("name") == name and sh.get("speed") is not None]
        # the headline numbers come from measured speeds only; estimates (~) are counted but only lead when nothing was measured
        meas = [s for s in speeds if not s.get("est")]
        base = meas if len(meas) >= 5 else speeds
        sv = [s["v"] for s in base if s["shot"] == 1]; rv = [s["v"] for s in base if s["shot"] > 1]
        top = max(meas or speeds, key=lambda s: s["v"]) if speeds else None
        P["speed_m_s"] = {"serve_median": round(float(np.median(sv)), 1) if sv else None,
                          "rally_median": round(float(np.median(rv)), 1) if rv else None,
                          "fastest": round(float(top["v"]), 1) if top else None, "fastest_est": bool(top and top.get("est")),
                          "n": len(speeds), "n_est": sum(1 for s in speeds if s.get("est"))}

    # ---- 3: tempo, per point and across the match
    tempos = []
    for pt in points:
        ct = [e["t"] for e in pt["events"] if e["kind"] == "net"]
        gaps = np.diff(ct)
        pt["tempo_s"] = round(float(np.median(gaps)), 2) if len(gaps) else None
        if len(gaps):
            tempos.append((pt["start_t"], float(np.median(gaps))))
    if tempos:
        half = len(tempos) // 2 or 1
        out["match"]["tempo_s"] = {"overall_median": round(float(np.median([t for _, t in tempos])), 2),
                                   "first_half": round(float(np.median([t for _, t in tempos[:half]])), 2),
                                   "second_half": round(float(np.median([t for _, t in tempos[half:]])), 2) if len(tempos) > half else None,
                                   "per_game": {g: round(float(np.median([pt["tempo_s"] for pt in points if pt["game"] == g and pt["tempo_s"]])), 2)
                                                for g in range(1, game + 1) if any(pt["game"] == g and pt["tempo_s"] for pt in points)}}
    out["match"]["points"] = len(points)
    out["match"]["ending_shot_numbers"] = sorted(pt["n_crossings"] for pt in points)
    return out


def summary_text(st):
    lines = [f"points: {st['match'].get('points', 0)}, games seen: {st['match'].get('games_seen', 1)}"]
    tp = st["match"].get("tempo_s")
    if tp:
        lines.append(f"tempo (s between net crossings): overall {tp['overall_median']}, first half {tp['first_half']}, second half {tp['second_half']}, per game {tp['per_game']}")
    lines.append(f"shots per point at the ending: {st['match'].get('ending_shot_numbers')}")
    for name, P in st["players"].items():
        lines += ["", f"== {name}", f"points {P['points']['won']}/{P['points']['played']}   by rally length: {P['points']['win_rate_by_length']}",
                  f"serves ({P['serves']['n']}): depth {P['serves']['depth']}  width {P['serves']['width']}",
                  f"serves by score situation: {P['serves']['by_situation']}",
                  f"third ball ({P['third_ball']['n']}): depth {P['third_ball']['depth']}  width {P['third_ball']['width']}",
                  f"my last shot ended the point: {P['points']['my_last_shot_ended_point']}",
                  f"speed along the table (m/s): serve {P['speed_m_s']['serve_median']}, rally {P['speed_m_s']['rally_median']}, fastest {P['speed_m_s']['fastest']} (n={P['speed_m_s']['n']})"]
    return "\n".join(lines)


def write_stats(out_dir, points, shots=None):
    st = compute_stats(points, shots)
    out_dir = pathlib.Path(out_dir)
    (out_dir / "stats.json").write_text(json.dumps(st, indent=1))
    (out_dir / "stats.txt").write_text(summary_text(st))
    return st


if __name__ == "__main__":
    d = pathlib.Path(sys.argv[1])
    pts = json.load(open(d / "rallies.json"))
    print(summary_text(write_stats(d, pts)))
