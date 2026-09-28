"""Player profiles that remember every match.

Every recording that is analysed with the players' names becomes one match record, profiles/matches/<id>.json, where <id> is a
fingerprint of the video file: analysing the same video again replaces its record instead of counting it twice. A record holds
what the match report measured (games, points, serve and receive, the three phases, speeds, placements, posture at each hit) and
the point list, so a profile can be recomputed from the records alone. Profiles are not stored separately: each player's page is
built from all the records they appear in, so there is one source of truth.

    record = record_match(...); build_pages()        # both done by `tt-scout report` / `tt-scout analyse` when players are named
    tt-scout profiles                                 # rebuild every page, list the players
    tt-scout profiles --remove <id>                   # forget one recording

Names are matched case-insensitively ("Sam" = "sam"). Everything stays on this computer; profiles/ is not committed to git.
"""
import datetime, hashlib, html, json, os, pathlib, re, subprocess
import numpy as np
from .config import ROOT, TABLE_LENGTH as L, TABLE_WIDTH as W
from .herocard import counts_in_career, knee_values, knee_pooled
from .pose import KNEE_LABEL

PROFILES = ROOT / "profiles"
SCHEMA = 2                  # 2 (2026-09-28): posture holds the gated knee (pose.py, top) with its [median, n]; schema-1 knees are another quantity
DEFAULT_NAMES = {"Left player", "Right player", "near", "far"}


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower()).strip("-") or "player"


def fingerprint(video):
    """12 hex characters from the file's size and its first and last 2 MB: stable, and quick on a 1 GB video."""
    p = pathlib.Path(video); h = hashlib.sha1(str(p.stat().st_size).encode())
    with open(p, "rb") as f:
        h.update(f.read(2 << 20))
        if p.stat().st_size > 4 << 20:
            f.seek(-(2 << 20), os.SEEK_END); h.update(f.read())
    return h.hexdigest()[:12]


def recording_date(video):
    """When the recording started, from the file's metadata (iPhones write it), else the file's modification time. ISO string."""
    try:
        tags = json.loads(subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_entries", "format_tags", str(video)],
                                         capture_output=True, text=True, check=True).stdout).get("format", {}).get("tags", {})
    except Exception:
        tags = {}
    for k in ("com.apple.quicktime.creationdate", "creation_time"):
        v = str(tags.get(k, "")).split(";")[0].strip()
        if v:
            try:
                return datetime.datetime.fromisoformat(v.replace("Z", "+00:00").replace("+0100", "+01:00").replace("+0000", "+00:00")).isoformat()
            except ValueError:
                continue
    return datetime.datetime.fromtimestamp(pathlib.Path(video).stat().st_mtime).astimezone().isoformat(timespec="seconds")


def _rel(path):
    p = pathlib.Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


# counts_in_career: whether a recording's numbers go into the career totals; the match list shows every recording either way. A
# judgement call, and the one that decides how much a bad recording can pollute a profile: keeping every match gives more data,
# dropping the "unreliable" ones (tt_scout's own verdict: the tracking could not be trusted) keeps wrong numbers out but loses real
# points, and "degraded" ones sit in between. Current rule: everything except "unreliable". It lives in herocard (no numpy there, the
# web server imports it) so the hero cards and the profile pages filter the records by the one rule.


def record_match(root, video, names, points, ms, quality, report=None, posture=None, portrait=None, recorded=None, critique=None, position3d=None):
    """Write (or replace) the match record for this video. names = {"near": .., "far": ..} at the start of the recording;
    ms = match_stats.compute(); posture = posture.json content: hits (lean at each racket hit) and contacts (pose.knee_contacts, the
    gated knee bend at contact). Per player the record keeps knee = the gated angles, knee_won beside them, knee_median = [median, n]
    (None under pose.KNEE_MIN_N), lean and lean_won: the card, the profile's fact and the trend all read these. Returns the record."""
    from .pose import hitter_name, knee_median
    root = pathlib.Path(root); (root / "matches").mkdir(parents=True, exist_ok=True)
    players = list(dict.fromkeys([names["near"], names["far"]]))
    pose_by = {n: dict(knee=[], knee_won=[], knee_median=[None, 0], lean=[], lean_won=[]) for n in players}
    for nm, cs in ((posture or {}).get("contacts") or {}).items():
        if nm in pose_by:
            pose_by[nm]["knee"] = [round(c["knee"], 1) for c in cs if c.get("knee") is not None]
            pose_by[nm]["knee_won"] = [c.get("won") for c in cs if c.get("knee") is not None]
    for n in players:
        pose_by[n]["knee_median"] = list(knee_median(pose_by[n]["knee"]))
    for h in (posture or {}).get("hits", []):
        nm = hitter_name(h, points, names)
        if nm not in pose_by or h.get("lean") is None:
            continue
        p = next((p for p in points if p["start_t"] <= h["t"] <= p["end_t"]), None)
        pose_by[nm]["lean"].append(round(h["lean"], 1))
        pose_by[nm]["lean_won"].append(bool(p and p.get("winner_name") == nm))
    games = ms.get("games", [])
    gw = {n: sum(1 for g in games if g["winner"] == n) for n in players}
    pw = {n: ms["players"][n]["points"] for n in players}
    order = sorted(players, key=lambda n: (gw[n], pw[n]), reverse=True)
    # a result needs a lead in finished games: a clip of part of one game (7-6 when the phone stopped), or 2-2 with the decider
    # unfinished, is nobody's win, whoever has more points
    winner = order[0] if gw[order[0]] != gw[order[1]] else None
    rec = dict(schema=SCHEMA, id=fingerprint(video), recorded=recorded or recording_date(video), added=datetime.datetime.now().isoformat(timespec="seconds"),
               video=_rel(video), report=_rel(report) if report else None, players=players, winner=winner,
               verdict=dict(level=quality.get("level"), score=quality.get("score"), notes=quality.get("reasons", [])),
               games=games, match=ms.get("match", {}), stats={n: {k: v for k, v in ms["players"][n].items() if k != "posture"} for n in players},
               posture=pose_by, portrait={n: _rel(v) for n, v in (portrait or {}).items() if v},
               critique={n: [dict(key=c_.key, head=c_.head, evidence=c_.evidence, fix=c_.fix) for c_ in cs] for n, cs in (critique or {}).items()},
               position3d={n: {k: m[k] for k in ("feet_back", "width", "reach", "height", "timing", "top_share", "net", "depth") if k in m}
                           for n, m in (position3d or {}).items()},     # after the match, in 3D (analysis3d.py): [median, count] each
               points=[dict(id=p["id"], t=round(p["start_t"], 2), near=p.get("near_player"), far=p.get("far_player"), server=p.get("server"),
                            winner=p.get("winner_name"), shots=p.get("n_crossings"), ending=p.get("ending"),
                            landings=[[l["shot"], round(l["x_m"], 3), round(l["y_m"], 3), l["side"]] for l in p.get("landings", [])])
                       for p in points])
    (root / "matches" / f"{rec['id']}.json").write_text(json.dumps(rec, indent=1))
    return rec


def load_records(root=PROFILES):
    recs = []
    for f in sorted((pathlib.Path(root) / "matches").glob("*.json")):
        try:
            recs.append(json.loads(f.read_text()))
        except Exception:
            continue
    return sorted(recs, key=lambda r: r.get("recorded") or "")


def player_names(recs):
    """slug -> the most recent spelling of that name."""
    out = {}
    for r in recs:
        for n in r["players"]:
            out[slug(n)] = n
    return out


def _med(xs):
    xs = [x for x in xs if x is not None]
    return (float(np.median(xs)), len(xs)) if xs else (None, 0)


def aggregate(key, recs):
    """Everything a profile page shows, for the player whose slug is key, from the match records they appear in."""
    mine = [r for r in recs if key in (slug(n) for n in r["players"])]
    career = [r for r in mine if counts_in_career(r)]
    name = player_names(mine).get(key, key)
    rows, h2h = [], {}
    tot = dict(serve=[0, 0], receive=[0, 0], phases={k: [0, 0] for k in ("serve", "receive", "rally")}, winners=0, missed=0,
               played=0, opp_missed=0, fastest=None, serve_length={"short": 0, "half-long": 0, "long": 0}, knee=[], knee_won=[], knee_lost=[], lean=[],
               serves=[], thirds=[], returns=[], won=0, lost=0, games=[0, 0], points=[0, 0], rally_kmh=[])
    for r in mine:
        nm = next(n for n in r["players"] if slug(n) == key)
        opp = next((n for n in r["players"] if n != nm), None)
        s, o = r["stats"].get(nm, {}), r["stats"].get(opp, {})
        res = "W" if r.get("winner") == nm else ("L" if r.get("winner") == opp else "D")
        k = tuple(((r.get("posture") or {}).get(nm) or {}).get("knee_median") or (None, 0))   # the record's own gated median (pose.knee_median)
        played = s.get("points", 0) + o.get("points", 0)
        rows.append(dict(id=r["id"], schema=r.get("schema", 1), date=r.get("recorded"), opp=opp, res=res, games=[s.get("games", 0), o.get("games", 0)],
                         points=[s.get("points", 0), o.get("points", 0)], serve=s.get("serve", [0, 0]), receive=s.get("receive", [0, 0]),
                         winners=s.get("not_returned", 0), missed=s.get("missed_table", 0), played=played, fastest=s.get("fastest_kmh"),
                         rally_kmh=s.get("rally_kmh"), knee=k[0], verdict=r.get("verdict", {}).get("level"), report=r.get("report"),
                         counted=counts_in_career(r), game_scores=[(g["score"].get(nm, 0), g["score"].get(opp, 0), g["finished"]) for g in r.get("games", [])],
                         critique=(r.get("critique") or {}).get(nm, []),
                         feet_back=((r.get("position3d") or {}).get(nm, {}).get("feet_back") or [None])[0],
                         top_share=((r.get("position3d") or {}).get(nm, {}).get("top_share") or [None])[0],
                         over_net=((r.get("position3d") or {}).get(nm, {}).get("net") or [None])[0]))
        hh = h2h.setdefault(slug(opp), dict(name=opp, rec=[0, 0, 0], games=[0, 0], points=[0, 0]))
        hh["rec"]["WLD".index(res)] += 1
        hh["games"][0] += s.get("games", 0); hh["games"][1] += o.get("games", 0)
        hh["points"][0] += s.get("points", 0); hh["points"][1] += o.get("points", 0)
        if r not in career:
            continue
        for a in ("serve", "receive"):
            tot[a][0] += s.get(a, [0, 0])[0]; tot[a][1] += s.get(a, [0, 0])[1]
        for ph, (w_, l_) in s.get("phases", {}).items():
            tot["phases"][ph][0] += w_; tot["phases"][ph][1] += l_
        tot["winners"] += s.get("not_returned", 0); tot["missed"] += s.get("missed_table", 0); tot["opp_missed"] += o.get("missed_table", 0)
        tot["played"] += played
        tot["won"] += res == "W"; tot["lost"] += res == "L"
        tot["games"][0] += s.get("games", 0); tot["games"][1] += o.get("games", 0)
        tot["points"][0] += s.get("points", 0); tot["points"][1] += o.get("points", 0)
        if s.get("fastest_kmh"):
            tot["fastest"] = max(tot["fastest"] or 0, s["fastest_kmh"])
        if s.get("rally_kmh"):
            tot["rally_kmh"].append(s["rally_kmh"])
        for lk, v in (s.get("serve_length") or {}).items():
            tot["serve_length"][lk] = tot["serve_length"].get(lk, 0) + v
        pz = (r.get("posture") or {}).get(nm) or {}
        tot["lean"] += [x for x in pz.get("lean", []) if x is not None]
        kv = knee_values(r, nm)                                          # the gated knee only (schema 2 records)
        tot["knee"] += kv
        for kn, won in zip(kv, pz.get("knee_won", [])):
            if won is True:
                tot["knee_won"].append(kn)
            elif won is False:
                tot["knee_lost"].append(kn)
        for p in r.get("points", []):                                   # placements, oriented so the player is at the left end
            if nm not in (p.get("near"), p.get("far")):
                continue
            end = "near" if p.get("near") == nm else "far"; serving = p.get("server") == nm; won = p.get("winner") == nm
            for shot, x, y, side in p.get("landings", []):
                if (serving and shot in (1, 3)) or (not serving and shot == 2):
                    xy = (x, y) if end == "near" else (L - x, W - y)
                    {1: tot["serves"], 3: tot["thirds"], 2: tot["returns"]}[shot].append((xy[0], xy[1], won))
    return dict(key=key, name=name, rows=rows, h2h=sorted(h2h.values(), key=lambda h: -sum(h["rec"])), tot=tot,
                knee=knee_pooled(mine, name),                            # (median, n): the same call the hero card makes
                has_pose=any(bool(pz.get("lean") or pz.get("knee")) for r in mine                 # skeletons were measured in some recording
                             for pz in [((r.get("posture") or {}).get(next((n for n in r["players"] if slug(n) == key), "")) or {})]),
                n=len(mine), n_career=len(career), first=mine[0].get("recorded") if mine else None, last=mine[-1].get("recorded") if mine else None,
                portrait=next((r["portrait"].get(next(n for n in r["players"] if slug(n) == key)) for r in reversed(mine) if r.get("portrait")), None))


def tendencies(a, min_n=8):
    """Plain sentences a coach would say, each with its sample size; nothing is said on fewer than min_n cases."""
    t, out = a["tot"], []
    sl = t["serve_length"]; ns = sum(sl.values())
    if ns >= min_n:
        top = max(sl, key=sl.get)
        out.append(f"Serves {top}: {sl[top]} of {ns} serves that landed ({round(100 * sl[top] / ns)}%).")
    rates = {ph: (w_, w_ + l_) for ph, (w_, l_) in t["phases"].items() if w_ + l_ >= 5}
    label = {"serve": "serve and third ball", "receive": "receive and fourth ball", "rally": "rallies (fifth shot on)"}
    if len(rates) >= 2:
        best = max(rates, key=lambda k: rates[k][0] / rates[k][1]); worst = min(rates, key=lambda k: rates[k][0] / rates[k][1])
        if best != worst:
            out.append(f"Strongest in {label[best]}: won {rates[best][0]} of {rates[best][1]} points his own shot decided there; "
                       f"weakest in {label[worst]}: {rates[worst][0]} of {rates[worst][1]}.")
    if t["played"] >= 20:
        per = 10 * t["missed"] / t["played"]; oper = 10 * t["opp_missed"] / t["played"]
        out.append(f"Misses the table {per:.1f} times per 10 points played (his opponents: {oper:.1f}).")
    if t["serve"][1] >= min_n and t["receive"][1] >= min_n:
        out.append(f"Wins {round(100 * t['serve'][0] / t['serve'][1])}% of points on his serve ({t['serve'][0]} of {t['serve'][1]}) and "
                   f"{round(100 * t['receive'][0] / t['receive'][1])}% on receive ({t['receive'][0]} of {t['receive'][1]}).")
    kw, kl = _med(t["knee_won"]), _med(t["knee_lost"])
    if kw[1] >= min_n and kl[1] >= min_n and abs(kw[0] - kl[0]) >= 4:
        out.append(f"Bends his knees {'more' if kw[0] < kl[0] else 'less'} in points he wins: {kw[0]:.0f}° at contact against "
                   f"{kl[0]:.0f}° in points he loses ({kw[1]} and {kl[1]} forehand contacts, {KNEE_LABEL}).")
    return out
