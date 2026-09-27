"""Player profile pages: one per player and an index, built from the match records in profiles/matches (see profiles.py).
Same design language as the match report: warm paper, ink rules, serif numerals.
"""
import datetime, html, os, pathlib
from .config import ROOT
from .report_html import CSS, _svg_table
from .profiles import PROFILES, load_records, player_names, aggregate, tendencies, slug, _med

e = html.escape
PCSS = r"""
.pf{display:grid;grid-template-columns:minmax(0,1fr) minmax(11rem,17rem);gap:clamp(1.2rem,4vw,3rem);align-items:start;padding:clamp(1.4rem,3.5vw,2.4rem) 0 clamp(1.2rem,3vw,2rem);border-bottom:1px solid var(--line)}
.pf figure{margin:0} .pf img{display:block;width:100%;height:auto;border-radius:2px;background:var(--paper-2)}
.pf figcaption{padding-top:.4rem;font-size:.78rem;color:var(--ink-3)}
.rec{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:clamp(.8rem,2.5vw,2rem)}
.rec div{display:grid;gap:.1rem} .rec dt{font-size:.7rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:var(--ink-3)}
.rec dd{margin:0;font-family:var(--serif);font-size:clamp(2.2rem,1.5rem + 3vw,3.6rem);line-height:.95;letter-spacing:-.02em}
.rec dd small{font-family:var(--sans);font-size:.8rem;color:var(--ink-3);letter-spacing:0;display:block;padding-top:.3rem}
.form{display:flex;flex-wrap:wrap;gap:.35rem;align-items:center;margin:1.4rem 0 0;padding:0;list-style:none}
.form li{width:2rem;height:2rem;border-radius:50%;display:grid;place-items:center;font-size:.72rem;font-weight:700;border:1.5px solid var(--ink);color:var(--ink)}
.form li.W{background:var(--ink);color:var(--paper)} .form li.D{border-style:dashed;color:var(--ink-3)}
.form .lab{width:auto;height:auto;border:0;font-size:.7rem;letter-spacing:.15em;text-transform:uppercase;color:var(--ink-3);margin-right:.4rem}
.tend{list-style:none;margin:0;padding:0} .tend li{padding:.8rem 0;border-top:1px solid var(--line);font-family:var(--serif);font-size:clamp(1.05rem,1rem + .3vw,1.2rem);line-height:1.4}
.tend li:first-child{border-top:0}
.trends{display:grid;grid-template-columns:repeat(auto-fit,minmax(15rem,1fr));gap:1.4rem clamp(1rem,3vw,2rem);padding-top:1.2rem}
.trends figure{margin:0} .trends svg{width:100%;height:auto;display:block;overflow:visible}
.trends figcaption{display:flex;justify-content:space-between;font-size:.8rem;color:var(--ink-2);border-bottom:1px solid var(--line);padding-bottom:.35rem;margin-bottom:.4rem}
.trends figcaption b{font-size:.7rem;font-weight:700;letter-spacing:.13em;text-transform:uppercase}
.trends .v{font:400 13px var(--serif);fill:var(--ink)} .trends .d{font:600 9px var(--sans);letter-spacing:.08em;fill:var(--ink-3)}
.hist td a{white-space:nowrap} .hist .res{font-weight:700} .hist .gs{color:var(--ink-3);font-size:.84rem} .hist tr.out td{color:var(--ink-3)}
.players{list-style:none;margin:0;padding:0}
.players li{display:grid;grid-template-columns:4.2rem minmax(0,1fr) auto;gap:1rem;align-items:center;padding:.9rem 0;border-top:1px solid var(--line)}
.players img{width:4.2rem;height:5.6rem;object-fit:cover;border-radius:2px;background:var(--paper-2)}
.players a{font-family:var(--serif);font-size:1.6rem;color:var(--ink);text-decoration:none} .players a:hover{text-decoration:underline}
.players p{margin:.1rem 0 0;color:var(--ink-3);font-size:.86rem} .players .r{text-align:right;font-family:var(--serif);font-size:1.25rem}
@media (max-width:40rem){.pf{grid-template-columns:minmax(0,1fr)}.pf figure{max-width:14rem}.rec{grid-template-columns:repeat(2,minmax(0,1fr))}}
"""


def _date(iso, fmt="%-d %b %Y"):
    try:
        return datetime.datetime.fromisoformat(iso).strftime(fmt)
    except Exception:
        return iso or ""


def _link(path, from_dir):
    return os.path.relpath(ROOT / path, from_dir) if path else None


def _trend(title, rows, key, fmt, unit="", lower_better=False):
    """One small chart across recordings, every value printed on its point; nothing drawn from fewer than two values."""
    pts = [(i, r[key]) for i, r in enumerate(rows) if r.get(key) is not None]
    if len(pts) < 2:
        return ""
    days = [_date(r["date"], "%-d %b") for r in rows]
    same_day = len(set(days)) < len(days)                              # two recordings on one day: label them by time instead
    vals = [v for _, v in pts]; lo, hi = min(vals), max(vals); span = (hi - lo) or max(1.0, abs(hi) * 0.1)
    W_, H_, pad = 300.0, 110.0, 18.0
    def X(i):
        return pad + (W_ - 2 * pad) * (i / max(1, len(rows) - 1))
    def Y(v):
        return H_ - 26 - (H_ - 50) * ((v - lo) / span)
    line = " ".join(f"{'M' if k == 0 else 'L'}{X(i):.1f},{Y(v):.1f}" for k, (i, v) in enumerate(pts))
    dots = "".join(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3.6" fill="var(--paper)" stroke="var(--ink)" stroke-width="1.6"/>'
                   f'<text class="v" x="{X(i):.1f}" y="{Y(v) - 9:.1f}" text-anchor="middle">{fmt(v)}</text>'
                   f'<text class="d" x="{X(i):.1f}" y="{H_ - 6:.0f}" text-anchor="middle">{_date(rows[i]["date"], "%-d %b %H:%M" if same_day else "%-d %b").upper()}</text>' for i, v in pts)
    last, first = pts[-1][1], pts[0][1]
    arrow = "" if last == first else ("↓" if last < first else "↑")
    return (f'<figure><figcaption><b>{title}</b><span>{arrow} {fmt(last)}{unit} latest</span></figcaption>'
            f'<svg viewBox="0 0 {W_:.0f} {H_:.0f}" role="img" aria-label="{e(title)} by recording">'
            f'<path d="{line}" fill="none" stroke="var(--ink)" stroke-width="1.6" stroke-linejoin="round"/>{dots}</svg></figure>')


def player_page(a, out_dir):
    t, rows = a["tot"], a["rows"]
    name = a["name"]
    port = _link(a["portrait"], out_dir) if a.get("portrait") else None
    pct = lambda x: None if not x[1] else round(100 * x[0] / x[1])
    km = _med(t["knee"])
    form = "".join(f'<li class="{r["res"]}" title="{e(_date(r["date"]))} v {e(r["opp"] or "")}">{r["res"]}</li>' for r in rows[-10:])
    top = (f'<div class="pf"><div><dl class="rec">'
           f'<div><dt>Recordings</dt><dd>{t["won"]}–{t["lost"]}<small>won–lost, of {a["n_career"]} counted</small></dd></div>'
           f'<div><dt>Games</dt><dd>{t["games"][0]}–{t["games"][1]}<small>to 11, won by 2</small></dd></div>'
           f'<div><dt>Points</dt><dd>{t["points"][0]}–{t["points"][1]}<small>{pct((t["points"][0], sum(t["points"]))) if sum(t["points"]) else 0}% won</small></dd></div></dl>'
           f'<ul class="form"><li class="lab">Form</li>{form}</ul></div>'
           + (f'<figure><img src="{e(port)}" alt="{e(name)} at contact, from his latest recording" width="300" height="400"><figcaption>at contact, latest recording</figcaption></figure>' if port else "<div></div>")
           + "</div>")
    facts = [("Points won on serve", f"{pct(t['serve'])}%" if t["serve"][1] else "–", f"{t['serve'][0]} of {t['serve'][1]}"),
             ("Points won on receive", f"{pct(t['receive'])}%" if t["receive"][1] else "–", f"{t['receive'][0]} of {t['receive'][1]}"),
             ("Serve and third ball", f"{pct((t['phases']['serve'][0], sum(t['phases']['serve'])))}%" if sum(t["phases"]["serve"]) else "–",
              f"won {t['phases']['serve'][0]} · lost {t['phases']['serve'][1]}"),
             ("Receive and fourth ball", f"{pct((t['phases']['receive'][0], sum(t['phases']['receive'])))}%" if sum(t["phases"]["receive"]) else "–",
              f"won {t['phases']['receive'][0]} · lost {t['phases']['receive'][1]}"),
             ("Rally, fifth shot on", f"{pct((t['phases']['rally'][0], sum(t['phases']['rally'])))}%" if sum(t["phases"]["rally"]) else "–",
              f"won {t['phases']['rally'][0]} · lost {t['phases']['rally'][1]}"),
             ("Winning shots", f"{10 * t['winners'] / t['played']:.1f}" if t["played"] else "–", "per 10 points played"),
             ("Missed the table", f"{10 * t['missed'] / t['played']:.1f}" if t["played"] else "–", "per 10 points played"),
             ("Fastest shot", f"{t['fastest']}" if t["fastest"] else "–", "km/h, over the table"),
             ("Knee bend at contact", f"{km[0]:.0f}°" if km[0] is not None else "–", f"median of {km[1]} hits" if km[1] else "no skeletons yet")]
    facts_html = ('<dl class="facts">' + "".join(f"<div><dt>{k}</dt><dd>{v}<small>{s}</small></dd></div>" for k, v, s in facts) + "</dl>")
    heads = {}                                                        # the coach's criticisms, and how often each comes back
    for r in rows:
        for c in r.get("critique") or []:
            heads.setdefault(c["head"], []).append((r, c))
    crit = ""
    if heads:
        items = ""
        for head, seen in sorted(heads.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            r_, c_ = seen[-1]
            items += (f'<li><p class="for"><b>{len(seen)} of {len(rows)}</b><small>recordings</small></p><h3>{e(head)}</h3>'
                      f'<p class="ev">Latest ({e(_date(r_["date"]))} v {e(r_["opp"] or "")}): {e(c_["evidence"])}.<small>Fix: {e(c_["fix"])}.</small></p></li>')
        crit = (f'<section class="tips"><div class="head"><h2>Coach\u2019s critique</h2><span>the criticisms his matches keep bringing up</span></div>'
                f'<ol class="tiplist">{items}</ol></section>')
    tl = tendencies(a)
    tend = (f'<section><div class="head"><h2>Habits</h2><span>across {a["n_career"]} recording{"s" if a["n_career"] != 1 else ""}, with the counts behind them</span></div>'
            + ('<ul class="tend">' + "".join(f"<li>{e(x)}</li>" for x in tl) + "</ul>" if tl else
               '<div class="empty">Not enough points yet to say anything with confidence. Habits appear once there are about 8 cases of each.</div>')
            + "</section>")
    for r in rows:
        r["serve_pct"] = pct(r["serve"]); r["miss10"] = round(10 * r["missed"] / r["played"], 1) if r["played"] else None
    trends = "".join(x for x in (
        _trend("Points won on serve", rows, "serve_pct", lambda v: f"{v:.0f}%"),
        _trend("Missed the table, per 10 points", rows, "miss10", lambda v: f"{v:.1f}"),
        _trend("Knee bend at contact", rows, "knee", lambda v: f"{v:.0f}°"),
        _trend("Typical rally shot", rows, "rally_kmh", lambda v: f"{v:.0f}", " km/h"),
        _trend("Feet behind the end line at contact", rows, "feet_back", lambda v: f"{v:.2f}", " m"),
        _trend("Taken at the top of the bounce or rising", rows, "top_share", lambda v: f"{100 * v:.0f}%"),
        _trend("Height over the net", rows, "over_net", lambda v: f"{v:.0f}", " cm")) if x)
    trends = (f'<section><div class="head"><h2>Trends</h2><span>one point per recording, oldest on the left</span></div><div class="trends">{trends}</div></section>'
              if trends else "")
    h2h = "".join(f'<tr><td>{e(h["name"] or "")}</td><td>{h["rec"][0]}–{h["rec"][1]}' + (f' <small>({h["rec"][2]} level)</small>' if h["rec"][2] else "")
                  + f'</td><td>{h["games"][0]}–{h["games"][1]}</td><td>{h["points"][0]}–{h["points"][1]}</td></tr>' for h in a["h2h"])
    h2h = (f'<section class="numbers"><div class="head"><h2>Head to head</h2><span>every recording, counted or not</span></div><table><thead><tr><th>Opponent</th>'
           f'<th>Recordings</th><th>Games</th><th>Points</th></tr></thead><tbody>{h2h}</tbody></table></section>')
    hist = ""
    for r in reversed(rows):
        rep = _link(r["report"], out_dir)
        gs = " ".join(f'{x}–{y}{"" if fin else "*"}' for x, y, fin in r["game_scores"])
        hist += (f'<tr class="{"" if r["counted"] else "out"}"><td>{e(_date(r["date"], "%-d %b %Y, %H:%M"))}</td><td>{e(r["opp"] or "")}</td>'
                 f'<td class="res">{r["res"]}</td><td>{r["games"][0]}–{r["games"][1]} <span class="gs">{e(gs)}</span></td>'
                 f'<td>{r["points"][0]}–{r["points"][1]}</td><td>{e(r["verdict"] or "")}{"" if r["counted"] else " (not counted)"}</td>'
                 f'<td>{f"<a href={chr(34)}{e(rep)}{chr(34)}>report</a>" if rep else ""}</td></tr>')
    hist = (f'<section class="numbers hist"><div class="head"><h2>Matches</h2><span>newest first · * game in play when the recording stopped</span></div>'
            f'<div class="ledger"><table><thead><tr><th>Recorded</th><th>Opponent</th><th>Result</th><th>Games</th><th>Points</th><th>Verdict</th>'
            f'<th><span class="sr">Report</span></th></tr></thead><tbody>{hist}</tbody></table></div></section>')
    place = "".join(f'<figure>{_svg_table(d, colour="var(--p1)")}<figcaption><b>{lab}</b><span>{len(d)} landed · {sum(1 for x in d if x[2])} in points won</span></figcaption></figure>'
                    for lab, d in (("Serves", t["serves"]), ("Third ball", t["thirds"]), ("Returns of serve", t["returns"])))
    place = (f'<section><div class="head"><h2>Placement</h2><span>all counted recordings, seen from above, {e(name)} at the left end</span></div>'
             f'<div class="who">{place}</div><div class="key"><span><i style="background:var(--p1)"></i>filled: a point he won</span>'
             f'<span><i class="o"></i>hollow: a point he lost</span></div></section>')
    since = f'since {_date(a["first"])}' if a.get("first") else ""
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{e(name)}: player profile</title><style>{CSS}{PCSS}</style></head><body><div class="wrap">'
            f'<header class="mast"><p class="kicker"><a href="index.html">Player profiles</a> · tt_scout</p><h1>{e(name)}</h1>'
            f'<p class="meta">{a["n"]} recording{"s" if a["n"] != 1 else ""} · {sum(t["points"])} points counted · {since}</p></header>'
            f'{top}<section class="numbers"><div class="head"><h2>Career</h2><span>pooled over every counted recording</span></div>{facts_html}</section>'
            f'{crit}{tend}{trends}{h2h}{hist}{place}'
            '<p class="colophon">Built on this computer from tt_scout’s match records in profiles/matches; nothing is uploaded. Every number is tt_scout’s '
            'own count, so it carries each recording’s errors: see the verdict and the hand-check notes in each match report. Speeds are measured over '
            'the table, net to bounce; knee angles are as the camera sees them.</p></div></body></html>')
    (pathlib.Path(out_dir) / f"{a['key']}.html").write_text(page)


def build_pages(root=PROFILES):
    """Rebuild every player's page and the index from the match records. Returns {name: path}."""
    root = pathlib.Path(root); recs = load_records(root)
    names = player_names(recs); out = {}
    aggs = [aggregate(k, recs) for k in names]
    for a in aggs:
        player_page(a, root); out[a["name"]] = root / f"{a['key']}.html"
    items = ""
    for a in sorted(aggs, key=lambda a: a.get("last") or "", reverse=True):
        t = a["tot"]; port = _link(a["portrait"], root) if a.get("portrait") else None
        items += (f'<li>{f"<img src={chr(34)}{e(port)}{chr(34)} alt={chr(34)}{chr(34)}>" if port else "<span></span>"}'
                  f'<div><a href="{a["key"]}.html">{e(a["name"])}</a><p>{a["n"]} recording{"s" if a["n"] != 1 else ""} · last {e(_date(a.get("last")))}</p></div>'
                  f'<div class="r">{t["won"]}–{t["lost"]}<p>recordings won–lost</p></div></li>')
    index = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
             f'<title>Player profiles</title><style>{CSS}{PCSS}</style></head><body><div class="wrap">'
             f'<header class="mast"><p class="kicker">tt_scout</p><h1>Player profiles</h1><p class="meta">{len(aggs)} players · {len(recs)} recordings</p></header>'
             f'<section><ul class="players">{items}</ul></section>'
             '<p class="colophon">Each profile is rebuilt from the match records every time a recording is analysed with the players’ names.</p></div></body></html>')
    (root / "index.html").write_text(index)
    return out
