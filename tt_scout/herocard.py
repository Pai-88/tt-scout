"""Comic-book hero cards for the players, and the comic theme the website wears (the Players page and the profile pages).

Pure Python, no numpy: the web server imports it. Every number on a card is tt_scout's own measurement from the match records in
profiles/matches (the same counts the profile pages use); a stat with nothing measured behind it shows a dash and an empty bar.

    stats = crew_stats(records)                 # {slug: card stats}, each player's class chosen against the rest of the crew
    html = card_html(stats[slug], portrait_url, href, big=False)
"""
import html as _html, re
from statistics import median

e = _html.escape

# the power grid: key, label, how a value fills its bar (0 to 1), how it prints
GRID = [("speed", "Speed", lambda v: v / 90.0, lambda v: f"{v:.0f}", "km/h"),
        ("power", "Rally pace", lambda v: v / 40.0, lambda v: f"{v:.0f}", "km/h"),
        ("serve", "Serve", lambda v: v / 100.0, lambda v: f"{v:.0f}", "%"),
        ("ret", "Return", lambda v: v / 100.0, lambda v: f"{v:.0f}", "%"),
        ("rally", "Rally", lambda v: v / 100.0, lambda v: f"{v:.0f}", "%"),
        ("control", "Control", lambda v: v / 100.0, lambda v: f"{v:.0f}", "%")]

# the class a player gets, and the claim printed under his name when it is true: top of the crew in that stat ("joint" on a tie)
CLASSES = {"speed": ("The Rocket", "fastest shot of the crew"), "power": ("Heavy Hitter", "hardest rally pace of the crew"),
           "serve": ("Serve Master", "best on serve of the crew"), "ret": ("The Wall", "best on return of the crew"),
           "rally": ("Rally Boss", "best in long rallies of the crew"), "control": ("The Metronome", "fewest misses of the crew")}

COLOURS = ["var(--c-red)", "var(--c-blue)", "var(--c-green)", "var(--c-orange)", "var(--c-pink)"]


def slug(name):
    """= tt_scout.profiles.slug and webapp.slug (a profile's file name)."""
    return re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower()).strip("-") or "player"


def _pct(a, b):
    return 100.0 * a / b if b else None


KNEE_MIN_N = 3          # = pose.KNEE_MIN_N (pose needs numpy; this module must not): under this many contacts the knee is not measurable


def counts_in_career(record):
    """Whether a recording's numbers go into the career totals (profiles.counts_in_career is this function): everything except
    "unreliable", tt_scout's own verdict that the tracking could not be trusted."""
    return (record.get("verdict") or {}).get("level") != "unreliable"


def knee_values(record, name):
    """A player's gated knee angles in one record (profiles.record_match: the knee bend at contact as the camera sees it, rally
    forehands seen in profile). Records written before that rule (no knee_median in them) held a different quantity, the min of
    whatever legs were found at every hit, and contribute nothing: re-analyse the recording to get the knee back."""
    p = (record.get("posture") or {}).get(name) or {}
    return [v for v in (p.get("knee") or []) if v is not None] if "knee_median" in p else []


def knee_pooled(records, name):
    """(median, n) of the gated knee angles over every record that counts in the career, the one number the card, the profile's fact
    and the trend's points all derive from (the trend shows each record's own knee_median). The median is None under KNEE_MIN_N."""
    vals = [v for r in records if counts_in_career(r) for v in knee_values(r, next((n for n in r.get("players") or [] if slug(n) == slug(name)), name))]
    return (median(vals) if len(vals) >= KNEE_MIN_N else None, len(vals))


def player_stats(records, name):
    """One player's card stats over every record he appears in (the knee: over the records that count in the career, knee_pooled)."""
    k = slug(name)
    mine = [r for r in records if any(slug(n) == k for n in (r.get("players") or []))]
    t = dict(matches=len(mine), won=0, lost=0, points_won=0, points=0, serve=[0, 0], ret=[0, 0], rally=[0, 0], missed=0,
             fastest=None, rally_kmh=[], last=None, portrait=None, name=name)
    for r in mine:
        n = next(n for n in r["players"] if slug(n) == k)
        pts = r.get("points") or []
        t["won"] += r.get("winner") == n
        t["lost"] += r.get("winner") not in (None, n)
        t["points_won"] += sum(1 for q in pts if q.get("winner") == n); t["points"] += len(pts)
        s = (r.get("stats") or {}).get(n) or {}
        for key, src in (("serve", "serve"), ("ret", "receive")):
            w, of = (s.get(src) or [0, 0])[:2]
            t[key][0] += w; t[key][1] += of
        rw, rl = ((s.get("phases") or {}).get("rally") or [0, 0])[:2]
        t["rally"][0] += rw; t["rally"][1] += rw + rl
        t["missed"] += s.get("missed_table") or 0
        if s.get("fastest_kmh") and not s.get("fastest_est"):
            t["fastest"] = max(t["fastest"] or 0, s["fastest_kmh"])
        if s.get("rally_kmh"):
            t["rally_kmh"].append(s["rally_kmh"])
        if not t["last"] or (r.get("recorded") or "") >= t["last"]:
            t["last"] = r.get("recorded"); t["name"] = n
            t["portrait"] = (r.get("portrait") or {}).get(n) or t["portrait"]
    vals = dict(speed=t["fastest"], power=median(t["rally_kmh"]) if t["rally_kmh"] else None,
                serve=_pct(*t["serve"]), ret=_pct(*t["ret"]), rally=_pct(*t["rally"]),
                control=100.0 - _pct(t["missed"], t["points"]) if t["points"] else None)
    knee, knee_n = knee_pooled(mine, name)
    return dict(slug=k, name=t["name"], matches=t["matches"], won=t["won"], lost=t["lost"], points_won=t["points_won"], points=t["points"],
                knee=knee, knee_n=knee_n, portrait=t["portrait"], last=t["last"], vals=vals)


def crew_stats(records):
    """Every player's card stats, each with a class of his own: the stat where he stands furthest above the others' average, handed
    out greedily over the whole crew so no two players share one (a stat counts only when at least two players have it measured)."""
    names = {}
    for r in records:
        for n in r.get("players") or []:
            names.setdefault(slug(n), n)
    out = {k: player_stats(records, n) for k, n in names.items()}
    leads = []
    for i, (k, p) in enumerate(sorted(out.items(), key=lambda kv: kv[0])):
        p["colour"] = COLOURS[i % len(COLOURS)]
        for key, label, fill, _, _ in GRID:
            others = [q["vals"][key] for j, q in out.items() if j != k and q["vals"][key] is not None]
            v = p["vals"][key]
            if v is not None and others:
                top = v >= max(others)
                tie = v == max(others)
                leads.append((top, not tie, fill(v) - sum(fill(x) for x in others) / len(others), k, key, tie, label))
    taken = set()
    for top, _, lead, k, key, tie, label in sorted(leads, reverse=True):  # outright top of the crew first, then a shared top, then by lead
        if "cls" not in out[k] and key not in taken:
            title, best = CLASSES[key]
            why = (("joint " if tie else "") + best) if top else (f"{label} above the crew's average" if lead > 0 else "strongest stat so far")
            out[k]["cls"] = (title, why); taken.add(key)
    for p in out.values():
        p.setdefault("cls", ("Rookie", "not enough measured yet"))
    return out


def card_html(p, portrait_url, href, big=False):
    """One hero card. portrait_url and href as the page that shows it needs them (the web server's paths or relative ones)."""
    rows = ""
    for key, label, fill, fmt, unit in GRID:
        v = p["vals"][key]
        f = 0.0 if v is None else max(0.0, min(1.0, fill(v)))
        rows += (f'<div><dt>{label}</dt><dd><span class="h-bar"><i style="--v:{f:.3f}"></i></span>'
                 f'<b>{"–" if v is None else fmt(v)}<small>{unit if v is not None else ""}</small></b></dd></div>')
    art = f'<img src="{e(portrait_url)}" alt="{e(p["name"])} at contact" loading="lazy">' if portrait_url else '<span class="h-noart"></span>'
    cls, why = p.get("cls") or ("Rookie", "")
    tag = "div" if big else "a"
    link = "" if big else f' href="{e(href)}"'
    return (f'<{tag} class="hcard{" big" if big else ""}"{link} style="--hc:{p.get("colour", "var(--c-orange)")}">'
            f'<div class="h-fr"><div class="h-hd"><span class="h-cls" title="{e(why)}">{e(cls)}</span>'
            f'<span class="h-rec"><b>{p["won"]}–{p["lost"]}</b><small>won–lost</small></span></div>'
            f'<div class="h-art">{art}</div><h3 class="h-nm">{e(p["name"])}</h3><p class="h-why">{e(why)}</p>'
            f'<dl class="h-grid">{rows}</dl>'
            f'<p class="h-ft">{p["matches"]} match{"es" if p["matches"] != 1 else ""} · {p["points"]} points · '
            f'{knee_line(p)}</p></div></{tag}>')


def knee_line(p):
    """The card's knee, the same gated quantity as the profile's fact: the median with its count, or that it is not measurable."""
    if p.get("knee") is not None and (p.get("knee_n") or 0) >= KNEE_MIN_N:
        return f'knees {p["knee"]:.0f}° as the camera sees it ({p["knee_n"]} contacts)'
    return "knee not measurable from this camera"


# the comic theme: the match clips' HUD in print (cream panels, navy ink, process colours, Ben-Day dots, hard offset shadows),
# laid over the pages' own tokens so every page element follows. Impact for the lettering, as in the clips; Avenir Next for text.
THEME = r"""
:root{
  color-scheme:light;
  /* a 1960s four-colour comic as it looks on the newsstand now: mustard stock under faded rust Ben-Day dots, off-white caption
     panels, black ink, the primaries a little dulled by the paper */
  --page:oklch(77% 0.1 80); --paper:oklch(95% 0.022 88); --paper-2:oklch(90% 0.04 86); --ink:oklch(17% 0.012 60); --ink-2:oklch(25% 0.012 60); --ink-3:oklch(33% 0.014 60);
  --line:color-mix(in oklch, var(--ink) 22%, transparent); --line-2:color-mix(in oklch, var(--ink) 42%, transparent);
  --yellow:oklch(85% 0.13 90); --c-red:oklch(52% 0.16 29); --c-blue:oklch(49% 0.11 250); --c-green:oklch(56% 0.12 150);
  --c-orange:oklch(63% 0.14 55); --c-pink:oklch(55% 0.14 355);
  --p1:var(--c-red); --p2:var(--c-blue); --ball:oklch(69% 0.17 52);
  --good:oklch(52% 0.14 148); --care:oklch(58% 0.14 70); --bad:oklch(52% 0.2 29);
  --serif:Impact,'Haettenschweiler','Arial Narrow Bold','Avenir Next Condensed',sans-serif;
  --sans:'Comic Sans MS','Chalkboard SE','Comic Neue','Marker Felt',cursive;          /* comic lettering for every text, as in the clips */
  --dots:radial-gradient(color-mix(in oklch, var(--c-red) 17%, transparent) 1.1px, transparent 1.5px);
}
body{background-color:var(--page);background-image:var(--dots);background-size:8px 8px;color:var(--ink)}
body,button,input,select,textarea,table{font-family:var(--sans)}
/* the content sits on one off-white panel, like a page inside the comic: text is never on the dots */
.wrap{background:var(--paper);border:4px solid var(--ink);box-shadow:10px 10px 0 var(--ink);margin-top:clamp(1rem,3vw,2.2rem);margin-bottom:3rem}
@media (max-width:40rem){.wrap{border-width:3px;box-shadow:5px 5px 0 var(--ink);margin-left:.5rem;margin-right:.5rem}}
h1{font-family:var(--serif);font-weight:400;letter-spacing:.02em;color:var(--c-red);-webkit-text-stroke:2px var(--ink);paint-order:stroke fill;
  text-shadow:4px 4px 0 var(--ink)}
h2{font-family:var(--serif);font-weight:400;letter-spacing:.05em;color:var(--ink)}
a{color:var(--ink)}
.top{background:var(--c-red);border-bottom:5px solid var(--ink);backdrop-filter:none}
.brand{font-family:var(--serif);font-weight:400;font-size:1.6rem;letter-spacing:.05em;color:var(--c-red);background:var(--yellow);border:3px solid var(--ink);
  padding:.05rem .6rem;box-shadow:4px 4px 0 var(--ink);transform:rotate(-2deg);display:inline-block;-webkit-text-stroke:1.5px var(--ink);paint-order:stroke fill}
.nav a{font-family:var(--serif);font-weight:400;font-size:1.15rem;letter-spacing:.06em;color:var(--yellow);border-bottom-width:4px;
  -webkit-text-stroke:1px var(--ink);paint-order:stroke fill;text-shadow:2px 2px 0 var(--ink)}
.nav a:hover{color:var(--paper)} .nav a[aria-current="page"]{color:var(--paper);border-bottom-color:var(--yellow)}
.top .priv{color:var(--yellow);font-weight:600}
.lede,.meta,.mast .meta{color:var(--ink)}
button,.btn{font-family:var(--serif);font-weight:400;font-size:1rem;letter-spacing:.06em;border:3px solid var(--ink);border-radius:0;
  background:var(--paper);box-shadow:4px 4px 0 var(--ink);transition:transform .15s var(--ease),box-shadow .15s var(--ease),background .15s var(--ease)}
button:hover:not(:disabled),.btn:hover{background:var(--yellow);color:var(--ink);transform:translate(-2px,-2px);box-shadow:6px 6px 0 var(--ink)}
button:active:not(:disabled),.btn:active{transform:translate(3px,3px);box-shadow:1px 1px 0 var(--ink)}
.primary{background:var(--c-red);color:var(--yellow)} .primary:hover:not(:disabled){background:var(--c-red);color:var(--yellow)}
.card .pic,.player video{border:3px solid var(--ink);box-shadow:6px 6px 0 var(--ink);border-radius:0}
.card .pic img{filter:contrast(1.1) saturate(1.15) sepia(.12)}
.card h3,.people a.nm,.now b,.stages li .lab{font-family:var(--serif);font-weight:400}
.games th,.earlier th{color:var(--ink);border-bottom:3px solid var(--ink)} .games td.sc{font-family:var(--serif)}
.games tbody tr:hover,.earlier tbody tr:hover{background:color-mix(in oklch, var(--yellow) 55%, transparent)}
.drop{border:3px dashed var(--ink);background:var(--paper)}
.drop:hover,.drop:focus-within,body.dragging .drop{background:var(--yellow)}
.empty{border:3px dashed var(--ink-3);background:var(--paper)}
pre{background:var(--paper-2);color:var(--ink)}
/* the coach's critique and the scout's tips: comic panels, a red count burst, the fix on a yellow tag */
.tiplist{list-style:none;margin:1rem 0 0;padding:0;display:grid;gap:1.15rem}
.tiplist li,.tiplist li:first-child{border:3px solid var(--ink);background:var(--paper);box-shadow:6px 6px 0 var(--ink);padding:.95rem 1.05rem;
  grid-template-columns:5.6rem minmax(0,1fr);column-gap:1.1rem;row-gap:.25rem}
.tiplist .for{grid-row:1/span 3;align-self:start;justify-items:center;text-align:center;background:var(--c-red);border:3px solid var(--ink);
  padding:.4rem .25rem .35rem;transform:rotate(-4deg);box-shadow:3px 3px 0 var(--ink)}
.tiplist .for b{font-family:var(--serif);font-weight:400;font-size:1.35rem;line-height:1;letter-spacing:.02em;color:var(--yellow);overflow:visible}
.tiplist .for small,.tiplist .early .for small{font-family:var(--sans);font-size:.62rem;font-weight:700;letter-spacing:.04em;font-style:normal;color:var(--paper)}
.tiplist h3{font-family:var(--serif);font-weight:400;font-size:clamp(1.35rem,1.1rem + .9vw,1.8rem);letter-spacing:.02em;line-height:1.05;text-transform:uppercase}
.tiplist .ev{color:var(--ink);font-size:.97rem;line-height:1.45}
.tiplist .ev small{display:inline-block;margin-top:.5rem;padding:.12rem .5rem;background:var(--yellow);border:2px solid var(--ink);color:var(--ink);font-size:.86rem;font-weight:700}
.tiplist .see{color:var(--ink-2)}
/* habits: a caption box each */
.tend li,.tend li:first-child{font-family:var(--sans);font-weight:700;font-size:1rem;line-height:1.45;color:var(--ink);border:3px solid var(--ink);
  background:color-mix(in oklch, var(--yellow) 42%, var(--paper));box-shadow:4px 4px 0 var(--ink);padding:.7rem .95rem;margin:.95rem 0}
"""

CSS = THEME + r"""
/* hero cards */
.heroes{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,14.5rem),1fr));gap:clamp(1.8rem,3.5vw,2.8rem) clamp(1.2rem,2.4vw,2rem);
  margin:1.8rem 0 .6rem;align-items:start}
.hcard{--hc:var(--c-orange);display:block;color:var(--ink);text-decoration:none;transform:rotate(var(--tilt,-1.2deg));transition:transform .35s var(--ease);
  animation:slam .55s var(--ease) both}
.heroes .hcard:nth-child(2n){--tilt:1deg} .heroes .hcard:nth-child(3n){--tilt:-.5deg} .heroes .hcard:nth-child(4n){--tilt:.6deg}
.heroes .hcard:nth-child(2){animation-delay:.07s} .heroes .hcard:nth-child(3){animation-delay:.14s} .heroes .hcard:nth-child(4){animation-delay:.21s}
.heroes .hcard:nth-child(n+5){animation-delay:.28s}
@keyframes slam{from{opacity:0;transform:rotate(var(--tilt,-1.2deg)) scale(1.12) translateY(-10px)}}
a.hcard:hover,a.hcard:focus-visible{transform:rotate(0deg) translateY(-6px);outline:none}
a.hcard:hover .h-fr,a.hcard:focus-visible .h-fr{box-shadow:11px 11px 0 var(--ink)}
.hcard .h-fr{position:relative;display:grid;gap:.55rem;padding:.6rem;background-color:var(--hc);background-size:7px 7px;
  background-image:radial-gradient(color-mix(in oklch, var(--ink) 16%, transparent) .9px, transparent 1.3px);
  border:4px solid var(--ink);box-shadow:7px 7px 0 var(--ink);transition:box-shadow .35s var(--ease)}
.hcard .h-hd{display:flex;justify-content:space-between;align-items:flex-start;gap:.6rem}
.hcard .h-cls{font-family:var(--serif);font-size:1rem;line-height:1.1;letter-spacing:.05em;text-transform:uppercase;color:var(--paper);background:var(--ink);
  padding:.2rem .55rem .15rem;transform:skewX(-10deg);margin-left:.15rem}
.hcard .h-rec{display:grid;justify-items:center;font-family:var(--serif);background:var(--yellow);border:3px solid var(--ink);padding:.1rem .45rem .15rem;
  transform:rotate(4deg);box-shadow:3px 3px 0 var(--ink)}
.hcard .h-rec b{font-weight:400;font-size:1.45rem;line-height:1} .hcard .h-rec small{font-family:var(--sans);font-size:.55rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase}
.hcard .h-art{position:relative;aspect-ratio:3/4;overflow:hidden;border:3px solid var(--ink);background:var(--paper-2)}
.hcard .h-art img{width:100%;height:100%;object-fit:cover;display:block;filter:contrast(1.18) saturate(1.2) sepia(.14);transition:transform .6s var(--ease)}
a.hcard:hover .h-art img{transform:scale(1.05)}
.hcard .h-art::after{content:"";position:absolute;inset:0;pointer-events:none;mix-blend-mode:multiply;opacity:.55;
  background:radial-gradient(color-mix(in oklch, var(--ink) 55%, transparent) .9px, transparent 1.3px) 0 0/5px 5px}
.hcard .h-noart{display:block;width:100%;height:100%;background:var(--dots) 0 0/8px 8px}
.hcard .h-nm{margin:-3.1rem .4rem 0;position:relative;z-index:1;font-family:var(--serif);font-weight:400;font-size:clamp(2.3rem,1.7rem + 1.5vw,2.9rem);
  line-height:.9;letter-spacing:.02em;text-transform:uppercase;color:var(--yellow);-webkit-text-stroke:2.5px var(--ink);paint-order:stroke fill;
  text-shadow:4px 4px 0 var(--ink);transform:rotate(-3deg);transform-origin:left bottom;overflow-wrap:anywhere}
.hcard .h-why{margin:.1rem 0 0;justify-self:start;font-family:var(--sans);font-size:.74rem;font-weight:700;letter-spacing:.04em;text-transform:uppercase;
  color:var(--paper);background:var(--ink);padding:.22rem .5rem;transform:skewX(-6deg)}
.hcard .h-grid{margin:0;display:grid;gap:.32rem;background:var(--paper);border:3px solid var(--ink);padding:.6rem .65rem}
.hcard .h-grid div{display:grid;grid-template-columns:4.9rem minmax(0,1fr);align-items:center;gap:.4rem}
.hcard .h-grid dt{font-family:var(--serif);font-size:.86rem;letter-spacing:.05em;text-transform:uppercase;color:var(--ink)}
.hcard .h-grid dd{margin:0;display:grid;grid-template-columns:minmax(0,1fr) 3.7rem;align-items:center;gap:.4rem}
.hcard .h-bar{position:relative;height:.78rem;border:2px solid var(--ink);background:var(--paper-2);overflow:hidden}
.hcard .h-bar i{position:absolute;inset:0;background:var(--hc);transform-origin:left;transform:scaleX(var(--v));animation:fill .9s var(--ease) .25s both}
.hcard .h-bar::after{content:"";position:absolute;inset:0;background:repeating-linear-gradient(90deg,transparent 0 calc(10% - 2px),var(--ink) calc(10% - 2px) 10%)}
@keyframes fill{from{transform:scaleX(0)}}
.hcard .h-grid b{font-family:var(--serif);font-weight:400;font-size:1.12rem;line-height:1;text-align:right;white-space:nowrap}
.hcard .h-grid b small{font-family:var(--sans);font-size:.62rem;font-weight:700;letter-spacing:.06em;padding-left:.15rem}
.hcard .h-ft{margin:0;justify-self:start;font-size:.74rem;font-weight:700;color:var(--ink);background:var(--paper);border:2px solid var(--ink);padding:.15rem .45rem}
.hcard.big{max-width:25rem;transform:rotate(-1.5deg);animation:none}
.hcard.big .h-nm{font-size:clamp(3.2rem,2.4rem + 2.4vw,4.4rem);margin-top:-4.2rem}
@media (prefers-reduced-motion:reduce){.hcard,.hcard .h-bar i{animation:none}}
"""

# the match report in the comic theme: report_html puts this after its own CSS and THEME
REPORT = r"""
.topbar{background:var(--c-red);border-bottom:5px solid var(--ink)}
.topbar .sc{font-family:var(--serif);font-weight:400;font-size:1rem;letter-spacing:.06em;color:var(--yellow);-webkit-text-stroke:1px var(--ink);paint-order:stroke fill;
  text-shadow:2px 2px 0 var(--ink)}
.topbar .sc b{font-family:var(--serif);font-weight:400;font-size:1.3rem;color:var(--ink);background:var(--yellow);border:2px solid var(--ink);padding:.08rem .45rem;
  -webkit-text-stroke:0;text-shadow:none}
.topbar .sc em{background:var(--ink);width:3px} .topbar .sc i{border:2px solid var(--ink);width:.75rem;height:.75rem}
.topbar nav a{font-family:var(--serif);font-weight:400;font-size:.98rem;letter-spacing:.06em;color:var(--yellow);-webkit-text-stroke:1px var(--ink);
  paint-order:stroke fill;text-shadow:2px 2px 0 var(--ink);border-radius:0}
.topbar nav a:hover{color:var(--paper)}
.topbar nav a[aria-current="true"]{color:var(--ink);background:var(--yellow);-webkit-text-stroke:0;text-shadow:none;box-shadow:inset 0 0 0 2px var(--ink)}
.head{border-bottom:3px solid var(--ink);align-items:end}
.head h2{display:inline-block;font-family:var(--serif);font-weight:400;font-size:clamp(1.5rem,1.2rem + 1.2vw,2.1rem);line-height:1.05;letter-spacing:.04em;
  text-transform:uppercase;color:var(--ink);background:var(--yellow);border:3px solid var(--ink);box-shadow:4px 4px 0 var(--ink);padding:.2rem .7rem .12rem;
  transform:rotate(-1.2deg);margin-bottom:.5rem}
.head span{color:var(--ink-2);font-weight:700}
.wrap img,.wrap video,.wrap figure > svg{border:3px solid var(--ink);box-shadow:5px 5px 0 var(--ink);border-radius:0}
.wrap img.map,.wrap .key i,.wrap svg img{border:0;box-shadow:none}
table th{color:var(--ink);border-bottom:3px solid var(--ink)} table td{border-color:var(--line)}
tbody tr:hover{background:color-mix(in oklch, var(--yellow) 50%, transparent)}
.ledger tr[aria-current="true"],.ledger tr[aria-current="true"]:hover{background:var(--yellow)}
.side .num{font-family:var(--serif);font-weight:400;color:var(--ink)}
.side.won .num{color:var(--c-red);-webkit-text-stroke:3px var(--ink);paint-order:stroke fill;text-shadow:6px 6px 0 var(--ink)}
.side .name{font-family:var(--serif);font-weight:400;letter-spacing:.05em}
.watchbar a{font-family:var(--serif);font-weight:400;font-size:1.05rem;letter-spacing:.05em;border:3px solid var(--ink);border-radius:0;background:var(--paper);
  box-shadow:4px 4px 0 var(--ink);transition:transform .15s var(--ease),box-shadow .15s var(--ease)}
.watchbar a.first{background:var(--c-red);color:var(--yellow)}
.watchbar a:hover{background:var(--yellow);color:var(--ink);transform:translate(-2px,-2px);box-shadow:6px 6px 0 var(--ink)}
.play{border:2px solid var(--ink);background:var(--yellow);color:var(--ink)}
.kicker{font-family:var(--serif);font-weight:400;letter-spacing:.12em;color:var(--ink)}
.colophon{font-size:.9rem;color:var(--ink-2)}
"""
