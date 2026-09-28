"""One self-contained HTML report a coach can open: confidence, score summary, charts, numbers, and every point as a clip.

    make_html(out_dir, title, points, quality, video=None, clips=True)
Reads report.png and stats.txt from out_dir (written by run.analyse), cuts clips/point_NNN.mp4 with ffmpeg when `video` is
given, and writes out_dir/report.html. Nothing is uploaded anywhere; the page works offline.
"""
import base64, html, pathlib, shutil, subprocess

BADGE = {"good": ("#1a7f37", "Good"), "degraded": ("#9a6700", "Use with care"), "unreliable": ("#cf222e", "Unreliable")}
ENDING = {"double_bounce": "double bounce", "not_returned": "not returned", "long": "long or wide"}


def mmss(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


def cut_clips(video, points, out_dir, lead_s=1.0, tail_s=0.8, height=540, max_points=250):
    """clips/point_NNN.mp4 per point, re-encoded small (precise cuts; stream copy would snap to keyframes)."""
    if not shutil.which("ffmpeg"):
        return {}
    clips = pathlib.Path(out_dir) / "clips"; clips.mkdir(exist_ok=True)
    done = {}
    for p in points[:max_points]:
        ev_t = [e["t"] for e in p.get("events", [])]
        t0 = max(0.0, p["start_t"] - lead_s); t1 = (max(ev_t) if ev_t else p["end_t"]) + tail_s
        dst = clips / f"point_{p['id']:03d}.mp4"
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.3f}", "-i", str(video), "-t", f"{t1 - t0:.3f}",
                            "-vf", f"scale=-2:{height},fps=30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "27", "-an", str(dst)],
                           capture_output=True)
        if r.returncode == 0 and dst.exists():
            done[p["id"]] = f"clips/{dst.name}"
    return done


def annotated_clips(video, points, out_dir, table, track, events, obs, fps, max_points=250, comic=True, scoreboard=True, tips=None, show_table=False,
                    pose=None, analysis=None, critiques=None, critiques_after=None, speeds=None, shots=None, hide=None, only=None):
    """clips/point_NNN.mp4 with the tracker's output drawn on (see annotate.py). tips = {point id: Tip} from tips.assign(): that clip
    opens earlier, as far as the dead time since the previous point allows, and shows the note before the serve."""
    from .annotate import render_point_clip, running_scores, overlay_context
    from .match_stats import moments as moments_of
    scores = running_scores(points)
    moments = moments_of(points, shots)                           # the celebration each point earns (word, colours, intensity)
    from .match_stats import confirm as confirm_point
    context = overlay_context(video, table)                        # the empty hall for finding heads, and the landing map's size
    clips = pathlib.Path(out_dir) / "clips"; clips.mkdir(exist_ok=True)
    by_frame = {}
    for o in obs:
        by_frame.setdefault(o["frame"], []).append(o)
    done, prev_end = {}, 0.0
    for p in points[:max_points]:
        dst = clips / f"point_{p['id']:03d}.mp4"
        room = p["start_t"] - prev_end - 0.2; prev_end = max(prev_end, p["end_t"])
        if only and p["id"] not in only:
            continue
        if render_point_clip(video, fps, table, track, events, by_frame, dict(p, _score=scores[p["id"]]), dst, comic=comic, scoreboard=scoreboard,
                             tip=(tips or {}).get(p["id"]), max_lead_s=room, context=context, show_table=show_table, pose=pose, analysis=analysis,
                             critique=(critiques or {}).get(p["id"]), critique_after=(critiques_after or {}).get(p["id"]), speeds=speeds,
                             moment=moments.get(p["id"]), confirmed=confirm_point(p, events, track, table, fps), hide=hide, shots=shots):
            done[p["id"]] = f"clips/{dst.name}"
    return done


from .herocard import THEME, REPORT as COMIC                                # the website's 1960s comic theme, over the tokens below
WEB_FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
             '<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;500;600;700'
             '&family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet">')

CSS = r"""
:root{
  color-scheme:dark;
  /* the black edition (2026-09-26): a near-black ground with a cool hint,
     off-white ink on three greys, colour kept for the data only (the two players, the table, the ball) */
  --paper:oklch(14.5% 0.004 250); --paper-2:oklch(19.5% 0.006 250); --ink:oklch(96% 0.004 90); --ink-2:oklch(80% 0.006 250); --ink-3:oklch(63% 0.008 250);
  --table:oklch(44% 0.085 232); --table-line:oklch(92% 0.01 90); --ball:oklch(77% 0.16 58); --rust:oklch(70% 0.008 250);
  --good:oklch(78% 0.14 152); --care:oklch(84% 0.14 88); --bad:oklch(70% 0.18 28);
  --line:color-mix(in oklch, var(--ink) 11%, transparent); --line-2:color-mix(in oklch, var(--ink) 24%, transparent);
  --serif:'Avenir Next Condensed','Barlow Condensed','Arial Narrow',sans-serif;     /* display: headings and every big number */
  --sans:'Avenir Next','Manrope','Segoe UI','Helvetica Neue',Arial,sans-serif;
  --ease:cubic-bezier(.16,1,.3,1); --gutter:clamp(1.1rem,4.5vw,3.5rem);
  --p1:oklch(75% 0.15 58); --p2:oklch(72% 0.12 238);            /* the two players, the colours their skeletons and balloons have in the clips */
}
@media print{:root{
  color-scheme:light;
  --paper:oklch(99% 0 0); --paper-2:oklch(95% 0.005 85); --ink:oklch(20% 0.02 250); --ink-2:oklch(34% 0.015 250); --ink-3:oklch(46% 0.012 250);
  --table:oklch(34% 0.07 225); --table-line:oklch(93% 0.02 85); --rust:oklch(46% 0.012 250);
  --good:oklch(43% 0.09 155); --care:oklch(54% 0.12 78); --bad:oklch(48% 0.16 28); --p1:oklch(64% 0.155 55); --p2:oklch(55% 0.12 240);
}}
*,*::before,*::after{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);font-size:clamp(.97rem,.93rem + .2vw,1.06rem);line-height:1.55;
  -webkit-font-smoothing:antialiased;font-variant-numeric:lining-nums}
.wrap{max-width:74rem;margin:0 auto;padding:clamp(1.5rem,4vw,3rem) var(--gutter) 4rem}
h1,h2{font-family:var(--serif);font-weight:400;margin:0;letter-spacing:-.012em}
.kicker{font-size:.76rem;font-weight:600;letter-spacing:.19em;text-transform:uppercase;color:var(--rust);margin:0}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
a{color:var(--rust);text-underline-offset:.18em;text-decoration-thickness:1px}
:focus-visible{outline:2px solid var(--ball);outline-offset:3px;border-radius:2px}

/* masthead */
.mast{border-bottom:1.5px solid var(--ink);padding-bottom:1.1rem;position:relative}
.mast::after{content:"";position:absolute;left:0;right:0;bottom:-5px;border-bottom:1px solid var(--line-2)}
.mast h1{font-size:clamp(2.3rem,1.2rem + 5vw,4.6rem);line-height:.98;margin:.55rem 0 .5rem;font-variation-settings:"opsz" 144;overflow-wrap:anywhere}
.mast h1 i{font-style:italic;font-weight:300;color:var(--ink-3);padding:0 .12em}
.meta{margin:0;color:var(--ink-3);font-size:.9rem}

/* scoreline: the score set as type, the verdict as an inked stamp */
.score{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:clamp(1rem,4vw,3.5rem);padding:clamp(1.6rem,4vw,2.8rem) 0 clamp(1.2rem,3vw,2rem)}
.side{display:grid;gap:.15rem;min-width:0}
.side.r{text-align:right}
.side .name{font-size:.8rem;font-weight:700;letter-spacing:.17em;text-transform:uppercase;color:var(--ink-2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.side .num{font-family:var(--serif);font-size:clamp(4rem,2rem + 9vw,8.5rem);line-height:.86;font-variation-settings:"opsz" 144;letter-spacing:-.03em}
.side .sub{color:var(--ink-3);font-size:.88rem}
.side.won .num{color:var(--ink)} .side:not(.won) .num{color:var(--ink-3)}
.score.unreliable .num{color:var(--ink-3);opacity:.5} .score.unreliable .sub::before{content:"provisional \00b7  ";color:var(--bad);font-weight:600}
.stamp{--c:var(--good);justify-self:center;width:clamp(7.2rem,15vw,9.4rem);aspect-ratio:1;border-radius:50%;display:grid;place-content:center;text-align:center;color:var(--c);
  border:2px solid var(--c);box-shadow:inset 0 0 0 4px var(--paper),inset 0 0 0 5px var(--c);transform:rotate(-7deg);line-height:1.05}
.stamp.degraded{--c:var(--care)} .stamp.unreliable{--c:var(--bad)}
.stamp span{font-size:.62rem;font-weight:700;letter-spacing:.22em;text-transform:uppercase}
.stamp b{font-family:var(--serif);font-weight:400;font-style:italic;font-size:clamp(1.25rem,2.4vw,1.65rem);margin:.18rem 0 .12rem}
.stamp small{font-size:.74rem;font-weight:600;letter-spacing:.06em}
.verdict{display:grid;grid-template-columns:minmax(0,38rem);gap:.5rem;padding-bottom:clamp(1.6rem,4vw,2.6rem);border-bottom:1px solid var(--line)}
.verdict p{margin:0;font-family:var(--serif);font-size:clamp(1.1rem,1rem + .45vw,1.32rem);line-height:1.38;font-variation-settings:"opsz" 30}
.verdict ul{margin:.2rem 0 0;padding-left:1.1rem;color:var(--ink-2);font-size:.92rem} .verdict li{margin:.15rem 0}
.verdict .note{color:var(--ink-3);font-size:.86rem;font-family:var(--sans)}

/* sections */
section{padding-top:clamp(2.2rem,5vw,3.6rem)}
.head{display:flex;align-items:baseline;justify-content:space-between;gap:1rem;border-bottom:1.5px solid var(--ink);padding-bottom:.6rem;margin-bottom:.2rem}
.head h2{font-size:clamp(1.5rem,1.2rem + 1.2vw,2.05rem);line-height:1.05}
.head span{color:var(--ink-3);font-size:.86rem;text-align:right}

/* ledger + player */
.points{display:grid;gap:clamp(1.2rem,3vw,2.4rem);grid-template-columns:minmax(0,1fr)}
.points>*{min-width:0} .ledger{overflow-x:auto;-webkit-overflow-scrolling:touch}
@media (min-width:62rem){.points{grid-template-columns:minmax(0,1.42fr) minmax(0,1fr);align-items:start}.stage{position:sticky;top:1.2rem;order:2}}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums lining-nums}
th{font-size:.7rem;font-weight:700;letter-spacing:.13em;text-transform:uppercase;color:var(--ink-3);text-align:left;padding:.85rem .42rem .5rem;white-space:nowrap}
td{padding:.58rem .42rem;border-top:1px solid var(--line);vertical-align:baseline}
.ledger td,.ledger th{white-space:nowrap} .ledger td:last-child,.ledger th:last-child{width:2.6rem;padding-right:.2rem;text-align:right}
.ledger tbody tr{transition:background .18s var(--ease)}
.ledger tbody tr:hover{background:color-mix(in oklch, var(--ink) 5%, transparent)}
.ledger tr[aria-current="true"]{background:color-mix(in oklch, var(--ball) 15%, transparent)}
.ledger td:first-child{font-family:var(--serif);font-size:1.12rem;color:var(--ink-3);width:2.4rem;position:relative}
.ledger tr[aria-current="true"] td:first-child{color:var(--ink)}
.ledger tr[aria-current="true"] td:first-child::before{content:"";position:absolute;left:-.55rem;top:50%;width:.5rem;height:.5rem;border-radius:50%;background:var(--ball);transform:translateY(-50%)}
.ledger .won{font-weight:700} .ledger .unsure td{color:var(--ink-3)} .ledger .unsure .won{font-weight:500;font-style:italic}
.ledger .sc{font-family:var(--serif);color:var(--ink-2)} .ledger tr.game td{padding:1.1rem .55rem .35rem;border-top:1.5px solid var(--ink);font-size:.7rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:var(--rust)}
.ledger tr.game small{font-weight:500;letter-spacing:.04em;text-transform:none;color:var(--ink-3);padding-left:.6rem} .ledger tr.game:hover{background:none} .ledger tr.game td:last-child{text-align:left;width:auto}
.ledger .t{color:var(--ink-3);white-space:nowrap} .ledger .n{text-align:right;padding-right:1rem}
.ledger td:nth-child(5){white-space:normal;min-width:6.2rem}
.play{font:inherit;font-size:.72rem;font-weight:700;letter-spacing:.13em;text-transform:uppercase;color:var(--ink);background:transparent;border:1.5px solid var(--ink);
  border-radius:999px;padding:.42rem .8rem .42rem .7rem;min-height:2.1rem;cursor:pointer;display:inline-flex;align-items:center;gap:.45rem;white-space:nowrap;
  transition:transform .25s var(--ease),background .2s,color .2s}
.play{width:2.15rem;height:2.15rem;min-height:0;padding:0;justify-content:center;border-radius:50%}
.play span{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}
.play::before{content:"";border-left:.62em solid currentColor;border-block:.4em solid transparent;margin-left:.16em}
.play:hover{background:var(--ink);color:var(--paper);transform:translateY(-1px)} .play:active{transform:translateY(0) scale(.97)}
tr[aria-current="true"] .play{background:var(--ink);color:var(--paper)}
@media (pointer:coarse){.play{width:2.75rem;height:2.75rem}}
.stage{container-type:inline-size}
/* the picture sits at the top of the player and a 3.5rem strip is left under it (Chrome's control row reaches ~50px up): a browser draws its controls over that strip, not over the
   speed gauge and notes along the bottom of the clip, whenever the clip is paused or hovered */
.stage video{width:100%;aspect-ratio:16/9;background:oklch(14% 0.015 165);display:block;border-radius:2px;object-fit:contain;object-position:50% 0}
@supports (height:1cqw){.stage video{aspect-ratio:auto;height:calc(56.25cqw + 3.5rem)}}
.stage .cap{display:flex;justify-content:space-between;gap:1rem;align-items:baseline;padding:.7rem 0 .55rem;border-bottom:1px solid var(--line);min-height:2.6rem}
.stage .cap b{font-family:var(--serif);font-weight:400;font-size:1.18rem} .stage .cap span{color:var(--ink-3);font-size:.86rem;text-align:right}
.legend{list-style:none;margin:.7rem 0 0;padding:0;display:grid;grid-template-columns:repeat(auto-fit,minmax(11.5rem,1fr));gap:.3rem 1.2rem;color:var(--ink-2);font-size:.84rem}
.legend li{display:flex;gap:.55rem;align-items:center} .legend i{flex:none;width:1.15rem;height:.62rem;border-radius:999px;border:2px solid var(--k);background:transparent}
.legend i.f{background:var(--k)} .note2{margin:.8rem 0 0;color:var(--ink-3);font-size:.82rem;max-width:34rem} .empty{border:1.5px dashed var(--line-2);border-radius:2px;padding:1.4rem;color:var(--ink-2);font-size:.93rem}
.empty code{font-size:.86em}

/* scout tips: a ruled list; the suggestion set as one serif line, the counts under it, the points that show it as small round keys */
.tiplist{list-style:none;margin:0;padding:0}
.tiplist li{display:grid;grid-template-columns:minmax(6.5rem,9.5rem) minmax(0,1fr);column-gap:clamp(1rem,3vw,2.4rem);row-gap:.18rem;padding:1.05rem 0 1.1rem;border-top:1px solid var(--line)}
.tiplist li:first-child{border-top:0}
.tiplist .for{margin:0;grid-row:1/span 3;display:grid;align-content:start;gap:.2rem;padding-top:.3rem}
.tiplist .for b{font-size:.8rem;font-weight:700;letter-spacing:.17em;text-transform:uppercase;overflow:hidden;text-overflow:ellipsis}
.tiplist .for small{font-size:.7rem;font-weight:700;letter-spacing:.13em;text-transform:uppercase;color:var(--rust)}
.tiplist .early .for small{color:var(--ink-3);font-weight:600;font-style:italic;letter-spacing:.06em;text-transform:none;font-size:.84rem}
.tiplist h3{margin:0;font-family:var(--serif);font-weight:400;font-size:clamp(1.22rem,1.05rem + .7vw,1.55rem);line-height:1.2;font-variation-settings:"opsz" 40}
.tiplist .ev{margin:0;color:var(--ink-2);font-size:.95rem;max-width:40rem} .tiplist .ev small{display:block;color:var(--ink-3);font-size:.84rem;padding-top:.1rem}
.tiplist .see{margin:.4rem 0 0;color:var(--ink-3);font-size:.84rem;display:flex;flex-wrap:wrap;gap:.3rem .35rem;align-items:center}
.tiplist .see span{font-variant-numeric:tabular-nums;color:var(--ink-2)} .tiplist .see span + span::before{content:"\00b7";padding-right:.35rem;color:var(--ink-3)}
.go{font:inherit;font-size:.8rem;font-variant-numeric:tabular-nums;color:var(--ink);background:transparent;border:1px solid var(--line-2);border-radius:999px;min-width:2rem;height:1.75rem;
  padding:0 .5rem;cursor:pointer;transition:background .2s,color .2s,transform .25s var(--ease)}
.go:hover{background:var(--ink);color:var(--paper);transform:translateY(-1px)} .go:active{transform:none}
.tips .note2{max-width:44rem;margin-top:1rem}
@media (pointer:coarse){.go{min-width:2.75rem;height:2.75rem}}
@media (max-width:34rem){.tiplist li{grid-template-columns:minmax(0,1fr)}.tiplist .for{grid-row:auto;display:flex;gap:.8rem;align-items:baseline;padding-top:0}}

/* placement */
.who{display:grid;gap:1.4rem clamp(1rem,2.5vw,2rem);grid-template-columns:minmax(0,1fr);padding-top:1.4rem}
@media (min-width:44rem){.who{grid-template-columns:repeat(3,minmax(0,1fr))}}
.who + .who{border-top:1px solid var(--line);margin-top:1.6rem}
.who h3{grid-column:1/-1;margin:0;font-family:var(--serif);font-weight:400;font-size:1.4rem;display:flex;gap:.8rem;align-items:baseline}
.who h3 small{font-family:var(--sans);font-size:.72rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:var(--ink-3)}
figure{margin:0} figcaption{display:flex;justify-content:space-between;gap:.6rem;font-size:.84rem;color:var(--ink-2);padding-top:.45rem}
figcaption b{font-weight:700;letter-spacing:.12em;text-transform:uppercase;font-size:.7rem;color:var(--ink)}
svg.tt{width:100%;height:auto;display:block;border-radius:2px}
.key{display:flex;flex-wrap:wrap;gap:.4rem 1.4rem;color:var(--ink-3);font-size:.84rem;padding-top:1.1rem}
.key span{display:inline-flex;align-items:center;gap:.45rem} .key i{width:.68rem;height:.68rem;border-radius:50%;background:var(--ball);display:inline-block}
.key i.o{background:transparent;border:1.5px solid var(--table-line);outline:1px solid var(--line-2)}

/* numbers */
.numbers table{max-width:50rem} .numbers td:first-child{color:var(--ink-2)} .numbers td:not(:first-child),.numbers th:not(:first-child){text-align:right}
.numbers td b{font-family:var(--serif);font-weight:400;font-size:1.18rem} .numbers .grp td{padding-top:1.5rem;border-top:0;font-size:.7rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:var(--rust)}
.numbers small{color:var(--ink-3)}
.rally{display:grid;grid-template-columns:repeat(var(--n),minmax(0,1fr));gap:3px;align-items:end;height:5.2rem;margin-top:1.4rem;border-bottom:1.5px solid var(--ink)}
.rally div{background:var(--ink);min-height:2px;border-radius:1px 1px 0 0;transform-origin:bottom;animation:rise .7s var(--ease) both;animation-delay:calc(var(--i)*35ms + .25s)}
.rally-x{display:grid;grid-template-columns:repeat(var(--n),minmax(0,1fr));gap:3px;font-size:.72rem;color:var(--ink-3);text-align:center;padding-top:.3rem}
.colophon{margin-top:clamp(2.5rem,6vw,4rem);padding-top:1rem;border-top:1.5px solid var(--ink);color:var(--ink-3);font-size:.84rem;max-width:46rem}

/* the opening picture: a real frame of the match with the tracking inked on */
.hero{margin:clamp(1.3rem,3vw,2.2rem) 0 0}
.hero img{display:block;width:100%;height:auto;border-radius:2px;background:var(--paper-2)}
.hero figcaption{display:flex;justify-content:space-between;align-items:center;gap:1rem;padding:.6rem 0 0;font-size:.86rem;color:var(--ink-2)}
.hero figcaption b{font-family:var(--serif);font-weight:400;font-size:1.02rem;color:var(--ink);letter-spacing:0;text-transform:none}
.hero figcaption .go,.strip .go{white-space:nowrap;flex:none}
/* the top of the report plays: a first visitor must see at once that there are videos (2026-09-27) */
.hero-v{position:relative;border-radius:4px;overflow:hidden;background:var(--paper-2);cursor:pointer}
.hero-v video{display:block;width:100%;height:auto;aspect-ratio:var(--ar,16/9);object-fit:cover;background:var(--paper-2)}
.hero-v.playing video{aspect-ratio:16/9;object-fit:contain;background:oklch(9% 0.003 250)}   /* the picture's shape until play, then the video's */
.hero-v .bigplay{position:absolute;inset:0;display:grid;place-items:center;align-content:center;gap:.7rem;border:0;padding:0;margin:0;width:100%;height:100%;
  background:linear-gradient(to top,oklch(10% 0.004 250 / .55),transparent 45%);color:var(--ink);cursor:pointer;font:inherit;transition:opacity .25s var(--ease)}
.hero-v .bigplay svg{width:clamp(4.4rem,8vw,6.2rem);height:auto;transition:transform .3s var(--ease)}
.hero-v:hover .bigplay svg{transform:scale(1.06)}
.hero-v .bigplay span{font-size:.74rem;font-weight:700;letter-spacing:.17em;text-transform:uppercase;text-shadow:0 1px 8px oklch(0% 0 0 / .6)}
.hero-v.playing .bigplay{opacity:0;pointer-events:none}
.watchbar{display:flex;flex-wrap:wrap;gap:.7rem;margin:1rem 0 0}
.watchbar a{font-size:.74rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;text-decoration:none;color:var(--ink);border:1.5px solid var(--ink);
  border-radius:2px;padding:.85rem 1.2rem;display:inline-flex;align-items:baseline;gap:.6rem;transition:background .2s var(--ease),color .2s var(--ease)}
.watchbar a small{font-size:.7rem;font-weight:500;letter-spacing:.06em;text-transform:none;color:var(--ink-3)}
.watchbar a:hover{background:var(--ink);color:var(--paper)} .watchbar a:hover small{color:var(--paper)}
.watchbar a.first{background:var(--ink);color:var(--paper)} .watchbar a.first small{color:color-mix(in oklch,var(--paper) 70%,var(--ink))}

/* the scoresheet: one column per game, the game's winner ringed like a scorer's pen would */
.sheetwrap{display:grid;justify-items:center;gap:.5rem;padding:0 0 clamp(1.4rem,3.5vw,2.2rem)}
.sheet{border-collapse:collapse;width:auto;font-variant-numeric:tabular-nums lining-nums}
.sheet th,.sheet td{border:0;padding:.28rem clamp(.45rem,1.4vw,.85rem);text-align:center;vertical-align:middle}
.sheet thead th{font-size:.66rem;font-weight:700;letter-spacing:.17em;color:var(--ink-3)}
.sheet tbody th{text-align:left;font-size:.72rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:var(--ink-2);padding-left:0;white-space:nowrap}
.sheet tbody th i{display:inline-block;width:.55rem;height:.55rem;border-radius:50%;background:var(--c);margin-right:.5rem;vertical-align:.04em}
.sheet td{font-family:var(--serif);font-size:clamp(1.15rem,1rem + .6vw,1.5rem);color:var(--ink-3);line-height:1}
.sheet td span{display:inline-grid;place-items:center;min-width:2.3rem;height:2.3rem}
.sheet td.w{color:var(--ink)}
.sheet td.w span{border:1.5px solid var(--ink);border-radius:52% 48% 50% 46%/48% 54% 46% 52%;transform:rotate(-5deg)}
.sheet td.w:nth-child(odd) span{transform:rotate(4deg)}
.sheet tbody tr+tr th,.sheet tbody tr+tr td{border-top:1px solid var(--line)}
.sheet .open{font-style:italic}
.sheetwrap p{margin:0;color:var(--ink-3);font-size:.8rem;text-align:center;max-width:36rem}

/* head to head: the broadcast convention, set in ink */
.names2{display:flex;justify-content:space-between;gap:1rem;padding:1rem 0 .35rem}
.names2 b{font-size:.78rem;font-weight:700;letter-spacing:.17em;text-transform:uppercase}
.names2 b::before{content:"";display:inline-block;width:.6rem;height:.6rem;border-radius:50%;background:var(--c);margin-right:.55rem;vertical-align:.04em}
.bars{list-style:none;margin:0;padding:0}
.bars li{display:grid;grid-template-columns:minmax(4.6rem,auto) minmax(0,1fr) minmax(9.5rem,15rem) minmax(0,1fr) minmax(4.6rem,auto);
  grid-template-areas:"vl bl lab br vr";align-items:center;column-gap:clamp(.6rem,1.6vw,1.2rem);padding:.66rem 0;border-top:1px solid var(--line)}
.bars .v{display:grid;line-height:1.08;grid-area:vl} .bars .v.r{text-align:right;grid-area:vr}
.bars .v b{font-family:var(--serif);font-weight:400;font-size:clamp(1.25rem,1.05rem + .7vw,1.7rem);color:var(--ink-3)}
.bars .v.lead b{color:var(--ink)} .bars .v small{color:var(--ink-3);font-size:.76rem;white-space:nowrap}
.bars .lab{grid-area:lab;margin:0;text-align:center;font-size:.8rem;font-weight:600;letter-spacing:.05em;color:var(--ink-2);line-height:1.3}
.bars .lab small{display:block;font-weight:500;letter-spacing:0;color:var(--ink-3);font-size:.74rem}
.bar{height:.5rem;background:color-mix(in oklch,var(--ink) 7%,transparent);border-radius:1px;display:flex;overflow:hidden}
.bar.l{grid-area:bl;justify-content:flex-end} .bar.r{grid-area:br}
.bar i{display:block;height:100%;width:calc(var(--w)*100%);background:var(--c);animation:grow .9s var(--ease) both;animation-delay:calc(var(--k)*45ms + .35s)}
.bar.l i{transform-origin:right} .bar.r i{transform-origin:left} .bar i.soft{opacity:.4}
.bar i.s2{opacity:.62} .bar i.s3{opacity:.3}
.bars .grp{display:block;padding:1.5rem 0 .3rem;border-top:0;font-size:.7rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:var(--rust)}
@keyframes grow{from{transform:scaleX(0)}to{transform:scaleX(1)}}
@media (max-width:42rem){.bars li{grid-template-columns:auto minmax(0,1fr) minmax(0,1fr) auto;grid-template-areas:"lab lab lab lab" "vl bl br vr";row-gap:.4rem}}

/* momentum: who led, point by point, game by game */
.mom svg{width:100%;height:auto;display:block;overflow:visible;margin-top:1.1rem}
.mom .gl{font:700 11px var(--sans);letter-spacing:.14em;fill:var(--ink-3)} .mom .gs{font:400 15px var(--serif);fill:var(--ink)}
.mom .who{font:700 10px var(--sans);letter-spacing:.14em}
.mom [data-go]{cursor:pointer} .mom [data-go]:hover{fill:color-mix(in oklch,var(--ink) 7%,transparent)}
.mom .line{fill:none;stroke:var(--ink);stroke-width:1.7;stroke-linejoin:round;stroke-linecap:round;stroke-dasharray:4000;stroke-dashoffset:4000;animation:draw 2.2s var(--ease) .5s forwards}
@keyframes draw{to{stroke-dashoffset:0}}

/* the whole match */
.whole video{display:block;width:100%;aspect-ratio:16/9;background:oklch(14% 0.015 165);border-radius:2px;margin-top:1rem}

/* facts: ruled, not boxed */
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(12.5rem,1fr));margin:1.2rem 0 0;column-gap:clamp(1rem,3vw,2.2rem)}
.facts div{padding:.85rem 0 .95rem;border-top:1px solid var(--line)}
.facts dt{font-size:.7rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--ink-3)}
.facts dd{margin:.25rem 0 0;font-family:var(--serif);font-size:clamp(1.6rem,1.3rem + 1vw,2.2rem);line-height:1.05;display:flex;align-items:baseline;gap:.6rem;flex-wrap:wrap}
.facts dd small{font-family:var(--sans);font-size:.8rem;color:var(--ink-3)}

/* posture: real frames at contact */
.strip{display:grid;grid-template-columns:minmax(10rem,14rem) minmax(0,1fr);column-gap:clamp(1rem,3vw,2.2rem);row-gap:.4rem;padding:1.3rem 0 1.4rem;border-top:1px solid var(--line)}
.strip:first-of-type{border-top:0}
.strip h3{grid-column:1;margin:0;font-size:.78rem;font-weight:700;letter-spacing:.17em;text-transform:uppercase}
.strip h3 i{display:inline-block;width:.6rem;height:.6rem;border-radius:50%;background:var(--c);margin-right:.55rem;vertical-align:.04em}
.strip p{grid-column:1;margin:0;color:var(--ink-2);font-size:.9rem;line-height:1.45} .strip p b{font-family:var(--serif);font-weight:400;font-size:1.5rem;color:var(--ink)}
.strip p small{color:var(--ink-3);font-size:.78rem}
.strip .tiles{grid-column:2;grid-row:1/span 3;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:clamp(.4rem,1vw,.7rem)}
.strip figure img{display:block;width:100%;height:auto;border-radius:2px;background:var(--paper-2)}
.strip figcaption{display:flex;justify-content:space-between;align-items:center;padding-top:.35rem;font-size:.8rem;color:var(--ink-3)}
.strip figcaption b{font-family:var(--serif);font-weight:400;font-size:1.1rem;color:var(--ink);letter-spacing:0;text-transform:none}
@media (max-width:44rem){.strip{grid-template-columns:minmax(0,1fr)}.strip .tiles{grid-column:1;grid-row:auto;grid-template-columns:repeat(2,minmax(0,1fr))}}

/* stances */
.stance svg{width:100%;max-width:48rem;height:auto;display:block;margin:1rem auto 0}
.stance .nm{font:700 12px var(--sans);letter-spacing:.16em} .stance .deg{font:400 22px var(--serif);fill:var(--ink)} .stance .cap{font:500 12px var(--sans);fill:var(--ink-3)}

/* after the match, in 3D */
.a3d-tabs{display:flex;gap:.5rem;margin:1.1rem 0 .2rem;flex-wrap:wrap}
.a3d-tabs button,.vbar button{font:600 .78rem var(--sans);letter-spacing:.14em;text-transform:uppercase;color:var(--ink-2);background:none;border:1.5px solid var(--line-2);
  border-radius:999px;padding:.5rem .95rem .5rem .8rem;display:inline-flex;align-items:center;gap:.5rem;cursor:pointer;transition:background .2s var(--ease),color .2s var(--ease)}
.a3d-tabs button i,.vbar button i{width:.62rem;height:.62rem;border-radius:50%;background:var(--c,var(--ink-3))}
.a3d-tabs button small{font-weight:500;letter-spacing:.04em;text-transform:none;color:var(--ink-3)}
.a3d-tabs button[aria-selected=true]{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.a3d-tabs button[aria-selected=true] small{color:color-mix(in oklch,var(--paper) 72%,transparent)}
.a3d-tabs button:focus-visible,.vbar button:focus-visible{outline:2px solid var(--ball);outline-offset:2px}
.plate{display:grid;grid-template-columns:minmax(0,1fr);gap:.9rem 2.2rem;padding:1.5rem 0 1.7rem;border-top:1px solid var(--line)}
.a3d-panel .plate:first-child{border-top:0}
.plate img{width:100%;height:auto;display:block;border-radius:2px;background:var(--paper-2)}
@media (min-width:62rem){.plate{grid-template-columns:minmax(0,1.9fr) minmax(0,1fr);align-items:start}}
.plate .k{margin:0 0 .4rem;font-size:.72rem;font-weight:700;letter-spacing:.17em;text-transform:uppercase;color:var(--rust)}
.plate h3{font-family:var(--serif);font-weight:400;font-size:clamp(1.25rem,1.1rem + .6vw,1.55rem);line-height:1.15;margin:0 0 .5rem}
.plate h4{font:700 .95rem/1.3 var(--sans);margin:1.05rem 0 .3rem}
.plate p{margin:0 0 .5rem;color:var(--ink-2);font-size:.95rem;max-width:34rem}
.plate .sum{margin-top:.85rem;padding-top:.6rem;border-top:1px solid var(--line);font-size:.84rem;color:var(--ink-3);font-variant-numeric:tabular-nums}
.viewer{margin:2.2rem 0 0}
.viewer h3{font-family:var(--serif);font-weight:400;font-size:1.35rem;margin:0 0 .7rem}
.vbar{display:flex;flex-wrap:wrap;gap:.6rem 1.4rem;align-items:center;margin-bottom:.8rem} .vbar[hidden]{display:none}
.vbar .grp{display:flex;gap:.35rem;flex-wrap:wrap}
.vbar button{padding:.38rem .8rem;letter-spacing:.1em}
.vbar button[aria-pressed=true]{background:color-mix(in oklch,var(--ink) 9%,transparent);color:var(--ink);border-color:var(--ink)}
.vmodes{display:inline-flex;border:1.5px solid var(--ink);border-radius:999px;padding:2px;margin-bottom:.8rem}.vmodes button{font:600 .78rem var(--sans);letter-spacing:.12em;text-transform:uppercase;border:0;background:none;color:var(--ink-2);padding:.42rem .95rem;border-radius:999px;cursor:pointer}.vmodes button[aria-pressed=true]{background:var(--ink);color:var(--paper)}.vsel{display:inline-flex;align-items:center;gap:.45rem;font:600 .72rem var(--sans);letter-spacing:.12em;text-transform:uppercase;color:var(--ink-3)}.vsel select{font:500 .86rem var(--sans);color:var(--ink);background:var(--paper);border:1.5px solid var(--line-2);border-radius:2px;padding:.3rem .4rem;max-width:22rem}#a3d-t{width:clamp(8rem,22vw,16rem);accent-color:var(--ball)} #a3d-tl{font:500 .82rem var(--sans);color:var(--ink-3);min-width:6.5rem;font-variant-numeric:tabular-nums}.vstat{margin:.6rem 0 0;font-size:.9rem;color:var(--ink-2);font-variant-numeric:tabular-nums} .vstat b{font-family:var(--serif);font-weight:400;font-size:1.15rem;color:var(--ink)}
#a3d-cv{width:100%;display:block;border-top:1.5px solid var(--ink);border-bottom:1px solid var(--line);cursor:grab;touch-action:pan-y}
#a3d-cv:active{cursor:grabbing} #a3d-cv:focus-visible{outline:2px solid var(--ball);outline-offset:3px}
.viewer figcaption{color:var(--ink-3);font-size:.84rem;margin-top:.6rem}
.a3d .note2{max-width:46rem;margin-top:1.2rem}
@media print{.vbar,.viewer,.a3d-tabs{display:none}.a3d-panel[hidden]{display:block!important}}

/* ---- the black edition: type, rules and controls ------------------------------------------------------------------------------ */
html{scroll-behavior:smooth;background:var(--paper)} @media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
::selection{background:color-mix(in oklch,var(--ink) 26%,transparent)}
h1,h2{font-weight:700;letter-spacing:.004em;text-transform:uppercase;text-wrap:balance}
a{color:var(--ink);text-decoration-color:var(--line-2)} a:hover{text-decoration-color:var(--ink)}
:focus-visible{outline-color:var(--ink)}
.kicker{color:var(--ink-3);letter-spacing:.26em;font-weight:600}
.mast{border-bottom:1px solid var(--line-2);padding-bottom:1.5rem} .mast::after{display:none}
.mast h1{font-size:clamp(3rem,1rem + 8vw,7.4rem);line-height:.86;margin:.8rem 0 .9rem;font-variation-settings:normal}
.mast h1 i{font-style:normal;font-weight:400;text-transform:lowercase;color:var(--ink-3);padding:0 .06em}
.meta{font-size:.92rem} .meta a{color:var(--ink-2)}
section{padding-top:clamp(3.2rem,7vw,5.6rem);scroll-margin-top:4rem} .anchor{display:block;scroll-margin-top:4rem}
.head{border-bottom:1px solid var(--line-2);padding-bottom:.85rem;margin-bottom:.45rem}
.head h2{font-size:clamp(1.8rem,1.2rem + 2vw,2.8rem);line-height:.92}
.head span{font-size:.84rem;max-width:26rem}
.score{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}                  /* two sides, no badge between them */
.side .name{color:var(--ink);font-weight:600;letter-spacing:.22em}
.side .num{font-weight:700;font-size:clamp(5.2rem,2rem + 12vw,11.5rem);line-height:.8;letter-spacing:-.012em;font-variation-settings:normal}
.side .sub{font-size:.86rem}
.stamp{position:relative;border:0;box-shadow:none;transform:none;color:var(--ink);width:clamp(7.6rem,15vw,9.6rem)}
.stamp svg{position:absolute;inset:0;width:100%;height:100%;transform:rotate(-90deg)}
.stamp .trk{fill:none;stroke:var(--line);stroke-width:2.6} .stamp .val{fill:none;stroke:var(--c);stroke-width:2.6;stroke-linecap:round;animation:gauge 1.5s var(--ease) .35s both}
@keyframes gauge{from{stroke-dasharray:0 100}}
.stamp span{color:var(--ink-3);letter-spacing:.24em} .stamp small{color:var(--ink-2)}
.stamp b{font-style:normal;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:var(--c);font-size:clamp(1.3rem,2.4vw,1.75rem)}
.verdict p{font-weight:500;font-size:clamp(1.3rem,1rem + 1vw,1.8rem);line-height:1.16;font-variation-settings:normal}
.sheet td{font-weight:600} .sheet td.w{color:var(--paper)}
.sheet td.w span,.sheet td.w:nth-child(odd) span{border:0;border-radius:50%;background:var(--ink);transform:none}
.ledger td:first-child,.ledger .sc{font-weight:600}
.ledger tr[aria-current="true"]{background:color-mix(in oklch,var(--ink) 9%,transparent)}
.ledger tr.game td{border-top:1px solid var(--line-2);color:var(--ink-3)}
.play,.go{border-color:var(--line-2)}
.play:hover,.go:hover,tr[aria-current="true"] .play{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.stage video,.whole video{background:var(--paper-2);border-radius:4px}
.hero img,.plate img,.strip figure img,.pf img{border-radius:4px}
.stage .cap b,.strip p b,.strip figcaption b,.hero figcaption b{font-weight:600;font-size:1.25rem}
.tiplist h3{font-weight:600;font-size:clamp(1.4rem,1.1rem + 1vw,1.9rem);line-height:1.06;font-variation-settings:normal}
.tiplist .for small,.plate .k,.bars .grp,.numbers .grp td,.ledger tr.game td{color:var(--ink-3)}
.facts dd{font-weight:600;font-size:clamp(2.1rem,1.4rem + 2vw,3.1rem);line-height:.95}
.numbers td b,.bars .v b{font-weight:600}
.plate h3{font-weight:600;font-size:clamp(1.5rem,1.2rem + 1vw,2rem);line-height:1.04}
.viewer h3,.who h3{font-weight:600;font-size:clamp(1.4rem,1.2rem + .8vw,1.8rem);text-transform:uppercase;letter-spacing:.01em}
.a3d-tabs button[aria-selected=true],.vmodes button[aria-pressed=true]{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.vmodes{border-color:var(--line-2)} .vbar button[aria-pressed=true]{background:color-mix(in oklch,var(--ink) 14%,transparent);border-color:var(--ink-2)}
#a3d-cv{border:1px solid var(--line);border-radius:4px}
.rally{border-bottom:1px solid var(--line-2)} .colophon{border-top:1px solid var(--line-2)}
.mom .line{stroke:var(--ink)} .mom .gs{font-weight:600}
.empty{border-color:var(--line-2)}

/* the score bar: always on top, the match in one line and a way to every section; the current one lit */
.topbar{position:sticky;top:0;z-index:30;background:var(--paper);border-bottom:1px solid var(--line)}
.topbar .in{max-width:74rem;margin:0 auto;padding:0 var(--gutter);display:flex;align-items:center;gap:clamp(.8rem,2vw,1.6rem);height:3.3rem}
.topbar .sc{display:flex;align-items:center;gap:.6rem;white-space:nowrap;font:600 .74rem var(--sans);letter-spacing:.18em;text-transform:uppercase;color:var(--ink-2)}
.topbar .sc i{width:.5rem;height:.5rem;border-radius:50%;background:var(--c);display:inline-block}
.topbar .sc b{font:700 1.3rem/1 var(--serif);letter-spacing:0;color:var(--ink);font-variant-numeric:tabular-nums}
.topbar .sc em{width:1px;height:1.05rem;background:var(--line-2);display:inline-block}
.bar i.soft{opacity:1;background:color-mix(in srgb,var(--ink) 34%,transparent)}      /* the trailing player: plain grey (a hue mix drifts to purple) */
.bar i.s2{opacity:1;background:color-mix(in srgb,var(--c) 58%,var(--ink))} .bar i.s3{opacity:1;background:color-mix(in srgb,var(--c) 28%,var(--ink))}
.topbar nav{margin-left:auto;display:flex;gap:.15rem;overflow-x:auto;scrollbar-width:none;-webkit-overflow-scrolling:touch}
.topbar nav::-webkit-scrollbar{display:none}
.topbar nav a{color:var(--ink-3);text-decoration:none;font:600 .7rem var(--sans);letter-spacing:.15em;text-transform:uppercase;padding:.45rem .65rem;border-radius:999px;
  white-space:nowrap;transition:color .2s var(--ease),background .2s var(--ease)}
.topbar nav a:hover{color:var(--ink)} .topbar nav a[aria-current="true"]{color:var(--paper);background:var(--ink)}
@media (max-width:52rem){.topbar .sc{display:none}.topbar nav{margin-left:0}}
@media (max-width:40rem){.head{flex-direction:column;align-items:flex-start;gap:.45rem}.head span{text-align:left}}
@media print{.topbar{display:none}}


/* one orchestrated entrance; transform and opacity only */
@keyframes up{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@keyframes ink{from{opacity:0;transform:rotate(-7deg) scale(1.35)}to{opacity:1;transform:rotate(-7deg) scale(1)}}
@keyframes rise{from{transform:scaleY(0)}to{transform:scaleY(1)}}
.mast,.hero,.score .side,.sheetwrap,.verdict,section{animation:up .8s var(--ease) both} .hero{animation-delay:.05s} .sheetwrap{animation-delay:.2s}
.score .side{animation-delay:.08s} .score .side.r{animation-delay:.16s} .verdict{animation-delay:.24s} section{animation-delay:.32s}
.stamp{animation:ink .55s var(--ease) .42s both}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
@media (max-width:40rem){.ledger .hide-s{display:none}}
@media (max-width:34rem){.score{grid-template-columns:minmax(0,1fr) minmax(0,1fr);row-gap:1.4rem}.stamp{grid-column:1/-1;grid-row:2;justify-self:start}.ledger .hide-xs{display:none}.ledger td,.ledger th{padding-inline:.4rem}.ledger .n{padding-right:.8rem}.play{width:2.75rem;height:2.75rem}}
@media print{.stage,.play,.legend,.tiplist .see,.hero button{display:none}.points{grid-template-columns:1fr}body{background:#fff;color:#000}*{animation:none!important}}
/* sections come up as they scroll in, where the browser can tie an animation to the scroll */
@supports (animation-timeline:view()){
  @media (prefers-reduced-motion:no-preference){
    section{animation:up linear both;animation-timeline:view();animation-range:entry 0% cover 16%}
  }
}
"""

JS = r"""
(function(){
  var box=document.querySelector('.hero-v'); if(!box) return;
  var hv=box.querySelector('video'), all=[].slice.call(document.querySelectorAll('video'));
  function start(){hv.controls=true; box.classList.add('playing'); hv.play().catch(function(){});}
  box.querySelector('.bigplay').addEventListener('click',start);
  all.forEach(function(x){x.addEventListener('play',function(){                // one video at a time
    if(x===hv) box.classList.add('playing');
    all.forEach(function(o){if(o!==x&&!o.paused) o.pause()});
  })});
})();
(function(){
  var v=document.getElementById('v'); if(!v) return;
  var rows=[].slice.call(document.querySelectorAll('.ledger tbody tr[data-clip]')), cap=document.getElementById('cap'), sub=document.getElementById('sub');
  function show(tr){
    rows.forEach(function(r){r.removeAttribute('aria-current')}); tr.setAttribute('aria-current','true');
    cap.textContent=tr.dataset.cap; sub.textContent=tr.dataset.sub; v.src=tr.dataset.clip; v.play().catch(function(){});
  }
  rows.forEach(function(tr){tr.querySelector('.play').addEventListener('click',function(){show(tr)})});
  var calm=window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  [].slice.call(document.querySelectorAll('[data-go]')).forEach(function(b){b.addEventListener('click',function(){
    var tr=rows.filter(function(r){return r.dataset.id===b.dataset.go})[0]; if(!tr) return;
    show(tr); v.scrollIntoView({block:'center',behavior:calm?'auto':'smooth'});
  })});
  v.addEventListener('play',function(){if(!rows.some(function(r){return r.getAttribute('aria-current')==='true'})&&rows[0]){rows[0].setAttribute('aria-current','true');cap.textContent=rows[0].dataset.cap;sub.textContent=rows[0].dataset.sub}});
  document.addEventListener('keydown',function(e){
    if(e.target.tagName==='INPUT'||e.altKey||e.metaKey||e.ctrlKey) return;
    var i=rows.findIndex(function(r){return r.getAttribute('aria-current')==='true'});
    if(e.key==='j'||e.key==='ArrowRight'){if(rows[i+1]){show(rows[i+1]);rows[i+1].scrollIntoView({block:'nearest'})}}
    else if(e.key==='k'||e.key==='ArrowLeft'){if(i>0){show(rows[i-1]);rows[i-1].scrollIntoView({block:'nearest'})}}
  });
})();
(function(){
  var tabs=[].slice.call(document.querySelectorAll('.a3d-tabs [role=tab]'));
  function sel(t){tabs.forEach(function(b){var on=b===t;b.setAttribute('aria-selected',on?'true':'false');b.tabIndex=on?0:-1;
    document.getElementById(b.getAttribute('aria-controls')).hidden=!on})}
  tabs.forEach(function(b,i){b.addEventListener('click',function(){sel(b)});b.addEventListener('keydown',function(e){
    var d=e.key==='ArrowRight'?1:e.key==='ArrowLeft'?-1:0;if(!d)return;e.preventDefault();e.stopPropagation();
    var n=tabs[(i+d+tabs.length)%tabs.length];sel(n);n.focus()})});
  if(tabs.length)sel(tabs[0]);
  var cv=document.getElementById('a3d-cv');if(!cv)return;
  var D=JSON.parse(document.getElementById('a3d-data').textContent),T=D.table,ctx=cv.getContext('2d');
  var bodyPlayers=D.players.map(function(p,i){return i}).filter(function(i){return D.players[i].strokes&&D.players[i].strokes.length});
  var mode=bodyPlayers.length?'bodies':'shots';
  var VIEWS={behind:[Math.PI+.5,.56,5.9],side:[-Math.PI/2,.3,6.2],above:[-Math.PI/2+.001,1.4,6.6]};
  var BV={front:[-1.0,.18,4.7],side:[-Math.PI/2,.12,4.9],behind:[Math.PI+.4,.36,4.7],above:[-Math.PI/2+.001,1.25,5.4]};
  var st={yaw:VIEWS.behind[0],pitch:VIEWS.behind[1],dist:VIEWS.behind[2],tx:.45,ty:T.W/2,tz:-.15},on={p0:true,p1:true,pro:false},lay={flights:true,contacts:true,feet:false};
  var E,R,F,W_,H_,queued=false;
  function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]}
  function cross(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]}
  function norm(a){var n=Math.sqrt(dot(a,a))||1;return[a[0]/n,a[1]/n,a[2]/n]}
  function clamp(v,a,b){return Math.max(a,Math.min(b,v))}
  var cc=document.createElement('canvas');cc.width=cc.height=1;var cx1=cc.getContext('2d',{willReadFrequently:true}),cache={};
  function rgb(c){if(cache[c])return cache[c];cx1.clearRect(0,0,1,1);cx1.fillStyle='#000';cx1.fillStyle=c;cx1.fillRect(0,0,1,1);
    var d=cx1.getImageData(0,0,1,1).data;return cache[c]=[d[0],d[1],d[2]]}
  var shades={};
  function shade(c,t){var key=c+'|'+t;if(shades[key])return shades[key];var a=rgb(c),b=t<0?[0,0,0]:[255,255,255],k=Math.abs(t);
    return shades[key]='rgb('+[0,1,2].map(function(i){return Math.round(a[i]+(b[i]-a[i])*k)}).join(',')+')'}
  var K=null;                                                          // the theme's colours, read once (reading styles every frame stalls it)
  function pal(){var r={};['--paper','--ink','--ink-2','--ink-3','--table','--table-line','--rust','--p1','--p2'].forEach(function(n){
    r[n]=getComputedStyle(document.documentElement).getPropertyValue(n).trim()||'gray'});return r}
  function col(n){if(!K)K=pal();return K[n]||(K[n]=getComputedStyle(document.documentElement).getPropertyValue(n).trim()||'gray')}
  function setup(){var cp=Math.cos(st.pitch);E=[st.tx+st.dist*cp*Math.cos(st.yaw),st.ty+st.dist*cp*Math.sin(st.yaw),st.tz+st.dist*Math.sin(st.pitch)];
    var f=norm([st.tx-E[0],st.ty-E[1],st.tz-E[2]]),r=norm(cross(f,[0,0,1])),u=cross(r,f);R=[r,[-u[0],-u[1],-u[2]],f];F=(W_/2)/Math.tan(21*Math.PI/180)}
  function P(p){var d=[p[0]-E[0],p[1]-E[1],p[2]-E[2]],z=dot(R[2],d);return z<.08?null:[F*dot(R[0],d)/z+W_/2,F*dot(R[1],d)/z+H_/2,z]}
  function path(pts,close){ctx.beginPath();var s=false;for(var i=0;i<pts.length;i++){var q=P(pts[i]);if(!q){s=false;continue}
    if(!s){ctx.moveTo(q[0],q[1]);s=true}else ctx.lineTo(q[0],q[1])}if(close)ctx.closePath()}
  var bg=document.createElement('canvas'),bgx=bg.getContext('2d'),bgKey='';
  function frame(){var w=cv.clientWidth||800,h=Math.round(w*9/16),dpr=Math.min(2,window.devicePixelRatio||1);
    if(cv.width!==Math.round(w*dpr)||cv.height!==Math.round(h*dpr)){cv.width=Math.round(w*dpr);cv.height=Math.round(h*dpr);cv.style.height=h+'px'}
    W_=w;H_=h;setup();
    var key=[w,h,dpr,st.yaw,st.pitch,st.dist,st.tx,st.ty,st.tz,col('--paper')].join(',');
    if(key!==bgKey){bgKey=key;bg.width=cv.width;bg.height=cv.height;var c0=ctx;ctx=bgx;ctx.setTransform(dpr,0,0,dpr,0,0);
      ctx.globalAlpha=1;ctx.fillStyle=col('--paper');ctx.fillRect(0,0,w,h);ctx.lineCap='round';
      ctx.strokeStyle=col('--ink');ctx.lineWidth=1;ctx.globalAlpha=.09;
      for(var x=-3;x<=T.L+3.01;x+=.5){path([[x,-2.5,T.floor],[x,T.W+2.5,T.floor]]);ctx.stroke()}
      for(var y=-2.5;y<=T.W+2.51;y+=.5){path([[-3,y,T.floor],[T.L+3,y,T.floor]]);ctx.stroke()}
      ctx=c0}
    ctx.setTransform(1,0,0,1,0,0);ctx.globalAlpha=1;ctx.drawImage(bg,0,0);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.lineCap='round';ctx.lineJoin='round'}
  function drawShots(){frame();var ink=col('--ink'),ink3=col('--ink-3'),tbl=col('--table'),tl=col('--table-line');
    ctx.strokeStyle=ink3;ctx.lineWidth=3;
    [[.35,.18],[.35,T.W-.18],[T.L-.35,.18],[T.L-.35,T.W-.18]].forEach(function(l){path([[l[0],l[1],T.floor],[l[0],l[1],-.03]]);ctx.stroke()});
    var top=[[0,0,0],[T.L,0,0],[T.L,T.W,0],[0,T.W,0]];ctx.fillStyle=tbl;path(top,true);ctx.fill();
    ctx.strokeStyle=tl;ctx.lineWidth=1.6;path(top,true);ctx.stroke();ctx.lineWidth=.8;path([[0,T.W/2,0],[T.L,T.W/2,0]]);ctx.stroke();
    var sets=[];D.players.forEach(function(p,i){if(on['p'+i])sets.push([p,col('--'+p.key),false])});if(on.pro)sets.push([D.pro,ink,true]);
    if(lay.feet)sets.forEach(function(s){ctx.fillStyle=s[1];ctx.globalAlpha=s[2]?.3:.45;s[0].feet.forEach(function(f){var q=P([f[0],f[1],T.floor]);
      if(q){ctx.beginPath();ctx.arc(q[0],q[1],Math.max(1.5,F*.03/q[2]),0,7);ctx.fill()}});ctx.globalAlpha=1});
    ctx.globalAlpha=.28;ctx.fillStyle=ink;path([[T.net_x,-.15,0],[T.net_x,T.W+.15,0],[T.net_x,T.W+.15,T.net_h],[T.net_x,-.15,T.net_h]],true);ctx.fill();
    ctx.globalAlpha=1;ctx.strokeStyle=tl;ctx.lineWidth=1.6;path([[T.net_x,-.15,T.net_h],[T.net_x,T.W+.15,T.net_h]]);ctx.stroke();
    if(lay.flights)sets.forEach(function(s){ctx.strokeStyle=s[1];ctx.lineWidth=s[2]?1.1:1.4;ctx.globalAlpha=s[2]?.3:.42;
      if(s[2])ctx.setLineDash([5,4]);s[0].paths.forEach(function(pt){path(pt);ctx.stroke()});ctx.setLineDash([]);ctx.globalAlpha=1});
    if(lay.contacts){var balls=[];sets.forEach(function(s){s[0].contacts.forEach(function(c,j){var q=P(c);if(q)balls.push([q,s[1],s[0].top[j]&&!s[2]])})});
      balls.sort(function(a,b){return b[0][2]-a[0][2]});
      balls.forEach(function(b){var r=Math.max(2.2,F*.022/b[0][2]);ctx.beginPath();ctx.arc(b[0][0],b[0][1],r,0,7);
        if(b[2]){ctx.fillStyle=b[1];ctx.fill()}else{ctx.strokeStyle=b[1];ctx.lineWidth=1.5;ctx.stroke()}})}
    stat.textContent=''}
  // ---------------------------------------------------------------- bodies
  var NAMES=["top_head","center_head","center_shoulder","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","spine","root",
    "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"],IX={};NAMES.forEach(function(n,i){IX[n]=i});
  var BONES=[["left_hip","left_knee",.072],["right_hip","right_knee",.072],["left_knee","left_ankle",.054],["right_knee","right_ankle",.054],
    ["left_shoulder","left_elbow",.046],["right_shoulder","right_elbow",.046],["left_elbow","left_wrist",.038],["right_elbow","right_wrist",.038],
    ["center_head","center_shoulder",.045],["left_shoulder","right_shoulder",.058],["left_hip","right_hip",.085],["center_shoulder","spine",.125],["spine","root",.115]];
  var bp=bodyPlayers.length?bodyPlayers[0]:0,si=0,tt=0,playing=false,ghost=true,last=0,hold=0,stat=document.getElementById('a3d-stat'),statKey='';
  var calm=window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  function strokes(){return(D.players[bp]&&D.players[bp].strokes)||[]}
  function cur(){return strokes()[si]}
  function lerp(a,b,t){return[a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,a[2]+(b[2]-a[2])*t]}
  function bodyAt(s,t){var d=s.dts,n=d.length,F_=s.frames;if(t<=d[0])return F_[0];if(t>=d[n-1])return F_[n-1];var i=0;while(i<n-2&&d[i+1]<t)i++;
    var a=(t-d[i])/((d[i+1]-d[i])||1),A=F_[Math.max(0,i-1)],B=F_[i],C=F_[i+1],E=F_[Math.min(n-1,i+2)],a2=a*a,a3=a2*a;
    return B.map(function(p,j){var r=[0,0,0];for(var k=0;k<3;k++){var p0=A[j][k],p1=p[k],p2=C[j][k],p3=E[j][k];
      r[k]=.5*(2*p1+(-p0+p2)*a+(2*p0-5*p1+4*p2-p3)*a2+(-p0+3*p1-3*p2+p3)*a3)}return r})}
  function along(Pp,u){if(!Pp||Pp.length<2)return null;u=clamp(u,0,1)*(Pp.length-1);var i=Math.min(Pp.length-2,Math.floor(u));return lerp(Pp[i],Pp[i+1],u-i)}
  function ballAt(s,t){if(t<0){if(!s.arc||!s.dti||t<-s.dti)return null;return along(s.arc,1+t/s.dti)}
    if(!s.path||!s.ft)return t<.03?s.ball:null;return t>s.ft?null:along(s.path,t/s.ft)}
  function proFor(s){var ps=D.pro.strokes||[];return ps.filter(function(q){return q.side===s.side})[0]||ps[0]||null}
  function shift(J,d){return J.map(function(p){return[p[0]+d[0],p[1]+d[1],p[2]]})}
  function facing(J){var h=[J[IX.left_hip][0]-J[IX.right_hip][0],J[IX.left_hip][1]-J[IX.right_hip][1],0],f=norm(cross(h,[0,0,1]));return f}
  function body(J,c,g,hand,towards,out){
    BONES.forEach(function(b){var a=P(J[IX[b[0]]]),e=P(J[IX[b[1]]]);if(a&&e)out.push({z:(a[2]+e[2])/2,k:'cap',a:a,b:e,w:2*F*b[2]/((a[2]+e[2])/2),c:c,g:g})});
    var fw=facing(J);['left','right'].forEach(function(sd){var an=J[IX[sd+'_ankle']],h=[an[0]-.04*fw[0],an[1]-.04*fw[1],an[2]-.05],t=[h[0]+.19*fw[0],h[1]+.19*fw[1],h[2]];
      var a=P(h),e=P(t);if(a&&e)out.push({z:(a[2]+e[2])/2,k:'cap',a:a,b:e,w:2*F*.036/((a[2]+e[2])/2),c:c,g:g});
      var q=P(J[IX[sd+'_wrist']]);if(q)out.push({z:q[2],k:'disc',p:q,r:F*.042/q[2],c:c,g:g})});
    var hd=J[IX.center_head],q=P([hd[0],hd[1],hd[2]+.02]);if(q)out.push({z:q[2],k:'disc',p:q,r:F*.092/q[2],c:c,g:g,ball:true});
    var w=J[IX[hand+'_wrist']],el=J[IX[hand+'_elbow']],d=towards?norm([towards[0]-w[0],towards[1]-w[1],towards[2]-w[2]]):norm([w[0]-el[0],w[1]-el[1],w[2]-el[2]]);
    var h1=P(w),h2=P([w[0]+.1*d[0],w[1]+.1*d[1],w[2]+.1*d[2]]),bl=P([w[0]+.17*d[0],w[1]+.17*d[1],w[2]+.17*d[2]]);
    if(h1&&h2)out.push({z:(h1[2]+h2[2])/2,k:'cap',a:h1,b:h2,w:2*F*.014/h1[2],c:g?c:col('--ink-2'),g:g});
    if(bl)out.push({z:bl[2]-.01,k:'disc',p:bl,r:F*.078/bl[2],c:g?c:col('--rust'),g:g,blade:true})}
  function table(out,tbl,tl,ink){var nx=10,ny=6,i,j;
    for(i=0;i<nx;i++)for(j=0;j<ny;j++){var x0=T.L*i/nx,x1=T.L*(i+1)/nx,y0=T.W*j/ny,y1=T.W*(j+1)/ny,qs=[[x0,y0,0],[x1,y0,0],[x1,y1,0],[x0,y1,0]].map(P);
      if(qs.every(Boolean))out.push({z:(qs[0][2]+qs[2][2])/2,k:'poly',q:qs,c:tbl})}
    var L2=[[[0,0],[T.L,0]],[[T.L,0],[T.L,T.W]],[[T.L,T.W],[0,T.W]],[[0,T.W],[0,0]],[[0,T.W/2],[T.L,T.W/2]]];
    L2.forEach(function(l,li){for(var k=0;k<8;k++){var a=P([l[0][0]+(l[1][0]-l[0][0])*k/8,l[0][1]+(l[1][1]-l[0][1])*k/8,.001]),
      e=P([l[0][0]+(l[1][0]-l[0][0])*(k+1)/8,l[0][1]+(l[1][1]-l[0][1])*(k+1)/8,.001]);if(a&&e)out.push({z:Math.min(a[2],e[2])-.06,k:'seg',a:a,b:e,w:li<4?1.6:.8,c:tl})}});
    [[.35,.18],[.35,T.W-.18],[T.L-.35,.18],[T.L-.35,T.W-.18]].forEach(function(l){var a=P([l[0],l[1],T.floor]),e=P([l[0],l[1],-.03]);
      if(a&&e)out.push({z:(a[2]+e[2])/2,k:'seg',a:a,b:e,w:2*F*.025/((a[2]+e[2])/2),c:ink,al:.75})});
    var nq=[[T.net_x,-.15,0],[T.net_x,T.W+.15,0],[T.net_x,T.W+.15,T.net_h],[T.net_x,-.15,T.net_h]].map(P);
    if(nq.every(Boolean)){out.push({z:(nq[0][2]+nq[2][2])/2,k:'poly',q:nq,c:ink,al:.25});out.push({z:(nq[0][2]+nq[2][2])/2-.004,k:'seg',a:nq[3],b:nq[2],w:1.6,c:tl})}}
  function seg(a,b,dx,dy){ctx.beginPath();ctx.moveTo(a[0]+dx,a[1]+dy);ctx.lineTo(b[0]+dx,b[1]+dy);ctx.stroke()}
  function paint(o){ctx.globalAlpha=o.al||1;
    if(o.k==='poly'){ctx.fillStyle=o.c;ctx.strokeStyle=o.c;ctx.lineWidth=.8;ctx.beginPath();o.q.forEach(function(q,i){i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1])});
      ctx.closePath();ctx.fill();if(!o.al)ctx.stroke()}
    else if(o.k==='seg'){ctx.strokeStyle=o.c;ctx.lineWidth=o.w;seg(o.a,o.b,0,0)}
    else if(o.g){ctx.globalAlpha=.3;ctx.fillStyle=o.c;ctx.strokeStyle=o.c;if(o.k==='cap'){ctx.lineWidth=o.w;seg(o.a,o.b,0,0)}else{ctx.beginPath();ctx.arc(o.p[0],o.p[1],o.r,0,7);ctx.fill()}}
    else if(o.k==='cap'){var off=.13*o.w;ctx.strokeStyle=shade(o.c,-.42);ctx.lineWidth=o.w;seg(o.a,o.b,0,0);ctx.strokeStyle=o.c;ctx.lineWidth=o.w*.72;seg(o.a,o.b,-off,-off);
      ctx.strokeStyle=shade(o.c,.35);ctx.lineWidth=Math.max(1,o.w*.2);seg(o.a,o.b,-2*off,-2*off)}
    else if(o.k==='disc'){ctx.fillStyle=shade(o.c,o.ballc?-.25:-.42);ctx.beginPath();ctx.arc(o.p[0],o.p[1],o.r,0,7);ctx.fill();
      ctx.fillStyle=o.c;ctx.beginPath();ctx.arc(o.p[0]-o.r*.14,o.p[1]-o.r*.14,o.r*.84,0,7);ctx.fill();
      if(o.ball||o.ballc){ctx.fillStyle=shade(o.c,.5);ctx.beginPath();ctx.arc(o.p[0]-o.r*.38,o.p[1]-o.r*.38,o.r*.25,0,7);ctx.fill()}}
    ctx.globalAlpha=1}
  function tLabel(t){return Math.abs(t)<.009?'contact':(Math.abs(t).toFixed(2)+' s '+(t<0?'before':'after'))}
  function drawBodies(){var s=cur();if(!s){drawShots();return}frame();
    var ink=col('--ink'),out=[],c=col('--'+D.players[bp].key),ci=s.frames[s.ci];
    table(out,col('--table'),col('--table-line'),ink);
    var J=bodyAt(s,tt);body(J,c,false,s.hand||'right',Math.abs(tt)<.02?s.ball:null,out);
    var g=ghost&&proFor(s);
    if(g){var G=bodyAt(g,tt),gc=g.frames[g.ci];body(shift(G,[ci[IX.root][0]-gc[IX.root][0],ci[IX.root][1]-gc[IX.root][1]]),ink,true,g.hand||'right',null,out)}
    var b=ballAt(s,tt);if(b){var q=P(b);if(q)out.push({z:q[2]-.04,k:'disc',p:q,r:Math.max(2.4,F*.02/q[2]),c:shade(col('--paper'),.5),ballc:true})}
    out.sort(function(a,b){return b.z-a.z}).forEach(paint);
    ctx.strokeStyle=col('--ink-3');ctx.lineWidth=1;ctx.setLineDash([4,4]);if(s.arc){path(s.arc);ctx.stroke()}if(s.path){path(s.path);ctx.stroke()}ctx.setLineDash([]);
    var m=s.m||{},pm=(g&&g.med)||{},sk=bp+'|'+si+'|'+(g?1:0);
    if(sk!==statKey){statKey=sk;stat.innerHTML='Point '+s.point+', shot '+s.shot+' · '+(s.side||'')+' · at contact: trunk <b>'+Math.round(m.lean)+
      '°</b> forward · shoulder turn <b>'+Math.round(m.turn)+'°</b>'+(g?' <span style="color:var(--ink-3)">· pros’ typical '+Math.round(pm.lean)+'°, '+Math.round(pm.turn)+'°</span>':'')}
    var lb=tLabel(tt);if(lb!==tl.textContent)tl.textContent=lb}
  function draw(){queued=false;if(mode==='bodies')drawBodies();else drawShots()}
  cv.a3dDraw=function(t){if(t!=null)tt=t;draw()};                     // for timing the drawing (tests)
  function req(){if(!queued){queued=true;requestAnimationFrame(draw)}}
  function aim(){var s=cur();if(!s)return;var r=s.frames[s.ci][IX.root];st.tx=r[0]+.2;st.ty=r[1];st.tz=r[2]+.02}
  var sel=document.getElementById('a3d-stroke'),slider=document.getElementById('a3d-t'),tl=document.getElementById('a3d-tl')||{},playB=document.getElementById('a3d-play');
  function fill(){if(!sel)return;var ss=strokes();sel.innerHTML=ss.map(function(s,i){return'<option value="'+i+'">Point '+s.point+', shot '+s.shot+' · '+(s.side||'')+
    ' · trunk '+Math.round((s.m||{}).lean)+'°</option>'}).join('');si=Math.min(D.players[bp].typ||0,ss.length-1);sel.value=si;pick()}
  function pick(){var s=cur();if(!s)return;tt=0;slider.value=Math.round(1000*(0-s.dts[0])/((s.dts[s.dts.length-1]-s.dts[0])||1));aim();req()}
  if(sel){sel.addEventListener('change',function(){si=+sel.value;pick()});
    [].slice.call(document.querySelectorAll('[data-step]')).forEach(function(b){b.addEventListener('click',function(){var n=strokes().length;si=(si+(+b.dataset.step)+n)%n;sel.value=si;pick()})});
    [].slice.call(document.querySelectorAll('[data-bp]')).forEach(function(b){b.addEventListener('click',function(){bp=+b.dataset.bp;
      [].slice.call(document.querySelectorAll('[data-bp]')).forEach(function(x){x.setAttribute('aria-pressed',x===b?'true':'false')});fill()})});
    slider.addEventListener('input',function(){var s=cur();if(!s)return;playing=false;playB.setAttribute('aria-pressed','false');playB.textContent='Play';
      tt=s.dts[0]+(s.dts[s.dts.length-1]-s.dts[0])*slider.value/1000;req()});
    playB.addEventListener('click',function(){playing=!playing;playB.setAttribute('aria-pressed',playing?'true':'false');playB.textContent=playing?'Pause':'Play';
      var s=cur();if(playing&&s&&tt>=s.dts[s.dts.length-1]-1e-3)tt=s.dts[0];last=0;hold=0;if(playing)requestAnimationFrame(tick)});
    document.getElementById('a3d-ghost').addEventListener('click',function(){ghost=!ghost;this.setAttribute('aria-pressed',ghost?'true':'false');req()});
    [].slice.call(document.querySelectorAll('[data-bv]')).forEach(function(b){b.addEventListener('click',function(){var v=BV[b.dataset.bv];st.yaw=v[0];st.pitch=v[1];st.dist=v[2];req()})})}
  function tick(ts){if(!playing)return;var s=cur();if(!s)return;if(last){var dt=(ts-last)/1000;if(hold>0){hold-=dt;if(hold<=0)tt=s.dts[0]}
      else{tt+=dt*.25;if(tt>=s.dts[s.dts.length-1]){tt=s.dts[s.dts.length-1];hold=.7}}}
    last=ts;var sv=String(Math.round(1000*(tt-s.dts[0])/((s.dts[s.dts.length-1]-s.dts[0])||1)));if(slider.value!==sv)slider.value=sv;draw();requestAnimationFrame(tick)}
  [].slice.call(document.querySelectorAll('.vmodes [data-mode]')).forEach(function(b){b.addEventListener('click',function(){mode=b.dataset.mode;
    [].slice.call(document.querySelectorAll('.vmodes [data-mode]')).forEach(function(x){x.setAttribute('aria-pressed',x===b?'true':'false')});
    [].slice.call(document.querySelectorAll('.vbar[data-mode]')).forEach(function(v){v.hidden=v.dataset.mode!==mode});
    if(mode==='bodies'){var v=BV.front;st.yaw=v[0];st.pitch=v[1];st.dist=v[2];aim()}else{var w=VIEWS.behind;st.yaw=w[0];st.pitch=w[1];st.dist=w[2];st.tx=.45;st.ty=T.W/2;st.tz=-.15}req()})});
  var drag=null;
  cv.addEventListener('pointerdown',function(e){drag=[e.clientX,e.clientY,st.yaw,st.pitch];try{cv.setPointerCapture(e.pointerId)}catch(_){}});
  cv.addEventListener('pointermove',function(e){if(!drag)return;st.yaw=drag[2]-(e.clientX-drag[0])*.008;st.pitch=clamp(drag[3]+(e.clientY-drag[1])*.006,.05,1.45);req()});
  ['pointerup','pointercancel'].forEach(function(t){cv.addEventListener(t,function(){drag=null})});
  cv.addEventListener('wheel',function(e){e.preventDefault();st.dist=clamp(st.dist*Math.exp(e.deltaY*.0012),1.6,10);req()},{passive:false});
  cv.addEventListener('keydown',function(e){var k=e.key,u=true;
    if(k==='ArrowLeft')st.yaw+=.08;else if(k==='ArrowRight')st.yaw-=.08;else if(k==='ArrowUp')st.pitch=clamp(st.pitch+.05,.05,1.45);
    else if(k==='ArrowDown')st.pitch=clamp(st.pitch-.05,.05,1.45);else if(k==='+'||k==='=')st.dist=clamp(st.dist*.9,1.6,10);
    else if(k==='-')st.dist=clamp(st.dist*1.1,1.6,10);else u=false;if(u){e.preventDefault();e.stopPropagation();req()}});
  [].slice.call(document.querySelectorAll('.vbar [data-k]')).forEach(function(b){b.addEventListener('click',function(){on[b.dataset.k]=!on[b.dataset.k];
    b.setAttribute('aria-pressed',on[b.dataset.k]?'true':'false');req()})});
  [].slice.call(document.querySelectorAll('.vbar [data-l]')).forEach(function(b){b.addEventListener('click',function(){lay[b.dataset.l]=!lay[b.dataset.l];
    b.setAttribute('aria-pressed',lay[b.dataset.l]?'true':'false');req()})});
  [].slice.call(document.querySelectorAll('.vbar [data-v]')).forEach(function(b){b.addEventListener('click',function(){var v=VIEWS[b.dataset.v];
    st.yaw=v[0];st.pitch=v[1];st.dist=v[2];st.tx=.45;st.ty=T.W/2;st.tz=-.15;req()})});
  window.addEventListener('resize',req);
  if(window.matchMedia)window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change',function(){cache={};shades={};K=null;bgKey='';req()});
  [].slice.call(document.querySelectorAll('.vbar[data-mode]')).forEach(function(v){v.hidden=v.dataset.mode!==mode});
  if(mode==='bodies'){var v0=BV.front;st.yaw=v0[0];st.pitch=v0[1];st.dist=v0[2];fill()}
  draw();
})();
(function(){
  var links=[].slice.call(document.querySelectorAll('.topbar nav a'));if(!links.length||!('IntersectionObserver' in window))return;
  var by={},cur=null,nav=links[0].parentNode;links.forEach(function(a){by[a.getAttribute('href').slice(1)]=a});
  var io=new IntersectionObserver(function(es){es.forEach(function(e){if(!e.isIntersecting)return;var a=by[e.target.id];if(!a||a===cur)return;
    if(cur)cur.removeAttribute('aria-current');cur=a;a.setAttribute('aria-current','true');
    nav.scrollTo({left:a.offsetLeft-nav.clientWidth/2+a.clientWidth/2,behavior:'smooth'})})},{rootMargin:'-42% 0px -52% 0px'});
  Object.keys(by).forEach(function(id){var el=document.getElementById(id);if(el)io.observe(el)});
})();
"""


def _svg_table(dots, L=2.74, W=1.525, colour="var(--ball)"):
    """Top-down table, player at the left end. dots: [(x_m, y_m, won)] already oriented; the far side of the picture is up.
    Filled in the player's colour = a point he won, hollow = lost."""
    sx = 100.0
    parts = [f'<svg class="tt" viewBox="-8 -8 {L * sx + 16:.0f} {W * sx + 16:.0f}" role="img" aria-label="{len(dots)} landing positions">',
             f'<rect x="0" y="0" width="{L * sx:.0f}" height="{W * sx:.0f}" fill="var(--table)" stroke="var(--table-line)" stroke-width="3"/>',
             f'<line x1="{L * sx / 2:.0f}" y1="-6" x2="{L * sx / 2:.0f}" y2="{W * sx + 6:.0f}" stroke="var(--table-line)" stroke-width="4"/>',
             f'<line x1="0" y1="{W * sx / 2:.1f}" x2="{L * sx:.0f}" y2="{W * sx / 2:.1f}" stroke="var(--table-line)" stroke-width="1.2" opacity=".55"/>']
    for x, y, won in dots:
        cx, cy = min(max(x, -0.03), L + 0.03) * sx, (W - min(max(y, -0.03), W + 0.03)) * sx
        if won:
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="{colour}" stroke="oklch(20% 0.02 165)" stroke-width="1.2"/>')
        else:
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6.4" fill="none" stroke="var(--table-line)" stroke-width="2.2"/>')
    return "".join(parts) + "</svg>"


def _placements(points, name, L=2.74, W=1.525):
    """(serves, third balls, returns) of one player as oriented dots [(x, y, player won the point)]."""
    out = {1: [], 3: [], 2: []}
    for p in points:
        end = "near" if p.get("near_player") == name else "far"
        serving = p.get("server") == name
        for l in p.get("landings", []):
            if (serving and l["shot"] in (1, 3)) or (not serving and l["shot"] == 2):
                x, y = (l["x_m"], l["y_m"]) if end == "near" else (L - l["x_m"], W - l["y_m"])
                out[l["shot"]].append((x, y, p.get("winner_name") == name))
    return out[1], out[3], out[2]


def _pct(a, b):
    return None if not b else round(100 * a / b)


def _sheet(ms, names):
    """The scoresheet: a column per game, the winner's score ringed; an unfinished last game in italics."""
    games = ms.get("games") or []
    if not games:
        return ""
    e = html.escape
    head = "".join(f'<th scope="col">G{g["n"]}{"*" if not g["finished"] else ""}</th>' for g in games)
    rows = ""
    for i, n in enumerate(names):
        cells = "".join(f'<td class="{"w" if g["winner"] == n else ""}{" open" if not g["finished"] else ""}"><span>{g["score"].get(n, 0)}</span></td>' for g in games)
        rows += f'<tr><th scope="row"><i style="--c:var(--p{i + 1})"></i>{e(n)}</th>{cells}</tr>'
    open_note = ("* in play when the recording stopped. " if not games[-1]["finished"] else "")
    return (f'<div class="sheetwrap"><table class="sheet"><caption class="sr">Game scores</caption><thead><tr><th><span class="sr">Player</span></th>{head}</tr></thead>'
            f'<tbody>{rows}</tbody></table><p>{open_note}Games to 11, won by 2, as tt_scout counted the points; one wrong call moves where a game ends.</p></div>')


def _bar_row(k, label, a, b, fa, fb, sub_a="", sub_b="", scale=None, lower_better=False, note=""):
    """One head-to-head row. a, b: the numbers compared (None = not measured); fa, fb: how they are printed."""
    if a is None and b is None:
        return ""
    top = scale or max(v for v in (a, b) if v is not None) or 1
    lead = None
    if a is not None and b is not None and a != b:
        lead = 0 if ((a < b) if lower_better else (a > b)) else 1
    def v(side, val, txt, sub):
        return (f'<div class="v {"lr"[side]}{" lead" if lead == side else ""}"><b>{txt if val is not None else "–"}</b>'
                + (f"<small>{sub}</small>" if sub else "") + "</div>")
    def bar(side, val):
        w = 0 if val is None else max(0.0, min(1.0, val / top))
        return f'<div class="bar {"lr"[side]}"><i class="{"soft" if lead is not None and lead != side else ""}" style="--w:{w:.3f};--c:var(--p{side + 1});--k:{k}"></i></div>'
    return (f'<li>{v(0, a, fa, sub_a)}{bar(0, a)}<p class="lab">{label}' + (f"<small>{note}</small>" if note else "") + f'</p>{bar(1, b)}{v(1, b, fb, sub_b)}</li>')


def _serve_length_row(k, P, names):
    """Serve length as each player mixes it: short, half-long, long, as one stacked bar each."""
    mix = [P[n].get("serve_length") or {} for n in names]
    if not any(sum(m.values()) for m in mix):
        return ""
    order = ("short", "half-long", "long")
    def v(side, m):
        tot = sum(m.values())
        if not tot:
            return f'<div class="v {"lr"[side]}"><b>–</b></div>'
        top = max(order, key=lambda o: m.get(o, 0))
        return f'<div class="v {"lr"[side]} lead"><b>{round(100 * m.get(top, 0) / tot)}%</b><small>{top}</small></div>'
    def bar(side, m):
        tot = sum(m.values()) or 1
        segs = [(o, m.get(o, 0) / tot) for o in order]
        if side == 0:
            segs = segs[::-1]                                          # both bars read short to long from the middle outwards
        return (f'<div class="bar {"lr"[side]}">' + "".join(f'<i class="{("s3", "s2", "")[order.index(o)]}" style="--w:{w:.3f};--c:var(--p{side + 1});--k:{k}" title="{o} {round(100 * w)}%"></i>'
                                                            for o, w in segs) + "</div>")
    return (f'<li>{v(0, mix[0])}{bar(0, mix[0])}<p class="lab">Serve length<small>short · half-long · long, from the net outwards</small></p>'
            f'{bar(1, mix[1])}{v(1, mix[1])}</li>')


def _h2h(ms, names):
    P = ms["players"]; A, B = (P[n] for n in names)
    rows, k = [], 0
    def add(*a, **kw):
        nonlocal k
        r = _bar_row(k, *a, **kw)
        if r:
            rows.append(r); k += 1
    def grp(t):
        rows.append(f'<li class="grp">{t}</li>')
    grp("The score")
    if A["games"] or B["games"]:
        add("Games won", A["games"], B["games"], str(A["games"]), str(B["games"]))
    add("Points won", A["points"], B["points"], str(A["points"]), str(B["points"]))
    add("Most points in a row", A["run"], B["run"], str(A["run"]), str(B["run"]))
    grp("Serve and receive")
    sa, sb = _pct(*A["serve"]), _pct(*B["serve"])
    add("Points won on serve", sa, sb, f"{sa}%" if sa is not None else "", f"{sb}%" if sb is not None else "", f"{A['serve'][0]} of {A['serve'][1]}", f"{B['serve'][0]} of {B['serve'][1]}", scale=100)
    ra, rb = _pct(*A["receive"]), _pct(*B["receive"])
    add("Points won on receive", ra, rb, f"{ra}%" if ra is not None else "", f"{rb}%" if rb is not None else "", f"{A['receive'][0]} of {A['receive'][1]}", f"{B['receive'][0]} of {B['receive'][1]}", scale=100)
    rows.append(_serve_length_row(k, P, names)); k += 1
    grp("Where points were decided")
    for key, label in (("serve", "Serve and third ball"), ("receive", "Receive and fourth ball"), ("rally", "Rally, fifth shot on")):
        (wa, la), (wb, lb) = A["phases"][key], B["phases"][key]
        pa, pb = _pct(wa, wa + la), _pct(wb, wb + lb)
        add(label, pa, pb, f"{pa}%" if pa is not None else "", f"{pb}%" if pb is not None else "", f"won {wa} · lost {la}", f"won {wb} · lost {lb}", scale=100,
            note="points decided by his own shot there")
    add("Winning shots", A["not_returned"], B["not_returned"], str(A["not_returned"]), str(B["not_returned"]), note="balls that did not come back")
    add("Shots that missed the table", A["missed_table"], B["missed_table"], str(A["missed_table"]), str(B["missed_table"]), lower_better=True, note="long or wide")
    grp("Pace")
    fa = ("~" if A.get("fastest_est") else "") + f"{A['fastest_kmh']}" if A["fastest_kmh"] else ""
    fb = ("~" if B.get("fastest_est") else "") + f"{B['fastest_kmh']}" if B["fastest_kmh"] else ""
    add("Fastest shot", A["fastest_kmh"], B["fastest_kmh"], fa, fb, "km/h", "km/h",
        note="every strike, landed or not; ~ = one that never bounced, measured to about 8%" if (A.get("fastest_est") or B.get("fastest_est")) else "")
    add("Typical rally shot", A["rally_kmh"], B["rally_kmh"], f"{A['rally_kmh']}" if A["rally_kmh"] else "", f"{B['rally_kmh']}" if B["rally_kmh"] else "", "km/h", "km/h")
    add("Typical serve", A["serve_kmh"], B["serve_kmh"], f"{A['serve_kmh']}" if A["serve_kmh"] else "", f"{B['serve_kmh']}" if B["serve_kmh"] else "", "km/h", "km/h")
    e = html.escape
    return (f'<section class="h2h"><div class="head"><h2>Match stats</h2><span>head to head, from tt_scout’s count</span></div>'
            f'<div class="names2"><b style="--c:var(--p1)">{e(names[0])}</b><b style="--c:var(--p2)">{e(names[1])}</b></div>'
            f'<ol class="bars">{"".join(rows)}</ol>'
            '<p class="note2">Each point is put down to the shot that decided it: the shot that won it, or the one that missed. Serve and third ball are the '
            'server’s first two shots, receive and fourth ball the receiver’s, rally is everything from the fifth shot: the three-phase split table '
            'tennis coaching uses. A ball returned into the net counts for the shot that forced it, because the camera cannot tell a forced miss from a '
            'free one. Speeds are the ball\u2019s speed as it left the racket: each shot\u2019s flight is fitted in 3D (gravity, air drag, spin) '
            'through the camera recovered from the table\u2019s corners, from the moment of contact (where the incoming and outgoing paths '
            'meet when the racket or body hides the ball, or, failing that, when the flight passes the hitter\u2019s wrist). Every strike is '
            'counted: one that bounced on the table is within 3% typically (95% of shots within 9%, on simulated shots through this camera); '
            'one that never did (long, wide, into the net) is fitted without the bounce and marked ~, an estimate within 8% typically (95% '
            'within 23%). The same method gives the professional OpenTTGames test matches rally shots of about 12 m/s (43 km/h).</p></section>')


def _momentum(ms, names, clip_of):
    mom, games = ms.get("momentum") or [], ms.get("games") or []
    if len(mom) < 4:
        return ""
    W_, top, bot = 1000.0, 40.0, 214.0
    peak = max(3, max(abs(m["lead"]) for m in mom))
    gids = sorted(set(m["game"] for m in mom))
    gap = 22.0; n_all = len(mom)
    unit = (W_ - gap * (len(gids) - 1)) / n_all
    y0 = (top + bot) / 2; ky = (bot - top) / 2 / peak
    parts = [f'<svg viewBox="-4 0 {W_ + 8:.0f} {bot + 26:.0f}" role="img" aria-label="Who led, point by point, in each game">',
             f'<defs><clipPath id="mu"><rect x="-10" y="0" width="{W_ + 20:.0f}" height="{y0:.1f}"/></clipPath>'
             f'<clipPath id="md"><rect x="-10" y="{y0:.1f}" width="{W_ + 20:.0f}" height="{bot:.0f}"/></clipPath></defs>',
             f'<text class="who" x="0" y="{top - 4:.0f}" fill="var(--p1)">{html.escape(names[0]).upper()} AHEAD</text>',
             f'<text class="who" x="0" y="{bot + 16:.0f}" fill="var(--p2)">{html.escape(names[1]).upper()} AHEAD</text>']
    x = 0.0
    by_game = {g["n"]: g for g in games}
    for gi, g in enumerate(gids):
        pts = [m for m in mom if m["game"] == g]
        w = unit * len(pts)
        xs = [x + unit * (i + 1) for i in range(len(pts))]
        line = f"M{x:.1f},{y0:.1f} " + " ".join(f"L{xi:.1f},{y0 - m['lead'] * ky:.1f}" for xi, m in zip(xs, pts))
        area = line + f" L{xs[-1]:.1f},{y0:.1f} Z"
        parts.append(f'<path d="{area}" fill="var(--p1)" opacity=".55" clip-path="url(#mu)"/><path d="{area}" fill="var(--p2)" opacity=".5" clip-path="url(#md)"/>')
        parts.append(f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x + w:.1f}" y2="{y0:.1f}" stroke="var(--line-2)" stroke-width="1"/>')
        parts.append(f'<path class="line" d="{line}"/>')
        for xi, m in zip(xs, pts):
            col = "var(--p1)" if m["winner"] == names[0] else "var(--p2)"
            parts.append(f'<circle cx="{xi:.1f}" cy="{y0 - m["lead"] * ky:.1f}" r="2.6" fill="{col}"/>')
            if m["id"] in clip_of:
                parts.append(f'<rect x="{xi - unit / 2:.1f}" y="{top - 6:.0f}" width="{unit:.1f}" height="{bot - top + 12:.0f}" fill="transparent" data-go="{m["id"]}">'
                             f'<title>Point {m["id"]}: won by {html.escape(m["winner"])} (play)</title></rect>')
        gg = by_game.get(g)
        if gg:
            sc = f'{gg["score"].get(names[0], 0)}–{gg["score"].get(names[1], 0)}'
            wcol = "var(--p1)" if gg["winner"] == names[0] else ("var(--p2)" if gg["winner"] == names[1] else "var(--ink-3)")
            parts.append(f'<text class="gl" x="{x:.1f}" y="14">G{g}{"" if gg["finished"] else "*"}</text>'
                         f'<text class="gs" x="{x + 30:.1f}" y="15">{sc}</text><circle cx="{x + 30 + 11 * len(sc):.1f}" cy="10" r="3.4" fill="{wcol}"/>')
        if gi:
            parts.append(f'<line x1="{x - gap / 2:.1f}" y1="{top - 10:.0f}" x2="{x - gap / 2:.1f}" y2="{bot + 4:.0f}" stroke="var(--line)" stroke-width="1"/>')
        x += w + gap
    parts.append("</svg>")
    return (f'<section class="mom"><div class="head"><h2>Momentum</h2><span>the lead inside each game, point by point{" · click a point to play it" if clip_of else ""}</span></div>'
            + "".join(parts) + "</section>")


def _facts(ms, clip_of):
    m = ms.get("match") or {}
    items = []
    if m.get("mean_shots") is not None:
        items.append(("Shots per point", f'{m["mean_shots"]:.1f}', "on average"))
    if m.get("within_four") is not None:
        items.append(("Over within four shots", f'{m["within_four"]}%', "serve, return, third and fourth ball"))
    lg = m.get("longest")
    if lg and lg.get("shots"):
        play = (f' <button class="go" type="button" data-go="{lg["id"]}" aria-label="Play point {lg["id"]}">play</button>' if lg["id"] in clip_of else "")
        items.append(("Longest rally", f'{lg["shots"]}', f'shots, point {lg["id"]} at {mmss(lg["t"])}{play}'))
    if m.get("tempo_s"):
        items.append(("Between shots", f'{m["tempo_s"]:.2f}', "seconds, typical"))
    return '<dl class="facts">' + "".join(f"<div><dt>{a}</dt><dd>{b}<small>{c}</small></dd></div>" for a, b, c in items) + "</dl>"


def _contact_strips(frames, posture, names, clip_of):
    """Real frames of each player at his gated contacts, deepest knee bend first, skeleton and knee angle inked on (hero.contact_frames),
    under the one knee number: the median of those contacts, as the camera sees it, or 'not measurable from this camera'."""
    from .pose import NOT_MEASURABLE, KNEE_MIN_N
    e = html.escape; rows = []
    for i, n in enumerate(names):
        tiles = (frames or {}).get(n) or []
        pm = (posture or {}).get(n) or {}
        if not tiles and not pm:
            continue
        med = tuple(pm.get("knee") or (None, 0)); why = pm.get("knee_why")
        figs = "".join(
            f'<figure><img src="{t["src"]}" width="300" height="400" loading="lazy" alt="{e(n)} at contact, knee at {t["knee"]} degrees">'
            f'<figcaption><b>{t["knee"]}\u00b0</b>'
            + (f'<button class="go" type="button" data-go="{t["point"]}" aria-label="Play point {t["point"]}">point {t["point"]}</button>' if t.get("point") in clip_of
               else (f"<span>point {t['point']}</span>" if t.get("point") else ""))
            + "</figcaption></figure>" for t in tiles)
        lean = tuple(pm.get("lean") or (None, 0))
        kw, kl = tuple(pm.get("knee_won") or (None, 0)), tuple(pm.get("knee_lost") or (None, 0))
        if med[0] is not None:
            knee = f'<b>{med[0]:.0f}\u00b0</b> typical knee bend at contact, as the camera sees it <small>(median of {med[1]} forehand contacts seen in profile)</small>'
        elif why:
            knee = f'<b>{e(NOT_MEASURABLE)}</b> <small>knee bend at contact: {e(why)}</small>'
        else:
            knee = (f'<b>{e(NOT_MEASURABLE)}</b> <small>knee bend at contact: {med[1]} forehand contact{"" if med[1] == 1 else "s"} seen in profile, '
                    f'{KNEE_MIN_N} are needed</small>')
        facts = ('<p>' + knee
                 + (f'<br><b>{abs(lean[0]):.0f}\u00b0</b> trunk lean {"forward" if lean[0] >= 0 else "back"}, as the camera sees it <small>({lean[1]} hits)</small>' if lean[0] is not None else "")
                 + (f'<br><small>won points {kw[0]:.0f}\u00b0 \u00b7 lost points {kl[0]:.0f}\u00b0</small>' if kw[0] is not None and kl[0] is not None else "")
                 + "</p>")
        rows.append(f'<div class="strip"><h3><i style="--c:var(--p{i + 1})"></i>{e(n)}</h3>{facts}<div class="tiles">{figs}</div></div>')
    return "".join(rows)


def _stance(stance, posture, names, clip_of=None):
    """Each player's real frames at contact (deepest knee bend to straightest), then the numbers."""
    if isinstance(stance, dict) and "frames" in stance:
        strips = _contact_strips(stance.get("frames"), posture, names, clip_of or {})
        if strips:
            any_tile = any((stance.get("frames") or {}).get(n) for n in names)     # the sub-head promises frames only when there are some
            sub = "deepest knee bend to straightest, skeletons from Apple Vision on this computer" if any_tile else "skeletons from Apple Vision on this computer"
            return (f'<section class="stance"><div class="head"><h2>Posture at contact</h2><span>{sub}</span></div>'
                    f'{strips}<p class="note2">Knee bend is the hip, knee and ankle angle as the camera sees it: 180\u00b0 is a straight leg, lower is deeper. '
                    'It is an angle in the picture, not a 3D joint angle, so it is read only where the picture can show it: on rally forehands, on the leg nearer '
                    'the camera, with both legs found, and only at contacts where the player is seen side-on (his hips within 30\u00b0 of the line of sight); '
                    'a player turned towards the camera reads straighter than he is, so those contacts are left out. Fewer than three such contacts and the '
                    'knee is not measurable from this camera. The pros\u2019 reference in the critique is measured with the same rule.</p></section>')
        stance = stance.get("figures")
    if not stance or not any(stance.get(n) for n in names):
        return ""
    from .pose import BONES
    sc, W_, H_ = 92.0, 760.0, 380.0
    parts = [f'<svg viewBox="0 0 {W_:.0f} {H_:.0f}" role="img" aria-label="Typical stance at contact for {html.escape(names[0])} and {html.escape(names[1])}">']
    floor = []
    for i, n in enumerate(names):
        st = stance.get(n)
        if not st:
            continue
        cx = 190.0 if i == 0 else W_ - 190.0
        flip = 1.0 if i == 0 else -1.0                                   # the left figure faces right, the right one faces left
        P = {j: (cx + flip * x * sc, 150.0 + y * sc) for j, (x, y) in st.items()}
        col = f"var(--p{i + 1})"
        for a, b in BONES:
            if (a, b) == ("neck", "nose") or a not in P or b not in P:
                continue
            (x1, y1), (x2, y2) = P[a], P[b]
            parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="var(--ink)" stroke-width="12" stroke-linecap="round"/>')
            parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" stroke-width="6.5" stroke-linecap="round"/>')
        if "nose" in P and "neck" in P:
            hx = (P["nose"][0] * 0.6 + P["neck"][0] * 0.4); hy = (P["nose"][1] * 0.75 + P["neck"][1] * 0.25) - 0.12 * sc
            parts.append(f'<circle cx="{hx:.1f}" cy="{hy:.1f}" r="{0.27 * sc:.1f}" fill="{col}" stroke="var(--ink)" stroke-width="3"/>')
        for j, (x, y) in P.items():
            if j not in ("leye", "reye", "lear", "rear", "nose"):
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="var(--paper)" stroke="var(--ink)" stroke-width="1.6"/>')
        # the camera-near leg of the median figure (the longer hip-to-ankle span, pose.near_leg's rule): an arc and the gated median
        pm = (posture or {}).get(n) or {}
        kn = [(s, P[s + "hip"], P[s + "kne"], P[s + "ank"]) for s in ("l", "r") if all(s + j in P for j in ("hip", "kne", "ank"))]
        if kn and pm.get("knee") and pm["knee"][0] is not None:
            s_, h, kk, a = max(kn, key=lambda t: (t[1][0] - t[3][0]) ** 2 + (t[1][1] - t[3][1]) ** 2)
            import math
            a1 = math.atan2(h[1] - kk[1], h[0] - kk[0]); a2 = math.atan2(a[1] - kk[1], a[0] - kk[0])
            r_ = 30.0
            p1 = (kk[0] + r_ * math.cos(a1), kk[1] + r_ * math.sin(a1)); p2 = (kk[0] + r_ * math.cos(a2), kk[1] + r_ * math.sin(a2))
            sweep = 1 if ((a2 - a1) % (2 * math.pi)) < math.pi else 0
            parts.append(f'<path d="M{p1[0]:.1f},{p1[1]:.1f} A{r_:.0f},{r_:.0f} 0 0 {sweep} {p2[0]:.1f},{p2[1]:.1f}" fill="none" stroke="var(--ink)" stroke-width="2"/>')
            tx = kk[0] + flip * 44
            parts.append(f'<text class="deg" x="{tx:.1f}" y="{kk[1] + 8:.1f}" text-anchor="{"start" if i == 0 else "end"}">{pm["knee"][0]:.0f}°</text>')
        floor.append(max(y for j, (x, y) in P.items()))
        lean = pm.get("lean") or (None, 0)
        from .pose import knee_text
        cap = "knee bend " + knee_text(tuple(pm.get("knee") or (None, 0)), why=pm.get("knee_why")) if pm else ""
        if lean[0] is not None:
            cap += f" · leans {abs(lean[0]):.0f}° {'forward' if lean[0] >= 0 else 'back'}"
        parts.append(f'<text class="nm" x="{cx:.0f}" y="{H_ - 34:.0f}" text-anchor="middle" fill="{col}">{html.escape(n).upper()}</text>'
                     f'<text class="cap" x="{cx:.0f}" y="{H_ - 14:.0f}" text-anchor="middle">{cap}</text>')
    if floor:
        fy = max(floor) + 12
        parts.insert(1, f'<line x1="40" y1="{fy:.1f}" x2="{W_ - 40:.0f}" y2="{fy:.1f}" stroke="var(--line-2)" stroke-width="1.5"/>')
    parts.append("</svg>")
    rows = ""
    if posture and all(n in posture for n in names):
        def deg(v):
            return "–" if v[0] is None else f"<b>{v[0]:.0f}°</b> <small>({v[1]})</small>"
        rows = "".join(f"<tr><td>{k}</td>" + "".join(f"<td>{f(posture[n])}</td>" for n in names) + "</tr>"
                       for k, f in (("Knee bend at contact, as the camera sees it", lambda p: deg(p["knee"])), (" in points he won", lambda p: deg(p["knee_won"])),
                                    (" in points he lost", lambda p: deg(p["knee_lost"])), ("Trunk lean towards the table, as the camera sees it", lambda p: deg(p["lean"]))))
        rows = (f'<table><thead><tr><th><span class="sr">Measure</span></th>' + "".join(f"<th>{html.escape(n)}</th>" for n in names) + f"</tr></thead><tbody>{rows}</tbody></table>")
    return (f'<section class="stance numbers"><div class="head"><h2>Posture at contact</h2><span>skeletons from Apple Vision, run on this computer</span></div>{"".join(parts)}{rows}'
            '<p class="note2">Each figure is the median of that player’s skeletons at the moment of his hits, scaled to one torso length and turned to face the '
            'table. Forehands and backhands are mixed, so read the legs and the trunk, not the arms. Knee bend is the hip, knee and ankle angle as the camera '
            'sees it (180° is a straight leg; lower is deeper), so it compares shots filmed this way rather than giving textbook values, and it is read '
            'only at rally forehands seen in profile (the arc sits on the leg nearer the camera). Numbers in brackets: gated forehand contacts for the '
            'knee rows, hits for the lean.</p></section>')


def _after3d(a3, names):
    """After the match, in 3D (analysis3d.py): per player, three plates (where to stand, where and when to strike, where to aim) with
    the coaching notes and their numbers beside them, light and dark renders; then a canvas viewer to turn every shot round."""
    import json
    if not a3 or not a3.get("players"):
        return ""
    e = html.escape
    topics = (("stand", "Where to stand", "Distance and stance are about right"), ("pose", "How to stand into the ball", "Posture at contact is close to the pros’"),
              ("strike", "Where and when to strike", "Timing is close to the pros’"), ("flight", "Where to aim", "The flight is close to the pros’"))
    alts = dict(stand="{n}’s feet at every contact, drawn in 3D from behind his end, against where the pros stand",
                strike="The ball rising off the bounce into {n}’s racket, seen from the side: each contact, taken at the top or let drop first",
                pose="{n}’s typical forehand at the moment of contact as a 3D body, beside the pros’ typical forehand, with trunk lean and shoulder turn marked",
                flight="Every fitted flight of {n}’s shots from the racket to the bounce, a typical one against a typical one of the pros’")
    tabs, panels = [], []
    for i, p in enumerate(a3["players"]):
        n, pid = p["name"], f"a3d-p{i + 1}"
        tabs.append(f'<button type="button" role="tab" id="{pid}-t" aria-controls="{pid}" aria-selected="{"true" if i == 0 else "false"}" '
                    f'tabindex="{0 if i == 0 else -1}" style="--c:var(--{p["key"]})"><i></i>{e(n)}<small>{p["n"]} shots</small></button>')
        arts = []
        k = 0
        for topic, kicker, fine in topics:
            if topic not in p["plates"]:
                continue
            k += 1
            adv = p["advice"].get(topic) or {}
            items = adv.get("items") or []
            pl = p["plates"][topic]
            body = "".join((f"<h3>{e(h)}</h3>" if j == 0 else f"<h4>{e(h)}</h4>") + f"<p>{e(b)}</p>" for j, (h, b) in enumerate(items)) or f"<h3>{e(fine)}</h3>"
            src = pl.get("dark") or pl.get("light")
            arts.append(f'<article class="plate"><picture><img src="{src}" width="1600" height="900" loading="lazy" decoding="async" '
                        f'alt="{e(alts[topic].format(n=n))}"></picture><div class="txt"><p class="k">{k} · {kicker}</p>{body}'
                        f'<p class="sum">{e(adv.get("summary") or "")}</p></div></article>')
        panels.append(f'<div class="a3d-panel" role="tabpanel" id="{pid}" aria-labelledby="{pid}-t">{"".join(arts)}</div>')
    scene = json.dumps(a3["scene"], separators=(",", ":")).replace("</", "<\\/")
    who = "".join(f'<button type="button" aria-pressed="true" data-k="p{i}" style="--c:var(--{p["key"]})"><i></i>{e(p["name"])}</button>'
                  for i, p in enumerate(a3["scene"]["players"]))
    has_bodies = any(p.get("strokes") for p in a3["scene"]["players"])
    bodies_bar = ""
    if has_bodies:
        pick = "".join(f'<button type="button" aria-pressed="{"true" if i == 0 else "false"}" data-bp="{i}" style="--c:var(--{p["key"]})"><i></i>{e(p["name"])}</button>'
                       for i, p in enumerate(a3["scene"]["players"]) if p.get("strokes"))
        bodies_bar = ('<div class="vbar" data-mode="bodies" hidden><div class="grp" role="group" aria-label="Whose strokes">' + pick + '</div>'
                      '<div class="grp"><label class="vsel"><span>Stroke</span><select id="a3d-stroke"></select></label>'
                      '<button type="button" data-step="-1" aria-label="Previous stroke">‹</button><button type="button" data-step="1" aria-label="Next stroke">›</button></div>'
                      '<div class="grp"><button type="button" id="a3d-play" aria-pressed="false">Play</button>'
                      '<input type="range" id="a3d-t" min="0" max="1000" value="667" aria-label="Time through the stroke"><output id="a3d-tl">contact</output></div>'
                      '<div class="grp" role="group" aria-label="Compare"><button type="button" aria-pressed="true" id="a3d-ghost" style="--c:var(--ink)"><i></i>Pro ghost</button></div>'
                      '<div class="grp" role="group" aria-label="Camera"><button type="button" data-bv="front">Front</button><button type="button" data-bv="side">Side</button>'
                      '<button type="button" data-bv="behind">Behind</button><button type="button" data-bv="above">Above</button></div></div>')
    modes = ('<div class="vmodes" role="group" aria-label="What to show"><button type="button" aria-pressed="true" data-mode="bodies">Bodies</button>'
             '<button type="button" aria-pressed="false" data-mode="shots">Every shot</button></div>') if has_bodies else ""
    viewer = (f'<figure class="viewer"><h3>Turn it round</h3>{modes}{bodies_bar}<div class="vbar" data-mode="shots"{" hidden" if has_bodies else ""}>'
              f'<div class="grp" role="group" aria-label="Whose shots">{who}'
              '<button type="button" aria-pressed="false" data-k="pro" style="--c:var(--ink)"><i></i>Pros</button></div>'
              '<div class="grp" role="group" aria-label="What to draw"><button type="button" aria-pressed="true" data-l="flights">Flights</button>'
              '<button type="button" aria-pressed="true" data-l="contacts">Contacts</button><button type="button" aria-pressed="false" data-l="feet">Feet</button></div>'
              '<div class="grp" role="group" aria-label="Camera"><button type="button" data-v="behind">Behind</button><button type="button" data-v="side">Side</button>'
              '<button type="button" data-v="above">Above</button></div></div>'
              '<canvas id="a3d-cv" tabindex="0" role="img" aria-label="A turnable 3D view of the table: the players’ bodies through each stroke, the ball’s '
              'arc into the racket and its flight, and a professional’s stroke as a ghost to compare"></canvas><p class="vstat" id="a3d-stat" aria-live="polite"></p>'
              '<figcaption>Drag to turn it, scroll to zoom; with it focused, the arrow keys turn it and + and - zoom. Bodies: every stroke is replayed at a quarter speed; '
              'the grey ghost is the pros’ typical stroke of the same side, stood where he stood and timed to the same contact. Every shot: filled balls were taken '
              f'at the top of the bounce or rising, rings let drop first.</figcaption><script type="application/json" id="a3d-data">{scene}</script></figure>')
    pro = a3.get("pro") or {}
    note = ('<p class="note2">Drawn from measurements, not generated. Each arc is a shot’s flight fitted in 3D through the camera found from the table’s corners; '
            'each ball is where that flight was at the racket; each footprint is the hitter’s ankles at contact taken down to the floor; the rising arc into the racket '
            'joins the fitted bounce to the fitted contact under gravity. On simulated shots through this camera the contact point comes out within about 8 cm along the '
            'table and 2 cm in height (median). Where the ball was hidden at the racket and the contact could not be pinned down, the shot is left out of the first two '
            'views. Every player is turned to play from the left end, so both can be laid over the pros: the same measurements on '
            f'{pro.get("n", "")} rally shots from three professional OpenTTGames matches (CC BY-NC-SA 4.0). The advice is rules over these numbers, not a language model.</p>')
    return (f'<section class="a3d" id="after"><div class="head"><h2>After the match, in 3D</h2><span>where to stand, where and when to strike, where to aim; '
            f'measured, against the pros</span></div><div class="a3d-tabs" role="tablist" aria-label="Player">{"".join(tabs)}</div>{"".join(panels)}{viewer}{note}</section>')


NAV = {"Match stats": "Stats", "Coach’s critique": "Critique", "After the match, in 3D": "3D", "Momentum": "Momentum", "The whole match": "Video",
       "Points": "Points", "Scout tips": "Tips", "Placement": "Placement", "Rallies": "Rallies", "Posture at contact": "Posture"}


def _topbar(page, names, big):
    """Give every section an anchor and put a bar on top of the page: the match in one line, a link to each section."""
    import re
    links = []
    def anchor(m):
        attrs, title = m.group(1) or "", m.group(2)
        found = re.search(r'id="([^"]+)"', attrs)
        sid = found.group(1) if found else (re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or f"s{len(links)}")
        links.append((sid, NAV.get(title, title)))
        return m.group(0) if found else m.group(0).replace("<section" + attrs, f'<section{attrs} id="{sid}"', 1)
    page = re.sub(r'<section((?: [a-z-]+="[^"]*")*)><div class="head"><h2>([^<]+)</h2>', anchor, page)
    e = html.escape
    sc = (f'<p class="sc"><i style="--c:var(--p1)"></i>{e(names[0])} <b>{big.get(names[0], "")}</b><em></em><b>{big.get(names[1], "")}</b> {e(names[1])}'
          f'<i style="--c:var(--p2)"></i></p>') if len(names) == 2 else ""
    nav = "".join(f'<a href="#{sid}">{e(label)}</a>' for sid, label in links)
    bar = f'<div class="topbar"><div class="in">{sc}<nav aria-label="Sections">{nav}</nav></div></div>'
    return page.replace("<body><div class=\"wrap\">", "<body>" + bar + "<div class=\"wrap\">", 1)


PLAY_SVG = ('<svg viewBox="0 0 64 64" aria-hidden="true"><circle cx="32" cy="32" r="30" fill="oklch(10% 0.004 250 / .35)" stroke="currentColor" '
            'stroke-width="2"/><path d="M26 20 L46 32 L26 44 Z" fill="currentColor"/></svg>')


def hero_block(hero, lg, names, clip_of, has_whole, dur):
    """The top of the report as a video that plays where it is (the longest rally, poster = the hero frame), and under it the way to
    every other video: the whole match, every point. "" when that rally has no clip (the caller keeps the still)."""
    e = html.escape
    clip = clip_of.get(hero["point"])
    if not clip:
        return ""
    what = (f', the longest rally: {lg.get("n_crossings")} shots over the net, at {mmss(lg["start_t"])}' if lg else "")
    bar = ((f'<a class="first" href="#the-whole-match">Watch the whole match <small>{mmss(dur)}</small></a>' if has_whole else "")
           + f'<a{"" if has_whole else " class=first"} href="#points">Every point, one by one <small>{len(clip_of)} clips</small></a>')
    ar = f' style="--ar:{hero["w"]}/{hero["h"]}"' if hero.get("w") and hero.get("h") else ""
    return (f'<figure class="hero"><div class="hero-v"{ar}><video playsinline preload="none" poster="{hero["src"]}" src="{e(clip)}" '
            f'aria-label="Point {hero["point"]}, the longest rally"></video><button class="bigplay" type="button" aria-label="Play the longest rally">'
            f'{PLAY_SVG}<span>Play the longest rally</span></button></div><figcaption><span><b>Point {hero["point"]}</b>{what}</span></figcaption>'
            f'</figure><nav class="watchbar" aria-label="The videos">{bar}</nav>')


def make_html(out_dir, title, points, quality, video=None, clips=True, calibration=None, overlay=None, web_fonts=False, reuse_clips=False,
              comic=True, scoreboard=True, tips=True, show_table=False, posture=None, hero=None, stance=None, profiles=None, critique=None,
              after3d=None):
    """overlay = dict(table, track, events, obs, fps) draws the tracking onto the clips; without it the clips are plain cuts.
    web_fonts adds a Google Fonts link for Fraunces + Manrope (a network request: off by default, the report stays offline).
    Design context: a scorer's sheet (warm paper, ink rules, serif numerals, the verdict as a stamp)."""
    import datetime, json
    from .tips import match_tips, assign
    out = pathlib.Path(out_dir)
    tip_list = match_tips(points, quality) if tips else []
    on_clip = assign(tip_list, points)
    if tips:
        (out / "tips.json").write_text(json.dumps([dict(t.to_json(), on_clips=sorted(i for i, x in on_clip.items() if x is t)) for t in tip_list], indent=1))
    have = {p["id"]: f"clips/point_{p['id']:03d}.mp4" for p in points if (out / "clips" / f"point_{p['id']:03d}.mp4").exists()}
    if reuse_clips and len(have) == len(points) and points:
        clip_of = have
    elif clips and video and overlay:
        clip_of = annotated_clips(video, points, out, comic=comic, scoreboard=scoreboard, tips=on_clip, show_table=show_table, **overlay)
    else:
        clip_of = cut_clips(video, points, out) if (clips and video) else {}
    e = html.escape
    names = []
    for p in points:
        for k in ("near_player", "far_player"):
            if p.get(k) and p[k] not in names:
                names.append(p[k])
    names = (names + ["Left player", "Right player"])[:2]
    won = {n: sum(1 for p in points if p.get("winner_name") == n) for n in names}
    served = {n: sum(1 for p in points if p.get("server") == n) for n in names}
    on_serve = {n: sum(1 for p in points if p.get("server") == n and p.get("winner_name") == n) for n in names}
    st = json.loads((out / "stats.json").read_text()) if (out / "stats.json").exists() else {}
    from .match_stats import compute as match_compute
    ms = match_compute(points, st, posture)
    (out / "match_stats.json").write_text(json.dumps(ms, indent=1))
    by_games = any(g["finished"] for g in ms["games"])                  # the score is games won once a game has been completed
    big = {n: (ms["players"][n]["games"] if by_games else won[n]) for n in names}
    lead = max(names, key=lambda n: (big[n], won[n]))
    level = big[names[0]] == big[names[1]] and won[names[0]] == won[names[1]]
    label = {"good": "Good", "degraded": "Use with care", "unreliable": "Unreliable"}[quality["level"]]
    dur = max([p["end_t"] for p in points] or [0.0])
    def sub(n):
        if by_games:
            return f'game{"" if big[n] == 1 else "s"} · {won[n]} points · {on_serve[n]} of {served[n]} on serve'
        return f'{on_serve[n]} of {served[n]} on own serve'
    sides = "".join(
        f'<div class="side{" r" if i else ""}{" won" if n == lead and not level else ""}"><span class="name">{e(n)}</span>'
        f'<b class="num">{big[n]}</b><span class="sub">{sub(n)}</span></div>' for i, n in enumerate(names))
    stamp = (f'<div class="stamp {quality["level"]}" role="img" aria-label="Confidence {label}, {quality["score"]} of 100"><svg viewBox="0 0 100 100" aria-hidden="true">'
             f'<circle class="trk" cx="50" cy="50" r="46"/><circle class="val" cx="50" cy="50" r="46" pathLength="100" style="stroke-dasharray:{quality["score"]} 100"/></svg>'
             f'<span>Confidence</span><b>{label}</b><small>{quality["score"]} / 100</small></div>')
    # no confidence badge on the score line (the "Good 98/100" label went 2026-09-26); the verdict sentence stays
    reasons = "".join(f"<li>{e(r)}</li>" for r in quality.get("reasons", []))
    cal = (f'<p class="note">Table found automatically in {calibration.get("n_frames")} frames. Check <a href="table_check.png">the outline</a> sits on the table edges.</p>'
           if calibration else "")
    from .annotate import running_scores
    tally = running_scores(points)
    rows, cur = [], None
    games_by_n = {g["n"]: g for g in ms["games"]}
    for p in points:
        sc = tally[p["id"]]
        if sc["game"] != cur and len(ms["games"]) > 1:
            cur = sc["game"]; g = games_by_n.get(cur)
            if g and g["finished"] and g["winner"]:
                res = f'{e(g["winner"])} {max(g["score"].values())}–{min(list(g["score"].values()) + [0] if len(g["score"]) < 2 else g["score"].values())}'
            else:
                res = "in play when the recording stopped"
            rows.append(f'<tr class="game"><td colspan="8">Game {cur} <small>{res}</small></td></tr>')
        clip = clip_of.get(p["id"]); who = p.get("winner_name")
        capt = f"Point {p['id']} · {p.get('server') or p.get('serve_side')} serving"
        sub = f"{p.get('n_crossings', 0)} shots · {('won by ' + who) if who else 'winner unsure'}"
        btn = f'<button class="play" type="button" aria-label="Play point {p["id"]}"><span>Play</span></button>' if clip else ""
        rows.append(f'<tr{" class=unsure" if not who else ""}{(" data-clip=" + chr(34) + clip + chr(34)) if clip else ""} data-id="{p["id"]}" data-cap="{e(capt)}" data-sub="{e(sub)}">'
                    f'<td>{p["id"]}</td><td class="t hide-xs">{mmss(p["start_t"])}</td><td>{e(str(p.get("server") or p.get("serve_side")))}</td>'
                    f'<td class="n">{p.get("n_crossings", "")}</td><td class="hide-s">{ENDING.get(p.get("ending"), p.get("ending") or "")}</td>'
                    f'<td class="won">{e(who) if who else "unsure"}</td><td class="sc">{sc["after"].get(names[0], 0)}\u2013{sc["after"].get(names[1], 0)}</td><td>{btn}</td></tr>')
    if clip_of:
        first = clip_of.get(points[0]["id"]) or next(iter(clip_of.values()))
        stage = (f'<aside class="stage" aria-label="Clip player"><video id="v" controls playsinline preload="metadata" src="{first}#t=0.1"></video>'
                 '<div class="cap"><b id="cap">Choose a point</b><span id="sub">← → or J K step through points</span></div>'
                 '<ul class="legend">' + ('<li><i style="--k:oklch(74% 0.17 145)"></i>table it found</li>' if show_table else '')
                 + '<li><i class="f" style="--k:oklch(86% 0.17 95);background:linear-gradient(90deg,oklch(90% 0.08 200),oklch(88% 0.18 100),oklch(70% 0.19 50),oklch(62% 0.24 25),oklch(65% 0.25 340));border:0;width:2.6rem"></i>ball trail, slow to fast</li>'
                 '<li><i style="--k:oklch(74% 0.17 145);border-radius:50%;width:.7rem;height:.7rem"></i>bounce</li><li><i class="f" style="--k:oklch(70% 0.25 330)"></i>net crossing</li></ul>'
                 '<p class="note2">The scoreboard in the clips is tt_scout\u2019s own count, in games to 11. Points it is unsure of are not counted, so it can differ from the umpire\u2019s board.'
                 + (' A note along the bottom before the serve is a scout tip; it leaves when its fuse has burnt down, and its evidence is under Scout tips.' if on_clip else '') + '</p></aside>')
    else:
        stage = ('<aside class="stage"><div class="empty">No clips in this report. Give the recording to get one clip per point, with the tracking drawn on:<br>'
                 '<code>tt-scout report out/&lt;match&gt; --video match.mp4</code></div></aside>')
    ledger = (f'<div class="ledger"><table><caption class="sr">Every point: time, server, shots over the net, how it ended, who won</caption><thead><tr><th>#</th><th class="hide-xs">Time</th>'
              f'<th>Server</th><th class="n">Shots</th><th class="hide-s">Ended by</th><th>Won by</th><th title="{e(names[0])}\u2013{e(names[1])}, counted by tt_scout">Score</th><th><span class="sr">Clip</span></th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
              if points else '<div class="empty">No points were found. ' + e(" ".join(quality.get("reasons", []))) + "</div>")
    if not tips:
        tips_html = ""
    else:
        if tip_list:
            def see(t):
                ids = [i for i in t.shows_on][:12]
                nums = "".join(f'<button class="go" type="button" data-go="{i}" aria-label="Play point {i}">{i}</button>' if i in clip_of else f"<span>{i}</span>" for i in ids)
                return f'<p class="see">{"Watch it in" if clip_of else "Seen in"} point{"s" if len(ids) != 1 else ""} {nums}</p>'
            body = '<ol class="tiplist">' + "".join(
                f'<li class="{t.level}"><p class="for"><b>{e(t.player)}</b><small>{"Scout tip" if t.level == "tip" else "Early read"}</small></p>'
                f'<h3>{e(t.head)}</h3><p class="ev">{e(t.evidence)}'
                + (f'<small>Luck alone does this well in {max(1, round(100 * t.chance))} of 100 shuffled matches.</small>' if t.family == "outcome" else "")
                + f'</p>{see(t)}</li>' for t in tip_list) + "</ol>"
            body += ('<p class="note2">A tip is a pattern in tt_scout\u2019s own count of this recording. Many patterns are tried and only the strongest are shown, so each is '
                     'tested against luck: the winners are dealt out at random 400 times and the whole search is run again. A scout tip is a pattern that luck matches in at most '
                     '1 shuffled match in 10; an early read, in about 1 in 3. Treat both as leads to check in the clips, not verdicts.</p>')
        elif quality.get("level") == "unreliable":
            body = '<div class="empty">No tips. The tracking on this recording is unreliable, so advice built on it would be guesswork.</div>'
        else:
            body = (f'<div class="empty">No tips yet. In {len(points)} point{"s" if len(points) != 1 else ""} no pattern is clear enough to say out loud; '
                    'they start to appear from about 15 points.</div>')
        tips_html = (f'<section class="tips"><div class="head"><h2>Scout tips</h2><span>{len(tip_list)} pattern{"s" if len(tip_list) != 1 else ""} the points support</span></div>{body}</section>')
    place = []
    for pi, n in enumerate(names):
        sv, tb, rt = _placements(points, n)
        figs = "".join(f'<figure>{_svg_table(d, colour=f"var(--p{pi + 1})")}<figcaption><b>{t}</b><span>{len(d)} landed · {sum(1 for x in d if x[2])} in points won</span></figcaption></figure>'
                       for t, d in (("Serves", sv), ("Third ball", tb), ("Returns of serve", rt)))
        place.append(f'<div class="who"><h3>{e(n)}<small>plays left to right</small></h3>{figs}</div>')
    shots = st.get("match", {}).get("ending_shot_numbers", [p.get("n_crossings", 0) for p in points])
    hi = max(shots + [1]); cap_n = min(hi, 12)
    counts = [sum(1 for s_ in shots if (s_ == k if k < cap_n else s_ >= k)) for k in range(1, cap_n + 1)]
    top = max(counts + [1])
    rally = (f'<p class="kicker" style="padding-top:2rem">Shots over the net per point</p>'
             f'<div class="rally" style="--n:{cap_n}" role="img" aria-label="Rally lengths: shots over the net per point">'
             + "".join(f'<div style="height:{100 * c / top:.0f}%;--i:{i}" title="{c} points"></div>' for i, c in enumerate(counts))
             + f'</div><div class="rally-x" style="--n:{cap_n}">' + "".join(f"<span>{k}{'+' if k == cap_n and hi > cap_n else ''}</span>" for k in range(1, cap_n + 1)) + "</div>")
    rallies = (f'<section class="numbers"><div class="head"><h2>Rallies</h2><span>how long the points lasted</span></div>{_facts(ms, clip_of)}{rally}</section>'
               if points else "")
    hero_html = hero_block(hero, next((p for p in points if p["id"] == hero["point"]), None) if hero else None, names, clip_of,
                           (out / "full_match.mp4").exists(), dur) if hero else ""
    if hero and not hero_html:
        lg = next((p for p in points if p["id"] == hero["point"]), None)
        play = (f'<button class="go" type="button" data-go="{hero["point"]}" aria-label="Play point {hero["point"]}">play the rally</button>'
                if hero["point"] in clip_of else "")
        hero_html = (f'<figure class="hero"><img src="{hero["src"]}" width="{hero["w"]}" height="{hero["h"]}" alt="{e(names[0])} and {e(names[1])} mid-rally, '
                     f'with the skeletons and the ball\u2019s flight tt_scout measured drawn on" decoding="async"><figcaption><span><b>Point {hero["point"]}</b>'
                     + (f', the longest rally: {lg.get("n_crossings")} shots over the net, at {mmss(lg["start_t"])}' if lg else "")
                     + f'</span>{play}</figcaption></figure>')
    crit_html = ""
    if critique is not None:                                        # {name: [Critique, ...]} over the whole match, strongest first
        items = ""
        for n in names:
            for c in (critique.get(n) or [])[:6]:
                items += (f'<li><p class="for"><b>{e(n)}</b><small>Critique</small></p><h3>{e(c.head)}</h3>'
                          f'<p class="ev">{e(c.evidence[0].upper() + c.evidence[1:])}.<small>Fix: {e(c.fix)}.</small></p></li>')
            if not critique.get(n):
                items += (f'<li class="early"><p class="for"><b>{e(n)}</b><small>Critique</small></p><h3>Nothing stands out</h3>'
                          f'<p class="ev">No weakness is clear enough in these points to call.</p></li>')
        crit_html = (f'<section class="tips"><div class="head"><h2>Coach\u2019s critique</h2><span>what each player should work on, worst first, '
                     f'the same checks for both</span></div><ol class="tiplist">{items}</ol>'
                     '<p class="note2">Measured on every shot: speed off the racket and topspin from each flight fitted in 3D, height over the net, '
                     'how long after the top of the bounce the ball was hit, trunk lean and shoulder turn at contact from each player\u2019s body in 3D '
                     '(Apple Vision on this computer; the 2D skeleton for the lean where no 3D body was found), knee bend at contact as the camera sees it '
                     '(the picture angle on forehands seen side-on, see Posture at contact), where the shot landed. The pro figures '
                     'are the same measurements on three professional OpenTTGames matches. Rules, not a language model: '
                     'each note needs only a handful of shots (an 11-point game is enough), states its counts, and gives the usual coaching fix. '
                     'The notes cycle through the clips, after the scout tip and over every replay.</p></section>')
    whole = ""
    if (out / "full_match.mp4").exists():
        whole = (f'<section class="whole"><div class="head"><h2>The whole match</h2><span>every point with the tracking, the skeletons and both players\u2019 '
                 f'analysis on screen throughout</span></div><video controls playsinline preload="none"{" poster=" + chr(34) + "hero.jpg" + chr(34) if hero else ""} '
                 f'src="full_match.mp4"></video></section>')
    names_h2h = [n for n in names if n in ms["players"]]
    h2h = _h2h(ms, names) if len(names_h2h) == 2 and points else ""
    momentum = _momentum(ms, names, clip_of)
    stance_html = _stance(stance, posture, names, clip_of)
    today = datetime.date.today().strftime("%-d %B %Y")
    h1 = f"{e(names[0])} <i>v</i> {e(names[1])}"
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{e(title)}</title>{WEB_FONTS if web_fonts else ""}<style>{CSS}{THEME}{COMIC}</style></head><body><div class="wrap">'
            f'<header class="mast"><p class="kicker">Match report · tt_scout</p><h1>{h1}</h1>'
            f'<p class="meta">{e(title.split(" - ")[-1])} · {len(points)} points in {mmss(dur)} · {today}'
            + ("".join(f' · <a href="{e(h)}">{e(n)}\u2019s profile</a>' for n, h in (profiles or [])))
            + f'</p></header>{hero_html}'
            f'<div class="score {quality["level"]}">{sides}</div>{_sheet(ms, names)}'
            f'<div class="verdict"><p>{e(quality["advice"])}</p>{("<ul>" + reasons + "</ul>") if reasons else ""}{cal}</div>'
            # the videos come first, straight after the score (they used to sit under four sections of numbers);
            # #watch is the link the upload page's "Watch" goes to, whether or not this report has the whole-match video
            f'<span id="watch" class="anchor"></span>{whole}'
            f'<section><div class="head"><h2>Points</h2><span>{len(clip_of)} clips · tracking drawn on</span></div><div class="points">{stage}{ledger}</div></section>'
            f'{h2h}{crit_html}{_after3d(after3d, names)}{momentum}'
            f'{tips_html}<section><div class="head"><h2>Placement</h2><span>seen from above, player at the left end</span></div>{"".join(place)}'
            f'<div class="key"><span><i style="background:var(--p1)"></i><i style="background:var(--p2);margin-left:-.25rem"></i>filled: landed in a point that player won</span>'
            f'<span><i class="o"></i>hollow: in a point they lost</span></div></section>'
            f'{rallies}{stance_html}'
            f'<p class="colophon">Analysed on this computer; no video left it. Scored in games to 11, won by 2 (the rule since 2001), from tt_scout\u2019s own calls: '
            f'winners are inferred from bounces and net crossings, so a fast final shot the camera did not resolve can flip a call. Where it matters, play the clip.</p>'
            f'</div><script>{JS}</script></body></html>')
    page = _topbar(page, names, big)
    (out / "report.html").write_text(page)
    return out / "report.html", len(clip_of)
