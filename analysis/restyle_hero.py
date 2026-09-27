"""Swap the still picture at the top of already-built reports for the playable one (report_html.hero_block), without rebuilding them.
so that someone opening a report for the first time sees at once that it has videos. A report built from now on has
it anyway; this is for the ones already on disk. Skips a report that already has it.
    python analysis/restyle_hero.py out_own_product/*/report.html"""
import pathlib, re, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from tt_scout import report_html as R

CSS_START = "/* the top of the report plays"
for f in map(pathlib.Path, sys.argv[1:]):
    s = f.read_text(); d = f.parent
    if 'class="hero-v"' in s and "--restyle" not in sys.argv:
        print(f"{f}: already has it"); continue
    m = re.search(r'<figure class="hero">.*?</figure>', s, re.S)
    if not m:
        print(f"{f}: no picture at the top"); continue
    fig = m.group(0)
    pid = re.search(r'<b>Point (\d+)</b>', fig); cap = re.search(r'</b>(.*?)</span>', fig, re.S); src = re.search(r'<img src="([^"]+)"', fig)
    if not (pid and src):
        print(f"{f}: could not read the picture"); continue
    clips = {int(c.stem.split("_")[1]): f"clips/{c.name}" for c in sorted((d / "clips").glob("point_*.mp4"))}
    dur_m = re.search(r'points in ([\d:.]+)', s)
    dur = sum(float(x) * 60 ** i for i, x in enumerate(reversed(dur_m.group(1).split(":")))) if dur_m else 0.0
    point = int(pid.group(1))
    wh = re.search(r'<img src="[^"]+" width="(\d+)" height="(\d+)"', fig)
    hero = dict(point=point, src=src.group(1), **(dict(w=int(wh.group(1)), h=int(wh.group(2))) if wh else {}))
    block = R.hero_block(hero, None, None, clips, (d / "full_match.mp4").exists(), dur)
    if not block:
        print(f"{f}: no clip for point {point}"); continue
    block = block.replace(f"<b>Point {point}</b></span>", f"<b>Point {point}</b>{cap.group(1) if cap else ''}</span>")
    css = R.CSS[R.CSS.index(CSS_START):R.CSS.index(".watchbar a.first")]
    css += R.CSS[R.CSS.index(".watchbar a.first"):].split("\n", 1)[0]
    js = R.JS[:R.JS.index("})();") + len("})();")] + "\n"          # the first block of the report's script: the one that plays the top video
    s = s.replace(fig, block).replace("</head>", f"<style>{css}</style></head>", 1).replace("</body>", f"<script>{js}</script></body>", 1)
    f.write_text(s); print(f"{f}: top picture now plays (point {point}, whole match {'yes' if (d / 'full_match.mp4').exists() else 'not yet'})")
