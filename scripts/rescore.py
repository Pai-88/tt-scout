"""Recompute points from SAVED events (no detection pass) with a given point-logic version, write out_<version>/<clip>/rallies.json,
and score against labels/<clip>.points.csv.   Usage: python scripts/rescore.py v1 game_1 game_2 ... [--json results/x.json]"""
import csv, json, pathlib, subprocess, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from tt_scout.config import Config
from tt_scout.events import segment_points
from tt_scout.scoring import who_won
from tt_scout.points_v1 import segment_points_v1

argv = sys.argv[1:]
out_json = None
if "--json" in argv:
    i = argv.index("--json"); out_json = argv[i + 1]; del argv[i:i + 2]
version, clips = argv[0], argv[1:]
for clip in clips:
    ev = []
    for r in csv.DictReader(open(ROOT / f"out/{clip}/events.csv")):
        ev.append(dict(t=float(r["t"]), kind=r["kind"], side=r["side"], x_m=float(r["x_m"]), y_m=float(r["y_m"]),
                       frame=int(float(r["frame"])), strength=float(r["strength"])))
    tr = np.genfromtxt(ROOT / f"out/{clip}/track.csv", delimiter=",", skip_header=1)[:, :5]
    if version == "v0":
        pts = segment_points(ev, Config(), tr)
        for p in pts: p["winner"] = who_won(p)
    else:
        pts = segment_points_v1(ev, None, tr)
    d = ROOT / f"out_{version}" / clip; d.mkdir(parents=True, exist_ok=True)
    (d / "rallies.json").write_text(json.dumps(pts, indent=1))
    for f in ("events.csv",):
        (d / f).write_bytes((ROOT / f"out/{clip}/{f}").read_bytes())
cmd = [sys.executable, "-m", "tt_scout.metrics", "--out-root", f"out_{version}", "--clips"] + [f"data/{c}" for c in clips]
if out_json: cmd += ["--json", out_json]
subprocess.run(cmd, cwd=ROOT, check=True)
