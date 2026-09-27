"""Hyper-parameter grid for the v1 point logic on the DEVELOPMENT games only (no detection; saved events).
Usage: python -m analysis.grid_v1 game_1 game_2 ..."""
import csv, sys, itertools, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from tt_scout.metrics import load_points_csv, match_points, prf
from tt_scout.points_v1 import segment_points_v1, V1Config
clips = sys.argv[1:]
data = {}
for clip in clips:
    ev = [dict(t=float(r["t"]), kind=r["kind"], side=r["side"], x_m=float(r["x_m"]), y_m=float(r["y_m"])) for r in csv.DictReader(open(ROOT / f"out/{clip}/events.csv"))]
    tr = np.genfromtxt(ROOT / f"out/{clip}/track.csv", delimiter=",", skip_header=1)[:, :5]
    data[clip] = (ev, tr, load_points_csv(ROOT / f"labels/{clip}.points.csv"))
rows = []
for rt, lt, nm, dt in itertools.product((1.1, 1.3, 1.5), (1.2, 1.5), (0.10, 0.20), (2.5, 4.0)):
    cfg = V1Config(return_timeout_s=rt, long_timeout_s=lt, net_margin_m=nm, dead_time_s=dt)
    codes, win, srv = [], [], []
    for clip, (ev, tr, truth) in data.items():
        pts = segment_points_v1(ev, cfg, tr)
        pred = [dict(id=p["id"], start_t=p["start_t"], end_t=p["end_t"], server=p["serve_side"], winner=p["winner"], ending=p["ending"]) for p in pts]
        pairs, fp, fn, _ = match_points(truth, pred)
        codes += [0] * len(pairs) + [1] * len(fp) + [2] * len(fn)
        win += [pp["winner"] == t["winner"] for t, pp in pairs if t["winner"]]; srv += [pp["server"] == t["server"] for t, pp in pairs if t["server"]]
    p, r, f1 = prf(codes)
    rows.append((f1, np.mean(win), rt, lt, nm, dt, p, r, sum(win), len(win), np.mean(srv)))
rows.sort(key=lambda x: -(x[0] + x[1]))
print("return_to long_to net_m dead |   P    R   F1 | winner        | server")
for f1, w, rt, lt, nm, dt, p, r, wn, wd, s in rows[:10]:
    print(f"   {rt:.1f}     {lt:.1f}   {nm:.2f}  {dt:.1f} | {p:.2f} {r:.2f} {f1:.2f} | {wn}/{wd} = {w:.3f} | {s:.3f}")
