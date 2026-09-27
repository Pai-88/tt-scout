"""Classical versus learned detector on labelled clips. Nothing here is tuned on these numbers.

    python -m tt_scout.ml.evaluate test_4 [test_1 ...] --json results/ml_vs_classical.json

1. Ball positions against the OpenTTGames labels, 1920x1080 pixels: share of labelled visible frames where the final track is
   within 10 px, within 40 px, median error of those within 40 px, and how often a track position is reported on a frame
   labelled "ball not visible". Also for LAST-SHOT frames only (the 0.6 s before each labelled point ends), because that is
   where missed shots decide who won a point.
2. Points with the unchanged v1 logic on top of each track, scored by tt_scout.metrics (point F1, server, winner).
Classical: out/<clip>/track.csv and out_v1/<clip>/rallies.json. Learned: out_ml/<clip>/.
"""
import argparse, csv, json, pathlib, subprocess, sys
import numpy as np
from ..config import ROOT


def ball_scores(track, labels, frames=None):
    vis = [(int(f), v) for f, v in labels.items() if v["x"] >= 0 and (frames is None or int(f) in frames)]
    hid = [int(f) for f, v in labels.items() if v["x"] < 0 and (frames is None or int(f) in frames)]
    d = []
    for f, v in vis:
        if f < len(track) and not np.isnan(track[f, 2]):
            d.append(float(np.hypot(track[f, 2] - v["x"], track[f, 3] - v["y"])))
        else:
            d.append(np.inf)
    d = np.array(d)
    fp = sum(1 for f in hid if f < len(track) and not np.isnan(track[f, 2]))
    ok = d[d <= 40]
    return dict(n=len(vis), within_10px=round(float(np.mean(d <= 10)), 3) if len(d) else None, within_40px=round(float(np.mean(d <= 40)), 3) if len(d) else None,
                median_err_px=round(float(np.median(ok)), 2) if len(ok) else None, hidden_frames=len(hid), reported_on_hidden=fp)


def last_shot_frames(clip, fps):
    out = set()
    for r in csv.DictReader(open(ROOT / "labels" / f"{clip}.points.csv")):
        end = int(float(r["end_frame"]))
        out.update(range(end - int(0.6 * fps), end + 1))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("clips", nargs="+"); ap.add_argument("--json")
    a = ap.parse_args()
    res = {"ball": {}, "points": {}}
    for clip in a.clips:
        labels = json.loads((ROOT / "data" / f"{clip}_markup" / "ball_markup.json").read_text())
        fps = 120.0
        last = last_shot_frames(clip, fps)
        row = {}
        for name, path in (("classical", ROOT / "out" / clip / "track.csv"), ("learned", ROOT / "out_ml" / clip / "track.csv")):
            tr = np.genfromtxt(path, delimiter=",", skip_header=1)
            row[name] = dict(all=ball_scores(tr, labels), last_shot=ball_scores(tr, labels, last))
        res["ball"][clip] = row
        for part in ("all", "last_shot"):
            c, l = row["classical"][part], row["learned"][part]
            print(f"{clip} {part:9s} n={c['n']:5d}  within 10 px: classical {c['within_10px']}  learned {l['within_10px']}   "
                  f"within 40 px: {c['within_40px']} vs {l['within_40px']}   median err {c['median_err_px']} vs {l['median_err_px']} px   "
                  f"on hidden frames {c['reported_on_hidden']}/{c['hidden_frames']} vs {l['reported_on_hidden']}/{l['hidden_frames']}")
    for name, root in (("classical", "out_v1"), ("learned", "out_ml")):
        j = ROOT / "results" / f"_points_{name}.json"; j.parent.mkdir(exist_ok=True)
        print(f"\n--- points, {name} detector, v1 logic ---", flush=True)
        subprocess.run([sys.executable, "-m", "tt_scout.metrics", "--out-root", root, "--json", str(j), "--clips"] + [f"data/{c}" for c in a.clips],
                       cwd=ROOT, check=True)
        res["points"][name] = json.loads(j.read_text()); j.unlink()
    if a.json:
        pathlib.Path(a.json).parent.mkdir(parents=True, exist_ok=True); pathlib.Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
