"""Recompute events and points from a SAVED tracking run (track.csv, players.csv) with Config overrides, into a new run
folder that `tt-scout report` can build a report from. No tracking pass, seconds per match.
    python analysis/reevents.py out_own/IMG_3143 --table data/own/IMG_3143_table.json --out out_own/IMG_3143_onetable \
        --set cross_needs_table_s=0.6 --near Black --far Grey
"""
import argparse, csv, json, pathlib, shutil, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from tt_scout.config import Config
from tt_scout.table import Table
from tt_scout.events import detect_events
from tt_scout.points_v1 import segment_points_v1
from tt_scout.players import load_obs, name_points, auto_reference
from tt_scout.quality import assess
from tt_scout.stats import write_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--table", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[], help="Config field=value, repeatable")
    ap.add_argument("--near", default="near"); ap.add_argument("--far", default="far")
    a = ap.parse_args()
    cfg = Config(near_name=a.near, far_name=a.far)
    for kv in a.set:
        k, v = kv.split("=")
        setattr(cfg, k, type(getattr(cfg, k))(v))
    src, dst = pathlib.Path(a.run), pathlib.Path(a.out)
    dst.mkdir(parents=True, exist_ok=True)
    track = np.genfromtxt(src / "track.csv", delimiter=",", skip_header=1)
    fps = float(np.median(1.0 / np.diff(track[:200, 1])))
    table = Table.load(a.table)
    events = detect_events(track[:, :5], table, cfg, fps)
    pts = segment_points_v1(events, None, track[:, :5])
    shutil.copy(src / "track.csv", dst / "track.csv"); shutil.copy(src / "players.csv", dst / "players.csv")
    shutil.copy(a.table, dst / "table.json")
    obs = load_obs(dst / "players.csv"); names = {"near": a.near, "far": a.far}
    name_points(pts, obs, table.meta.get("players") or auto_reference(obs, names), names)
    with open(dst / "events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(events[0].keys())); w.writeheader(); w.writerows(events)
    (dst / "rallies.json").write_text(json.dumps(pts, indent=1))
    q = assess(pts, track, fps); (dst / "quality.json").write_text(json.dumps(q, indent=1))
    write_stats(dst, pts)
    (dst / "settings.json").write_text(json.dumps({"from_run": str(src), "overrides": a.set}, indent=1))
    n = {k: sum(e["kind"] == k for e in events) for k in ("net", "bounce", "hit")}
    print(f"{dst}: {len(pts)} points, events {n}, confidence {q['level']} {q['score']}")


if __name__ == "__main__":
    main()
