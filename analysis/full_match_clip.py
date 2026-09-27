"""The whole recording as one annotated video, instead of one clip per point. Product layer.

    python3 analysis/full_match_clip.py --run out/yt_bundesliga --points out_product_yt_classical/yt_bundesliga/rallies.json \
        --video data/yt_bundesliga.mp4 --table data/yt_bundesliga_table.json --out out_product_yt_classical/yt_bundesliga/full_match.mp4

Reads an analysis already on disk (no tracking) and plays the recording through with the tracking drawn on: ball trail, bounce rings,
speed gauge, landing map, name balloons, the running scoreboard and a lettered burst as each point is won.
"""
import argparse, csv, json, pathlib, sys, time
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tt_scout.annotate import render_match_clip, overlay_context
from tt_scout.players import load_obs
from tt_scout.table import Table


def read_events(path):
    out = []
    for r in csv.DictReader(open(path)):
        out.append(dict(frame=int(float(r["frame"])), t=float(r["t"]), kind=r["kind"], side=r["side"], x_px=float(r["x_px"]),
                        y_px=float(r["y_px"]), x_m=float(r["x_m"]), y_m=float(r["y_m"]), strength=float(r["strength"]),
                        track=int(float(r["track"]))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="folder with track.csv, events.csv, players.csv")
    ap.add_argument("--points", help="rallies.json with the named points (default: <run>/rallies.json)")
    ap.add_argument("--video", required=True); ap.add_argument("--table", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--t0", type=float); ap.add_argument("--t1", type=float)
    ap.add_argument("--no-scoreboard", action="store_true"); ap.add_argument("--no-comic", action="store_true")
    ap.add_argument("--show-table", action="store_true")
    a = ap.parse_args()
    run = pathlib.Path(a.run)
    track = np.genfromtxt(run / "track.csv", delimiter=",", skip_header=1)
    events = read_events(run / "events.csv")
    points = json.loads(pathlib.Path(a.points or run / "rallies.json").read_text())
    obs = load_obs(run / "players.csv")
    by_frame = {}
    for o in obs:
        by_frame.setdefault(o["frame"], []).append(o)
    fps = float(np.median(1.0 / np.diff(track[:200, 1])))
    table = Table.load(a.table)
    print(f"{len(track)} frames at {fps:.2f} fps, {len(points)} points; building the hall ...", flush=True)
    ctx = overlay_context(a.video, table)
    t = time.time()
    ok = render_match_clip(a.video, fps, table, track[:, :5], events, by_frame, points, a.out, context=ctx,
                           scoreboard=not a.no_scoreboard, comic=not a.no_comic, t0=a.t0, t1=a.t1, show_table=a.show_table)
    size = pathlib.Path(a.out).stat().st_size / 1e6 if ok else 0
    print(f"{'wrote' if ok else 'FAILED'} {a.out} ({size:.0f} MB) in {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
