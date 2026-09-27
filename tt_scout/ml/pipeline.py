"""The whole analysis with the learned detector in place of the classical one. Everything after the detector is unchanged:
the same tracker, event detection, point logic (v1), player naming, confidence and stats, so any difference in the results
comes from the detector alone (contribution 3 of RESEARCH.md).

    python -m tt_scout.ml.pipeline test_4 --near Black --far Red       # -> out_ml/test_4/ (then: tt-scout report out_ml/test_4 ...)
Player observations (shirt colours by end) come from the classical run in out/<clip>/players.csv: the learned model finds the
ball only.
"""
import argparse, csv, json, pathlib, shutil, time
import numpy as np
from ..config import Config, ROOT
from ..table import Table
from ..detector import Candidate
from ..tracker import BallTracker
from ..events import detect_events
from ..points_v1 import segment_points_v1
from ..players import load_obs, name_points, auto_reference
from ..quality import assess
from ..stats import write_stats
from ..report import make_report
from . import infer


def track_from_candidates(cands, fps, cfg):
    trk = BallTracker(cfg, fps); rows = []
    for i, cs in enumerate(cands):
        cc = [Candidate(x, y, 30.0, s) for x, y, s in cs]
        c = trk.update(cc)
        rows.append([i, i / fps, c.x if c else np.nan, c.y if c else np.nan, trk.track_id if c else -1, len(cc)])
        for ago, bx, by in trk.backfill:
            if ago < len(rows):
                rows[-1 - ago][2:5] = [bx, by, trk.track_id]
    return rows


def run(clip, near, far, ckpt=ROOT / "models" / "ballnet.pt", out_root=ROOT / "out_ml", reuse=True, video=None, table=None, classical_dir=None, thresh=None):
    """clip names the run; video defaults to data/<clip>.mp4, the table calibration to data/<clip>_table.json or the one the
    classical run saved in out/<clip>/table.json, and the player observations to out/<clip>/players.csv."""
    video = pathlib.Path(video) if video else ROOT / "data" / f"{clip}.mp4"
    src = pathlib.Path(classical_dir) if classical_dir else ROOT / "out" / clip
    table_path = pathlib.Path(table) if table else next((t for t in (ROOT / "data" / f"{clip}_table.json", src / "table.json") if t.exists()), None)
    if table_path is None:
        raise SystemExit(f"no table calibration for {clip}: run  tt-scout analyse {video}  first (it writes out/<stem>/table.json)")
    out = pathlib.Path(out_root) / clip; out.mkdir(parents=True, exist_ok=True)
    cpath = out / "candidates.npz"; t0 = time.time()
    if reuse and cpath.exists():
        fps, cands, th = infer.load_saved(cpath)
    else:
        fps, cands, th = infer.candidates(video, ckpt, thresh=thresh)
        infer.save(cpath, fps, cands, th)
    t_detect = time.time() - t0
    cfg = Config(ball_colour="white", near_name=near, far_name=far)       # the settings the classical test runs used
    table = Table.load(table_path)
    rows = track_from_candidates(cands, fps, cfg); track = np.array(rows, float)
    events = detect_events(track[:, :5], table, cfg, fps)
    rallies = segment_points_v1(events, None, track[:, :5])
    shutil.copy(src / "players.csv", out / "players.csv"); shutil.copy(table_path, out / "table.json")
    obs = load_obs(out / "players.csv"); names = {"near": near, "far": far}
    name_points(rallies, obs, table.meta.get("players") or auto_reference(obs, names), names)
    quality = assess(rallies, track, fps)
    with open(out / "track.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "t", "x", "y", "track", "n_cands"]); w.writerows(rows)
    with open(out / "events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(events[0].keys()) if events else ["t"]); w.writeheader(); w.writerows(events)
    (out / "rallies.json").write_text(json.dumps(rallies, indent=1)); (out / "quality.json").write_text(json.dumps(quality, indent=1))
    write_stats(out, rallies); make_report(rallies, events, out / "report.png", title=f"{clip} (learned detector)")
    seen = float(np.mean(~np.isnan(track[:, 2])))
    return dict(clip=clip, frames=len(rows), points=len(rallies), ball_seen=round(seen, 3), quality=quality["level"], detect_seconds=round(t_detect))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("clip"); ap.add_argument("--near", default="near"); ap.add_argument("--far", default="far")
    ap.add_argument("--ckpt", default=str(ROOT / "models" / "ballnet.pt")); ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--video"); ap.add_argument("--table"); ap.add_argument("--classical-dir"); ap.add_argument("--thresh", type=float)
    a = ap.parse_args()
    print(run(a.clip, a.near, a.far, a.ckpt, reuse=not a.fresh, video=a.video, table=a.table, classical_dir=a.classical_dir, thresh=a.thresh))
