"""Analyse one match video.
    python -m tt_scout.run match.mp4 --table table.json --ball-colour orange [--show] [--save-video]
Writes out/<stem>/track.csv, events.csv, rallies.json, report.png (+ annotated.mp4)."""
import argparse, csv, json, pathlib, time
import numpy as np, cv2
from .config import NET_X, Config, ROOT
from .table import Table
from .detector import BallDetector
from .tracker import BallTracker
from .events import detect_events, segment_rallies, segment_points
from .scoring import who_won
from .report import make_report
from .players import write_obs, load_obs, name_points, end_of, auto_reference
from .points_v1 import segment_points_v1
from .quality import assess

EV_COL = {"bounce": (0, 220, 0), "hit": (0, 0, 255), "net": (255, 0, 255), "nethit": (255, 0, 255)}


def analyse(video, table_path, cfg, show=False, save_video=False, max_frames=None, out_root=None, logic="v0"):
    if not pathlib.Path(table_path).exists():
        raise SystemExit(f"no table calibration at {table_path}: run  python calibrate_table.py <video or camera index> --view side  first")
    table = Table.load(table_path)
    live = isinstance(video, int)                      # a camera index (Continuity Camera etc.) instead of a file
    stem = f"live_{time.strftime('%Y%m%d_%H%M%S')}" if live else pathlib.Path(video).stem
    t0 = time.time()
    rows, prow, fps, i = track_video(video, table, cfg, show=show, max_frames=max_frames)
    track = np.array(rows, float)
    events = detect_events(track[:, :5], table, cfg, fps)
    if logic == "v1":                                   # repaired event sequence (points_v1.py); v0 stays the frozen baseline
        rallies = segment_points_v1(events, None, track[:, :5])
    else:
        rallies = segment_points(events, cfg, track[:, :5]) or segment_rallies(track[:, :5], events, cfg, fps)
        for r in rallies:
            r["winner"] = who_won(r)
    out = pathlib.Path(out_root or ROOT / "out") / stem
    out.mkdir(parents=True, exist_ok=True)
    write_obs(out / "players.csv", prow)
    obs = load_obs(out / "players.csv")
    names = {"near": cfg.near_name, "far": cfg.far_name}
    name_points(rallies, obs, table.meta.get("players") or auto_reference(obs, names), names)
    quality = assess(rallies, track, fps)
    (out / "quality.json").write_text(json.dumps(quality, indent=1))
    with open(out / "track.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "t", "x", "y", "track", "n_cands"]); w.writerows(rows)
    with open(out / "events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(events[0].keys()) if events else ["t"]); w.writeheader(); w.writerows(events)
    (out / "rallies.json").write_text(json.dumps(rallies, indent=1))
    from .stats import write_stats
    write_stats(out, rallies)
    make_report(rallies, events, out / "report.png", title=stem)
    if save_video and not live:
        write_annotated(video, track, events, out / "annotated.mp4", table, fps)
    seen = float(np.mean(~np.isnan(track[:, 2]))) if len(track) else 0.0
    return dict(out=str(out), fps=fps, n_frames=i, events=events, rallies=rallies, ball_seen_frac=seen,
                seconds=time.time() - t0, quality=quality)


def track_video(video, table, cfg, show=False, max_frames=None):
    """The ball in every frame of a video file (or a camera index): rows [frame, t, x, y, track id, candidates] with x, y NaN where
    it was not seen, and the player-sized blobs. Returns (rows, player rows, fps, frames read)."""
    live = isinstance(video, int)                      # a camera index (Continuity Camera etc.) instead of a file
    cap = cv2.VideoCapture(video if live else str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}" + (" (check System Settings > Privacy & Security > Camera)" if live else ""))
    if live:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if live and fps <= 1:
        fps = 30.0
    det, trk = BallDetector(cfg, table, fps), BallTracker(cfg, fps)
    trk.area_ref = det.area_ref                                     # the tracker waits only for a ball the size of this table's,
    trk.net_line = [tuple(map(float, q)) for q in table.to_px([[NET_X, 0.0], [NET_X, 1.0]])]   # and only near the net
    rows, prow = [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and i >= max_frames):
            break
        cands = det.detect(frame)
        c = trk.update(cands)
        rows.append([i, i / fps, c.x if c else np.nan, c.y if c else np.nan, trk.track_id if c else -1, len(cands)])
        for pl in det.players:
            prow.append([i, i / fps, end_of(table, pl["cx"], pl["cy"]), round(pl["cx"]), round(pl["cy"]), int(pl["area"]),
                         round(pl["h"], 1), round(pl["s"], 1), round(pl["v"], 1)])
        for ago, bx, by in trk.backfill:                    # the frames that led to a track start are ball too
            if ago < len(rows):
                rows[-1 - ago][2:5] = [bx, by, trk.track_id]
        if show:
            cv2.polylines(frame, [table.quad_int()], True, (255, 255, 0), 1)
            for cd in cands:
                cv2.circle(frame, (int(cd.x), int(cd.y)), 6, (0, 255, 255), 1)
            if c:
                cv2.circle(frame, (int(c.x), int(c.y)), 10, (0, 0, 255), 2)
            cv2.imshow("tt_scout (q quits)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        i += 1
    cap.release(); cv2.destroyAllWindows()
    return rows, prow, fps, i


def write_annotated(video, track, events, out_path, table, fps):
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(3)), int(cap.get(4))
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    by_frame = {}
    for e in events:
        by_frame.setdefault(e["frame"], []).append(e)
    recent, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok or i >= len(track):
            break
        cv2.polylines(frame, [table.quad_int()], True, (255, 255, 0), 1)
        trail = track[max(0, i - 15):i + 1]
        pts = trail[~np.isnan(trail[:, 2])][:, 2:4].astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(frame, [pts.reshape(-1, 1, 2)], False, (0, 200, 255), 2)
        if not np.isnan(track[i, 2]):
            cv2.circle(frame, (int(track[i, 2]), int(track[i, 3])), 8, (0, 0, 255), 2)
        recent = [(e, n) for e, n in recent if n < int(fps * 0.6)] + [(e, 0) for e in by_frame.get(i, [])]
        for e, n in recent:
            cv2.circle(frame, (int(e["x_px"]), int(e["y_px"])), 14, EV_COL[e["kind"]], 2)
            cv2.putText(frame, e["kind"], (int(e["x_px"]) + 16, int(e["y_px"])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, EV_COL[e["kind"]], 2)
        recent = [(e, n + 1) for e, n in recent]
        vw.write(frame); i += 1
    cap.release(); vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="video file, or a camera index like 1 for a live Continuity Camera feed (q quits)")
    ap.add_argument("--table", default=str(ROOT / "table.json"))
    ap.add_argument("--ball-colour", default="any", choices=["any", "orange", "white"])
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--near", default="near", help="name of the player at the x=0 end (calibration corners 1-2)")
    ap.add_argument("--far", default="far")
    a = ap.parse_args()
    cfg = Config(ball_colour=a.ball_colour, near_name=a.near, far_name=a.far)
    src = int(a.video) if a.video.isdigit() else a.video
    r = analyse(src, a.table, cfg, show=a.show or isinstance(src, int), save_video=a.save_video, max_frames=a.max_frames)
    print(f"{r['n_frames']} frames in {r['seconds']:.0f}s, ball seen in {r['ball_seen_frac']*100:.0f}% of frames, "
          f"{len(r['rallies'])} rallies, {sum(1 for e in r['events'] if e['kind']=='bounce')} bounces -> {r['out']}")


if __name__ == "__main__":
    main()
