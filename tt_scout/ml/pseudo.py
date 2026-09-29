"""Labels for the learned detector from a recording the classical pipeline has already analysed: no hand labelling.

    python -m tt_scout.ml.pseudo out/match --video match.mp4 --table match_table.json --pose match_pose.csv \
        --near Ann --far Bob --game phone_match

Writes data/ml/frames/<game>/ in the layout tt_scout.ml.data reads (frames at SIZE, labels.json, negatives.json), plus
sources.json = {frame: where its label came from} and full-size crops round a sample of the labels in data/ml/qa/<game>/, to be
looked at before anything is trained on them.

Where a label comes from:
  tracked  the classical track's position, inside a fitted 3D flight and within AGREE_PX of that flight projected into the picture
  smooth   the track's position between a fitted flight's bounce and the next contact, where it lies within SMOOTH_PX of a
           quadratic through its neighbours in time and moves at least MIN_STEP px a frame (a knee or a shoe does not)
Frames of a fitted flight where the track has nothing, or sits further than AGREE_PX from the flight, get no label and are not
used at all. Taking the fitted flight's own projection there was tried and looked at (34 crops of our 2026-09-26 recording):
most were tens of pixels off the ball, which was in plain view beside a racket or a leg, so the flight is the wrong one to trust
in exactly those frames.
Negatives, two kinds (negative_sources.json says which):
  quiet    frames further than DEAD_S from every point with no ball track for QUIET_F frames either side. Things still move in
           them (players walking, a game on the table behind): none of it is a ball the tracker could follow
  dead     where a hall is too busy to have enough of those (on our 2026-09-28 recordings something ball-like is tracked in 90% of
           all frames): frames further than DEAD2_S from every point, whatever is tracked in them. No ball is in play, but one can
           be in a hand, on the floor or on the table behind, so a peak on one is a fire in dead time, not always a false alarm
"""
import argparse, csv, json, pathlib
import numpy as np, cv2
from .data import SIZE, HISTORY, FRAMES, ROOT, to_size

AGREE_PX = 5.0
FIT_RMS = 3.0
GUARD_F = 2
SMOOTH_PX, MIN_STEP = 2.0, 3.0
DEAD_S, QUIET_F, DEAD2_S = 1.5, 5, 3.0
NEG_SHARE, NEG_MIN = 0.15, 300


def load_run(run_dir, video, table_path, pose_csv, near, far):
    """The analysis on disk, measured again exactly as `tt-scout report` measures it: track, points and the fitted flights."""
    from ..points_v1 import segment_points_v1
    from ..players import load_obs, name_points, auto_reference
    from ..table import Table
    from ..camera import Camera
    from ..heads import standing_zone
    from .. import pose as pose_mod, technique
    src = pathlib.Path(run_dir)
    events = []
    for r in csv.DictReader(open(src / "events.csv")):
        events.append(dict(frame=int(float(r["frame"])), t=float(r["t"]), kind=r["kind"], side=r["side"], x_px=float(r["x_px"]), y_px=float(r["y_px"]),
                           x_m=float(r["x_m"]), y_m=float(r["y_m"]), strength=float(r["strength"]), track=int(float(r["track"]))))
    track = np.genfromtxt(src / "track.csv", delimiter=",", skip_header=1)
    pts = segment_points_v1(events, None, track[:, :5])
    obs = load_obs(src / "players.csv"); names = {"near": near, "far": far}
    name_points(pts, obs, auto_reference(obs, names), names)
    fps = float(np.median(1.0 / np.diff(track[:200, 1])))
    tb = Table.load(table_path)
    cap = cv2.VideoCapture(str(video)); size = (int(cap.get(3)), int(cap.get(4))); cap.release()
    cam = Camera.from_table_pnp(tb, *size)
    assigned = None
    if pose_csv:
        assigned = pose_mod.assign(pose_mod.load(pose_csv), tb, standing_zone(tb, 1.0))
        assigned, _ = pose_mod.clean_legs(assigned, tb)
    _, fits = technique.measure(pts, events, track, fps, cam, assigned)
    return dict(track=track, events=events, points=pts, fps=fps, cam=cam, fits=fits, size=size)


def make_labels(run):
    """({frame: (x, y) in the video's pixels}, {frame: source}, {negative frame: kind})."""
    from .. import flight
    track, fps, cam, pts, fits = run["track"], run["fps"], run["cam"], run["points"], run["fits"]
    n = len(track); W, H = run["size"]
    sh = {(s["point"], s["shot"]): s for s in flight.shots(pts, run["events"], track, fps, None) + flight.net_strikes(pts, run["events"], track, fps)}
    lab, src = {}, {}
    for r in fits:                                                       # 1. inside the fitted flights
        s = sh.get((r["point"], r["shot"]))
        if not r.get("ok") or s is None or r.get("rms_px", 1e9) > FIT_RMS or len(s["frames"]) < 5:
            continue
        idx = np.asarray(s["frames"], int)
        fr = np.arange(idx.min(), idx.max() + 1)
        fr = fr[(track[fr, 1] < r["anchor"]["t"]) & (fr - r["t_contact"] * fps >= GUARD_F)]
        if not len(fr):
            continue
        P, _ = flight.state_at(r, track[fr, 1])
        for f, (u, v) in zip(fr, cam.project(P)):
            if not (0 <= u < W and 0 <= v < H):
                continue
            x, y = track[f, 2], track[f, 3]
            d = np.inf if np.isnan(x) else float(np.hypot(x - u, y - v))
            if d <= AGREE_PX:
                lab[int(f)] = (float(x), float(y)); src[int(f)] = "tracked"
    ends = {p["id"]: p["end_t"] for p in pts}
    by_point = {}
    for r in fits:
        if r.get("ok"):
            by_point.setdefault(r["point"], []).append(r)
    for pid, rs in by_point.items():                                     # 2. from each bounce on to the next contact
        rs.sort(key=lambda r: r["t_contact"])
        for i, r in enumerate(rs):
            if r.get("free") or r.get("bounce") is None or r.get("rms_px", 1e9) > FIT_RMS:
                continue
            nxt = rs[i + 1] if i + 1 < len(rs) and rs[i + 1]["shot"] == r["shot"] + 1 else None
            t0 = r["t_bounce"]; t1 = nxt["t_contact"] if nxt else min(t0 + 0.6, ends.get(pid, t0))
            f0, f1 = int(np.ceil(t0 * fps)) + 1, int(np.floor(t1 * fps)) - GUARD_F
            for f in range(max(f0, 0), min(f1, n - 1) + 1):
                if f in lab or np.isnan(track[f, 2]):
                    continue
                nb = [g for g in range(f - 3, f + 4) if g != f and f0 - 1 <= g <= f1 + GUARD_F and 0 <= g < n
                      and not np.isnan(track[g, 2]) and track[g, 4] == track[f, 4]]
                if len(nb) < 5:
                    continue
                t = track[nb, 1] - track[f, 1]; A = np.c_[t * t, t, np.ones(len(t))]
                cx = np.linalg.lstsq(A, track[nb, 2], rcond=None)[0]; cy = np.linalg.lstsq(A, track[nb, 3], rcond=None)[0]
                if np.hypot(cx[2] - track[f, 2], cy[2] - track[f, 3]) <= SMOOTH_PX and np.hypot(cx[1], cy[1]) / fps >= MIN_STEP:
                    lab[f] = (float(track[f, 2]), float(track[f, 3])); src[f] = "smooth"
    def away(gap_s):                                                     # 3. negatives
        m = np.ones(n, bool)
        for p in pts:
            m[max(0, int((p["start_t"] - gap_s) * fps)):int((p["end_t"] + gap_s) * fps) + 1] = False
        m[:HISTORY + QUIET_F] = False; m[n - QUIET_F:] = False
        return m
    seen = (~np.isnan(track[:, 2])).astype(int)
    near = np.convolve(seen, np.ones(2 * QUIET_F + 1, int), mode="same")    # tracked frames within QUIET_F of each frame
    want = max(NEG_MIN, int(NEG_SHARE * len(lab))); rng = np.random.default_rng(0)
    quiet = np.where(away(DEAD_S) & (near == 0))[0]
    negs = {int(f): "quiet" for f in (rng.choice(quiet, size=min(want, len(quiet)), replace=False) if len(quiet) else [])}
    dead = np.array([f for f in np.where(away(DEAD2_S))[0] if int(f) not in negs], int)
    if len(negs) < want and len(dead):
        for f in rng.choice(dead, size=min(want - len(negs), len(dead)), replace=False):
            negs[int(f)] = "dead"
    return lab, src, negs


def extract(video, lab, src, negs, out_dir, qa_dir, n_qa=48, crop=160):
    """One pass over the video: the frames the samples need at SIZE, and full-size crops round a sample of each source's labels."""
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    qa = pathlib.Path(qa_dir); qa.mkdir(parents=True, exist_ok=True)
    need = set()
    for f in list(lab) + list(negs):
        need.update(range(f - HISTORY, f + 1))
    need = {f for f in need if f >= 0}
    rng = np.random.default_rng(1); look = {}
    for tag in sorted(set(src.values())):
        fs = sorted(f for f, s in src.items() if s == tag)
        for f in rng.choice(fs, size=min(n_qa, len(fs)), replace=False):
            look[int(f)] = tag
    cap = cv2.VideoCapture(str(video)); W, H = int(cap.get(3)), int(cap.get(4))
    last = max(need) if need else -1; f = 0; written = 0
    while f <= last:
        if f in need:
            ok, frame = cap.read()
            if not ok:
                break
            p = out / f"{f}.jpg"
            if not p.exists():
                cv2.imwrite(str(p), cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95])
            written += 1
            if f in look:
                x, y = lab[f]; x0 = int(np.clip(x - crop / 2, 0, W - crop)); y0 = int(np.clip(y - crop / 2, 0, H - crop))
                cv2.imwrite(str(qa / f"{look[f]}_{f}.jpg"), frame[y0:y0 + crop, x0:x0 + crop], [cv2.IMWRITE_JPEG_QUALITY, 92])
        elif not cap.grab():
            break
        f += 1
    cap.release()
    have = {f for f in lab if all((out / f"{f - k}.jpg").exists() for k in range(HISTORY + 1))}
    (out / "labels.json").write_text(json.dumps({str(f): list(to_size(*lab[f], src=(W, H))) for f in sorted(have)}))
    (out / "sources.json").write_text(json.dumps({str(f): src[f] for f in sorted(have)}))
    kept = sorted(f for f in negs if all((out / f"{f - k}.jpg").exists() for k in range(HISTORY + 1)))
    (out / "negatives.json").write_text(json.dumps(kept))
    (out / "negative_sources.json").write_text(json.dumps({str(f): negs[f] for f in kept}))
    return written, len(have)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("--video", required=True); ap.add_argument("--table", required=True)
    ap.add_argument("--pose"); ap.add_argument("--near", default="near"); ap.add_argument("--far", default="far")
    ap.add_argument("--game", required=True, help="the folder under data/ml/frames/ to write")
    ap.add_argument("--dry", action="store_true", help="count the labels and stop: no frames are written")
    a = ap.parse_args()
    run = load_run(a.run_dir, a.video, a.table, a.pose, a.near, a.far)
    lab, src, negs = make_labels(run)
    n_ok = sum(1 for r in run["fits"] if r.get("ok")); n_tight = sum(1 for r in run["fits"] if r.get("ok") and r.get("rms_px", 1e9) <= FIT_RMS)
    by = {t: sum(1 for s in src.values() if s == t) for t in ("tracked", "smooth")}
    print(f"{a.game}: {len(run['points'])} points, {len(run['fits'])} shots, {n_ok} fitted, {n_tight} tight; labels {len(lab)} "
          f"({by['tracked']} tracked, {by['smooth']} smooth), negatives {len(negs)} "
          f"({sum(1 for v in negs.values() if v == 'quiet')} quiet, {sum(1 for v in negs.values() if v == 'dead')} dead)", flush=True)
    if a.dry:
        return
    w, m = extract(a.video, lab, src, negs, FRAMES / a.game, ROOT / "data" / "ml" / "qa" / a.game)
    print(f"{a.game}: wrote {w} frames for {m} labels -> {FRAMES / a.game}", flush=True)


if __name__ == "__main__":
    main()
