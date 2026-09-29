"""The learned detector on a phone recording it was not trained on, against the classical one. Nothing is tuned on these numbers.

    python analysis/phone_ml_eval.py out/match --video match.mp4 --table match_table.json --pose match_pose.csv \
        --near Ann --far Bob --game phone_match --ckpt models/ballnet_phone.pt --handcheck match_handcheck.csv --out out_ml_phone

1. The detector against the labels of tt_scout.ml.pseudo (the classical track where physics confirms it): the share of labelled
   frames with the top peak within NEAR_PX, and with any of the three peaks within NEAR_PX. The classical detector scores 100%
   here by construction, so this says how much of what the classical one finds the network finds too, and nothing more.
2. The frames inside the points, by who has the ball: both (within APART_PX of each other), classical only, learned only, both
   but apart, neither. Crops of a sample of "learned only" and "apart" frames are written to be looked at: no number here can
   say who is right.
3. The whole analysis three times over one pass of the video: the tracker fed by the classical detector, by the learned one, and
   by both (tt_scout.ml.fuse), then the same events, points (v1) and fitted flights on each track. Per detector: points, frames
   with the ball inside the classical points, shots, how many of them got a fitted flight and a speed, and the hand check.
Needs PyTorch for step 1's peaks (cached in <out>/<game>/candidates.npz).
"""
import argparse, csv, json, pathlib, re, sys, time
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from tt_scout.config import Config, NET_X
from tt_scout.table import Table
from tt_scout.camera import Camera
from tt_scout.detector import BallDetector
from tt_scout.tracker import BallTracker
from tt_scout.events import detect_events
from tt_scout.points_v1 import segment_points_v1
from tt_scout.players import write_obs, load_obs, name_points, end_of, auto_reference
from tt_scout.heads import standing_zone
from tt_scout import pose as pose_mod, technique
from tt_scout.ml.fuse import fuse, as_candidates
from tt_scout.ml.data import FRAMES, to_src

NEAR_PX, APART_PX = 10.0, 15.0
MODES = ("classical", "learned", "both")


def peaks_of(video, ckpt, cache, max_frames=None):
    from tt_scout.ml import infer
    if cache.exists():
        return infer.load_saved(cache)
    fps, cands, th = infer.candidates(video, ckpt, spacing="auto", log_every=12000, max_frames=max_frames)
    infer.save(cache, fps, cands, th)
    return fps, cands, th


def against_labels(game, cands):
    lab = json.loads((FRAMES / game / "labels.json").read_text()); src = json.loads((FRAMES / game / "sources.json").read_text())
    res = {}
    for tag in ("tracked", "smooth", "all"):
        fs = [f for f in lab if (tag == "all" or src[f] == tag) and int(f) < len(cands)]
        top = any3 = 0
        for f in fs:
            x, y = to_src(*lab[f]); cs = cands[int(f)] if int(f) < len(cands) else []
            d = [float(np.hypot(c[0] - x, c[1] - y)) for c in cs]
            top += bool(d and d[0] <= NEAR_PX); any3 += bool(d and min(d) <= NEAR_PX)
        res[tag] = dict(n=len(fs), top_within=round(top / max(1, len(fs)), 4), any_within=round(any3 / max(1, len(fs)), 4))
    return res


def three_tracks(video, table, cfg, cands, max_frames=None):
    """One pass: the blob detector once, three trackers. Rows as tt_scout.run.track_video writes them."""
    cap = cv2.VideoCapture(str(video)); fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    det = BallDetector(cfg, table, fps); trk = {m: BallTracker(cfg, fps) for m in MODES}
    for t in trk.values():
        t.area_ref = det.area_ref
        t.net_line = [tuple(map(float, q)) for q in table.to_px([[NET_X, 0.0], [NET_X, 1.0]])]
    rows = {m: [] for m in MODES}; prow = []; i = 0; t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and i >= max_frames):
            break
        cl = det.detect(frame); pk = cands[i] if i < len(cands) else []
        per = dict(classical=cl, learned=as_candidates(pk, det.area_ref), both=fuse(cl, pk, det.area_ref))
        for m in MODES:
            c = trk[m].update(per[m])
            rows[m].append([i, i / fps, c.x if c else np.nan, c.y if c else np.nan, trk[m].track_id if c else -1, len(per[m])])
            for ago, bx, by in trk[m].backfill:
                if ago < len(rows[m]):
                    rows[m][-1 - ago][2:5] = [bx, by, trk[m].track_id]
        for pl in det.players:
            prow.append([i, i / fps, end_of(table, pl["cx"], pl["cy"]), round(pl["cx"]), round(pl["cy"]), int(pl["area"]),
                         round(pl["h"], 1), round(pl["s"], 1), round(pl["v"], 1)])
        i += 1
        if i % 12000 == 0:
            print(f"  tracking frame {i}: {i / (time.time() - t0):.0f} frames/s", flush=True)
    cap.release()
    return {m: np.array(r, float) for m, r in rows.items()}, prow, fps


def points_of(track, table, cfg, fps, obs, names):
    events = detect_events(track[:, :5], table, cfg, fps)
    pts = segment_points_v1(events, None, track[:, :5])
    name_points(pts, obs, table.meta.get("players") or auto_reference(obs, names), names)
    return events, pts


def hand_check(path, pts):
    """Each hand-checked ending against the points of one run: the point ending nearest in time (within 2.5 s) and its winner."""
    rows = [r for r in csv.DictReader(l for l in open(path) if not l.startswith("#"))]
    ends = np.array([p["end_t"] for p in pts]) if pts else np.zeros(0)
    out = []
    for r in rows:
        if r["verdict"] == "split":                                      # no ending there: the rally went on
            m = re.search(r"at (\d+\.\d+)", r["how_it_ended"]); t = float(m.group(1)) if m else None
            cut = bool(t is not None and len(ends) and np.min(np.abs(ends - t)) <= 1.5)
            out.append(dict(end_s=t, truth="rally goes on", got="cut in two" if cut else "one rally", ok=not cut)); continue
        if r["winner_truth"] in ("?", "-", ""):
            continue
        t = float(r["end_s"]); k = int(np.argmin(np.abs(ends - t))) if len(ends) else None
        if k is None or abs(ends[k] - t) > 2.5:
            out.append(dict(end_s=t, truth=r["winner_truth"], got=None, ok=False)); continue
        out.append(dict(end_s=t, truth=r["winner_truth"], got=pts[k].get("winner_name"), ok=pts[k].get("winner_name") == r["winner_truth"]))
    return out


def crops(video, picks, out_dir, crop=160):
    """picks: {frame: (x, y, tag)}; one pass over the video, a crop round each."""
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video)); W, H = int(cap.get(3)), int(cap.get(4)); f = 0; last = max(picks) if picks else -1
    while f <= last:
        if f in picks:
            ok, frame = cap.read()
            if not ok:
                break
            x, y, tag = picks[f]; x0 = int(np.clip(x - crop / 2, 0, W - crop)); y0 = int(np.clip(y - crop / 2, 0, H - crop))
            cv2.imwrite(str(out / f"{tag}_{f}.jpg"), frame[y0:y0 + crop, x0:x0 + crop], [cv2.IMWRITE_JPEG_QUALITY, 92])
        elif not cap.grab():
            break
        f += 1
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("--video", required=True); ap.add_argument("--table", required=True); ap.add_argument("--pose")
    ap.add_argument("--near", default="near"); ap.add_argument("--far", default="far"); ap.add_argument("--game", required=True)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--handcheck"); ap.add_argument("--out", default="out_ml_phone")
    ap.add_argument("--set", action="append", default=["cross_front_s=0.5"], help="Config field=value for the events, as the classical run had")
    ap.add_argument("--n-look", type=int, default=64); ap.add_argument("--max-frames", type=int, help="the first frames only (a smoke test)")
    a = ap.parse_args()
    out = pathlib.Path(a.out) / a.game; out.mkdir(parents=True, exist_ok=True)
    names = {"near": a.near, "far": a.far}
    fps_v, cands, th = peaks_of(a.video, a.ckpt, out / "candidates.npz", a.max_frames)
    res = dict(game=a.game, ckpt=str(a.ckpt), threshold=th, frames=len(cands), labels=against_labels(a.game, cands))
    print(json.dumps(res["labels"]), flush=True)

    table = Table.load(a.table)
    cfg_t = Config(near_name=a.near, far_name=a.far)                     # the tracker's settings: the defaults, as the classical run
    cfg_e = Config(near_name=a.near, far_name=a.far)                     # the events' settings: with the classical run's overrides
    for kv in a.set:
        k, v = kv.split("="); setattr(cfg_e, k, type(getattr(cfg_e, k))(v))
    tracks, prow, fps = three_tracks(a.video, table, cfg_t, cands, a.max_frames)
    write_obs(out / "players.csv", prow); obs = load_obs(out / "players.csv")
    old = np.genfromtxt(pathlib.Path(a.run_dir) / "track.csv", delimiter=",", skip_header=1)
    n = min(len(old), len(tracks["classical"])); a_, b_ = old[:n, 2:4], tracks["classical"][:n, 2:4]
    same = (np.isnan(a_[:, 0]) & np.isnan(b_[:, 0])) | (np.hypot(*(np.nan_to_num(a_) - np.nan_to_num(b_)).T) < 0.5) & ~(np.isnan(a_[:, 0]) ^ np.isnan(b_[:, 0]))
    res["classical_track_reproduced"] = round(float(same.mean()), 4)

    cap = cv2.VideoCapture(str(a.video)); size = (int(cap.get(3)), int(cap.get(4))); cap.release()
    cam = Camera.from_table_pnp(table, *size)
    assigned = None
    if a.pose:
        assigned = pose_mod.assign(pose_mod.load(a.pose), table, standing_zone(table, 1.0)); assigned, _ = pose_mod.clean_legs(assigned, table)
    runs = {}
    for m in MODES:
        ev, pts = points_of(tracks[m], table, cfg_e, fps, obs, names)
        shots, fits = technique.measure(pts, ev, tracks[m], fps, cam, assigned)
        d = out.parent / f"{a.game}_{m}"; d.mkdir(parents=True, exist_ok=True)
        with open(d / "track.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["frame", "t", "x", "y", "track", "n_cands"]); w.writerows(tracks[m].tolist())
        with open(d / "events.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ev[0].keys()) if ev else ["t"]); w.writeheader(); w.writerows(ev)
        (d / "rallies.json").write_text(json.dumps(pts, indent=1)); write_obs(d / "players.csv", prow)
        (d / "table.json").write_text(pathlib.Path(a.table).read_text())
        (d / "settings.json").write_text(json.dumps({"detector": m, "model": str(a.ckpt), "overrides": a.set}, indent=1))
        runs[m] = dict(events=ev, points=pts, shots=shots, fits=fits)
    ref = runs["classical"]["points"]
    inpt = np.zeros(len(tracks["classical"]), bool)
    for p in ref:
        inpt[int(p["start_t"] * fps):int(p["end_t"] * fps) + 1] = True
    res["in_point_frames"] = int(inpt.sum()); res["runs"] = {}
    for m in MODES:
        r = runs[m]; tr = tracks[m]
        ok = [f for f in r["fits"] if f.get("ok")]
        res["runs"][m] = dict(points=len(r["points"]), ball_seen_in_points=round(float((~np.isnan(tr[inpt, 2])).mean()), 4),
                              shots=len(r["shots"]), fitted=len(ok), fitted_tight=sum(1 for f in ok if f.get("rms_px", 1e9) <= 3.0),
                              with_speed=sum(1 for s in r["shots"] if s.get("speed") is not None),
                              bounces=sum(1 for e in r["events"] if e["kind"] == "bounce"), crossings=sum(1 for e in r["events"] if e["kind"] == "net"),
                              hand_check=hand_check(a.handcheck, r["points"]) if a.handcheck else None)
        if a.handcheck:
            hc = res["runs"][m]["hand_check"]; res["runs"][m]["hand_check_ok"] = f"{sum(h['ok'] for h in hc)}/{len(hc)}"
    tc = tracks["classical"]; cat = dict(both=0, classical_only=0, learned_only=0, apart=0, neither=0); pool = dict(learned_only=[], apart=[])
    for f in np.where(inpt)[0]:
        pk = cands[f][0] if f < len(cands) and cands[f] else None
        has = not np.isnan(tc[f, 2])
        if has and pk is not None:
            if np.hypot(pk[0] - tc[f, 2], pk[1] - tc[f, 3]) <= APART_PX:
                cat["both"] += 1
            else:
                cat["apart"] += 1; pool["apart"].append((int(f), pk[0], pk[1]))
        elif has:
            cat["classical_only"] += 1
        elif pk is not None:
            cat["learned_only"] += 1; pool["learned_only"].append((int(f), pk[0], pk[1]))
        else:
            cat["neither"] += 1
    res["in_point_by_detector"] = cat
    rng = np.random.default_rng(0); picks = {}
    for tag, fs in pool.items():
        for k in rng.choice(len(fs), size=min(a.n_look, len(fs)), replace=False):
            picks[fs[k][0]] = (fs[k][1], fs[k][2], tag)
    crops(a.video, picks, out / "look")
    (out / "eval.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=1))
    for m in MODES:
        print(m, json.dumps({k: v for k, v in res["runs"][m].items() if k != "hand_check"}))
        for h in res["runs"][m].get("hand_check") or []:
            if not h["ok"]:
                print("   hand check miss:", h)


if __name__ == "__main__":
    main()
