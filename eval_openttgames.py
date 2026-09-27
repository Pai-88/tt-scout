"""Run tt_scout on an OpenTTGames game and score it against the dataset's own labels.
Usage: python eval_openttgames.py data/test_2 [--save-video] [--max-frames N]"""
import json, sys, pathlib
import numpy as np
from tt_scout.config import Config
from tt_scout.run import analyse
from tt_scout.evaluate import evaluate

base = pathlib.Path(sys.argv[1])
maxf = int(sys.argv[sys.argv.index("--max-frames") + 1]) if "--max-frames" in sys.argv else None
colour = sys.argv[sys.argv.index("--colour") + 1] if "--colour" in sys.argv else "white"
out_root = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "out"
overrides = {}
for i, a in enumerate(sys.argv):
    if a == "--set":                                   # --set key=value  (any Config field)
        k, v = sys.argv[i + 1].split("="); overrides[k] = type(getattr(Config(), k))(v)
truth = json.load(open(str(base) + "_truth.json"))
cfg = Config(ball_colour=colour, **overrides)
if "--from-track" in sys.argv:                        # recompute events from a saved track.csv (no detection pass)
    import time, csv, json as _json
    from tt_scout.table import Table
    from tt_scout.events import detect_events, segment_rallies, segment_points
    from tt_scout.scoring import who_won
    from tt_scout.report import make_report
    out = pathlib.Path(out_root) / base.name
    tr = np.genfromtxt(out / "track.csv", delimiter=",", skip_header=1)
    table = Table.load(str(base) + "_table.json"); fps = float(truth["fps"])
    ev = detect_events(tr[:, :5], table, cfg, fps); ra = segment_points(ev, cfg, tr[:, :5]) or segment_rallies(tr[:, :5], ev, cfg, fps)
    for r in ra: r["winner"] = who_won(r)
    from tt_scout.players import load_obs, name_points
    name_points(ra, load_obs(out / "players.csv"), table.meta.get("players"), {"near": cfg.near_name, "far": cfg.far_name})
    with open(out / "events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ev[0].keys()) if ev else ["t"]); w.writeheader(); w.writerows(ev)
    from tt_scout.stats import write_stats
    write_stats(out, ra)
    (out / "rallies.json").write_text(_json.dumps(ra, indent=1)); make_report(ra, ev, out / "report.png", title=base.name + ".mp4")
    res = dict(out=str(out), fps=fps, n_frames=len(tr), events=ev, rallies=ra, ball_seen_frac=float(np.mean(~np.isnan(tr[:, 2]))), seconds=0.0)
else:
    res = analyse(str(base) + ".mp4", str(base) + "_table.json", cfg,
                  save_video="--save-video" in sys.argv, max_frames=maxf, out_root=out_root)
tev = [e for e in truth["events"] if maxf is None or e["frame"] < maxf]
m = evaluate(res["events"], tev, res["fps"], tol_frames=6)

track = np.genfromtxt(res["out"] + "/track.csv", delimiter=",", skip_header=1)
errs, missed, wrong = [], 0, 0
for f, b in truth["ball"].items():
    f = int(f)
    if b["x"] < 0 or f >= len(track) or (maxf and f >= maxf):
        continue
    x, y = track[f, 2], track[f, 3]
    if np.isnan(x):
        missed += 1; continue
    d = float(np.hypot(x - b["x"], y - b["y"]))
    (errs if d < 40 else [None]).append(d)
    wrong += d >= 40
n_lab = len(errs) + missed + wrong
print(f"\n=== {base.name} [colour={colour} {overrides}]: {res['n_frames']} frames at {res['fps']:.0f} fps in {res['seconds']:.0f}s; ball seen in {res['ball_seen_frac']*100:.0f}% of all frames")
print(f"ball vs {n_lab} labelled frames: found {len(errs)+wrong} ({(len(errs)+wrong)/max(1,n_lab)*100:.0f}%), "
      f"wrong object (>40 px off) {wrong}, missed {missed}; position error median {np.median(errs):.1f} px, p90 {np.percentile(errs, 90):.1f} px  (TTNet paper: ~2 px RMSE)")
print(f"bounces: {m['n_pred_bounces']} detected vs {m['n_true_bounces']} labelled | precision {m['bounce_precision']:.2f}  recall {m['bounce_recall']:.2f} "
      f"(near {m['bounce_recall_near'][0]}/{m['bounce_recall_near'][1]}, far {m['bounce_recall_far'][0]}/{m['bounce_recall_far'][1]})")
# Detections with no label: are they where the labelled net crossings say a bounce must be?
# Between two consecutive crossings (< 1.5 s apart) the ball must bounce exactly once on the receiving side.
cross = sorted(e["t"] for e in tev if e["kind"] == "net")
lab_b = [e for e in tev if e["kind"] == "bounce"]
unl = [e for e in res["events"] if e["kind"] == "bounce" and not any(abs(e["t"] - b["t"]) <= 6 / res["fps"] for b in lab_b)]
implied = [e for e in unl if any(c1 < e["t"] < c2 and c2 - c1 < 1.5 for c1, c2 in zip(cross, cross[1:]))]
after_last = [e for e in unl if lab_b and cross and e["t"] > max(max(c for c in cross), max(b["t"] for b in lab_b))]
print(f"unlabelled bounce detections: {len(unl)}; of these {len(implied)} lie between two labelled net crossings (a bounce must exist there) "
      f"and {len(after_last)} are after the last label of the clip; unexplained: {len(unl) - len(implied) - len(after_last)}")
print(f"bounce position error vs labelled ball: median {m['loc_err_median_cm']:.1f} cm, p90 {m['loc_err_p90_cm']:.1f} cm")
print(f"net crossings: {m['net_recall'][0]}/{m['net_recall'][1]} labelled crossings detected")
print(f"rallies segmented: {len(res['rallies'])};  report: {res['out']}/report.png")
