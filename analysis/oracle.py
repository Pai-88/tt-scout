"""Where are the points lost? Decomposes v0 point errors on a set of clips, using only saved outputs (no detection).

  A1  detectability: does a true point contain any detected net crossing / bounce at all?
  A2  winner given perfect segmentation: who_won on the detected events inside the TRUE point window.
  A3  what the false and missed points are: splits / boundary near-misses / spurious play between points.

Usage: python analysis/oracle.py game_1 game_2 ...   (development clips; do not tune on the test clips)
"""
import csv, json, sys, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tt_scout.metrics import load_points_csv, load_pred_points, match_points
from tt_scout.scoring import who_won


def load_events(path):
    out = []
    for r in csv.DictReader(open(path)):
        out.append(dict(t=float(r["t"]), kind=r["kind"], side=r["side"], x_m=float(r["x_m"]), y_m=float(r["y_m"]),
                        strength=float(r["strength"]), frame=int(float(r["frame"]))))
    return out


tot = dict(n=0, no_events=0, no_cross=0, cross=0, win_n=0, win_tight=0, win_loose=0, fn=0, fn_overlapped=0, fn_alone=0,
           fp=0, fp_in_true=0, fp_between=0, near_miss_end=0, near_miss_start=0, split=0)
for clip in sys.argv[1:]:
    truth = [t for t in load_points_csv(ROOT / f"labels/{clip}.points.csv") if t["ending"] != "unknown" and t["end_t"] is not None]
    pred = load_pred_points(ROOT / f"out/{clip}/rallies.json")
    ev = load_events(ROOT / f"out/{clip}/events.csv")
    pairs, fp, fn, _ = match_points(load_points_csv(ROOT / f"labels/{clip}.points.csv"), pred)
    c = dict(n=len(truth), no_events=0, no_cross=0, cross=0, win_n=0, win_tight=0, win_loose=0)
    for t in truth:
        inside = [e for e in ev if t["start_t"] - 0.3 <= e["t"] <= t["end_t"] + 0.5]
        ncross = sum(e["kind"] == "net" for e in inside)
        if not inside: c["no_events"] += 1
        elif ncross == 0: c["no_cross"] += 1
        else: c["cross"] += 1
        if t["winner"]:
            c["win_n"] += 1
            for key, pad in (("win_tight", 0.3), ("win_loose", 1.5)):
                w = [e for e in ev if t["start_t"] - 0.3 <= e["t"] <= t["end_t"] + pad]
                r = dict(events=w, serve_side=t["server"], coverage=1.0)
                c[key] += who_won(r) == t["winner"]
    ov = lambda a, b: max(0.0, min(a["end_t"], b["end_t"]) - max(a["start_t"], b["start_t"]))
    fn_over = [t for t in fn if any(ov(t, p) > 0 for p in pred)]
    split = [t for t in fn if sum(ov(t, p) > 0 for p in pred) >= 2]
    near_end = [t for t in fn if any(ov(t, p) > 0 and p["start_t"] - 0.5 <= t["start_t"] and 0 < t["end_t"] - (p["end_t"] + 0.5) <= 1.0 for p in pred)]
    near_start = [t for t in fn if any(ov(t, p) > 0 and 0 < (p["start_t"] - 0.5) - t["start_t"] <= 1.0 for p in pred)]
    fp_in = [p for p in fp if any(ov(t, p) > 0 for t in truth)]
    print(f"{clip}: truth {c['n']} | detectable: crossing inside {c['cross']}, events but no crossing {c['no_cross']}, nothing {c['no_events']} | "
          f"winner with PERFECT windows: {c['win_tight']}/{c['win_n']} (+0.3 s), {c['win_loose']}/{c['win_n']} (+1.5 s) | "
          f"matched {len(pairs)}, FN {len(fn)} (overlapped by a prediction {len(fn_over)}, of which split in >=2 {len(split)}, "
          f"end near-miss <=1 s {len(near_end)}, start near-miss {len(near_start)}; no prediction at all {len(fn) - len(fn_over)}) | "
          f"FP {len(fp)} (inside a true point {len(fp_in)}, between points {len(fp) - len(fp_in)})")
    for k in ("n", "no_events", "no_cross", "cross", "win_n", "win_tight", "win_loose"): tot[k] += c[k]
    tot["fn"] += len(fn); tot["fn_overlapped"] += len(fn_over); tot["fn_alone"] += len(fn) - len(fn_over); tot["split"] += len(split)
    tot["near_miss_end"] += len(near_end); tot["near_miss_start"] += len(near_start)
    tot["fp"] += len(fp); tot["fp_in_true"] += len(fp_in); tot["fp_between"] += len(fp) - len(fp_in)
print("\nTOTAL", json.dumps(tot))
print(f"ceiling from events: {tot['cross']}/{tot['n']} = {tot['cross']/max(1,tot['n']):.2f} of true points contain a detected crossing")
print(f"winner with perfect segmentation: {tot['win_tight']}/{tot['win_n']} = {tot['win_tight']/max(1,tot['win_n']):.2f} (tight), {tot['win_loose']/max(1,tot['win_n']):.2f} (loose)")
