"""What is still wrong under a given point-logic version (development clips). Usage: python -m analysis.errors_v1 v1 game_1 ..."""
import csv, sys, json, pathlib, collections
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from tt_scout.metrics import load_points_csv, load_pred_points, match_points
version, clips = sys.argv[1], sys.argv[2:]
fp_kind = collections.Counter(); fn_kind = collections.Counter(); wrong_by_end = collections.defaultdict(lambda: [0, 0]); wrong_by_pred = collections.Counter()
fp_feat = dict(between_ncross=[], between_gap=[], inside_ncross=[]); srv_wrong = collections.Counter()
for clip in clips:
    allp = load_points_csv(ROOT / f"labels/{clip}.points.csv"); endk = {r["id"]: r["ending"] for r in csv.DictReader(open(ROOT / f"labels/{clip}.points.csv"))}
    truth = sorted([t for t in allp if t["ending"] != "unknown" and t["end_t"] is not None], key=lambda t: t["start_t"])
    raw = json.load(open(ROOT / f"out_{version}/{clip}/rallies.json")); byid = {p["id"]: p for p in raw}
    pred = load_pred_points(ROOT / f"out_{version}/{clip}/rallies.json")
    pairs, fp, fn, _ = match_points(allp, pred)
    ov = lambda a, b: max(0.0, min(a["end_t"], b["end_t"]) - max(a["start_t"], b["start_t"]))
    for p in fp:
        inside = [t for t in truth if ov(t, p) > 0]
        n = byid[p["id"]]["n_crossings"]
        if inside:
            t = inside[0]; matched = any(tt is t for tt, _ in pairs)
            fp_kind["inside a true point that IS matched by another prediction (fragment)" if matched else "inside a true point that is NOT matched (boundary/split)"] += 1
            fp_feat["inside_ncross"].append(n)
        else:
            prev = [t for t in truth if t["end_t"] <= p["start_t"]]
            fp_kind["between true points"] += 1; fp_feat["between_ncross"].append(n)
            if prev: fp_feat["between_gap"].append(p["start_t"] - prev[-1]["end_t"])
    for t in fn:
        over = [p for p in pred if ov(t, p) > 0]
        if not over: fn_kind["no prediction overlaps"] += 1
        elif len(over) >= 2: fn_kind["split into >= 2 predictions"] += 1
        else:
            p = over[0]
            if p["start_t"] - 0.5 > t["start_t"]: fn_kind[f"one prediction, starts late"] += 1
            elif t["end_t"] > p["end_t"] + 0.5: fn_kind["one prediction, ends early"] += 1
            else: fn_kind["one prediction, taken by another true point"] += 1
    for t, p in pairs:
        if t["winner"]:
            ok = p["winner"] == t["winner"]; wrong_by_end[endk[t["id"]]][0] += ok; wrong_by_end[endk[t["id"]]][1] += 1
            if not ok: wrong_by_pred[(endk[t["id"]], byid[p["id"]]["ending"])] += 1
        if t["server"] and p["server"] != t["server"]:
            first = next((e for e in byid[p["id"]]["events"] if e["kind"] == "net"), None)
            srv_wrong["first crossing implied" if first and first.get("implied") else "first crossing detected"] += 1
print("FALSE points:", dict(fp_kind))
for k, v in fp_feat.items():
    if v: print(f"   {k}: n={len(v)} median {np.median(v):.1f}  p10 {np.percentile(v,10):.1f}  p90 {np.percentile(v,90):.1f}")
print("MISSED points:", dict(fn_kind))
print("WINNER accuracy on matched points by true ending:", {k: f"{a}/{n}" for k, (a, n) in sorted(wrong_by_end.items(), key=lambda kv: -kv[1][1])})
print("wrong winners (true ending, predicted ending):", wrong_by_pred.most_common(8))
print("wrong servers:", dict(srv_wrong))
