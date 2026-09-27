"""Design probes for a repaired point state machine, development clips only.
(i) v0 winner errors under perfect windows by true ending type; (ii) recoverable missed crossings (consecutive bounces on
opposite sides with no detected crossing between them); (iii) serve signature (own-side bounce shortly before the first
crossing) in true points vs. predicted points that lie between true points."""
import csv, sys, pathlib, collections
ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from tt_scout.metrics import load_points_csv, load_pred_points, match_points
from tt_scout.scoring import who_won
from analysis.oracle import load_events  # noqa

NET = 1.37
by_end = collections.defaultdict(lambda: [0, 0]); implied = [0, 0]; sig_true = [0, 0]; sig_fp = [0, 0]; near_net = [0, 0]
def ending_of(clip):
    return {r["id"]: r["ending"] for r in csv.DictReader(open(ROOT / f"labels/{clip}.points.csv"))}
for clip in sys.argv[1:]:
    allp = load_points_csv(ROOT / f"labels/{clip}.points.csv"); endk = ending_of(clip)
    truth = [t for t in allp if t["ending"] != "unknown" and t["end_t"] is not None]
    pred = load_pred_points(ROOT / f"out/{clip}/rallies.json"); ev = load_events(ROOT / f"out/{clip}/events.csv")
    for e in ev:
        if e["kind"] == "bounce":
            near_net[1] += 1; near_net[0] += abs(e["x_m"] - NET) < 0.10
    for t in truth:
        w = [e for e in ev if t["start_t"] - 0.3 <= e["t"] <= t["end_t"] + 0.3 and e["kind"] in ("bounce", "net")]
        ok = who_won(dict(events=w, serve_side=t["server"], coverage=1.0)) == t["winner"]
        by_end[endk[t["id"]]][0] += ok; by_end[endk[t["id"]]][1] += 1
        seq = sorted(w, key=lambda e: e["t"])
        for a, b in zip(seq, seq[1:]):
            if a["kind"] == "bounce" and b["kind"] == "bounce":
                implied[1] += 1
                implied[0] += (a["side"] != b["side"])
        cross = [e for e in seq if e["kind"] == "net"]
        if cross:
            c0 = cross[0]; origin = "near" if c0["side"] == "far" else "far"
            sig_true[1] += 1
            sig_true[0] += any(e["kind"] == "bounce" and e["side"] == origin and 0 < c0["t"] - e["t"] <= 0.8 for e in ev)
    pairs, fp, fn, _ = match_points(allp, pred)
    ov = lambda a, b: max(0.0, min(a["end_t"], b["end_t"]) - max(a["start_t"], b["start_t"]))
    for p in fp:
        if any(ov(t, p) > 0 for t in truth):
            continue
        w = sorted([e for e in ev if p["start_t"] - 1.0 <= e["t"] <= p["end_t"]], key=lambda e: e["t"])
        cross = [e for e in w if e["kind"] == "net"]
        if cross:
            c0 = cross[0]; origin = "near" if c0["side"] == "far" else "far"
            sig_fp[1] += 1
            sig_fp[0] += any(e["kind"] == "bounce" and e["side"] == origin and 0 < c0["t"] - e["t"] <= 0.8 for e in w)
print("(i) v0 winner accuracy under perfect windows, by true ending type:")
for k, (a, n) in sorted(by_end.items(), key=lambda kv: -kv[1][1]):
    print(f"    {k:18s} {a:3d}/{n:3d} = {a/max(1,n):.2f}")
print(f"(ii) consecutive bounce pairs with NO crossing between them inside true points: {implied[1]}, of which on OPPOSITE sides (a crossing was missed): {implied[0]} = {implied[0]/max(1,implied[1]):.2f}")
print(f"(iii) serve signature (own-side bounce <= 0.8 s before first crossing): true points {sig_true[0]}/{sig_true[1]} = {sig_true[0]/max(1,sig_true[1]):.2f}; "
      f"false points between true points {sig_fp[0]}/{sig_fp[1]} = {sig_fp[0]/max(1,sig_fp[1]):.2f}")
print(f"(iv) detected bounces within 10 cm of the net line (side ambiguous): {near_net[0]}/{near_net[1]} = {near_net[0]/max(1,near_net[1]):.3f}")
