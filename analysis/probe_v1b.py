"""More design probes on development clips: (a) what separates spurious between-point 'points' from real ones (timing, length,
first-shot speed), (b) final-stretch track geometry for true 'out' vs 'landed' endings when no bounce was detected."""
import csv, sys, pathlib, json
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from tt_scout.metrics import load_points_csv, load_pred_points, match_points
from tt_scout.table import Table
from analysis.oracle import load_events
L = 2.74
def q(v): return "n=0" if not len(v) else f"n={len(v)} median {np.median(v):.2f} [p10 {np.percentile(v,10):.2f}, p90 {np.percentile(v,90):.2f}]"
real = dict(gap_prev=[], dur=[], ncross=[], speed=[]); spur = dict(gap_prev=[], gap_next=[], dur=[], ncross=[], speed=[])
fin = dict(out=[], landed=[])
for clip in sys.argv[1:]:
    allp = load_points_csv(ROOT / f"labels/{clip}.points.csv"); endk = {r["id"]: r["ending"] for r in csv.DictReader(open(ROOT / f"labels/{clip}.points.csv"))}
    truth = sorted([t for t in allp if t["ending"] != "unknown" and t["end_t"] is not None], key=lambda t: t["start_t"])
    pred = load_pred_points(ROOT / f"out/{clip}/rallies.json"); ev = load_events(ROOT / f"out/{clip}/events.csv")
    table = Table.load(str(ROOT / f"data/{clip}_table.json"))
    tr = np.genfromtxt(ROOT / f"out/{clip}/track.csv", delimiter=",", skip_header=1)
    def first_speed(t0, t1):
        w = sorted([e for e in ev if t0 <= e["t"] <= t1 and e["kind"] in ("bounce", "net")], key=lambda e: e["t"])
        c = next((e for e in w if e["kind"] == "net"), None)
        if not c: return None
        b = next((e for e in w if e["kind"] == "bounce" and 0 < e["t"] - c["t"] <= 1.0), None)
        return abs(b["x_m"] - 1.37) / (b["t"] - c["t"]) if b else None
    for i, t in enumerate(truth):
        real["dur"].append(t["end_t"] - t["start_t"])
        if i: real["gap_prev"].append(t["start_t"] - truth[i - 1]["end_t"])
        w = [e for e in ev if t["start_t"] - 0.3 <= e["t"] <= t["end_t"] + 0.3]
        real["ncross"].append(sum(e["kind"] == "net" for e in w))
        s = first_speed(t["start_t"] - 0.3, t["end_t"] + 0.3)
        if s: real["speed"].append(s)
        # final stretch: last detected crossing inside the point, no bounce detected after it
        cr = [e for e in w if e["kind"] == "net"]
        if cr and not any(e["kind"] == "bounce" and e["t"] > cr[-1]["t"] for e in w):
            side = cr[-1]["side"]; sel = (tr[:, 1] > cr[-1]["t"]) & (tr[:, 1] <= cr[-1]["t"] + 1.2) & ~np.isnan(tr[:, 2])
            if sel.sum() >= 3:
                xm = table.to_table(tr[sel][:, 2:4])[:, 0]
                beyond = (xm.max() - L) if side == "far" else (0 - xm.min())      # how far past that side's end line the track got (m)
                kind = endk[t["id"]]
                truly_landed = kind in ("winner", "double_bounce", "not_hitting_ball", "miss_on_own_side", "net") and t["winner"] != side
                fin["landed" if truly_landed else "out"].append(beyond)
    pairs, fp, fn, _ = match_points(allp, pred)
    ov = lambda a, b: max(0.0, min(a["end_t"], b["end_t"]) - max(a["start_t"], b["start_t"]))
    for p in fp:
        if any(ov(t, p) > 0 for t in truth): continue
        prev = [t for t in truth if t["end_t"] <= p["start_t"]]; nxt = [t for t in truth if t["start_t"] >= p["end_t"]]
        if prev: spur["gap_prev"].append(p["start_t"] - prev[-1]["end_t"])
        if nxt: spur["gap_next"].append(nxt[0]["start_t"] - p["end_t"])
        spur["dur"].append(p["end_t"] - p["start_t"])
        w = [e for e in ev if p["start_t"] <= e["t"] <= p["end_t"]]; spur["ncross"].append(sum(e["kind"] == "net" for e in w))
        s = first_speed(p["start_t"] - 1.0, p["end_t"])
        if s: spur["speed"].append(s)
print("REAL points:      gap since previous point end (s):", q(real["gap_prev"])); print("                  detected crossings:", q(real["ncross"]), "| first-shot speed m/s:", q(real["speed"]))
print("SPURIOUS between: gap since previous point end (s):", q(spur["gap_prev"]), "| gap to next true serve:", q(spur["gap_next"]))
print("                  detected crossings:", q(spur["ncross"]), "| first-shot speed m/s:", q(spur["speed"]), "| share with 1 crossing:", round(float(np.mean(np.array(spur['ncross']) <= 1)), 2))
print("real points with <=1 detected crossing:", round(float(np.mean(np.array(real['ncross']) <= 1)), 2))
print("FINAL STRETCH with no detected bounce after the last crossing; track extent beyond that side's end line (m):")
print("   truly OUT/long (receiver of the crossing wins):", q(fin["out"])); print("   truly LANDED, bounce missed (hitter wins):    ", q(fin["landed"]))
