"""Score a run's points against a hand check of point endings (data/own/<clip>_handcheck.csv).
    python analysis/score_handcheck.py out_own/IMG_3144_v2 data/own/IMG_3144_handcheck.csv --near Sam --far Robin
A truth row matches the run point whose span holds its end time (start_t .. end_t + 0.5 s)."""
import argparse, csv, json

ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("truth"); ap.add_argument("--near", default="Sam"); ap.add_argument("--far", default="Robin")
ap.add_argument("-v", action="store_true")
a = ap.parse_args()
pts = json.load(open(f"{a.run}/rallies.json"))
rows = [r for r in csv.DictReader(l for l in open(a.truth) if not l.startswith("#"))]
name = {"near": a.near, "far": a.far}
n = dict(ok=0, wrong=0, fp_still=0, fp_gone=0, fn_still=0, fn_found=0, unmatched=0)
used = set()
for r in rows:
    t_end = float(r["end_s"])
    m = [p for p in pts if p["start_t"] - 0.3 <= t_end <= p["end_t"] + 0.5]
    if r["verdict"] == "fp":
        k = "fp_still" if m else "fp_gone"; n[k] += 1
        if a.v: print(f"  fp  {t_end:7.1f}: {'STILL THERE' if m else 'gone'}")
        continue
    if r["winner_truth"] in ("?", "-"):
        continue
    if not m:
        n["fn_still" if r["verdict"] == "fn" else "unmatched"] += 1
        if a.v: print(f"  --  {t_end:7.1f}: no point covers it (truth {r['winner_truth']})")
        continue
    p = m[0]; used.add(p["id"])
    if r["verdict"] == "fn":
        n["fn_found"] += 1
    got = p.get("winner_name") or (name.get(p.get("winner")) if p.get("winner") else None)
    ok = got == r["winner_truth"]
    n["ok" if ok else "wrong"] += 1
    if a.v: print(f"  {'ok ' if ok else 'BAD'} {t_end:7.1f}: truth {r['winner_truth']:5s} got {str(got):5s} ({p['ending']}, point {p['id']} {p['start_t']:.1f}-{p['end_t']:.1f})")
checked = n["ok"] + n["wrong"]
print(f"{a.run}: winner right {n['ok']}/{checked}; false point still there {n['fp_still']}, gone {n['fp_gone']}; "
      f"missed point found {n['fn_found']}, still missed {n['fn_still']}; checked endings with no point {n['unmatched']}; points in run {len(pts)}")
