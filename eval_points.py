"""Score detected points and winners against the Extended OpenTT Games labels (rally endings).
Usage: python eval_points.py data/test_3
Labels: data/extended_labels/data/raw/game_data/test/<stem>.json  {frame: "left_out" | "right_winner" | stroke...}
left = near (x=0 end, left in the image), right = far. The side in the label CAUSED the ending, so it loses,
except "winner" where it wins."""
import json, pathlib, sys
FPS = 120
ENDINGS = ("net", "out", "winner", "double_bounce", "miss_on_own_side", "not_hitting_ball")
base = pathlib.Path(sys.argv[1])
lab = json.load(open(f"data/extended_labels/data/raw/game_data/test/{base.name}.json"))
ends = []
for f, v in sorted(((int(k), v) for k, v in lab.items()), key=lambda kv: kv[0]):
    parts = v.split()
    if len(parts) == 1 and "_" in v:
        side, kind = v.split("_", 1)
        if kind in ENDINGS:
            causer = "near" if side == "left" else "far"
            ends.append(dict(t=f / FPS, kind=kind, causer=causer, winner=causer if kind == "winner" else ("far" if causer == "near" else "near")))
pts = json.load(open(f"out/{base.name}/rallies.json"))
print(f"{base.name}: {len(ends)} labelled point endings, {len(pts)} detected points")
right = wrong = unmatched = 0
used = set()
for e in ends:
    m = [p for p in pts if p["start_t"] - 0.5 <= e["t"] <= p["end_t"] + 0.5]
    if not m:
        unmatched += 1; print(f"  ending {e['kind']:16s} by {e['causer']:4s} @{e['t']:6.1f}s -> winner {e['winner']:4s} | NO detected point covers it"); continue
    p = m[0]; used.add(p["id"])
    ok = p["winner"] == e["winner"]; right += ok; wrong += (not ok)
    print(f"  ending {e['kind']:16s} by {e['causer']:4s} @{e['t']:6.1f}s -> winner {e['winner']:4s} | point {p['id']} ({p['start_t']:.1f}-{p['end_t']:.1f}s, {p.get('ending')}) says {p['winner']}  {'OK' if ok else 'WRONG'}")
extra = [p for p in pts if p["id"] not in used]
for p in extra:
    print(f"  detected point {p['id']} ({p['start_t']:.1f}-{p['end_t']:.1f}s -> {p['winner']}) has no labelled ending")
print(f"winners: {right} right, {wrong} wrong; labelled endings with no point: {unmatched}; detected points with no label: {len(extra)}")
