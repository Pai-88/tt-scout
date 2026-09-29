"""Where does the pipeline lose the ball, and which bounces does it miss? Reads out/<stem>/track.csv + events.csv.
Usage: python diag_openttgames.py data/test_2 [max_frames] [--out DIR] [--set key=value]"""
import csv, json, sys, pathlib
import numpy as np, cv2
from tt_scout.config import Config
from tt_scout.table import Table
from tt_scout.detector import BallDetector

if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    raise SystemExit(__doc__)
base = pathlib.Path(sys.argv[1]); maxf = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 10**9
out_root = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "out"
overrides = {}
for i, a in enumerate(sys.argv):
    if a == "--set":
        k, v = sys.argv[i + 1].split("="); overrides[k] = type(getattr(Config(), k))(v)
truth = json.load(open(str(base) + "_truth.json")); fps = truth["fps"]
out = pathlib.Path(out_root) / base.name
track = np.genfromtxt(out / "track.csv", delimiter=",", skip_header=1)
pred = [dict(frame=int(r["frame"]), kind=r["kind"], x_m=float(r["x_m"]), strength=float(r["strength"])) for r in csv.DictReader(open(out / "events.csv"))]
ball = {int(k): v for k, v in truth["ball"].items() if v["x"] >= 0 and int(k) < maxf}

# 1) misses: rerun detector on the missed frames' neighbourhood to see which stage dropped the ball
missed = [f for f in ball if f < len(track) and np.isnan(track[f, 2])]
cfg = Config(ball_colour="white", **overrides); table = Table.load(str(base) + "_table.json"); det = BallDetector(cfg, table, fps)
from tt_scout.detector import COLOUR_RANGES
cap = cv2.VideoCapture(str(base) + ".mp4"); reasons = {"candidate existed (tracker did not take it)": 0, "inside player margin": 0, "no fg blob": 0, "blob too big/merged": 0, "blob too small": 0, "not moving enough": 0, "colour": 0, "other": 0}
ms = set(missed); last_gray = None
for i in range(min(maxf, len(track))):
    ok, fr = cap.read()
    if not ok: break
    cands = det.detect(fr)
    if i not in ms: continue
    x, y = ball[i]["x"], ball[i]["y"]
    if any(np.hypot(c.x - x, c.y - y) < 20 for c in cands):
        reasons["candidate existed (tracker did not take it)"] += 1; continue
    fg = det.last_mask; n, lab, stats, cent = cv2.connectedComponentsWithStats(fg, connectivity=8)
    win = lab[max(0, y - 6):y + 7, max(0, x - 6):x + 7]; ids = [v for v in np.unique(win) if v]
    if not ids: reasons["no fg blob"] += 1; continue
    j = max(ids, key=lambda v: stats[v, cv2.CC_STAT_AREA]); area = stats[j, cv2.CC_STAT_AREA]
    if area > det.area_max: reasons["blob too big/merged"] += 1
    elif area < det.area_min: reasons["blob too small"] += 1
    else:
        big = np.isin(lab, [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] > det.area_max]).astype(np.uint8)
        k = 2 * cfg.player_margin_px + 1; big = cv2.dilate(big, np.ones((k, k), np.uint8))
        if big[y, x]: reasons["inside player margin"] += 1
        else:
            comp = lab == j
            mv = float(det.moving[comp].mean()) if det.moving is not None else 1.0
            hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV); sub = hsv[comp]
            cf = float(COLOUR_RANGES["white"](sub[:, 0], sub[:, 1], sub[:, 2]).mean())
            if mv < cfg.min_moving_frac: reasons[f"not moving enough (frac<{cfg.min_moving_frac})"] = reasons.get(f"not moving enough (frac<{cfg.min_moving_frac})", 0) + 1
            elif cf < cfg.colour_min_frac: reasons["colour"] += 1
            else: reasons["other"] += 1
print(f"missed labelled frames: {len(missed)} of {len(ball)}"); [print(f"  {k:48s} {v}") for k, v in reasons.items() if v]

# 2) events side by side
print("\ntruth bounces (first %d frames) vs detections within +-6 frames:" % maxf)
for e in [e for e in truth["events"] if e["kind"] == "bounce" and e["frame"] < maxf]:
    f = e["frame"]; near = [p for p in pred if abs(p["frame"] - f) <= 6]
    seg = track[max(0, f - 8):f + 9, 2]; seen = int(np.sum(~np.isnan(seg)))
    print(f"  bounce @ {f:5d} x={e.get('x_m', float('nan')):.2f}m  tracked {seen:2d}/17 frames around it  -> " + (", ".join(f"{p['kind']}@{p['frame']} (str {p['strength']:.1f})" for p in near) or "nothing"))
fp = [p for p in pred if p["kind"] == "bounce" and p["frame"] < maxf and not any(abs(p["frame"] - e["frame"]) <= 6 for e in truth["events"] if e["kind"] == "bounce")]
print("\nfalse bounces:", ", ".join(f"@{p['frame']} x={p['x_m']:.2f}m str {p['strength']:.1f}" for p in fp) or "none")
nets = [e for e in truth["events"] if e["kind"] == "net" and e["frame"] < maxf]
print("\ntruth nets:", ", ".join(f"@{e['frame']}->" + ("/".join(p['kind'] for p in pred if abs(p['frame'] - e['frame']) <= 6) or "-") for e in nets))
