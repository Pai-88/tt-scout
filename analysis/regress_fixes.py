"""Old vs new event/point rules on the 12 OpenTTGames videos from their saved tracks (no detection pass); scored with tt_scout.metrics.
Old = the 2026-09-26 additions switched off; new = the defaults.
    python analysis/regress_fixes.py"""
import csv, json, pathlib, subprocess, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from tt_scout.config import Config
from tt_scout.table import Table
from tt_scout.events import detect_events
from tt_scout.points_v1 import segment_points_v1, V1Config

CLIPS = ["game_1", "game_2", "game_3", "game_4", "game_5", "test_1", "test_2", "test_3", "test_4", "test_5", "test_6", "test_7"]
SETS = {"old": (dict(contact_shift=False, net_touch_s=0.0, net_touch_bounce_s=0.0), dict(implied_max_gap_s=1e9, implied_cross_max_gap_s=1e9, min_bounces=0)),
        "new": ({}, {}), "new_no_minb": ({}, dict(min_bounces=0))}
for name, (ev_over, v1_over) in SETS.items():
    root = ROOT / f"out_regress_{name}"
    for clip in CLIPS:
        src = ROOT / "out" / clip
        truth = json.load(open(ROOT / "data" / f"{clip}_truth.json"))
        tr = np.genfromtxt(src / "track.csv", delimiter=",", skip_header=1)
        cfg = Config(ball_colour="white", **ev_over)
        ev = detect_events(tr[:, :5], Table.load(ROOT / "data" / f"{clip}_table.json"), cfg, float(truth["fps"]))
        pts = segment_points_v1(ev, V1Config(**v1_over), tr[:, :5])
        d = root / clip; d.mkdir(parents=True, exist_ok=True)
        (d / "rallies.json").write_text(json.dumps(pts, indent=1))
        with open(d / "events.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ev[0].keys())); w.writeheader(); w.writerows(ev)
    for group, clips in (("dev (games)", CLIPS[:5]), ("held-out (tests)", CLIPS[5:])):
        r = subprocess.run([sys.executable, "-m", "tt_scout.metrics", "--out-root", str(root), "--clips"] + [f"data/{c}" for c in clips],
                           cwd=ROOT, capture_output=True, text=True)
        lines = [l for l in r.stdout.splitlines() if "POOLED" in l.upper() or "pooled" in l]
        print(f"--- {name} {group}"); print("\n".join(lines[-6:]) if lines else r.stdout[-1500:] + r.stderr[-800:])
