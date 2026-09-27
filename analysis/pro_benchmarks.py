"""Benchmarks for the coach's critique, measured on the professional OpenTTGames test matches with exactly the measurements the
critique uses on our own recordings (technique.py): speed off the racket, topspin dip, height over the net, contact timing, knee bend,
trunk lean, where the feet are. Writes tt_scout/benchmarks.json.
    python analysis/pro_benchmarks.py
OpenTTGames (OSAI) is CC BY-NC-SA 4.0: these medians are used as reference points in a research tool, not sold."""
import json, csv, pathlib, sys
import numpy as np, cv2
ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from tt_scout.table import Table
from tt_scout.camera import Camera
from tt_scout.points_v1 import segment_points_v1
from tt_scout import pose as P, technique
from tt_scout.heads import standing_zone


def load(clip):
    t = Table.load(ROOT / f"data/{clip}_table.json")
    cap = cv2.VideoCapture(str(ROOT / f"data/{clip}.mp4")); W_, H_, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5); cap.release()
    cam = Camera.from_table_pnp(t, W_, H_)
    events = [dict(frame=int(float(r["frame"])), t=float(r["t"]), kind=r["kind"], side=r["side"], x_px=float(r["x_px"]), y_px=float(r["y_px"]),
                   x_m=float(r["x_m"]), y_m=float(r["y_m"]), strength=float(r["strength"]), track=int(float(r["track"])))
              for r in csv.DictReader(open(ROOT / f"out/{clip}/events.csv"))]
    track = np.genfromtxt(ROOT / f"out/{clip}/track.csv", delimiter=",", skip_header=1)
    pts = segment_points_v1(events, None, track[:, :5])
    for p in pts:
        p["near_player"], p["far_player"] = "near", "far"
        p["server"] = p.get("serve_side")
    A = None
    if (ROOT / f"out/{clip}/pose.csv").exists():
        A, _ = P.clean_legs(P.assign(P.load(ROOT / f"out/{clip}/pose.csv"), t, standing_zone(t, 1.0)), t)
    return pts, events, track, fps, cam, A


def summary(shots):
    rally = [s for s in shots if not s["serve"]]
    def med(key, ss):
        v = [s[key] for s in ss if s.get(key) is not None]
        return (round(float(np.median(v)), 2), len(v)) if v else (None, 0)
    spins = [s["spin"] for s in rally if s.get("spin") is not None]
    return dict(rally_speed=med("speed", rally), serve_speed=med("speed", [s for s in shots if s["serve"]]),
                dip_share=(round(float(np.mean(np.array(spins) < -3)), 2), len(spins)) if spins else (None, 0),
                over_net=med("over_net", rally), timing=med("timing", rally), knee=med("knee", rally), lean=med("lean", rally),
                stance=med("stance", rally), depth=med("depth", rally))


if __name__ == "__main__":
    out = {"source": "OpenTTGames test_2, test_3, test_4 (professional matches, 120 fps), tt_scout measurements", "clips": {}}
    pooled = []
    for clip in ("test_2", "test_3", "test_4"):
        pts, events, track, fps, cam, A = load(clip)
        shots, _ = technique.measure(pts, events, track, fps, cam, A)
        out["clips"][clip] = summary(shots); pooled += shots
        print(clip, out["clips"][clip])
    out["pooled"] = summary(pooled)
    print("pooled", out["pooled"])
    (ROOT / "tt_scout" / "benchmarks.json").write_text(json.dumps(out, indent=1))
    # the pros' measured rally shots in 3D, as the reference drawn in the after-match analysis (analysis3d.py); OpenTTGames-derived:
    # CC BY-NC-SA 4.0, used here as a reference in a non-commercial tool
    keep = [dict(shot=s_["shot"], end=s_["end"], contact=s_["contact"], path=s_["path"], feet=s_["feet"], timing=s_["timing"],
                 speed=s_["speed"], incoming=s_.get("incoming"))
            for s_ in pooled if not s_["serve"] and s_.get("contact")]
    (ROOT / "tt_scout" / "pro_shots.json").write_text(json.dumps(dict(source=out["source"], licence="CC BY-NC-SA 4.0 (OpenTTGames, OSAI)",
                                                                      shots=keep), separators=(",", ":")))
    print(len(keep), "pro rally shots saved for the 3D reference")
