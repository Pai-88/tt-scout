"""Benchmarks for the coach's critique, measured on the professional OpenTTGames test matches with exactly the measurements the
critique uses on our own recordings (technique.py): speed off the racket, topspin dip, height over the net, contact timing, knee bend,
trunk lean, where the feet are. Writes tt_scout/benchmarks.json.
    python analysis/pro_benchmarks.py
The knee is the ONE gated quantity (pose.py, top): the camera-near leg's picture angle on rally forehands seen in profile, gated by
the pros' 3D bodies placed from the cached out/<clip>/pose3d.jsonl (analysis/pro_bodies.py stage 1) exactly as our recordings are
(technique.gate_knees), plus knee_iqr and knee_ci (a 90% bootstrap interval of the pooled median: what critique.KNEE_MARGIN is derived
from). NOTE: the pros' cache was answered on crops built from the CLEANED skeletons (before body3d.requests took the legs as found,
2026-09-28); the request ids that the new boxes add are not in the cache and are simply unanswered (pose3d_cache in the json counts
them per clip). Re-run analysis/pro_bodies.py without --place-only to refresh the cache with the new boxes.
OpenTTGames (OSAI) is CC BY-NC-SA 4.0: these medians are used as reference points in a research tool, not sold."""
import json, csv, pathlib, sys
import numpy as np, cv2
ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from tt_scout.table import Table
from tt_scout.camera import Camera
from tt_scout.points_v1 import segment_points_v1
from tt_scout import pose as P, technique, body3d
from tt_scout.heads import standing_zone

CLIPS = ("test_2", "test_3", "test_4")


def load(clip):
    """(points, events, track, fps, cam, A, A_raw): A = the skeletons after pose.clean_legs, A_raw = as Vision found them (for the
    3D crop boxes, body3d.requests)."""
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
    A = A_raw = None
    if (ROOT / f"out/{clip}/pose.csv").exists():
        A = P.assign(P.load(ROOT / f"out/{clip}/pose.csv"), t, standing_zone(t, 1.0))
        A_raw = {f: dict(v) for f, v in A.items()}
        A, _ = P.clean_legs(A, t)
    return pts, events, track, fps, cam, A, A_raw


def pro_strokes(clip, shots, A_raw, fps, cam, step=4):
    """(strokes, cache note): the pros' bodies placed from the cached pose3d answers (out/<clip>/pose3d.jsonl, written by
    pro_bodies.py); [] without a cache. The cache was answered with the boxes of the day it was made (cleaned-skeleton crops, see the
    module docstring); only the request ids (point, shot, frame) must match, and the note says how many of today's requests it lacks."""
    work = ROOT / "out" / clip
    if not (work / "pose3d.jsonl").exists():
        return [], dict(answered=0, requested=0, unanswered=0, crops="none")
    lines, meta = body3d.requests(shots, A_raw, fps, step=step)
    raw = [json.loads(l) for l in (work / "pose3d.jsonl").read_text().splitlines() if l.strip()]
    ids = {l.split()[-1] for l in lines}; ans = {str(r.get("id")) for r in raw}
    note = dict(answered=len(ids & ans), requested=len(ids), unanswered=len(ids - ans), crops="cleaned skeletons (pre 2026-09-28 boxes)")
    return list(body3d.strokes_from(raw, meta, cam).values()), note


def summary(shots):
    rally = [s for s in shots if not s["serve"]]
    def med(key, ss):
        v = [s[key] for s in ss if s.get(key) is not None]
        return (round(float(np.median(v)), 2), len(v)) if v else (None, 0)
    spins = [s["spin"] for s in rally if s.get("spin") is not None]
    kn = [s["knee"] for s in rally if s.get("knee") is not None]
    return dict(rally_speed=med("speed", rally), serve_speed=med("speed", [s for s in shots if s["serve"]]),
                dip_share=(round(float(np.mean(np.array(spins) < -3)), 2), len(spins)) if spins else (None, 0),
                over_net=med("over_net", rally), timing=med("timing", rally), knee=med("knee", rally), lean=med("lean", rally),
                stance=med("stance", rally), depth=med("depth", rally),
                knee_iqr=[round(float(np.percentile(kn, 25)), 2), round(float(np.percentile(kn, 75)), 2)] if kn else [None, None],
                knee_ci=knee_ci(kn),
                knee_ungated=med("knee_raw", rally))     # the camera-near leg on every rally shot, for comparison only: not what the critique uses


def knee_ci(kn, reps=4000, seed=0):
    """[lo, hi]: a 90% bootstrap interval of the median of the gated knees (the reference's own uncertainty, critique.KNEE_MARGIN);
    [None, None] under 3 values."""
    if len(kn) < 3:
        return [None, None]
    rng = np.random.default_rng(seed); kn = np.asarray(kn, float)
    boot = np.array([np.median(rng.choice(kn, len(kn))) for _ in range(reps)])
    return [round(float(np.percentile(boot, 5)), 2), round(float(np.percentile(boot, 95)), 2)]


if __name__ == "__main__":
    out = {"source": "OpenTTGames test_2, test_3, test_4 (professional matches, 120 fps), tt_scout measurements", "clips": {},
           "knee_rule": "the camera-near leg's picture angle at contact, rally forehands, both legs found, only where the pros' 3D body "
                        f"(or, without one, the picture's hip width) shows the legs in profile: pelvis line within {90 - P.PROFILE_MIN:.0f} deg "
                        "of the line of sight (technique.gate_knees, pose.py)"}
    pooled = []
    for clip in CLIPS:
        pts, events, track, fps, cam, A, A_raw = load(clip)
        shots, _ = technique.measure(pts, events, track, fps, cam, A)
        strokes, cache = pro_strokes(clip, shots, A_raw, fps, cam)
        body3d.attach(shots, strokes, cam)
        technique.gate_knees(shots, strokes, cam)
        g = [s for s in shots if s.get("knee") is not None]
        print(clip, len(strokes), "strokes in 3D;", len(g), "of", sum(not s["serve"] for s in shots),
              "rally shots with a gated knee (", sum(s.get("profile_src") == "3d" for s in g), "by the 3D body ); cache", cache)
        out["clips"][clip] = summary(shots); out["clips"][clip]["pose3d_cache"] = cache; pooled += shots
        print(clip, out["clips"][clip])
    out["pooled"] = summary(pooled)
    if out["pooled"]["knee"][1] < 20:                                    # said in the file too: the critique's reference must own up to it
        out["knee_note"] = (f"THIN: the pros' gated knee rests on {out['pooled']['knee'][1]} forehand contacts seen in profile (the pros turn their "
                            "hips into a forehand, which the gate refuses); critique.KNEE_MARGIN is half the width of knee_ci")
        print("WARNING", out["knee_note"])
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
