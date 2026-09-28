"""The pros' bodies in 3D through their rally strokes, for the after-match viewer's "ghost" (body3d.py, tools/pose3d).
Stage 1 (slow, Vision on this Mac) writes out/<clip>/pose3d.jsonl; stage 2 (fast) places the bodies and writes
tt_scout/pro_bodies.json. OpenTTGames (OSAI) is CC BY-NC-SA 4.0: derived poses are a reference in a non-commercial tool.
    python analysis/pro_bodies.py [--place-only]
"""
import json, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from tt_scout import technique, body3d
sys.path.insert(0, str(ROOT / "analysis"))
from pro_benchmarks import load

CLIPS = ("test_2", "test_3", "test_4")
if __name__ == "__main__":
    place_only = "--place-only" in sys.argv
    strokes = []
    for clip in CLIPS:
        pts, events, track, fps, cam, A, A_raw = load(clip)
        shots, _ = technique.measure(pts, events, track, fps, cam, A)
        work = ROOT / "out" / clip
        lines, meta = body3d.requests(shots, A_raw, fps, step=4)          # boxes round the skeletons as found, legs and all
        (work / "pose3d_meta.json").write_text(json.dumps(meta))
        if place_only and (work / "pose3d.jsonl").exists():
            raw = [json.loads(l) for l in (work / "pose3d.jsonl").read_text().splitlines() if l.strip()]
        else:
            raw = body3d.run(ROOT / f"data/{clip}.mp4", lines, work, cam.f)
        st = body3d.strokes_from(raw, meta, cam)                          # flipped frames already left out: what the report's loader sees
        print(clip, len(shots), "shots,", len(st), "strokes in 3D")
        for s in st.values():
            strokes.append(dict(body3d.pack(s), clip=clip, m=body3d.measures(s, cam)))
    (ROOT / "tt_scout" / "pro_bodies.json").write_text(json.dumps(dict(
        source="OpenTTGames test_2, test_3, test_4 (professional matches, 120 fps), Apple Vision 3D body pose, tt_scout placement",
        licence="CC BY-NC-SA 4.0 (OpenTTGames, OSAI)", strokes=strokes), separators=(",", ":")))
    print(len(strokes), "pro strokes saved")
