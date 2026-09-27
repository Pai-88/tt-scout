"""Table corners for a table that tt_scout.autocal cannot see: a light (cream, grey, yellow) surface instead of blue or green.


    python analysis/fit_light_table.py data/yt_bundesliga.mp4 --out data/yt_bundesliga_table.json --check /tmp/check.png

The camera never moves, so the median of frames spread over the clip is the empty hall. In that plate the surface is the large
light blob in the lower half; the white edge lines belong to the playing surface (2.74 x 1.525 m is measured to their outer
edge), so the blob is grown into the bright, unsaturated pixels around it before the four edges are fitted with the same Huber
line fit the automatic calibration uses, and the corners are their intersections.
"""
import argparse, json, pathlib, sys
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from tt_scout.heads import sample_frames, plate_of
from tt_scout.autocal import quad_of, refine_by_lines


def fit(plate, hue=(12, 35), sat=(60, 170), val=130, line_val=120, line_sat=90, grow=9):
    hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    core = ((h >= hue[0]) & (h <= hue[1]) & (s >= sat[0]) & (s <= sat[1]) & (v >= val)).astype(np.uint8)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    n, lab, st, cent = cv2.connectedComponentsWithStats(core, 8)
    cands = [i for i in range(1, n) if cent[i][1] > plate.shape[0] * 0.4 and st[i, 2] > plate.shape[1] * 0.25]
    if not cands:
        return None, None
    surface = (lab == max(cands, key=lambda i: st[i, 4])).astype(np.uint8)
    near = cv2.dilate(surface, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))          # a band round the surface
    lines = ((v >= line_val) & (s <= line_sat)).astype(np.uint8) & near                   # the white edge lines in that band
    full = cv2.morphologyEx(surface | lines, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) * 255
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats((full > 0).astype(np.uint8), 8)
    full = ((lab2 == max(range(1, n2), key=lambda i: st2[i, 4])).astype(np.uint8)) * 255
    q = quad_of(full)
    return (None, full) if q is None else (refine_by_lines(full, q), full)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--out", required=True); ap.add_argument("--check"); ap.add_argument("--frames", type=int, default=41)
    a = ap.parse_args()
    plate = plate_of(sample_frames(a.video, (1920, 1080), a.frames))
    q, mask = fit(plate)
    if q is None:
        raise SystemExit("no light table found")
    json.dump({"corners_px": [[float(x), float(y)] for x, y in q], "view": "side",
               "source": f"analysis/fit_light_table.py on {pathlib.Path(a.video).name}: light-surface segmentation of the empty-hall "
                         "plate grown into the white edge lines, Huber line fits, corner intersections"},
              open(a.out, "w"), indent=1)
    print("corners:", [[round(float(x), 1), round(float(y), 1)] for x, y in q], "->", a.out)
    if a.check:
        vis = plate.copy()
        cv2.polylines(vis, [q.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 1)
        cv2.imwrite(a.check, vis)


if __name__ == "__main__":
    main()
