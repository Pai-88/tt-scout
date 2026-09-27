"""Table corners for a grey table with white edge lines (Cornilleau-style), which tt_scout.autocal cannot see (it knows blue
and green) and analysis/fit_light_table.py over-grows (its line-growing step walks into the white apron under the near edge).
Used on our own recordings (data/own/).

    python analysis/fit_grey_table.py data/own/IMG_3144.mp4 --out data/own/IMG_3144_table.json --check data/own/IMG_3144_corners.png

Method, on the empty-hall plate (median of frames spread over the clip, so players vanish):
  1. the playing surface = the largest grey, unsaturated blob in the lower part of the frame, opened hard enough that the
     net posts no longer bridge it to the table behind; its Huber line fits give the INNER quad (inside the white lines);
  2. the rules measure 2.74 x 1.525 m to the OUTER edge of the white lines, so along each edge the white line is found by
     walking outward from the inner edge along its normal, and the outer boundary of the white run is fitted with a line;
  3. corners = intersections of the four outer lines. Near the net (the post covers the lines) samples are skipped.
"""
import argparse, json, pathlib, sys
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from tt_scout.heads import sample_frames, plate_of
from tt_scout.autocal import quad_of, refine_by_lines, intersect, outer_lines


def surface_core(plate, sat_max=60, val=(140, 214), hue=(8, 50), open_px=15):
    hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
    h, s, v = (hsv[..., i].astype(int) for i in range(3))
    core = ((s <= sat_max) & (v >= val[0]) & (v <= val[1]) & (h >= hue[0]) & (h <= hue[1])).astype(np.uint8)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((open_px, open_px), np.uint8))
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, np.ones((9, 61), np.uint8))       # bridge the net gap
    H, W = core.shape
    n, lab, st, cent = cv2.connectedComponentsWithStats(core, 8)
    cands = [i for i in range(1, n) if cent[i][1] > H * 0.45 and st[i, 2] > W * 0.25]
    if not cands:
        return None
    return (lab == max(cands, key=lambda i: st[i, 4])).astype(np.uint8) * 255


def fit(plate, seed=None):
    if seed is not None:                                           # rough corners read off the plate by eye
        seed = np.array(seed, float).reshape(4, 2)
        outer, samples = outer_lines(plate, seed, net_x=float(seed[:, 0].mean()))
        return (outer, seed, "ok" if outer is not None else "white edge lines not found near the seed")
    core = surface_core(plate)
    if core is None:
        return None, None, "no grey surface found"
    q = quad_of(core)
    if q is None:
        return None, None, "the grey surface is not a quadrilateral"
    inner = refine_by_lines(core, q)
    outer, samples = outer_lines(plate, inner, net_x=float(inner[:, 0].mean()))
    if outer is None:
        return None, inner, "white edge lines not found"
    return outer, inner, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--out", required=True); ap.add_argument("--check"); ap.add_argument("--frames", type=int, default=41)
    ap.add_argument("--seed", help="rough corners x1,y1,...,x4,y4 (1 bottom-left, 2 top-left, 3 top-right, 4 bottom-right) read off the plate")
    ap.add_argument("--plate-out", help="also save the empty-hall plate")
    a = ap.parse_args()
    plate = plate_of(sample_frames(a.video, (1920, 1080), a.frames))
    if a.plate_out:
        cv2.imwrite(a.plate_out, plate)
    q, inner, why = fit(plate, [float(v) for v in a.seed.split(",")] if a.seed else None)
    if q is None:
        raise SystemExit(why)
    json.dump({"corners_px": [[float(x), float(y)] for x, y in q], "view": "side",
               "source": f"analysis/fit_grey_table.py on {pathlib.Path(a.video).name}: " + (f"rough corners by eye {a.seed}, " if a.seed else
                         "grey-surface blob on the empty-hall plate, ") + "outer edge of the table top found along each edge's normal on "
                         "the empty-hall plate, Huber line fits, intersections"},
              open(a.out, "w"), indent=1)
    print("corners:", [[round(float(x), 1), round(float(y), 1)] for x, y in q], "seed/inner-to-outer px:",
          np.abs(q - inner).max(axis=1).round(1).tolist(), "->", a.out)
    if a.check:
        vis = plate.copy()
        cv2.polylines(vis, [np.round(q * 4).astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 1, cv2.LINE_AA, 2)
        tiles = []
        for (x, y) in q:
            x, y = int(round(x)), int(round(y))
            c = vis[max(0, y - 30):y + 30, max(0, x - 30):x + 30]
            tiles.append(cv2.resize(c, (240, 240), interpolation=cv2.INTER_NEAREST))
        top = np.hstack(tiles)
        crop = vis[int(q[:, 1].min()) - 120:int(q[:, 1].max()) + 60, int(q[:, 0].min()) - 60:int(q[:, 0].max()) + 60]
        crop = cv2.resize(crop, (top.shape[1], int(crop.shape[0] * top.shape[1] / crop.shape[1])))
        cv2.imwrite(a.check, np.vstack([top, crop]))


if __name__ == "__main__":
    main()
