"""Frames of a time window laid out as a labelled grid, for checking by eye what happened (ground truth, not tracker output).

    python analysis/strip.py data/own/IMG_3144.mp4 67.0 70.0 --step 0.2 --out /tmp/s.png [--crop 380,430,1520,840] [--scale 0.5]
Optional --track out/<run>/track.csv draws the tracker's ball as a small ring (to check the tracker against the picture).
"""
import argparse
import numpy as np, cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("t0", type=float); ap.add_argument("t1", type=float)
    ap.add_argument("--step", type=float, default=0.2); ap.add_argument("--out", required=True)
    ap.add_argument("--crop", default="380,430,1520,840"); ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--cols", type=int, default=4); ap.add_argument("--track")
    a = ap.parse_args()
    x0, y0, x1, y1 = (int(v) for v in a.crop.split(","))
    cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS)
    tr = np.genfromtxt(a.track, delimiter=",", skip_header=1) if a.track else None
    tiles = []
    for t in np.arange(a.t0, a.t1 + 1e-6, a.step):
        f = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            break
        if tr is not None and f < len(tr) and not np.isnan(tr[f, 2]):
            cv2.circle(fr, (int(tr[f, 2]), int(tr[f, 3])), 14, (0, 0, 255), 2)
        c = cv2.resize(fr[y0:y1, x0:x1], None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
        lab = f"{t:.2f}s"
        cv2.putText(c, lab, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(c, lab, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        tiles.append(c)
    while len(tiles) % a.cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + a.cols]) for i in range(0, len(tiles), a.cols)]
    cv2.imwrite(a.out, np.vstack(rows))
    print(a.out, len(tiles), "tiles")


if __name__ == "__main__":
    main()
