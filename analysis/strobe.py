"""Stroboscopic composite of a time window: every frame's moving pixels stacked into one picture, coloured by time
(blue = start, red = end), so a ball's whole path shows as a trail of dots and a bounce as a V. Built from raw pixels
(frame minus the median of the window), not from the tracker, so it can be used to check the tracker.
    python analysis/strobe.py data/own/IMG_3144.mp4 59.2 60.8 --out /tmp/s.png [--crop 150,380,1770,900] [--scale 1.0]
"""
import argparse
import numpy as np, cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("t0", type=float); ap.add_argument("t1", type=float); ap.add_argument("--out", required=True)
    ap.add_argument("--crop", default="150,380,1770,900"); ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--thresh", type=float, default=45.0); ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--ticks", type=float, default=0.1, help="a white tick label every this many seconds along the colour bar")
    ap.add_argument("--panels", type=int, default=1, help="split the window into this many consecutive panels, stacked top to bottom, "
                    "each labelled with its time span (direction and order then need no colour reading)")
    a = ap.parse_args()
    x0, y0, x1, y1 = (int(v) for v in a.crop.split(","))
    cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS)
    f0, f1 = int(round(a.t0 * fps)), int(round(a.t1 * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    frames = []
    for f in range(f0, f1 + 1):
        ok, fr = cap.read()
        if not ok:
            break
        if (f - f0) % a.every == 0:
            frames.append(fr[y0:y1, x0:x1].astype(np.int16))
    st = np.stack(frames)                                   # T x H x W x 3
    bg = np.median(st, axis=0).astype(np.int16)
    if a.panels > 1:
        diff_all = np.abs(st - bg[None]).sum(axis=3)
        T = len(frames); rows = []
        for k in range(a.panels):
            i0, i1 = k * T // a.panels, (k + 1) * T // a.panels
            d = diff_all[i0:i1]; best = d.argmax(axis=0); strength = d.max(axis=0)
            hue = (120 - 120 * best / max(i1 - i0 - 1, 1)).astype(np.uint8)     # within the panel: blue = its start, red = its end
            col = cv2.cvtColor(np.dstack([hue, np.full_like(hue, 255), np.full_like(hue, 255)]), cv2.COLOR_HSV2BGR)
            out = (bg * 0.55).astype(np.uint8); mv = strength > a.thresh; out[mv] = col[mv]
            if a.scale != 1.0:
                out = cv2.resize(out, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_NEAREST)
            t_a = a.t0 + i0 * a.every / fps; t_b = a.t0 + (i1 - 1) * a.every / fps
            lab = f"{k + 1}/{a.panels}  {t_a:.2f}-{t_b:.2f} s  (blue = start of panel, red = end)"
            cv2.putText(out, lab, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4); cv2.putText(out, lab, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            rows.append(out); rows.append(np.full((6, out.shape[1], 3), 255, np.uint8))
        cv2.imwrite(a.out, np.vstack(rows[:-1])); print(a.out, T, "frames in", a.panels, "panels"); return
    diff = np.abs(st - bg[None]).sum(axis=3)                # T x H x W
    best = diff.argmax(axis=0); strength = diff.max(axis=0)
    T = len(frames)
    # colour by time on a hue ramp (blue -> cyan -> green -> yellow -> red)
    hue = (120 - 120 * best / max(T - 1, 1)).astype(np.uint8)
    col = cv2.cvtColor(np.dstack([hue, np.full_like(hue, 255), np.full_like(hue, 255)]), cv2.COLOR_HSV2BGR)
    out = (bg * 0.55).astype(np.uint8)
    moving = strength > a.thresh
    out[moving] = col[moving]
    if a.scale != 1.0:
        out = cv2.resize(out, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_NEAREST)
    # colour bar with time ticks
    W = out.shape[1]; bar = np.zeros((34, W, 3), np.uint8)
    for x in range(W):
        h = int(120 - 120 * x / max(W - 1, 1))
        bar[:14, x] = cv2.cvtColor(np.uint8([[[h, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    dur = (T - 1) * a.every / fps
    k = 0
    while k * a.ticks <= dur + 1e-6:
        x = int(k * a.ticks / max(dur, 1e-6) * (W - 1))
        cv2.line(bar, (x, 0), (x, 18), (255, 255, 255), 1)
        cv2.putText(bar, f"{a.t0 + k * a.ticks:.1f}", (max(0, x - 14), 31), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        k += 1
    cv2.imwrite(a.out, np.vstack([out, bar]))
    print(a.out, T, "frames")


if __name__ == "__main__":
    main()
