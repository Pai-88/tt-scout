"""The report's opening picture: one real frame from the recording, the longest rally at full stretch, with both players' skeletons
and the ball's last flight inked on.  Nothing is generated: it is the footage with the tracking drawn on.
"""
import numpy as np, cv2
from .pose import DRAW_BONES, FACE, J, MIN_C

NEAR_BGR, FAR_BGR, INK = (60, 170, 255), (255, 200, 60), (96, 32, 34)       # indigo ink, no black
YELLOW_BGR = (51, 221, 255)


def pick(points, track, assigned, fps):
    """(point, frame) for the picture: the rally with the most shots, at the frame nearest its middle where both players have a
    skeleton and the ball is tracked in the air; None if no frame qualifies."""
    def whole(a):                                                    # joints found, of the 15 that are not eyes and ears
        return 0 if a is None else int(sum(a[J[n], 2] >= MIN_C for n in ("nose", "neck", "lsho", "rsho", "lelb", "relb", "lwri", "rwri",
                                                                         "root", "lhip", "rhip", "lkne", "rkne", "lank", "rank")))
    for p in sorted(points, key=lambda p: -(p.get("n_crossings") or 0))[:5]:
        f0, f1 = int(p["start_t"] * fps), int(p["end_t"] * fps)
        mid = (f0 + f1) / 2; best = None
        for f in range(f0, f1):
            d = assigned.get(f) or {}
            if f >= len(track) or np.isnan(track[f, 2]):
                continue
            k = min(whole(d.get("near")), whole(d.get("far")))
            score = k - abs(f - mid) / max(1.0, (f1 - f0) / 2) * 2          # both bodies complete first, then near the middle
            if k >= 11 and (best is None or score > best[0]):
                best = (score, f)
        if best:
            return p, best[1]
    return None


def _skeleton(img, a, col, s=1.0):
    for p, q in DRAW_BONES:                                          # nothing is drawn on a face
        A, B = a[J[p]], a[J[q]]
        if A[2] >= MIN_C and B[2] >= MIN_C:
            p0, p1 = (int(A[0] * s), int(A[1] * s)), (int(B[0] * s), int(B[1] * s))
            cv2.line(img, p0, p1, INK, 7, cv2.LINE_AA); cv2.line(img, p0, p1, col, 3, cv2.LINE_AA)
    for i in range(len(a)):
        if a[i, 2] >= MIN_C and i not in {J[n] for n in FACE}:
            c = (int(a[i, 0] * s), int(a[i, 1] * s))
            cv2.circle(img, c, 6, INK, -1, cv2.LINE_AA); cv2.circle(img, c, 3, YELLOW_BGR, -1, cv2.LINE_AA)


def render(video, track, assigned, table, fps, frame, out_path, width=1600, aspect=2.3):
    """Draw and save the picture; returns (x0, y0, x1, y1) of the crop in the source frame, or None."""
    cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES, frame); ok, img = cap.read(); cap.release()
    if not ok:
        return None
    H, W = img.shape[:2]
    d = assigned.get(frame) or {}
    tid = track[frame, 4]
    seg = track[max(0, frame - int(0.35 * fps)):frame + 1]
    seg = seg[~np.isnan(seg[:, 2]) & (seg[:, 4] == tid)]
    for k in range(1, len(seg)):                                     # the ball's last flight, fading in
        a = k / len(seg)
        p0, p1 = tuple(int(v) for v in seg[k - 1, 2:4]), tuple(int(v) for v in seg[k, 2:4])
        cv2.line(img, p0, p1, INK, 3 + int(5 * a), cv2.LINE_AA); cv2.line(img, p0, p1, (90, 200, 255), 1 + int(3 * a), cv2.LINE_AA)
    if len(seg):
        c = (int(seg[-1, 2]), int(seg[-1, 3]))
        cv2.circle(img, c, 11, INK, 4, cv2.LINE_AA); cv2.circle(img, c, 11, YELLOW_BGR, 2, cv2.LINE_AA)
    for end, col in (("near", NEAR_BGR), ("far", FAR_BGR)):
        if d.get(end) is not None:
            _skeleton(img, d[end], col)
    # crop round the action: both players, the table, the ball; then widen to the aspect
    pts = [table.corners]
    for end in ("near", "far"):
        a = d.get(end)
        if a is not None:
            pts.append(a[a[:, 2] >= MIN_C, :2])
    pts = np.vstack(pts)
    x0, y0 = pts.min(axis=0); x1, y1 = pts.max(axis=0)
    padx, pady = 0.08 * (x1 - x0), 0.14 * (y1 - y0)
    x0, x1, y0, y1 = max(0, x0 - padx), min(W, x1 + padx), max(0, y0 - pady), min(H, y1 + pady)
    cw, ch = x1 - x0, y1 - y0
    if cw / ch < aspect:                                            # too tall: widen, then trim height if the frame runs out
        grow = ch * aspect - cw; x0, x1 = x0 - grow / 2, x1 + grow / 2
        if x0 < 0: x1 -= x0; x0 = 0
        if x1 > W: x0 -= x1 - W; x1 = W
        x0 = max(0, x0)
        ch = (x1 - x0) / aspect; cy = (y0 + y1) / 2; y0, y1 = cy - ch / 2, cy + ch / 2
    else:
        grow = cw / aspect - ch; y0, y1 = y0 - grow / 2, y1 + grow / 2
    if y0 < 0: y1 -= y0; y0 = 0
    if y1 > H: y0 -= y1 - H; y1 = H
    x0, y0, x1, y1 = (int(round(v)) for v in (max(0, x0), max(0, y0), min(W, x1), min(H, y1)))
    crop = img[y0:y1, x0:x1]
    out = cv2.resize(crop, (width, int(round(width * crop.shape[0] / crop.shape[1]))), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), out, [cv2.IMWRITE_JPEG_QUALITY, 84, cv2.IMWRITE_JPEG_PROGRESSIVE, 1])
    return x0, y0, x1, y1


def contact_frames(video, hits, assigned, fps, points, names, out_dir, per_player=4, size=(300, 400)):
    """Real frames of each player at the moment of a hit, from his deepest knee bend to his straightest, skeleton and the measured knee
    angle inked on. Returns {name: [dict(src, knee, t, point)]}. hits = pose.at_hits(); names = {"near": .., "far": ..}."""
    import math
    from .pose import skeleton_near, knees, hitter_name
    out_dir = __import__("pathlib").Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video)); res = {}
    for pi, nm in enumerate(dict.fromkeys([names["near"], names["far"]])):
        mine = sorted([h for h in hits if h.get("knee") is not None and hitter_name(h, points, names) == nm], key=lambda h: h["knee"])
        if len(mine) < per_player:
            continue
        idx = sorted(set(int(round(i * (len(mine) - 1) / (per_player - 1))) for i in range(per_player)))
        tiles = []
        for k, i in enumerate(idx):
            h = mine[i]
            a = skeleton_near(assigned, h["t"], h["side"], fps, reach=6)
            if a is None:
                continue
            f = int(round(h["t"] * fps)); cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = cap.read()
            if not ok:
                continue
            col = NEAR_BGR if h["side"] == "near" else FAR_BGR
            _skeleton(img, a, col)
            kn = knees(a)
            if kn:
                s_ = min(kn, key=kn.get)
                hp, kp, ap = (a[J[s_ + j], :2] for j in ("hip", "kne", "ank"))
                a1 = math.degrees(math.atan2(hp[1] - kp[1], hp[0] - kp[0])); a2 = math.degrees(math.atan2(ap[1] - kp[1], ap[0] - kp[0]))
                lo, hi = sorted((a1, a2))
                if hi - lo > 180:
                    lo, hi = hi, lo + 360
                cv2.ellipse(img, (int(kp[0]), int(kp[1])), (34, 34), 0, lo, hi, INK, 7, cv2.LINE_AA)
                cv2.ellipse(img, (int(kp[0]), int(kp[1])), (34, 34), 0, lo, hi, YELLOW_BGR, 3, cv2.LINE_AA)
            ok_ = a[:, 2] >= MIN_C
            if ok_.sum() < 6:                                            # too little of the body left to frame (legs removed as unsure)
                continue
            x0, y0 = a[ok_, :2].min(axis=0); x1, y1 = a[ok_, :2].max(axis=0)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            hgt = max(y1 - y0, (x1 - x0) * size[1] / size[0]) * 1.25
            wid = hgt * size[0] / size[1]
            X0, Y0 = int(max(0, cx - wid / 2)), int(max(0, cy - hgt / 2))
            X1, Y1 = int(min(img.shape[1], X0 + wid)), int(min(img.shape[0], Y0 + hgt))
            X0, Y0 = max(0, X1 - int(wid)), max(0, Y1 - int(hgt))
            if X1 - X0 < 20 or Y1 - Y0 < 20:
                continue
            crop = cv2.resize(img[Y0:Y1, X0:X1], size, interpolation=cv2.INTER_AREA)
            name = f"contact_{pi + 1}_{k + 1}.jpg"
            cv2.imwrite(str(out_dir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 82, cv2.IMWRITE_JPEG_PROGRESSIVE, 1])
            p = next((p for p in points if p["start_t"] - 0.5 <= h["t"] <= p["end_t"] + 0.5), None)
            tiles.append(dict(src=name, knee=round(h["knee"]), t=round(h["t"], 2), point=p["id"] if p else None))
        res[nm] = tiles
    cap.release()
    return res
