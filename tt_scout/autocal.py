"""Zero-click table calibration: find the table in the video itself and return its four corners.

A playing surface is a large, solid, saturated blue or green quadrilateral in the lower-middle of the frame. Per frame the
best such blob is found by colour and re-thresholded on its own median colour; a pixel counts as table when it is
table-coloured in at least a quarter of the sampled frames (players move, the table does not). Corners are the intersections
of robust line fits to the four edges of that voted mask, which recovers a corner a player hides most of the time (the geometry functions are shared with openttgames.py, where the same
method was checked on zoomed crops: 47 of 48 corners within ~2 px).

    corners, info = auto_table("match.mp4")          # corners: 1 bottom-left, 2 top-left, 3 top-right, 4 bottom-right
Left end of the image = "near" (x = 0), as everywhere else in tt_scout. Side view only.
"""
import numpy as np, cv2

HUE_RANGES = {"blue": (95, 135), "green": (40, 90)}


def line_dist(pts, a, b):
    d = b - a
    return np.abs((pts[:, 0] - a[0]) * d[1] - (pts[:, 1] - a[1]) * d[0]) / max(float(np.hypot(*d)), 1e-9)


def intersect(l1, l2):
    (x1, y1, u1, v1), (x2, y2, u2, v2) = l1, l2
    den = u1 * v2 - v1 * u2
    if abs(den) < 1e-9:
        return None
    t = ((x2 - x1) * v2 - (y2 - y1) * u2) / den
    return np.array([x1 + t * u1, y1 + t * v1])


def quad_of(mask):
    """Largest blob -> convex hull -> 4-point polygon, corners ordered 1 BL, 2 TL, 3 TR, 4 BR; None if it is not a quad."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    eps = 0.005 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, eps, True)
    while len(approx) > 4:
        eps *= 1.25; approx = cv2.approxPolyDP(hull, eps, True)
    pts = approx.reshape(-1, 2).astype(float)
    if len(pts) != 4 or pts[:, 0].max() - pts[:, 0].min() < 0.3 * mask.shape[1]:
        return None
    left = sorted(sorted(pts, key=lambda p: p[0])[:2], key=lambda p: -p[1])
    right = sorted(sorted(pts, key=lambda p: p[0])[2:], key=lambda p: p[1])
    return np.array(left + right)


def refine_by_lines(mask, quad, iters=4, trim_px=12.0):
    """Corners as intersections of robust (Huber) line fits to the four edges of the blob boundary."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return quad
    pts = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    q = quad.astype(np.float64).copy()
    for it in range(iters):
        d = np.stack([line_dist(pts, q[i], q[(i + 1) % 4]) for i in range(4)], axis=1)
        lab = d.argmin(axis=1)
        lines = []
        for i in range(4):
            sel = pts[lab == i]
            if it > 0:
                sel = sel[d[lab == i, i] <= trim_px]
            if len(sel) < 30:
                return quad
            vx, vy, x0, y0 = cv2.fitLine(sel.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            lines.append((float(x0), float(y0), float(vx), float(vy)))
        new = [intersect(lines[(i - 1) % 4], lines[i]) for i in range(4)]
        if any(c is None for c in new):
            return quad
        q = np.array(new)
    return q


def outer_lines(plate, inner, net_x, sat_max=60, val_min=150, reach=16, net_skip=40, step=2):
    """For each edge of a rough quad, walk outward along the normal from a few px inside it. The table top (grey surface and
    white lines) is bright and unsaturated; just outside it is the dark edge band (near side) or the saturated wooden floor, so
    the edge is the last table-top pixel before three non-table-top pixels in a row. A Huber line through those points, refitted
    once without the points more than 1.5 px off it, is the outer edge of the white line."""
    hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
    top = (hsv[..., 1].astype(int) <= sat_max) & (hsv[..., 2].astype(int) >= val_min)
    H, W = top.shape
    c = inner.mean(axis=0)
    lines, samples = [], []
    for i in range(4):
        a, b = inner[i], inner[(i + 1) % 4]
        d = (b - a) / np.linalg.norm(b - a)
        nrm = np.array([-d[1], d[0]])
        if np.dot(nrm, (a + b) / 2 - c) < 0:
            nrm = -nrm                                             # point away from the table's centre
        L = np.linalg.norm(b - a)
        pts = []
        for t in np.arange(10, L - 10, step):
            p0 = a + d * t
            if abs(p0[0] - net_x) < net_skip and abs(d[0]) > abs(d[1]):
                continue                                           # the net post hides the long lines here
            last_in, off = None, 0
            for r in np.arange(-reach, reach, 0.25):
                q = p0 + nrm * r
                xi, yi = int(round(q[0])), int(round(q[1]))
                if not (0 <= xi < W and 0 <= yi < H):
                    break
                if top[yi, xi]:
                    last_in, off = q, 0
                elif last_in is not None:
                    off += 1
                    if off >= 12:                                  # 3 px of not-table-top: we are outside
                        break
            if last_in is not None and off >= 12:
                pts.append(last_in + nrm * 0.5)                   # the boundary lies half a pixel further out
        pts = np.array(pts, np.float32)
        if len(pts) < 20:
            return None, samples
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        dist = np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx)
        keep = pts[dist <= 1.5]
        if len(keep) >= 20:
            vx, vy, x0, y0 = cv2.fitLine(keep, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((float(x0), float(y0), float(vx), float(vy)))
        samples.append((len(pts), len(keep)))
    corners = [intersect(lines[(i - 1) % 4], lines[i]) for i in range(4)]
    return (None if any(q is None for q in corners) else np.array(corners)), samples


def _best_component(m, H, W):
    """Most table-like connected component of a binary mask: (blob, score) or None."""
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 61), np.uint8))      # the net cuts the table top in two halves: bridge it
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
    best = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 0.02 * H * W or w < 0.30 * W or w > 0.92 * W:
            continue
        if y <= 2 or not (0.30 * H <= cent[i][1] <= 0.92 * H):
            continue                                             # a backdrop touches the top of the frame; a table sits lower
        blob = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_area = cv2.contourArea(cv2.convexHull(max(cnts, key=cv2.contourArea)))
        solidity = area / max(hull_area, 1.0)                    # a table top is a solid convex quad; curtains and floors are not
        if solidity < 0.80 or w / max(h, 1) < 2.0:
            continue
        if best is None or solidity * area > best[1]:
            best = (blob, solidity * area)
    return best


def table_blob(frame):
    """(mask, colour_name, score) of the most table-like blob in one frame, or None.

    Many halls hang a blue backdrop behind a blue table, and by hue alone the two are one blob that touches the top of the
    frame. They differ in brightness and slightly in hue (OpenTTGames: table H 122 / V 174, curtain H 112 / V 120), so when
    the plain hue mask gives nothing the hue-masked pixels are split into 2 and then 3 colour clusters (k-means on hue and
    value) and each cluster is searched for the table shape."""
    H, W = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    best = None
    for name, (h0, h1) in HUE_RANGES.items():
        m = cv2.inRange(hsv, np.array([h0, 60, 35], np.uint8), np.array([h1, 255, 255], np.uint8))
        cands = [_best_component(m, H, W)]
        ys, xs = np.nonzero(m)
        if cands[0] is None and len(ys) > 5000:
            step = max(1, len(ys) // 60000)
            feat = np.float32(np.c_[hsv[ys[::step], xs[::step], 0] * 2.0, hsv[ys[::step], xs[::step], 2]])      # hue weighted x2
            for k in (2, 3):
                _, _, centres = cv2.kmeans(feat, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 2, cv2.KMEANS_PP_CENTERS)
                full = np.float32(np.c_[hsv[ys, xs, 0] * 2.0, hsv[ys, xs, 2]])
                lab = np.argmin(((full[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2), axis=1)
                for c in range(k):
                    mc = np.zeros((H, W), np.uint8); mc[ys[lab == c], xs[lab == c]] = 255
                    cands.append(_best_component(mc, H, W))
                if any(c is not None for c in cands):
                    break
        for c in cands:
            if c is not None and (best is None or c[1] > best[2]):
                best = (c[0], name, c[1])
    if best is None:
        return None
    blob, name, score = best
    core = cv2.erode(blob, np.ones((25, 25), np.uint8)) > 0
    if core.sum() < 200:
        return None
    med = np.median(hsv[core], axis=0)
    lo = np.array([max(0, med[0] - 6), max(0, med[1] - 80), max(0, med[2] - 40)], np.uint8)      # tight on hue and value: the backdrop is 10 hue / 50 value away
    hi = np.array([min(179, med[0] + 6), 255, min(255, med[2] + 60)], np.uint8)
    own = cv2.inRange(hsv, lo, hi)
    own[cv2.dilate(blob, np.ones((21, 21), np.uint8)) == 0] = 0
    own = cv2.morphologyEx(own, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return cv2.morphologyEx(own, cv2.MORPH_CLOSE, np.ones((9, 61), np.uint8)), name, score


def auto_table_from_frames(frames_iter, n_tried, vote_frac=0.25):
    """Core of auto_table on an iterable of (index, frame). Returns (corners or None, info)."""
    votes, used, colours, frame_out, fidx, mid = None, 0, [], None, None, n_tried // 2
    for k, (i, fr) in enumerate(frames_iter):
        r = table_blob(fr)
        if r is None:
            continue
        votes = (r[0] > 0).astype(np.uint16) if votes is None else votes + (r[0] > 0)
        used += 1; colours.append(r[1])
        if frame_out is None or k == mid:
            frame_out, fidx = fr, i
    info = dict(method="auto: colour blob vote over frames, Huber line fits", n_frames=used, n_tried=n_tried,
                colour=max(set(colours), key=colours.count) if colours else None)
    if votes is None or used < max(3, n_tried // 3):
        return None, dict(info, reason="no table-like blob found in enough frames")
    # a VOTE, not a union: the table is table-coloured in nearly every frame, a blue shirt at the table end only in a few
    # (a union over 30 frames put corners 85-190 px out on the two games with players in blue). A corner that is hidden most
    # of the time is cut off by the vote; the line fits below recover it from the long visible runs of its two edges.
    union = ((votes >= max(3, int(np.ceil(vote_frac * used)))) * 255).astype(np.uint8)
    union = cv2.morphologyEx(union, cv2.MORPH_OPEN, np.ones((21, 21), np.uint8))
    hull_quad = quad_of(union)
    if hull_quad is None:
        return None, dict(info, reason="the table blob is not a quadrilateral")
    corners = refine_by_lines(union, hull_quad)
    info.update(frame=fidx, hull_to_lines_px=[round(float(v), 1) for v in np.abs(corners - hull_quad).max(axis=1)])
    return corners, dict(info, frame_image=frame_out)


def auto_table(video, frames=30, vote_frac=0.25):
    """Corners of the table from `frames` frames spread over the video. Returns (corners or None, info).
    Graded 2026-09-17 against 48 hand-verified corners on the 12 OpenTTGames videos: 12/12 found, corner error median 1.3 px,
    p90 10 px, worst 32 px (a player standing at one end for the whole clip). Always look at table_check.png."""
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = sorted(set(int(i) for i in np.linspace(0.03 * n, 0.97 * n, frames))) if n > 0 else list(range(frames))

    def gen():
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if ok:
                yield i, fr
    out = auto_table_from_frames(gen(), len(idx), vote_frac)
    cap.release()
    return out
