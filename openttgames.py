"""Convert an OpenTTGames game (video + unzipped markup) into tt_scout inputs.

Usage: python openttgames.py data/test_2
Expects data/test_2.mp4 and data/test_2_markup/{ball_markup.json, events_markup.json, segmentation_masks/}
Writes  data/test_2_table.json  (corners derived from the table segmentation mask, refined at full res)
        data/test_2_truth.json  (bounce/net events with time and table coordinates in metres)
        data/test_2_table_check.png (frame with the fitted table outline, look at it)

Dataset: OpenTTGames, OSAI (Voeikov, Falaleev, Baikulov, "TTNet", CVPRW 2020), CC BY-NC-SA 4.0.
"""
import json, pathlib, sys
import numpy as np, cv2
from tt_scout.table import Table
from tt_scout.config import TABLE_LENGTH as L

FPS = 120


def _refined_mask(video, mpath):
    """Full-resolution mask of table-coloured pixels near this frame's segmentation mask, or None."""
    m = cv2.imread(str(mpath))
    cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES, int(mpath.stem))
    ok, frame = cap.read(); cap.release()
    if not ok or m is None:
        return None
    H, W = frame.shape[:2]
    # the table is the class whose largest blob is the widest thing in the mask
    best = None
    for c in range(3):
        ch = (m[..., c] > 127).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(ch)
        if n < 2:
            continue
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if best is None or stats[i, cv2.CC_STAT_WIDTH] > best[0]:
            best = (stats[i, cv2.CC_STAT_WIDTH], (lab == i).astype(np.uint8))
    if best is None:
        return None
    coarse = cv2.resize(best[1], (W, H), interpolation=cv2.INTER_NEAREST)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    core = cv2.erode(coarse, np.ones((31, 31), np.uint8)) > 0
    if core.sum() < 100:
        return None
    med = np.median(hsv[core], axis=0)
    lo = np.array([max(0, med[0] - 12), max(0, med[1] - 80), max(0, med[2] - 80)], np.uint8)   # loose enough to keep shadowed table; leaks are removed by the opening on the union
    hi = np.array([min(179, med[0] + 12), 255, 255], np.uint8)
    band = cv2.dilate(coarse, np.ones((21, 21), np.uint8))       # +-10 px: wide enough for the 6x mask upscale, too narrow to leak into a blue backdrop
    blue = cv2.inRange(hsv, lo, hi)
    blue[band == 0] = 0
    return cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)), frame


def _quad(mask):
    """Largest blob -> convex hull -> 4-point polygon (None if it is not a quad), corners ordered
    1 bottom-left, 2 top-left, 3 top-right, 4 bottom-right (left end = "near", x = 0)."""
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


def _line_dist(pts, a, b):
    """Perpendicular distance of points to the infinite line through a and b."""
    d = b - a
    n = np.hypot(*d)
    return np.abs((pts[:, 0] - a[0]) * d[1] - (pts[:, 1] - a[1]) * d[0]) / max(n, 1e-9)


def _intersect(l1, l2):
    (x1, y1, u1, v1), (x2, y2, u2, v2) = l1, l2
    den = u1 * v2 - v1 * u2
    if abs(den) < 1e-9:
        return None
    t = ((x2 - x1) * v2 - (y2 - y1) * u2) / den
    return np.array([x1 + t * u1, y1 + t * v1])


def _refine_by_lines(mask, quad, iters=4, trim_px=12.0):
    """Corners as intersections of robust line fits to the four edges of the blob boundary. A corner that the mask
    never shows (hidden behind a player in every frame) is cut by a chord in the hull; the long straight runs of
    the two adjacent edges still determine it. Edge i runs from corner i to corner i+1; corner i = edge i-1 x edge i."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return quad
    pts = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    q = quad.astype(np.float64).copy()
    for it in range(iters):
        d = np.stack([_line_dist(pts, q[i], q[(i + 1) % 4]) for i in range(4)], axis=1)
        lab = d.argmin(axis=1)
        lines = []
        for i in range(4):
            sel = pts[lab == i]
            if it > 0:
                sel = sel[d[lab == i, i] <= trim_px]              # after the first pass, drop chord and occlusion points
            if len(sel) < 30:
                return quad
            vx, vy, x0, y0 = cv2.fitLine(sel.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            lines.append((float(x0), float(y0), float(vx), float(vy)))
        new = [_intersect(lines[(i - 1) % 4], lines[i]) for i in range(4)]
        if any(c is None for c in new):
            return quad
        q = np.array(new)
    return q


def table_corners(video, masks_dir, frames=25):
    """Table corners from the UNION of table-coloured masks over `frames` frames spread across the clip.

    From a side camera the player standing at each end hides the far corner of that end in nearly every frame, so
    any single frame (and the median over frames) fits a quad with that corner pulled onto the player's body
    (test_4 corner 2, game_1 corner 3 on 2026-09-17). The corner is exposed whenever the player steps aside, so the
    union over many frames recovers the full table. Returns (corners, frame, frame_index, info)."""
    masks = sorted(masks_dir.glob("*.png"), key=lambda p: int(p.stem))
    idx = sorted(set(int(round(i)) for i in np.linspace(0, len(masks) - 1, min(frames, len(masks)))))
    union, used, single, frame_out, fidx = None, 0, [], None, None
    for i in idx:
        r = _refined_mask(video, masks[i])
        if r is None:
            continue
        mask, frame = r
        union = mask if union is None else cv2.bitwise_or(union, mask)
        used += 1
        q = _quad(mask)
        if q is not None:
            single.append(q)
        if fidx is None or i == idx[len(idx) // 2]:
            frame_out, fidx = frame, int(masks[i].stem)
    if union is None:
        raise SystemExit("no usable segmentation mask frames; click the corners with calibrate_table.py instead")
    # an opening removes thin protrusions of the union (a dark-blue shoe or a strip of backdrop curtain inside the band)
    # without touching the table itself; it rounds the corners by ~10 px, which the line fits ignore
    union = cv2.morphologyEx(union, cv2.MORPH_OPEN, np.ones((21, 21), np.uint8))
    hull_quad = _quad(union)
    if hull_quad is None:
        raise SystemExit(f"union of {used} mask frames is not a quad; click the corners with calibrate_table.py instead")
    corners = _refine_by_lines(union, hull_quad)
    info = dict(method="robust line fits to the edges of the union of table-colour masks over spread frames, corners = intersections",
                n_frames=used, n_masks=len(masks), hull_to_lines_px=[round(float(v), 1) for v in np.abs(corners - hull_quad).max(axis=1)])
    if single:
        dev = np.abs(np.array(single) - corners).max(axis=2)          # per frame, per corner
        info["single_frame_corner_dev_px"] = [round(float(v), 1) for v in np.median(dev, axis=0)]
    return corners, frame_out, fidx, info


def main(base):
    base = pathlib.Path(base)
    video, mk = base.with_suffix(".mp4"), pathlib.Path(str(base) + "_markup")
    corners, frame, fidx, info = table_corners(video, mk / "segmentation_masks")
    Table.save(str(base) + "_table.json", corners, "side", video=str(video), frame=fidx,
               source="OpenTTGames (OSAI, CC BY-NC-SA 4.0), corners = intersections of robust edge-line fits on the union of table-colour masks over spread frames", fit=info)
    table = Table.load(str(base) + "_table.json")
    cv2.polylines(frame, [corners.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
    for i, p in enumerate(corners):
        cv2.putText(frame, str(i + 1), (int(p[0]) + 6, int(p[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imwrite(str(base) + "_table_check.png", frame)

    ev = json.load(open(mk / "events_markup.json"))
    ball = json.load(open(mk / "ball_markup.json"))
    events = []
    for f, kind in sorted(ev.items(), key=lambda kv: int(kv[0])):
        if kind == "empty_event":
            continue
        f = int(f); e = dict(t=f / FPS, frame=f, kind=kind, side=None)
        b = ball.get(str(f))
        if b and b["x"] >= 0:
            xm, ym = table.to_table([(b["x"], b["y"])])[0]
            e.update(x_px=b["x"], y_px=b["y"], x_m=float(xm), y_m=float(ym), side="near" if xm < L / 2 else "far")
        events.append(e)
    truth = dict(source="OpenTTGames " + base.name + " (OSAI, CC BY-NC-SA 4.0)", fps=FPS, events=events,
                 ball={k: v for k, v in ball.items()})
    pathlib.Path(str(base) + "_truth.json").write_text(json.dumps(truth, indent=1))
    kinds = {}
    for e in events:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    print(f"corners: {corners.astype(int).tolist()}  fit: lines on the union of {info['n_frames']} frames; line fit moved the hull corners by "
          f"{info['hull_to_lines_px']} px; single frames deviate {info.get('single_frame_corner_dev_px')} px  events: {kinds}  ball-labelled frames: {len(ball)}")
    bx = [e["x_m"] for e in events if e["kind"] == "bounce" and "x_m" in e]
    by = [e["y_m"] for e in events if e["kind"] == "bounce" and "x_m" in e]
    print(f"labelled bounce positions in metres: x {min(bx):.2f}..{max(bx):.2f}  y {min(by):.2f}..{max(by):.2f}  (table is 0..2.74 x 0..1.525)")


if __name__ == "__main__":
    main(sys.argv[1])
