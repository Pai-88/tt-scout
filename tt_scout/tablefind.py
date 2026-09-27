"""The table's four corners, found in the video by itself, whatever colour the table is.

Two methods, tried in order on frames spread over the recording:
  1. colour (autocal.auto_table_from_frames): a blue or green table found by its colour in every frame and voted on. Graded on
     the 12 OpenTTGames videos: 12 of 12 found, worst corner 32 px (a player standing at one end for the whole clip).
  2. surface (surface_table): for any colour, including a grey table under hall lights. On the empty-hall plate (the median of the
     frames, so players drop out), the table top is a large even patch with a white line all round it. Regions are grown from a
     grid of seeds with the white lines and strong edges as walls; the net and the centre line cut the top into pieces, so pieces
     that touch across a thin wall are joined for as long as the result stays one solid, convex shape (the wall or floor next to
     the table is not); the largest shape a plausible camera could see as a 2.74 x 1.525 m table wins, and its corners are snapped
     out to the outer edge of the white line. Graded on our 8 side-on recordings: 8 of 8 found, every corner within 1.3 px of the
     hand-fitted ones; the 3 filmed from behind a player are (rightly) not found.
Neither finds it: the corners are clicked by hand (the upload page asks, `tt-scout calibrate` on the command line).

    corners, info = find_table("match.mp4")                 # corners: 1 bottom-left, 2 top-left, 3 top-right, 4 bottom-right, or None
"""
import numpy as np, cv2
from .autocal import auto_table_from_frames, quad_of, refine_by_lines, outer_lines
from .config import TABLE_LENGTH as L, TABLE_WIDTH as TW


def walls(plate):
    """What a region may not grow across: thin bright lines (the white edge lines, the centre line, the net's top band) and strong edges."""
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY); hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    white = (th > 16) & (hsv[..., 1] < 80) & (hsv[..., 2] > 120)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 100) > 0
    return cv2.dilate((white | edges).astype(np.uint8), np.ones((3, 3), np.uint8))


def pieces(plate, tol=10):
    """Even regions grown from a grid of seeds over the middle and lower part of the picture, walled in by walls(); a region that
    reaches the frame's edge (floor, wall, ceiling) or is tiny or huge is dropped."""
    H, W = plate.shape[:2]
    lab = cv2.cvtColor(cv2.medianBlur(plate, 5), cv2.COLOR_BGR2LAB)
    wall = walls(plate)
    out = []

    def grow(sx, sy):
        if not (0 <= sx < W and 0 <= sy < H) or wall[sy, sx] or any(m[sy, sx] for m in out):
            return
        mask = np.zeros((H + 2, W + 2), np.uint8); mask[1:-1, 1:-1] = wall
        cv2.floodFill(lab, mask, (sx, sy), 0, (tol,) * 3, (tol,) * 3, 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
        m = mask[1:-1, 1:-1] == 255
        if not 0.002 < m.mean() < 0.45:
            return
        ys, xs = np.nonzero(m)
        if xs.min() <= 1 or xs.max() >= W - 2 or ys.min() <= 1 or ys.max() >= H - 2:
            return
        out.append(m)
    for fx in np.linspace(0.2, 0.8, 13):
        for fy in np.linspace(0.40, 0.90, 11):
            grow(int(fx * W), int(fy * H))
    for m in list(out):                                            # the rest of a table lies right next to a piece of it: the far half,
        ys, xs = np.nonzero(m)                                     # a thin band beyond the centre line, can fall between grid seeds
        for fx in (0.25, 0.5, 0.75):
            x = int(xs.min() + fx * (xs.max() - xs.min()))
            col = ys[xs == x] if (xs == x).any() else ys
            for y in (int(col.min()) - 8, int(col.max()) + 8):
                grow(x, y)
    return out


def solidity(m):
    """Area over convex-hull area, after closing gaps a few px wide (the net's band between the two halves)."""
    mu = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    pts = np.argwhere(mu > 0)[:, ::-1].astype(np.int32)
    return float((mu > 0).sum()) / max(1.0, cv2.contourArea(cv2.convexHull(pts))) if len(pts) > 2 else 0.0


def joined(ps, lab=None, gap=(45, 17), min_solid=0.88, max_de=20.0):
    """From each of the biggest pieces, add a touching piece (across a line, or across the net: hence the wide gap) of the same colour
    (Lab difference under max_de: one table top, not the dark apron under its edge) only while the result stays one solid convex shape."""
    order = sorted(range(len(ps)), key=lambda i: -ps[i].sum()); out = []
    col = [lab[m].mean(axis=0) if lab is not None else np.zeros(3) for m in ps]
    same = lambda a, b: float(np.linalg.norm(col[a] - col[b])) < max_de
    k = np.ones((gap[1], gap[0]), np.uint8)
    for s0 in order[:8]:
        cur = ps[s0].copy(); used = {s0}; changed = True
        while changed:
            changed = False
            near = cv2.dilate(cur.astype(np.uint8), k) > 0
            touching = [j for j in order if j not in used and same(s0, j) and (near & ps[j]).any()]
            for j in touching:
                u = cur | ps[j]
                if solidity(u) >= min_solid:
                    add = [j]
                else:                                              # one far quarter makes an L: try it with the piece that squares it
                    add = next(([j, q] for q in touching if q != j and solidity(u | ps[q]) >= min_solid), None)
                if add:
                    for q in add:
                        cur = cur | ps[q]; used.add(q)
                    changed = True; near = cv2.dilate(cur.astype(np.uint8), k) > 0
                    break
        if len(used) > 1:
            out.append(cur)
    return out


def plausible(corners, W, H):
    """Whether a camera where a phone would stand sees these four points as the table: (ok, camera position in metres, corner rms px)."""
    from .table import Table
    from .camera import Camera
    try:
        cam = Camera.from_table_pnp(Table([[float(x), float(y)] for x, y in corners], "side", {}), W, H)
    except Exception:
        return False, None, None
    x, y, z = (float(v) for v in cam.C); d = float(np.hypot(x - L / 2, y - TW / 2))
    return (0.2 < z < 5.0 and 1.2 < d < 15.0 and cam.corner_rms < 4.0), (round(x, 2), round(y, 2), round(z, 2)), round(float(cam.corner_rms), 2)


def surface_table(plate):
    """Corners of the table on an empty-hall plate by the surface method, or (None, reason)."""
    H, W = plate.shape[:2]
    ps = pieces(plate)
    lab = cv2.cvtColor(cv2.medianBlur(plate, 5), cv2.COLOR_BGR2LAB).astype(np.float32)
    best = None
    for m in ps + joined(ps, lab):
        mu = cv2.morphologyEx((m * 255).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        q = quad_of(mu)
        if q is None or solidity(m) < 0.85:
            continue
        c = refine_by_lines(mu, q)
        ok, pos, rms = plausible(c, W, H)
        if not ok:
            continue
        score = float((mu > 0).mean()) * solidity(m)
        if best is None or score > best[0]:
            best = (score, c, pos, rms)
    if best is None:
        return None, "no region that a camera could see as a table"
    _, inner, pos, rms = best
    outer, _ = outer_lines(plate, inner, net_x=float(inner[:, 0].mean()))
    snapped = outer is not None and np.abs(outer - inner).max() < 40
    return (outer if snapped else inner), dict(camera_m=pos, corner_rms_px=rms, snapped_to_white_line=bool(snapped))


def find_table_in_frames(frames):
    """frames: BGR images spread over the recording, all the same size. Returns (corners or None, info)."""
    if not frames:
        return None, dict(reason="no frames")
    c, info = auto_table_from_frames(enumerate(frames), len(frames))
    if c is not None:
        return c, dict(info, method="colour")
    plate = np.median(np.stack(frames), axis=0).astype(np.uint8)
    c, why = surface_table(plate)
    if c is None:
        return None, dict(reason=f"no blue or green table, and {why}", n_frames=len(frames), n_tried=len(frames))
    return c, dict(why, method="surface", colour="any", n_frames=len(frames), n_tried=len(frames), frame_image=plate)


def find_table(video, n=25):
    """The table's corners in a video file (frames read with OpenCV, spread over 3 to 97% of it). Returns (corners or None, info)."""
    cap = cv2.VideoCapture(str(video)); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); frames = []
    for i in sorted(set(int(v) for v in np.linspace(0.03 * total, 0.97 * total, n))) if total > 0 else range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    return find_table_in_frames(frames)
