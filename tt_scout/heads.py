"""Where the players' heads are, from the picture itself. Product layer (clip overlays); not part of the evaluated method.

The camera never moves, so the median of a few dozen frames spread over the recording is the empty hall (players walk about between
points, the umpire sits still and stays in it). A frame minus that plate is the players. Each end's player is the lowest-standing
foreground shape in its half that reaches down past the table top and whose feet read inside the zone a player can stand in, which
leaves out an audience (what shows of them stops at the barrier, metres away). The head is the first row from the top of that shape
that is wider than an arm or a racket handle, and its centre is the middle of the shape over the next head-height of rows.

Why not the tracker's player blobs: they come from motion, so a player standing still is often only legs or a swinging arm, and the
blob centre was measured 40 to 140 px from the head. Name balloons placed from it sat on faces.

    plate = background_plate("match.mp4", (960, 540))
    heads = find_heads(frame_960x540, plate, table_top, net_x, standing=standing_zone(table, 540 / source_h))  # {"near": (x, y, sure), ...}
"""
import numpy as np, cv2

from .config import TABLE_LENGTH as L, TABLE_WIDTH as W

THRESH_LO, THRESH_HI, THRESH_K = 16, 40, 0.30   # difference from the plate that counts as a player, scaled with the plate's brightness:
#                                               dark hair on a dark backdrop differs by ~25, compression noise on a bright wall by ~15
MIN_BODY = 700              # px of foreground a player's shape needs at 540 lines
MIN_PART = 120              # a separate piece above the body (head and shoulders cut off by a banner of the shirt's colour) counts from here
JOIN_GAP = 60               # px of rows such a piece may float above the body and still belong to it
HEAD_MIN_W = 15             # a row this wide is a head, not an arm or a racket handle (arms are 8 to 12 px at 540 lines)
HEAD_H = 26                 # rows of head below its top
HEAD_SPAN = 140             # px either side of the feet the head may be: over 218 frames of three matches the widest lunge was 132.
#                             It keeps the head of whoever stands behind a player out of a shape the two of them share.
HEAD_ABOVE_TABLE = (175, 15)  # a head centre lies between 175 and 15 px above the table's far edge at 540 lines; outside it, the shape's top
#                               was a hand, a racket or a knee and only the x is used
STAND_X, STAND_Y = 3.0, 3.0  # the zone a player stands in, in table metres: up to 3 m beyond each end line and 3 m past each side line.
#                              Measured over four halls: every player's feet read inside it, the nearest spectator's 7 m beyond the end line.
STAND_BEHIND = 0.15         # ... but no more than this past the FAR side line. Feet are on the floor, below the table top, so read as if on
#                             the table top they land nearer the camera: the two players' feet read 1.0 to 1.5 m on the camera side, while a
#                             game on a table further back reads just past the far side line (our 2026-09-25 recording: 1.77 m = W + 0.25).


def sample_frames(video, size, n=41, lo=None, hi=None):
    """n frames spread over the whole recording, or over frames lo to hi of it, resized to size = (w, h)."""
    cap = cv2.VideoCapture(str(video)); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    a = 0 if lo is None else max(0, int(lo)); b = max(0, total - 1) if hi is None else min(total - 1, int(hi))
    frames = []
    for fi in np.linspace(a, max(a, b), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, f = cap.read()
        if ok:
            frames.append(cv2.resize(f, size, interpolation=cv2.INTER_AREA))
    cap.release()
    return frames


def plate_of(frames):
    return np.median(np.stack(frames), axis=0).astype(np.uint8) if frames else None


def background_plate(video, size, n=41, lo=None, hi=None):
    """Median of n frames spread over the whole recording, or over frames lo to hi of it, resized to size = (w, h). A plate taken from
    the seconds around a rally holds whoever stood still through it, so a crowd behind the table drops out of the foreground and only
    the two players are left; one taken from the whole recording is steadier but keeps anyone who moved about during it."""
    return plate_of(sample_frames(video, size, n, lo, hi))


def thresholds(plate):
    return np.clip(THRESH_K * plate.max(axis=2).astype(np.float32) + 8, THRESH_LO, THRESH_HI)


def foreground(frame, plate, thr=None):
    thr = thresholds(plate) if thr is None else thr
    m = (cv2.absdiff(frame, plate).max(axis=2) > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def head_in(mask, x0, x1, y0, y1, reach_y=None, standing=None):
    """dict(x, y, found_y) for the player in mask[y0:y1, x0:x1], or None. The player is the lowest shape that is big enough, reaches down
    to reach_y (the table top: a player standing behind the table has the legs hidden by it, a raised hand or the ball does not get that
    low) and stands where a player can stand (standing(x, y) of its lowest point), joined with the pieces that float up to JOIN_GAP px
    above it inside its columns. found_y False: no head-wide row was found."""
    x0, x1, y0, y1 = max(0, int(x0)), min(mask.shape[1], int(x1)), max(0, int(y0)), min(mask.shape[0], int(y1))
    if x1 - x0 < 20 or y1 - y0 < 40:
        return None
    win = mask[y0:y1, x0:x1]
    n, lab, st, _ = cv2.connectedComponentsWithStats(win, connectivity=8)
    L_, T_, W_, H_, A_ = cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA
    reach = (win.shape[0] - 2) if reach_y is None else int(reach_y) - y0

    def player(k):                                                      # big enough, low enough, and standing where a player can stand
        if st[k, A_] < MIN_BODY or st[k, T_] + st[k, H_] < reach:
            return False
        foot = (x0 + st[k, L_] + st[k, W_] / 2, y0 + st[k, T_] + st[k, H_])
        return True if standing is None else standing(*foot)

    bodies = [i for i in range(1, n) if player(i)]
    if not bodies:
        return None
    body = max(bodies, key=lambda k: (st[k, T_] + st[k, H_], st[k, A_]))   # the lowest of them: the one whose feet are on this floor
    keep = {body}; top = st[body, T_]; left, right = st[body, L_] - 15, st[body, L_] + st[body, W_] + 15
    grown = True
    while grown:                                                        # pieces stacked above the body, nearest first
        grown = False
        for k in range(1, n):
            if k in keep or st[k, A_] < MIN_PART:
                continue
            kl, kr, kb = st[k, L_], st[k, L_] + st[k, W_], st[k, T_] + st[k, H_]
            if kr >= left and kl <= right and top - JOIN_GAP <= kb <= top + 5:
                keep.add(k); top = min(top, st[k, T_]); grown = True
    comp = np.isin(lab, list(keep))
    rows = np.flatnonzero(comp.any(axis=1))
    feet = float(np.median(np.nonzero(comp[max(0, rows[-1] - 3):rows[-1] + 1])[1]))   # the columns its lowest rows stand on
    comp[:, :max(0, int(feet - HEAD_SPAN))] = False; comp[:, int(feet + HEAD_SPAN) + 1:] = False
    cx_body = float(np.median(np.nonzero(comp)[1]))
    for r in range(top, win.shape[0]):
        cols = np.flatnonzero(comp[r])
        if not len(cols):
            continue
        runs = np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1)          # contiguous stretches of this row
        best = max(runs, key=len)
        if len(best) >= HEAD_MIN_W:
            cx = float(best.mean())
            band = comp[r:r + HEAD_H, max(0, int(cx) - 30):int(cx) + 31]
            _, xs = np.nonzero(band)
            if len(xs):
                cx = max(0, int(cx) - 30) + float(np.median(xs))
            return dict(x=x0 + cx, y=y0 + r + HEAD_H / 2, found_y=True)
    return dict(x=x0 + cx_body, y=None, found_y=False)


def standing_zone(table, scale=1.0):
    """standing(x, y) for a calibrated table, in the pixels of a frame scaled by scale: whether a shape whose lowest point is (x, y) is
    someone standing at this table. It reads that lowest point as if it lay on the table surface. A player's feet are on the floor beside
    the table and land a metre or two beyond the end line; an audience behind a barrier stops being visible at the barrier, so their
    lowest point reads several metres further away still, which is what keeps their heads out of the answer."""
    # which side line is the far one (higher in the picture) depends on the order the corners were given in
    far_is_W = table.to_px([[L / 2, W]])[0][1] < table.to_px([[L / 2, 0.0]])[0][1]
    y_lo, y_hi = (-STAND_Y, W + STAND_BEHIND) if far_is_W else (-STAND_BEHIND, W + STAND_Y)

    def standing(x, y):
        tx, ty = table.to_table([[x / scale, y / scale]])[0]
        return -STAND_X <= tx <= L + STAND_X and y_lo <= ty <= y_hi

    return standing


def find_heads(frame, plate, table_top, net_x, above=230, thr=None, standing=None):
    """{"near": (x, y, sure_y) | None, "far": ...} for a frame the size of the plate; near = the half left of the net. sure_y False: the
    head height is not trustworthy (nothing head-wide, or outside HEAD_ABOVE_TABLE) and only x should be used.
    standing comes from standing_zone() of the table and keeps the answer to the two players. Without it any shape reaching below the
    table's top edge will do, which was enough for the empty training halls but not for a hall with an audience."""
    if plate is None or frame.shape != plate.shape:
        return {"near": None, "far": None}
    mask = foreground(frame, plate, thr)
    reach = table_top - 12
    y0, y1 = table_top - above, mask.shape[0]                           # down to the bottom of the picture, so a shape's foot is its foot
    out = {}
    for end, (a, b) in (("near", (0, net_x)), ("far", (net_x, mask.shape[1]))):
        h = head_in(mask, a, b, y0, y1, reach_y=reach, standing=standing)
        if h is None:
            out[end] = None
            continue
        ok = h["found_y"] and table_top - HEAD_ABOVE_TABLE[0] <= h["y"] <= table_top - HEAD_ABOVE_TABLE[1]
        out[end] = (h["x"], h["y"] if ok else None, ok)
    return out
