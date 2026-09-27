"""Ball tracking for a camera behind one end with other games in view. Every ball-like thing is followed at once as short tracks;
only the tracks that bounce on this table or cross its net are kept as the match ball, plus the tracks that join them in time and
place (the ball going to and from a racket); the rest belong to the neighbouring tables. Made for
IMG_3140 to IMG_3142 (filmed from behind one player, two other games in play), where the one-ball tracker kept jumping to the other balls.

    python analysis/multi_track.py data/own/IMG_3140.mp4 --table data/own/IMG_3140_table.json --near Sam --far Robin --out out_own/IMG_3140_multi

Writes what `tt-scout analyse` writes (track.csv, players.csv, events.csv, rallies.json, quality.json, stats), so the report step runs
unchanged; plus cands.npz (every candidate, reused on a rerun) and tracklets.json (why each short track was kept or dropped).
"""
import argparse, csv, json, math, pathlib, sys, time
import numpy as np, cv2
ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from tt_scout.config import Config, NET_X, TABLE_LENGTH as L, TABLE_WIDTH as W
from tt_scout.table import Table
from tt_scout.camera import Camera
from tt_scout.detector import BallDetector
from tt_scout.events import detect_events
from tt_scout.points_v1 import segment_points_v1
from tt_scout.players import write_obs, load_obs, name_points, end_of, auto_reference
from tt_scout.quality import assess


def detect_all(video, table, cfg, out):
    """Every ball candidate in every frame (the detector unchanged), and the player blobs for naming ends; cached."""
    cache = out / "cands.npz"
    if cache.exists():
        z = np.load(cache)
        return z["cands"], float(z["fps"]), int(z["n"]), None
    cap = cv2.VideoCapture(str(video)); fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    det = BallDetector(cfg, table, fps); C, prow, i, t0 = [], [], 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for c in det.detect(frame):
            C.append((i, c.x, c.y, c.area, c.score))
        for pl in det.players:
            prow.append([i, i / fps, end_of(table, pl["cx"], pl["cy"]), round(pl["cx"]), round(pl["cy"]), int(pl["area"]),
                         round(pl["h"], 1), round(pl["s"], 1), round(pl["v"], 1)])
        i += 1
        if i % 1500 == 0:
            print(f"  {i} frames, {len(C)} candidates, {i / (time.time() - t0):.0f} fps", flush=True)
    cap.release()
    C = np.array(C, float) if C else np.zeros((0, 5))
    np.savez_compressed(cache, cands=C, fps=fps, n=i)
    write_obs(out / "players.csv", prow)
    return C, fps, i, prow


def tracklets(C, n, gate_new=90.0, gate_base=30.0, gate_mult=0.8, max_miss=4, min_len=5):
    """Short tracks over all candidates at once: constant-velocity prediction, greedy nearest assignment inside a speed-scaled gate."""
    by_f = {}
    for row in C:
        by_f.setdefault(int(row[0]), []).append(row)
    active, done = [], []
    for f in range(n):
        cs = by_f.get(f, [])
        pairs = []
        for ti, tr in enumerate(active):
            dtf = f - tr["last"]
            px, py = tr["x"] + tr["vx"] * dtf, tr["y"] + tr["vy"] * dtf
            gate = gate_new if len(tr["pts"]) < 2 else gate_base + gate_mult * math.hypot(tr["vx"], tr["vy"]) * dtf
            for ci, c in enumerate(cs):
                d = math.hypot(c[1] - px, c[2] - py)
                if d <= gate:
                    pairs.append((d, ti, ci))
        used_t, used_c = set(), set()
        for d, ti, ci in sorted(pairs):
            if ti in used_t or ci in used_c:
                continue
            used_t.add(ti); used_c.add(ci)
            tr, c = active[ti], cs[ci]; dtf = f - tr["last"]
            vx, vy = (c[1] - tr["x"]) / dtf, (c[2] - tr["y"]) / dtf
            if len(tr["pts"]) >= 2:
                vx, vy = 0.5 * tr["vx"] + 0.5 * vx, 0.5 * tr["vy"] + 0.5 * vy
            tr.update(x=c[1], y=c[2], vx=vx, vy=vy, last=f); tr["pts"].append((f, c[1], c[2]))
        for ci, c in enumerate(cs):
            if ci not in used_c:
                active.append(dict(x=c[1], y=c[2], vx=0.0, vy=0.0, last=f, pts=[(f, c[1], c[2])]))
        keep = []
        for tr in active:
            (done if f - tr["last"] > max_miss else keep).append(tr)
        active = keep
    return [np.array(tr["pts"]) for tr in done + active if len(tr["pts"]) >= min_len]


def evidence(tk, table, cam):
    """Bounces inside this table's outline (the lowest point of a V in the picture) and crossings of this net between its posts."""
    quad = np.array(table.corners, np.float32).reshape(-1, 1, 2)
    y = tk[:, 2]; b = 0
    for i in range(2, len(tk) - 2):
        if y[i] >= y[i - 1] and y[i] >= y[i + 1] and y[i] - y[i - 2] >= 3 and y[i] - y[i + 2] >= 3:
            if cv2.pointPolygonTest(quad, (float(tk[i, 1]), float(tk[i, 2])), True) >= -8:
                b += 1
    n1, n2 = cam.project([[NET_X, -0.1525, 0.1525], [NET_X, W + 0.1525, 0.1525]])     # the net's top tape, post to post
    d = (tk[:, 1] - n1[0]) * (n2[1] - n1[1]) - (tk[:, 2] - n1[1]) * (n2[0] - n1[0])
    c = 0
    for i in range(1, len(tk)):
        if d[i] * d[i - 1] < 0:
            a = d[i - 1] / (d[i - 1] - d[i]); xc = tk[i - 1, 1] + a * (tk[i, 1] - tk[i - 1, 1])
            if min(n1[0], n2[0]) - 20 <= xc <= max(n1[0], n2[0]) + 20:
                c += 1
    return b, c


def ballness(tk):
    """How much a short track moves like a ball in flight: straight along the picture's x and a parabola in its y between bounces
    (a limb or a shirt wobbles). exp(-rms/3 px) of a piecewise fit split at V-shaped turns, times a moving-at-all factor."""
    f, x, y = tk[:, 0], tk[:, 1], tk[:, 2]
    cuts = [0] + [i for i in range(2, len(tk) - 2) if y[i] >= y[i - 1] and y[i] >= y[i + 1] and y[i] - y[i - 2] >= 3 and y[i] - y[i + 2] >= 3] + [len(tk) - 1]
    res = []
    for a_, b_ in zip(cuts[:-1], cuts[1:]):
        if b_ - a_ + 1 >= 4:
            ff = f[a_:b_ + 1] - f[a_]
            res += list(y[a_:b_ + 1] - np.polyval(np.polyfit(ff, y[a_:b_ + 1], 2), ff))
            res += list(x[a_:b_ + 1] - np.polyval(np.polyfit(ff, x[a_:b_ + 1], 1 if b_ - a_ + 1 < 6 else 2), ff))
    if not res:
        return 0.0
    rms = float(np.sqrt(np.mean(np.square(res))))
    speed = float(np.median(np.hypot(np.diff(x), np.diff(y)) / np.maximum(1, np.diff(f))))
    return math.exp(-rms / 3.0) * min(1.0, speed / 3.0)


def select(tks, ev, fps, max_gap_s=3.0):
    """The match ball as the best single path through the short tracks, in time order (dynamic programming): each track scores for
    bounces on this table and crossings of this net and for moving like a ball; joining two tracks is free when one's motion carries
    into the other, and costs something when the ball has to be assumed to reappear elsewhere (between strokes, between points)."""
    n = len(tks)
    st = np.array([t[0, 0] for t in tks]); en = np.array([t[-1, 0] for t in tks])
    bl = np.array([ballness(t) for t in tks])
    evs = np.array([b + c for b, c in ev], float)
    score = 3.0 * evs + 0.08 * np.array([len(t) for t in tks]) * bl - 0.6
    order = np.argsort(en)                                               # a track can only follow one that ended before it starts
    best = np.full(n, -np.inf); prev = np.full(n, -1)
    ends_sorted = en[order]
    for j in np.argsort(st):
        lo = np.searchsorted(ends_sorted, st[j] - max_gap_s * fps, side="left")
        hi = np.searchsorted(ends_sorted, st[j], side="left")            # ended strictly before j starts
        bj, pj = score[j], -1
        for k in order[lo:hi]:
            if best[k] == -np.inf:
                continue
            gap = st[j] - en[k]; a_, b_ = tks[k], tks[j]
            v = (a_[-1, 1:3] - a_[-3, 1:3]) / max(1.0, a_[-1, 0] - a_[-3, 0]) if len(a_) >= 3 else np.zeros(2)
            miss = float(np.hypot(*(b_[0, 1:3] - (a_[-1, 1:3] + v * gap))))
            cost = 0.0 if (gap <= 12 and miss <= 40 + 1.2 * float(np.hypot(*v)) * gap) else 1.0 + gap / fps
            cand = best[k] - cost + score[j]
            if cand > bj:
                bj, pj = cand, k
        best[j], prev[j] = bj, pj
    chosen, j = [], int(np.argmax(best))
    while j >= 0:                                                        # the single best path, back from its best end
        chosen.append(j); j = int(prev[j])
    return [i for i in chosen if score[i] > -0.6 + 1e-9 or ev[i][0] + ev[i][1] > 0 or bl[i] > 0.35]


def steady_frames(video, n, fps, ref_t=5.0, tol_px=5.0):
    """Frames where the camera still stands where it did at ref_t (ORB features on the room against that frame, every second)."""
    cap = cv2.VideoCapture(str(video)); orb = cv2.ORB_create(3000); bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    cap.set(1, int(ref_t * fps)); ok, ref = cap.read()
    k0, d0 = orb.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)
    probe = np.array([[700, 500], [1200, 450], [900, 600]], np.float32).reshape(-1, 1, 2)
    ok_s = []
    for f in range(0, n, int(round(fps))):
        cap.set(1, f); ok, img = cap.read()
        good = False
        if ok:
            k, d = orb.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
            if d is not None:
                m = sorted(bf.match(d0, d), key=lambda x: x.distance)[:600]
                if len(m) >= 30:
                    A = np.float32([k0[x.queryIdx].pt for x in m]); B = np.float32([k[x.trainIdx].pt for x in m])
                    H, inl = cv2.findHomography(A, B, cv2.RANSAC, 4.0)
                    if H is not None and inl.sum() >= 100:
                        good = float(np.abs(cv2.perspectiveTransform(probe, H) - probe).max()) <= tol_px
        ok_s.append(good)
    cap.release()
    steady = np.zeros(n, bool)
    for i, g in enumerate(ok_s):
        steady[i * int(round(fps)):(i + 1) * int(round(fps))] = g
    return steady


def play_mask(cam, shape):
    """Where this table's play can appear in the picture: a box round it (1.6 m behind each end, 0.7 m beside, up to 1.3 m high)."""
    import itertools
    xs = np.linspace(-1.6, L + 1.6, 30); ys = np.linspace(-0.7, W + 0.7, 15); zs = np.linspace(-0.3, 1.3, 9)
    P = np.array(list(itertools.product(xs, ys, zs))); Xc = P @ cam.R.T + cam.t; P = P[Xc[:, 2] > 0.2]
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(np.round(cam.project(P)).astype(np.int32).clip(-5000, 5000)), 255)
    return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("--table", required=True); ap.add_argument("--near", required=True)
    ap.add_argument("--far", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--player-margin", type=int, default=None, help="px kept clear round a player's blob (smaller: the ball near a racket counts)")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = Config(); cfg.near_name, cfg.far_name = a.near, a.far
    if a.player_margin is not None:
        cfg.player_margin_px = a.player_margin
    table = Table.load(a.table)
    cap = cv2.VideoCapture(str(a.video)); w, h = int(cap.get(3)), int(cap.get(4)); cap.release()
    cam = Camera.from_table_pnp(table, w, h)
    t0 = time.time()
    C, fps, n, _ = detect_all(a.video, table, cfg, out)
    print(f"candidates: {len(C)} in {n} frames ({time.time() - t0:.0f} s)")
    steady = steady_frames(a.video, n, fps)                             # a camera that was moved: its frames are not this calibration's
    mask = play_mask(cam, (h, w))
    keep = steady[C[:, 0].astype(int)] & (mask[np.clip(C[:, 2].astype(int), 0, h - 1), np.clip(C[:, 1].astype(int), 0, w - 1)] > 0)
    print(f"camera steady in {steady.mean():.0%} of frames; {keep.mean():.0%} of candidates where this table's play can be")
    C = C[keep]
    tks = tracklets(C, n)
    ev = [evidence(t, table, cam) for t in tks]
    order = select(tks, ev, fps)
    rows = np.full((n, 6), np.nan); rows[:, 0] = np.arange(n); rows[:, 1] = np.arange(n) / fps; rows[:, 4] = -1
    counts = np.bincount(C[:, 0].astype(int), minlength=n) if len(C) else np.zeros(n)
    rows[:, 5] = counts[:n]
    taken = np.zeros(n, bool)
    for i in order:                                                     # best first; later tracks only fill frames still empty
        for f, x, y in tks[i]:
            f = int(f)
            if not taken[f]:
                rows[f, 2:5] = (x, y, i); taken[f] = True
    info = [dict(id=i, start=int(t[0, 0]), end=int(t[-1, 0]), n=len(t), bounces=int(ev[i][0]), crossings=int(ev[i][1]), kept=i in set(order))
            for i, t in enumerate(tks)]
    (out / "tracklets.json").write_text(json.dumps(info))
    print(f"tracks: {len(tks)} short tracks, {len(order)} kept as the match ball; ball in {taken.mean():.0%} of frames")
    track = rows
    events = detect_events(track[:, :5], table, cfg, fps)
    rallies = segment_points_v1(events, None, track[:, :5])
    obs = load_obs(out / "players.csv")
    names = {"near": a.near, "far": a.far}
    name_points(rallies, obs, table.meta.get("players") or auto_reference(obs, names), names)
    quality = assess(rallies, track, fps)
    (out / "quality.json").write_text(json.dumps(quality, indent=1))
    with open(out / "track.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["frame", "t", "x", "y", "track", "n_cands"])
        wr.writerows([[int(r[0]), r[1], r[2], r[3], int(r[4]), int(r[5])] for r in rows])
    with open(out / "events.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(events[0].keys()) if events else ["t"]); wr.writeheader(); wr.writerows(events)
    (out / "rallies.json").write_text(json.dumps(rallies, indent=1))
    (out / "table.json").write_text(pathlib.Path(a.table).read_text())
    from tt_scout.stats import write_stats
    write_stats(out, rallies)
    print(f"{len(rallies)} points; confidence {quality['level'].upper()} ({quality['score']}/100): " + "; ".join(quality.get("reasons", [])))


if __name__ == "__main__":
    main()
