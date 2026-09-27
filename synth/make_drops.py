"""A synthetic drop-test recording (tt_scout/droptest.py): the side view of make_synthetic.py with no players, and a ball held still
for a second, let go from rest with its bottom `height` above the table, bouncing (air drag, restitution 0.89) until it is caught.

    python synth/make_drops.py --height 1.0 --out synth/drops
    tt-scout droptest synth/drops.mp4 --table synth/drops_table.json
"""
import argparse, pathlib, sys
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_synthetic import Sim, F, IMG_W, IMG_H, ORANGE
from tt_scout.config import BALL_RADIUS as R
from tt_scout.flight import C0, CD0, G
from tt_scout.table import Table

SPOTS = [(0.5, 0.4), (0.8, 1.1), (1.0, 0.7), (1.8, 0.5), (2.1, 1.0), (2.3, 0.6)]


def drop(p0, t, e=0.89, cd=CD0, h=1 / 2400):
    """Positions at times t (seconds after the ball is let go from rest at p0, its centre), bouncing on the table."""
    k = C0 * cd; p = np.array(p0, float); v = np.zeros(3); out = []; now = 0.0
    acc = lambda v: np.array([0.0, 0.0, -G]) - k * np.linalg.norm(v) * v
    for target in t:
        while now < target - 1e-12:
            dt = min(h, target - now)
            a1 = acc(v); a2 = acc(v + 0.5 * dt * a1); a3 = acc(v + 0.5 * dt * a2); a4 = acc(v + dt * a3)
            p = p + dt * (v + dt / 6 * (a1 + a2 + a3)); v = v + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4); now += dt
            if p[2] <= R and v[2] < 0:
                p[2] = 2 * R - p[2]; v[2] = -e * v[2]
        out.append(p.copy())
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--height", type=float, default=1.0); ap.add_argument("--fps", type=float, default=59.94)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="synth/drops")
    a = ap.parse_args()
    per = 4.0                                                       # held 1 s, falling and bouncing 2.2 s, caught 0.8 s
    sim = Sim("side", per * len(SPOTS) + 1.0, a.fps, a.seed)
    rng = np.random.default_rng(a.seed); n = sim.n; ts = np.arange(n) / a.fps
    bg = sim.bg.copy(); sim.draw_table(bg)
    vw = cv2.VideoWriter(a.out + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (IMG_W, IMG_H))
    prev = None
    for i, t in enumerate(ts):
        k, local = int((t - 1.0) // per), (t - 1.0) % per
        img = bg.copy(); uv = None
        if 0 <= k < len(SPOTS) and t >= 1.0 and local < 3.2:
            x, y = SPOTS[k]; p0 = (x, y, a.height + R)
            fall = local - 1.0                                      # seconds since it was let go
            P = np.array(p0) if fall < 0 else drop(p0, [fall])[0]
            (uv,), (d,) = sim.cam.project([P])
            r = F * R / d
            if prev is not None and fall > 0:
                mid = (prev + uv) / 2
                cv2.line(img, tuple(mid.astype(int)), tuple(uv.astype(int)), ORANGE, max(1, int(round(2 * r))))
            cv2.circle(img, tuple(uv.astype(int)), max(1, int(round(r))), ORANGE, -1)
        prev = uv
        img = np.clip(img.astype(np.int16) + sim.noise[rng.integers(len(sim.noise))], 0, 255).astype(np.uint8)
        vw.write(img)
    vw.release()
    Table.save(a.out + "_table.json", sim.corners_px.tolist(), "side", synthetic=True, video=a.out + ".mp4")
    print(f"{a.out}.mp4: {len(SPOTS)} drops from {a.height:.2f} m, {n} frames")


if __name__ == "__main__":
    main()
