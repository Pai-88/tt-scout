"""Synthetic table tennis clip with known ball physics, for testing the pipeline before real footage.

Usage: python synth/make_synthetic.py --view side --seconds 40 --out synth/side
Writes <out>.mp4, <out>_table.json (corners as the calibration tool would save), <out>_truth.json.

What is modelled: gravity, table/floor restitution, the net, two textured players that follow the ball
and can occlude it, a moving racket per player (a classic false positive), motion streaks, sensor noise.
What is not: spin, drag, lighting changes, rolling shutter, compression of a real phone.
"""
import argparse, json, pathlib, sys
import numpy as np, cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from tt_scout.config import TABLE_LENGTH as L, TABLE_WIDTH as W, BALL_RADIUS
from tt_scout.table import TABLE_PTS

G, E_TABLE, NET_H, FLOOR_Z = 9.81, 0.88, 0.1525, -0.76
IMG_W, IMG_H, F = 1920, 1080, 1350.0
ORANGE = (0, 140, 255)
VIEWS = {
    "side":   dict(cam=((L / 2, -3.5, 1.5), (L / 2, W / 2, 0.0)), corners_w=[(0, 0), (0, W), (L, W), (L, 0)]),
    "behind": dict(cam=((-3.0, W / 2, 2.2), (L / 2, W / 2, 0.0)), corners_w=[(0, W), (0, 0), (L, 0), (L, W)]),
}


def unit(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)


def other(s):
    return "far" if s == "near" else "near"


class Camera:
    def __init__(self, pos, target):
        self.C = np.asarray(pos, float)
        fwd = unit(np.asarray(target, float) - self.C)
        right = unit(np.cross(fwd, [0.0, 0.0, 1.0]))
        self.R = np.stack([right, -np.cross(right, fwd), fwd])

    def project(self, P):
        d = (np.atleast_2d(np.asarray(P, float)) - self.C) @ self.R.T
        uv = np.stack([F * d[:, 0] / d[:, 2] + IMG_W / 2, F * d[:, 1] / d[:, 2] + IMG_H / 2], 1)
        return uv, d[:, 2]


class Ball:
    def __init__(self, p, v, hitter, is_serve, t):
        self.p, self.v = np.array(p, float), np.array(v, float)
        self.hitter, self.need_own, self.got_opp, self.whiff = hitter, is_serve, False, False
        self.decided, self.decided_t, self.uv_prev, self.t0 = None, None, None, t


class Sim:
    def __init__(self, view, seconds, fps, seed):
        self.view_name = view
        self.view = VIEWS[view]; self.cam = Camera(*self.view["cam"])
        self.rng = np.random.default_rng(seed); self.fps = fps; self.dt = 1.0 / fps; self.n = int(seconds * fps)
        self.M = cv2.getAffineTransform(np.float32(self.view["corners_w"][:3]), TABLE_PTS[:3])
        self.events, self.rallies = [], []
        self.corners_px, _ = self.cam.project([(x, y, 0.0) for x, y in self.view["corners_w"]])
        self.px = {"near": -0.6, "far": L + 0.6}; self.py = {"near": W / 2, "far": W / 2}
        self.ident = {"near": "A", "far": "B"}                       # who stands at which end; swaps like a change of ends
        self.shirt = {"A": (60, 50, 170), "B": (170, 120, 40)}      # BGR: A reddish, B teal-blue
        self.poly = {}; self.pdepth = {}
        self.tex = self.rng.normal(0, 22, (IMG_H, IMG_W, 1)).astype(np.int16)
        self.noise = [self.rng.normal(0, 2.5, (IMG_H, IMG_W, 3)).astype(np.int16) for _ in range(6)]
        self.bg = self.make_bg()

    # ---------- geometry helpers
    def w2t(self, x, y):
        return (self.M @ np.array([x, y, 1.0])).tolist()

    def depth_of(self, P):
        return float(self.cam.project([P])[1][0])

    def make_bg(self):
        img = np.full((IMG_H, IMG_W, 3), 178, np.uint8)
        floor = [(-2.2, -3.0), (L + 3.0, -3.0), (L + 3.0, W + 3.0), (-2.2, W + 3.0)]
        uv, d = self.cam.project([(x, y, FLOOR_Z) for x, y in floor])
        if (d > 0.3).all():
            cv2.fillPoly(img, [uv.astype(np.int32)], (128, 128, 128))
        img = np.clip(img.astype(np.int16) + self.rng.normal(0, 5, img.shape), 0, 255).astype(np.uint8)
        return img

    # ---------- physics
    def aim(self, p, target, T):
        v = (np.asarray(target, float) - p) / T; v[2] += G * T / 2; return v

    def clears_net(self, p, v):
        if abs(v[0]) < 1e-6:
            return True
        tn = (L / 2 - p[0]) / v[0]
        return tn <= 0 or p[2] + v[2] * tn - 0.5 * G * tn * tn > NET_H + 0.03

    def serve(self, server, t):
        rng = self.rng; y0 = rng.uniform(0.3, W - 0.3)
        x0, sgn = (-0.15, 1) if server == "near" else (L + 0.15, -1)
        p = np.array([x0, y0, 0.30])
        target = (x0 + sgn * rng.uniform(0.6, 1.15), y0 + rng.uniform(-0.25, 0.25), 0.0)
        return Ball(p, self.aim(p, target, rng.uniform(0.2, 0.28)), server, True, t)

    def stroke(self, b, player):
        rng = self.rng; sgn = 1 if player == "near" else -1
        opp_end = L if player == "near" else 0.0
        v = b.v
        for k in range(6):
            if rng.random() < 0.12:                                   # a miss: long or wide
                if rng.random() < 0.5:
                    target = (opp_end + sgn * rng.uniform(0.2, 0.7), rng.uniform(0.1, W - 0.1), 0.0)
                else:
                    edge = 0.0 if rng.random() < 0.5 else W
                    target = (opp_end - sgn * rng.uniform(0.3, 1.0), edge + (-1 if edge == 0 else 1) * rng.uniform(0.08, 0.35), 0.0)
            else:
                target = (opp_end - sgn * rng.uniform(0.3, 1.25), rng.uniform(0.12, W - 0.12), 0.0)
            dist = np.linalg.norm(np.asarray(target) - b.p)
            T = float(np.clip(dist / rng.uniform(7, 15), 0.25, 0.6)) * (1 + 0.15 * k)
            v = self.aim(b.p, target, T)
            if self.clears_net(b.p, v):
                break
        b.v = v; b.hitter, b.need_own, b.got_opp = player, False, False

    def decide(self, b, winner, t):
        if b.decided is None:
            b.decided, b.decided_t = winner, t

    def ev(self, t, frame, kind, side, x=None, y=None):
        e = dict(t=round(t, 4), frame=frame, kind=kind, side=side)
        if x is not None:
            xm, ym = self.w2t(x, y)
            e.update(x_w=round(float(x), 4), y_w=round(float(y), 4), x_m=round(xm, 4), y_m=round(ym, 4))
        self.events.append(e)

    def step(self, b, t, frame):
        """Advance one frame. Returns False when the ball should disappear (point over)."""
        h = self.dt / 4
        for _ in range(4):
            x_prev = b.p[0]
            b.p += b.v * h; b.v[2] -= G * h
            x, y, z = b.p
            if (x_prev - L / 2) * (x - L / 2) < 0 and z >= NET_H:                    # crossed the net
                self.ev(t, frame, "net", b.hitter)
            if (x_prev - L / 2) * (x - L / 2) < 0 and z < NET_H:                     # hit the net cord
                self.ev(t, frame, "nethit", b.hitter)
                b.p[0] = L / 2 - 0.03 * np.sign(b.v[0]); b.v[0] = -0.4 * b.v[0]; b.v[1] *= 0.3; b.v[2] = min(b.v[2], 0.3)
                continue
            if z <= 0 and b.v[2] < 0 and 0 <= x <= L and 0 <= y <= W:              # table
                side = "near" if x < L / 2 else "far"
                b.p[2] = 0.0; b.v[2] = -E_TABLE * b.v[2]; b.v[0] *= 0.93; b.v[1] *= 0.93
                self.ev(t, frame, "bounce", side, x, y)
                if side == b.hitter:
                    if b.need_own:
                        b.need_own = False
                    else:
                        self.decide(b, other(b.hitter), t)
                elif b.need_own:
                    self.decide(b, other(b.hitter), t)
                elif b.got_opp:
                    self.decide(b, b.hitter, t)
                else:
                    b.got_opp = True
                continue
            if z <= FLOOR_Z and b.v[2] < 0:                                          # floor
                self.ev(t, frame, "floor", "near" if x < L / 2 else "far")
                b.p[2] = FLOOR_Z; b.v[2] = -0.5 * b.v[2]; b.v[0] *= 0.6; b.v[1] *= 0.6
                self.decide(b, b.hitter if b.got_opp else other(b.hitter), t)
                continue
            if b.decided is None and not b.whiff and z > FLOOR_Z + 0.1:             # racket
                player = "near" if (x < -0.35 and b.v[0] < 0) else ("far" if (x > L + 0.35 and b.v[0] > 0) else None)
                if player is not None and player != b.hitter and b.got_opp:
                    if self.rng.random() < 0.08:
                        b.whiff = True
                    else:
                        self.ev(t, frame, "hit", player)
                        self.stroke(b, player)
        if t - b.t0 > 12 and b.decided is None:
            b.decided, b.decided_t = "none", t
        return not (b.decided is not None and t - b.decided_t > 0.3)

    # ---------- rendering
    def draw_table(self, img):
        cv2.fillPoly(img, [self.corners_px.astype(np.int32)], (140, 70, 30))
        cv2.polylines(img, [self.corners_px.astype(np.int32)], True, (240, 240, 240), 2)
        net, _ = self.cam.project([(L / 2, 0, 0), (L / 2, W, 0), (L / 2, W, NET_H), (L / 2, 0, NET_H)])
        cv2.fillPoly(img, [net.astype(np.int32)], (90, 90, 90))
        cv2.line(img, tuple(net[2].astype(int)), tuple(net[3].astype(int)), (240, 240, 240), 1)

    def draw_player(self, img, name, t):
        x, y = self.px[name], self.py[name]
        body = [(x, y - 0.22, FLOOR_Z), (x, y + 0.22, FLOOR_Z), (x, y + 0.22, 1.0), (x, y - 0.22, 1.0)]
        uv, d = self.cam.project(body)
        poly = uv.astype(np.int32)
        self.poly[name], self.pdepth[name] = poly, float(d.mean())
        mask = np.zeros((IMG_H, IMG_W), np.uint8); cv2.fillPoly(mask, [poly], 255)
        shifted = np.roll(self.tex, int(uv[:, 0].mean()), axis=1)
        region = mask > 0
        img[region] = np.clip(np.array(self.shirt[self.ident[name]], np.int16) + shifted[region], 0, 255).astype(np.uint8)
        sgn = 1 if name == "near" else -1
        rk = (x + sgn * 0.35, y + 0.30 * np.sin(2 * np.pi * 0.7 * t), 0.5 + 0.15 * np.sin(2 * np.pi * 0.5 * t + sgn))
        ruv, rd = self.cam.project([rk])
        cv2.circle(img, tuple(ruv[0].astype(int)), max(2, int(F * 0.085 / rd[0])), (0, 0, 140), -1)

    def draw_ball(self, img, b):
        uv, d = self.cam.project([b.p]); uv, d = uv[0], d[0]
        hidden = d <= 0.2 or not (0 <= uv[0] < IMG_W and 0 <= uv[1] < IMG_H)
        for name in ("near", "far"):
            if name in self.poly and d > self.pdepth[name] and cv2.pointPolygonTest(self.poly[name], (float(uv[0]), float(uv[1])), False) >= 0:
                hidden = True
        if not hidden:
            r = F * BALL_RADIUS / d
            if b.uv_prev is not None:
                mid = (b.uv_prev + uv) / 2
                cv2.line(img, tuple(mid.astype(int)), tuple(uv.astype(int)), ORANGE, max(1, int(round(2 * r))))
            cv2.circle(img, tuple(uv.astype(int)), max(1, int(round(r))), ORANGE, -1)
        b.uv_prev = uv
        return None if hidden else uv

    def render(self, t, b):
        img = self.bg.copy()
        for s in ("near", "far"):
            target_y = b.p[1] if b is not None else W / 2
            self.py[s] += (target_y - self.py[s]) * 0.08
        items = [(self.depth_of((L / 2, W / 2, 0.0)), "table")] + \
                [(self.depth_of((self.px[s], self.py[s], 0.2)), s) for s in ("near", "far")]
        for _, name in sorted(items, key=lambda i: -i[0]):
            self.draw_table(img) if name == "table" else self.draw_player(img, name, t)
        uv = self.draw_ball(img, b) if b is not None else None
        img = np.clip(img.astype(np.int16) + self.noise[self.rng.integers(len(self.noise))], 0, 255).astype(np.uint8)
        return img, uv

    # ---------- main loop
    def run(self, out):
        vw = cv2.VideoWriter(str(out) + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (IMG_W, IMG_H))
        server, points, b, gap_until = "near", 0, None, 1.5
        visible = 0
        for frame in range(self.n):
            t = frame * self.dt
            if b is None and t >= gap_until:
                b = self.serve(server, t); self.ev(t, frame, "serve", server)
            if b is not None and not self.step(b, t, frame):
                evs = [e for e in self.events if b.t0 <= e["t"] <= t]
                self.rallies.append(dict(start_t=round(b.t0, 3), end_t=round(t, 3), server=server,
                                         winner=None if b.decided == "none" else b.decided,
                                         near_player=self.ident["near"], far_player=self.ident["far"],
                                         server_name=self.ident[server],
                                         winner_name=None if b.decided in ("none", None) else self.ident[b.decided],
                                         n_hits=sum(e["kind"] in ("hit", "serve") for e in evs),
                                         n_bounces=sum(e["kind"] == "bounce" for e in evs)))
                b = None; gap_until = t + self.rng.uniform(3.5, 6.0); points += 1
                if points % 2 == 0:
                    server = other(server)
                if points % 3 == 0:                                   # change of ends
                    self.ident = {"near": self.ident["far"], "far": self.ident["near"]}
            img, uv = self.render(t, b)
            visible += uv is not None
            for e in self.events:
                if e["frame"] == frame and "visible" not in e:
                    e["visible"] = uv is not None
            vw.write(img)
        vw.release()
        from tt_scout.table import Table
        hsv = {k: [float(x) for x in cv2.cvtColor(np.uint8([[v]]), cv2.COLOR_BGR2HSV)[0, 0]] for k, v in self.shirt.items()}
        Table.save(str(out) + "_table.json", self.corners_px.tolist(), self.view_name, synthetic=True, video=str(out) + ".mp4",
                   players={"near": {"name": "A", "hsv": hsv["A"]}, "far": {"name": "B", "hsv": hsv["B"]}})
        truth = dict(view=self.view_name, fps=self.fps, frames=self.n, ball_visible_frames=int(visible),
                     events=self.events, rallies=self.rallies)
        pathlib.Path(str(out) + "_truth.json").write_text(json.dumps(truth, indent=1))
        return truth


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="side", choices=list(VIEWS))
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"synth/{a.view}"
    tr = Sim(a.view, a.seconds, a.fps, a.seed).run(out)
    nb = sum(e["kind"] == "bounce" for e in tr["events"])
    print(f"{out}.mp4: {tr['frames']} frames, {len(tr['rallies'])} rallies, {nb} table bounces, "
          f"ball visible {tr['ball_visible_frames']} frames")
