"""Single-ball tracker: constant-velocity prediction with a speed-adaptive gate.

Three rules learned on real footage (OpenTTGames test_2):
  * the gate grows with speed, so a full reversal at a racket hit stays inside it;
  * a track that stays slow is not the ball (hands, shoes, the umpire) and is dropped;
  * while a track is coasting, three fresh detections on a straight line take over immediately,
    instead of waiting for the stale track to time out while the ball flies past.
"""
from collections import deque
import numpy as np


class BallTracker:
    def __init__(self, cfg, fps=60.0):
        self.cfg = cfg
        self.fps = fps
        self.max_missed = max(2, int(round(cfg.max_missed_s * fps)))
        self.min_speed = cfg.min_speed_px_s / fps
        self.slow_n = max(3, int(round(cfg.slow_track_s * fps)))
        self.pos = None                 # last SEEN position (not advanced while coasting)
        self.vel = np.zeros(2)
        self.missed = 0
        self.track_id = 0
        self.age = 0                    # detections taken since the track started
        self.areas = deque(maxlen=6)    # blob areas of the last detections taken
        self.area_ref = None            # the detector's ball area at this table (px), set by whoever runs the tracker
        self.net_line = None            # two image points on the net line (px), set by whoever runs the tracker
        self.speeds = deque(maxlen=self.slow_n)
        self.tentative = []             # candidate lists of the last two frames, for starting a track
        self.backfill = []              # after a start: [(frames_ago, x, y)] for the two frames that led to it

    def _drop(self):
        self.pos, self.missed, self.age = None, 0, 0
        self.areas.clear()
        self.vel[:] = 0
        self.speeds.clear()

    def _near_net(self, p):
        """Whether an image point is within cfg.healthy_net_px of the net line (always, when no net line was given)."""
        if self.net_line is None:
            return True
        (x1, y1), (x2, y2) = self.net_line
        d = abs((p[0] - x1) * (y2 - y1) - (p[1] - y1) * (x2 - x1)) / max(float(np.hypot(x2 - x1, y2 - y1)), 1e-9)
        return d <= self.cfg.healthy_net_px

    def _start(self, c, prev, prev2=None):
        self.track_id += 1
        self.pos = np.array([c.x, c.y])
        self.vel = self.pos - prev
        self.backfill = [(1, float(prev[0]), float(prev[1]))] + ([(2, float(prev2[0]), float(prev2[1]))] if prev2 is not None else [])
        self.missed = 0; self.age = 3                                  # the three frames that started it
        self.areas.clear(); self.areas.append(float(c.area))
        self.speeds.clear(); self.speeds.append(float(np.linalg.norm(self.vel)))
        self.tentative = []

    def _try_init(self, cands):
        """Three consecutive frames on a straight line at ball-like speed. Every triple is tried."""
        cfg = self.cfg
        best = None
        if len(self.tentative) == 2 and cands:
            for a in self.tentative[0]:
                pa = np.array([a.x, a.y])
                for b in self.tentative[1]:
                    pb = np.array([b.x, b.y])
                    step = float(np.linalg.norm(pb - pa))
                    if step >= cfg.init_max_jump_px or step < self.min_speed:
                        continue
                    pred = 2 * pb - pa
                    for c in cands:
                        d = float(np.hypot(c.x - pred[0], c.y - pred[1]))
                        if d < cfg.init_resid_px and (best is None or d - 5 * c.score < best[0]):
                            best = (d - 5 * c.score, pb, c, pa)
        self.tentative = (self.tentative + [list(cands)])[-2:] if cands else []
        return None if best is None else (best[2], best[1], best[3])

    def update(self, cands):
        cfg = self.cfg
        taken = None
        self.backfill = []
        if self.pos is not None:
            k = self.missed + 1
            pred = self.pos + self.vel * k
            gate = cfg.gate_base_px + cfg.gate_vel_mult * float(np.linalg.norm(self.vel)) * k
            best, bc = None, 1.0
            speed = float(np.linalg.norm(self.vel))
            for c in cands:
                d = float(np.hypot(c.x - pred[0], c.y - pred[1]))
                if k >= 2 and speed > self.min_speed:
                    implied = float(np.hypot(c.x - self.pos[0], c.y - self.pos[1])) / k
                    if not (cfg.coast_speed_lo * speed <= implied <= cfg.coast_speed_hi * speed):
                        continue                      # a ball in flight does not triple or lose its speed
                cost = d / gate - 0.3 * c.score       # nearest wins, ball-shaped breaks ties
                if d < gate and cost < bc:
                    best, bc = c, cost
            if best is not None:
                new = np.array([best.x, best.y])
                self.vel = cfg.vel_smooth * (new - self.pos) / k + (1 - cfg.vel_smooth) * self.vel
                self.speeds.append(float(np.linalg.norm(new - self.pos)) / k)
                self.pos, self.missed = new, 0
                self.age += 1; self.areas.append(float(best.area))
                taken = best
                if len(self.speeds) == self.slow_n and max(self.speeds) < self.min_speed:
                    self._drop()                      # been crawling for a while: not the ball
                    taken = None
            else:
                self.missed += 1
                if self.missed > self.max_missed:
                    self._drop()
        fresh = self._try_init([c for c in cands if c is not taken])
        healthy = (self.pos is not None and self.age >= cfg.healthy_min_frames      # a fast ball followed for a while, the size of this
                   and float(np.linalg.norm(self.vel)) >= 2 * self.min_speed           # table's ball: wait for it
                   and (self.area_ref is None or (self.areas and float(np.median(self.areas)) >= cfg.healthy_area_frac * self.area_ref))
                   and self._near_net(self.pos + self.vel * (self.missed + 1)))
        need = cfg.preempt_missed_healthy if healthy else cfg.preempt_missed
        if fresh is not None and (self.pos is None or self.missed >= need):
            c, prev, prev2 = fresh
            self._start(c, prev, prev2)
            return c
        return taken
