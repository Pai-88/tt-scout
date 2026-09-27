"""The drop test (tt_scout/droptest.py): the physics it compares with, and simulated drops with their rebounds, seen through a
camera beside the table and recovered from the table's corners, found and measured by the same fit as the match speeds."""
import math, unittest
import numpy as np
from tt_scout import droptest, flight
from tt_scout.camera import Camera
from tt_scout.config import Config, TABLE_LENGTH as L, TABLE_WIDTH as W
from tt_scout.events import detect_events
from tt_scout.table import Table
from tests.test_analysis3d import camera

FPS = 59.94


def bouncing(p0, t_len, e=0.89, cd=flight.CD0, h=1 / 2400):
    """A ball let go from rest at p0 (its centre), bouncing on the table (restitution e): (times, positions) every h seconds."""
    k = flight.C0 * cd; p = np.array(p0, float); v = np.zeros(3); ts, ps = [0.0], [p.copy()]
    acc = lambda v: np.array([0.0, 0.0, -flight.G]) - k * np.linalg.norm(v) * v
    for n in range(1, int(t_len / h) + 1):
        a1 = acc(v); a2 = acc(v + 0.5 * h * a1); a3 = acc(v + 0.5 * h * a2); a4 = acc(v + h * a3)
        p = p + h * (v + h / 6 * (a1 + a2 + a3)); v = v + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        if p[2] <= flight.R_BALL and v[2] < 0:
            p[2] = 2 * flight.R_BALL - p[2]; v[2] = -e * v[2]
        ts.append(n * h); ps.append(p.copy())
    return np.array(ts), np.array(ps)


def recording(cam, spots, height=1.0, lost=5, seed=0):
    """Track rows [frame, t, x, y, track id] for one drop per spot, 3 s apart: the ball unseen while held and for the first `lost`
    frames of its fall (a slow ball is not yet tracked), then seen, bouncing, until it is caught 2.2 s after it was let go."""
    rng = np.random.default_rng(seed)
    n = int((1.0 + 3.0 * len(spots)) * FPS); t = np.arange(n) / FPS
    rows = np.c_[np.arange(n), t, np.full(n, np.nan), np.full(n, np.nan), -np.ones(n)]
    for k, (x, y) in enumerate(spots):
        t_go = 1.0 + 3.0 * k + rng.uniform(0, 1 / FPS)
        ts, ps = bouncing((x, y, height + flight.R_BALL), 2.2)
        for i in np.where((t >= t_go + lost / FPS) & (t <= t_go + 2.2))[0]:
            j = int(round((t[i] - t_go) / (ts[1] - ts[0])))
            rows[i, 2:4] = cam.project(ps[j])[0] + rng.normal(0, 0.7, 2); rows[i, 4] = k + 1
    return rows


class Physics(unittest.TestCase):
    def test_the_speed_physics_gives(self):
        self.assertAlmostEqual(droptest.predicted_speed(1.0), 4.154, places=3)             # 15.0 km/h
        self.assertAlmostEqual(droptest.predicted_speed(1.0, cd=1e-9), math.sqrt(2 * 9.81), places=4)   # no air: sqrt(2 g h)
        self.assertLess(droptest.predicted_speed(1.0, cd=0.52), droptest.predicted_speed(1.0, cd=0.42))


class Drops(unittest.TestCase):
    def setUp(self):
        true_cam = camera()
        corners = true_cam.project([[0, 0, 0], [0, W, 0], [L, W, 0], [L, 0, 0]])
        self.table = Table([[float(u), float(v)] for u, v in corners], "side", {})
        self.cam = Camera.from_table_pnp(self.table, 1920, 1080)                         # as the real test does: from the corners
        self.track = recording(true_cam, [(0.6, 0.4), (1.9, 1.0), (1.0, 1.1)])
        self.events = detect_events(self.track, self.table, Config(), FPS)

    def test_each_drop_is_measured_and_its_rebounds_are_not(self):
        res = droptest.measure(self.track, self.events, FPS, self.cam, 1.0)
        self.assertEqual(len(res["rows"]), 3, res["skipped"])
        self.assertTrue(any("coming down again" in why for _, why in res["skipped"]))
        for r in res["rows"]:
            self.assertLess(abs(r["err_pct"]), 3.0, r)
            self.assertAlmostEqual(r["height_fit"], 1.0, delta=0.05)
            self.assertIsNotNone(r["sd"])

    def test_a_missed_first_bounce_does_not_pass_a_rebound_off_as_a_drop(self):
        bounces = sorted((e for e in self.events if e["kind"] == "bounce"), key=lambda e: e["t"])
        first = bounces[0]                                                              # the first drop's first bounce, lost
        drops, skipped = droptest.find_drops(self.track, [e for e in self.events if e is not first])
        self.assertEqual(len(drops), 2)
        self.assertTrue(any("rising" in why for _, why in skipped))


if __name__ == "__main__":
    unittest.main()
