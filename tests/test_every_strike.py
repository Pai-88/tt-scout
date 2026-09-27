"""Every strike gets a speed: a flight with no bounce to hang on (flight.fit_free), a return that went into the net
(flight.net_strikes), and the guard for a shot whose predecessor never bounced. """
import unittest
import numpy as np
from tt_scout import flight, technique
from tt_scout.config import NET_X, TABLE_WIDTH as W
from tests.test_analysis3d import camera


def flight_track(cam, p0, v0, fps=59.94, t_len=0.55, spin=(0.1, 0.0, 0.47)):
    """A ball struck at p0 with v0 at t = 0: its flight (forward in time, from the backward integrator run from a late state)."""
    # run a long way back from a late anchor so that at t = 0 the state is p0, v0: integrate forward by stepping the backward
    # integrator's inverse is awkward, so build it with small explicit steps instead (same physics)
    ts = np.arange(0, t_len, 1 / fps); out = []; p = np.array(p0, float); v = np.array(v0, float); h = 1 / 1200; t = 0.0
    cl_t, cl_s, cd = spin; hx, hy = v[0], v[1]; hn = np.hypot(hx, hy); sx, sy = -hy / hn, hx / hn
    def acc(v):
        sp = np.linalg.norm(v); tx, ty, tz = sy * v[2], -sx * v[2], sx * v[1] - sy * v[0]
        return np.array([-flight.C0 * cd * sp * v[0] + sp * flight.C0 * cl_t * tx, -flight.C0 * cd * sp * v[1] + sp * flight.C0 * cl_t * ty,
                         -flight.G - flight.C0 * cd * sp * v[2] + sp * flight.C0 * cl_t * tz])
    for target in ts:
        while t < target - 1e-12:
            dt = min(h, target - t); a1 = acc(v); a2 = acc(v + 0.5 * dt * a1); a3 = acc(v + 0.5 * dt * a2); a4 = acc(v + dt * a3)
            p = p + dt * (v + dt / 6 * (a1 + a2 + a3)); v = v + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4); t += dt
        out.append(p.copy())
    P = np.array(out)
    return ts, P, cam.project(P)


class NoBounce(unittest.TestCase):
    def test_a_long_ball_is_measured_without_its_bounce(self):
        cam = camera(); fps = 59.94
        p0 = np.array([-0.5, 0.7, 0.25]); v0 = np.array([11.0, 0.3, 1.4])        # a hard ball that will fly past the far end
        ts, P, uv = flight_track(cam, p0, v0, fps)
        keep = (P[:, 2] > -0.5) & (ts > 0.5 / fps)
        ts, P, uv = ts[keep], P[keep], uv[keep] + np.random.default_rng(0).normal(0, 0.8, (keep.sum(), 2))
        Qn = np.array([NET_X, 0, 0]); n = np.cross(cam.C - Qn, [0, 1.0, 0]); n /= np.linalg.norm(n)
        k = np.where(np.diff(np.sign((P - Qn) @ n)) != 0)[0][0]
        f = flight.fit_free(cam, ts, uv, fps, 0.0, t_cross=float(ts[k + 1]), y_prior=(0.7, 0.35))
        self.assertTrue(f["ok"], f.get("why"))
        self.assertAlmostEqual(f["speed"] / np.linalg.norm(v0), 1.0, delta=0.08)
        self.assertTrue(f["free"])
        self.assertIsInstance(flight.Speeds([dict(f, t_bounce=f["t_bounce"], contact_src="seen")]).last(10.0), flight.Approx)


class IntoTheNet(unittest.TestCase):
    def test_a_return_into_the_net_is_found(self):
        fps = 60.0; t = np.arange(0, 3.0, 1 / fps); u = np.full(len(t), np.nan); v = np.full(len(t), 600.0)
        # the ball crosses the net line (u 960) at t = 1.0 going left, bounces at t = 1.2 (u 700), goes on left to u 560 at t = 1.5,
        # is hit back towards the net and dies at u 900 at t = 1.8
        a = (t >= 0.8) & (t < 1.5); u[a] = 1100 - (t[a] - 0.8) * 540
        b = (t >= 1.5) & (t <= 1.8); u[b] = 722 + (t[b] - 1.5) * 600
        track = np.c_[np.arange(len(t)), t, u, v, np.ones(len(t))]
        evs = [dict(kind="net", t=1.0, side="near", x_px=960.0, y_px=600.0), dict(kind="bounce", t=1.2, side="near", x_px=700.0, y_px=640.0)]
        pts = [dict(id=1, ending="not_returned", events=evs, near_player="A", far_player="B")]
        got = flight.net_strikes(pts, evs, track, fps)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["hitter_end"], "near"); self.assertTrue(got[0]["net"]); self.assertEqual(got[0]["shot"], 2)
        self.assertAlmostEqual(got[0]["contact"], 1.5, delta=2 / fps)


class Guards(unittest.TestCase):
    def test_no_incoming_arc_after_a_shot_that_never_bounced(self):
        prev = dict(ok=True, bounce=None, t_bounce=1.0)
        self.assertIsNone(technique._incoming(prev, dict(t_contact=1.3)))


if __name__ == "__main__":
    unittest.main()
