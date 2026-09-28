"""When a point is seen to end (match_stats.confirm) and what the speed gauge reads (flight.Speeds.reading).

Both failed in the clips of 2026-09-28: a tracker that jumped onto a player's knee during a tracking gap was read as the ball
dropping to the floor and celebrated a point 6.5 s before the rally ended, and the gauge kept the previous shot's number up
through a hard hit that was not measured, so a smash read slower than the block before it."""
import unittest
import numpy as np
from tt_scout.table import Table
from tt_scout.match_stats import confirm, ball_path
from tt_scout import flight
from tt_scout.flight import Speeds, DASH, Approx, READ_AFTER_CONTACT_S

CORNERS = [[518, 773], [649, 667], [1228, 671], [1379, 773]]      # a side view, 1920x1080, near end on the left
FPS = 60.0


def track_of(samples, n=400):
    """A track array (frame, t, u, v, id) with the given {frame: (u, v)} and nothing elsewhere."""
    tr = np.full((n, 5), np.nan); tr[:, 0] = np.arange(n); tr[:, 1] = np.arange(n) / FPS; tr[:, 4] = -1
    for f, (u, v) in samples.items():
        tr[f, 2:4] = (u, v); tr[f, 4] = 1
    return tr


class PointEnd(unittest.TestCase):
    def setUp(self):
        self.table = Table(CORNERS)
        self.net = self.table.to_px([[1.37, 0.5]])[0]

    def test_a_jump_onto_a_knee_is_not_the_ball_falling(self):
        # the last shot crosses to the far half at t = 1.0 and flies on; the tracker then jumps onto the near player's knee, below the
        # table's edge and wobbling down, and the ball is never seen again
        s = {60 + k: (self.net[0] + 12 * k, self.net[1] + 2 * k) for k in range(1, 6)}
        s.update({66 + k: (412.0, 853.0 + 4 * (k % 3)) for k in range(0, 30)})
        cross = dict(kind="net", t=1.0, side="far", x_px=float(self.net[0]), y_px=float(self.net[1]))
        got = confirm(dict(events=[cross]), [cross], track_of(s), self.table, FPS)
        self.assertFalse(got[0], got)

    def test_a_ball_seen_dropping_past_the_end_ends_the_point(self):
        s = {}
        for k in range(1, 40):                                        # past the far end at table height, then falling to the floor
            s[60 + k] = (self.net[0] + 20 * k, self.net[1] + (3 * k if k <= 20 else 60 + 14 * (k - 20)))
        cross = dict(kind="net", t=1.0, side="far", x_px=float(self.net[0]), y_px=float(self.net[1]))
        ok, t, how = confirm(dict(events=[cross]), [cross], track_of(s), self.table, FPS)
        self.assertTrue(ok); self.assertEqual(how, "missed the table")
        first = next(f for f in sorted(s) if s[f][1] > 773 + 40 and s[f][0] > 1379 + 12)
        self.assertAlmostEqual(t, first / FPS + 0.1, delta=1.5 / FPS)

    def test_a_return_into_the_net_is_not_made_a_winner_by_a_jump_onto_a_shoe(self):
        # crosses to the near half at 1.0, bounces at 1.2, the near player hits it back at 1.5 and it dies in the net at 1.75;
        # then the tracker holds his shoe, below the table's edge beyond the near end
        nx, ny = self.net
        s = {}
        for f in range(61, 72):
            s[f] = (nx - 16 * (f - 60), ny + 5 * (f - 60))           # to the bounce
        for f in range(72, 90):
            s[f] = (s[71][0] - 12 * (f - 71), s[71][1] - 2 * (f - 71))    # on towards the hitter
        for f in range(91, 105):
            s[f] = (s[89][0] + 25 * (f - 90), s[89][1] + 1.0 * (f - 90))  # back towards the net
        s = {f: p for f, p in s.items() if p[0] <= nx - 4}
        s.update({106 + k: (437.0, 865.0 + 3 * k) for k in range(6)})
        bounce_f = 72
        cross = dict(kind="net", t=1.0, side="near", x_px=float(nx), y_px=float(ny))
        bounce = dict(kind="bounce", t=bounce_f / FPS, side="near", x_px=float(s[bounce_f][0]), y_px=float(s[bounce_f][1]))
        ok, t, how = confirm(dict(events=[cross, bounce]), [cross, bounce], track_of(s), self.table, FPS)
        self.assertTrue(ok); self.assertEqual(how, "return into the net")
        self.assertAlmostEqual(t, max(f for f in s if f < 106) / FPS + 0.25, delta=2 / FPS)

    def test_the_ball_path_does_not_hop_across_a_gap_to_somebody_else(self):
        s = {60 + k: (900.0 + 15 * k, 650.0) for k in range(1, 9)}   # 15 px a frame, then lost ...
        s.update({74 + k: (1435.0, 696.0) for k in range(5)})         # ... and six frames on the tracker holds the far player's hand
        idx = ball_path(track_of(s), 1.0, 900.0, 650.0, FPS)
        self.assertEqual(list(idx), [60 + k for k in range(1, 9)])


def fit(shot, t_contact, speed, **kw):
    return dict(dict(point=1, shot=shot, ok=True, t_contact=t_contact, t_bounce=t_contact + 0.4, speed=speed, contact_src="seen"), **kw)


class GaugeReading(unittest.TestCase):
    def setUp(self):
        events = [dict(kind="net", t=1.0, x_px=1.0), dict(kind="net", t=1.8, implied=True), dict(kind="net", t=2.6, x_px=1.0),
                  dict(kind="bounce", t=2.9)]
        self.point = dict(id=1, events=events)
        self.fits = [fit(1, 0.8, 8.0), fit(3, 2.4, 12.0, free=True), fit(4, 3.3, 6.0, net=True, t_cross=None)]

    def test_each_shot_reads_its_own_speed_from_just_after_its_contact(self):
        sp = Speeds(self.fits, points=[self.point])
        self.assertIsNone(sp.reading(0.8 + READ_AFTER_CONTACT_S - 0.01))          # nothing before the first shot
        self.assertEqual(sp.reading(0.95), 8.0)
        self.assertEqual(sp.reading(1.75), DASH)                                  # the shot over the inferred crossing: not measured
        self.assertIsInstance(sp.reading(2.55), Approx); self.assertAlmostEqual(sp.reading(2.55), 12.0)   # no bounce: an estimate
        self.assertEqual(sp.reading(3.5), 6.0)                                    # the return into the net reads too

    def test_a_hard_hit_never_shows_the_slower_shot_before_it(self):
        # the old gauge (last(): the latest measured shot by its bounce) kept 8.0 up through the unmeasured shot 2
        sp = Speeds(self.fits, points=[self.point])
        self.assertEqual(sp.last(2.0), 8.0)
        self.assertEqual(sp.reading(2.0), DASH)

    def test_a_hidden_contact_that_was_not_pinned_down_reads_as_not_measured(self):
        sp = Speeds([fit(1, 0.8, 5.0, contact_src="late")], points=[dict(id=1, events=[dict(kind="net", t=1.0, x_px=1.0)])])
        self.assertEqual(sp.reading(1.2), DASH)

    def test_the_dash_is_held_like_a_number(self):
        from tt_scout.annotate import Held
        h = Held(hold=0.35)
        self.assertEqual(h.update(0.0, 5.0), 5.0)
        self.assertEqual(h.update(0.05, 5.1), 5.0); self.assertIsNone(h.pending)   # 18 km/h either way: nothing to change
        self.assertEqual(h.update(0.1, DASH), 5.0)                                # waits its turn
        self.assertEqual(h.update(0.4, DASH), DASH); self.assertAlmostEqual(h.changed, 0.4)
        self.assertEqual(h.update(0.5, DASH), DASH); self.assertAlmostEqual(h.changed, 0.4)
        self.assertEqual(h.update(0.6, 7.0), DASH)
        self.assertEqual(h.update(0.8, 7.0), 7.0)
        self.assertEqual(h.update(0.9, 5.0), 7.0); self.assertEqual(h.update(1.0, 7.0), 7.0)   # a change that went back is forgotten
        self.assertEqual(h.update(1.3, 7.0), 7.0); self.assertAlmostEqual(h.changed, 0.8)
        self.assertEqual(h.update(1.4, 24 / 3.6), 24 / 3.6)                     # 24 after 25 km/h: both round to 7 m/s, but print apart


class NetStrikeNumbering(unittest.TestCase):
    def test_a_return_into_the_net_is_numbered_after_every_crossing_inferred_ones_too(self):
        t = np.arange(0, 3.0, 1 / FPS); u = np.full(len(t), np.nan); v = np.full(len(t), 600.0)
        a = (t >= 0.8) & (t < 1.5); u[a] = 1100 - (t[a] - 0.8) * 540
        b = (t >= 1.5) & (t <= 1.8); u[b] = 722 + (t[b] - 1.5) * 600
        track = np.c_[np.arange(len(t)), t, u, v, np.ones(len(t))]
        evs = [dict(kind="net", t=0.4, side="far", implied=True), dict(kind="net", t=1.0, side="near", x_px=960.0, y_px=600.0),
               dict(kind="bounce", t=1.2, side="near", x_px=700.0, y_px=640.0)]
        got = flight.net_strikes([dict(id=1, ending="not_returned", events=evs)], evs, track, FPS)
        self.assertEqual(len(got), 1); self.assertEqual(got[0]["shot"], 3)


if __name__ == "__main__":
    unittest.main()
