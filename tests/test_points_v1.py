"""Behaviour of the repaired point logic (tt_scout/points_v1.py). Run: python -m unittest discover -s tests -v
Event vocabulary as in test_scoring.py: bounce.side = half it landed on; net.side = half crossed TO; net at x = 1.37 m."""
import unittest
from tt_scout.points_v1 import repair, segment_points_v1, V1Config


def ev(t, kind, side, x_m=None):
    if kind == "net":
        x_m = 1.37
    elif x_m is None:
        x_m = 0.5 if side == "near" else 2.3
    return dict(t=float(t), kind=kind, side=side, x_m=x_m, y_m=0.76)


SERVE = [ev(1.0, "bounce", "near"), ev(1.2, "net", "far"), ev(1.4, "bounce", "far", 2.3)]      # 0.93 m in 0.2 s = 4.65 m/s


class Repair(unittest.TestCase):
    def test_bounces_on_opposite_sides_imply_the_missed_crossing(self):
        seq = repair(SERVE + [ev(1.9, "bounce", "near")])                     # the return's crossing was not detected
        kinds = [(e["kind"], e["side"], e["implied"]) for e in seq]
        self.assertEqual(kinds, [("bounce", "near", False), ("net", "far", False), ("bounce", "far", False),
                                 ("net", "near", True), ("bounce", "near", False)])
        self.assertTrue(1.4 < seq[3]["t"] < 1.9)

    def test_a_bounce_next_to_the_net_does_not_vote(self):
        seq = repair(SERVE + [ev(1.6, "bounce", "near", 1.30)])               # 7 cm from the net: side is not trusted
        self.assertEqual([e["kind"] for e in seq].count("net"), 1)
        self.assertEqual(seq[-1]["side"], "far")                              # kept on the half the ball was on

    def test_duplicate_crossing_is_dropped_and_a_contradicting_one_implies_a_crossing_back(self):
        self.assertEqual(sum(e["kind"] == "net" for e in repair([ev(1.2, "net", "far"), ev(1.25, "net", "far")])), 1)
        seq = repair([ev(1.2, "net", "far"), ev(2.0, "net", "far")])
        self.assertEqual([(e["side"], e["implied"]) for e in seq if e["kind"] == "net"], [("far", False), ("near", True), ("far", False)])


class PointsV1(unittest.TestCase):
    def test_serve_not_returned(self):
        pts = segment_points_v1(SERVE)
        self.assertEqual(len(pts), 1)
        self.assertEqual((pts[0]["serve_side"], pts[0]["ending"], pts[0]["winner"]), ("near", "not_returned", "near"))
        self.assertAlmostEqual(pts[0]["start_t"], 1.0 - 0.3, places=3)        # serve contact precedes its own-side bounce

    def test_missed_crossing_no_longer_reads_as_a_double_bounce(self):
        # far returns (crossing missed), the ball lands on near's side and near fails: v0 called a double bounce for near
        pts = segment_points_v1(SERVE + [ev(1.9, "bounce", "near")])
        self.assertEqual(len(pts), 1)
        self.assertEqual((pts[0]["n_crossings"], pts[0]["ending"], pts[0]["winner"]), (2, "not_returned", "far"))

    def test_double_bounce_needs_the_ball_not_to_come_back(self):
        real = segment_points_v1(SERVE + [ev(1.8, "bounce", "far", 2.0)])
        self.assertEqual((real[0]["ending"], real[0]["winner"]), ("double_bounce", "near"))
        # a second "bounce" followed by a return within the timeout was a false detection: the rally goes on
        went_on = segment_points_v1(SERVE + [ev(1.6, "bounce", "far", 2.5), ev(1.9, "net", "near"), ev(2.1, "bounce", "near")])
        self.assertEqual((went_on[0]["n_crossings"], went_on[0]["ending"], went_on[0]["winner"]), (2, "not_returned", "far"))

    def test_long_ball(self):
        pts = segment_points_v1(SERVE + [ev(1.8, "net", "near")])             # far's return crosses and never lands
        self.assertEqual((pts[0]["ending"], pts[0]["winner"]), ("long", "near"))

    def test_knock_back_is_not_a_point(self):
        slow = [ev(1.0, "bounce", "far"), ev(1.2, "net", "near"), ev(1.6, "bounce", "near", 0.4)]      # 0.97 m in 0.4 s = 2.4 m/s
        caught = [ev(5.0, "net", "near")]                                     # crosses and is caught: never lands
        self.assertEqual(segment_points_v1(slow), [])
        self.assertEqual(segment_points_v1(caught), [])
        # a dropped knock-back does not start a dead time: the serve right after it is kept
        pts = segment_points_v1(slow + [ev(t + 1.5, k, s, x) for t, k, s, x in [(1.0, "bounce", "near", 0.5), (1.2, "net", "far", None), (1.4, "bounce", "far", 2.3)]])
        self.assertEqual(len(pts), 1)

    def test_retrieval_in_the_dead_time_is_ignored(self):
        pts = segment_points_v1(SERVE + [ev(3.5, "net", "near"), ev(3.7, "bounce", "near", 0.4)])
        self.assertEqual(len(pts), 1)


if __name__ == "__main__":
    unittest.main()
