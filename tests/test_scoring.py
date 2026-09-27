"""Behaviour the point logic must reproduce.   Run:  python -m unittest discover -s tests -v

These tests pin down tt_scout.events.segment_points and tt_scout.scoring.who_won so that they can be re-implemented
by hand (RESEARCH.md, item 6) without changing what the pipeline does. Timings use the Config defaults:
return_timeout_s = long_timeout_s = 1.5 s, dead_time_s = 2.5 s, serve_lead_s = 1.0 s.

Event vocabulary (events.py): kind "bounce" with side = the table half it landed on; kind "net" with side = the
half the ball crossed TO. Racket hits are ignored by both functions. The table is 2.74 m long, net at 1.37 m.
"""
import csv, pathlib, unittest
from tt_scout.config import Config
from tt_scout.events import segment_points
from tt_scout.scoring import who_won, other

ROOT = pathlib.Path(__file__).resolve().parent.parent
FPS = 120


def ev(t, kind, side, x_m=None):
    if kind == "net":
        x_m = 1.37
    elif x_m is None:
        x_m = 0.5 if side == "near" else 2.2
    return dict(frame=int(round(t * FPS)), t=float(t), kind=kind, side=side, x_px=0.0, y_px=0.0,
                x_m=x_m, y_m=0.76, strength=5.0, track=1)


def points(events, cfg=None):
    pts = segment_points(events, cfg or Config())
    for p in pts:
        p["winner"] = who_won(p)
    return pts


# serve by the near player: own-side bounce at 1.0 s, over the net at 1.2 s
SERVE_NEAR = [ev(1.0, "bounce", "near"), ev(1.2, "net", "far")]


class SegmentPointsAndWhoWon(unittest.TestCase):
    def test_serve_lands_and_is_never_returned_server_wins(self):
        pts = points(SERVE_NEAR + [ev(1.4, "bounce", "far")])
        self.assertEqual(len(pts), 1)
        p = pts[0]
        self.assertEqual(p["serve_side"], "near")
        self.assertEqual(p["ending"], "not_returned")
        self.assertEqual(p["winner"], "near")
        self.assertAlmostEqual(p["start_t"], 1.0, places=3)          # the serve's own-side bounce opens the point
        self.assertAlmostEqual(p["end_t"], 1.4 + 1.5, places=3)      # landing + return_timeout_s
        self.assertEqual(p["n_crossings"], 1)
        self.assertEqual([(l["shot"], l["side"]) for l in p["landings"]], [(1, "far")])

    def test_serve_goes_long_receiver_wins(self):
        pts = points(SERVE_NEAR)                                       # crossed, never landed, never came back
        self.assertEqual(len(pts), 1)
        self.assertEqual(pts[0]["ending"], "long")
        self.assertEqual(pts[0]["winner"], "far")
        self.assertAlmostEqual(pts[0]["end_t"], 1.2 + 1.5, places=3)

    def test_double_bounce_on_the_receivers_side(self):
        pts = points(SERVE_NEAR + [ev(1.4, "bounce", "far"), ev(1.9, "bounce", "far")])
        self.assertEqual(len(pts), 1)
        self.assertEqual(pts[0]["ending"], "double_bounce")
        self.assertEqual(pts[0]["winner"], "near")
        self.assertAlmostEqual(pts[0]["end_t"], 1.9 + 0.2, places=3)

    def test_rally_with_three_crossings(self):
        pts = points(SERVE_NEAR + [ev(1.4, "bounce", "far"), ev(1.7, "net", "near"), ev(1.9, "bounce", "near"),
                                   ev(2.2, "net", "far"), ev(2.4, "bounce", "far")])
        self.assertEqual(len(pts), 1)
        p = pts[0]
        self.assertEqual(p["n_crossings"], 3)
        self.assertEqual([(l["shot"], l["side"]) for l in p["landings"]], [(1, "far"), (2, "near"), (3, "far")])
        self.assertEqual(p["ending"], "not_returned")
        self.assertEqual(p["winner"], "near")                          # far never returned the third ball

    def test_retrieval_crossing_in_dead_time_is_ignored_and_next_point_is_served_by_far(self):
        first = SERVE_NEAR + [ev(1.4, "bounce", "far")]                # ends at 2.9 s, dead until 5.4 s
        retrieval = [ev(3.9, "net", "near")]                           # loser knocks the ball back: not play
        second = [ev(7.0, "bounce", "far"), ev(7.2, "net", "near"), ev(7.4, "bounce", "near")]
        pts = points(first + retrieval + second)
        self.assertEqual(len(pts), 2)
        self.assertEqual([p["serve_side"] for p in pts], ["near", "far"])
        self.assertEqual([p["winner"] for p in pts], ["near", "far"])
        self.assertAlmostEqual(pts[1]["start_t"], 7.0, places=3)
        self.assertEqual(pts[0]["n_crossings"], 1)

    def test_no_crossing_means_no_point(self):
        # a serve into the net leaves no crossing: documented blind spot, segment_points makes no point of it
        self.assertEqual(points([ev(1.0, "bounce", "near"), ev(1.1, "nethit", "near")]), [])


class WhoWonRules(unittest.TestCase):
    def rally(self, events, serve_side="near", coverage=1.0):
        return dict(events=events, serve_side=serve_side, coverage=coverage)

    def test_serve_never_crossed_receiver_wins(self):
        self.assertEqual(who_won(self.rally([ev(1.0, "bounce", "near")], serve_side="near")), "far")
        self.assertEqual(who_won(self.rally([ev(1.0, "bounce", "far")], serve_side="far")), "near")

    def test_ball_crossed_and_never_landed_the_receiver_of_that_crossing_wins(self):
        evs = [ev(1.0, "bounce", "near"), ev(1.2, "net", "far"), ev(1.4, "bounce", "far"), ev(1.7, "net", "near")]
        self.assertEqual(who_won(self.rally(evs)), "near")             # far's return went long

    def test_ball_crossed_and_landed_the_other_player_wins(self):
        evs = [ev(1.0, "bounce", "near"), ev(1.2, "net", "far"), ev(1.4, "bounce", "far"), ev(1.7, "net", "near"),
               ev(1.9, "bounce", "near")]
        self.assertEqual(who_won(self.rally(evs)), "far")              # landed on near's side, never came back

    def test_bounce_side_after_the_last_crossing_is_not_trusted(self):
        # the detector put the bounce on the wrong half; without another crossing it still counts as landed
        evs = [ev(1.0, "bounce", "near"), ev(1.2, "net", "far"), ev(1.4, "bounce", "near", x_m=1.3)]
        self.assertEqual(who_won(self.rally(evs)), "near")

    def test_unsure_when_coverage_is_low_or_nothing_landed(self):
        evs = [ev(1.0, "bounce", "near"), ev(1.2, "net", "far")]
        self.assertIsNone(who_won(self.rally(evs, coverage=0.1)))
        self.assertIsNone(who_won(self.rally([ev(1.2, "net", "far")])))

    def test_other(self):
        self.assertEqual(other("near"), "far"); self.assertEqual(other("far"), "near")


class RealClipRegression(unittest.TestCase):
    """The umpire-verified clip: 5 points, 5 servers right, 4 of 5 winners right (the miss is an edge-ball winner the
    detector classed as a racket hit, README 'Points and winners'). Skipped when the outputs are not on disk."""
    EVENTS, TRUTH = ROOT / "out/test_3/events.csv", ROOT / "labels/test_3.points.csv"

    def test_test_3_points_servers_and_winners(self):
        if not (self.EVENTS.exists() and self.TRUTH.exists()):
            self.skipTest("run eval_openttgames.py data/test_3 and tt_scout.points_truth first")
        from tt_scout.metrics import load_points_csv, match_points, summarise
        events = []
        for r in csv.DictReader(open(self.EVENTS)):
            events.append(dict(t=float(r["t"]), kind=r["kind"], side=r["side"], x_m=float(r["x_m"]), y_m=float(r["y_m"]),
                               frame=int(float(r["frame"])), strength=float(r["strength"])))
        pts = points(events)
        for p in pts:
            p["serve_side"], p["winner"] = p["serve_side"], p["winner"]
        pred = [dict(id=p["id"], start_t=p["start_t"], end_t=p["end_t"], server=p["serve_side"], winner=p["winner"]) for p in pts]
        pairs, fp, fn, _ = match_points(load_points_csv(self.TRUTH), pred)
        s = summarise(pairs, fp, fn, n_boot=50)
        self.assertEqual((s["tp"], s["fp"], s["fn"]), (5, 0, 0))
        self.assertEqual(s["server_right"], 5)
        self.assertGreaterEqual(s["winner_right"], 4)


if __name__ == "__main__":
    unittest.main()
