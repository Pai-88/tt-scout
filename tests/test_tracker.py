"""The ball tracker waits for a fast ball it has followed for a while instead of jumping to three dots in a line elsewhere the moment
it misses the ball for two frames (our 2026-09-26 videos: a white ball crossing in front of a light table further back vanishes for
about 4 frames right at the net, and the crossing was lost). """
import unittest
from tt_scout.config import Config
from tt_scout.detector import Candidate
from tt_scout.tracker import BallTracker


def run(cfg, n=40, gap=range(20, 24), ball_area=120.0, area_ref=None, net_x=None):
    """A ball moving 17 px a frame to the right, unseen for the frames in gap; during the gap a decoy moves in a straight line
    far away (a shoe, another table's ball). Returns the track id the tracker had on each frame it reported a position."""
    trk = BallTracker(cfg, fps=60.0); trk.area_ref = area_ref; ids = []
    if net_x is not None:
        trk.net_line = [(net_x, 500.0), (net_x, 700.0)]               # a vertical net line in the picture
    for f in range(n):
        cands = [] if f in gap else [Candidate(700.0 + 17 * f, 577.0, ball_area, 1.0)]
        if f >= gap.start:                                             # the decoy: three and more dots on a line, 400 px away
            cands.append(Candidate(300.0 + 12 * (f - gap.start), 900.0, 110.0, 1.0))
        c = trk.update(cands)
        ids.append((f, trk.track_id, None if c is None else (round(c.x), round(c.y))))
    return ids


class WaitsForTheBall(unittest.TestCase):
    def test_keeps_one_track_through_a_short_gap(self):
        ids = run(Config())
        before = {i for f, i, c in ids if 10 <= f < 20}; after = {i for f, i, c in ids if 24 <= f < 40}
        self.assertEqual(len(before), 1)
        self.assertEqual(before, after)                                # the same track picks the ball up again after the gap
        self.assertTrue(all(c is not None and c[1] == 577 for f, i, c in ids if 24 <= f < 40))   # and it is the ball, not the decoy

    def test_old_rule_lost_it(self):
        cfg = Config(); cfg.preempt_missed_healthy = cfg.preempt_missed     # the behaviour before the change
        ids = run(cfg)
        self.assertTrue(any(c is not None and c[1] == 900 for f, i, c in ids if 20 <= f < 26))   # it jumped to the decoy

    def test_waits_for_a_ball_the_size_of_this_tables(self):
        ids = run(Config(), area_ref=125.0)                            # this table's ball: 120 px against 125 expected
        self.assertTrue(all(c is not None and c[1] == 577 for f, i, c in ids if 24 <= f < 40))

    def test_does_not_wait_for_a_ball_on_a_table_behind(self):
        ids = run(Config(), ball_area=25.0, area_ref=125.0)            # a 25 px ball: a game further back, let it go as before
        self.assertTrue(any(c is not None and c[1] == 900 for f, i, c in ids if 20 <= f < 26))

    def test_waits_at_the_net_only(self):
        at_net = run(Config(), area_ref=125.0, net_x=1070.0)           # the gap (x 1040 to 1110) is at the net: waited for
        self.assertTrue(all(c is not None and c[1] == 577 for f, i, c in at_net if 24 <= f < 40))
        far = run(Config(), area_ref=125.0, net_x=300.0)               # the same gap far from the net (at a racket): let go as before
        self.assertTrue(any(c is not None and c[1] == 900 for f, i, c in far if 20 <= f < 26))

    def test_a_young_track_still_gives_way(self):
        ids = run(Config(), gap=range(4, 8))                           # only 4 frames old when it loses the ball: not waited for
        self.assertTrue(any(c is not None and c[1] == 900 for f, i, c in ids if 4 <= f < 10))


if __name__ == "__main__":
    unittest.main()
