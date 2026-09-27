"""Player profiles: one record per video (re-analysing replaces it), names matched case-insensitively, totals pooled across
recordings, unreliable recordings kept out of the career numbers. """
import json, pathlib, tempfile, unittest
from tt_scout.profiles import record_match, load_records, aggregate, slug, tendencies
from tt_scout.profile_html import build_pages


def fake_match(a, b, a_pts, b_pts, level="good"):
    """Points where a wins a_pts then b wins b_pts; a serves the odd points; every point 3 shots, ended 'not_returned'."""
    pts = []
    for i in range(a_pts + b_pts):
        w = a if i < a_pts else b
        pts.append(dict(id=i + 1, start_t=10.0 * i, end_t=10.0 * i + 4, near_player=a, far_player=b, server=a if i % 2 == 0 else b,
                        winner_name=w, n_crossings=3, ending="not_returned", landings=[dict(shot=1, x_m=2.0, y_m=0.7, side="far")]))
    from tt_scout.match_stats import compute
    ms = compute(pts)
    return pts, ms, dict(level=level, score=90, reasons=[])


class Profiles(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.vid1 = self.tmp / "one.mp4"; self.vid1.write_bytes(b"x" * 5000)
        self.vid2 = self.tmp / "two.mp4"; self.vid2.write_bytes(b"y" * 7000)

    def test_the_same_video_is_one_record_however_often_it_is_analysed(self):
        pts, ms, q = fake_match("Sam", "Robin", 11, 5)
        for _ in range(3):
            record_match(self.tmp, self.vid1, {"near": "Sam", "far": "Robin"}, pts, ms, q, recorded="2026-09-25T23:00:00+01:00")
        self.assertEqual(len(load_records(self.tmp)), 1)

    def test_totals_pool_across_recordings_and_names_ignore_case(self):
        p1, m1, q1 = fake_match("Sam", "Robin", 11, 5)
        p2, m2, q2 = fake_match("robin", "SAM", 11, 9)                     # the second time the names are typed differently
        record_match(self.tmp, self.vid1, {"near": "Sam", "far": "Robin"}, p1, m1, q1, recorded="2026-09-25T23:00:00+01:00")
        record_match(self.tmp, self.vid2, {"near": "robin", "far": "SAM"}, p2, m2, q2, recorded="2026-09-26T20:00:00+01:00")
        a = aggregate(slug("Sam"), load_records(self.tmp))
        self.assertEqual(a["n"], 2)
        self.assertEqual((a["tot"]["won"], a["tot"]["lost"]), (1, 1))
        self.assertEqual(a["tot"]["points"], [11 + 9, 5 + 11])
        self.assertEqual([r["res"] for r in a["rows"]], ["W", "L"])          # oldest first
        self.assertEqual(a["name"], "SAM")                                   # the latest spelling is shown

    def test_an_unreliable_recording_is_listed_but_not_counted(self):
        p1, m1, q1 = fake_match("Sam", "Robin", 11, 5)
        p2, m2, q2 = fake_match("Sam", "Robin", 2, 11, level="unreliable")
        record_match(self.tmp, self.vid1, {"near": "Sam", "far": "Robin"}, p1, m1, q1, recorded="2026-09-25T23:00:00+01:00")
        record_match(self.tmp, self.vid2, {"near": "Sam", "far": "Robin"}, p2, m2, q2, recorded="2026-09-26T20:00:00+01:00")
        a = aggregate("sam", load_records(self.tmp))
        self.assertEqual((a["n"], a["n_career"]), (2, 1))
        self.assertEqual(a["tot"]["points"], [11, 5])
        self.assertEqual(len(a["rows"]), 2)

    def test_pages_are_built_for_every_player_and_the_index(self):
        p1, m1, q1 = fake_match("Sam", "Robin", 11, 5)
        record_match(self.tmp, self.vid1, {"near": "Sam", "far": "Robin"}, p1, m1, q1, recorded="2026-09-25T23:00:00+01:00")
        pages = build_pages(self.tmp)
        self.assertEqual(set(pages), {"Sam", "Robin"})
        self.assertIn("Sam", (self.tmp / "index.html").read_text())
        self.assertIn("Head to head", pages["Robin"].read_text())

    def test_no_habit_is_stated_on_too_few_cases(self):
        p1, m1, q1 = fake_match("Sam", "Robin", 3, 2)
        record_match(self.tmp, self.vid1, {"near": "Sam", "far": "Robin"}, p1, m1, q1, recorded="2026-09-25T23:00:00+01:00")
        self.assertEqual(tendencies(aggregate("sam", load_records(self.tmp))), [])


if __name__ == "__main__":
    unittest.main()
