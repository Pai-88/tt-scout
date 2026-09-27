"""Unit tests for the point-level ground truth converter, the matching criteria, the bootstrap and the labeller store."""
import pathlib, tempfile, unittest
import numpy as np
from tt_scout.points_truth import from_extended_labels, write_points
from tt_scout.metrics import load_points_csv, match_points, summarise, prf, boot_ci, iou, covers


def T(start, end, winner="near", server="near", ending="net", known=1):
    return dict(id="t", start_t=start, end_t=end, winner=winner, server=server, ending=ending, start_known=known)


def P(start, end, winner="near", server="near"):
    return dict(id="p", start_t=start, end_t=end, winner=winner, server=server, ending="x")


class ExtendedLabels(unittest.TestCase):
    def test_serve_and_ending_make_a_point(self):
        lab = {985: "left_forehand_serve neutral unknown", 1001: "bounce", 1180: "left_net", 1191: "bounce",
               1322: "empty_event", 2222: "left_forehand_serve neutral unknown", 2331: "right_net"}
        pts = from_extended_labels(lab, 120)
        self.assertEqual(len(pts), 2)
        self.assertEqual((pts[0]["start_frame"], pts[0]["end_frame"], pts[0]["server"], pts[0]["winner"], pts[0]["ending"]),
                         (985, 1180, "near", "far", "net"))                # near (left) netted -> far wins
        self.assertEqual((pts[1]["server"], pts[1]["winner"]), ("near", "near"))   # far (right) netted -> near wins
        self.assertEqual([p["id"] for p in pts], [1, 2])

    def test_winner_label_means_the_named_side_wins(self):
        pts = from_extended_labels({10: "right_backhand_serve x y", 50: "right_winner"}, 120)
        self.assertEqual((pts[0]["server"], pts[0]["winner"], pts[0]["ending"]), ("far", "far", "winner"))

    def test_ending_without_a_serve_keeps_the_point_with_unknown_start(self):
        pts = from_extended_labels({100: "bounce", 150: "right_out"}, 100)
        self.assertEqual(len(pts), 1)
        self.assertEqual((pts[0]["start_known"], pts[0]["start_frame"], pts[0]["server"], pts[0]["winner"]), (0, 100, "", "near"))

    def test_serve_without_an_ending_is_unknown_and_spans_to_the_next_serve(self):
        pts = from_extended_labels({100: "left_forehand_serve a b", 200: "bounce", 300: "right_backhand_serve a b", 400: "left_net"}, 100)
        self.assertEqual(len(pts), 2)
        self.assertEqual((pts[0]["ending"], pts[0]["end_frame"], pts[0]["winner"]), ("unknown", 300, ""))
        self.assertEqual((pts[1]["start_frame"], pts[1]["winner"]), (300, "far"))

    def test_double_bounce_means_the_named_striker_wins(self):
        # README: "double_bounce: the ball bounces twice on the OPPONENT's side before they can return it"
        pts = from_extended_labels({10: "left_backhand_push a b", 40: "net", 60: "bounce", 90: "left_double_bounce"}, 120)
        self.assertEqual((pts[0]["winner"], pts[0]["ending"]), ("near", "double_bounce"))

    def test_a_point_never_spans_two_rallies(self):
        # serve and rally with no ending label, 17 s of nothing, then a rally with strokes and an ending but no serve label
        lab = {1000: "left_backhand_serve a b", 1020: "bounce", 1040: "net", 1100: "right_backhand_push a b", 1150: "empty_event",
               3000: "bounce", 3030: "net", 3060: "left_forehand_loop a b", 3100: "net", 3140: "right_net"}
        pts = from_extended_labels(lab, 120)
        self.assertEqual([(p["ending"], p["start_known"]) for p in pts], [("unknown", 1), ("net", 0)])
        self.assertEqual((pts[0]["start_frame"], pts[0]["end_frame"]), (1000, 1150))
        self.assertEqual((pts[1]["start_frame"], pts[1]["end_frame"], pts[1]["winner"]), (3000, 3140, "near"))

    def test_ball_events_alone_are_not_a_rally(self):
        # the ball knocked back to the server: bounce, net, bounce and nothing else -> no row, so a prediction there is false
        self.assertEqual(from_extended_labels({500: "bounce", 530: "net", 560: "bounce", 600: "empty_event"}, 120), [])
        # strokes without serve or ending are a rally of unknown outcome -> predictions inside it are ignored
        pts = from_extended_labels({500: "bounce", 530: "left_forehand_push a b", 560: "net"}, 120)
        self.assertEqual([(p["ending"], p["start_known"]) for p in pts], [("unknown", 0)])

    def test_csv_roundtrip(self):
        pts = from_extended_labels({10: "left_forehand_serve a b", 50: "right_out"}, 100, source="test")
        with tempfile.TemporaryDirectory() as d:
            path = write_points(pts, pathlib.Path(d) / "x.points.csv")
            back = load_points_csv(path)
        self.assertEqual(len(back), 1)
        self.assertAlmostEqual(back[0]["start_t"], 0.1); self.assertAlmostEqual(back[0]["end_t"], 0.5)
        self.assertEqual((back[0]["server"], back[0]["winner"], back[0]["start_known"]), ("near", "near", 1))


class Matching(unittest.TestCase):
    def test_cover_accepts_a_late_end_that_iou_rejects(self):
        t, p = T(1.0, 2.0), P(0.9, 3.4)                                   # decisive event + 1.5 s timeout
        self.assertTrue(covers(t, p)); self.assertLess(iou(t, p), 0.5)
        pairs, fp, fn, _ = match_points([t], [p], criterion="cover")
        self.assertEqual((len(pairs), len(fp), len(fn)), (1, 0, 0))
        pairs, fp, fn, _ = match_points([t], [p], criterion="iou", iou_thresh=0.5)
        self.assertEqual((len(pairs), len(fp), len(fn)), (0, 1, 1))

    def test_cover_needs_the_serve_inside_the_prediction(self):
        self.assertFalse(covers(T(1.0, 2.0), P(1.6, 2.5)))               # prediction starts after the serve + tolerance
        self.assertTrue(covers(T(1.0, 2.0, known=0), P(1.6, 2.5)))       # unknown serve frame: only the end is tested
        self.assertFalse(covers(T(1.0, 2.0), P(3.0, 4.0)))               # ends before the prediction begins

    def test_one_to_one_and_largest_overlap_first(self):
        t1, t2 = T(1.0, 2.0), T(4.0, 5.0)
        p_big, p_two = P(0.8, 5.6), P(3.9, 5.2)
        pairs, fp, fn, _ = match_points([t1, t2], [p_big, p_two])
        self.assertEqual(len(pairs), 2); self.assertEqual((len(fp), len(fn)), (0, 0))
        got = {id(t): id(p) for t, p in pairs}
        self.assertEqual(got[id(t2)], id(p_two))                          # t2 takes the tighter prediction (IoU tie-break)
        self.assertEqual(got[id(t1)], id(p_big))

    def test_predictions_inside_unknown_points_are_ignored_not_false(self):
        unknown = T(5.0, 7.0, winner=None, server="far", ending="unknown")
        pairs, fp, fn, ign = match_points([unknown], [P(5.5, 6.5)])
        self.assertEqual((len(pairs), len(fp), len(fn), len(ign)), (0, 0, 0, 1))
        cut = T(26.0, None, winner=None, server="far", ending="unknown")     # the clip ends inside this point
        pairs, fp, fn, ign = match_points([cut], [P(27.4, 29.7), P(20.0, 22.0)])
        self.assertEqual((len(fp), len(ign)), (1, 1))


class Statistics(unittest.TestCase):
    def test_prf(self):
        self.assertEqual(prf([0, 0, 1, 2]), (2 / 3, 2 / 3, 2 / 3))
        self.assertEqual(prf([2, 2])[2], 0.0)
        self.assertEqual(prf([0, 0]), (1.0, 1.0, 1.0))

    def test_summarise_counts_unsure_as_wrong(self):
        pairs = [(T(1, 2, winner="near"), P(1, 2.5, winner=None, server="far")),
                 (T(4, 5, winner="far", server="far"), P(4, 5.5, winner="far", server="far"))]
        s = summarise(pairs, [], [], n_boot=50)
        self.assertEqual((s["winner_n"], s["winner_right"], s["winner_unsure"]), (2, 1, 1))
        self.assertEqual((s["server_n"], s["server_right"]), (2, 1))
        self.assertAlmostEqual(s["end_err_median_s"], 0.5)
        self.assertEqual(s["f1"], 1.0)

    def test_bootstrap_is_seeded_and_brackets_the_estimate(self):
        hits = [1, 1, 1, 0, 1, 0, 1, 1]
        lo, hi = boot_ci(hits, np.mean, n_boot=500, seed=1)
        self.assertEqual((lo, hi), boot_ci(hits, np.mean, n_boot=500, seed=1))
        self.assertLessEqual(lo, np.mean(hits)); self.assertLessEqual(np.mean(hits), hi)
        self.assertTrue(all(np.isnan(v) for v in boot_ci([], np.mean)))


class LabellerStore(unittest.TestCase):
    def test_open_close_ending_undo_save(self):
        from label_points import PointStore
        with tempfile.TemporaryDirectory() as d:
            st = PointStore("match.mp4", fps=60, labels_dir=d)
            st.serve(120, "near"); r = st.point_over(300, "far"); st.set_ending("out")
            self.assertEqual((r["start_frame"], r["end_frame"], r["server"], r["winner"], st.rows[-1]["ending"]), (120, 300, "near", "far", "out"))
            self.assertEqual(st.undo(), "ending"); self.assertEqual(st.rows[-1]["ending"], "unknown")
            self.assertEqual(st.undo(), "close"); self.assertEqual(st.rows, []); self.assertEqual(st.open["server"], "near")
            st.point_over(310, "near")
            st.point_over(900, "far")                                  # no serve: clip started inside the point
            self.assertEqual([r["start_known"] for r in st.rows], [1, 0])
            path = st.save()
            self.assertEqual([r["winner"] for r in load_points_csv(path)], ["near", "far"])
            self.assertAlmostEqual(load_points_csv(path)[0]["start_t"], 2.0)


if __name__ == "__main__":
    unittest.main()
