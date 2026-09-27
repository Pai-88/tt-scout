"""Scout tips: the posteriors are exact, luck alone rarely produces a tip, thin or unreliable matches say nothing, and the note
is lettered, kept off the players and only shown when there is time to read it."""
import json, pathlib, random, tempfile, unittest
import numpy as np, cv2
from tt_scout.config import TABLE_WIDTH as W, NET_X
from tt_scout.quality import assess
from tt_scout.tips import (Tip, assign, match_tips, prob_greater, prob_majority, CHANCE_TIP, MAX_SHOWS)
from tests.test_product import TRUE, scene


def pt(i, server, winner, n=2, ending="not_returned", depth=0.3, y=0.2, near="A", far="B"):
    """One point whose serve lands `depth` metres past the net and `y` metres in from the server's right-hand edge."""
    end = "near" if server == near else "far"
    x_m, y_m = (NET_X + depth, y) if end == "near" else (NET_X - depth, W - y)
    return dict(id=i, server=server, serve_side=end, winner_name=winner, n_crossings=n, ending=ending, near_player=near, far_player=far,
                start_t=10.0 * i, end_t=10.0 * i + 3.0, events=[], landings=[dict(shot=1, side="far" if end == "near" else "near", t=10.0 * i + 0.5, x_m=x_m, y_m=y_m)])


def coin_flip_match(seed, n=24):
    r = random.Random(seed)
    return [pt(i + 1, "AB"[(i // 2) % 2], r.choice("AB"), n=r.choice((1, 2, 3, 4, 5, 6, 8)), depth=r.choice((0.3, 0.7, 1.2)), y=r.choice((0.15, 0.76, 1.4)))
            for i in range(n)]


class Posteriors(unittest.TestCase):
    def test_closed_forms_agree_with_hand_results_and_with_sampling(self):
        self.assertAlmostEqual(prob_greater(1, 1, 0, 0, prior=(1, 1)), 2 / 3, places=9)      # Beta(2, 1) against uniform: E[p] = 2/3
        self.assertAlmostEqual(prob_greater(3, 6, 3, 6), 0.5, places=9)
        self.assertAlmostEqual(prob_greater(4, 5, 2, 6) + prob_greater(2, 6, 4, 5), 1.0, places=9)
        self.assertAlmostEqual(prob_majority(3, 3), 57 / 64, places=12); self.assertAlmostEqual(prob_majority(0, 0), 0.5, places=12)
        rng = np.random.default_rng(0)
        mc = float((rng.beta(2 + 4, 2 + 1, 400000) > rng.beta(2 + 2, 2 + 4, 400000)).mean())
        self.assertAlmostEqual(prob_greater(4, 5, 2, 6), mc, places=2)
        self.assertGreater(prob_greater(2, 2, 0, 2), 0.85)      # the prior alone would let two lucky points through: MIN_N is what stops them


class Engine(unittest.TestCase):
    def test_a_real_serve_pattern_becomes_a_tip_on_the_points_that_show_it(self):
        pts = [pt(i + 1, "A", "A", depth=0.3) for i in range(10)] + [pt(i + 11, "A", "B", depth=1.2) for i in range(10)]
        pts += [pt(i + 21, "B", "AB"[(i // 2) % 2], depth=(0.3, 1.2)[i % 2]) for i in range(20)]     # B's serve: depth and winner unrelated
        tips = match_tips(pts)
        t = next(t for t in tips if t.rule == "serve_depth" and t.player == "A")
        self.assertEqual((t.head, t.level, t.family), ("Keep serving short", "tip", "outcome"))
        self.assertLessEqual(t.chance, CHANCE_TIP); self.assertEqual(sorted(t.shows_on), list(range(1, 11)))
        self.assertIn("won 10 of 10", t.evidence)
        self.assertFalse([t for t in tips if t.rule == "serve_depth" and t.player == "B"])

    def test_luck_alone_rarely_produces_a_tip(self):
        hits = sum(any(t.level == "tip" and t.family == "outcome" for t in match_tips(coin_flip_match(s), shuffles=150)) for s in range(60))
        self.assertLessEqual(hits, 12)                          # calibrated to about 1 match in 10; 12 of 60 would be 1 in 5

    def test_the_opponents_usual_serve_is_described_without_shuffling(self):
        pts = [pt(i + 1, "A", "AB"[i % 2], depth=0.7 if i else 1.2) for i in range(10)] + [pt(i + 11, "B", "AB"[i % 2], depth=(0.3, 0.7, 1.2)[i % 3]) for i in range(9)]
        t = next(t for t in match_tips(pts) if t.rule == "read_serve")
        self.assertEqual((t.player, t.head, t.family, t.level), ("B", "Expect the half-long serve", "describe", "tip"))
        self.assertAlmostEqual(t.chance, 1 - prob_majority(9, 10)); self.assertIn("9 of 10", t.evidence)

    def test_thin_unnamed_or_unreliable_matches_say_nothing(self):
        lucky = [pt(1, "A", "A", depth=0.3), pt(2, "A", "A", depth=0.3), pt(3, "A", "B", depth=1.2), pt(4, "A", "B", depth=1.2)]
        self.assertEqual(match_tips(lucky), [])                 # 2 of 2 against 0 of 2 is not evidence
        good = [pt(i + 1, "A", "A", depth=0.3) for i in range(10)] + [pt(i + 11, "A", "B", depth=1.2) for i in range(10)]
        self.assertTrue(match_tips(good)); self.assertEqual(match_tips(good, dict(level="unreliable")), [])
        self.assertEqual(match_tips([dict(p, near_player=None, far_player=None) for p in good]), [])

    def test_a_note_rides_on_few_clips_spread_out_and_only_where_the_point_shows_it(self):
        pts = [pt(i, "A", "A") for i in range(1, 13)]
        sure = Tip("read_serve", "B", "Expect the short serve", "A served short 9 of 9 times", 0.99, 9, list(range(1, 10)), chance=0.01)
        less = Tip("rally", "A", "Make the rally long", "5+ shots: won 4 of 4", 0.94, 4, [5, 6], chance=0.30)
        got = assign([less, sure], pts)
        self.assertEqual(sorted(i for i, t in got.items() if t is sure), [1, 5, 9])                  # first, middle, last
        self.assertEqual(sorted(i for i, t in got.items() if t is less), [6])                       # 5 was taken by the surer tip
        self.assertLessEqual(max(sum(1 for t in got.values() if t is x) for x in (sure, less)), MAX_SHOWS)
        self.assertFalse({10, 11, 12} & set(got))


class Note(unittest.TestCase):
    def test_the_note_is_lettered_and_kept_off_the_players(self):
        from tt_scout.annotate import tip_sprite, tip_spot, FAR_RGB, GAUGE_W, GAUGE_H, NOTE_BOTTOM, _overlap
        full = tip_sprite("tip", "Red", FAR_RGB, "Expect the half-long serve", "Black served half-long 9 of 10 times")
        burnt = tip_sprite("early", "A very long player name", FAR_RGB, "Serve into the middle less often " * 2, "Serves into the middle won 1 of 4 · the others 5 of 7 " * 2, left=3)
        for s in (full, burnt):
            self.assertEqual(s.shape[2], 4); self.assertGreater(int(s[..., 3].max()), 250)
        self.assertFalse(np.array_equal(full, tip_sprite("tip", "Red", FAR_RGB, "Expect the half-long serve", "Black served half-long 9 of 10 times", left=10)))
        h, w = full.shape[:2]

        def rect(spot, fw=960, fh=540):
            cx, cy, sc = spot
            return (cx - w * sc / 2, cy - h * sc / 2, cx + w * sc / 2, cy + h * sc / 2)

        def gauge(fh=540):
            return (0, fh - NOTE_BOTTOM - GAUGE_H - 8, 12 + GAUGE_W + 8, fh - NOTE_BOTTOM + 8)

        for fw in (960, 720, 640):                                                                   # 16:9, 4:3 and narrower: never on the gauge;
            r = rect(tip_spot((w, h), [], (fw, 540)))                                                # since 2026-09-26 on the bottom edge, under
            self.assertEqual(_overlap(r, gauge()), 0, fw); self.assertLessEqual(r[3], 540 - NOTE_BOTTOM + 6, fw)   # the table
            self.assertGreaterEqual(r[0], 0); self.assertLessEqual(r[2], fw)
        self.assertEqual(tip_spot((w, h), [], (960, 540))[2], 1.0)                                   # full size when nothing is in the way
        legs = (300, 380, 440, 540)                                                                  # the near player's legs next to the gauge
        r = rect(tip_spot((w, h), [legs], (960, 540)))
        self.assertLessEqual(_overlap(r, legs) / ((r[2] - r[0]) * (r[3] - r[1])), 0.02)
        feet = (700, 380, 900, 540)                                                                  # no room at full size: drawn smaller, clear
        spot = tip_spot((w, h), [feet], (960, 540)); r = rect(spot)
        self.assertLess(spot[2], 1.0); self.assertLessEqual(_overlap(r, feet) / ((r[2] - r[0]) * (r[3] - r[1])), 0.02)

    def test_a_clip_with_a_note_opens_in_the_lull_and_without_room_it_is_not_flashed(self):
        import shutil, subprocess
        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg not installed")
        from tt_scout.annotate import render_point_clip, TIP_LEAD_S
        from tt_scout.table import Table
        with tempfile.TemporaryDirectory() as d:
            src = str(pathlib.Path(d) / "src.mp4"); fps, n = 60.0, 540      # 9 s: room before the serve for the longer tip (2026-09-26)
            vw = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080)); base = scene()
            track = np.full((n, 6), np.nan); track[:, 0] = np.arange(n); track[:, 1] = np.arange(n) / fps
            for i in range(n):
                vw.write(base)
            vw.release()
            events = [dict(t=6.3, kind="net", side="far", x_px=930.0, y_px=540.0, x_m=1.37, y_m=0.7), dict(t=6.6, kind="bounce", side="far", x_px=1100.0, y_px=580.0, x_m=2.0, y_m=0.8)]
            point = dict(id=1, start_t=6.2, end_t=7.0, events=events, server="Red", serve_side="near", winner_name="Red", ending="not_returned", near_player="Red", far_player="Black")
            tip = Tip("read_serve", "Black", "Expect the half-long serve", "Red served half-long 9 of 10 times", 0.99, 10, [1], chance=0.01)

            def render(name, **kw):
                dst = pathlib.Path(d) / name
                self.assertTrue(render_point_clip(src, fps, Table(TRUE), track, events, {}, point, dst, **kw))
                dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(dst)], capture_output=True, text=True).stdout)
                return dst, dur

            def paper(clip, t):                                  # pixels of the note's paper in the band along the bottom where it sits
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", str(clip), "-frames:v", "1", str(pathlib.Path(d) / "f.png")], check=True)
                img = cv2.imread(str(pathlib.Path(d) / "f.png"))
                img = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA) if img.shape[0] != 540 else img   # clips are 1080p now
                band = img[440:535, 310:950].astype(int)
                # the note's paper is comic cream now (annotate.PAPER / PAPER_BOX, "no black or white", 2026-09-26): BGR within 30 of either
                cream = [np.array([190, 246, 255]), np.array([150, 238, 255])]
                return int(np.logical_or.reduce([np.abs(band - c).max(axis=2) < 30 for c in cream]).sum())

            plain, d0 = render("plain.mp4")
            noted, d1 = render("noted.mp4", tip=tip, max_lead_s=TIP_LEAD_S + 0.4)
            tight, d2 = render("tight.mp4", tip=tip, max_lead_s=0.5)
            self.assertAlmostEqual(d1 - d0, TIP_LEAD_S - 1.0, delta=0.15)                  # opens earlier, before the serve
            self.assertAlmostEqual(d2, d0, delta=0.05)                                    # no room to read it: the ordinary clip
            # there while it is read; gone by the end of the rally (since 2026-09-26 it stays 1.2 s into the rally, so it can be finished)
            self.assertGreater(paper(noted, 2.0), 8000); self.assertLess(paper(noted, 6.55), 300)
            self.assertLess(paper(plain, 0.5), 300)


class ReportSection(unittest.TestCase):
    def test_tips_are_listed_with_their_evidence_and_can_be_switched_off(self):
        from tt_scout.report_html import make_html
        pts = [pt(i + 1, "A", "A", depth=0.3) for i in range(10)] + [pt(i + 11, "A", "B", depth=1.2) for i in range(10)]
        q = assess([dict(n_crossings=3, n_implied_crossings=0, coverage=0.8)] * 2)
        with tempfile.TemporaryDirectory() as d:
            page = pathlib.Path(make_html(d, "A v B", pts, q, clips=False)[0]).read_text()
            saved = json.loads((pathlib.Path(d) / "tips.json").read_text())
            off = pathlib.Path(make_html(d, "A v B", pts, q, clips=False, tips=False)[0]).read_text()
            few = pathlib.Path(make_html(d, "A v B", pts[:3], q, clips=False)[0]).read_text()
            bad = pathlib.Path(make_html(d, "A v B", pts, dict(q, level="unreliable"), clips=False)[0]).read_text()
        for needle in ("Scout tips", "Keep serving short", "won 10 of 10", "shuffled matches"):
            self.assertIn(needle, page)
        self.assertTrue(saved and saved[0]["level"] == "tip" and "chance" in saved[0])
        self.assertNotIn("Scout tips", off)
        self.assertIn("No tips yet", few); self.assertIn("advice built on it would be guesswork", bad)


if __name__ == "__main__":
    unittest.main()
