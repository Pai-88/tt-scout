"""Product-layer behaviour: zero-click calibration, confidence levels, shirt reference, HTML report."""
import pathlib, tempfile, unittest
import numpy as np, cv2
from tt_scout.autocal import auto_table_from_frames
from tt_scout.quality import assess
from tt_scout.players import auto_reference
from tt_scout.report_html import make_html

TRUE = np.array([[360, 640], [560, 520], [1340, 535], [1480, 660]], float)       # 1 BL, 2 TL, 3 TR, 4 BR


def hsv_bgr(h, s, v):
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0])


def scene(occlude=None):
    """Orange floor, a blue backdrop across the top that TOUCHES the far edge of a brighter blue table, a dark net band."""
    f = np.full((1080, 1920, 3), hsv_bgr(18, 200, 170), np.uint8)
    f[:528] = hsv_bgr(112, 237, 120)                                             # backdrop, same hue family as the table
    cv2.fillConvexPoly(f, TRUE.astype(np.int32), hsv_bgr(122, 254, 174))
    cv2.line(f, (930, 524), (905, 652), (25, 25, 25), 26)                        # the net cuts the table top in two
    if occlude is not None:
        cv2.rectangle(f, occlude[0], occlude[1], hsv_bgr(125, 240, 160) if occlude[2] else (40, 40, 60), -1)
    return f


class AutoCalibration(unittest.TestCase):
    def test_table_is_found_next_to_a_blue_backdrop_and_hidden_corners_are_recovered(self):
        frames = []
        for k in range(12):
            occ = None
            if k % 4 != 0:                                                       # a player hides the far-right corner in 3 of 4 frames
                occ = ((1290, 430), (1460, 600), False)
            if k in (2, 7):                                                      # and twice someone in a table-blue shirt stands at the left end
                occ = ((250, 560), (380, 700), True)
            frames.append((k, scene(occ)))
        corners, info = auto_table_from_frames(iter(frames), len(frames))
        self.assertIsNotNone(corners, info)
        err = np.hypot(*(corners - TRUE).T)
        self.assertLess(err.max(), 6.0, f"corner errors {err.round(1)}")
        self.assertEqual(info["colour"], "blue")

    def test_no_table_gives_a_reason_not_a_guess(self):
        floor = np.full((1080, 1920, 3), hsv_bgr(18, 200, 170), np.uint8)
        corners, info = auto_table_from_frames(iter([(k, floor) for k in range(6)]), 6)
        self.assertIsNone(corners); self.assertIn("reason", info)


class Confidence(unittest.TestCase):
    def pts(self, n, implied, coverage):
        return [dict(n_crossings=10, n_implied_crossings=int(10 * implied), coverage=coverage) for _ in range(n)]

    def test_levels(self):
        self.assertEqual(assess(self.pts(20, 0.1, 0.7))["level"], "good")
        self.assertEqual(assess(self.pts(20, 0.5, 0.5))["level"], "degraded")
        self.assertEqual(assess(self.pts(20, 0.8, 0.35))["level"], "unreliable")
        q = assess([])
        self.assertEqual(q["level"], "unreliable"); self.assertTrue(q["reasons"])

    def test_low_frame_rate_never_reads_as_good(self):
        self.assertEqual(assess(self.pts(20, 0.1, 0.7), fps=30)["level"], "degraded")


class ShirtReference(unittest.TestCase):
    def obs(self, near, far):
        return [dict(t=float(i) / 10, end=e, area=9000.0, h=c[0], s=c[1], v=c[2]) for i in range(300) for e, c in (("near", near), ("far", far))]

    def test_distinct_shirts_become_the_reference_and_alike_shirts_do_not(self):
        ref = auto_reference(self.obs((2, 220, 180), (110, 60, 40)), {"near": "Red", "far": "Black"})
        self.assertEqual((ref["near"]["name"], ref["far"]["name"]), ("Red", "Black"))
        self.assertIsNone(auto_reference(self.obs((2, 220, 180), (4, 215, 175)), {"near": "A", "far": "B"}))
        self.assertIsNone(auto_reference([], {"near": "A", "far": "B"}))


class Report(unittest.TestCase):
    def test_html_is_written_without_a_video(self):
        pts = [dict(id=1, start_t=8.2, end_t=10.4, serve_side="near", server="Red", winner="far", winner_name="Black",
                    near_player="Red", far_player="Black", n_crossings=2, ending="double_bounce", events=[]),
               dict(id=2, start_t=18.5, end_t=19.7, serve_side="near", server="Red", winner=None, winner_name=None,
                    near_player="Red", far_player="Black", n_crossings=1, ending="not_returned", events=[])]
        with tempfile.TemporaryDirectory() as d:
            path, n_clips = make_html(d, "Red v Black", pts, assess([dict(n_crossings=3, n_implied_crossings=0, coverage=0.8)] * 2), clips=False)
            page = pathlib.Path(path).read_text()
        self.assertEqual(n_clips, 0)
        for needle in ("Red v Black", "double bounce", "unsure", "0:08.2", 'class="verdict"', 'class="topbar"'):
            self.assertIn(needle, page)
        self.assertNotIn('class="stamp', page)                    # no confidence badge on the score line (removed 2026-09-26)
        self.assertIn("calc(56.25cqw + 3.5rem)", page)            # the player keeps a strip under the picture for the browser's controls


if __name__ == "__main__":
    unittest.main()


class AnnotatedClip(unittest.TestCase):
    def test_a_point_clip_is_rendered_as_h264_with_the_overlay(self):
        import shutil, subprocess
        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg not installed")
        from tt_scout.annotate import render_point_clip
        from tt_scout.table import Table
        with tempfile.TemporaryDirectory() as d:
            src = str(pathlib.Path(d) / "src.mp4"); fps, n = 60.0, 90
            vw = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080))
            base = scene()
            track = np.full((n, 6), np.nan); track[:, 0] = np.arange(n); track[:, 1] = np.arange(n) / fps
            for i in range(n):
                f = base.copy(); x, y = 500 + 10 * i, 560 - abs(30 - i % 60)
                cv2.circle(f, (x, y), 6, (255, 255, 255), -1); track[i, 2:4] = (x, y); vw.write(f)
            vw.release()
            table = Table(TRUE)
            events = [dict(t=0.5, kind="net", side="far", x_px=930.0, y_px=540.0, x_m=1.37, y_m=0.7),
                      dict(t=0.8, kind="bounce", side="far", x_px=1100.0, y_px=580.0, x_m=2.0, y_m=0.8)]
            point = dict(id=1, start_t=0.4, end_t=1.2, events=events, server="Red", serve_side="near", winner_name="Red",
                         ending="not_returned", near_player="Red", far_player="Black")
            dst = pathlib.Path(d) / "clip.mp4"
            ok = render_point_clip(src, fps, table, track, events, {30: [dict(end="near", cx=300.0, cy=600.0, area=9000.0)]}, point, dst, lead_s=0.3, tail_s=0.2)
            self.assertTrue(ok and dst.exists() and dst.stat().st_size > 2000)
            codec = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,height", "-of", "csv=p=0", str(dst)],
                                   capture_output=True, text=True).stdout.strip()
            self.assertEqual(codec, "h264,1080")                      # plays in Chrome and Safari; OpenCV's mp4v would not. Since
                                                                      # 2026-09-26 clips keep the footage's own resolution (annotate.HIRES)

            def table_line_pixels(path):                              # the calibration outline is a thin bright green line
                cap = cv2.VideoCapture(str(path)); n = 0
                while True:
                    ok, fr = cap.read()
                    if not ok:
                        break
                    n += int(((fr[..., 1] > 190) & (fr[..., 0] < 140) & (fr[..., 2] < 140)).sum())
                cap.release(); return n

            shown = pathlib.Path(d) / "clip_table.mp4"
            self.assertTrue(render_point_clip(src, fps, table, track, events, {30: [dict(end="near", cx=300.0, cy=600.0, area=9000.0)]}, point, shown,
                                              lead_s=0.3, tail_s=0.2, show_table=True))
            self.assertGreater(table_line_pixels(shown), table_line_pixels(dst) + 4000)          # off by default, drawn when asked


class ComicOverlay(unittest.TestCase):
    def test_running_score_resets_at_a_change_of_ends_and_skips_unsure_points(self):
        from tt_scout.annotate import running_scores
        pts = [dict(id=1, winner_name="A", ends_swapped=False), dict(id=2, winner_name=None, ends_swapped=False),
               dict(id=3, winner_name="B", ends_swapped=False), dict(id=4, winner_name="A", ends_swapped=False),
               dict(id=5, winner_name="B", ends_swapped=True)]
        sc = running_scores(pts)
        self.assertEqual(sc[2]["after"], {"A": 1})                           # the unsure point changes nothing
        self.assertEqual(sc[4]["after"], {"A": 2, "B": 1})
        self.assertEqual((sc[5]["before"], sc[5]["after"], sc[5]["games"]), ({}, {"B": 1}, {"A": 1}))   # new game, A took the first

    def test_speed_colour_runs_from_pale_to_hot_and_clamps(self):
        from tt_scout.annotate import speed_colour, SPEED_STOPS
        self.assertEqual(speed_colour(-3), SPEED_STOPS[0][1]); self.assertEqual(speed_colour(99), SPEED_STOPS[-1][1])
        self.assertEqual(speed_colour(8.0), (0, 235, 255))                   # yellow at 8 m/s (BGR)
        slow, fast = speed_colour(1.0), speed_colour(15.0)
        self.assertGreater(slow[0], fast[0]); self.assertGreater(fast[2], 200)   # blue channel falls, red stays high when fast

    def test_action_word_and_sprites(self):
        from tt_scout.annotate import action_word, burst_sprite, scoreboard_sprite
        self.assertEqual(action_word(dict(ending="long", n_crossings=3), 12.0), "OUT!")
        self.assertEqual(action_word(dict(ending="not_returned", n_crossings=9), None), "EPIC RALLY!")
        self.assertEqual(action_word(dict(ending="not_returned", n_crossings=2), 11.0), "SMASH!")
        self.assertEqual(action_word(dict(ending="not_returned", n_crossings=2), None), "POINT!")
        b = burst_sprite("POINT!", seed=3); self.assertEqual(b.shape[2], 4); self.assertGreater(int(b[..., 3].max()), 250)
        sb = scoreboard_sprite(("Red", "Black"), {"Red": 3, "Black": 5}, {"Red": 1}, 0, hot=1); self.assertEqual(sb.shape[2], 4)

    def test_a_crossing_and_a_bounce_from_different_shots_do_not_make_a_number(self):
        from tt_scout.annotate import last_shot_speed, SHOT_V_MAX
        from tt_scout.config import NET_X
        cross = [dict(t=10.0)]
        self.assertAlmostEqual(last_shot_speed([dict(t=10.25, x_m=NET_X + 3.0)], cross, None), 12.0)   # 3 m past the net in a quarter second
        self.assertIsNone(last_shot_speed([dict(t=10.02, x_m=NET_X + 1.2)], cross, None))              # 60 m/s: not one shot
        self.assertIsNone(last_shot_speed([dict(t=10.9, x_m=NET_X + 0.1)], cross, None))               # a dribble over the net
        self.assertEqual(last_shot_speed([dict(t=9.5, x_m=NET_X + 2.0)], cross, 8.0), 8.0)             # the bounce came first
        self.assertEqual(last_shot_speed([dict(t=12.0, x_m=NET_X + 2.0)], cross, 8.0), 8.0)            # and a second later is another shot
        self.assertEqual(last_shot_speed([], cross, 8.0), 8.0); self.assertEqual(last_shot_speed([dict(t=1, x_m=0)], [], 8.0), 8.0)
        self.assertLess(SHOT_V_MAX, 40.0)                                    # no ball has been hit that fast

    def test_the_speed_number_changes_once_per_readable_interval(self):
        from tt_scout.annotate import Held
        h = Held(hold=0.7)
        self.assertIsNone(h.update(0.0, None))
        self.assertEqual(h.update(0.1, 6.2), 6.2)                            # the first reading shows at once
        self.assertEqual(h.update(0.3, 11.0), 6.2)                           # the next shot waits ...
        self.assertEqual(h.update(0.5, 13.4), 6.2)                           # ... and a later one replaces it while it waits
        self.assertEqual(h.update(0.8, 13.4), 13.4); self.assertAlmostEqual(h.changed, 0.8)
        self.assertEqual(h.update(1.6, 13.2), 13.4); self.assertAlmostEqual(h.changed, 0.8)     # same printed digits: nothing moves
        shown, t, changes = None, 0.0, []
        while t < 6.0:                                                       # a fast rally: a new reading every 0.25 s
            v = h.update(t, 5.0 + (int(t / 0.25) % 5) * 3.0)
            if v != shown:
                changes.append(t); shown = v
            t += 1 / 60
        self.assertGreaterEqual(min(b - a for a, b in zip(changes, changes[1:])), 0.7 - 1e-6)

    def test_name_balloons_sit_beside_the_head_and_never_on_it(self):
        from tt_scout.annotate import tag_spot, _overlap, TAG_H
        panels = [(12, 12, 300, 90), (310, 6, 680, 130), (706, 10, 952, 199)]      # point caption, scoreboard, landing map
        w, hy = 64, 224

        def box(s):
            return (s[1] - w / 2, s[2] - TAG_H / 2, s[1] + w / 2, s[2] + TAG_H / 2)

        head = lambda hx: (hx - 16, hy - 16, hx + 16, hy + 16)
        near = tag_spot((180, hy), -1, w, panels, 960)
        self.assertEqual((near[0], near[3]), (0, "right")); self.assertLess(near[1], 180)          # left of the head, tail pointing at it
        far = tag_spot((800, hy), +1, w, panels, 960)                                              # under the map: beside the head, not below the map
        self.assertEqual(far[3], "left"); self.assertGreater(far[1], 800); self.assertTrue(hy <= far[2] <= hy + 16)   # at ear level
        self.assertEqual(_overlap(box(far), panels[2]), 0)
        edge = tag_spot((930, hy), +1, w, panels, 960)                                             # no room on the away side: the other side
        self.assertEqual(edge[3], "right"); self.assertLess(edge[1] + w / 2, 930)
        self.assertEqual(tag_spot((180, hy), -1, w, panels, 960, keep=1)[0], 1)                    # last frame's slot is kept while it is clear
        boxed = [(0, 0, 960, hy + 60)]                                                             # everything above the neck is a panel
        for hx in (180, 480, 800):
            s = tag_spot((hx, hy), -1, w, boxed, 960)
            self.assertEqual(_overlap(box(s), head(hx)), 0, hx)

    def test_the_gauge_and_caption_are_lettered_sprites(self):
        from tt_scout.annotate import gauge_sprite, hud_sprite, readout_sprite, tag_sprite, GAUGE_W, NEAR_RGB
        self.assertGreaterEqual(gauge_sprite().shape[1], GAUGE_W); self.assertEqual(hud_sprite(3, "Red", 2).shape[2], 4)
        self.assertEqual(readout_sprite("13", (255, 80, 60)).shape[2], 4)
        for tail in ("down", "left", "right"):
            s = tag_sprite("Black", NEAR_RGB, tail); self.assertGreater(int(s[..., 3].max()), 250)
