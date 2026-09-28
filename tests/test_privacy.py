"""Blurring people who are not the two players (pose.others, annotate.blur_people): only the others are hidden, the players stay sharp
even where a stranger stands right behind one, and a frame where Vision misses the stranger is still covered."""
import unittest
import numpy as np
from tt_scout import pose
from tt_scout.annotate import blur_people


def person(cx, top, h=300.0):
    """19 joints spread over a standing person's box, all confident."""
    a = np.zeros((len(pose.JOINTS), 3))
    for i in range(len(pose.JOINTS)):
        a[i] = (cx + 40 * np.sin(i), top + h * i / (len(pose.JOINTS) - 1), 0.9)
    return a


class Others(unittest.TestCase):
    def test_only_the_stranger_is_hidden_and_held_through_a_missed_frame(self):
        near, far, stranger = person(400, 300), person(1500, 280), person(900, 250, h=150)
        raw = {10: [near, far, stranger], 11: [near, far]}             # frame 11: Vision missed the stranger
        assigned = {10: dict(near=near, far=far), 11: dict(near=near, far=far)}
        out = pose.others(raw, assigned, fps=60.0, hold_s=0.1)
        for f in (10, 11):
            self.assertEqual(len(out.get(f)["hide"]), 1)
            x0, y0, x1, y1 = out.get(f)["hide"][0]
            self.assertTrue(x0 < 900 < x1 and y0 < 250)                # round the stranger, head included
        self.assertEqual(len(out.get(10)["keep"]), 2)                   # the two players, kept sharp
        self.assertIsNone(out.get(40))                                  # far from any sighting: nothing to hide


class Blur(unittest.TestCase):
    def test_blur_changes_the_stranger_and_nothing_else(self):
        rng = np.random.default_rng(0)
        fr = rng.integers(0, 255, (1080, 1920, 3)).astype(np.uint8); before = fr.copy()
        blur_people(fr, dict(hide=[(800, 200, 1000, 500)], keep=[(950, 200, 1100, 600)]))
        d = np.abs(fr.astype(int) - before.astype(int)).mean(axis=2)
        self.assertGreater(d[330:370, 860:900].mean(), 40)             # inside the stranger: blurred
        self.assertLess(d[300:400, 990:1080].mean(), 1)                # inside the player's box: untouched
        self.assertLess(d[700:900, 100:400].mean(), 1)                 # elsewhere: untouched

    def test_a_stranger_right_beside_a_player_blurs_but_the_player_does_not(self):
        rng = np.random.default_rng(1)
        fr = rng.integers(0, 255, (1080, 1920, 3)).astype(np.uint8); before = fr.copy()
        sk = np.zeros((len(pose.JOINTS), 3))                            # a player standing at x = 700, torso from y 400 to 560
        for n, (x, y) in dict(nose=(700, 360), neck=(700, 400), lsho=(740, 405), rsho=(660, 405), root=(700, 560), lhip=(725, 560),
                              rhip=(675, 560)).items():
            sk[pose.J[n]] = (x, y, 0.9)
        blur_people(fr, dict(hide=[(560, 300, 960, 700)], keep=[sk]))  # the stranger's box takes in the player too
        d = np.abs(fr.astype(int) - before.astype(int)).mean(axis=2)
        self.assertLess(d[440:520, 690:710].mean(), 1)                 # the player's torso: sharp
        self.assertGreater(d[440:520, 860:900].mean(), 40)             # beside him, inside the stranger's box: blurred


if __name__ == "__main__":
    unittest.main()
