"""filming.camera_ok: a recording is accepted only when the phone stood beside the table, between its two ends."""
import unittest
from tt_scout.filming import camera_ok


def pos(x, off_deg=3.0):
    return dict(x=x, y=-3.5, z=1.0, off_deg=off_deg, dist_m=3.6, down_deg=16.0)


class CameraOk(unittest.TestCase):
    def test_beside_the_table(self):
        for x in (0.0, 1.2, 1.94, 2.74):                     # both ends count as beside it; 1.2 and 1.94 are real recordings
            self.assertEqual(camera_ok(pos(x)), (True, ""))

    def test_past_an_end(self):
        ok, why = camera_ok(pos(-2.1, 63.0))                    # IMG_3140, filmed from behind a player
        self.assertFalse(ok)
        self.assertIn("2.1 m past the end", why)
        ok, why = camera_ok(pos(3.0, 25.0))                     # just past the far end
        self.assertFalse(ok)
        self.assertIn("0.3 m past the end", why)


if __name__ == "__main__":
    unittest.main()
