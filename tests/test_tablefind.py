"""The colour-blind table finder (tt_scout/tablefind.py) on a table drawn through a known camera: grey top, white edge and centre
lines, a net, a dark apron, a wooden floor. It must find all four corners within 5 px (the failures it guards against are 70 px:
half the table, or the apron glued on; on our real recordings every corner is within 1.3 px), and nothing in a picture without a table."""
import unittest
import numpy as np, cv2
from tt_scout.config import TABLE_LENGTH as L, TABLE_WIDTH as TW
from tt_scout.tablefind import surface_table

K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])


def camera(C, target):
    z = target - C; z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(z, x), z])
    return R, -R @ C


def proj(R, t, P):
    p = (K @ (R @ np.asarray(P, float).T + t[:, None])).T
    return p[:, :2] / p[:, 2:]


def scene(C=(1.3, -3.0, 1.0), seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((1080, 1920, 3), np.uint8)
    img[:] = (40, 95, 165)                                                  # a wooden floor (BGR) ...
    img[:400] = (200, 170, 150)                                             # ... under a pale wall
    stripes = (np.arange(1920)[None, :] // 40 % 2) * 12
    img[400:] = np.clip(img[400:].astype(int) + stripes[:, :, None], 0, 255)
    R, t = camera(np.array(C, float), np.array([L / 2, TW / 2, 0.0]))
    poly = lambda P: np.round(proj(R, t, P) * 16).astype(np.int32).reshape(-1, 1, 2)
    fill = lambda P, col: cv2.fillPoly(img, [poly(P)], col, cv2.LINE_AA, 4)
    fill([[0, 0, 0], [L, 0, 0], [L, 0, -0.06], [0, 0, -0.06]], (35, 35, 35))              # the apron under the near edge
    fill([[0, 0, 0], [0, TW, 0], [L, TW, 0], [L, 0, 0]], (245, 245, 245))                  # white edge line ...
    e = 0.02
    fill([[e, e, 0], [e, TW - e, 0], [L - e, TW - e, 0], [L - e, e, 0]], (150, 152, 148))  # ... round a grey top
    c = 0.0015
    fill([[e, TW / 2 - c, 0], [e, TW / 2 + c, 0], [L - e, TW / 2 + c, 0], [L - e, TW / 2 - c, 0]], (245, 245, 245))   # centre line
    n = 0.004
    fill([[L / 2 - n, -0.15, 0], [L / 2 - n, TW + 0.15, 0], [L / 2 - n, TW + 0.15, 0.1525], [L / 2 - n, -0.15, 0.1525]], (45, 45, 45))
    fill([[L / 2 - n, -0.15, 0.13], [L / 2 - n, TW + 0.15, 0.13], [L / 2 - n, TW + 0.15, 0.1525], [L / 2 - n, -0.15, 0.1525]], (240, 240, 240))
    img = np.clip(img.astype(int) + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    return img, proj(R, t, [[0, 0, 0], [0, TW, 0], [L, TW, 0], [L, 0, 0]])


class SurfaceTable(unittest.TestCase):
    def test_finds_the_corners(self):
        for C in ((1.3, -3.0, 1.0), (1.8, -3.6, 1.4)):
            img, truth = scene(C)
            found, info = surface_table(img)
            self.assertIsNotNone(found, info)
            err = max(min(np.hypot(*(found - p).T)) for p in truth)       # each true corner to the nearest found one
            self.assertLess(err, 5.0, (C, found.round(1).tolist(), truth.round(1).tolist()))

    def test_no_table_no_corners(self):
        img, _ = scene()
        img[380:1080] = (40, 95, 165)                                        # floor only
        found, why = surface_table(img)
        self.assertIsNone(found)


if __name__ == "__main__":
    unittest.main()
