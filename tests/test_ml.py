"""Learned ball detector: frame/label alignment through extraction, heatmap targets, augmentation bookkeeping, loss and decoding."""
import json, pathlib, tempfile, unittest
import numpy as np, cv2

try:
    import torch
except ImportError:                                  # the ML part is optional
    torch = None


def dot_video(path, n=40, size=(1920, 1080), fps=120.0):
    """A video whose frame f has a white 12 px ball at x = 300 + 20 f, y = 500, and the frame number written on it."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for f in range(n):
        img = np.full((size[1], size[0], 3), (90, 60, 30), np.uint8)
        cv2.circle(img, (300 + 20 * f, 500), 6, (255, 255, 255), -1)
        vw.write(img)
    vw.release()


@unittest.skipIf(torch is None, "torch not installed")
class Data(unittest.TestCase):
    def test_heatmap_peak_is_the_positive_and_hidden_ball_is_empty(self):
        from tt_scout.ml.data import heatmap
        hm = heatmap(10.4, 7.6, 32, 16)
        self.assertEqual(float(hm.max()), 1.0); self.assertEqual(tuple(np.argwhere(hm == 1.0)[0]), (8, 10))
        self.assertEqual(int((hm == 1.0).sum()), 1); self.assertLess(float(hm[8, 14]), 0.1)            # 3.6 px from the centre at sigma 1.5
        self.assertEqual(float(heatmap(None, None, 32, 16).sum()), 0.0)
        self.assertEqual(float(heatmap(-50, -50, 32, 16).sum()), 0.0)

    def test_scaling_maps_pixel_centres_and_round_trips(self):
        from tt_scout.ml.data import to_size, to_src
        self.assertEqual(to_size(1, 1), (0.0, (1.5 * 352 / 1080) - 0.5))            # the centre of source pixel 1 is the centre of pixel 0
        for x, y in ((0.0, 0.0), (959.5, 540.2), (1919.0, 1079.0)):
            a, b = to_src(*to_size(x, y)); self.assertAlmostEqual(a, x, 6); self.assertAlmostEqual(b, y, 6)

    def test_extraction_keeps_frame_numbers_aligned_with_labels(self):
        from tt_scout.ml.data import extract, SIZE
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d); dot_video(d / "v.mp4"); mk = d / "markup"; mk.mkdir()
            labels = {str(f): {"x": 300 + 20 * f, "y": 500} for f in (10, 11, 12, 30)}
            labels["31"] = {"x": -1, "y": -1}
            (mk / "ball_markup.json").write_text(json.dumps(labels))
            written, n = extract(d / "v.mp4", mk, d / "frames")
            self.assertEqual(n, 5); self.assertEqual(written, len({f - k for f in (10, 11, 12, 30, 31) for k in range(3)}))
            lab = json.loads((d / "frames" / "labels.json").read_text())
            self.assertIsNone(lab["31"])
            for f in (10, 30):                        # the brightest pixel of the saved frame is where its label says
                img = cv2.imread(str(d / "frames" / f"{f}.jpg")).max(axis=2).astype(float)
                ys, xs = np.nonzero(img > 200)
                self.assertLess(abs(xs.mean() - lab[str(f)][0]), 1.0); self.assertLess(abs(ys.mean() - lab[str(f)][1]), 1.0)
                self.assertEqual(img.shape, (SIZE[1], SIZE[0]))

    def test_dataset_crops_flips_and_stacks_three_frames(self):
        from tt_scout.ml.data import extract, BallFrames
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d); dot_video(d / "v.mp4"); mk = d / "markup"; mk.mkdir()
            (mk / "ball_markup.json").write_text(json.dumps({str(f): {"x": 300 + 20 * f, "y": 500} for f in range(5, 30)}))
            extract(d / "v.mp4", mk, d / "root" / "g")
            full = BallFrames(["g"], root=d / "root")
            x, t, xy = full[0]
            self.assertEqual(tuple(x.shape), (9, 352, 640)); self.assertEqual(tuple(t.shape), (1, 352, 640))
            py, px = divmod(int(t.view(-1).argmax()), 640)
            self.assertLess(abs(px - xy[0].item()), 1.0); self.assertLess(abs(py - xy[1].item()), 1.0)
            last = x[6:9].mean(0).numpy()                     # frame t is the last three channels: its brightest spot is the ball
            ly, lx = np.unravel_index(last.argmax(), last.shape)
            self.assertLess(abs(lx - xy[0].item()), 2.5)
            aug = BallFrames(["g"], root=d / "root", crop=(320, 176), augment=True, p_ball=1.0, seed=3)
            for i in range(len(aug)):
                x, t, xy = aug[i]
                self.assertEqual(tuple(x.shape), (9, 176, 320))
                py, px = divmod(int(t.view(-1).argmax()), 320)
                self.assertEqual(float(t.max()), 1.0)
                self.assertLess(abs(px - xy[0].item()), 1.0); self.assertLess(abs(py - xy[1].item()), 1.0)
            self.assertEqual(len(aug), len(full))


@unittest.skipIf(torch is None, "torch not installed")
class ModelAndLoss(unittest.TestCase):
    def test_shapes_loss_and_peaks(self):
        from tt_scout.ml.model import BallNet, focal_loss, peaks
        from tt_scout.ml.data import heatmap
        m = BallNet().eval()
        with torch.no_grad():
            self.assertEqual(tuple(m(torch.zeros(2, 9, 176, 320)).shape), (2, 1, 176, 320))
        t = torch.from_numpy(heatmap(40.3, 20.7, 64, 32))[None, None]
        good = torch.full((1, 1, 32, 64), -8.0); good[0, 0] = torch.logit(t[0, 0].clamp(1e-4, 1 - 1e-4))
        bad = torch.full((1, 1, 32, 64), -8.0); bad[0, 0, 5, 5] = 6.0
        self.assertLess(float(focal_loss(good, t)), float(focal_loss(bad, t)))
        self.assertTrue(torch.isfinite(focal_loss(bad, torch.zeros_like(t))))
        (x, y, s), = peaks(good, k=3, thresh=0.5)[0]
        self.assertLess(abs(x - 40.3), 0.08); self.assertLess(abs(y - 20.7), 0.08)       # parabola on log p; self.assertGreater(s, 0.95)
        self.assertEqual(peaks(torch.full((1, 1, 32, 64), -8.0), thresh=0.5), [[]])


if __name__ == "__main__":
    unittest.main()
