"""The learned detector on phone recordings: labels from the classical analysis, the two detectors side by side, tracking with both."""
import json, pathlib, tempfile, unittest
import numpy as np, cv2

try:
    import torch
except ImportError:                                  # the ML part is optional
    torch = None

from tt_scout.detector import Candidate
from tt_scout.ml.fuse import fuse, as_candidates, MERGE_PX

CORNERS = np.array([[360, 640], [560, 520], [1340, 535], [1480, 660]], float)


class Fuse(unittest.TestCase):
    def test_a_peak_on_a_blob_is_one_ball_at_the_blob_with_the_better_score(self):
        blobs = [Candidate(100.0, 100.0, 50.0, 0.4), Candidate(300.0, 200.0, 60.0, 0.9)]
        out = fuse(blobs, [(100.0 + MERGE_PX - 1, 100.0, 0.95), (301.0, 200.0, 0.2)], area_ref=80.0)
        self.assertEqual([(c.x, c.y, c.area, c.score) for c in out], [(100.0, 100.0, 50.0, 0.95), (300.0, 200.0, 60.0, 0.9)])
        self.assertEqual(blobs[0].score, 0.4)                          # the classical detector's own list is left as it was

    def test_a_peak_no_blob_explains_is_added_as_a_ball_of_this_tables_size(self):
        out = fuse([Candidate(100.0, 100.0, 50.0, 0.4)], [(100.0 + MERGE_PX + 1, 100.0, 0.7)], area_ref=80.0)
        self.assertEqual(len(out), 2); self.assertEqual((out[1].x, out[1].area, out[1].score), (100.0 + MERGE_PX + 1, 80.0, 0.7))
        self.assertEqual([(c.x, c.area) for c in as_candidates([(5, 6, 0.5)], 80.0)], [(5.0, 80.0)])
        self.assertEqual(fuse([], [], 80.0), [])

    def test_a_ball_the_blob_detector_loses_for_ten_frames_stays_one_track_with_both(self):
        from tt_scout.config import Config
        from tt_scout.tracker import BallTracker
        cfg = Config()
        def run(with_peaks):
            trk = BallTracker(cfg, 60.0); trk.area_ref = 80.0; ids = []
            for f in range(60):
                x, y = 200.0 + 14 * f, 400.0 + 2 * f
                blobs = [] if 25 <= f < 35 else [Candidate(x, y, 80.0, 1.0)]        # white ball in front of something white
                c = trk.update(fuse(blobs, [(x + 1.0, y, 0.8)] if with_peaks else [], 80.0))
                ids.append(trk.track_id if c else -1)
            return ids
        alone, both = run(False), run(True)
        self.assertEqual(alone[30], -1)                                # lost without the network ...
        self.assertGreater(len({i for i in alone[5:] if i > 0}), 1)     # ... and picked up again as a new track
        self.assertEqual({i for i in both[5:]}, {both[5]})              # one track from start to end with it
        self.assertGreater(both[5], 0)


class Labels(unittest.TestCase):
    def fake_run(self):
        """One point: a flight seen in frames 20 to 44, its bounce at frame 45, then on to frame 60. Nothing after."""
        n, fps = 300, 60.0
        tr = np.full((n, 6), np.nan); tr[:, 0] = np.arange(n); tr[:, 1] = np.arange(n) / fps; tr[:, 4] = -1; tr[:, 5] = 0
        pos = {}
        for f in range(20, 46):
            pos[f] = (400.0 + 12 * (f - 20), 600.0 - 0.3 * (45 - f) ** 2)
        for f in range(45, 61):
            pos[f] = (400.0 + 12 * (f - 20), 600.0 - 6.0 * (f - 45) + 0.2 * (f - 45) ** 2)
        for f, (x, y) in pos.items():
            tr[f, 2:6] = (x, y, 1, 1)
        tr[33, 2:5] = (np.nan, np.nan, -1)                              # not seen in one frame of the flight
        fit = dict(ok=True, rms_px=1.0, point=1, shot=1, t_contact=20 / fps, t_bounce=45 / fps, anchor=dict(t=45 / fps), bounce=[2.0, 0.7], free=False)
        off = {30: 20.0}                                               # the fitted flight passes 20 px from the track in one frame
        class Cam:
            def project(self, P):
                return np.asarray(P)[:, :2]
        def state_at(r, ts):
            fs = [int(round(t * fps)) for t in ts]
            return np.array([[pos[f][0], pos[f][1] + off.get(f, 0.0), 0.0] for f in fs]), None
        pts = [dict(id=1, start_t=20 / fps, end_t=60 / fps, events=[])]
        return dict(track=tr, events=[], points=pts, fps=fps, cam=Cam(), fits=[fit], size=(1920, 1080)), state_at

    def test_labels_are_the_track_where_the_flight_agrees_and_negatives_keep_clear_of_the_point(self):
        from tt_scout import flight
        from tt_scout.ml import pseudo
        run, state_at = self.fake_run()
        keep = flight.shots, flight.net_strikes, flight.state_at
        flight.shots = lambda *a, **k: [dict(point=1, shot=1, frames=[f for f in range(21, 45) if f != 33])]
        flight.net_strikes = lambda *a, **k: []
        flight.state_at = state_at
        try:
            lab, src, negs = pseudo.make_labels(run)
        finally:
            flight.shots, flight.net_strikes, flight.state_at = keep
        tracked = sorted(f for f, s in src.items() if s == "tracked"); smooth = sorted(f for f, s in src.items() if s == "smooth")
        first = 20 + pseudo.GUARD_F                                   # nothing at the racket
        self.assertEqual(tracked, [f for f in range(first, 45) if f not in (30, 33)])
        self.assertEqual(smooth, list(range(47, 61 - pseudo.GUARD_F)))   # five neighbours in time are needed, and none at the next contact
        self.assertEqual(lab[40], tuple(run["track"][40, 2:4]))
        self.assertEqual(set(negs.values()), {"quiet"})
        self.assertGreaterEqual(min(negs), 60 + int(pseudo.DEAD_S * 60) + 1)
        self.assertLess(max(negs), 300 - pseudo.QUIET_F)

    def test_a_loose_fit_gives_no_labels(self):
        from tt_scout import flight
        from tt_scout.ml import pseudo
        run, state_at = self.fake_run(); run["fits"][0]["rms_px"] = pseudo.FIT_RMS + 0.5
        keep = flight.shots, flight.net_strikes, flight.state_at
        flight.shots = lambda *a, **k: [dict(point=1, shot=1, frames=list(range(21, 45)))]
        flight.net_strikes = lambda *a, **k: []; flight.state_at = state_at
        try:
            lab, src, negs = pseudo.make_labels(run)
        finally:
            flight.shots, flight.net_strikes, flight.state_at = keep
        self.assertEqual(lab, {})

    def test_extraction_writes_the_frames_the_labels_and_crops_to_look_at(self):
        from tt_scout.ml import pseudo
        from tt_scout.ml.data import to_size
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d); v = d / "v.mp4"
            vw = cv2.VideoWriter(str(v), cv2.VideoWriter_fourcc(*"mp4v"), 60.0, (1920, 1080))
            for f in range(40):
                img = np.full((1080, 1920, 3), (90, 60, 30), np.uint8); cv2.circle(img, (300 + 20 * f, 500), 6, (255, 255, 255), -1); vw.write(img)
            vw.release()
            lab = {10: (500.0, 500.0), 11: (520.0, 500.0)}; src = {10: "tracked", 11: "smooth"}; negs = {30: "quiet", 35: "dead"}
            w, m = pseudo.extract(v, lab, src, negs, d / "frames", d / "qa")
            self.assertEqual((w, m), (10, 2))                          # frames 8 to 11, 28 to 30, 33 to 35
            self.assertEqual(json.loads((d / "frames" / "labels.json").read_text())["10"], list(to_size(500.0, 500.0, src=(1920, 1080))))
            self.assertEqual(json.loads((d / "frames" / "negatives.json").read_text()), [30, 35])
            self.assertEqual(json.loads((d / "frames" / "negative_sources.json").read_text()), {"30": "quiet", "35": "dead"})
            crop = cv2.imread(str(d / "qa" / "tracked_10.jpg"))
            self.assertEqual(crop.shape[:2], (160, 160))
            self.assertGreater(int(crop[74:86, 74:86].max()), 200)      # the ball of frame 10 is at the crop's centre


class LearnedTracking(unittest.TestCase):
    def test_the_track_follows_the_peaks_when_nothing_moves_for_the_blob_detector(self):
        from tt_scout.config import Config
        from tt_scout.table import Table
        from tt_scout.run import track_video
        with tempfile.TemporaryDirectory() as d:
            v = pathlib.Path(d) / "still.mp4"
            vw = cv2.VideoWriter(str(v), cv2.VideoWriter_fourcc(*"mp4v"), 60.0, (1920, 1080))
            for f in range(30):
                vw.write(np.full((1080, 1920, 3), (90, 60, 30), np.uint8))
            vw.release()
            peaks = [[(300.0 + 15 * f, 400.0 + 3 * f, 0.9)] for f in range(30)]
            table = Table(CORNERS)
            rows, _, fps, n = track_video(str(v), table, Config(), learned=peaks, detector="learned")
            tr = np.array(rows, float)
            self.assertEqual(n, 30)
            self.assertTrue(np.allclose(tr[:, 2], [p[0][0] for p in peaks]) and np.allclose(tr[:, 3], [p[0][1] for p in peaks]))
            rows, _, _, _ = track_video(str(v), table, Config(), learned=peaks, detector="classical")
            self.assertTrue(np.isnan(np.array(rows, float)[:, 2]).all())     # the peaks are not used unless asked for


@unittest.skipIf(torch is None, "torch not installed")
class Spacing(unittest.TestCase):
    def test_auto_spacing_is_the_videos_frame_rate_over_the_models(self):
        from tt_scout.ml import infer
        from tt_scout.ml.model import BallNet
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d); v = d / "v.mp4"
            vw = cv2.VideoWriter(str(v), cv2.VideoWriter_fourcc(*"mp4v"), 120.0, (1920, 1080))
            for f in range(12):
                vw.write(np.full((1080, 1920, 3), (90, 60, 30), np.uint8))
            vw.release()
            m = BallNet(width=(4, 4, 4, 4, 4))
            first = {}
            for fps in (120.0, 60.0):
                torch.save(dict(state=m.state_dict(), config=m.config, threshold=0.0, fps=fps), d / "m.pt")
                _, cands, _ = infer.candidates(v, d / "m.pt", spacing="auto", log_every=0, device=torch.device("cpu"))
                first[fps] = next(i for i, c in enumerate(cands) if c)
            self.assertEqual(first, {120.0: 2, 60.0: 4})               # two frames of history one apart, or two apart


if __name__ == "__main__":
    unittest.main()
