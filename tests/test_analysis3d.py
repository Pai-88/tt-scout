"""The after-match 3D analysis: the hidden-contact bridge (flight._bridge), placing a 3D body in the table's frame (body3d.place), the
3D posture measures, and the plates and report section built from them (analysis3d, report_html). """
import json, math, pathlib, tempfile, unittest
import numpy as np
from tt_scout import flight, body3d
from tt_scout.camera import Camera
from tt_scout.config import TABLE_LENGTH as L, TABLE_WIDTH as W

CAM = dict(K=np.array([[974.5, 0, 960], [0, 974.5, 540], [0, 0, 1.0]]))


def camera():
    """A camera beside the table like a phone on a tripod: 3 m out, 1 m above the playing surface, looking across it."""
    C = np.array([1.2, -3.0, 1.0]); tgt = np.array([1.37, 0.76, -0.2])
    f = tgt - C; f /= np.linalg.norm(f); r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); u = np.cross(r, f)
    R = np.stack([r, -u, f]); return Camera(CAM["K"], R, -R @ C)


def standing_body(root=(-0.6, 0.8), knee_bend=0.0):
    """17 joints (body3d.NAMES order) of a 1.75 m person facing +x, ankles 8 cm above the floor, table frame."""
    x, y = root; fl = body3d.FLOOR
    J = {"left_ankle": [x, y + 0.14, fl + 0.08], "right_ankle": [x, y - 0.14, fl + 0.08], "left_knee": [x + knee_bend, y + 0.13, fl + 0.5],
         "right_knee": [x + knee_bend, y - 0.13, fl + 0.5], "left_hip": [x, y + 0.12, fl + 0.92], "right_hip": [x, y - 0.12, fl + 0.92],
         "root": [x, y, fl + 0.92], "spine": [x, y, fl + 1.18], "center_shoulder": [x, y, fl + 1.45], "left_shoulder": [x, y + 0.19, fl + 1.43],
         "right_shoulder": [x, y - 0.19, fl + 1.43], "left_elbow": [x, y + 0.24, fl + 1.15], "right_elbow": [x, y - 0.24, fl + 1.15],
         "left_wrist": [x + 0.2, y + 0.25, fl + 1.0], "right_wrist": [x + 0.2, y - 0.25, fl + 1.0], "center_head": [x, y, fl + 1.62],
         "top_head": [x, y, fl + 1.74]}
    return np.array([J[n] for n in body3d.NAMES], float)


class Bridge(unittest.TestCase):
    def test_a_hidden_contact_is_found_where_the_paths_meet(self):
        fps = 60.0; t = np.arange(0, 1.0, 1 / fps); u = np.full(len(t), np.nan); v = np.full(len(t), np.nan)
        tc = 0.5                                                          # the ball comes in leftwards at 12 px a frame, leaves at 22
        inc = t < tc; u[inc] = 540 + 12 * (tc - t[inc]) * fps; u[~inc] = 540 + 22 * (t[~inc] - tc) * fps; v[:] = 650
        hid = (t > tc - 3 / fps) & (t < tc + 6 / fps)
        u[hid] = 1500.0; v[hid] = 720.0                                   # the tracker on something else meanwhile
        track = np.c_[np.arange(len(t)), t, u, v]
        t_out = t[np.where((t >= tc + 6 / fps))[0][0]]
        got = flight._bridge(track, t_out, 0.2, 1.0, fps)
        self.assertIsNotNone(got)
        self.assertAlmostEqual(got, tc, delta=0.5 / fps)

    def test_no_bridge_without_an_incoming_path(self):
        fps = 60.0; t = np.arange(0, 1.0, 1 / fps); u = np.where(t >= 0.5, 540 + 22 * (t - 0.5) * fps, np.nan); v = np.full(len(t), 650.0)
        self.assertIsNone(flight._bridge(np.c_[np.arange(len(t)), t, u, v], 0.5, 0.2, 1.0, fps))


class Bodies(unittest.TestCase):
    def test_a_body_is_placed_back_where_it_stood(self):
        cam = camera(); Jw = standing_body()
        uv = cam.project(Jw)
        a = math.radians(35); Rb = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        Jb = (Jw - Jw[body3d.I["root"]]) @ Rb.T                           # the same body in some body frame, as Vision would give it
        rec = dict(joints={n: Jb[i].tolist() for i, n in enumerate(body3d.NAMES)}, img={n: uv[i].tolist() for i, n in enumerate(body3d.NAMES)}, height=1.8)
        Xw, chk = body3d.place(rec, cam)
        self.assertLess(chk["rms_px"], 0.5)
        self.assertAlmostEqual(chk["scale"], 1.0, delta=0.02)
        self.assertLess(float(np.abs(Xw - Jw).max()), 0.03)

    def test_straight_upright_body_measures_straight_and_upright(self):
        J = standing_body()
        st = dict(frames=[J, J], contact_index=1, ball=None)
        m = body3d.measures(st)
        self.assertGreater(m["knee"], 175); self.assertLess(abs(m["lean"]), 1.0); self.assertLess(m["turn"], 1.0)
        self.assertAlmostEqual(m["hip"], 0.92, delta=0.01)

    def test_forehand_and_backhand_by_the_side_of_the_body(self):
        J = standing_body()
        self.assertEqual(body3d.side_of(J, np.array([-0.3, 0.8 - 0.5, 0.2]), "right")[1], "forehand")   # on his right (-y, facing +x)
        self.assertEqual(body3d.side_of(J, np.array([-0.3, 0.8 + 0.4, 0.2]), "right")[1], "backhand")


def turned(J, deg, about):
    """The body turned deg about a vertical axis through the point about."""
    a = math.radians(deg); R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
    return (J - about) @ R.T + about


class Cleaning(unittest.TestCase):
    def stroke(self, flip_at=None):
        J = standing_body(); c = J[body3d.I["root"]]
        frames = [turned(J, 3.0 * k, c) for k in range(12)]              # a real stroke turns about 35 deg in all
        if flip_at is not None:
            frames[flip_at] = turned(frames[flip_at], 150.0, c)         # Vision puts one frame the wrong way round
        return dict(frames=frames, dts=[0.033 * (k - 9) for k in range(12)], contact_index=9)

    def test_a_frame_put_the_wrong_way_round_is_left_out(self):
        st = self.stroke(flip_at=5)
        self.assertEqual(body3d.flipped(st["frames"]), [5])
        out = body3d.without_flips(st)
        self.assertEqual(len(out["frames"]), 11); self.assertEqual(out["contact_index"], 8)
        self.assertAlmostEqual(out["dts"][out["contact_index"]], st["dts"][9])
        self.assertEqual(body3d.flipped(self.stroke()["frames"]), [])       # a normal turn is not a flip
        self.assertIsNone(body3d.without_flips(self.stroke(flip_at=9)))   # the contact itself flipped: the stroke is not used

    def test_a_floating_foot_is_put_back_on_the_floor(self):
        J = standing_body(); I = body3d.I
        k, a = J[I["left_knee"]], J[I["left_ankle"]]; b = math.radians(70)             # the hidden leg guessed with the knee bent,
        J[I["left_ankle"]] = k + np.array([[math.cos(b), 0, -math.sin(b)], [0, 1, 0], [math.sin(b), 0, math.cos(b)]]) @ (a - k)   # foot ~28 cm up
        G = body3d.planted([J])[0]
        self.assertLessEqual(G[I["left_ankle"], 2], body3d.ANKLE_Z + body3d.PLANT_M + 1e-9)
        for a, b in (("left_hip", "left_knee"), ("left_knee", "left_ankle")):   # thigh and shin keep their lengths
            self.assertAlmostEqual(np.linalg.norm(G[I[a]] - G[I[b]]), np.linalg.norm(J[I[a]] - J[I[b]]), places=6)
        self.assertTrue(np.allclose(G[I["right_ankle"]], J[I["right_ankle"]]))      # the foot already down is not moved
        self.assertTrue(np.allclose(G[I["left_hip"]], J[I["left_hip"]]))


def fake_shots(name, end, n=12, seed=0):
    rng = np.random.default_rng(seed); out = []
    for k in range(n):
        x0 = -0.3 - 0.4 * rng.random(); y0 = 0.3 + 0.9 * rng.random()
        path = [[x0 + (2.2 - x0) * s, y0 + 0.2 * s, 0.25 + 0.3 * math.sin(math.pi * s) - 0.23 * s] for s in np.linspace(0, 1, 24)]
        c = np.array(path[0]); p = np.array(path)
        if end == "far":
            c = np.array([L - c[0], W - c[1], c[2]]); p = np.c_[L - p[:, 0], W - p[:, 1], p[:, 2]]
        feet = [[c[0] - 0.5 if end == "near" else c[0] + 0.5, c[1] + 0.2], [c[0] - 0.6 if end == "near" else c[0] + 0.6, c[1] - 0.3]]
        out.append(dict(point=k + 1, shot=2 + k % 3, name=name, end=end, serve=False, t=float(k), timing=0.02 * (k % 4), speed=6.0,
                        contact=c.tolist(), path=p.tolist(), feet=feet, contact_src="seen", flight_t=0.4,
                        incoming=dict(bounce=[0.6 if end == "near" else L - 0.6, 0.7], dt=0.3)))
    return out


class Plates(unittest.TestCase):
    def test_plates_and_report_section(self):
        from tt_scout import analysis3d
        from tt_scout.report_html import _after3d
        shots = fake_shots("A", "near") + fake_shots("B", "far", seed=1)
        with tempfile.TemporaryDirectory() as d:
            a3, players, pro = analysis3d.build(shots, ["A", "B"], d, themes=("light",), size=(640, 360))
            self.assertEqual([p["name"] for p in a3["players"]], ["A", "B"])
            for p in a3["players"]:
                for view in ("stand", "strike", "flight"):
                    self.assertTrue((pathlib.Path(d) / p["plates"][view]["light"]).exists())
            html = _after3d(a3, ["A", "B"])
            self.assertIn('id="after"', html); self.assertIn("a3d-data", html); self.assertIn("After the match, in 3D", html)
            json.loads(html.split('id="a3d-data">')[1].split("</script>")[0].replace("<\\/", "</"))   # the viewer's data parses


if __name__ == "__main__":
    unittest.main()
