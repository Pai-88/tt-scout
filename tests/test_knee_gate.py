"""The knee bend at contact as ONE quantity (pose.py, top): the camera-near leg by the longer picture span, the facing gate (a leg
seen in profile passes, an edge-on one is refused, from the 3D pelvis or from the 2D hip width), a hit's side from its shot's hitter
end, the 3D posture on the shots before shots.json is written, and the record keeping the gated median with its count. """
import json, math, pathlib, tempfile, unittest
import numpy as np
from tt_scout import pose as P, body3d, technique
from tt_scout.body3d import I
from tests.test_analysis3d import camera, standing_body, turned


def skeleton(knee_l=150.0, knee_r=120.0, span_l=200.0, span_r=170.0, hip_w=30.0, torso=130.0, cx=900.0, both=True):
    """A 2D skeleton (pose.JOINTS order, x y confidence): hips hip_w apart, neck torso above them, each leg a hip-knee-ankle chain
    of the given knee angle whose hip-to-ankle span in the picture is span_l / span_r."""
    a = np.zeros((len(P.JOINTS), 3))
    def put(n, x, y):
        a[P.J[n]] = (x, y, 0.9)
    hy = 500.0
    put("neck", cx, hy - torso); put("root", cx, hy); put("lhip", cx + hip_w / 2, hy); put("rhip", cx - hip_w / 2, hy)
    for s, ang, span, dx in (("l", knee_l, span_l, +hip_w / 2), ("r", knee_r, span_r, -hip_w / 2)):
        half = math.radians(ang / 2)                                   # thigh and shin of equal length L: span = 2 L sin(ang/2)
        Lb = span / (2 * math.sin(half))
        put(s + "kne", cx + dx + Lb * math.cos(half), hy + Lb * math.sin(half))
        if both or s == "l":
            put(s + "ank", cx + dx, hy + span)
    return a


class LegRule(unittest.TestCase):
    def test_the_leg_with_the_longer_picture_span_is_taken_whatever_its_bend(self):
        leg, knee = P.near_leg(skeleton(knee_l=150, knee_r=120, span_l=200, span_r=170))
        self.assertEqual(leg, "l"); self.assertAlmostEqual(knee, 150.0, delta=1.0)
        leg, knee = P.near_leg(skeleton(knee_l=150, knee_r=120, span_l=160, span_r=190))   # the more bent leg is the nearer one here
        self.assertEqual(leg, "r"); self.assertAlmostEqual(knee, 120.0, delta=1.0)

    def test_one_leg_found_is_no_knee_at_all(self):
        self.assertEqual(P.near_leg(skeleton(both=False)), (None, None))
        m = P.measures(skeleton(both=False), "near")
        self.assertIsNone(m["knee"]); self.assertIsNone(m["leg"])

    def test_the_3d_rule_picks_the_same_leg_as_the_2d_rule(self):
        cam = camera(); J = standing_body(root=(-0.6, 0.8))
        J[I["right_knee"]][0] += 0.25; J[I["right_ankle"]][0] += 0.45     # the right leg stepped forward: seen bigger, and nearer this camera (y = -3)
        px = cam.project(J)
        a = np.zeros((len(P.JOINTS), 3))                                   # the same body as Vision's 2D skeleton (pose.JOINTS order)
        for n2, n3 in (("neck", "center_shoulder"), ("root", "root"), ("lhip", "left_hip"), ("rhip", "right_hip"), ("lkne", "left_knee"),
                       ("rkne", "right_knee"), ("lank", "left_ankle"), ("rank", "right_ankle")):
            a[P.J[n2]] = (px[I[n3]][0], px[I[n3]][1], 0.9)
        self.assertEqual(P.near_leg(a)[0], {"left": "l", "right": "r"}[body3d.near_leg(J, cam)])   # the 2D rule and its 3D twin agree
        self.assertEqual(body3d.nearer_leg(J, cam), "right")               # ... and here it is the leg whose knee is nearer the camera
        self.assertGreaterEqual(P.measures(a, "near")["leg_ratio"], P.LEG_RATIO_MIN)
        m = body3d.measures(dict(frames=[J], contact_index=0, end="near", ball=None), cam)
        self.assertIsNotNone(m["knee"])
        self.assertIsNone(body3d.measures(dict(frames=[J], contact_index=0, end="near", ball=None))["knee"])   # no camera: no knee


def stroke(J, end="near", side="forehand", name="A", point=1, shot=2):
    return dict(point=point, shot=shot, name=name, end=end, side=side, hand="right", contact_index=0, frames=[body3d._orient(J, end)],
                dts=[0.0], ball=None)


class ProfileGate(unittest.TestCase):
    def test_a_leg_seen_in_profile_passes_and_an_edge_on_one_is_refused(self):
        cam = camera(); J = standing_body(root=(1.2, 0.8))                 # facing +x, squarely in front of the camera at y = -3: seen side-on
        self.assertGreater(body3d.pelvis_profile(J, cam), 70.0)
        self.assertLess(body3d.pelvis_profile(standing_body(root=(-0.6, 0.8)), cam), 70.0)   # at the table's end the same camera sees him obliquely
        Jf = turned(J, 90.0, J[I["root"]])                                # turned to face the camera: legs edge-on
        self.assertLess(body3d.pelvis_profile(Jf, cam), 10.0)
        shots = [dict(point=1, shot=2, name="A", end="near", serve=False, t=1.0, knee=150.0, leg="r", profile2d=None),
                 dict(point=1, shot=4, name="A", end="near", serve=False, t=2.0, knee=150.0, leg="r", profile2d=None),
                 dict(point=1, shot=3, name="A", end="near", serve=False, t=1.5, knee=150.0, leg="r", profile2d=None, side=None),
                 dict(point=1, shot=1, name="A", end="near", serve=True, t=0.5, knee=150.0, leg="r", profile2d=None)]
        strokes = [stroke(J, point=1, shot=2), stroke(Jf, point=1, shot=4), stroke(J, point=1, shot=3, side="backhand"), stroke(J, point=1, shot=1)]
        technique.gate_knees(shots, strokes, cam)
        by = {s["shot"]: s for s in shots}
        self.assertEqual(by[2]["knee"], 150.0); self.assertEqual(by[2]["profile_src"], "3d"); self.assertGreaterEqual(by[2]["profile"], P.PROFILE_MIN)
        self.assertTrue(by[2]["leg_ok"])                                                        # the right leg is the nearer one in the body too
        self.assertIsNone(by[4]["knee"]); self.assertLess(by[4]["profile"], P.PROFILE_MIN)     # edge-on: refused
        self.assertIsNone(by[3]["knee"])                                                        # a backhand
        self.assertIsNone(by[1]["knee"])                                                        # the serve
        self.assertEqual(by[4]["knee_raw"], 150.0)                                              # the angle itself is kept for checking

    def test_the_2d_leg_must_be_the_leg_nearer_the_camera_in_the_body(self):
        cam = camera(); J = standing_body(root=(1.2, 0.8))                 # side-on: the right leg (y - 0.13) is nearer the camera at y = -3
        shots = [dict(point=1, shot=2, name="A", end="near", serve=False, t=1.0, knee=150.0, leg="l", profile2d=None),
                 dict(point=1, shot=4, name="A", end="near", serve=False, t=2.0, knee=150.0, leg="r", profile2d=None)]
        technique.gate_knees(shots, [stroke(J, point=1, shot=2), stroke(J, point=1, shot=4)], cam)
        self.assertFalse(shots[0]["leg_ok"]); self.assertIsNone(shots[0]["knee"]); self.assertEqual(shots[0]["knee_raw"], 150.0)
        self.assertTrue(shots[1]["leg_ok"]); self.assertEqual(shots[1]["knee"], 150.0)

    def test_in_a_recording_with_3d_bodies_a_shot_without_one_is_refused_not_judged_by_the_picture(self):
        cam = camera(); J = standing_body(root=(1.2, 0.8))
        side_on = P.measures(skeleton(hip_w=15.0, torso=130.0), "near")    # the picture alone would pass this one
        self.assertGreater(side_on["profile2d"], P.PROFILE_MIN)
        contact, feet = [-0.3, 0.55, 0.2], [[-0.6, 0.9], [-0.7, 0.7]]
        shots = [dict(point=1, shot=2, name="A", end="near", serve=False, t=1.0, contact=contact, feet=feet, **side_on),
                 dict(point=1, shot=4, name="A", end="near", serve=False, t=2.0, knee=150.0, leg="r", leg_ratio=1.3, profile2d=None)]
        technique.gate_knees(shots, [stroke(J, point=1, shot=4)], cam)     # the 3D pass ran: shot 2 has no body, so it is refused
        self.assertIsNone(shots[0]["knee"]); self.assertIsNone(shots[0]["profile"]); self.assertIsNone(shots[0]["profile_src"])
        self.assertEqual(shots[0]["side"], "forehand"); self.assertIsNotNone(shots[0]["knee_raw"])
        self.assertEqual(shots[1]["knee"], 150.0); self.assertEqual(shots[1]["profile_src"], "3d")
        s0 = dict(point=1, shot=2, name="A", end="near", serve=False, t=1.0, contact=contact, feet=feet, **side_on)
        technique.gate_knees([s0], [], None)                               # no 3D pass at all: the picture judges it
        self.assertEqual(s0["profile_src"], "2d"); self.assertIsNotNone(s0["knee"])
        s1 = dict(point=1, shot=4, name="A", end="near", serve=False, t=2.0, knee=150.0, leg="r", leg_ratio=1.3, profile2d=80.0)
        technique.gate_knees([s1], [stroke(J, point=1, shot=4)], None)     # a body but no camera: the picture judges profile and leg
        self.assertEqual(s1["profile_src"], "2d"); self.assertEqual(s1["knee"], 150.0)

    def test_without_a_3d_body_the_picture_s_hip_width_gates(self):
        side_on = skeleton(hip_w=15.0, torso=130.0)                       # hips 0.12 torso lengths wide: seen side-on
        facing = skeleton(hip_w=58.0, torso=130.0)                        # 0.45: squarely facing the camera
        self.assertGreater(P.profile_2d(side_on), P.PROFILE_MIN); self.assertLess(P.profile_2d(facing), P.PROFILE_MIN)
        contact, feet = [-0.3, 0.55, 0.2], [[-0.6, 0.9], [-0.7, 0.7]]     # the ball met at -y of the feet: a right hander's forehand at the near end
        shots = [dict(point=1, shot=2, name="A", end="near", serve=False, t=1.0, contact=contact, feet=feet, **P.measures(side_on, "near")),
                 dict(point=1, shot=4, name="A", end="near", serve=False, t=2.0, contact=contact, feet=feet, **P.measures(facing, "near"))]
        technique.gate_knees(shots, [], None)
        self.assertEqual(shots[0]["side"], "forehand"); self.assertEqual(shots[0]["profile_src"], "2d"); self.assertIsNotNone(shots[0]["knee"])
        self.assertIsNone(shots[1]["knee"]); self.assertEqual(shots[1]["profile_src"], "2d")
        self.assertEqual(technique.side_2d(dict(end="far", contact=contact, feet=feet)), "backhand")   # at the far end his right is +y
        coin = P.measures(skeleton(hip_w=15.0, torso=130.0, span_l=180.0, span_r=175.0), "near")      # spans nearly equal: no near leg
        self.assertLess(coin["leg_ratio"], P.LEG_RATIO_MIN)
        s2 = dict(point=1, shot=6, name="A", end="near", serve=False, t=3.0, contact=contact, feet=feet, **coin)
        technique.gate_knees([s2], [], None)
        self.assertFalse(s2["leg_ok"]); self.assertIsNone(s2["knee"]); self.assertIsNotNone(s2["knee_raw"])

    def test_fewer_than_three_gated_contacts_is_not_measurable(self):
        self.assertEqual(P.knee_median([150.0, 140.0]), (None, 2))
        self.assertEqual(P.knee_median([150.0, 140.0, 130.0]), (140.0, 3))
        self.assertTrue(P.knee_text((None, 2)).startswith(P.NOT_MEASURABLE))
        self.assertIn(P.KNEE_LABEL, P.knee_text((140.0, 3)))


class HitSide(unittest.TestCase):
    def test_a_hit_takes_the_end_of_its_shot_s_hitter(self):
        fps = 60.0
        near, far = skeleton(cx=400.0), skeleton(cx=1500.0, knee_l=110.0, knee_r=100.0, hip_w=15.0)   # the far one seen side-on
        assigned = {f: {"near": near, "far": far} for f in range(0, 400)}
        events = [dict(kind="hit", t=1.00, side="far"),                    # the ball's table x said the far end ...
                  dict(kind="hit", t=2.50, side="near"),                   # ... here no shot is near: the point's serve order places it
                  dict(kind="hit", t=5.00, side="near")]                   # ... and this one lies in no point: not measured at all
        shots = [dict(point=1, shot=2, name="A", end="near", serve=False, t=1.02, knee=147.5, leg="l")]   # ... but the shot was hit at the near end
        # B serves from the far end: stroke 1 (B, far) crosses at 0.5, stroke 2 (A, near, hit at 1.0) at 1.3, stroke 3 (B, far, hit at 2.5) at 2.8
        points = [dict(id=1, start_t=0.0, end_t=3.0, server="B", near_player="A", far_player="B",
                       events=[dict(kind="net", t=0.5), dict(kind="net", t=1.3, implied=True), dict(kind="net", t=2.8)])]
        hits = P.at_hits(events, assigned, fps, shots=shots, points=points)
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0]["side"], "near"); self.assertEqual(hits[0]["knee"], 147.5); self.assertEqual(hits[0]["leg"], "l")
        self.assertEqual((hits[0]["point"], hits[0]["shot"]), (1, 2))
        self.assertEqual(hits[1]["side"], "far"); self.assertIsNone(hits[1]["knee"]); self.assertIsNone(hits[1]["point"])
        self.assertEqual(P.stroke_end(points[0], 2.5), "far"); self.assertEqual(P.stroke_end(points[0], 1.0), "near")
        self.assertEqual(P.stroke_end(points[0], 2.5, [dict(point=1, shot=3, end="far")]), "far")   # the measured shot's own end when there is one
        self.assertIsNone(P.stroke_end(dict(id=2, events=[]), 1.0))                              # an unnamed point places nothing
        self.assertEqual(len(P.at_hits(events, assigned, fps, shots=shots)), 1)                  # no points to place the others in: left out
        legacy = P.at_hits(events, assigned, fps)                          # without shots: the ball's side, the lean only, never a knee
        self.assertEqual([h["side"] for h in legacy], ["far", "near", "near"])
        self.assertTrue(all(h["knee"] is None and h["leg"] is None and h["lean"] is not None for h in legacy))
        twins = [dict(point=1, shot=2, name="A", end="near", serve=False, t=1.0, knee=147.5, leg="l"),
                 dict(point=1, shot=3, name="B", end="far", serve=False, t=1.0, knee=None, leg=None)]   # two shots at one t: no crash
        self.assertEqual(P.at_hits(events, assigned, fps, shots=twins, points=points)[0]["knee"], 147.5)

    def test_summary_without_shots_has_no_knee_and_says_why(self):
        pts = [dict(id=1, start_t=0.0, end_t=3.0, near_player="A", far_player="B", winner_name="A")]
        hits = [dict(t=1.0, side="near", knee=None, leg=None, lean=12.0), dict(t=2.0, side="far", knee=None, leg=None, lean=3.0)]
        s = P.summary(hits, pts, {"near": "A", "far": "B"})
        self.assertEqual(s["A"]["knee"], (None, 0)); self.assertEqual(s["A"]["knee_why"], P.NO_SHOTS); self.assertEqual(s["A"]["lean"], (12.0, 1))
        self.assertIn(P.NO_SHOTS, P.knee_text(s["A"]["knee"], why=s["A"]["knee_why"]))
        shots = [dict(point=1, shot=2, name="A", end="near", serve=False, t=1.0, knee=150.0, leg="r", profile_src="3d"),
                 dict(point=1, shot=4, name="A", end="near", serve=False, t=1.5, knee=140.0, leg="r", profile_src="3d"),
                 dict(point=1, shot=6, name="A", end="near", serve=False, t=2.0, knee=130.0, leg="r", profile_src="2d")]
        s = P.summary(hits, pts, {"near": "A", "far": "B"}, shots=shots)
        self.assertEqual(s["A"]["knee"], (140.0, 3)); self.assertEqual(s["A"]["knee_src"], {"3d": 2, "2d": 1}); self.assertIsNone(s["A"]["knee_why"])


class ShotsGetTheBody(unittest.TestCase):
    def test_knee3d_is_on_every_shot_with_a_placed_stroke(self):
        cam = camera(); J = standing_body(root=(-0.6, 0.8))
        shots = [dict(point=1, shot=2, name="A", end="near", serve=False, t=1.0, knee=150.0, leg="r", profile2d=None),
                 dict(point=1, shot=3, name="A", end="near", serve=False, t=1.5, knee=150.0, leg="r", profile2d=None)]
        n = body3d.attach(shots, [stroke(J, point=1, shot=2)], cam)
        self.assertEqual(n, 1)
        self.assertIsNotNone(shots[0]["knee3d"]); self.assertIn("lean3d", shots[0]); self.assertIn("turn3d", shots[0]); self.assertEqual(shots[0]["side"], "forehand")
        self.assertNotIn("knee3d", shots[1])
        src = pathlib.Path(technique.__file__).with_name("cli.py").read_text()   # and shots.json is written only after the bodies and the gate
        self.assertLess(src.index("body3d.attach(shots"), src.index('"shots.json").write_text'))
        self.assertLess(src.index("gate_knees(shots, strokes, cam)"), src.index('"shots.json").write_text'))
        self.assertLess(src.index("assigned_raw = {"), src.index("clean_legs(assigned, tb)"))     # the 3D crop boxes see the legs as found
        self.assertIn("body3d.requests(shots, assigned_raw", src)


class RecordKeepsTheKnee(unittest.TestCase):
    def test_record_match_stores_the_gated_median_and_its_count(self):
        from tt_scout.profiles import record_match, load_records, aggregate
        from tt_scout.herocard import knee_pooled, player_stats
        from tests.test_profiles import fake_match
        tmp = pathlib.Path(tempfile.mkdtemp()); vid = tmp / "one.mp4"; vid.write_bytes(b"x" * 5000)
        pts, ms, q = fake_match("Sam", "Robin", 6, 5)
        contacts = {"Sam": [dict(t=1.0, point=1, shot=2, side="near", knee=150.0, leg="r", won=True), dict(t=11.0, point=2, shot=2, side="near", knee=140.0, leg="r", won=True),
                            dict(t=21.0, point=3, shot=2, side="near", knee=130.0, leg="l", won=True), dict(t=61.0, point=7, shot=2, side="near", knee=120.0, leg="l", won=False)],
                    "Robin": [dict(t=2.0, point=1, shot=3, side="far", knee=160.0, leg="l", won=False)]}
        hits = [dict(t=1.0, side="near", knee=150.0, leg="r", lean=12.0), dict(t=2.0, side="far", knee=None, leg=None, lean=5.0)]
        rec = record_match(tmp, vid, {"near": "Sam", "far": "Robin"}, pts, ms, q, posture=dict(hits=hits, contacts=contacts), recorded="2026-09-28T20:00:00+01:00")
        self.assertEqual(rec["schema"], 2)
        self.assertEqual(rec["posture"]["Sam"]["knee"], [150.0, 140.0, 130.0, 120.0]); self.assertEqual(rec["posture"]["Sam"]["knee_median"], [135.0, 4])
        self.assertEqual(rec["posture"]["Sam"]["knee_won"], [True, True, True, False])
        self.assertEqual(rec["posture"]["Robin"]["knee_median"], [None, 1])                     # under three contacts: not measurable
        self.assertEqual(rec["posture"]["Sam"]["lean"], [12.0]); self.assertEqual(rec["posture"]["Robin"]["lean"], [5.0])
        recs = load_records(tmp)
        a = aggregate("sam", recs)
        self.assertEqual(a["knee"], (135.0, 4)); self.assertEqual(a["rows"][0]["knee"], 135.0)     # the profile's fact and its trend point
        self.assertEqual(knee_pooled(recs, "Sam"), (135.0, 4)); self.assertEqual(player_stats(recs, "Sam")["knee"], 135.0)   # the hero card
        self.assertEqual(sorted(a["tot"]["knee_won"]), [130.0, 140.0, 150.0]); self.assertEqual(a["tot"]["knee_lost"], [120.0])
        self.assertEqual(aggregate("robin", recs)["knee"], (None, 1))
        old = json.loads((tmp / "matches" / f"{rec['id']}.json").read_text())                  # a schema-1 record's knees are another quantity
        old["posture"]["Sam"] = dict(knee=[100.0, 100.0, 100.0], lean=[1.0, 1.0, 1.0], won=[True, True, True]); old["schema"] = 1
        self.assertEqual(knee_pooled([old], "Sam"), (None, 0))


class ProfilePage(unittest.TestCase):
    """The profile page reads the one gated knee: a knee criticism stored by a record from before the gate (schema 1, the 3D or the
    min-leg knee against pros 127) is not printed, the fact says why the knee is missing, and the trend shows the counted records."""
    def setUp(self):
        from tt_scout.critique import Critique
        from tests.test_profiles import fake_match
        from tt_scout.profiles import record_match
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.knees = Critique("Sam", "Standing too upright at contact", "knees 148 deg at contact in 3D (straight = 180, pros 127), 16 hits", "Wider stance", 0.5, "knees")
        self.pace = Critique("Sam", "Slow rally pace", "6.0 m/s off the racket", "Turn hips", 0.4, "pace")
        def rec(name, a_pts, b_pts, level, when, knee=None, crit=(), lean=(12.0,)):
            vid = self.tmp / f"{name}.mp4"; vid.write_bytes(name.encode() * 900)
            pts, ms, q = fake_match("Sam", "Robin", a_pts, b_pts, level=level)
            contacts = {"Sam": [dict(t=1.0 + i, point=1, shot=2, side="near", knee=k, leg="r", won=True) for i, k in enumerate(knee or [])]}
            hits = [dict(t=1.0, side="near", knee=None, leg=None, lean=l) for l in lean]
            return record_match(self.tmp, vid, {"near": "Sam", "far": "Robin"}, pts, ms, q, posture=dict(hits=hits, contacts=contacts),
                                critique={"Sam": list(crit)}, recorded=when)
        self.rec = rec

    def test_an_old_knee_criticism_is_not_shown_and_the_fact_says_why(self):
        from tt_scout.profiles import load_records, aggregate
        from tt_scout.profile_html import player_page
        r = self.rec("one", 6, 5, "good", "2026-09-26T20:00:00+01:00", crit=(self.knees, self.pace))
        self.assertEqual(r["critique"]["Sam"][0]["key"], "knees")
        f = self.tmp / "matches" / f"{r['id']}.json"; old = json.loads(f.read_text())
        old["schema"] = 1; old["posture"]["Sam"] = dict(knee=[148.0] * 16, lean=[12.0] * 16, won=[True] * 16)   # as written before the gate
        f.write_text(json.dumps(old))
        a = aggregate("sam", load_records(self.tmp))
        self.assertTrue(a["has_pose"]); self.assertEqual(a["knee"], (None, 0)); self.assertEqual(a["rows"][0]["schema"], 1)
        player_page(a, self.tmp); page = (self.tmp / "sam.html").read_text()
        self.assertNotIn("knees 148", page); self.assertNotIn("pros 127", page); self.assertNotIn("Standing too upright", page)   # ("in 3D" alone sits in a CSS comment)
        self.assertIn("Slow rally pace", page); self.assertIn("analysed before the knee was gated", page)
        self.assertIn("re-analyse the recording", page); self.assertNotIn("no skeletons yet", page)
        self.rec("two", 7, 4, "good", "2026-09-28T20:00:00+01:00", knee=[150.0, 140.0, 130.0], crit=(self.knees,))   # written now: shown
        a = aggregate("sam", load_records(self.tmp)); self.assertEqual(a["knee"], (140.0, 3))
        player_page(a, self.tmp); page = (self.tmp / "sam.html").read_text()
        self.assertIn("Standing too upright", page); self.assertIn("1 of 2", page); self.assertIn("median of 3 forehand contacts", page)

    def test_no_skeletons_at_all_is_said_as_such(self):
        from tt_scout.profiles import load_records, aggregate
        from tt_scout.profile_html import player_page
        self.rec("one", 6, 5, "good", "2026-09-26T20:00:00+01:00", lean=())
        a = aggregate("sam", load_records(self.tmp)); self.assertFalse(a["has_pose"])
        player_page(a, self.tmp); self.assertIn("no skeletons yet", (self.tmp / "sam.html").read_text())

    def test_the_knee_trend_shows_the_counted_recordings_only(self):
        from tt_scout.profiles import load_records, aggregate
        from tt_scout.profile_html import player_page
        self.rec("one", 6, 5, "good", "2026-09-24T20:00:00+01:00", knee=[141.0, 140.0, 139.0])
        self.rec("two", 6, 5, "unreliable", "2026-09-26T20:00:00+01:00", knee=[99.0, 99.0, 99.0])
        self.rec("three", 6, 5, "good", "2026-09-28T20:00:00+01:00", knee=[151.0, 150.0, 149.0])
        a = aggregate("sam", load_records(self.tmp)); self.assertEqual(a["knee"][1], 6)          # the fact skips the unreliable one ...
        player_page(a, self.tmp); page = (self.tmp / "sam.html").read_text()
        self.assertIn("140°", page); self.assertIn("150°", page); self.assertNotIn("99°", page)   # ... and so does the trend


class ReportStrips(unittest.TestCase):
    def test_the_posture_sub_head_promises_frames_only_when_there_are_some(self):
        from tt_scout.report_html import _stance
        pm = dict(n=5, knee=(150.0, 3), lean=(10.0, 5), knee_won=(None, 1), knee_lost=(None, 2), knee_src={"3d": 3, "2d": 0}, knee_label=P.KNEE_LABEL, knee_why=None)
        posture = {"Sam": pm, "Robin": dict(pm, knee=(None, 1), knee_src={"3d": 0, "2d": 0})}
        h = _stance(dict(figures={}, frames={"Sam": [], "Robin": []}), posture, ["Sam", "Robin"])
        self.assertNotIn("deepest knee bend to straightest", h); self.assertIn("150\u00b0", h); self.assertIn(P.NOT_MEASURABLE, h)
        self.assertIn("trunk lean forward, as the camera sees it", h)
        h = _stance(dict(figures={}, frames={"Sam": [dict(src="x.jpg", knee=150, t=1.0, point=1)], "Robin": []}), posture, ["Sam", "Robin"])
        self.assertIn("deepest knee bend to straightest", h)
        h = _stance(dict(figures={}, frames={}), {"Sam": dict(pm, knee=(None, 0), knee_why=P.NO_SHOTS)}, ["Sam", "Robin"])
        self.assertIn(P.NO_SHOTS, h)


if __name__ == "__main__":
    unittest.main()


class PaceNote(unittest.TestCase):
    """The pace note speaks km/h like the clips' gauge, and in a clip (notes from the shots before a point) says 'so far'."""
    def shots(self, name="Sam", speeds=(4.0, 4.5, 5.0, 6.0, 5.5)):
        return [dict(name=name, serve=False, speed=v, t=float(i), point=1, shot=i + 2) for i, v in enumerate(speeds)]

    def test_whole_match_note_is_in_kmh(self):
        from tt_scout.critique import critique, PRO
        c = next(c for c in critique(self.shots(), [], "Sam") if c.key == "pace")
        self.assertEqual(c.evidence, f"typically 18 km/h off the racket, fastest 22 (pros {3.6 * PRO['speed']:.0f})")
        self.assertNotIn("m/s", c.evidence)

    def test_clip_note_says_so_far(self):
        from tt_scout.critique import critique
        c = next(c for c in critique(self.shots(), [], "Sam", so_far=True) if c.key == "pace")
        self.assertIn("fastest 22 so far", c.evidence)
