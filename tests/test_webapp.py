"""The upload page's server (tt_scout/webapp.py): input checks, the file-serving whitelist, the players list, uploads and deletes.
No analysis runs here: the worker thread is off, so a started match just waits in the queue."""
import http.server, json, pathlib, tempfile, threading, unittest, urllib.error, urllib.request
from unittest import mock
import cv2, numpy as np
from tt_scout import webapp
from tt_scout.profiles import slug as profiles_slug

GOOD = [[518, 773], [649, 667], [1228, 671], [1379, 773]]          # IMG_3144's table: bottom-left, top-left, top-right, bottom-right


def req(url, method="GET", data=None, headers=None):
    r = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(r, timeout=30) as f:
            return f.status, f.read(), dict(f.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def _record(root, rid, players, winner, report, points, games):
    (root / "matches").mkdir(parents=True, exist_ok=True)
    rec = dict(id=rid, recorded="2026-09-25T23:00:00+01:00", players=players, winner=winner, report=report,
               verdict=dict(level="good"), games=[dict(winner=g) for g in games], points=[dict(winner=w) for w in points],
               portrait={players[0]: report.rsplit("/", 1)[0] + "/contact_1_2.jpg"})
    (root / "matches" / f"{rid}.json").write_text(json.dumps(rec))


class Inputs(unittest.TestCase):
    def test_slug_matches_profiles(self):
        for n in ["Sam", "  robin ", "O'Neil-Jones", "Ann Marie", "李", "", "A.B"]:
            self.assertEqual(webapp.slug(n), profiles_slug(n))

    def test_valid_corners(self):
        self.assertEqual(webapp.valid_corners(GOOD, 1920, 1080), [[float(x), float(y)] for x, y in GOOD])
        self.assertIsNone(webapp.valid_corners([GOOD[0], GOOD[2], GOOD[1], GOOD[3]], 1920, 1080))      # crossed
        self.assertIsNone(webapp.valid_corners(GOOD[::-1], 1920, 1080))                                # the wrong way round
        self.assertIsNone(webapp.valid_corners(GOOD[:3], 1920, 1080))
        self.assertIsNone(webapp.valid_corners([[5000, 1], *GOOD[1:]], 1920, 1080))                     # outside the frame
        self.assertIsNone(webapp.valid_corners([[float("nan"), 1], *GOOD[1:]], 1920, 1080))
        self.assertIsNone(webapp.valid_corners("nonsense", 1920, 1080))

    def test_clean_name(self):
        self.assertEqual(webapp.clean_name("<b>Sam</b>", "x"), "bSamb")
        self.assertEqual(webapp.clean_name("  Robin   K  ", "x"), "Robin K")
        self.assertEqual(webapp.clean_name("", "Left player"), "Left player")
        self.assertEqual(len(webapp.clean_name("a" * 80, "x")), 30)


class Site(unittest.TestCase):
    """A project folder in tmp: profiles with two matches (one made by an upload), and the server on a free port."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); root = self.root = pathlib.Path(self.tmp.name)
        for name, val in (("ROOT", root), ("PROFILES", root / "profiles"), ("JOBS", root / "web_jobs")):
            p = mock.patch.object(webapp, name, val); p.start(); self.addCleanup(p.stop)
        rep = root / "out_own_product" / "m1"; rep.mkdir(parents=True)
        (rep / "report.html").write_text("<p>report</p>"); (rep / "clip.mp4").write_bytes(bytes(range(256)) * 8)
        (root / "profiles").mkdir(); (root / "profiles" / "sam.html").write_text("<p>Sam</p>")
        (root / "data").mkdir(); (root / "data" / "secret.mp4").write_bytes(b"raw video")
        _record(root / "profiles", "aaa", ["Sam", "Robin"], "Sam", "out_own_product/m1/report.html", ["Sam"] * 5 + ["Robin"] * 3, ["Sam"])
        job = root / "web_jobs" / "0123456789ab" / "work" / "report" / "match"; job.mkdir(parents=True)
        (job / "report.html").write_text("<p>upload</p>")
        _record(root / "profiles", "bbb", ["Robin", "Alex"], "Alex", "web_jobs/0123456789ab/work/report/match/report.html", ["Alex"] * 4, ["Alex"])
        self.store = webapp.Store(root / "web_jobs", start_worker=False)
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), webapp.Handler); self.srv.store = self.store; self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def tearDown(self):
        self.srv.shutdown(); self.srv.server_close(); self.tmp.cleanup()

    def test_players_data(self):
        d = webapp.players_data()
        by = {a["name"]: a for a in d["players"]}
        self.assertEqual(set(by), {"Sam", "Robin", "Alex"})
        self.assertEqual((by["Robin"]["matches"], by["Robin"]["won"], by["Robin"]["lost"]), (2, 0, 2))
        self.assertEqual((by["Sam"]["points_won"], by["Sam"]["points"]), (5, 8))
        m = next(m for m in d["matches"] if m["id"] == "aaa")
        self.assertEqual((m["games"], m["points"], m["report"]), ([1, 0], [5, 3], "/out_own_product/m1/report.html"))

    def test_serves_only_reports_and_profile_pages(self):
        u = self.url
        self.assertEqual(req(u + "/profiles/sam.html")[0], 200)
        self.assertEqual(req(u + "/out_own_product/m1/report.html")[0], 200)
        self.assertEqual(req(u + "/web_jobs/0123456789ab/work/report/match/report.html")[0], 200)
        self.assertEqual(req(u + "/profiles/matches/aaa.json")[0], 404)                   # the records stay private
        self.assertEqual(req(u + "/data/secret.mp4")[0], 404)
        self.assertEqual(req(u + "/out_own_product/m1/../../data/secret.mp4")[0], 404)
        self.assertEqual(req(u + "/out_own_product/%2e%2e/%2e%2e/data/secret.mp4")[0], 404)
        code, body, h = req(u + "/out_own_product/m1/clip.mp4", headers={"Range": "bytes=10-19"})
        self.assertEqual((code, body, h["Content-Range"]), (206, bytes(range(10, 20)), "bytes 10-19/2048"))
        self.assertEqual(req(u + "/out_own_product/m1/clip.mp4", headers={"Range": "bytes=5000-"})[0], 416)

    def test_guards(self):
        u = self.url
        self.assertEqual(req(u + "/upload?name=x.mp4", "POST", b"abc")[0], 403)                  # no page header: another site's form
        self.assertEqual(req(u + "/jobs", headers={"Host": "evil.example"})[0], 403)              # DNS rebinding
        self.assertEqual(req(u + "/upload?name=notes.txt", "POST", b"abc", {"X-TT-Scout": "1"})[0], 415)

    def test_upload_start_delete(self):
        vid = self.root / "clip.mp4"
        w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 180))
        for i in range(45):
            w.write(np.full((180, 320, 3), i * 5 % 255, np.uint8))
        w.release()
        code, body, _ = req(self.url + "/upload?name=IMG_1.MOV", "POST", vid.read_bytes(), {"X-TT-Scout": "1"})
        j = json.loads(body)
        self.assertEqual((code, j["w"], j["h"], j["state"]), (200, 320, 180, "uploaded"))
        self.assertEqual(req(self.url + f"/job/{j['id']}/frame.jpg")[0], 200)
        H = {"X-TT-Scout": "1", "Content-Type": "application/json"}
        small = [[40, 150], [80, 60], [240, 60], [280, 150]]
        crossed = json.dumps({"corners": [small[0], small[2], small[1], small[3]]}).encode()
        self.assertEqual(req(self.url + f"/job/{j['id']}/start", "POST", crossed, H)[0], 400)
        body = json.dumps({"corners": small, "near": "Sam", "far": "sam", "profile": False}).encode()
        code, out, _ = req(self.url + f"/job/{j['id']}/start", "POST", body, H)
        s = json.loads(out)
        self.assertEqual((code, s["state"], s["near"], s["far"], s["profile"]), (200, "queued", "Sam", "sam", False))
        self.assertEqual(req(self.url + f"/job/{j['id']}/start", "POST", body, H)[0], 409)     # already queued
        self.assertEqual(req(self.url + f"/job/{j['id']}/delete", "POST", b"", {"X-TT-Scout": "1"})[0], 200)
        self.assertFalse((self.root / "web_jobs" / j["id"]).exists()); self.assertIsNone(self.store.get(j["id"]))

    def test_dropped_connection_is_quiet(self):
        """Chrome opens spare connections and resets them unused; that must not print a traceback in the terminal."""
        import io, socket, struct, time
        srv = webapp.Server(("127.0.0.1", 0), webapp.Handler); srv.store = self.store
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        err = io.StringIO()
        with mock.patch("sys.stderr", err):
            c = socket.create_connection(srv.server_address)
            c.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))     # close with a reset, like the browser
            c.close(); time.sleep(0.3)
        srv.shutdown(); srv.server_close()
        self.assertNotIn("Traceback", err.getvalue())

    def test_second_serve_opens_the_first(self):
        """Running `tt-scout serve` while it already runs: a line saying so, no traceback; another program on the port: a clear exit."""
        import io, socket
        srv = webapp.Server(("127.0.0.1", 0), webapp.Handler); srv.store = self.store
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        out = io.StringIO()
        with mock.patch("sys.stdout", out):
            self.assertIsNone(webapp.serve(srv.server_address[1], open_browser=False))
        self.assertIn("already running", out.getvalue())
        srv.shutdown(); srv.server_close()
        other = socket.socket(); other.bind(("127.0.0.1", 0)); other.listen(1)       # a port held by something that is not tt_scout
        with self.assertRaises(SystemExit) as cm:
            webapp.serve(other.getsockname()[1], open_browser=False)
        self.assertIn("in use by another program", str(cm.exception)); other.close()

    def test_match_page_data(self):
        """The site's watch page: the whole-match video, its poster and each point's clip, only where the files exist."""
        m1 = self.root / "out_own_product" / "m1"
        (m1 / "clips").mkdir(); (m1 / "clips" / "point_001.mp4").write_bytes(b"x"); (m1 / "full_match.mp4").write_bytes(b"x")
        rec = json.loads((self.root / "profiles" / "matches" / "aaa.json").read_text())
        for i, q in enumerate(rec["points"], start=1):
            q["id"] = i
        (self.root / "profiles" / "matches" / "aaa.json").write_text(json.dumps(rec))
        m = webapp.match_data("aaa000000000") or webapp.match_data("aaa")
        self.assertIsNone(webapp.match_data("aaa000000000"))
        self.assertIsNone(webapp.match_data("../../etc"))
        code, body, _ = req(self.url + "/match/aaa.json")
        self.assertEqual(code, 404)                                                         # ids are 12 hex characters
        rec["id"] = "aaaaaaaaaaaa"; (self.root / "profiles" / "matches" / "aaa.json").write_text(json.dumps(rec))
        code, body, _ = req(self.url + "/match/aaaaaaaaaaaa.json")
        m = json.loads(body)
        self.assertEqual(code, 200)
        self.assertEqual((m["video"], m["poster"]), ("/out_own_product/m1/full_match.mp4", None))
        self.assertEqual([q["clip"] for q in m["points"][:2]], ["/out_own_product/m1/clips/point_001.mp4", None])
        self.assertEqual(req(self.url + m["points"][0]["clip"])[0], 200)                  # and the server hands the clip out

    def test_forget_records(self):
        self.assertEqual(webapp.forget_records(self.root / "web_jobs" / "0123456789ab"), 1)
        self.assertEqual([r["id"] for r in webapp.load_records()], ["aaa"])


if __name__ == "__main__":
    unittest.main()
