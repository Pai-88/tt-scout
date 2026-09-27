"""The upload page: drop a match video into a browser tab, click the table's four corners, name the two players, get the report.


    tt-scout serve        (or: .venv/bin/python -m tt_scout.cli serve)      ->  http://127.0.0.1:8770 opens in the browser

This first version runs on this Mac. It listens on 127.0.0.1 only, so nothing else on the network can reach it, and every video and
report it makes stays in web_jobs/<id>/, which the page deletes with one button. Each match goes through scripts/new_match.py, the
same pipeline as the terminal, one match at a time. The skeletons come from Apple Vision (tools/pose, tools/pose3d), which exists
only on macOS; that is what keeps the page on the Mac until an open pose model replaces it.

Players (#/players) lists every player and every match in profiles/, whether it came through this page or the terminal. A match analysed here goes into both players' profiles unless the box
is unticked; deleting it here takes it out of the profiles again. Reports and profile pages are served at the same paths they have
under the project folder (/profiles/sam.html, /out_own_product/<match>/report.html), so the links between them keep working; only
the profile pages and the report folders the records point to are served, nothing else in the project.
"""
import errno, http.server, json, math, os, pathlib, queue, re, secrets, shutil, signal, subprocess, sys, threading, time, urllib.parse, urllib.request
from .config import ROOT

CODE = pathlib.Path(__file__).resolve().parent.parent    # the repo: scripts/, analysis/, .venv (ROOT is where the data lives; the same folder)
JOBS = ROOT / "web_jobs"
PAGE = pathlib.Path(__file__).with_name("webapp.html")
VENV_PY = CODE / ".venv" / "bin" / "python"
PY = str(VENV_PY) if VENV_PY.exists() else sys.executable     # the pipeline needs numpy and OpenCV; this server needs only the standard library
STAGES = [("convert", "Preparing the video"), ("table", "Fitting the table"), ("track", "Tracking the ball"),
          ("points", "Finding the points"), ("pose", "Reading the players' bodies"), ("report", "Building the report")]
VIDEO_EXT = {".mov", ".mp4", ".m4v"}
MAX_UPLOAD = 40 << 30
ID_RE = re.compile(r"[0-9a-f]{12}")
MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".json": "application/json",
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".svg": "image/svg+xml", ".mp4": "video/mp4",
        ".webm": "video/webm", ".txt": "text/plain; charset=utf-8"}
PUBLIC = ("id", "created", "name", "size", "w", "h", "fps", "duration", "hdr", "state", "stage", "stage_t", "detail", "message",
          "near", "far", "corners", "camera", "finished", "profile", "auto_state", "auto_corners", "auto_method")
PROFILES = ROOT / "profiles"


def slug(name):
    """= tt_scout.profiles.slug (that module needs numpy; this server does not). tests/test_webapp.py checks they agree."""
    return re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower()).strip("-") or "player"


def load_records(root=None):
    """The match records behind the profiles (profiles/matches/*.json, written by tt_scout.profiles.record_match)."""
    recs = []
    for f in sorted((pathlib.Path(root or PROFILES) / "matches").glob("*.json")):
        try:
            recs.append(json.loads(f.read_text()))
        except (OSError, ValueError):
            continue
    return sorted(recs, key=lambda r: r.get("recorded") or "", reverse=True)


def _url(rel):
    """A path relative to the project folder -> the URL this server gives it (the same path)."""
    return "/" + "/".join(urllib.parse.quote(x) for x in pathlib.PurePosixPath(rel).parts) if rel else None


def _media(rec):
    """What a match record's report folder holds to watch: the whole-match video, its poster frame, and a clip per point (by id)."""
    if not rec.get("report"):
        return dict(video=None, poster=None, clips={})
    base = pathlib.PurePosixPath(rec["report"]).parent
    rdir = ROOT / base
    have = lambda name: _url(str(base / name)) if (rdir / name).is_file() else None
    clips = {}
    for q in rec.get("points") or []:
        if isinstance(q.get("id"), int):
            u = have(f"clips/point_{q['id']:03d}.mp4")
            if u:
                clips[q["id"]] = u
    return dict(video=have("full_match.mp4"), poster=have("hero.jpg"), clips=clips)


def match_data(mid, root=None):
    """One match for its page on the site (#/match/<id>): the whole-match video, every point with its clip, the games. None if unknown."""
    if not ID_RE.fullmatch(mid or ""):
        return None
    rec = next((r for r in load_records(root) if r.get("id") == mid), None)
    if not rec:
        return None
    names = rec.get("players") or []
    media = _media(rec)
    games = [dict(n=g.get("n"), score=[(g.get("score") or {}).get(n, 0) for n in names], winner=g.get("winner"),
                  finished=g.get("finished", True), first=g.get("first"), last=g.get("last")) for g in rec.get("games") or []]
    points = [dict(id=q.get("id"), t=q.get("t"), server=q.get("server"), winner=q.get("winner"), shots=q.get("shots"),
                   ending=q.get("ending"), clip=media["clips"].get(q.get("id"))) for q in rec.get("points") or []]
    return dict(id=rec.get("id"), recorded=rec.get("recorded"), players=names, winner=rec.get("winner"),
                verdict=(rec.get("verdict") or {}).get("level"), games=games, points=points, video=media["video"],
                poster=media["poster"], report=_url(rec.get("report")), profiles=[f"/profiles/{slug(n)}.html" for n in names])


def players_data(root=None):
    """Every player and every match in the profiles, for the Players page."""
    recs = load_records(root)
    people, matches = {}, []
    for r in recs:
        names = r.get("players") or []
        pts = r.get("points") or []
        games = r.get("games") or []
        won = {n: sum(1 for g in games if g.get("winner") == n) for n in names}
        media = _media(r)
        matches.append(dict(id=r.get("id"), recorded=r.get("recorded"), players=names, winner=r.get("winner"),
                            games=[won.get(n, 0) for n in names], points=[sum(1 for q in pts if q.get("winner") == n) for n in names],
                            n_points=len(pts), verdict=(r.get("verdict") or {}).get("level"),
                            report=_url(r.get("report")), profiles=[f"/profiles/{slug(n)}.html" for n in names],
                            poster=media["poster"], video=media["video"], n_clips=len(media["clips"])))
        for n in names:
            k = slug(n)
            a = people.setdefault(k, dict(name=n, slug=k, matches=0, won=0, lost=0, points_won=0, points=0, last=None, portrait=None,
                                          profile=f"/profiles/{k}.html"))
            a["matches"] += 1
            a["won"] += r.get("winner") == n
            a["lost"] += r.get("winner") not in (None, n)
            a["points_won"] += sum(1 for q in pts if q.get("winner") == n); a["points"] += len(pts)
            if not a["last"] or (r.get("recorded") or "") > a["last"]:
                a["last"] = r.get("recorded"); a["name"] = n
            if not a["portrait"] and (r.get("portrait") or {}).get(n):
                a["portrait"] = _url(r["portrait"][n])
    return dict(players=sorted(people.values(), key=lambda a: (-a["matches"], a["name"].lower())), matches=matches)


def served_dirs(store_root=None, root=None):
    """The only folders this server hands files out of: the profile pages, and each report folder a match record or an upload points at."""
    root = pathlib.Path(root or PROFILES); store_root = pathlib.Path(store_root or JOBS)
    dirs = {root.resolve()}
    for r in load_records(root):
        if r.get("report"):
            dirs.add((ROOT / r["report"]).resolve().parent)
    for d in store_root.glob("*/work/report/match"):
        dirs.add(d.resolve())
    return dirs


def _converter():
    """analysis/convert_iphone.py: the same probe and colour/scale chain the pipeline converts with."""
    if str(CODE / "analysis") not in sys.path:
        sys.path.insert(0, str(CODE / "analysis"))
    import convert_iphone
    return convert_iphone


def preview(video, out_jpg):
    """The frame the corners are clicked on, 2 s in, through the same colour conversion and scaling as the video the pipeline reads,
    so a click is in the converted video's pixels. Returns what the page shows about the file."""
    conv = _converter()
    info = conv.probe(video)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    hdr = conv.is_hdr(v)
    dur = float(info.get("format", {}).get("duration") or v.get("duration") or 0)
    t = min(2.0, dur / 3) if dur else 0.0
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1",
                    "-vf", conv.video_filter(hdr, 1080), "-q:v", "2", str(out_jpg)], check=True, timeout=180)
    fr = next(s for s in conv.probe(out_jpg)["streams"] if s["codec_type"] == "video")          # the size after rotation and scaling
    num, den = (int(x) for x in (v.get("r_frame_rate") or "0/1").split("/"))
    return dict(w=int(fr["width"]), h=int(fr["height"]), fps=round(num / den, 2) if den else 0.0, duration=round(dur, 1), hdr=bool(hdr))


def clean_name(s, default):
    """A player's name as typed -> letters, digits, spaces and . ' - only (it goes into the report and its title), at most 30 characters."""
    s = re.sub(r"[^\w .'\-]", "", str(s or ""), flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()[:30]
    return s or default


def valid_corners(c, w, h):
    """The four corners from the page, in tt_scout's order (bottom-left, top-left, top-right, bottom-right in the picture): inside
    the frame (a little slack) and a convex quadrilateral going that way round. Returns [[x, y]] * 4 or None."""
    try:
        P = [[float(x), float(y)] for x, y in c]
    except (TypeError, ValueError):
        return None
    if len(P) != 4 or not all(math.isfinite(v) for p in P for v in p):
        return None
    if not all(-0.05 * w <= x <= 1.05 * w and -0.05 * h <= y <= 1.05 * h for x, y in P):
        return None
    cross = []
    for i in range(4):
        (x0, y0), (x1, y1), (x2, y2) = P[i], P[(i + 1) % 4], P[(i + 2) % 4]
        cross.append((x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1))
    if not all(z > 0 for z in cross):                      # y points down in a picture: this is clockwise as seen, all turns the same way
        return None
    return [[round(x, 1), round(y, 1)] for x, y in P]


def forget_records(job_dir, root=None):
    """Take the match records whose report lives in this upload's folder out of the profiles. Returns how many went."""
    job_dir = pathlib.Path(job_dir).resolve(); n = 0
    for f in (pathlib.Path(root or PROFILES) / "matches").glob("*.json"):
        try:
            rep = json.loads(f.read_text()).get("report")
        except (OSError, ValueError):
            continue
        if rep and job_dir in (ROOT / rep).resolve().parents:
            f.unlink(); n += 1
    return n


class Store:
    """The matches: one folder each under web_jobs/, its state in job.json (so a restart keeps the list), one worker thread running
    one pipeline at a time."""

    def __init__(self, root=None, start_worker=True):
        self.root = pathlib.Path(root or JOBS); self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock(); self.jobs = {}; self.procs = {}; self.q = queue.Queue()
        for f in sorted(self.root.glob("*/job.json")):
            try:
                j = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            if j.get("state") in ("queued", "running"):
                j.update(state="failed", message="The page was closed down while this match was being analysed. Start it again.")
            elif j.get("state") == "uploading":
                j.update(state="failed", message="The upload did not finish.")
            self.jobs[j["id"]] = j; self._save(j)
        if start_worker:
            threading.Thread(target=self._worker, daemon=True).start()

    def _save(self, j):
        d = self.root / j["id"]
        if d.exists():
            tmp = d / "job.json.tmp"; tmp.write_text(json.dumps(j, indent=1)); tmp.replace(d / "job.json")

    def new(self, **kw):
        jid = secrets.token_hex(6)
        (self.root / jid).mkdir()
        j = dict(id=jid, created=time.time(), state="uploading", stage=None, stage_t={}, detail="", message="", **kw)
        with self.lock:
            self.jobs[jid] = j; self._save(j)
        return j

    def get(self, jid):
        return self.jobs.get(jid) if ID_RE.fullmatch(jid or "") else None

    def update(self, jid, **kw):
        with self.lock:
            j = self.jobs[jid]; j.update(kw); self._save(j)
            return j

    def public(self, j):
        return {k: j.get(k) for k in PUBLIC}

    def status(self, jid):
        j = self.jobs[jid]
        out = self.public(j)
        now = time.time()
        order = [k for k, _ in STAGES]
        cur = order.index(j["stage"]) if j.get("stage") in order else -1
        rows = []
        for i, (k, label) in enumerate(STAGES):
            t = (j.get("stage_t") or {}).get(k)
            secs = None if not t else round((t[1] or now) - t[0])
            if j["state"] == "done" or i < cur:
                st = "done"
            elif i == cur:
                st = "active" if j["state"] == "running" else "stopped"
            else:
                st = "wait"
            rows.append(dict(key=k, label=label, state=st, secs=secs))
        out["stages"] = rows
        if j["state"] in ("failed", "refused", "stopped"):
            log = self.root / jid / "log.txt"
            out["log"] = log.read_text(errors="replace").splitlines()[-40:] if log.exists() else []
        if j["state"] == "done":
            rep = (self.root / jid / "work" / "report" / "match" / "report.html").resolve()
            # at its project path, so the report's links to the profiles work; a store outside the project (tests) uses /job/<id>/r/
            out["report_url"] = _url(rep.relative_to(ROOT.resolve())) if ROOT.resolve() in rep.parents else f"/job/{jid}/r/report.html"
        return out

    def list(self):
        return [self.public(j) for j in sorted(self.jobs.values(), key=lambda j: -j["created"])]

    def find_table(self, jid):
        """scripts/find_table.py on the upload: the table's corners by colour, else by its surface and white edge line, in the pixels
        of the video the analysis will read. The page places them; the person checks the net line and drags a corner if needed."""
        d = self.root / jid; out = d / "auto_table.json"
        try:
            subprocess.run([PY, str(CODE / "scripts" / "find_table.py"), str(d / self.jobs[jid]["upload"]), str(out)], cwd=CODE,
                           capture_output=True, timeout=600)
            res = json.loads(out.read_text())
        except Exception:
            res = {}
        if jid in self.jobs:
            c = res.get("corners")
            j = self.jobs[jid]
            ok = valid_corners(c, j.get("w") or 0, j.get("h") or 0) if c else None
            self.update(jid, auto_state="found" if ok else "not found", auto_corners=ok, auto_method=res.get("method"))

    def start(self, jid, corners, near, far, profile=True):
        with self.lock:
            j = self.jobs[jid]
            if j["state"] in ("queued", "running", "uploading"):
                return False
            self.update(jid, state="queued", corners=corners, near=near, far=far, profile=bool(profile), stage=None, stage_t={}, detail="",
                        message="", stop=False, finished=None)
        self.q.put(jid)
        return True

    def stop(self, jid):
        with self.lock:
            j = self.jobs.get(jid)
            if not j:
                return
            j["stop"] = True
            p = self.procs.get(jid)
        if p and p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)               # the pipeline and everything it started (ffmpeg, pose)
            except ProcessLookupError:
                pass
        elif j["state"] == "queued":
            self.update(jid, state="stopped", message="Stopped before it started.")

    def delete(self, jid):
        self.stop(jid)
        for _ in range(100):                                   # let the worker see the process end before the folder goes
            if jid not in self.procs:
                break
            time.sleep(0.1)
        with self.lock:
            self.jobs.pop(jid, None)
        gone = forget_records(self.root / jid)
        shutil.rmtree(self.root / jid, ignore_errors=True)
        if gone:                                                 # the players' pages are rebuilt without it
            subprocess.run([PY, "-m", "tt_scout.cli", "profiles"], cwd=CODE, capture_output=True, timeout=300)

    def shutdown(self):
        for jid in list(self.procs):
            self.stop(jid)

    def _worker(self):
        while True:
            jid = self.q.get()
            j = self.jobs.get(jid)
            if not j or j.get("stop") or j["state"] != "queued":
                continue
            try:
                self._run(jid)
            except Exception as e:                             # the page must never be left saying "running"
                self.procs.pop(jid, None)
                if jid in self.jobs:
                    self.update(jid, state="failed", message=f"The page's server hit an error: {e}", finished=time.time())

    def _run(self, jid):
        j = self.jobs[jid]; d = self.root / jid
        corners = ",".join(f"{v:.1f}" for p in j["corners"] for v in p)
        cmd = [PY, str(CODE / "scripts" / "new_match.py"), str(d / j["upload"]), "--work", str(d / "work"), "--near", j["near"],
               "--far", j["far"], "--table", "clicks:" + corners, "--name", "match", "--no-full-match", "--no-open"]
        if not j.get("profile", True):
            cmd.append("--no-profile")
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONWARNINGS": "ignore"}
        self.update(jid, state="running", started=time.time())
        report = None; refused = None
        with open(d / "log.txt", "w") as log:
            p = subprocess.Popen(cmd, cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                 start_new_session=True, env=env)
            self.procs[jid] = p
            for line in p.stdout:
                log.write(line); log.flush()
                s = line.strip()
                if s.startswith("== "):
                    k = s[3:].strip(); now = time.time()
                    with self.lock:
                        st = dict(j.get("stage_t") or {})
                        if j.get("stage") in st and st[j["stage"]][1] is None:
                            st[j["stage"]] = [st[j["stage"]][0], now]
                        st[k] = [now, None]
                        self.update(jid, stage=k, stage_t=st, detail="")
                elif s.startswith("REFUSED:"):
                    refused = s[8:].strip()
                elif s.startswith("report:"):
                    report = s[7:].strip()
                elif s.startswith("camera:"):
                    self.update(jid, camera=s[7:].strip(), detail=s)
                elif s and not s.startswith("+ ") and not s.startswith("Traceback"):
                    m = re.search(r"time=(\d+):(\d+):(\d+(?:\.\d+)?).*speed=", s)       # ffmpeg's progress line, in words
                    if m:
                        s = f"Converted {int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]):.0f} s of {j.get('duration') or 0:.0f} s"
                    self.update(jid, detail=s[-160:])
            rc = p.wait()
        self.procs.pop(jid, None)
        now = time.time()
        st = dict(j.get("stage_t") or {})
        if j.get("stage") in st and st[j["stage"]][1] is None:
            st[j["stage"]] = [st[j["stage"]][0], now]
        base = (d / "work" / "report" / "match").resolve()
        if j.get("stop"):
            self.update(jid, state="stopped", stage_t=st, message="Stopped. The video is still here: start it again, or delete it.", finished=now)
        elif rc == 3 and refused is not None:
            self.update(jid, state="refused", stage_t=st, message=refused or "The camera was not where tt_scout can read the match from.",
                        finished=now)
        elif rc == 0 and report and pathlib.Path(report).resolve() == base / "report.html" and (base / "report.html").exists():
            self.update(jid, state="done", stage_t=st, detail="", finished=now)
        else:
            label = dict(STAGES).get(j.get("stage"), "the start")
            self.update(jid, state="failed", stage_t=st, message=f"The analysis stopped during: {label.lower()}.", finished=now)


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "tt_scout"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    @property
    def store(self):
        return self.server.store

    def _host_ok(self):
        """Only this machine's own names: a web page elsewhere that points a hostname at 127.0.0.1 (DNS rebinding) is turned away."""
        return self.headers.get("Host", "").rsplit(":", 1)[0] in ("127.0.0.1", "localhost")

    def _send(self, code, body=b"", ctype="text/plain; charset=utf-8", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store"); self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _file(self, path):
        path = pathlib.Path(path)
        if not path.is_file():
            return self._send(404, b"not found")
        size = path.stat().st_size
        start, end, code = 0, size - 1, 200
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", (self.headers.get("Range") or "").strip())
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1)); end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            if start > end or start >= size:
                return self._send(416, b"", extra={"Content-Range": f"bytes */{size}"})
            code = 206
        self.send_response(code)
        self.send_header("Content-Type", MIME.get(path.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(end - start + 1)); self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache"); self.send_header("X-Content-Type-Options", "nosniff")
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        try:
            with open(path, "rb") as f:
                f.seek(start); left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk); left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self._host_ok():
            return self._send(403, b"forbidden")
        u = urllib.parse.urlparse(self.path)
        parts = [urllib.parse.unquote(p) for p in u.path.split("/") if p]
        if not parts:
            return self._file(PAGE)
        if parts == ["jobs"]:
            return self._json(200, {"jobs": self.store.list()})
        if parts == ["players.json"]:
            return self._json(200, players_data())
        if len(parts) == 2 and parts[0] == "match" and parts[1].endswith(".json"):
            m = match_data(parts[1][:-5])
            return self._json(200, m) if m else self._send(404, b"no such match")
        if len(parts) >= 3 and parts[0] == "job":
            j = self.store.get(parts[1])
            if not j:
                return self._send(404, b"no such match")
            d = self.store.root / j["id"]
            if parts[2:] == ["status"]:
                return self._json(200, self.store.status(j["id"]))
            if parts[2:] == ["frame.jpg"]:
                return self._file(d / "frame.jpg")
            if parts[2] == "r":                                    # the finished report and its clips, nothing outside that folder
                base = (d / "work" / "report" / "match").resolve()
                f = (base / "/".join(parts[3:] or ["report.html"])).resolve()
                return self._file(f) if base in f.parents else self._send(404, b"not found")
        if parts[0] in ("profiles", "web_jobs") or len(parts) > 1:       # profile pages, reports and their clips, at their project paths
            f = (ROOT / "/".join(parts)).resolve()
            prof = PROFILES.resolve()
            ok = f.parent == prof if prof in f.parents else any(d in f.parents for d in served_dirs(self.store.root) if d != prof)
            if ok and ROOT.resolve() in f.parents:                         # profile pages: the top level only (not the records)
                return self._file(f)
        return self._send(404, b"not found")

    def do_POST(self):
        # a custom header a page on another site cannot send here without a CORS preflight, which this server never grants
        if not self._host_ok() or self.headers.get("X-TT-Scout") != "1":
            return self._send(403, b"forbidden")
        u = urllib.parse.urlparse(self.path)
        parts = [p for p in u.path.split("/") if p]
        if parts == ["upload"]:
            return self._upload(u)
        if len(parts) == 3 and parts[0] == "job":
            j = self.store.get(parts[1])
            if not j:
                return self._send(404, b"no such match")
            act = parts[2]
            if act == "start":
                n = int(self.headers.get("Content-Length") or 0)
                if not 0 < n <= 10000:
                    return self._json(400, {"error": "bad request"})
                try:
                    body = json.loads(self.rfile.read(n))
                except ValueError:
                    return self._json(400, {"error": "bad request"})
                c = valid_corners(body.get("corners"), j.get("w") or 0, j.get("h") or 0)
                if c is None:
                    return self._json(400, {"error": "Those four points do not make a table outline. Click the corners again."})
                near = clean_name(body.get("near"), "Left player"); far = clean_name(body.get("far"), "Right player")
                if far == near:
                    far += " 2"
                if not self.store.start(j["id"], c, near, far, profile=body.get("profile", True) is not False):
                    return self._json(409, {"error": "This match is already being analysed."})
                return self._json(200, self.store.status(j["id"]))
            if act == "stop":
                self.store.stop(j["id"])
                return self._json(200, self.store.status(j["id"]))
            if act == "delete":
                self.store.delete(j["id"])
                return self._json(200, {"deleted": j["id"]})
        return self._send(404, b"not found")

    def _upload(self, u):
        q = urllib.parse.parse_qs(u.query)
        name = pathlib.Path(q.get("name", ["video.mov"])[0]).name[:120] or "video.mov"
        ext = pathlib.Path(name).suffix.lower()
        n = int(self.headers.get("Content-Length") or 0)
        if ext not in VIDEO_EXT:
            self.close_connection = True
            return self._json(415, {"error": "tt_scout reads .mov, .mp4 and .m4v videos."})
        if not 0 < n <= MAX_UPLOAD:
            self.close_connection = True
            return self._json(413, {"error": "That file is empty or larger than 40 GB."})
        j = self.store.new(name=name, size=n)
        d = self.store.root / j["id"]; dst = d / f"upload{ext}"
        got = 0
        try:
            with open(dst, "wb") as f:
                while got < n:
                    chunk = self.rfile.read(min(1 << 20, n - got))
                    if not chunk:
                        break
                    f.write(chunk); got += len(chunk)
        except (ConnectionError, OSError):
            pass
        if got < n:
            self.store.delete(j["id"]); self.close_connection = True
            return
        try:
            info = preview(dst, d / "frame.jpg")
        except Exception:
            self.store.delete(j["id"])
            return self._json(422, {"error": "This file could not be read as a video."})
        j = self.store.update(j["id"], state="uploaded", upload=dst.name, auto_state="finding", **info)
        threading.Thread(target=self.store.find_table, args=(j["id"],), daemon=True).start()   # the corners, found while the page opens
        return self._json(200, self.store.public(j))


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        """A browser dropping a connection is normal (a spare connection it opened and never used, a video it stopped loading):
        say nothing. Anything else still prints its traceback."""
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, TimeoutError)):
            return
        super().handle_error(request, client_address)


def _ours(port):
    """Whether what is listening on this port is a tt_scout upload page."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/jobs", timeout=2) as r:
            return r.status == 200 and "jobs" in json.loads(r.read())
    except Exception:
        return False


def serve(port=8770, open_browser=True):
    url = f"http://127.0.0.1:{port}/"
    try:
        srv = Server(("127.0.0.1", port), Handler)
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
        if _ours(port):                                        # started already (another terminal tab): open it, do not start a second
            print(f"tt_scout is already running at {url} (in another terminal); opening it.", flush=True)
            if open_browser and sys.platform == "darwin":
                subprocess.Popen(["open", url])
            return
        raise SystemExit(f"Port {port} is in use by another program. Start the page on another port:  tt-scout serve --port {port + 1}")
    store = Store()
    srv.store = store
    print(f"tt_scout upload page: {url}   (this Mac only; Ctrl-C stops it)", flush=True)
    if open_browser and sys.platform == "darwin":
        subprocess.Popen(["open", url])
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        store.shutdown(); srv.server_close()


if __name__ == "__main__":                                   # python3 -m tt_scout.webapp: any Python 3.10+, no packages needed
    import argparse
    ap = argparse.ArgumentParser(description="tt_scout's upload page"); ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args(); serve(a.port, open_browser=not a.no_open)
