"""tt-scout: one command from a phone recording to a scouting report.

    tt-scout analyse match.mp4 --near Alex --far Sam
        finds the table by itself, tracks the ball, finds the points and who won them, and writes out/<name>/report.html
        (confidence, score, placement charts, numbers, one clip per point). Nothing is uploaded.
    tt-scout report out/match --video match.mp4     rebuild the report from an analysis already on disk (seconds, no tracking)
    tt-scout calibrate match.mp4      click the four table corners yourself when the automatic outline is wrong
    tt-scout serve                    the upload page on this Mac: drop a video in the browser, click the corners, get the report
    tt-scout droptest drops.MOV --height 1.00
        the speed check on a real ball: drops from a measured height, filmed like a match, against the speed physics gives
"""
import argparse, json, pathlib, subprocess, sys
import cv2, numpy as np
from .config import Config, ROOT, NET_X, TABLE_WIDTH as TW


def profile_links(a, report_dir):
    """[(name, href from the report folder)] for the two players when their match goes into the profiles, else []."""
    from .profiles import PROFILES, DEFAULT_NAMES, slug
    if a.no_profile or a.near in DEFAULT_NAMES or a.far in DEFAULT_NAMES:
        return []
    root = pathlib.Path(a.profiles) if a.profiles else PROFILES
    import os
    return [(n, os.path.relpath(root / f"{slug(n)}.html", report_dir)) for n in dict.fromkeys([a.near, a.far])]


def save_profiles(a, video, report_dir, points, quality, report, critique=None, after3d=None):
    """Add this recording to both players' profiles (replacing it if the same video was analysed before) and rebuild the pages."""
    from .profiles import PROFILES, record_match
    from .profile_html import build_pages
    if not profile_links(a, report_dir):
        return
    root = pathlib.Path(a.profiles) if a.profiles else PROFILES
    ms = json.loads((report_dir / "match_stats.json").read_text())
    posture = json.loads((report_dir / "posture.json").read_text()) if (report_dir / "posture.json").exists() else None
    names = {"near": a.near, "far": a.far}
    portrait = {n: report_dir / f"contact_{i + 1}_2.jpg" for i, n in enumerate(dict.fromkeys([a.near, a.far]))
                if (report_dir / f"contact_{i + 1}_2.jpg").exists()}
    rec = record_match(root, video, names, points, ms, quality, report=report, posture=posture, portrait=portrait, recorded=a.recorded,
                       critique=critique, position3d={p_["name"]: p_["m"] for p_ in (after3d or {}).get("players", [])})
    pages = build_pages(root)
    print(f"profiles: recording {rec['id']} ({rec['recorded'][:16]}) saved; " + ", ".join(f"{n} -> {pages[n]}" for n in rec["players"] if n in pages))


def cmd_profiles(a):
    """Rebuild every profile page from the match records; list players; optionally forget one recording."""
    from .profiles import PROFILES, load_records, player_names, aggregate
    from .profile_html import build_pages
    root = pathlib.Path(a.profiles) if a.profiles else PROFILES
    if a.remove:
        f = root / "matches" / f"{a.remove}.json"
        if not f.exists():
            raise SystemExit(f"no recording {a.remove} in {root / 'matches'}")
        f.unlink(); print("removed", a.remove)
    recs = load_records(root)
    build_pages(root)
    for key, nm in player_names(recs).items():
        ag = aggregate(key, recs)
        print(f"{nm}: {ag['n']} recordings, won {ag['tot']['won']} lost {ag['tot']['lost']}, games {ag['tot']['games'][0]}-{ag['tot']['games'][1]} -> {root / (key + '.html')}")
    for r in recs:
        print(f"  {r['id']}  {r['recorded'][:16]}  {' v '.join(r['players'])}  winner {r.get('winner')}  {r.get('report')}")
    print("index:", root / "index.html")
    if a.open and sys.platform == "darwin":
        subprocess.run(["open", str(root / "index.html")])


def cmd_analyse(a):
    from .tablefind import find_table
    from .table import Table
    from .run import analyse
    from .report_html import make_html
    video = pathlib.Path(a.video)
    if not video.exists():
        raise SystemExit(f"no such file: {video}")
    out = pathlib.Path(a.out or ROOT / "out") / video.stem
    out.mkdir(parents=True, exist_ok=True)
    cal = None
    if a.table == "auto":
        print("finding the table ...", flush=True)
        corners, info = find_table(video)                          # by colour (blue, green), else by its surface and white edge line
        if corners is None:
            raise SystemExit(f"could not find the table automatically ({info.get('reason')}).\n"
                             f"Click the corners instead:  tt-scout calibrate {video}   then   tt-scout analyse {video} --table table.json")
        frame = info.pop("frame_image")
        table_path = out / "table.json"
        Table.save(table_path, corners, "side", video=str(video), **{k: v for k, v in info.items()})
        cv2.polylines(frame, [corners.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        cv2.imwrite(str(out / "table_check.png"), frame)
        cal = info
        print(f"  {info['colour']} table found in {info['n_frames']}/{info['n_tried']} frames -> {out / 'table_check.png'}", flush=True)
    else:
        table_path = pathlib.Path(a.table)
    print("tracking the ball (this is the slow part) ...", flush=True)
    cfg = Config(ball_colour=a.ball_colour, near_name=a.near, far_name=a.far)
    r = analyse(str(video), str(table_path), cfg, max_frames=a.max_frames, out_root=a.out, logic=a.logic)
    q = r["quality"]
    from .players import load_obs
    track = np.genfromtxt(out / "track.csv", delimiter=",", skip_header=1)
    overlay = None if a.plain_clips else dict(table=Table.load(table_path), track=track, events=r["events"], obs=load_obs(out / "players.csv"), fps=r["fps"])
    report, n_clips = make_html(out, f"{a.near} v {a.far} - {video.name}", r["rallies"], q, video=video, clips=not a.no_clips, calibration=cal, overlay=overlay,
                                web_fonts=a.web_fonts, comic=not a.no_comic, scoreboard=not a.no_scoreboard, tips=not a.no_tips, show_table=a.show_table,
                                profiles=profile_links(a, out))
    save_profiles(a, video, out, r["rallies"], q, report)
    print(f"{r['n_frames']} frames in {r['seconds']:.0f} s; {len(r['rallies'])} points; {n_clips} clips")
    print(f"confidence: {q['level'].upper()} ({q['score']}/100). {q['advice']}")
    for reason in q["reasons"]:
        print("  -", reason)
    print("report:", report)
    if not a.no_open and sys.platform == "darwin":
        subprocess.run(["open", str(report)])


def cmd_report(a):
    """Rebuild points, numbers and report.html from an analysis already on disk (events.csv, track.csv, players.csv): no tracking."""
    import csv
    from .points_v1 import segment_points_v1
    from .players import load_obs, name_points, auto_reference
    from .stats import write_stats
    from .report import make_report
    from .quality import assess
    from .report_html import make_html
    src = pathlib.Path(a.run_dir)
    dst = pathlib.Path(a.out or ROOT / "out_product") / src.name
    dst.mkdir(parents=True, exist_ok=True)
    events = []
    for r in csv.DictReader(open(src / "events.csv")):
        events.append(dict(frame=int(float(r["frame"])), t=float(r["t"]), kind=r["kind"], side=r["side"], x_px=float(r["x_px"]), y_px=float(r["y_px"]),
                           x_m=float(r["x_m"]), y_m=float(r["y_m"]), strength=float(r["strength"]), track=int(float(r["track"]))))
    track = np.genfromtxt(src / "track.csv", delimiter=",", skip_header=1)
    pts = segment_points_v1(events, None, track[:, :5])
    obs = load_obs(src / "players.csv"); names = {"near": a.near, "far": a.far}
    name_points(pts, obs, auto_reference(obs, names), names)
    (dst / "rallies.json").write_text(json.dumps(pts, indent=1))
    write_stats(dst, pts)
    make_report(pts, events, dst / "report.png", title=f"{a.near} v {a.far}")
    fps = float(np.median(1.0 / np.diff(track[:200, 1]))) if len(track) > 2 else None
    q = assess(pts, track, fps)
    q["reasons"] = q.get("reasons", []) + (a.note or [])     # e.g. the result of a hand check, shown under the verdict
    (dst / "quality.json").write_text(json.dumps(q, indent=1))
    video = pathlib.Path(a.video) if a.video else None
    overlay = None
    if video and not a.plain_clips:
        from .table import Table
        tpath = next((t for t in ([pathlib.Path(a.table)] if a.table else []) + [src / "table.json", ROOT / "data" / f"{src.name}_table.json"] if t.exists()), None)
        if tpath is None:
            print("no table.json found for the overlay (give --table): cutting plain clips")
        else:
            overlay = dict(table=Table.load(tpath), track=track, events=events, obs=obs, fps=fps or 60.0)
    posture = stance = None
    assigned = assigned_raw = None
    if a.pose:                                                      # skeletons from tools/pose: drawn on the clips, measured at hits
        from . import pose as pose_mod
        from .heads import standing_zone
        from .table import Table
        tb = overlay["table"] if overlay else Table.load(a.table or src / "table.json")
        raw_pose = pose_mod.load(a.pose)
        assigned = pose_mod.assign(raw_pose, tb, standing_zone(tb, 1.0))
        assigned_raw = {f: dict(v) for f, v in assigned.items()}         # as Vision found them, for the 3D crop boxes (body3d.requests);
        assigned, legs = pose_mod.clean_legs(assigned, tb)                # hidden or swapped legs are left out, not guessed (clean_legs
        print("legs: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in legs.items()))   # swaps in new arrays, so the copy keeps the old)
        if overlay:
            overlay["pose"] = assigned
            if getattr(a, "blur_others", False):                         # everyone who is not one of the two players, blurred
                cap_ = cv2.VideoCapture(str(video)) if video else None
                vw, vh = (int(cap_.get(3)), int(cap_.get(4))) if cap_ is not None else (1920, 1080)
                if cap_ is not None:
                    cap_.release()
                balls = [None if np.isnan(r_[2]) else (float(r_[2]), float(r_[3])) for r_ in track] if len(track) else None
                n1, n2 = tb.to_px([[NET_X, 0.0], [NET_X, TW]])           # the net stays sharp: the ball crosses it
                net_box = (min(n1[0], n2[0]) - 16, min(n1[1], n2[1]) - 70, max(n1[0], n2[0]) + 16, max(n1[1], n2[1]) + 12)
                overlay["hide"] = pose_mod.others(raw_pose, assigned, fps or 60.0, zone=pose_mod.back_zone(tb, vw, vh), balls=balls,
                                                  fixed_keep=[net_box], hold_s=0.3)   # the zone holds the back; people elsewhere, briefly
                print(f"blurring the back of the hall and {len(overlay['hide'])} frames' other people", flush=True)
    # every shot measured: 3D speed off the racket, topspin, height over the net, timing, posture (technique.py, flight.py)
    shots = []
    strokes = cam = None
    if video and video.exists():
        from .camera import Camera
        from .technique import measure as measure_shots, gate_knees
        from .flight import Speeds
        from .table import Table
        tb_ = overlay["table"] if overlay else (Table.load(a.table) if a.table else Table.load(src / "table.json"))
        cap_ = cv2.VideoCapture(str(video)); vw_, vh_ = int(cap_.get(3)), int(cap_.get(4)); cap_.release()
        cam = Camera.from_table_pnp(tb_, vw_, vh_)
        shots, fits = measure_shots(pts, events, track, fps or 60.0, cam, assigned)
        (dst / "camera.json").write_text(json.dumps(dict(cam.summary(vw_), corner_rms_px=round(cam.corner_rms, 2)), indent=1))
        write_stats(dst, pts, shots=shots)                               # speeds in stats.json: the 3D-fitted ones
        print(f"shots: {len(shots)}, {sum(s['speed'] is not None for s in shots)} with a 3D speed; camera {cam.summary(vw_)}")
        if overlay:
            overlay["speeds"] = Speeds(fits); overlay["shots"] = shots
        if shots and a.pose:                                             # the bodies in 3D through every stroke (tools/pose3d, cached)
            from . import body3d
            lines, meta = body3d.requests(shots, assigned_raw, fps or 60.0)   # boxes round the skeletons as found, legs and all
            cache = src / "pose3d.jsonl"; key = src / "pose3d_requests.txt"
            if cache.exists() and key.exists() and key.read_text().split() == "\n".join(sorted(lines, key=lambda l: int(l.split()[0]))).split():
                raw = [json.loads(l) for l in cache.read_text().splitlines() if l.strip()]
                print(f"bodies in 3D: {len(raw)} from the cache")
            else:
                print(f"bodies in 3D: {len(lines)} frames to analyse ...", flush=True)
                raw = body3d.run(video, lines, src, cam.f)
            strokes = list(body3d.strokes_from(raw, meta, cam).values())
            n3 = body3d.attach(shots, strokes, cam)                       # knee3d, lean3d, turn3d, side on each shot with a body
            print(f"bodies in 3D: {len(strokes)} strokes placed, {n3} shots with a body")
        gate_knees(shots, strokes, cam)                                  # the ONE knee bend at contact (pose.py): rally forehands seen in profile
        # written only now, with the 3D posture and the gate on it: the critique and the profiles read this file
        (dst / "shots.json").write_text(json.dumps(shots, indent=0, default=float))
        ok_ = [s for s in shots if s["knee"] is not None]                # only the contacts that passed the gate are counted here
        print(f"knee bend at contact: measurable {len(ok_)} ({sum(s.get('profile_src') == '3d' for s in ok_)} checked against the 3D body, "
              f"{sum(s.get('profile_src') == '2d' for s in ok_)} by the picture alone); refused {sum(not s['serve'] for s in shots) - len(ok_)} of "
              f"{sum(not s['serve'] for s in shots)} rally shots")
    if a.pose:                                                          # the hits and the posture, each hit at its shot's hitter end
        hits = pose_mod.at_hits(events, assigned, fps or 60.0, shots=shots if shots else None, points=pts)
        # only strokes inside a point: between points players bend to pick the ball up, and the tracker still logs "hits"
        hits = [h for h in hits if any(p["start_t"] <= h["t"] <= p["end_t"] for p in pts)]
        contacts = pose_mod.knee_contacts(shots, pts, names)             # the gated contacts: what every knee number is read from
        posture = pose_mod.summary(hits, pts, names, shots=shots if shots else None)
        stance = {}
        for nm in dict.fromkeys([names["near"], names["far"]]):
            mine = [h for h in hits if pose_mod.hitter_name(h, pts, names) == nm]
            side = max(("near", "far"), key=lambda sd: sum(h["side"] == sd for h in mine)) if mine else "near"
            stance[nm] = pose_mod.typical_stance([h for h in mine if h["side"] == side], assigned, fps or 60.0, side)
        (dst / "posture.json").write_text(json.dumps(dict(summary=posture, hits=hits, contacts=contacts, stance=stance), indent=1))
        for nm, q_ in posture.items():
            print(f"knee bend at contact, {nm}: {pose_mod.knee_text(q_['knee'], why=q_.get('knee_why'))}")
        if video and video.exists():                                     # real frames at the gated contacts, deepest knee bend to straightest
            from .hero import contact_frames
            stance = dict(figures=stance, frames=contact_frames(video, contacts, assigned, fps or 60.0, pts, names, dst))
    a3 = None
    if shots:                                                            # after the match, in 3D: where to stand, when to strike, where to aim
        from . import analysis3d
        a3, _, _ = analysis3d.build(shots, [a.near, a.far], dst, strokes=strokes, themes=("dark",))   # the page is black
        print("after the match, in 3D: " + ", ".join(f"{p_['name']} {p_['n']} shots" for p_ in a3["players"]))
    from .critique import critiques_by_point, critiques_after
    crit = critiques_by_point(shots, pts)                                # each player's criticisms before every point, from the shots so far
    ids = [p["id"] for p in pts]                                        # ... and after it (over its replay): the next point's "before"
    crit_after = {pid: crit[ids[i + 1]] for i, pid in enumerate(ids[:-1])}
    if ids:
        crit_after[ids[-1]] = critiques_after(shots, pts)
    crit_final = critiques_after(shots, pts)                             # for the page and the profiles
    if overlay:
        overlay["critiques"] = crit; overlay["critiques_after"] = crit_after
    if getattr(a, "full_match", False) and video and overlay and pts:   # the whole recording as one video, analysis up throughout
        from .annotate import render_match_clip, overlay_context
        by_frame = {}
        for o in obs:
            by_frame.setdefault(o["frame"], []).append(o)
        print("rendering the whole match ...", flush=True)
        from .match_stats import moments as moments_of
        render_match_clip(video, fps or 60.0, overlay["table"], track, events, by_frame, pts, dst / "full_match.mp4",
                          context=overlay_context(video, overlay["table"]), pose=overlay.get("pose"), critiques=crit,
                          critiques_after=crit_after, speeds=overlay.get("speeds"), moments=moments_of(pts, shots), hide=overlay.get("hide"))
    hero = None
    if video and video.exists() and len(track) and pts:                 # the opening picture: the longest rally, tracking inked on
        from . import hero as hero_mod
        from .table import Table
        tb = overlay["table"] if overlay else (Table.load(a.table) if a.table else None)
        if tb is not None:
            fps_ = fps or 60.0
            pick = hero_mod.pick(pts, track, assigned if a.pose else {}, fps_)
            if pick is None:                                            # no skeletons: the middle of the longest rally with the ball seen
                p0 = max(pts, key=lambda p: p.get("n_crossings") or 0)
                fr = [f for f in range(int(p0["start_t"] * fps_), int(p0["end_t"] * fps_)) if f < len(track) and not np.isnan(track[f, 2])]
                pick = (p0, fr[len(fr) // 2]) if fr else None
            if pick is not None:
                if hero_mod.render(video, track, assigned if a.pose else {}, tb, fps_, pick[1], dst / "hero.jpg"):
                    im = cv2.imread(str(dst / "hero.jpg"))
                    hero = dict(src="hero.jpg", w=im.shape[1], h=im.shape[0], point=pick[0]["id"])
    if overlay and getattr(a, "only_clips", None):                      # just these points' clips (e.g. one clip for a post)
        overlay["only"] = {int(x) for x in a.only_clips.split(",")}
    report, n_clips = make_html(dst, f"{a.near} v {a.far} - {src.name}", pts, q, video=video, clips=bool(video) and not a.no_clips, overlay=overlay,
                                web_fonts=a.web_fonts, reuse_clips=a.reuse_clips, comic=not a.no_comic, scoreboard=not a.no_scoreboard, tips=not a.no_tips, show_table=a.show_table,
                                posture=posture, hero=hero, stance=stance, profiles=profile_links(a, dst), critique=crit_final, after3d=a3)
    if video:
        save_profiles(a, video, dst, pts, q, report, critique=crit_final, after3d=a3)
    print(f"{len(pts)} points, {n_clips} clips, confidence {q['level'].upper()} ({q['score']}/100) -> {report}")
    if not a.no_open and sys.platform == "darwin":
        subprocess.run(["open", str(report)])


def cmd_serve(a):
    from .webapp import serve
    serve(a.port, open_browser=not a.no_open)


def cmd_calibrate(a):
    subprocess.run([sys.executable, str(ROOT / "calibrate_table.py"), a.video, "--view", "side"] + (["--frame", str(a.frame)] if a.frame else []), check=True)


def cmd_droptest(a):
    """Drops from a measured height, filmed like a match, each measured by the match speeds' own fit (tt_scout/droptest.py)."""
    from .camera import Camera
    from .droptest import measure, lines, figure
    from .events import detect_events
    from .filming import camera_position
    from .run import track_video
    from .table import Table
    from .tablefind import find_table
    src = pathlib.Path(a.video).expanduser()
    if not src.exists():
        raise SystemExit(f"no such file: {src}")
    out = pathlib.Path(a.out) / src.stem
    out.mkdir(parents=True, exist_ok=True)
    video = src
    if src.suffix.lower() != ".mp4":                                   # an iPhone .MOV (HDR) is converted as a match recording is
        video = out / f"{src.stem}.mp4"
        if not video.exists():
            print("converting the recording ...", flush=True)
            subprocess.run([sys.executable, str(ROOT / "analysis" / "convert_iphone.py"), str(src), str(video), "--max-height", "1080"], check=True)
    if a.table == "auto":
        print("finding the table ...", flush=True)
        corners, info = find_table(video)
        if corners is None:
            raise SystemExit(f"could not find the table automatically ({info.get('reason')}): click it with  tt-scout calibrate {video}  "
                             f"and pass the table.json with --table")
        info.pop("frame_image", None)
        table_path = out / "table.json"
        Table.save(table_path, corners, "side", video=str(video), **info)
    else:
        table_path = pathlib.Path(a.table)
    table = Table.load(table_path)
    cap = cv2.VideoCapture(str(video)); w, h = int(cap.get(3)), int(cap.get(4)); cap.release()
    pos = camera_position(table, w, h)
    print(f"camera: {pos['dist_m']:.1f} m from the net, {pos['z']:.2f} m above the table", flush=True)
    print("tracking the ball ...", flush=True)
    cfg = Config()
    rows, _, fps, _ = track_video(video, table, cfg)
    track = np.array(rows, float)
    events = detect_events(track[:, :5], table, cfg, fps)
    res = measure(track, events, fps, Camera.from_table_pnp(table, w, h), a.height)
    res.update(video=str(src), fps=round(fps, 3), camera=pos)
    (out / "droptest.json").write_text(json.dumps(res, indent=1))
    print("\n".join(lines(res)))
    if res["rows"]:
        figure(res, out / "droptest.png")
        print("figure:", out / "droptest.png")


def main():
    ap = argparse.ArgumentParser(prog="tt-scout", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    an = sub.add_parser("analyse", help="video -> report.html")
    an.add_argument("video"); an.add_argument("--near", default="Left player", help="player at the LEFT end of the picture at the start")
    an.add_argument("--far", default="Right player"); an.add_argument("--ball-colour", default="any", choices=["any", "white", "orange"])
    an.add_argument("--table", default="auto", help="'auto' (default) or a table.json from tt-scout calibrate")
    an.add_argument("--logic", default="v1", choices=["v1", "v0"]); an.add_argument("--no-clips", action="store_true")
    an.add_argument("--no-open", action="store_true"); an.add_argument("--out"); an.add_argument("--max-frames", type=int)
    an.add_argument("--plain-clips", action="store_true", help="cut the video without drawing the tracking on it")
    an.add_argument("--web-fonts", action="store_true", help="load Fraunces + Manrope from Google Fonts (a network request; the default report is offline)")
    an.add_argument("--no-comic", action="store_true", help="clips without the comic-book scoring moment")
    an.add_argument("--no-scoreboard", action="store_true", help="clips without the running scoreboard (it is tt_scout's own count and can differ from the umpire's)")
    an.add_argument("--no-tips", action="store_true", help="no scout tips: neither the note before the serve in the clips nor the list in the report")
    an.add_argument("--show-table", action="store_true", help="draw the calibrated table outline and net line on the clips (for checking a calibration)")
    for p_ in (an,):
        p_.add_argument("--no-profile", action="store_true", help="do not add this recording to the players' profiles")
        p_.add_argument("--profiles", help="profiles folder (default: profiles/ in the repo)")
        p_.add_argument("--recorded", help="when the recording was made (ISO), if the video file does not say")
    an.set_defaults(fn=cmd_analyse)
    rp = sub.add_parser("report", help="rebuild report.html from an analysis folder already on disk (no tracking)")
    rp.add_argument("run_dir", help="e.g. out/game_1"); rp.add_argument("--video", help="the recording, to cut one clip per point")
    rp.add_argument("--near", default="Left player"); rp.add_argument("--far", default="Right player")
    rp.add_argument("--no-clips", action="store_true"); rp.add_argument("--no-open", action="store_true"); rp.add_argument("--out")
    rp.add_argument("--table", help="table.json for the overlay (default: <run_dir>/table.json or data/<name>_table.json)")
    rp.add_argument("--plain-clips", action="store_true", help="cut the video without drawing the tracking on it")
    rp.add_argument("--reuse-clips", action="store_true", help="keep the clips already in the folder (restyle the page in seconds)")
    rp.add_argument("--note", action="append", help="a line shown under the verdict, e.g. what a hand check found (repeatable)")
    rp.add_argument("--pose", help="skeletons from tools/pose (CSV): drawn on the clips and measured at each hit (a Posture section)")
    rp.add_argument("--blur-others", action="store_true", help="blur everyone who is not one of the two players (needs --pose): for sharing a recording")
    rp.add_argument("--only-clips", help="render only these points' clips, e.g. 19,72 (use a separate --out: the page lists only those)")
    rp.add_argument("--full-match", action="store_true", help="also render the whole recording as one video, both players' analysis on screen throughout")
    rp.add_argument("--web-fonts", action="store_true", help="load Fraunces + Manrope from Google Fonts (a network request; the default report is offline)")
    rp.add_argument("--no-comic", action="store_true", help="clips without the comic-book scoring moment")
    rp.add_argument("--no-scoreboard", action="store_true", help="clips without the running scoreboard (it is tt_scout's own count and can differ from the umpire's)")
    rp.add_argument("--no-tips", action="store_true", help="no scout tips: neither the note before the serve in the clips nor the list in the report")
    rp.add_argument("--show-table", action="store_true", help="draw the calibrated table outline and net line on the clips (for checking a calibration)")
    rp.add_argument("--no-profile", action="store_true", help="do not add this recording to the players' profiles")
    rp.add_argument("--profiles", help="profiles folder (default: profiles/ in the repo)")
    rp.add_argument("--recorded", help="when the recording was made (ISO), if the video file does not say")
    rp.set_defaults(fn=cmd_report)
    pf = sub.add_parser("profiles", help="rebuild the player profile pages from the match records, list players")
    pf.add_argument("--profiles", help="profiles folder (default: profiles/ in the repo)")
    pf.add_argument("--remove", help="forget one recording (its id, as listed)")
    pf.add_argument("--open", action="store_true", help="open the index page")
    pf.set_defaults(fn=cmd_profiles)
    sv = sub.add_parser("serve", help="the upload page: drop a video in the browser, get the report (this Mac only)")
    sv.add_argument("--port", type=int, default=8770); sv.add_argument("--no-open", action="store_true"); sv.set_defaults(fn=cmd_serve)
    ca = sub.add_parser("calibrate", help="click the table corners by hand"); ca.add_argument("video"); ca.add_argument("--frame", type=int)
    ca.set_defaults(fn=cmd_calibrate)
    dt = sub.add_parser("droptest", help="the speed check on a real ball: drops from a measured height against the speed physics gives")
    dt.add_argument("video"); dt.add_argument("--height", type=float, default=1.0, help="metres from the table to the bottom of the ball when let go")
    dt.add_argument("--table", default="auto", help="'auto' (default) or a table.json from tt-scout calibrate")
    dt.add_argument("--out", default=str(ROOT / "out_droptest")); dt.set_defaults(fn=cmd_droptest)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
