"""One new recording -> converted video, tracking, skeletons, the match report with clips, and both players' profiles updated.
This is the whole pipeline our first recordings went through, as one command:

    python scripts/new_match.py ~/Downloads/IMG_3150.MOV --near Sam --far Robin --table data/own/IMG_3143_table.json

--near is the player at the LEFT end of the picture. --table is one of:
    a table.json          the camera stood where it stood for that recording (check table_check.png / the report's first clip)
    auto                  blue or green table, found automatically
    grey:x1,y1,...,x4,y4  a grey table: rough corners (bottom-left, top-left, top-right, bottom-right) read off one frame
    clicks:x1,y1,...      corners clicked on the upload page, same order: snapped to the white edge lines when those are found
                          close by, else used as clicked
--name sets the report folder (default: <near>_v_<far>_<video name>); --note lines go under the verdict; --no-pose skips skeletons.
--work puts everything for the recording in one folder (data/, raw/, report/) instead of data/own, out_own, out_own_product:
the upload page (tt_scout/webapp.py) runs every match that way, so deleting the folder deletes the video and all it produced.
Lines starting "== " name the step that is starting (the upload page shows them); exit code 3 = the camera check refused it.
"""
import argparse, json, pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = sys.executable
SNAP_PX = 40            # a snapped corner further than this from the click is not trusted: the click is used


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT, **kw)


def stage(name):
    print(f"== {name}", flush=True)


def table_from_clicks(mp4, clicks, out, check):
    """Clicked corners -> table.json: the white edge lines fitted near them (analysis/fit_grey_table.py --seed, which only looks for
    the white line, so it works on blue, green and grey tables), unless a fitted corner lands more than SNAP_PX from its click."""
    import numpy as np
    from tt_scout.table import Table
    c = np.array(clicks, float).reshape(4, 2)
    fit = out.with_name(out.stem + "_fit.json")
    why = ""
    try:
        run([PY, "analysis/fit_grey_table.py", mp4, "--seed", ",".join(f"{v:.1f}" for v in c.ravel()), "--out", fit, "--check", check])
        q = np.array(json.loads(fit.read_text())["corners_px"], float)
        moved = float(np.hypot(*(q - c).T).max())
        if moved <= SNAP_PX:
            Table.save(out, q, "side", source=f"clicked on the upload page, snapped to the white edge lines (largest move {moved:.0f} px)")
            print(f"table: snapped to the edge lines, largest move {moved:.0f} px", flush=True)
            return
        why = f"the fitted edge lines were {moved:.0f} px from a click"
    except subprocess.CalledProcessError:
        why = "the white edge lines were not found near the clicks"
    Table.save(out, c, "side", source=f"clicked on the upload page, used as clicked ({why})")
    print(f"table: used as clicked ({why})", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("--near", required=True); ap.add_argument("--far", required=True)
    ap.add_argument("--table", default="auto"); ap.add_argument("--name"); ap.add_argument("--note", action="append", default=[])
    ap.add_argument("--data", default="data/own"); ap.add_argument("--no-pose", action="store_true"); ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--work", help="one folder for everything this recording produces (data/, raw/, report/)")
    ap.add_argument("--no-profile", action="store_true", help="do not add the match to the player profiles")
    ap.add_argument("--no-full-match", action="store_true", help="skip the whole-match video, the slowest step")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    from tt_scout.profiles import slug
    src = pathlib.Path(a.video).expanduser()
    stem = src.stem
    work = pathlib.Path(a.work).resolve() if a.work else None
    data = work / "data" if work else ROOT / a.data
    raw_root = work / "raw" if work else ROOT / "out_own"
    product = work / "report" if work else ROOT / "out_own_product"
    for d in (data, raw_root, product):
        d.mkdir(parents=True, exist_ok=True)
    mp4 = data / f"{stem}.mp4"
    stage("convert")
    if src.suffix.lower() != ".mp4" or src.resolve() != mp4.resolve():
        if not mp4.exists():                                            # 1. HDR -> SDR, keeping the recording date
            run([PY, "analysis/convert_iphone.py", src, mp4, "--max-height", "1080"])
    name = a.name or f"{slug(a.near)}_v_{slug(a.far)}_{slug(stem)}"
    stage("table")
    table = None                                                        # 2. the table
    if a.table.startswith("grey:"):
        table = data / f"{stem}_table.json"
        run([PY, "analysis/fit_grey_table.py", mp4, "--seed", a.table[5:], "--out", table, "--check", data / f"{stem}_corners.png"])
    elif a.table.startswith("clicks:"):
        table = data / f"{stem}_table.json"
        table_from_clicks(mp4, [float(v) for v in a.table[7:].split(",")], table, data / f"{stem}_corners.png")
    elif a.table != "auto":
        table = pathlib.Path(a.table)
    else:                                                               # found here, not inside tracking, so the camera check sees it too
        from tt_scout.tablefind import find_table
        from tt_scout.table import Table
        corners, info = find_table(mp4)
        if corners is None:
            print(f"REFUSED: could not find the table automatically ({info.get('reason')}); pass --table clicks:... instead", flush=True)
            sys.exit(3)
        info.pop("frame_image", None)
        table = data / f"{stem}_table.json"
        Table.save(table, corners, "side", video=str(mp4), **info)
        print(f"table: found by its {info.get('method')}", flush=True)
    if table:                                                           # 2b. where the phone stood: refused here, before the slow part
        import cv2
        from tt_scout.table import Table
        from tt_scout.filming import camera_position, camera_ok
        cap = cv2.VideoCapture(str(mp4)); w_, h_ = int(cap.get(3)), int(cap.get(4)); cap.release()
        pos = camera_position(Table.load(table), w_, h_)
        print(f"camera: {pos['off_deg']:.0f} deg off the net line, {pos['dist_m']:.1f} m from the net, {pos['z']:.2f} m above the table",
              flush=True)
        ok, why = camera_ok(pos)
        if not ok:
            print("REFUSED:", why, flush=True)
            sys.exit(3)
    stage("track")
    track_args = ["--table", table]                                     # 3. tracking (no clips yet)
    run([PY, "-m", "tt_scout.cli", "analyse", mp4, *track_args, "--near", a.near, "--far", a.far, "--out", raw_root, "--no-clips", "--no-open", "--no-profile"])
    raw = raw_root / stem
    stage("points")
    run([PY, "analysis/reevents.py", raw, "--table", table, "--out", raw_root / name,     # 4. events with the other-table filter
         "--set", "cross_front_s=0.5", "--near", a.near, "--far", a.far])
    pose_args = []
    if not a.no_pose:                                                   # 5. skeletons, only inside the points
        stage("pose")
        tool = ROOT / "tools" / "pose" / "pose"
        if not tool.exists():
            run(["xcrun", "swiftc", "-O", "tools/pose/pose.swift", "-o", tool])
        pts = json.loads((raw_root / name / "rallies.json").read_text())
        wins = []
        for p in sorted(pts, key=lambda p: p["start_t"]):
            w = [max(0.0, p["start_t"] - 3.6), p["end_t"] + 1.2]
            if wins and w[0] <= wins[-1][1]:
                wins[-1][1] = max(wins[-1][1], w[1])
            else:
                wins.append(w)
        wfile = raw_root / f"{name}_windows.txt"; wfile.write_text("".join(f"{x:.2f} {y:.2f}\n" for x, y in wins))
        csv = raw_root / f"{name}_pose.csv"
        run([tool, mp4, csv, "--windows", wfile])
        pose_args = ["--pose", csv]
    stage("report")
    notes = [x for n in a.note for x in ("--note", n)]                  # 6. the report with clips; the profiles are updated by it
    extra = (["--no-profile"] if a.no_profile else []) + ([] if a.no_full_match else ["--full-match"])
    run([PY, "-m", "tt_scout.cli", "report", raw_root / name, "--video", mp4, "--table", table, "--near", a.near, "--far", a.far,
         "--out", product, "--no-open", *extra, *pose_args, *notes])
    report = product / name / "report.html"
    print("report:", report, flush=True)
    if not a.no_profile:
        print("profiles:", ROOT / "profiles" / "index.html")
    if not a.no_open and sys.platform == "darwin":
        subprocess.run(["open", str(report)])


if __name__ == "__main__":
    main()
