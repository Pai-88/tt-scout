"""Run filming.camera_ok on every recording whose table is known and show what it decides, with the numbers it saw.
    python analysis/check_camera_ok.py
The side-on recordings all worked; IMG_3140 (filmed from behind a player) did not. A good rule accepts the first and refuses the second."""
import glob, pathlib, sys
import cv2
ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from tt_scout.table import Table
from tt_scout.filming import camera_position, camera_ok

for tj in sorted(glob.glob(str(ROOT / "data/own/*_table.json"))) + sorted(glob.glob(str(ROOT / "data/*_table.json"))):
    vid = tj.replace("_table.json", ".mp4")
    cap = cv2.VideoCapture(vid); w, h = int(cap.get(3)), int(cap.get(4)); cap.release()
    if not w:
        continue
    pos = camera_position(Table.load(tj), w, h)
    ok, why = camera_ok(pos)
    name = pathlib.Path(tj).name.replace("_table.json", "")
    worked = {"IMG_3140": False, "yt_bundesliga": None}.get(name, True)     # None: never checked either way
    mark = "   (outcome never checked: your call)" if worked is None else ("" if ok == worked else "   <-- not what happened with this video")
    print(f"{name:10s} off the net line {pos['off_deg']:5.1f} deg, {pos['dist_m']:4.1f} m away, {pos['z']:4.2f} m up -> "
          f"{'ACCEPT' if ok else 'REFUSE: ' + why}{mark}")
