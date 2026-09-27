"""Click the 4 table corners once per camera position. Writes table.json.

Usage: python calibrate_table.py match.mp4 [--frame 300] [--view side|behind]

Order: corner 1 and 2 are the two corners of ONE end line (that end becomes "near", x = 0). Start with
the corner you would walk around the table from, then keep going round: 1 -> 2 -> 3 -> 4.
Pick a frame where nobody is leaning over the table. Press r to restart, q to quit without saving.
"""
import argparse, cv2
from tt_scout.table import Table
from tt_scout.config import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("--frame", type=int, default=0)
ap.add_argument("--view", default="side", choices=["side", "behind"])
ap.add_argument("--out", default=str(ROOT / "table.json"))
ap.add_argument("--near", help="name of the player at the end you click first (corners 1-2); with --far, you then click each shirt")
ap.add_argument("--far")
a = ap.parse_args()
want_players = bool(a.near and a.far)

cap = cv2.VideoCapture(int(a.video) if a.video.isdigit() else a.video)   # a camera index works too
if not a.video.isdigit():
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
else:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    for _ in range(15):
        cap.read()                                                          # let the exposure settle
ok, frame = cap.read(); cap.release()
if not ok:
    raise SystemExit("could not read that frame")
pts, shirts = [], []
WIN = "click 4 corners: 1,2 = one end line, then round the table   (r reset, q quit)"
hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


def on_click(ev, x, y, *_):
    if ev != cv2.EVENT_LBUTTONDOWN:
        return
    if len(pts) < 4:
        pts.append((x, y)); print(f"corner {len(pts)}: {x},{y}")
    elif want_players and len(shirts) < 2:
        patch = hsv_img[max(0, y - 10):y + 11, max(0, x - 10):x + 11].reshape(-1, 3)
        med = [float(v) for v in __import__("numpy").median(patch, axis=0)]
        shirts.append(((x, y), med)); print(f"{[a.near, a.far][len(shirts) - 1]}'s shirt: {x},{y}  HSV {[int(v) for v in med]}")


cv2.namedWindow(WIN, cv2.WINDOW_NORMAL); cv2.setMouseCallback(WIN, on_click)
while True:
    view = frame.copy()
    for i, p in enumerate(pts):
        cv2.circle(view, p, 6, (0, 0, 255), -1); cv2.putText(view, str(i + 1), (p[0] + 8, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    if len(pts) == 4:
        cv2.polylines(view, [__import__("numpy").int32(pts).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        for (sx, sy), _ in shirts:
            cv2.circle(view, (sx, sy), 12, (255, 0, 255), 2)
        if want_players and len(shirts) < 2:
            msg = f"now click {a.near}'s shirt ({a.near} is at the end of corners 1-2)" if not shirts else f"now click {a.far}'s shirt"
        else:
            msg = "press s to save"
        cv2.putText(view, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow(WIN, view)
    k = cv2.waitKey(30) & 0xFF
    if k == ord("q"):
        break
    if k == ord("r"):
        pts.clear(); shirts.clear()
    if k == ord("s") and len(pts) == 4 and (not want_players or len(shirts) == 2):
        extra = {}
        if want_players:
            extra["players"] = {"near": {"name": a.near, "hsv": shirts[0][1]}, "far": {"name": a.far, "hsv": shirts[1][1]}}
        Table.save(a.out, pts, a.view, video=a.video, frame=a.frame, **extra)
        print("wrote", a.out); break
cv2.destroyAllWindows()
