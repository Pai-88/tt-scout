"""Point-level ground truth for your own footage. One CSV per video: labels/<stem>.points.csv (see tt_scout/points_truth.py).

Usage: python label_points.py match.mp4

Keys (pause first; label at the exact frame):
  space   pause / play          , .   step one frame back / forward       j / k   jump 2 s back / forward
  1 / 2   half / normal speed   u     undo the last action                 q       save and quit
  n / f   SERVE by the NEAR / FAR player at this frame (near = the x=0 end, calibration corners 1-2) -> opens a point
  a / d   point OVER at this frame, won by NEAR (a) / FAR (d)             -> closes the open point
  o t b w m   ending of the point just closed: out / net / double bounce / winner / missed ball (default: unknown)

Rules: the serve frame is the racket-ball contact of the serve; the end frame is the event that decides the point
(second bounce, ball into the net, ball out, ...), not the dead ball afterwards. If the recording starts inside a
point, press a/d at its end without a serve: the row is kept with start_known=0.
"""
import csv, pathlib, sys
import cv2
from tt_scout.points_truth import FIELDS

ENDING_KEYS = {ord("o"): "out", ord("t"): "net", ord("b"): "double_bounce", ord("w"): "winner", ord("m"): "not_hitting_ball"}


class PointStore:
    def __init__(self, video, fps, labels_dir="labels"):
        self.fps = fps
        self.path = pathlib.Path(labels_dir) / (pathlib.Path(video).stem + ".points.csv")
        self.path.parent.mkdir(exist_ok=True)
        self.rows = [dict(r) for r in csv.DictReader(open(self.path))] if self.path.exists() else []
        self.open = None
        self.history = []                                   # ("open", None) | ("close", None) | ("ending", previous)

    def serve(self, frame, server):
        self.open = dict(start_frame=frame, start_t=f"{frame / self.fps:.4f}", server=server, start_known=1)
        self.history.append(("open", None))

    def point_over(self, frame, winner):
        row = self.open or dict(start_frame=frame, start_t=f"{frame / self.fps:.4f}", server="", start_known=0)
        row.update(end_frame=frame, end_t=f"{frame / self.fps:.4f}", winner=winner, ending="unknown",
                   id=len(self.rows) + 1, source="label_points.py")
        self.rows.append({k: row.get(k, "") for k in FIELDS}); self.open = None
        self.history.append(("close", None))
        return self.rows[-1]

    def set_ending(self, kind):
        if self.rows and self.open is None:
            self.history.append(("ending", self.rows[-1]["ending"])); self.rows[-1]["ending"] = kind
            return True
        return False

    def undo(self):
        if not self.history:
            return None
        what, prev = self.history.pop()
        if what == "open":
            self.open = None
        elif what == "close":
            row = self.rows.pop()
            self.open = {k: row[k] for k in ("start_frame", "start_t", "server", "start_known")} if int(row["start_known"] or 0) else None
        else:
            self.rows[-1]["ending"] = prev
        return what

    def save(self):
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(self.rows)
        return self.path


def main(video):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    store = PointStore(video, fps)
    state = dict(paused=False, speed=1.0, i=0, frame=None)
    WIN = "label points  (n/f serve, a/d point over, o/t/b/w/m ending, u undo, q save)"

    def seek(i):
        state["i"] = max(0, min(n - 1, i)); cap.set(cv2.CAP_PROP_POS_FRAMES, state["i"]); state["frame"] = None

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    while True:
        if not state["paused"] or state["frame"] is None:
            ok, fr = cap.read()
            if not ok:
                state["paused"] = True; seek(n - 1); continue
            state["frame"] = fr; state["i"] = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        i, t = state["i"], state["i"] / fps
        view = state["frame"].copy()
        cv2.putText(view, f"{t:8.3f}s  frame {i}  {'PAUSED' if state['paused'] else 'x' + str(state['speed'])}  points {len(store.rows)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if store.open:
            cv2.putText(view, f"OPEN: {store.open['server']} served at {store.open['start_t']}s -> a/d to end", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        for k, r in enumerate(store.rows[-3:]):
            cv2.putText(view, f"#{r['id']} {r['start_t']}-{r['end_t']}s serve {r['server'] or '?'} won {r['winner']} ({r['ending']})",
                        (10, 90 + 25 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
        cv2.imshow(WIN, view)
        key = cv2.waitKey(1 if state["paused"] else max(1, int(1000 / (fps * state["speed"])))) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            state["paused"] = not state["paused"]
        elif key == ord("u"):
            print("undo", store.undo())
        elif key == ord(","):
            seek(i - 1); state["paused"] = True
        elif key == ord("."):
            seek(i + 1); state["paused"] = True
        elif key == ord("j"):
            seek(i - int(2 * fps))
        elif key == ord("k"):
            seek(i + int(2 * fps))
        elif key == ord("1"):
            state["speed"] = 0.5
        elif key == ord("2"):
            state["speed"] = 1.0
        elif key in (ord("n"), ord("f")):
            store.serve(i, "near" if key == ord("n") else "far"); print(f"{t:.3f}s serve by {store.open['server']}")
        elif key in (ord("a"), ord("d")):
            r = store.point_over(i, "near" if key == ord("a") else "far"); print(f"{t:.3f}s point #{r['id']} won by {r['winner']}")
        elif key in ENDING_KEYS:
            if store.set_ending(ENDING_KEYS[key]):
                print(f"point #{store.rows[-1]['id']} ending = {ENDING_KEYS[key]}")
    cap.release(); cv2.destroyAllWindows()
    if store.open:
        print("note: an open point (serve without an end) was discarded")
    print("saved", store.save(), f"({len(store.rows)} points)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
