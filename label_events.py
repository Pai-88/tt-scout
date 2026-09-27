"""Ground-truth labelling for one match video. One CSV per video in labels/.

Usage: python label_events.py match.mp4

Keys:
  space       pause / play                 , .    step one frame back / forward (while paused)
  click       (while paused) mark a BOUNCE at the ball's position in this frame
  h           mark a racket HIT at this time
  n / f       point over, won by NEAR / FAR player (near = the x=0 end from calibration)
  j / k       jump back / forward 2 s      1 / 2  half speed / normal      u undo      q save+quit

Tip: play at half speed, pause at the frame where the ball touches the table, click the ball.
Bounce frames are what the detector is scored on, so precision here matters more than speed.
"""
import csv, pathlib, sys
import cv2

KIND_KEYS = {ord("h"): ("hit", None), ord("n"): ("point", "near"), ord("f"): ("point", "far")}


class LabelStore:
    def __init__(self, video):
        self.path = pathlib.Path("labels") / (pathlib.Path(video).stem + ".csv")
        self.path.parent.mkdir(exist_ok=True)
        self.rows = [dict(r) for r in csv.DictReader(open(self.path))] if self.path.exists() else []

    def add(self, t, kind, side=None, x=None, y=None):
        self.rows.append(dict(t=f"{t:.4f}", kind=kind, side=side or "", x="" if x is None else x, y="" if y is None else y))

    def undo(self):
        return self.rows.pop() if self.rows else None

    def save(self):
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["t", "kind", "side", "x", "y"]); w.writeheader(); w.writerows(self.rows)
        return self.path


def main(video):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    store = LabelStore(video)
    state = dict(paused=False, speed=1.0, i=0, frame=None)
    WIN = "label events  (space pause, click=bounce, h hit, n/f point, u undo, q save)"

    def seek(i):
        state["i"] = max(0, min(n - 1, i)); cap.set(cv2.CAP_PROP_POS_FRAMES, state["i"]); state["frame"] = None

    def on_click(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and state["paused"]:
            t = state["i"] / fps; store.add(t, "bounce", None, x, y); print(f"{t:.3f}s bounce at {x},{y}")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL); cv2.setMouseCallback(WIN, on_click)
    while True:
        if not state["paused"] or state["frame"] is None:
            ok, fr = cap.read()
            if not ok:
                state["paused"] = True; seek(n - 1); continue
            state["frame"] = fr; state["i"] = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        t = state["i"] / fps
        view = state["frame"].copy()
        cv2.putText(view, f"{t:8.3f}s  frame {state['i']}  {'PAUSED' if state['paused'] else 'x'+str(state['speed'])}  labels {len(store.rows)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        for k, r in enumerate(store.rows[-3:]):
            cv2.putText(view, f"{r['t']}s {r['kind']} {r['side']}", (10, 60 + 25 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
        cv2.imshow(WIN, view)
        key = cv2.waitKey(1 if state["paused"] else max(1, int(1000 / (fps * state["speed"])))) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            state["paused"] = not state["paused"]
        elif key == ord("u"):
            store.undo()
        elif key == ord(","):
            seek(state["i"] - 1); state["paused"] = True
        elif key == ord("."):
            seek(state["i"] + 1); state["paused"] = True
        elif key == ord("j"):
            seek(state["i"] - int(2 * fps))
        elif key == ord("k"):
            seek(state["i"] + int(2 * fps))
        elif key == ord("1"):
            state["speed"] = 0.5
        elif key == ord("2"):
            state["speed"] = 1.0
        elif key in KIND_KEYS:
            kind, side = KIND_KEYS[key]; store.add(t, kind, side); print(f"{t:.3f}s {kind} {side or ''}")
    cap.release(); cv2.destroyAllWindows()
    print("saved", store.save(), f"({len(store.rows)} labels)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
