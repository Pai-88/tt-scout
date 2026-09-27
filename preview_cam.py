"""Live preview of every camera the Mac can open, tiled in one window, so you can spot the iPhone
(Continuity Camera) and check the framing. q or Esc closes it. Prints the index to use with
calibrate_table.py / tt_scout.run."""
import cv2, numpy as np, sys
caps = []
for i in range(4):
    c = cv2.VideoCapture(i)
    ok, f = c.read()
    if ok:
        c.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); c.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        caps.append((i, c)); print(f"camera index {i}: {f.shape[1]}x{f.shape[0]}", flush=True)
    else:
        c.release()
if not caps:
    print("NO CAMERA: macOS has not allowed camera access for this app. System Settings > Privacy & Security > Camera, "
          "enable it for your terminal, then run again.", flush=True)
    sys.exit(1)
print("preview open; press q in the window to close", flush=True)
while True:
    tiles = []
    for i, c in caps:
        ok, f = c.read()
        if not ok:
            f = np.zeros((1080, 1920, 3), np.uint8)
        f = cv2.resize(f, (960, 540))
        cv2.putText(f, f"camera index {i}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        tiles.append(f)
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)]
    cv2.imshow("cameras  (q closes)", np.vstack(rows))
    k = cv2.waitKey(30) & 0xFF
    if k in (ord("q"), 27):
        break
for _, c in caps:
    c.release()
cv2.destroyAllWindows()
