"""Zoomed crops around the four calibrated table corners for QA.  Usage: python scripts/corner_crops.py data/test_4 [--size 160 --zoom 4]
Writes data/<stem>_corners.png: one panel per corner, the calibrated corner drawn as a red cross, frame = the table.json frame."""
import argparse, json, pathlib
import cv2, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("base"); ap.add_argument("--size", type=int, default=160); ap.add_argument("--zoom", type=int, default=4)
a = ap.parse_args(); base = pathlib.Path(a.base)
t = json.load(open(f"{base}_table.json")); corners = np.array(t["corners_px"])
cap = cv2.VideoCapture(f"{base}.mp4"); cap.set(cv2.CAP_PROP_POS_FRAMES, int(t.get("frame", 0))); ok, fr = cap.read(); cap.release()
H, W = fr.shape[:2]; h = a.size // 2; panels = []
for i, (x, y) in enumerate(corners, start=1):
    x0, y0 = int(np.clip(x - h, 0, W - a.size)), int(np.clip(y - h, 0, H - a.size))
    crop = cv2.resize(fr[y0:y0 + a.size, x0:x0 + a.size], None, fx=a.zoom, fy=a.zoom, interpolation=cv2.INTER_CUBIC)
    cx, cy = int((x - x0) * a.zoom), int((y - y0) * a.zoom)
    cv2.line(crop, (cx - 25, cy), (cx + 25, cy), (0, 0, 255), 1); cv2.line(crop, (cx, cy - 25), (cx, cy + 25), (0, 0, 255), 1)
    cv2.putText(crop, f"{i} ({x:.0f},{y:.0f})", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    panels.append(crop)
out = np.hstack(panels); cv2.imwrite(f"{base}_corners.png", out); print(f"wrote {base}_corners.png {out.shape[1]}x{out.shape[0]}")
