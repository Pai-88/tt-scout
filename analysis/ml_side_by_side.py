"""Side-by-side video: the classical ball track (left) and the learned one (right) over the same stretch of a clip, with the
OpenTTGames label drawn as a small white cross when one exists.

    python analysis/ml_side_by_side.py test_4 --start 250.8 --end 258.4 --out out_ml/test_4/side_by_side.mp4 [--slow 4]
"""
import argparse, json, pathlib, shutil, subprocess
import numpy as np, cv2

ROOT = pathlib.Path(__file__).resolve().parent.parent


def draw(img, track, f, s, colour, label):
    seg = track[max(0, f - 36):f + 1]
    seg = seg[~np.isnan(seg[:, 2])]
    for k in range(1, len(seg)):
        if seg[k, 0] - seg[k - 1, 0] > 6:
            continue
        a = k / len(seg)
        p0, p1 = (seg[k - 1, 2:4] * s).astype(int), (seg[k, 2:4] * s).astype(int)
        cv2.line(img, tuple(p0), tuple(p1), (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(img, tuple(p0), tuple(p1), tuple(int(c * (0.4 + 0.6 * a)) for c in colour), 2, cv2.LINE_AA)
    if f < len(track) and not np.isnan(track[f, 2]):
        c = (int(track[f, 2] * s), int(track[f, 3] * s))
        cv2.circle(img, c, 9, (0, 0, 0), 4, cv2.LINE_AA); cv2.circle(img, c, 9, colour, 2, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(img, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("clip"); ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True); ap.add_argument("--out", required=True); ap.add_argument("--slow", type=int, default=4)
    a = ap.parse_args()
    video = ROOT / "data" / f"{a.clip}.mp4"
    classical = np.genfromtxt(ROOT / "out" / a.clip / "track.csv", delimiter=",", skip_header=1)
    learned = np.genfromtxt(ROOT / "out_ml" / a.clip / "track.csv", delimiter=",", skip_header=1)
    lab_path = ROOT / "data" / f"{a.clip}_markup" / "ball_markup.json"
    labels = json.loads(lab_path.read_text()) if lab_path.exists() else {}
    cap = cv2.VideoCapture(str(video)); fps = cap.get(5)
    f0, f1 = int(a.start * fps), int(a.end * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    w, h = 960, 540; s = w / cap.get(3)
    out_fps = fps / a.slow
    step = max(1, int(round(out_fps / 30)))                        # keep the output near 30 fps
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{2 * w}x{h}", "-r", str(out_fps / step), "-i", "-",
                             "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
    for f in range(f0, f1):
        ok, frame = cap.read()
        if not ok:
            break
        if (f - f0) % step:
            continue
        small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        left, right = small.copy(), small.copy()
        draw(left, classical, f, s, (0, 200, 255), f"CLASSICAL (motion + colour rules)   t={f / fps:6.2f}s   {a.slow}x slower")
        draw(right, learned, f, s, (80, 255, 80), "LEARNED (BallNet, trained on other matches)")
        lab = labels.get(str(f))
        if lab and lab["x"] >= 0:
            for img in (left, right):
                x, y = int(lab["x"] * s), int(lab["y"] * s)
                cv2.drawMarker(img, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 7, 1, cv2.LINE_AA)
        proc.stdin.write(np.hstack([left, right]).tobytes())
    proc.stdin.close(); proc.wait()
    print(a.out)


if __name__ == "__main__":
    main()
