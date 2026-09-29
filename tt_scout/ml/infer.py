"""Run a trained BallNet over a whole video: ball candidates for every frame, in the video's own pixels.

    python -m tt_scout.ml.infer data/test_4.mp4 --out out_ml/test_4/candidates.npz

Decoding runs in a thread while the GPU works, so a 5-minute 120 fps clip (36,000 frames) takes a few minutes. Frames 0 and 1
have no history and get no candidates.
"""
import argparse, pathlib, queue, threading, time
import numpy as np, cv2, torch
from .data import SIZE, HISTORY, normalise, to_src, ROOT
from .model import BallNet, peaks


def load(ckpt, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"]); cfg["width"] = tuple(cfg["width"])
    model = BallNet(**cfg); model.load_state_dict(ck["state"]); model.to(device).eval()
    return model, ck


def _reader(video, q, max_frames):
    cap = cv2.VideoCapture(str(video)); i = 0
    while max_frames is None or i < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        q.put(cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA)); i += 1
    cap.release(); q.put(None)


@torch.no_grad()
def candidates(video, ckpt=ROOT / "models" / "ballnet.pt", batch=16, k=3, thresh=None, max_frames=None, device=None, log_every=6000, spacing=1):
    """(fps, per-frame lists of (x, y, score) in source pixels, threshold used). spacing: frames between the stacked inputs, so
    spacing=2 on a 120 fps clip feeds the model the same time gaps it would see in a 60 fps video. "auto": the video's frame rate
    over the frame rate the model was trained at, at least 1."""
    device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, ck = load(ckpt, device)
    th = float(ck["threshold"] if thresh is None else thresh)
    cap = cv2.VideoCapture(str(video)); fps = cap.get(cv2.CAP_PROP_FPS) or 120.0
    sw, sh = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT); cap.release()
    if spacing == "auto":
        spacing = max(1, int(round(fps / float(ck.get("fps") or 120.0))))
    src = (sw, sh)
    q = queue.Queue(maxsize=64); threading.Thread(target=_reader, args=(video, q, max_frames), daemon=True).start()
    hist, out, pending, t0 = [], [], [], time.time()

    def flush():
        x = torch.from_numpy(np.stack([p for _, p in pending])).to(device)
        for (i, _), found in zip(pending, peaks(model(x), k=k, thresh=th)):
            out[i] = [tuple(float(v) for v in to_src(a, b, src)) + (float(s),) for a, b, s in found]
        pending.clear()

    while True:
        small = q.get()
        if small is None:
            break
        i = len(out); out.append([])
        hist = (hist + [small])[-(HISTORY * spacing + 1):]
        if len(hist) == HISTORY * spacing + 1:
            pending.append((i, normalise(np.concatenate(hist[::spacing], axis=2))))
            if len(pending) == batch:
                flush()
        if log_every and i and i % log_every == 0:
            print(f"  frame {i}: {i / (time.time() - t0):.0f} frames/s", flush=True)
    if pending:
        flush()
    return fps, out, th


def save(path, fps, cands, th):
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fr = [i for i, cs in enumerate(cands) for _ in cs]
    xs = [c[0] for cs in cands for c in cs]; ys = [c[1] for cs in cands for c in cs]; sc = [c[2] for cs in cands for c in cs]
    np.savez_compressed(path, frame=np.array(fr, np.int32), x=np.array(xs, np.float32), y=np.array(ys, np.float32),
                        score=np.array(sc, np.float32), n_frames=len(cands), fps=fps, threshold=th)


def load_saved(path):
    d = np.load(path)
    cands = [[] for _ in range(int(d["n_frames"]))]
    for f, x, y, s in zip(d["frame"], d["x"], d["y"], d["score"]):
        cands[int(f)].append((float(x), float(y), float(s)))
    return float(d["fps"]), cands, float(d["threshold"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=str(ROOT / "models" / "ballnet.pt")); ap.add_argument("--max-frames", type=int)
    ap.add_argument("--thresh", type=float)
    a = ap.parse_args()
    t0 = time.time(); fps, c, th = candidates(a.video, a.ckpt, thresh=a.thresh, max_frames=a.max_frames)
    save(a.out, fps, c, th)
    print(f"{len(c)} frames, {sum(1 for x in c if x)} with a candidate, threshold {th}, {time.time() - t0:.0f} s -> {a.out}")
