"""Training data for the learned ball detector: frames cut from the OpenTTGames training videos, and a PyTorch dataset.

Layout on disk (made once by `python -m tt_scout.ml.data extract`):
    data/ml/frames/<game>/<frame>.jpg     every labelled frame and the two before it, resized to SIZE, JPEG quality 95
    data/ml/frames/<game>/labels.json     {frame: [x, y] in SIZE pixels, or null when the ball is not visible}

A sample is three consecutive frames (t-2, t-1, t) stacked as 9 channels, and a heatmap for frame t: a Gaussian of SIGMA px
round the ball whose peak pixel is exactly 1 (what the focal loss treats as the positive), all zeros when the ball is hidden.
Frames are at 120 fps, so t-2 to t spans 1/60 s: long enough for a fast ball to smear or jump, short enough to stay local.
"""
import json, pathlib, sys
import numpy as np, cv2

SIZE = (640, 352)                         # (w, h) the network sees; 1920x1080 is scaled by 3.0 across and 3.07 down
SRC = (1920, 1080)
SIGMA = 1.5                               # heatmap spread in SIZE pixels (a ball is ~3 px across at this size)
HISTORY = 2                               # frames before t in a sample
ROOT = pathlib.Path(__file__).resolve().parents[2]
FRAMES = ROOT / "data" / "ml" / "frames"
MEAN = np.array([0.45, 0.45, 0.45], np.float32) * 255
STD = np.array([0.25, 0.25, 0.25], np.float32) * 255


def to_size(x, y, src=SRC):
    """Source pixel -> SIZE pixel, mapping pixel centres to pixel centres (what cv2.INTER_AREA does). A plain x * 640 / 1920
    would put every label a third of a pixel right of the ball, and the flip augmentation would mirror that error."""
    return (x + 0.5) * SIZE[0] / src[0] - 0.5, (y + 0.5) * SIZE[1] / src[1] - 0.5


def to_src(x, y, src=SRC):
    return (x + 0.5) * src[0] / SIZE[0] - 0.5, (y + 0.5) * src[1] / SIZE[1] - 0.5


def extract(video, markup_dir, out_dir):
    """Decode `video` start to end once (no seeking, so frame numbers match the labels exactly) and write the frames a sample needs."""
    ball = json.loads((pathlib.Path(markup_dir) / "ball_markup.json").read_text())
    labelled = sorted(int(k) for k in ball)
    need = set()
    for f in labelled:
        need.update(range(f - HISTORY, f + 1))
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    src = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    last = max(need); f = 0; written = 0
    while f <= last:
        if f in need:
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / f"{f}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 95]); written += 1
        elif not cap.grab():
            break
        f += 1
    cap.release()
    labels = {}
    for k, v in ball.items():
        labels[k] = None if v["x"] < 0 or v["y"] < 0 else list(to_size(v["x"], v["y"], src))
    (out / "labels.json").write_text(json.dumps(labels))
    return written, len(labels)


def relabel(game_dir, markup_dir, src=SRC):
    """Rewrite labels.json from the markup (after a change to to_size) without extracting frames again."""
    ball = json.loads((pathlib.Path(markup_dir) / "ball_markup.json").read_text())
    labels = {k: (None if v["x"] < 0 or v["y"] < 0 else list(to_size(v["x"], v["y"], src))) for k, v in ball.items()}
    (pathlib.Path(game_dir) / "labels.json").write_text(json.dumps(labels))


def extract_negatives(video, markup_dir, out_dir, n=2000, gap_s=3.0, max_frame=60000, fps=120.0, seed=0):
    """Frames from the dead time between rallies: more than gap_s from every labelled frame. The labels only cover about 12
    frames either side of a bounce or net crossing, so without these the model never sees players walking, towels or a ball
    held in a hand, and validation cannot measure false alarms. A ball may still be in a hand here, so these are 'no ball in
    play' frames, not guaranteed empty ones. Writes the frames beside the others and negatives.json = [frame, ...]."""
    ball = json.loads((pathlib.Path(markup_dir) / "ball_markup.json").read_text())
    lab = np.array(sorted(int(k) for k in ball))
    cand = np.arange(HISTORY, max_frame)
    idx = np.searchsorted(lab, cand)
    dist = np.minimum(np.abs(cand - lab[np.clip(idx - 1, 0, len(lab) - 1)]), np.abs(lab[np.clip(idx, 0, len(lab) - 1)] - cand))
    pool = cand[dist > gap_s * fps]
    rng = np.random.default_rng(seed)
    picks = sorted(rng.choice(pool, size=min(n, len(pool)), replace=False).tolist()) if len(pool) else []
    need = {f - k for f in picks for k in range(HISTORY + 1)}
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video)); f = 0; last = max(need) if need else -1
    while f <= last:
        if f in need:
            ok, frame = cap.read()
            if not ok:
                break
            if not (out / f"{f}.jpg").exists():
                cv2.imwrite(str(out / f"{f}.jpg"), cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95])
        elif not cap.grab():
            break
        f += 1
    cap.release()
    (out / "negatives.json").write_text(json.dumps(picks))
    return len(picks)


def heatmap(x, y, w, h, sigma=SIGMA):
    """Gaussian round (x, y) in a w x h map, peak pixel exactly 1; all zeros when x is None."""
    hm = np.zeros((h, w), np.float32)
    if x is None:
        return hm
    r = int(3 * sigma + 1)
    cx, cy = int(round(x)), int(round(y))
    x0, x1, y0, y1 = max(0, cx - r), min(w, cx + r + 1), max(0, cy - r), min(h, cy + r + 1)
    if x0 >= x1 or y0 >= y1:
        return hm
    yy, xx = np.mgrid[y0:y1, x0:x1]
    hm[y0:y1, x0:x1] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    if 0 <= cx < w and 0 <= cy < h:
        hm[cy, cx] = 1.0
    return hm


def normalise(stack):
    """(H, W, 3k) uint8 BGR frames -> (3k, H, W) float32."""
    k = stack.shape[2] // 3
    x = (stack.astype(np.float32) - np.tile(MEAN, k)) / np.tile(STD, k)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


class BallFrames:
    """torch-style dataset over the extracted frames of some games.
    crop = (w, h) for training (a window containing the ball with probability p_ball, else anywhere), or None for whole frames.
    augment: horizontal flip and a brightness, contrast and colour jitter shared by the three frames."""

    def __init__(self, games, root=FRAMES, crop=None, augment=False, p_ball=0.8, stride=1, seed=0, labelled=True, negatives=False):
        """labelled: the frames of ball_markup.json; negatives: the dead-time frames of negatives.json (target all zeros)."""
        self.root, self.crop, self.augment, self.p_ball = pathlib.Path(root), crop, augment, p_ball
        self.items = []
        for g in games:
            if labelled:
                labels = json.loads((self.root / g / "labels.json").read_text())
                for f in sorted(int(k) for k in labels)[::stride]:
                    if all((self.root / g / f"{f - k}.jpg").exists() for k in range(HISTORY + 1)):
                        self.items.append((g, f, labels[str(f)]))
            if negatives and (self.root / g / "negatives.json").exists():
                for f in json.loads((self.root / g / "negatives.json").read_text()):
                    if all((self.root / g / f"{f - k}.jpg").exists() for k in range(HISTORY + 1)):
                        self.items.append((g, f, None))
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def frames(self, g, f):
        return np.concatenate([cv2.imread(str(self.root / g / f"{f - k}.jpg")) for k in range(HISTORY, -1, -1)], axis=2)

    def __getitem__(self, i):
        import torch
        g, f, lab = self.items[i]
        rng = np.random.default_rng((self.rng.integers(1 << 30) + i * 7919) if self.augment else i)
        img = self.frames(g, f)
        h, w = img.shape[:2]
        x, y = (lab if lab is not None else (None, None))
        if self.crop:                                     # crop first: every later step then works on a quarter of the pixels
            cw, ch = self.crop
            if x is not None and rng.random() < self.p_ball:
                x0 = int(np.clip(rng.integers(int(x) - cw + 8, int(x) - 7), 0, w - cw)) if cw < w else 0
                y0 = int(np.clip(rng.integers(int(y) - ch + 8, int(y) - 7), 0, h - ch)) if ch < h else 0
            else:
                x0 = int(rng.integers(0, w - cw + 1)); y0 = int(rng.integers(0, h - ch + 1))
            img = img[y0:y0 + ch, x0:x0 + cw]
            if x is not None:
                x, y = x - x0, y - y0
                if not (-0.5 <= x < cw - 0.5 and -0.5 <= y < ch - 0.5):
                    x = y = None
            h, w = ch, cw
        if self.augment:
            if rng.random() < 0.5:
                img = img[:, ::-1]; x = None if x is None else (w - 1) - x
            a = rng.uniform(0.75, 1.25); b = rng.uniform(-25, 25); cshift = rng.uniform(-12, 12, 3)
            img = cv2.convertScaleAbs(np.ascontiguousarray(img), alpha=a, beta=b)
            img = np.clip(img.astype(np.int16) + np.tile(np.round(cshift).astype(np.int16), img.shape[2] // 3), 0, 255).astype(np.uint8)
        target = heatmap(x, y, w, h)
        return torch.from_numpy(normalise(img)), torch.from_numpy(target[None]), torch.tensor([-1.0, -1.0] if x is None else [x, y])


if __name__ == "__main__" and len(sys.argv) >= 3:
    cmd, game = sys.argv[1], sys.argv[2]
    video, markup = ROOT / "data" / f"{game}.mp4", ROOT / "data" / f"{game}_markup"
    if cmd == "extract":
        n, m = extract(video, markup, FRAMES / game)
        print(f"{game}: wrote {n} frames for {m} labels")
    elif cmd == "negatives":
        print(f"{game}: {extract_negatives(video, markup, FRAMES / game)} dead-time frames")
    elif cmd == "relabel":
        relabel(FRAMES / game, markup); print(f"{game}: labels.json rewritten")
