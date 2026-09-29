"""Train BallNet on the extracted OpenTTGames training frames.

    python -m tt_scout.ml.train --epochs 14            # writes models/ballnet.pt and models/ballnet_train.jsonl
    python -m tt_scout.ml.train --train phone_a phone_b --val phone_c --fps 60 --stride 1 --out models/ballnet_phone.pt
                                                       # on your own recordings, labelled by tt_scout.ml.pseudo

Protocol. Train on game_1, game_2, game_3, game_5: their labelled frames (which all sit within ~12 frames of a bounce or net
crossing) plus dead-time frames between rallies as negatives. Choose the epoch and the detection threshold on game_4, which is
never trained on. game_4 shares its camera setup with game_3, so validation numbers are same-setup numbers; the seven test
clips are never read here (only test_6 shares a setup with a training game, game_2).
Training uses 320x176 windows cut from the 640x352 frames, because a fully convolutional network trained on windows runs
unchanged on whole frames and windows are 4x cheaper.

Validation, at each threshold, with the top peak of each frame:
  labelled frames: precision, recall, F1, where a hit is within TOL px at 640x352 (about 9 px in the 1920x1080 original), and
    the median error of hits in original pixels;
  dead-time frames: fire rate, the share with any peak. Not a false-positive rate: a ball can be in a hand between rallies.
The chosen threshold is the best F1 among thresholds whose fire rate is at most MAX_FIRE.
"""
import argparse, json, pathlib, time
import numpy as np
try:
    import torch
except ImportError as e:
    raise SystemExit(f'training needs PyTorch ({e}): install it with  pip install -e ".[ml]"')
from torch.utils.data import DataLoader
from .data import BallFrames, ROOT, SIZE, SRC, HISTORY, MEAN, STD, SIGMA, to_src
from .model import BallNet, focal_loss, peaks

TOL = 3.0
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
MAX_FIRE = 0.05


@torch.no_grad()
def validate(model, labelled, dead, device):
    model.eval()
    cnt = {th: np.zeros(3) for th in THRESHOLDS}; errs = {th: [] for th in THRESHOLDS}; fired = {th: 0 for th in THRESHOLDS}; n_dead = 0
    for x, _, xy in labelled:
        for found, t in zip(peaks(model(x.to(device)), k=1, thresh=min(THRESHOLDS)), xy.numpy()):
            vis = t[0] >= 0
            for th in THRESHOLDS:
                top = found[0] if found and found[0][2] >= th else None
                if top is None:
                    cnt[th] += (0, 0, int(vis))
                elif vis and np.hypot(top[0] - t[0], top[1] - t[1]) <= TOL:
                    cnt[th] += (1, 0, 0)
                    a, b = to_src(top[0], top[1]); c, d = to_src(t[0], t[1])
                    errs[th].append(float(np.hypot(a - c, b - d)))
                else:
                    cnt[th] += (0, 1, int(vis))
    for x, _, _ in dead:
        for found in peaks(model(x.to(device)), k=1, thresh=min(THRESHOLDS)):
            n_dead += 1
            for th in THRESHOLDS:
                fired[th] += bool(found and found[0][2] >= th)
    res = {}
    for th in THRESHOLDS:
        tp, fp, fn = (float(v) for v in cnt[th])
        p = tp / max(1.0, tp + fp); r = tp / max(1.0, tp + fn)
        res[th] = dict(precision=round(p, 4), recall=round(r, 4), f1=round(2 * p * r / max(1e-9, p + r), 4),
                       median_hit_err_src_px=round(float(np.median(errs[th])), 2) if errs[th] else None,
                       fire_rate_dead_time=round(fired[th] / max(1, n_dead), 4))
    ok = [th for th in THRESHOLDS if res[th]["fire_rate_dead_time"] <= MAX_FIRE] or [min(THRESHOLDS, key=lambda th: res[th]["fire_rate_dead_time"])]
    best = max(ok, key=lambda th: res[th]["f1"])
    n_hidden = int(sum(1 for _, _, xy in labelled.dataset.items if xy is None))
    return dict(by_threshold={str(k): v for k, v in res.items()}, best_threshold=float(best), best=res[best],
                n_labelled=len(labelled.dataset), n_labelled_hidden=n_hidden, n_dead_time=n_dead)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", default=["game_1", "game_2", "game_3", "game_5"])
    ap.add_argument("--val", nargs="+", default=["game_4"])
    ap.add_argument("--epochs", type=int, default=14); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--out")
    ap.add_argument("--init", help="start from this checkpoint's weights instead of from scratch")
    ap.add_argument("--fps", type=float, default=120.0, help="frame rate of the training videos (kept in the checkpoint)")
    ap.add_argument("--val-stride", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="batches per epoch, for a smoke test (writes models/smoke.pt unless --out)")
    a = ap.parse_args()
    out = pathlib.Path(a.out or ROOT / "models" / ("smoke.pt" if a.limit else "ballnet.pt")); out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train = BallFrames(a.train, crop=(320, 176), augment=True, stride=a.stride, negatives=True)
    val_lab = BallFrames(a.val, stride=a.val_stride)
    val_dead = BallFrames(a.val, labelled=False, negatives=True)
    val_dead.items = val_dead.items[::2]
    pw = a.workers > 0
    tl = DataLoader(train, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=pw)
    vl = DataLoader(val_lab, batch_size=16, num_workers=min(4, a.workers), persistent_workers=pw)
    vd = DataLoader(val_dead, batch_size=16, num_workers=min(4, a.workers), persistent_workers=pw)
    model = BallNet()
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=False)["state"])
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    per_epoch = min(a.limit, len(tl)) if a.limit else len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * per_epoch, pct_start=0.1)
    log = out.with_name(out.stem + "_train.jsonl")
    n_neg = sum(1 for it in train.items if it[2] is None)
    print(f"train {len(train)} samples ({n_neg} without a ball) from {a.train}; val {len(val_lab)} labelled + {len(val_dead)} dead-time "
          f"from {a.val}; {per_epoch} batches/epoch on {device} -> {out}", flush=True)
    best_f1 = -1.0
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); losses = []
        for i, (x, y, _) in enumerate(tl):
            if i >= per_epoch:
                break
            loss = focal_loss(model(x.to(device)), y.to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            losses.append(loss.item())
            if i % 100 == 0:
                print(f"  epoch {ep + 1} batch {i}/{per_epoch} loss {np.mean(losses[-100:]):.3f}", flush=True)
        v = validate(model, vl, vd, device)
        row = dict(epoch=ep + 1, loss=round(float(np.mean(losses)), 4), seconds=round(time.time() - t0), val=v)
        with log.open("a") as f:
            f.write(json.dumps(row) + "\n")
        b = v["best"]
        print(f"epoch {ep + 1}: loss {row['loss']}  val F1 {b['f1']} (P {b['precision']} R {b['recall']}) at threshold {v['best_threshold']}, "
              f"hit error {b['median_hit_err_src_px']} px (1080p), fires on {b['fire_rate_dead_time']:.1%} of dead-time frames, {row['seconds']} s", flush=True)
        if b["f1"] > best_f1:
            best_f1 = b["f1"]
            torch.save(dict(state={k: t.detach().cpu() for k, t in model.state_dict().items()}, config=model.config,
                            threshold=float(v["best_threshold"]), val=v, epoch=ep + 1, size=list(SIZE), src_size=list(SRC), history=HISTORY,
                            mean=[float(m) for m in MEAN], std=[float(sd) for sd in STD], sigma=SIGMA, peak_window=7, fps=a.fps, init=a.init,
                            channel_order="BGR", train_games=a.train, val_games=a.val), out)
    print(f"best val F1 {best_f1} -> {out}")


if __name__ == "__main__":
    main()
