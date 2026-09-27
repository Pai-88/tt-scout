"""Point-level match-state metrics with bootstrap confidence intervals.

Ground truth: labels/<stem>.points.csv (tt_scout.points_truth / label_points.py).
Predictions:  out/<stem>/rallies.json (tt_scout.run or eval_openttgames.py).

A true point and a predicted point are matched one-to-one (greedy, largest temporal overlap first) under one of
two criteria:
  cover (default)  the predicted interval covers the true point:
                       pred.start - tol_start <= true.start   (when the serve frame is known)
                       true.end <= pred.end + tol_end
                   The state machine ends a point at the decisive event plus a timeout, so "cover" tests what the
                   pipeline promises: one predicted point per true point, containing its serve and its ending.
  iou              temporal IoU >= iou_thresh, the temporal-action-detection convention. F1 at 0.3/0.5/0.7 is
                   reported alongside whichever criterion is primary, so results can be read either way.

Reported per clip and pooled: point detection precision / recall / F1; winner and server accuracy on matched
points ("unsure" = the pipeline returned no winner, counted as wrong); signed start and end errors (pred - true,
seconds); bounce landing recall and localisation error in cm when event files are given. 95 % intervals are
percentile bootstraps over points (detection units TP / FP / FN, or matched points) and over matched bounces.
Predictions that fall inside a true point of unknown outcome (a let, a cut clip) are ignored, not false positives.
"""
import argparse, csv, json, math, pathlib
import numpy as np

from .evaluate import match as match_events

NAN = float("nan")


# ----------------------------------------------------------------------------------------------- loading
def load_points_csv(path):
    pts = []
    for r in csv.DictReader(open(path)):
        pts.append(dict(id=r["id"], server=r["server"] or None, winner=r["winner"] or None,
                        ending=r.get("ending") or "unknown", start_known=int(r.get("start_known") or 1),
                        start_t=float(r["start_t"]) if r.get("start_t") else None,
                        end_t=float(r["end_t"]) if r.get("end_t") else None))
    return pts


def load_pred_points(path):
    return [dict(id=r["id"], start_t=float(r["start_t"]), end_t=float(r["end_t"]), server=r.get("serve_side"),
                 winner=r.get("winner"), ending=r.get("ending")) for r in json.load(open(path))]


def load_pred_events(path):
    out = []
    for r in csv.DictReader(open(path)):
        e = dict(t=float(r["t"]), kind=r["kind"], side=r.get("side") or None)
        if r.get("x_m"):
            e.update(x_m=float(r["x_m"]), y_m=float(r["y_m"]))
        out.append(e)
    return out


def load_truth_events(path):
    """data/<stem>_truth.json from openttgames.py (or any JSON with an 'events' list and 'fps')."""
    d = json.load(open(path))
    return d["events"], float(d.get("fps", 0) or 0)


# ----------------------------------------------------------------------------------------------- matching
def _overlap(a, b):
    return max(0.0, min(a["end_t"], b["end_t"]) - max(a["start_t"], b["start_t"]))


def iou(t, p):
    inter = _overlap(t, p)
    union = (t["end_t"] - t["start_t"]) + (p["end_t"] - p["start_t"]) - inter
    return inter / union if union > 0 else 0.0


def covers(t, p, tol_start=0.5, tol_end=0.5):
    if t["end_t"] is None:
        return False
    if t["start_known"] and t["start_t"] is not None and p["start_t"] - tol_start > t["start_t"]:
        return False
    if t["end_t"] < p["start_t"] - tol_start:
        return False
    return t["end_t"] <= p["end_t"] + tol_end


def match_points(truth, pred, criterion="cover", iou_thresh=0.5, tol_start=0.5, tol_end=0.5):
    """Greedy one-to-one matching. Returns (pairs [(truth, pred)], false_positives [pred], false_negatives [truth],
    ignored [pred]) where ignored = predictions inside a true point of unknown outcome."""
    known = [t for t in truth if t["ending"] != "unknown" and t["end_t"] is not None]
    unknown = [t for t in truth if t not in known]
    cands = []
    for ti, t in enumerate(known):
        for pi, p in enumerate(pred):
            if t["start_t"] is None:
                continue
            ok = covers(t, p, tol_start, tol_end) if criterion == "cover" else iou(t, p) >= iou_thresh
            if ok:
                cands.append((_overlap(t, p), iou(t, p), ti, pi))
    cands.sort(key=lambda c: (-c[0], -c[1]))                   # largest overlap first, tighter prediction on ties
    used_t, used_p, pairs = set(), set(), []
    for _, _, ti, pi in cands:
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti); used_p.add(pi); pairs.append((known[ti], pred[pi]))
    fn = [t for i, t in enumerate(known) if i not in used_t]
    rest = [p for i, p in enumerate(pred) if i not in used_p]
    # an unknown-outcome point with no end label runs to the end of the clip (the recording stopped inside it)
    spans = [dict(start_t=u["start_t"], end_t=u["end_t"] if u["end_t"] is not None else float("inf"))
             for u in unknown if u["start_t"] is not None]
    ignored = [p for p in rest if any(_overlap(u, p) > 0 for u in spans)]
    fp = [p for p in rest if p not in ignored]
    return pairs, fp, fn, ignored


# ----------------------------------------------------------------------------------------------- statistics
def prf(codes):
    """codes: array of 0 = TP, 1 = FP, 2 = FN -> (precision, recall, f1); F1 = 0 when there are no true positives."""
    codes = np.asarray(codes)
    tp, fp, fn = int((codes == 0).sum()), int((codes == 1).sum()), int((codes == 2).sum())
    p = tp / (tp + fp) if tp + fp else NAN
    r = tp / (tp + fn) if tp + fn else NAN
    f1 = 0.0 if tp == 0 else 2 * p * r / (p + r)
    return p, r, f1


def boot_ci(values, stat, n_boot=2000, seed=0):
    """Percentile bootstrap 95 % interval of stat(resampled values)."""
    values = np.asarray(values)
    if values.size == 0:
        return (NAN, NAN)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    vals = np.array([stat(values[i]) for i in idx], float)
    vals = vals[~np.isnan(vals)]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals.size else (NAN, NAN)


def summarise(pairs, fp, fn, n_boot=2000, seed=0):
    codes = [0] * len(pairs) + [1] * len(fp) + [2] * len(fn)
    p, r, f1 = prf(codes)
    win = [(pp["winner"] == t["winner"]) for t, pp in pairs if t["winner"]]
    unsure = sum(1 for t, pp in pairs if t["winner"] and pp["winner"] is None)
    srv = [(pp["server"] == t["server"]) for t, pp in pairs if t["server"]]
    d_start = [pp["start_t"] - t["start_t"] for t, pp in pairs if t["start_known"] and t["start_t"] is not None]
    d_end = [pp["end_t"] - t["end_t"] for t, pp in pairs]
    med = lambda v: float(np.median(v)) if len(v) else NAN
    return dict(
        n_truth=len(pairs) + len(fn), n_pred=len(pairs) + len(fp), tp=len(pairs), fp=len(fp), fn=len(fn),
        precision=p, recall=r, f1=f1, f1_ci=boot_ci(codes, lambda c: prf(c)[2], n_boot, seed),
        winner_n=len(win), winner_right=int(sum(win)), winner_unsure=unsure,
        winner_acc=float(np.mean(win)) if win else NAN, winner_ci=boot_ci(win, np.mean, n_boot, seed),
        server_n=len(srv), server_right=int(sum(srv)),
        server_acc=float(np.mean(srv)) if srv else NAN, server_ci=boot_ci(srv, np.mean, n_boot, seed),
        start_err_median_s=med(d_start), end_err_median_s=med(d_end),
        start_err_mean_s=float(np.mean(d_start)) if d_start else NAN, end_err_mean_s=float(np.mean(d_end)) if d_end else NAN,
        _codes=codes, _win=win, _srv=srv, _dstart=d_start, _dend=d_end)


def landing(pred_events, truth_events, fps, tol_frames=6):
    """Bounce recall and localisation error (cm) against labelled bounces with table coordinates."""
    pb = [e for e in pred_events if e["kind"] == "bounce"]
    tb = [e for e in truth_events if e["kind"] == "bounce"]
    pairs, _, _ = match_events(pb, tb, tol_frames / fps)
    errs = [math.hypot(pe["x_m"] - te["x_m"], pe["y_m"] - te["y_m"]) * 100 for te, pe in pairs if "x_m" in te and "x_m" in pe]
    return dict(n_truth_bounces=len(tb), n_pred_bounces=len(pb), matched=len(pairs),
                recall=len(pairs) / len(tb) if tb else NAN, _errs=errs)


def landing_summary(errs, n_boot=2000, seed=0):
    return dict(n=len(errs), median_cm=float(np.median(errs)) if errs else NAN,
                p90_cm=float(np.percentile(errs, 90)) if errs else NAN,
                median_ci=boot_ci(errs, np.median, n_boot, seed))


def iou_sweep(truth, pred, thresholds=(0.3, 0.5, 0.7)):
    """Detection unit codes (0 TP, 1 FP, 2 FN) per IoU threshold, so clips can be pooled before computing F1."""
    out = {}
    for th in thresholds:
        pairs, fp, fn, _ = match_points(truth, pred, criterion="iou", iou_thresh=th)
        out[th] = [0] * len(pairs) + [1] * len(fp) + [2] * len(fn)
    return out


# ----------------------------------------------------------------------------------------------- CLI
def _fmt_ci(ci):
    return "[" + ", ".join("  nan" if math.isnan(v) else f"{v:.2f}" for v in ci) + "]"


def _row(name, s, land):
    acc = lambda k, n, right: ("   -  " if not s[n] else f"{right}/{s[n]} {s[k]:.2f}")
    cm = "" if land is None else (f"  {land['matched']}/{land['n_truth_bounces']} bounces, {land['median_cm']:.1f} cm {_fmt_ci(land['median_ci'])}" if land["n"] else "  no matched bounces")
    return (f"{name:10s} {s['n_truth']:3d} {s['n_pred']:3d} {s['tp']:3d} {s['fp']:3d} {s['fn']:3d}  "
            f"{s['precision']:.2f} {s['recall']:.2f} {s['f1']:.2f} {_fmt_ci(s['f1_ci'])}  "
            f"{acc('winner_acc', 'winner_n', s['winner_right']):>12s} {_fmt_ci(s['winner_ci'])}  "
            f"{acc('server_acc', 'server_n', s['server_right']):>12s}  "
            f"{s['start_err_median_s']:+.2f} {s['end_err_median_s']:+.2f}{cm}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="*", default=[], help="clip bases, e.g. data/test_3: truth labels/<stem>.points.csv, "
                    "predictions out/<stem>/rallies.json, events out/<stem>/events.csv, truth events data/<stem>_truth.json")
    ap.add_argument("--truth", help="single clip: points CSV"); ap.add_argument("--pred", help="single clip: rallies.json")
    ap.add_argument("--name", default="clip")
    ap.add_argument("--labels-dir", default="labels"); ap.add_argument("--out-root", default="out")
    ap.add_argument("--criterion", choices=["cover", "iou"], default="cover")
    ap.add_argument("--iou", type=float, default=0.5); ap.add_argument("--tol-start", type=float, default=0.5); ap.add_argument("--tol-end", type=float, default=0.5)
    ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-landing", action="store_true"); ap.add_argument("--fps", type=float)
    ap.add_argument("--json", help="write all numbers here")
    a = ap.parse_args()

    clips = []
    for c in a.clips:
        stem = pathlib.Path(c).stem
        clips.append(dict(name=stem, truth=pathlib.Path(a.labels_dir) / f"{stem}.points.csv",
                          pred=pathlib.Path(a.out_root) / stem / "rallies.json",
                          events=pathlib.Path(a.out_root) / stem / "events.csv", truth_events=pathlib.Path(f"{c}_truth.json")))
    if a.truth and a.pred:
        clips.append(dict(name=a.name, truth=pathlib.Path(a.truth), pred=pathlib.Path(a.pred), events=None, truth_events=None))
    if not clips:
        raise SystemExit("give --clips data/test_3 ... or --truth/--pred")

    kw = dict(criterion=a.criterion, iou_thresh=a.iou, tol_start=a.tol_start, tol_end=a.tol_end)
    print(f"criterion={a.criterion}" + (f" iou>={a.iou}" if a.criterion == "iou" else f" tol_start={a.tol_start}s tol_end={a.tol_end}s")
          + f"; 95% percentile bootstrap, {a.n_boot} resamples, seed {a.seed}")
    print(f"{'clip':10s} {'tru':>3s} {'prd':>3s} {'TP':>3s} {'FP':>3s} {'FN':>3s}  {'P':>4s} {'R':>4s} {'F1':>4s} [95% CI]      "
          f"{'winner acc':>12s} [95% CI]      {'server acc':>12s}  dstart  dend   landing")
    results, pool = {}, dict(pairs=[], fp=[], fn=[], ignored=[], errs=[], iou={})
    for c in clips:
        truth, pred = load_points_csv(c["truth"]), load_pred_points(c["pred"])
        pairs, fp, fn, ign = match_points(truth, pred, **kw)
        s = summarise(pairs, fp, fn, a.n_boot, a.seed)
        land = None
        if not a.no_landing and c["events"] and c["events"].exists() and c["truth_events"].exists():
            tev, fps = load_truth_events(c["truth_events"])
            lm = landing(load_pred_events(c["events"]), tev, a.fps or fps)
            land = {**lm, **landing_summary(lm["_errs"], a.n_boot, a.seed)}
            pool["errs"] += lm["_errs"]
        iou_codes = iou_sweep(truth, pred)
        s["iou_f1"] = {th: prf(c)[2] for th, c in iou_codes.items()}; s["ignored"] = len(ign)
        for th, codes_th in iou_codes.items():
            pool["iou"].setdefault(th, []).extend(codes_th)
        s["unmatched_truth"] = [dict(id=t["id"], start_t=t["start_t"], end_t=t["end_t"], winner=t["winner"]) for t in fn]
        s["false_positives"] = [dict(id=p["id"], start_t=p["start_t"], end_t=p["end_t"], winner=p["winner"]) for p in fp]
        s["wrong_winner"] = [dict(truth_id=t["id"], pred_id=p["id"], truth=t["winner"], pred=p["winner"]) for t, p in pairs if t["winner"] and p["winner"] != t["winner"]]
        results[c["name"]] = dict(points={k: v for k, v in s.items() if not k.startswith("_")},
                                  landing=None if land is None else {k: v for k, v in land.items() if not k.startswith("_")})
        print(_row(c["name"], s, land))
        pool["pairs"] += pairs; pool["fp"] += fp; pool["fn"] += fn; pool["ignored"] += ign
    if len(clips) > 1:
        s = summarise(pool["pairs"], pool["fp"], pool["fn"], a.n_boot, a.seed)
        land = None if a.no_landing or not pool["errs"] else dict(matched=len(pool["errs"]), n_truth_bounces=sum(
            r["landing"]["n_truth_bounces"] for r in results.values() if r["landing"]), **landing_summary(pool["errs"], a.n_boot, a.seed))
        s["ignored"] = len(pool["ignored"])
        s["iou_f1"] = {th: prf(codes_th)[2] for th, codes_th in pool["iou"].items()}
        results["pooled"] = dict(points={k: v for k, v in s.items() if not k.startswith("_")},
                                 landing=None if land is None else {k: v for k, v in land.items() if not k.startswith("_")})
        print(_row("pooled", s, land))
    r0 = results["pooled" if "pooled" in results else clips[0]["name"]]["points"]
    print(f"F1 by temporal IoU (secondary, {'pooled units' if 'pooled' in results else 'this clip'}): " +
          "  ".join(f"@{th}: {v:.2f}" for th, v in r0["iou_f1"].items()) +
          f";  predictions ignored inside unknown-outcome points: {r0['ignored']}")
    for name, r in results.items():
        pts = r["points"]
        for w in pts.get("wrong_winner", []):
            print(f"  {name}: wrong winner on true point {w['truth_id']} (pred {w['pred_id']}): truth {w['truth']}, pred {w['pred']}")
        for t in pts.get("unmatched_truth", []):
            print(f"  {name}: MISSED true point {t['id']} {t['start_t']:.1f}-{t['end_t']:.1f}s (winner {t['winner']})")
        for p in pts.get("false_positives", []):
            print(f"  {name}: false point {p['id']} {p['start_t']:.1f}-{p['end_t']:.1f}s")
    if a.json:
        pathlib.Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.json).write_text(json.dumps(dict(settings=vars(a), results=results), indent=1, default=float))
        print("wrote", a.json)


if __name__ == "__main__":
    main()
