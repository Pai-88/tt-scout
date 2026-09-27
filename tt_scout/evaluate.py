"""Compare detected events with ground truth (synthetic truth JSON, or human labels from label_events.py)."""
import argparse, csv, json, math
import numpy as np


def match(pred, truth, tol_s):
    """Greedy one-to-one matching by time. Returns (pairs[(truth, pred)], n_fp, n_fn)."""
    used, pairs = set(), []
    for te in truth:
        best = None
        for j, pe in enumerate(pred):
            if j in used:
                continue
            d = abs(pe["t"] - te["t"])
            if d <= tol_s and (best is None or d < best[0]):
                best = (d, j)
        if best:
            used.add(best[1]); pairs.append((te, pred[best[1]]))
    return pairs, len(pred) - len(pairs), len(truth) - len(pairs)


def evaluate(pred_events, truth_events, fps, tol_frames=3):
    tol = tol_frames / fps
    pb = [e for e in pred_events if e["kind"] == "bounce"]
    tb = [e for e in truth_events if e["kind"] == "bounce"]
    pairs, fp, fn = match(pb, tb, tol)
    errs = [math.hypot(pe["x_m"] - te["x_m"], pe["y_m"] - te["y_m"]) * 100
            for te, pe in pairs if "x_m" in te and "x_m" in pe]
    vis = [e for e in tb if e.get("visible", True)]
    vis_got = sum(1 for te, _ in pairs if te.get("visible", True))
    by_side = {}
    for side in ("near", "far"):
        ts = [e for e in tb if e["side"] == side]
        got = sum(1 for te, _ in pairs if te["side"] == side)
        by_side[side] = (got, len(ts))
    # "hit" from the detector means "kink not on the table": rackets, floor, net cord all count as truth here
    ph = [e for e in pred_events if e["kind"] in ("hit", "nethit")]
    th = [e for e in truth_events if e["kind"] in ("hit", "floor", "nethit")]
    hp, hfp, hfn = match(ph, th, tol)
    npairs, _, _ = match([e for e in pred_events if e["kind"] == "net"], [e for e in truth_events if e["kind"] == "net"], tol)
    n_true_net = sum(1 for e in truth_events if e["kind"] == "net")
    # bounces the detector saw but called "hit" (the near-end ambiguity from a behind camera)
    conf, _, _ = match(ph, [e for e in tb if not any(te is e for te, _ in pairs)], tol)
    return dict(
        bounce_precision=len(pairs) / max(1, len(pb)), bounce_recall=len(pairs) / max(1, len(tb)),
        n_true_bounces=len(tb), n_pred_bounces=len(pb), bounce_fp=fp, bounce_fn=fn,
        bounce_recall_near=by_side["near"], bounce_recall_far=by_side["far"],
        bounces_called_hit=len(conf), bounce_recall_when_visible=(vis_got, len(vis)),
        loc_err_median_cm=float(np.median(errs)) if errs else float("nan"),
        loc_err_p90_cm=float(np.percentile(errs, 90)) if errs else float("nan"),
        offtable_recall=len(hp) / max(1, len(th)), offtable_precision=len(hp) / max(1, len(ph)),
        net_recall=(len(npairs), n_true_net),
    )


def load_labels(path, table):
    """labels/<stem>.csv from label_events.py -> truth events with metres via the table homography."""
    out = []
    for r in csv.DictReader(open(path)):
        e = dict(t=float(r["t"]), kind=r["kind"], side=r.get("side") or None)
        if r.get("x") and r["kind"] == "bounce":
            xm, ym = table.to_table([(float(r["x"]), float(r["y"]))])[0]
            e.update(x_m=float(xm), y_m=float(ym), side="near" if xm < 1.37 else "far")
        out.append(e)
    return out


def main():
    from .table import Table
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="out/<stem>/events.csv")
    ap.add_argument("--labels", required=True, help="labels/<stem>.csv")
    ap.add_argument("--table", default="table.json")
    ap.add_argument("--fps", type=float, required=True)
    a = ap.parse_args()
    pred = [dict(t=float(r["t"]), kind=r["kind"], side=r["side"], x_m=float(r["x_m"]), y_m=float(r["y_m"]))
            for r in csv.DictReader(open(a.pred))]
    truth = load_labels(a.labels, Table.load(a.table))
    for k, v in evaluate(pred, truth, a.fps).items():
        print(f"{k:24s} {v}")


if __name__ == "__main__":
    main()
