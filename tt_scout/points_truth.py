"""Point-level ground truth: one row per point with the serve frame, the decisive frame, the server and the winner.

Two sources write the same CSV, labels/<stem>.points.csv:
  * python -m tt_scout.points_truth data/extended_labels/data/raw/game_data/test/test_3.json --fps 120
        converts the Extended OpenTT Games labels (moamal01/table_tennis_data, arXiv:2512.19327). The serve stroke
        label ("left_forehand_serve ...") gives the start frame and the server; the ending label ("left_net",
        "right_out", "right_winner", ...) gives the end frame and the winner: the named side is the striker; it
        wins for "winner" and "double_bounce" and loses for the four error endings. "left" is the x=0 ("near") end in tt_scout's table frame,
        which is how openttgames.py orders the corners (left end of the image = corners 1-2).
  * python label_points.py match.mp4      keyboard labelling of your own footage, same columns.

Columns: id, start_frame, start_t, end_frame, end_t, server, winner, ending, start_known, source
  start = the serve contact frame; end = the frame of the event that decides the point (net, out, double
  bounce, winner ...). Bounces of the dead ball after that are not part of the point.
  start_known = 0 when the clip starts inside a point (an ending with no serve label before it).
  ending = "unknown" for rally activity whose outcome was not labelled (a serve or strokes with no ending label: a let, a
  clip cut, or simply an unlabelled rally); end_frame is the last label of that rally, and evaluators ignore predictions
  that fall inside such rows.
"""
import argparse, csv, json, pathlib

FIELDS = ["id", "start_frame", "start_t", "end_frame", "end_t", "server", "winner", "ending", "start_known", "source"]
ENDINGS = ("net", "out", "winner", "double_bounce", "miss_on_own_side", "not_hitting_ball")
# Extended OpenTT Games README: "winner: the player hits a shot which the opponent cannot reach"; "double_bounce: the ball bounces
# twice on the OPPONENT's side before they can return it". In both the named player is the striker and WINS the point. In the
# other four (net, out, miss_on_own_side, not_hitting_ball) the named player made the error and loses. (Until 2026-09-17 this
# file treated double_bounce as a loss for the named side; 4 of 281 points were affected, all in the training games.)
NAMED_SIDE_WINS = ("winner", "double_bounce")
SIDE = {"left": "near", "right": "far"}


def other(side):
    return "far" if side == "near" else "near"


def _kind(v):
    """'serve' | 'stroke' | 'ending' | 'ball' for one raw label."""
    head = v.split()[0]
    side = head.split("_", 1)[0]
    if " " not in v and "_" in v and side in SIDE and v.split("_", 1)[1] in ENDINGS:
        return "ending"
    if side in SIDE:
        return "serve" if head.endswith("_serve") else "stroke"
    return "ball"                                              # bounce, net, empty_event


def from_extended_labels(labels, fps, source="", gap_s=2.5):
    """labels: {frame: label} as in the Extended OpenTT Games JSON. Returns a list of point dicts.

    The annotations are not complete: in the training games many rallies have no serve label, no ending label, or neither
    (game_3, 156-204 s: six rallies between one labelled serve and the next labelled ending). Labels are therefore first cut
    into contiguous SEGMENTS (a gap of more than gap_s between consecutive labels ends a segment; inside a rally labels come
    every few tenths of a second), and a point never spans two segments:
      * a segment with an ending label gives a known point, from its serve label (start_known=1) or, when the serve was not
        labelled, from the segment's first label (start_known=0);
      * rally activity with no ending (a serve or any stroke label, but no ending label) gives a point with ending
        "unknown": evaluators ignore predictions that fall inside it;
      * a segment with ball events only (bounce / net: the ball being knocked back to the server) gives nothing, so a
        prediction there counts as a false point."""
    items = sorted(((int(f), v) for f, v in labels.items()), key=lambda kv: kv[0])
    if not items:
        return []
    segs, cur = [], [items[0]]
    for a, b in zip(items, items[1:]):
        if (b[0] - a[0]) / fps > gap_s:
            segs.append(cur); cur = [b]
        else:
            cur.append(b)
    segs.append(cur)
    points = []

    def emit(pt, end_frame, winner, ending):
        pt = dict(pt); pt.pop("has_stroke", None)
        pt.update(end_frame=end_frame, end_t=end_frame / fps, winner=winner, ending=ending)
        points.append(pt)

    for seg in segs:
        pending = None
        for f, v in seg:
            k = _kind(v)
            if k == "serve":
                if pending is not None and (pending["start_known"] or pending["has_stroke"]):
                    emit(pending, f, "", "unknown")            # rally activity that never got an ending before this serve
                pending = dict(start_frame=f, start_t=f / fps, server=SIDE[v.split("_", 1)[0]], start_known=1, has_stroke=True)
                continue
            if pending is None:
                pending = dict(start_frame=f, start_t=f / fps, server="", start_known=0, has_stroke=False)
            if k == "stroke":
                pending["has_stroke"] = True
            elif k == "ending":
                side_word, kind = v.split("_", 1)
                causer = SIDE[side_word]
                emit(pending, f, causer if kind in NAMED_SIDE_WINS else other(causer), kind)
                pending = None
        if pending is not None and (pending["start_known"] or pending["has_stroke"]):
            emit(pending, seg[-1][0], "", "unknown")
    points.sort(key=lambda p: p["start_frame"])
    for i, p in enumerate(points, start=1):
        p["id"], p["source"] = i, source
    return points


def write_points(points, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for p in points:
            w.writerow({k: p.get(k, "") for k in FIELDS})
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels_json", help="Extended OpenTT Games label file, e.g. .../game_data/test/test_3.json")
    ap.add_argument("--fps", type=float, default=120.0, help="OpenTTGames videos are 120 fps")
    ap.add_argument("--out", help="default labels/<stem>.points.csv")
    ap.add_argument("--source", default="Extended OpenTT Games (moamal01/table_tennis_data)")
    a = ap.parse_args()
    stem = pathlib.Path(a.labels_json).stem
    pts = from_extended_labels(json.load(open(a.labels_json)), a.fps, a.source)
    out = write_points(pts, a.out or pathlib.Path("labels") / f"{stem}.points.csv")
    known = [p for p in pts if p["ending"] != "unknown"]
    print(f"{stem}: {len(pts)} points -> {out}; {len(known)} with a known outcome, "
          f"{sum(p['start_known'] for p in pts)} with a labelled serve; "
          f"servers near/far {sum(p['server'] == 'near' for p in pts)}/{sum(p['server'] == 'far' for p in pts)}; "
          f"winners near/far {sum(p['winner'] == 'near' for p in known)}/{sum(p['winner'] == 'far' for p in known)}")


if __name__ == "__main__":
    main()
