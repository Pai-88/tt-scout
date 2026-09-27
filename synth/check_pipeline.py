"""End-to-end smoke test on synthetic clips. Prints detection accuracy vs known ground truth.
Usage: python synth/check_pipeline.py [--views side behind] [--regen]"""
import argparse, json, pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tt_scout.config import Config
from tt_scout.run import analyse
from tt_scout.evaluate import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--views", nargs="+", default=["side", "behind"])
ap.add_argument("--regen", action="store_true")
ap.add_argument("--seconds", type=float, default=40)
a = ap.parse_args()

for view in a.views:
    base = ROOT / "synth" / view
    if a.regen or not base.with_suffix(".mp4").exists():
        subprocess.run([sys.executable, str(ROOT / "synth/make_synthetic.py"), "--view", view,
                        "--seconds", str(a.seconds), "--out", str(base)], check=True)
    truth = json.loads(pathlib.Path(str(base) + "_truth.json").read_text())
    res = analyse(str(base) + ".mp4", str(base) + "_table.json", Config(ball_colour="orange"),
                  out_root=ROOT / "synth" / "out", save_video=(view == "side"))
    m = evaluate(res["events"], truth["events"], res["fps"])
    print(f"\n=== {view} camera: {res['n_frames']} frames in {res['seconds']:.0f}s, "
          f"ball seen {res['ball_seen_frac']*100:.0f}% of frames (truth visible {truth['ball_visible_frames']/truth['frames']*100:.0f}%)")
    print(f"rallies: detected {len(res['rallies'])} vs true {len(truth['rallies'])}")
    print(f"bounces: {m['n_pred_bounces']} detected vs {m['n_true_bounces']} true | "
          f"precision {m['bounce_precision']:.2f}  recall {m['bounce_recall']:.2f}  "
          f"(near {m['bounce_recall_near'][0]}/{m['bounce_recall_near'][1]}, far {m['bounce_recall_far'][0]}/{m['bounce_recall_far'][1]}); "
          f"true bounces mislabelled as hit: {m['bounces_called_hit']}")
    v = m['bounce_recall_when_visible']; print(f"bounces where the ball was actually drawn (not hidden by a player): {v[0]}/{v[1]} found")
    print(f"bounce position error: median {m['loc_err_median_cm']:.1f} cm, p90 {m['loc_err_p90_cm']:.1f} cm")
    print(f"off-table kinks (racket/floor/net): recall {m['offtable_recall']:.2f}  precision {m['offtable_precision']:.2f}")
    # winners: match each true rally to the detected rally overlapping it most
    hit = tot = unk = 0
    for tr in truth["rallies"]:
        best = max(res["rallies"], key=lambda r: min(r["end_t"], tr["end_t"]) - max(r["start_t"], tr["start_t"]), default=None)
        if best is None or min(best["end_t"], tr["end_t"]) - max(best["start_t"], tr["start_t"]) <= 0:
            continue
        tot += 1
        if best["winner"] is None: unk += 1
        elif best["winner"] == tr["winner"]: hit += 1
    print(f"net crossings: {m['net_recall'][0]}/{m['net_recall'][1]} found")
    print(f"point winners: {hit} right, {tot - hit - unk} wrong, {unk} unsure, of {tot} rallies with a known winner")
    nh = nt = eh = 0
    for tr in truth["rallies"]:
        best = max(res["rallies"], key=lambda r: min(r["end_t"], tr["end_t"]) - max(r["start_t"], tr["start_t"]), default=None)
        if best is None or min(best["end_t"], tr["end_t"]) - max(best["start_t"], tr["start_t"]) <= 0:
            continue
        nt += 1; eh += best.get("near_player") == tr["near_player"]
        nh += (best.get("winner_name") == tr["winner_name"]) if tr["winner_name"] else 0
    print(f"ends named right (shirt colour): {eh}/{nt};  winners by NAME right: {nh}/{nt}")
    print(f"report: {res['out']}/report.png")
