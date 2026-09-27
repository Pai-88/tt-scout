"""Turn a metrics JSON (python -m tt_scout.metrics --json ...) into results-log rows for RESEARCH.md, so no number is
ever typed by hand.   Usage: python scripts/results_log.py results/metrics_v0_7clips.json [--label v0] [--append]
--append inserts the rows above the '<!-- results-log-end -->' marker in RESEARCH.md."""
import argparse, datetime, json, pathlib

ap = argparse.ArgumentParser()
ap.add_argument("metrics_json"); ap.add_argument("--label", default="v0"); ap.add_argument("--append", action="store_true")
ap.add_argument("--date", default=datetime.date.today().isoformat()); ap.add_argument("--pooled-only", action="store_true")
a = ap.parse_args()
d = json.load(open(a.metrics_json)); r, st = d["results"], d["settings"]
crit = f"{st['criterion']} " + (f"IoU>={st['iou']}" if st["criterion"] == "iou" else f"{st['tol_start']}/{st['tol_end']}")
rows = []
for name, res in r.items():
    if a.pooled_only and name != "pooled":
        continue
    p, l = res["points"], res["landing"]
    land = "-" if not l or not l.get("n") else f"{l['matched']}/{l['n_truth_bounces']} bounces, {l['median_cm']:.1f} cm [{l['median_ci'][0]:.1f}, {l['median_ci'][1]:.1f}]"
    rows.append(f"| {a.date} | {a.label} | {name} | {crit} | {p['tp']}/{p['fp']}/{p['fn']} (ignored {p['ignored']}) | "
                f"{p['winner_right']}/{p['winner_n']} [{p['winner_ci'][0]:.2f}, {p['winner_ci'][1]:.2f}] | {p['server_right']}/{p['server_n']} | "
                f"{p['start_err_median_s']:+.2f} / {p['end_err_median_s']:+.2f} | {land} |")
text = "\n".join(rows)
print(text)
if a.append:
    doc = pathlib.Path(__file__).resolve().parent.parent / "RESEARCH.md"
    s = doc.read_text(); marker = "<!-- results-log-end -->"
    assert marker in s, "RESEARCH.md needs the results-log-end marker"
    doc.write_text(s.replace(marker, text + "\n" + marker)); print(f"appended {len(rows)} rows to {doc.name}")
