"""Classical versus learned detector, one figure, from the JSON written by tt_scout.ml.evaluate.
    python analysis/ml_results_figure.py results/ml_vs_classical_test4.json out_ml/results_test4.png "test_4 (5 min, camera setup not in training)"
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = json.load(open(sys.argv[1])); out = sys.argv[2]; title = sys.argv[3] if len(sys.argv) > 3 else ""
clips = list(res["ball"])
key = "pooled" if "pooled" in res["points"]["classical"]["results"] else clips[0]


def ball(det, part):
    rows = [res["ball"][c][det][part] for c in clips]
    n = sum(r["n"] for r in rows)
    return sum(r["within_10px"] * r["n"] for r in rows) / n, n


metrics = []
for label, part in (("ball within 10 px\n(all labelled frames)", "all"), ("ball within 10 px\n(last shot of each point)", "last_shot")):
    (c, n), (l, _) = ball("classical", part), ball("learned", part)
    metrics.append((label + f"\nn={n}", c, l))
for label, field, nf in (("point F1", "f1", "n_truth"), ("winner correct", "winner_acc", "winner_n"), ("server correct", "server_acc", "server_n")):
    pc, pl = res["points"]["classical"]["results"][key]["points"], res["points"]["learned"]["results"][key]["points"]
    metrics.append((f"{label}\nn={pc[nf]} vs {pl[nf]}", pc[field], pl[field]))

fig, ax = plt.subplots(figsize=(11, 4.6), dpi=150)
xs = np.arange(len(metrics)); w = 0.38
paper, ink, c_cls, c_ml = "#f6f1e6", "#1d2b25", "#d98a2b", "#2f7d4f"
fig.patch.set_facecolor(paper); ax.set_facecolor(paper)
b1 = ax.bar(xs - w / 2, [m[1] for m in metrics], w, color=c_cls, label="classical detector (motion + colour rules)")
b2 = ax.bar(xs + w / 2, [m[2] for m in metrics], w, color=c_ml, label="learned detector (BallNet)")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=9, color=ink)
ax.set_xticks(xs, [m[0] for m in metrics], fontsize=9, color=ink); ax.set_ylim(0, 1.12); ax.set_yticks([0, 0.5, 1.0])
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="lower left", fontsize=9); ax.set_title(title, loc="left", fontsize=12, color=ink)
fig.text(0.01, 0.01, "Same tracker, event detection and point logic on top of both detectors. Ball labels: OpenTTGames; points: Extended OpenTT Games.",
         fontsize=7.5, color="#5b6660")
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(out, facecolor=paper); print(out)
