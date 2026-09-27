"""One-page scouting report per match, player-centric: for each player, where their serves and third balls land
(table rotated so that player is always on the left), how their points ended; plus the match-wide heatmap and
point lengths. Names come from the shirt-colour matching in players.py, so a change of ends is handled."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .config import TABLE_LENGTH as L, TABLE_WIDTH as W

PCOL = ["tab:blue", "tab:red"]


def draw_table(ax, left="", right=""):
    ax.add_patch(Rectangle((0, 0), L, W, fill=False, lw=2))
    ax.plot([L / 2, L / 2], [-0.05, W + 0.05], "k--", lw=1)
    ax.plot([0, L], [W / 2, W / 2], color="0.75", lw=0.8)
    ax.set_xlim(-0.25, L + 0.25); ax.set_ylim(-0.3, W + 0.3); ax.set_aspect("equal")
    ax.set_xticks([0, L / 2, L]); ax.set_xticklabels([left, "net", right]); ax.set_yticks([])


def oriented(x, y, player_end):
    """Table coordinates as seen with the player at x=0: rotate 180 deg if they are at the far end."""
    return (x, y) if player_end == "near" else (L - x, W - y)


def player_end(pt, name):
    return "near" if pt.get("near_player") == name else "far"


def make_report(points, events, out_png, title=""):
    names = []
    for pt in points:
        for k in ("near_player", "far_player"):
            if pt.get(k) and pt[k] not in names:
                names.append(pt[k])
    names = (names + ["near", "far"])[:2]
    bounces = [e for e in events if e["kind"] == "bounce"]
    fig, ax = plt.subplots(3, 3, figsize=(18, 14))

    for r, name in enumerate(names):
        col = PCOL[r]
        mine = [pt for pt in points if name in (pt.get("near_player"), pt.get("far_player"))]
        served = [pt for pt in mine if pt.get("server") == name]
        # serves: landing of shot 1 on the opponent's side, oriented with this player on the left
        a = ax[r, 0]; draw_table(a, name, "opponent")
        pts = [oriented(l["x_m"], l["y_m"], player_end(pt, name)) for pt in served for l in pt["landings"] if l["shot"] == 1]
        if pts:
            a.scatter(*zip(*pts), s=40, c=col, alpha=0.8, edgecolor="k", linewidth=0.4)
        a.set_title(f"{name}: where the serve lands ({len(pts)})")
        # third ball: the server's first attack, landing of shot 3
        a = ax[r, 1]; draw_table(a, name, "opponent")
        pts = [oriented(l["x_m"], l["y_m"], player_end(pt, name)) for pt in served for l in pt["landings"] if l["shot"] == 3]
        if pts:
            a.scatter(*zip(*pts), s=40, c=col, alpha=0.8, edgecolor="k", linewidth=0.4)
        a.set_title(f"{name}: where the third ball lands ({len(pts)})")
        # receiving: where the opponent's serve lands on this player's side, and the return (shot 2)
        a = ax[r, 2]; draw_table(a, name, "opponent")
        recv = [pt for pt in mine if pt.get("server") and pt["server"] != name]
        p1 = [oriented(l["x_m"], l["y_m"], player_end(pt, name)) for pt in recv for l in pt["landings"] if l["shot"] == 1]
        p2 = [oriented(l["x_m"], l["y_m"], player_end(pt, name)) for pt in recv for l in pt["landings"] if l["shot"] == 2]
        if p1:
            a.scatter(*zip(*p1), s=40, c="0.5", alpha=0.7, edgecolor="k", linewidth=0.4, label=f"serves received ({len(p1)})")
        if p2:
            a.scatter(*zip(*p2), s=40, c=col, alpha=0.8, marker="^", edgecolor="k", linewidth=0.4, label=f"{name}'s return ({len(p2)})")
        if p1 or p2:
            a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8, frameon=False)
        a.set_title(f"{name}: receiving")

    a = ax[2, 0]; draw_table(a, "near end", "far end")
    if bounces:
        h = a.hist2d([e["x_m"] for e in bounces], [e["y_m"] for e in bounces], bins=[12, 6],
                     range=[[0, L], [0, W]], cmap="Reds", alpha=0.9)
        fig.colorbar(h[3], ax=a, fraction=0.03); draw_table(a, "near end", "far end")
    a.set_title(f"All bounces ({len(bounces)}), table coordinates")

    a = ax[2, 1]
    lens = [p["n_crossings"] for p in points]
    if lens:
        a.hist(lens, bins=range(1, max(lens) + 2), align="left", rwidth=0.8, color="0.4")
    a.set_xlabel("shots per point (net crossings)"); a.set_ylabel("points"); a.set_title("Point length")

    a = ax[2, 2]; a.axis("off")
    lines = [title, "", f"points detected: {len(points)}", f"mean shots per point: {np.mean(lens):.1f}" if lens else "", ""]
    lines.append(f"{'':12s}{'won':>5s}{'served':>8s}{'won on serve':>14s}{'won receiving':>15s}")
    for name in names:
        won = sum(1 for p in points if p.get("winner_name") == name)
        srv = sum(1 for p in points if p.get("server") == name)
        wos = sum(1 for p in points if p.get("winner_name") == name and p.get("server") == name)
        lines.append(f"{name[:12]:12s}{won:5d}{srv:8d}{wos:14d}{won - wos:15d}")
    unk = sum(1 for p in points if p.get("winner_name") is None)
    if unk:
        lines.append(f"{'unknown':12s}{unk:5d}")
    swaps = sum(1 for i in range(1, len(points)) if points[i].get("near_player") != points[i - 1].get("near_player"))
    lines += ["", f"changes of ends detected: {swaps}", "", "how points ended:"]
    endings = {}
    for p in points:
        key = f"{p.get('ending') or 'unknown'} -> {p.get('winner_name')}"
        endings[key] = endings.get(key, 0) + 1
    lines += [f"  {k:32s}{v}" for k, v in sorted(endings.items(), key=lambda kv: -kv[1])]
    lines += ["", "not_returned = landed, no return over the net", "long = crossed the net, never landed",
              "double_bounce = twice on the receiver's side"]
    a.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=10)
    fig.suptitle("tt_scout scouting report", fontsize=13)
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)
    return out_png
