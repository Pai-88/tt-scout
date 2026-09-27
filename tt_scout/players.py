"""Who is at which end. Players are told apart by shirt colour, sampled once at calibration (a click on each
shirt). During analysis every player-sized blob is logged with its torso colour and which end it is on; per point,
each end's median colour is matched to the two stored players. When the ends swap between games, the names follow.
No faces, no learned model: hue, saturation and value of a t-shirt.
"""
import csv
import numpy as np
from .config import NET_X


def colour_distance(a, b):
    """a, b = (h, s, v) in OpenCV ranges. Hue is circular (0-179); it dominates, but grey shirts need s/v."""
    dh = abs(a[0] - b[0]); dh = min(dh, 180 - dh) / 90.0
    ds = abs(a[1] - b[1]) / 255.0; dv = abs(a[2] - b[2]) / 255.0
    sat = min(a[1], b[1]) / 255.0                  # low saturation: hue is meaningless, lean on s and v
    return (2.0 * sat + 0.3) * dh + 1.0 * ds + 0.6 * dv


def write_obs(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "t", "end", "cx", "cy", "area", "h", "s", "v"]); w.writerows(rows)


def load_obs(path):
    try:
        return [dict(frame=int(float(r["frame"])), t=float(r["t"]), end=r["end"], cx=float(r["cx"]), cy=float(r["cy"]),
                     area=float(r["area"]), h=float(r["h"]), s=float(r["s"]), v=float(r["v"]))
                for r in csv.DictReader(open(path))]
    except FileNotFoundError:
        return []


def auto_reference(obs, names, first_s=60.0, min_obs=30, min_distance=0.25):
    """Shirt colours with no click: the median torso colour seen at each end during the first minute becomes that end's
    player. Returns the same structure calibration writes, or None when there are too few observations or the two shirts
    are too alike to tell apart (names then stay fixed to the ends)."""
    if not obs:
        return None
    t0 = min(o["t"] for o in obs)
    ref = {}
    for end in ("near", "far"):
        rows = [o for o in obs if o["end"] == end and o["t"] - t0 <= first_s]
        if len(rows) < min_obs:
            return None
        ref[end] = tuple(float(v) for v in np.median([[o["h"], o["s"], o["v"]] for o in rows], axis=0))
    if colour_distance(ref["near"], ref["far"]) < min_distance:
        return None
    return {end: dict(name=names[end], hsv=list(ref[end])) for end in ("near", "far")}


def name_points(points, obs, table_players, default_names):
    """Sets point["near_player"], ["far_player"], ["server"], ["winner_name"], ["ends_swapped"].
    table_players: {"near": {"name", "hsv"}, "far": {...}} from calibration, or None -> fixed default names."""
    names = dict(default_names)                      # {"near": name, "far": name}
    if table_players and all(k in table_players for k in ("near", "far")):
        ref = {table_players[e]["name"]: tuple(table_players[e]["hsv"]) for e in ("near", "far")}
        n1, n2 = list(ref)
        for pt in points:
            sel = [o for o in obs if pt["start_t"] - 1.0 <= o["t"] <= pt["end_t"] + 1.0]
            med = {}
            for end in ("near", "far"):
                rows = [o for o in sel if o["end"] == end]
                if len(rows) >= 5:
                    med[end] = tuple(np.median([[o["h"], o["s"], o["v"]] for o in rows], axis=0))
            if len(med) == 2:
                straight = colour_distance(med["near"], ref[n1]) + colour_distance(med["far"], ref[n2])
                crossed = colour_distance(med["near"], ref[n2]) + colour_distance(med["far"], ref[n1])
                names = {"near": n1, "far": n2} if straight <= crossed else {"near": n2, "far": n1}
            elif len(med) == 1:
                end = next(iter(med)); other = "far" if end == "near" else "near"
                pick = min(ref, key=lambda nm: colour_distance(med[end], ref[nm]))
                names = {end: pick, other: n2 if pick == n1 else n1}
            pt["near_player"], pt["far_player"] = names["near"], names["far"]
            pt["ends_swapped"] = names["near"] != table_players["near"]["name"]
    else:
        for pt in points:
            pt["near_player"], pt["far_player"], pt["ends_swapped"] = names["near"], names["far"], False
    for pt in points:
        pt["server"] = pt["near_player"] if pt["serve_side"] == "near" else pt["far_player"]
        pt["winner_name"] = None if pt["winner"] is None else (pt["near_player"] if pt["winner"] == "near" else pt["far_player"])
    return points


def end_of(table, cx, cy):
    return "near" if table.to_table([(cx, cy)])[0][0] < NET_X else "far"
