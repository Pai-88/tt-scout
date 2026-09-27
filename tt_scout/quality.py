"""How much should this analysis be trusted? A product has to say so itself.

Two cheap signals, both available without ground truth, tracked the measured point-detection F1 almost monotonically over
the 12 OpenTTGames videos (2026-09-17, v1 logic):
    implied_frac   share of net crossings that were not detected and had to be inferred from the bounce sides
    coverage       share of frames inside the points in which the ball was seen
        F1 0.38  -> implied 0.79, coverage 0.32        F1 0.84-1.00 -> implied 0.07-0.25, coverage 0.59-0.88
Levels:  good        implied <= 0.30 and coverage >= 0.55      (measured F1 0.84-1.00)
         degraded    in between                                  (measured F1 0.74-0.84)
         unreliable  implied >= 0.70 or coverage < 0.40          (measured F1 <= 0.67)
"""
import numpy as np


def assess(points, track=None, fps=None):
    ncross = sum(p.get("n_crossings", 0) for p in points)
    nimp = sum(p.get("n_implied_crossings", 0) for p in points)
    implied = nimp / ncross if ncross else 1.0
    coverage = float(np.mean([p.get("coverage", 0.0) for p in points])) if points else 0.0
    reasons = []
    if not points:
        level = "unreliable"; reasons.append("no points were found: check the camera angle (side view, whole table in frame) and the ball colour")
    elif implied >= 0.70 or coverage < 0.40:
        level = "unreliable"
    elif implied <= 0.30 and coverage >= 0.55:
        level = "good"
    else:
        level = "degraded"
    if points and implied > 0.30:
        reasons.append(f"{implied:.0%} of net crossings were not seen and had to be inferred (good footage: under 30 %)")
    if points and coverage < 0.55:
        reasons.append(f"the ball was visible in {coverage:.0%} of the frames during points (good footage: over 55 %); dim light or a ball close in colour to the background")
    if fps and fps < 50:
        reasons.append(f"{fps:.0f} frames per second: fast shots are missed below 60; record at 60-120 fps"); level = "degraded" if level == "good" else level
    score = int(round(100 * (0.5 * min(1.0, coverage / 0.65) + 0.5 * (1.0 - min(1.0, implied)))))
    advice = {"good": "Point counts and placements can be used as they are.",
              "degraded": "Use placements and serve patterns; check point winners against the clips before relying on them.",
              "unreliable": "Do not rely on these numbers. Re-record with more light, a side view of the whole table and 60-120 fps."}[level]
    return dict(level=level, score=score, implied_crossing_frac=round(implied, 3), ball_coverage_in_points=round(coverage, 3),
                n_points=len(points), reasons=reasons, advice=advice)
