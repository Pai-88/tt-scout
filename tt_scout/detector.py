"""Ball candidates from a fixed camera: background subtraction + size/shape/colour filters.

No learned model yet. With a fixed camera the ball is the smallest fast-moving thing in the frame, which
is enough to get started; a YOLO model fine-tuned on a few hundred labelled frames can replace this class
later without touching the rest of the pipeline (same detect() contract).
"""
import math
from dataclasses import dataclass
import numpy as np, cv2
from .table import TABLE_PTS


@dataclass
class Candidate:
    x: float
    y: float
    area: float
    score: float


COLOUR_RANGES = {  # HSV, OpenCV convention (H in 0..179)
    "orange": lambda h, s, v: (h >= 4) & (h <= 28) & (s >= 70) & (v >= 90),
    # "white" is loose on purpose: under gym lights over a blue table the ball reads as pale blue-grey,
    # and at the contact frame it picks up the table's blue (OpenTTGames: median HSV 90/85/174, but S up to
    # ~155 and V down to ~130 on bounce frames). The table itself is S > 190, so this still excludes it.
    "white": lambda h, s, v: (s <= 170) & (v >= 110),
}


class BallDetector:
    def __init__(self, cfg, table, fps=60.0):
        self.cfg = cfg
        self.mog = cv2.createBackgroundSubtractorMOG2(history=int(cfg.mog_history_s * fps),
                                                      varThreshold=cfg.mog_var_threshold, detectShadows=False)
        self.prev_gray = None
        radii = [table.ball_radius_px(p) for p in TABLE_PTS]
        self.area_min = cfg.area_min_frac * math.pi * min(radii) ** 2
        self.area_max = cfg.area_max_frac * math.pi * max(radii) ** 2
        self.area_ref = math.pi * float(np.mean(radii)) ** 2
        self.last_mask = None
        self.players = []          # per frame: player-sized blobs [{cx, cy, area, h, s, v}], torso colour

    def detect(self, frame):
        cfg = self.cfg
        fg = self.mog.apply(frame)
        fg = cv2.medianBlur(fg, 3)              # kills single-pixel noise, keeps a 3 px ball
        self.last_mask = fg
        n, lab, stats, cent = cv2.connectedComponentsWithStats(fg, connectivity=8)
        # "moving" = changed since the previous frame. Background uncovered by a moving player is
        # foreground for the background model but does not change frame to frame.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        moving = None
        if self.prev_gray is not None:
            moving = cv2.absdiff(gray, self.prev_gray) > cfg.diff_thresh
        self.prev_gray = gray
        self.moving = moving
        # anything player-sized: keep a margin around it (hands, shoes, rackets live there)
        big_ids = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > self.area_max]
        big = np.isin(lab, big_ids).astype(np.uint8)
        # who is it: the torso colour (middle band of the blob) identifies a player by shirt
        self.players = []
        hsv = None
        for i in sorted(big_ids, key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:3]:
            x, y, w, h, area = stats[i]
            if area < 4 * self.area_max:
                continue
            if hsv is None:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            y0, y1 = y + int(0.2 * h), y + int(0.55 * h)
            band = hsv[y0:y1, x:x + w][lab[y0:y1, x:x + w] == i]
            if band.shape[0] < 50:
                continue
            med = np.median(band, axis=0)
            self.players.append(dict(cx=float(cent[i][0]), cy=float(cent[i][1]), area=float(area),
                                     h=float(med[0]), s=float(med[1]), v=float(med[2])))
        if big.any():
            k = 2 * cfg.player_margin_px + 1
            big = cv2.dilate(big, np.ones((k, k), np.uint8))
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if not (self.area_min <= area <= self.area_max):
                continue
            if x == 0 or y == 0 or x + w >= fg.shape[1] or y + h >= fg.shape[0]:
                continue                                   # cut by the frame edge: a body, not a ball
            cx, cy = int(cent[i][0]), int(cent[i][1])
            if big[min(cy, big.shape[0] - 1), min(cx, big.shape[1] - 1)]:
                continue
            if moving is not None:
                comp = lab[y:y + h, x:x + w] == i
                if moving[y:y + h, x:x + w][comp].mean() < cfg.min_moving_frac:
                    continue
            if area / float(w * h) < cfg.min_fill:
                continue
            if max(w, h) / max(1.0, min(w, h)) > cfg.max_aspect:
                continue
            colour = 1.0
            if cfg.ball_colour in COLOUR_RANGES:
                if hsv is None:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                sub = hsv[y:y + h, x:x + w][lab[y:y + h, x:x + w] == i]
                m = COLOUR_RANGES[cfg.ball_colour](sub[:, 0], sub[:, 1], sub[:, 2])
                colour = float(m.mean())
                if colour < cfg.colour_min_frac:
                    continue
            size = math.exp(-abs(math.log(area / self.area_ref)))
            out.append(Candidate(float(cent[i][0]), float(cent[i][1]), float(area), colour * size))
        return out
