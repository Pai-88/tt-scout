"""Draw what the tracker knows onto the video of one point, comic-book style.

Live: table outline, a ball trail COLOURED BY SPEED (with a scale), bounce rings, net-crossing ticks, one name tag per end, a
landing map, a caption (server, shots, last shot speed) and a running SCOREBOARD. When the point is won: a white flash, a camera
punch, a jagged starburst with halftone dots and slanted outlined lettering, impact lines, a narration box, and the winner's
number on the scoreboard punching up. Lettering is rendered with Pillow (OpenCV's Hershey fonts are too thin for it). Frames go
straight to ffmpeg as H.264 (OpenCV's own mp4v files do not play in Chrome or Safari).

A clip that carries a TIP (tips.py) opens earlier, in the lull before the serve: the note slides up along the bottom edge, its fuse
burns down while it is read, and it is gone again about when the ball is struck, so the rally itself plays unobstructed.

The scoreboard is tt_scout's OWN count from the start of the recording (games split at each change of ends); points whose winner
is unsure are not counted, so it can differ from the umpire's board in the picture.

    render_point_clip(video, fps, table, track, events, obs_by_frame, point, dst, tip=None)     # point["_score"] from running_scores()
"""
import functools, math, pathlib, shutil, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from .config import TABLE_LENGTH as L, TABLE_WIDTH as W, NET_X
from . import heads as heads_mod
from .flight import DASH

GREEN, MAGENTA, WHITE, DARK = (80, 230, 80), (255, 80, 255), (190, 246, 255), (96, 32, 34)   # BGR; WHITE/DARK are comic tones, not white/black
INK_BGR, PAPER_BGR, YELLOW_BGR, RED_BGR, BLUE_BGR = (96, 32, 34), (190, 246, 255), (51, 221, 255), (38, 38, 228), (205, 95, 35)
NEAR_COL, FAR_COL = (60, 170, 255), (255, 200, 60)             # BGR: shots / tags of the near (left) and far (right) player
NEAR_RGB, FAR_RGB = (255, 170, 60), (60, 200, 255)
INK, PAPER, YELLOW_RGB, RED_RGB, HALFTONE_RGB = (34, 32, 96), (255, 246, 190), (255, 221, 51), (228, 38, 38), (255, 138, 0)
# ink is a deep indigo and paper a pale yellow: no pure black or white anywhere in the overlays (2026-09-26)
# ball speed along its path, m/s -> BGR. Cool and pale when slow (readable on a blue hall), yellow, orange, red, hot pink when fast
SPEED_STOPS = [(0.0, (255, 235, 150)), (4.0, (170, 255, 120)), (8.0, (0, 235, 255)), (12.0, (0, 150, 255)), (16.0, (50, 50, 255)), (22.0, (190, 70, 255))]
FONTS = ["/System/Library/Fonts/Supplemental/Impact.ttf", "/System/Library/Fonts/Supplemental/Arial Black.ttf", "C:/Windows/Fonts/impact.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]


def speed_colour(v):
    v = float(np.clip(v, SPEED_STOPS[0][0], SPEED_STOPS[-1][0]))
    for (a, ca), (b, cb) in zip(SPEED_STOPS, SPEED_STOPS[1:]):
        if v <= b:
            f = (v - a) / (b - a)
            return tuple(int(round(ca[i] + f * (cb[i] - ca[i]))) for i in range(3))
    return SPEED_STOPS[-1][1]


@functools.lru_cache(maxsize=32)
def _font(size):
    for f in FONTS:
        if pathlib.Path(f).exists():
            return ImageFont.truetype(f, size)
    try:
        import matplotlib.font_manager as fm
        return ImageFont.truetype(fm.findfont("DejaVu Sans:bold"), size)
    except Exception:
        return ImageFont.load_default()


def _bgra(im):
    a = np.array(im.convert("RGBA"))
    return np.ascontiguousarray(a[..., [2, 1, 0, 3]])


def _blend(img, spr, cx, cy, scale=1.0, angle=0.0, alpha=1.0):
    """Alpha-composite a BGRA sprite centred at (cx, cy), scaled and rotated."""
    if scale <= 0.02 or alpha <= 0.01:
        return
    h, w = spr.shape[:2]
    if scale != 1.0 or angle:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        nw, nh = int(w * max(scale, 1.0) * 1.25), int(h * max(scale, 1.0) * 1.25)
        M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
        spr = cv2.warpAffine(spr, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        h, w = nh, nw
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(img.shape[1], x0 + w), min(img.shape[0], y0 + h)
    if X1 <= X0 or Y1 <= Y0:
        return
    s = spr[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    a = (s[..., 3:4].astype(np.float32) / 255.0) * alpha
    roi = img[Y0:Y1, X0:X1]
    roi[:] = (roi * (1 - a) + s[..., :3] * a).astype(np.uint8)


def _stroked(draw, xy, text, font, fill, outer=9, inner=4, anchor="mm"):
    """Comic lettering: fat black outline, white keyline, coloured fill."""
    draw.text((xy[0] + 5, xy[1] + 6), text, font=font, fill=INK, stroke_width=outer, stroke_fill=INK, anchor=anchor)     # dropped shadow
    draw.text(xy, text, font=font, fill=INK, stroke_width=outer, stroke_fill=INK, anchor=anchor)
    draw.text(xy, text, font=font, fill=fill, stroke_width=inner, stroke_fill=PAPER, anchor=anchor)


LETTER_FONTS = ["/System/Library/Fonts/Supplemental/Comic Sans MS Bold.ttf", "/System/Library/Fonts/MarkerFelt.ttc", "C:/Windows/Fonts/comicbd.ttf"]


@functools.lru_cache(maxsize=16)
def _letter_font(size):
    """Comic-book lettering for captions (the narration-box hand), falling back to the block face."""
    for f in LETTER_FONTS:
        if pathlib.Path(f).exists():
            return ImageFont.truetype(f, size)
    return _font(size)


def comic_panel(img, seed=0):
    """Turn a video frame into a printed comic panel: flat posterised colour, black ink contours, Ben-Day dots in the shadows,
    the red plate slightly off-register, warm paper, a little grain. BGR uint8 in and out."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    for _ in range(3):
        small = cv2.bilateralFilter(small, 9, 55, 55)                  # melt texture, keep edges
    base = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[..., 2] / 255.0
    lo, hi = np.percentile(v, 3), np.percentile(v, 99)                  # halls are dim: stretch the exposure so that the dots
    v = np.clip((v - lo) / max(hi - lo, 1e-3), 0, 1) ** 0.72            # land in real shadows, not on the whole picture
    hsv[..., 0] = (np.round(hsv[..., 0] / 7.5) * 7.5) % 180            # 24 hues
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.35 + 22, 0, 255)              # process-colour saturation
    hsv[..., 2] = np.clip((np.floor(v * 5) + 0.7) / 5 * 255, 0, 255)    # five flat tones
    flat = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)                      # a 45-degree dot screen, 6 px pitch
    c = math.sqrt(0.5) / 6.0
    screen = 0.5 - 0.25 * (np.cos(2 * np.pi * (xx + yy) * c) + np.cos(2 * np.pi * (yy - xx) * c))
    dots = ((1.0 - v) * 1.35 - 0.62) > screen                           # bigger dots where it is darker; none in the light tones
    flat[dots] *= 0.55
    gray = cv2.medianBlur(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY), 5)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 9)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink)
    keep = np.isin(lab, [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 36])     # drop specks, keep drawn lines
    ink = cv2.dilate((keep * 255).astype(np.uint8), np.ones((2, 2), np.uint8))
    flat[(ink > 0) | (v < 0.07)] = INK_BGR                               # contours and solid darks, in indigo ink
    out = flat.copy()
    out[1:, 2:, 2] = flat[:-1, :-2, 2]                                   # red plate printed a hair off
    out *= np.array([0.86, 0.95, 0.985], np.float32)                     # warm newsprint (BGR)
    out += np.random.default_rng(seed).normal(0, 3.5, out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _letter(ch, size, rng, max_rot=7.0):
    """One hand-lettered block letter: 3-D ink extrusion, fat contour, yellow-to-orange fill with red Ben-Day dots low down."""
    f = _font(size); pad = int(size * 0.34)
    wch = int(f.getlength(ch)) + 2 * pad; hch = int(size * 1.25) + 2 * pad
    ext = max(5, size // 11); stroke = max(4, size // 15)
    im = Image.new("RGBA", (wch + ext, hch + ext), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for k in range(ext, 0, -1):                                          # extrusion, down and to the right
        d.text((pad + k, pad + k), ch, font=f, fill=INK, stroke_width=stroke, stroke_fill=INK)
    d.text((pad, pad), ch, font=f, fill=INK, stroke_width=stroke, stroke_fill=INK)
    face = Image.new("L", im.size, 0); ImageDraw.Draw(face).text((pad, pad), ch, font=f, fill=255)
    fm = np.array(face) > 128
    arr = np.array(im); hh = arr.shape[0]
    grad = np.linspace(0, 1, hh)[:, None]
    col = np.stack([np.full_like(grad, 255), 236 - 110 * grad, 40 - 30 * grad], axis=-1)               # yellow above, orange below (RGB)
    col = np.broadcast_to(col, (hh, arr.shape[1], 3)).copy()
    yy, xx = np.mgrid[0:hh, 0:arr.shape[1]]
    dot = (((xx + yy) % 14 < 6) & ((xx - yy) % 14 < 6)) & (yy > pad + size * 0.62)
    col[dot] = RED_RGB
    arr[fm, :3] = col[fm]; arr[fm, 3] = 255
    hi = np.roll(fm, 3, axis=0) & fm & ~np.roll(fm, 6, axis=0)           # a white highlight along the upper edges
    out = Image.fromarray(arr)
    return out.rotate(float(rng.uniform(-max_rot, max_rot)), resample=Image.BICUBIC, expand=True)


def burst_sprite(word, seed=0, w=760, h=400, palette="classic"):
    """An uneven ink explosion in two plates (red behind, yellow in front, red dots by default; match_stats.PALETTES) with hand-lettered
    block letters on an arc."""
    from .match_stats import PALETTES
    P_OUT, P_IN, P_DOT = PALETTES.get(palette, PALETTES["classic"])
    rng = np.random.default_rng(seed)
    cx, cy = w / 2, h / 2
    def blast(n, rmin, rmax, inner, jitter):
        ang = np.sort(np.linspace(0, 2 * np.pi, 2 * n, endpoint=False) + rng.uniform(-jitter, jitter, 2 * n))
        rad = np.where(np.arange(2 * n) % 2 == 0, rng.uniform(rmin, rmax, 2 * n), rng.uniform(inner - 0.06, inner + 0.05, 2 * n))
        return [(cx + math.cos(a) * r * (w * 0.40), cy + math.sin(a) * r * (h / 2 - 12)) for a, r in zip(ang, rad)]
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    outer = blast(13, 0.74, 1.0, 0.58, 0.10)
    d.polygon([(x + 7, y + 8) for x, y in outer], fill=INK + (255,))
    d.polygon(outer, fill=P_OUT + (255,)); d.line(outer + [outer[0]], fill=INK + (255,), width=6, joint="curve")
    inner = [(cx + (x - cx) * 0.80 - 4, cy + (y - cy) * 0.80 - 3) for x, y in blast(11, 0.78, 1.0, 0.60, 0.12)]
    d.polygon(inner, fill=P_IN + (255,))
    mask = Image.new("L", (w, h), 0); ImageDraw.Draw(mask).polygon(inner, fill=255); m = np.array(mask)
    for yy in range(4, h, 9):                                            # Ben-Day dots: one size, thinning toward the middle
        for xx in range(4 + (4 if (yy // 9) % 2 else 0), w, 9):
            if m[yy, xx] and math.hypot((xx - cx) / (w / 2), (yy - cy) / (h / 2)) > rng.uniform(0.30, 0.42):
                d.ellipse((xx - 2.6, yy - 2.6, xx + 2.6, yy + 2.6), fill=P_DOT + (255,))
    d.line(inner + [inner[0]], fill=INK + (255,), width=5, joint="curve")
    size = 170
    while size > 50 and sum(_font(size).getlength(ch) for ch in word) * 0.93 > 0.86 * w:
        size -= 6
    adv = [_font(size).getlength(ch) * 0.93 for ch in word]             # advances come from the glyphs, not from the rotated images
    x = cx - sum(adv) / 2
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for i, (ch, a) in enumerate(zip(word, adv)):
        if ch != " ":
            l = _letter(ch, int(size * rng.uniform(0.94, 1.07)), rng)
            arc = -0.10 * size * math.sin(math.pi * (i + 0.5) / len(word)) + rng.uniform(-0.035, 0.035) * size
            dx, dy = int(x + a / 2 - l.size[0] / 2), int(cy - l.size[1] / 2 + arc)
            big = Image.new("RGBA", (w, h), (0, 0, 0, 0)); big.paste(l, (dx, dy), l)          # paste clips instead of raising at the edges
            layer = Image.alpha_composite(layer, big)
        x += a
    im.alpha_composite(layer.rotate(5, resample=Image.BICUBIC, center=(cx, cy)))
    return _bgra(im)


def caption_sprite(line1, line2, w=420, h=84):
    """A narration box as the letterer draws it: pale yellow, black keyline, ink shadow, slanted caps."""
    im = Image.new("RGBA", (w + 14, h + 14), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rectangle((9, 9, w + 9, h + 9), fill=INK + (255,))
    d.rectangle((0, 0, w, h), fill=(255, 238, 130, 255), outline=INK + (255,), width=4)
    size = 34
    while size > 16 and _letter_font(size).getlength(line1) > w - 40:
        size -= 2
    d.text((18, 9), line1, font=_letter_font(size), fill=INK)
    d.text((19, h - 30), line2, font=_letter_font(15), fill=(120, 20, 20))
    im = im.transform(im.size, Image.AFFINE, (1, 0.14, -0.14 * h / 2, 0, 1, 0), resample=Image.BICUBIC)       # the letterer's slant
    return _bgra(im.rotate(1.6, resample=Image.BICUBIC, expand=True))


def _overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def place(size, forbidden, prefer, frame, scales=(1.0,), step=20, inset=0.86):
    """Best (cx, cy, scale) for a sprite of `size` (w, h): least weighted overlap with the forbidden boxes [(x0, y0, x1, y1, weight)],
    then nearest to `prefer`, then largest. Players, the ball and the furniture are in `forbidden`, so nothing lands on them."""
    W_, H_ = frame; best = None
    for sc in scales:
        w, h = size[0] * sc, size[1] * sc
        for cy in np.arange(h / 2 + 14, H_ - h / 2 - 14 + 1, step):
            for cx in np.arange(w / 2 + 14, W_ - w / 2 - 14 + 1, step):
                r = (cx - w * inset / 2, cy - h * inset / 2, cx + w * inset / 2, cy + h * inset / 2)
                cost = sum(_overlap(r, fb[:4]) * fb[4] for fb in forbidden) / (w * h)
                cost += 0.0009 * abs(cx - prefer[0]) + 0.0016 * abs(cy - prefer[1]) - 0.55 * sc
                if best is None or cost < best[0]:
                    best = (cost, float(cx), float(cy), sc)
    return best[1:] if best else (prefer[0], prefer[1], scales[-1])


def player_boxes(obs_by_frame, f, s, span=8):
    """Generous boxes around the two players near frame f, in output pixels: [(x0, y0, x1, y1)]."""
    out = []
    for end in ("near", "far"):
        blobs = [o for k in range(f - span, f + span + 1) for o in obs_by_frame.get(k, []) if o["end"] == end]
        if blobs:
            o = max(blobs, key=lambda b: b["area"])
            wb = math.sqrt(o["area"] / (0.55 * 2.6)); hb = 2.6 * wb
            out.append(((o["cx"] - 0.95 * wb) * s, (o["cy"] - 0.70 * hb) * s, (o["cx"] + 0.95 * wb) * s, (o["cy"] + 0.80 * hb) * s))
    return out


def scoreboard_sprite(names, score, games, server_idx, hot=None, w=340, h=78):
    """The running score as a comic panel: two slanted colour plates with Ben-Day dots, a lightning divider, block numerals with an
    ink extrusion, lettered names, a ball with speed ticks on the server; `hot` = index whose number just went up (it gets a burst)."""
    W2, H2 = w + 26, h + 40
    im = Image.new("RGBA", (W2, H2), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    x0, y0, sl = 8, 6, 14                                               # sl = slant of the panel's sides
    outer = [(x0 + sl, y0), (x0 + w, y0), (x0 + w - sl, y0 + h), (x0, y0 + h)]
    d.polygon([(x + 8, y + 9) for x, y in outer], fill=INK + (255,))    # ink shadow
    mid_t, mid_b = x0 + w / 2 + sl / 2, x0 + w / 2 - sl / 2
    zig = [(mid_t, y0), (mid_t - 9, y0 + h * 0.34), (mid_t + 7, y0 + h * 0.44), (mid_b - 6, y0 + h * 0.74), (mid_b, y0 + h)]      # lightning divider
    left = [outer[0]] + zig + [outer[3]]
    right = list(reversed(zig)) + [outer[1], outer[2]]
    right = [zig[0], outer[1], outer[2], zig[-1]] + list(reversed(zig[1:-1]))
    for poly, col, idx in ((left, NEAR_RGB, 0), (right, FAR_RGB, 1)):
        d.polygon(poly, fill=(YELLOW_RGB if hot == idx else col) + (255,))
        mask = Image.new("L", (W2, H2), 0); ImageDraw.Draw(mask).polygon(poly, fill=255); m = np.array(mask)
        dot_col = RED_RGB if (hot == idx or idx == 0) else (20, 110, 190)
        for yy in range(y0 + 4, y0 + h, 7):                              # Ben-Day dots thinning toward the middle of the board
            for xx in range(x0 + 3 + (3 if (yy // 7) % 2 else 0), x0 + w, 7):
                edge = abs(xx - (x0 + w / 2)) / (w / 2)
                if m[min(yy, H2 - 1), min(xx, W2 - 1)] and edge > 0.52:
                    rr = 0.6 + 2.0 * (edge - 0.52) / 0.48
                    d.ellipse((xx - rr, yy - rr, xx + rr, yy + rr), fill=dot_col + (255,))
    d.line(zig, fill=INK + (255,), width=6, joint="curve"); d.line(zig, fill=PAPER + (255,), width=2, joint="curve")
    d.line(outer + [outer[0]], fill=INK + (255,), width=5, joint="curve")
    rng = np.random.default_rng(11)
    for i, nm in enumerate(names):
        cxn = x0 + w / 2 + (-46 if i == 0 else 46)
        if hot == i:                                                     # a small burst behind the number that just went up
            n = 10; ang = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
            rad = np.where(np.arange(2 * n) % 2 == 0, 40, 25)
            star = [(cxn + math.cos(a_) * r_, y0 + h / 2 - 2 + math.sin(a_) * r_ * 0.86) for a_, r_ in zip(ang, rad)]
            d.polygon(star, fill=RED_RGB + (255,)); d.line(star + [star[0]], fill=INK + (255,), width=3, joint="curve")
        num = str(score.get(nm, 0)); size = 50 if hot == i else 44
        xs = cxn - sum(_font(size).getlength(ch) for ch in num) * 0.47
        for ch in num:
            l = _letter(ch, size, rng, max_rot=4.0 if hot == i else 2.0)
            adv = _font(size).getlength(ch) * 0.94
            big = Image.new("RGBA", (W2, H2), (0, 0, 0, 0)); big.paste(l, (int(xs + adv / 2 - l.size[0] / 2), int(y0 + h / 2 - 3 - l.size[1] / 2)), l)
            im = Image.alpha_composite(im, big); d = ImageDraw.Draw(im); xs += adv
        label = nm.upper()[:8]; fs = 19
        while fs > 11 and _letter_font(fs).getlength(label) > w / 2 - 92:
            fs -= 1
        lx = x0 + sl + 12 if i == 0 else x0 + w - sl - 12
        d.text((lx, y0 + h / 2 - 4), label, font=_letter_font(fs), fill=INK, anchor="lm" if i == 0 else "rm", stroke_width=2, stroke_fill=PAPER)
        if server_idx == i:                                              # the ball, with speed ticks, under the server's name
            bx = lx + 8 if i == 0 else lx - 8; byy = y0 + h - 15
            d.ellipse((bx - 6, byy - 6, bx + 6, byy + 6), fill=PAPER, outline=INK, width=2)
            for k in (-1, 0, 1):
                tx = bx + (13 if i == 0 else -13)
                d.line((tx, byy + k * 4, tx + (9 if i == 0 else -9), byy + k * 4), fill=INK, width=2)
    strip = f"GAMES {games.get(names[0], 0)}–{games.get(names[1], 0)}   ·   COUNT BY TT_SCOUT"
    tw = _letter_font(11).getlength(strip)
    d.rectangle((W2 / 2 - tw / 2 - 10, y0 + h + 4, W2 / 2 + tw / 2 + 10, y0 + h + 23), fill=(255, 238, 130, 255), outline=INK + (255,), width=3)
    d.text((W2 / 2, y0 + h + 13), strip, font=_letter_font(11), fill=INK, anchor="mm")
    return _bgra(im.rotate(-1.6, resample=Image.BICUBIC, expand=True))


def running_scores(points):
    """point id -> dict(before, after, games, games_after): tt_scout's own count by player name, in games to 11 won by 2 (a change of
    ends also closes a game); points with no winner are not counted. See match_stats.games_of."""
    from .match_stats import games_of
    return games_of(points)[0]


def action_word(point, last_speed):
    if point.get("ending") == "long":
        return "OUT!"
    if point.get("n_crossings", 0) >= 8:
        return "EPIC RALLY!"
    if point.get("ending") == "double_bounce":
        return "TOO GOOD!"
    if _num(last_speed) and last_speed >= 9.0:
        return "SMASH!"
    return "POINT!"


def _box(img, x, y, w, h, alpha=0.55):
    x, y = max(0, x), max(0, y)
    roi = img[y:y + h, x:x + w]
    if roi.size:
        roi[:] = (roi * (1 - alpha) + np.array(DARK) * alpha).astype(np.uint8)


def _text(img, s, x, y, scale=0.55, col=WHITE, thick=1):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, INK_BGR, thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


PAPER_BOX = (255, 238, 150)


def _panel(w, h, fill=PAPER_BOX, shadow=6, keyline=4):
    """A caption box as the letterer rules it: flat fill, black keyline, ink shadow down and to the right."""
    im = Image.new("RGBA", (w + shadow + 2, h + shadow + 2), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rectangle((shadow, shadow, w + shadow, h + shadow), fill=INK + (255,))
    d.rectangle((0, 0, w, h), fill=fill + (255,), outline=INK + (255,), width=keyline)
    return im, d


def _slant(im, k=0.12):
    return im.transform(im.size, Image.AFFINE, (1, k, -k * im.size[1] / 2, 0, 1, 0), resample=Image.BICUBIC)


@functools.lru_cache(maxsize=512)
def hud_sprite(point_id, server, shots):
    """Top left: POINT 16 on a red tab, then who serves and the shots over the net, all hand-lettered caps. (The speed is in the gauge.)"""
    w, h = 278, 66
    im, d = _panel(w, h)
    tab = f"POINT {point_id}"; tw = int(_font(24).getlength(tab)) + 22
    d.rectangle((0, 0, tw, 32), fill=RED_RGB + (255,), outline=INK + (255,), width=4)
    d.text((tw / 2, 16), tab, font=_font(24), fill=PAPER, anchor="mm", stroke_width=2, stroke_fill=INK)
    line = f"{str(server).upper()[:10]} SERVES"
    fs = 19
    while fs > 12 and _letter_font(fs).getlength(line) > w - tw - 18:
        fs -= 1
    d.text((tw + 10, 16), line, font=_letter_font(fs), fill=INK, anchor="lm")
    low = f"{shots} SHOT{'' if shots == 1 else 'S'} OVER THE NET"
    fs = 15
    while fs > 10 and _letter_font(fs).getlength(low) > w - 22:
        fs -= 1
    d.text((12, 49), low, font=_letter_font(fs), fill=(120, 20, 20), anchor="lm")
    return _bgra(_slant(im, 0.10).rotate(1.2, resample=Image.BICUBIC, expand=True))


SAFE_BOTTOM = 58                                                        # px of the 540 kept clear along the bottom edge: a browser draws its
#                                                                         video controls there whenever the clip is paused, hovered or over
GAUGE_W, GAUGE_H, GAUGE_BAR = 268, 68, (14, 29, 158, 13)                # box width, height; bar x, y, width, height inside the box
GAUGE_NUM = (230, 34)                                                   # centre of the LAST SHOT numeral inside the box
HOLD_S, NEEDLE_S = 0.35, 0.15                                            # the number stays at least this long; the needle's easing time
SHOT_V_MIN, SHOT_V_MAX = 2.5, 35.0                                      # a last-shot speed outside this is not a shot: a bounce right at
#                                                                         the net gives a tiny one, and the fastest smash ever measured is
#                                                                         about 31 m/s, so anything above 35 pairs a crossing with someone
#                                                                         else's bounce. The reading is dropped, not clipped.


def last_shot_speed(landed, crossed, previous):
    """The last shot's speed in m/s: from where it crossed the net to where it bounced, over the time between. `previous` is kept when
    the newest bounce cannot belong to the newest crossing (it came first, or more than a second later), and when the answer is outside
    what a shot can be, because then the two events belong to different shots."""
    if not landed or not crossed or not 0 < landed[-1]["t"] - crossed[-1]["t"] < 1.0:
        return previous
    sp = abs(landed[-1]["x_m"] - NET_X) / (landed[-1]["t"] - crossed[-1]["t"])
    return sp if SHOT_V_MIN <= sp <= SHOT_V_MAX else previous


class Held:
    """A number the eye can read: it changes at most once every `hold` seconds. A newer value waits and the latest one is shown when the
    time is up; a new value that prints the same digits changes nothing. `changed` = when the printed number last changed."""

    def __init__(self, hold=HOLD_S):
        self.hold, self.value, self.pending, self.changed = hold, None, None, -1e9

    def update(self, t, value):
        """value: a speed (m/s), flight.DASH for a shot that was not measured, or None (no news: keep what is shown). Two values
        are the same when they print the same (reading_text: km/h, ~ for an estimate)."""
        if value is not None:
            self.pending = None if (self.value is not None and reading_text(value) == reading_text(self.value)) else value
        if self.pending is not None and t - self.changed >= self.hold:
            self.value, self.pending, self.changed = self.pending, None, t
        return self.value


@functools.lru_cache(maxsize=2)
def gauge_sprite():
    """Bottom left, above the safe margin. Left column: BALL SPEED and the colour ramp ruled in ink, dots over its hot end, lettered ticks;
    it carries the live needle. Right column, behind an ink rule: LAST SHOT, a space for the numeral (drawn per frame), and the unit.
    The numeral is the last shot's speed off the racket in km/h, held so that it can be read."""
    w, h = GAUGE_W, GAUGE_H
    im, d = _panel(w, h)
    bx, by, bw, bh = GAUGE_BAR
    d.text((bx, 12), "BALL SPEED", font=_letter_font(14), fill=INK, anchor="lm")
    for x in range(bw):
        b_, g_, r_ = speed_colour(SPEED_STOPS[-1][0] * x / (bw - 1))
        d.line((bx + x, by, bx + x, by + bh), fill=(r_, g_, b_, 255))
    for yy in range(by + 3, by + bh, 5):                                 # Ben-Day dots thickening toward the fast end
        for xx in range(bx + bw // 2 + (2 if (yy // 5) % 2 else 0), bx + bw, 5):
            rr = 0.5 + 1.5 * (xx - bx - bw / 2) / (bw / 2)
            d.ellipse((xx - rr, yy - rr, xx + rr, yy + rr), fill=(120, 0, 30, 255))
    d.rectangle((bx - 1, by - 1, bx + bw, by + bh), outline=INK + (255,), width=3)
    top = SPEED_STOPS[-1][0]
    for kmh in (0, 40, 80):                                              # the scale in km/h (the colours stay on m/s)
        x = bx + min(bw, int(round(bw * kmh / 3.6 / top)))
        d.line((x, by + bh, x, by + bh + 4), fill=INK + (255,), width=2)
        d.text((x, by + bh + 12), f"{kmh}", font=_letter_font(11), fill=INK, anchor="mm")
    rule = bx + bw + 16
    d.line((rule, 7, rule, h - 7), fill=INK + (255,), width=2)
    cx = GAUGE_NUM[0]
    d.text((cx, 11), "LAST SHOT", font=_letter_font(10), fill=(120, 20, 20), anchor="mm")
    d.text((cx, h - 10), "KM/H", font=_letter_font(11), fill=INK, anchor="mm")
    return _bgra(im)


def reading_text(v):
    """What the gauge prints for a reading: km/h, with ~ for an estimate; DASH as it is."""
    if isinstance(v, str):
        return v
    return ("~" if getattr(v, "approx", False) else "") + f"{3.6 * v:.0f}"


def reading_sprite(v):
    """The gauge's numeral: the speed in km/h in its colour on the speed scale (~ for an estimate), or a grey dash for a shot that
    was played but not measured."""
    if isinstance(v, str):
        return dash_sprite()
    bgr = speed_colour(v)
    return readout_sprite(reading_text(v), (bgr[2], bgr[1], bgr[0]))


@functools.lru_cache(maxsize=1)
def dash_sprite():
    """The gauge's mark for a shot that was played but not measured: a flat bar the width of a numeral (a '-' in the block face
    prints as a blob)."""
    w, h = 50, 46
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((8, 19, w - 8, 30), radius=4, fill=INK + (255,))
    d.rounded_rectangle((11, 22, w - 11, 27), radius=2, fill=(200, 198, 190, 255))
    return _bgra(im)


def _num(v):
    """A speed that can be compared, or None (flight.DASH is not a number)."""
    return None if (v is None or isinstance(v, str)) else v


@functools.lru_cache(maxsize=128)
def readout_sprite(txt, rgb):
    """A speed in block numerals, filled with its colour on the speed scale."""
    f = _font({1: 30, 2: 30, 3: 26}.get(len(txt), 22)); w = int(f.getlength(txt)) + 22
    im = Image.new("RGBA", (w, 46), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.text((w / 2 + 3, 26), txt, font=f, fill=INK, anchor="mm", stroke_width=4, stroke_fill=INK)
    d.text((w / 2, 23), txt, font=f, fill=rgb, anchor="mm", stroke_width=3, stroke_fill=INK)
    return _bgra(im)


MAP_W, MAP_H = 190, int(190 * W / L)
MAP_X0, MAP_Y0 = 12, 28                                                 # the table drawing's top-left corner inside the full panel
MAP_PAD, MAP_MIN_H = 8, 56                                              # compact panel: ink margin round the table; smallest table drawn
MAP_FULL_BOTTOM = 10 + MAP_Y0 + MAP_H + 24 + 8                          # where the full panel ends, shadow included, placed 10 px down


def map_frame(compact_th=0):
    """(x0, y0, width, height) of the table drawing inside the landing-map sprite."""
    if compact_th:
        return MAP_PAD, MAP_PAD, int(round(compact_th * L / W)), int(compact_th)
    return MAP_X0, MAP_Y0, MAP_W, MAP_H


def map_compact_height(head_tops, map_left, full_bottom=MAP_FULL_BOTTOM, min_share=0.08):
    """0 for the full panel, else the table height of a compact one. head_tops = [(x, top y)] of the right-hand player's head sampled over
    the match. When heads under the panel's columns reach above its bottom often enough, the panel drops its title and names (the same
    colours name the players on the scoreboard and balloons) and shrinks to end above the highest tenth of those heads."""
    under = [top for x, top in head_tops if x >= map_left - 12]
    if not head_tops or not under:
        return 0
    high = [top for top in under if top < full_bottom + 6]
    if len(high) < max(2, min_share * len(head_tops)):
        return 0
    top10 = float(np.percentile(under, 10))
    return int(np.clip(top10 - 6 - 10 - 2 * MAP_PAD - 8, MAP_MIN_H, MAP_H))


@functools.lru_cache(maxsize=32)
def map_sprite(near_name, far_name, compact_th=0):
    """Top right: WHERE IT LANDED. The table from above in flat process blue, ruled in ink; names lettered in the player colours. With
    compact_th (see map_compact_height) it is only the table, ruled in ink on a paper margin, compact_th px tall."""
    x0, y0, tw, th = map_frame(compact_th)
    if compact_th:
        im, d = _panel(tw + 2 * MAP_PAD, th + 2 * MAP_PAD, keyline=3)
    else:
        w, h = MAP_W + 2 * MAP_X0, MAP_Y0 + MAP_H + 24
        im, d = _panel(w, h)
        d.rectangle((0, 0, w, 21), fill=INK + (255,))
        d.text((w / 2, 11), "WHERE IT LANDED", font=_letter_font(12), fill=(255, 238, 130), anchor="mm")
    d.rectangle((x0, y0, x0 + tw, y0 + th), fill=(36, 96, 170, 255), outline=INK + (255,), width=4 if not compact_th else 3)
    for yy in range(y0 + 5, y0 + th - 2, 6):                             # a light dot screen on the table
        for xx in range(x0 + 5 + (3 if (yy // 6) % 2 else 0), x0 + tw - 2, 6):
            d.ellipse((xx - 0.9, yy - 0.9, xx + 0.9, yy + 0.9), fill=(70, 140, 215, 255))
    d.line((x0 + 3, y0 + th / 2, x0 + tw - 3, y0 + th / 2), fill=(225, 235, 245, 255), width=1)
    d.line((x0 + tw / 2, y0 - 3, x0 + tw / 2, y0 + th + 3), fill=INK + (255,), width=6 if not compact_th else 5)
    d.line((x0 + tw / 2, y0 - 3, x0 + tw / 2, y0 + th + 3), fill=PAPER + (255,), width=2)
    if not compact_th:
        for i, (nm, rgb) in enumerate(((near_name, NEAR_RGB), (far_name, FAR_RGB))):
            cxn = x0 + tw * (0.25 if i == 0 else 0.75)
            d.text((cxn, y0 + th + 12), str(nm).upper()[:9], font=_letter_font(13), fill=rgb, anchor="mm", stroke_width=2, stroke_fill=INK)
    return _bgra(im)


def overlay_context(video, table, out_h=540, n=41):
    """What the clips of one recording share: the empty-hall plate for finding heads, its thresholds, and the landing map's size."""
    cap = cv2.VideoCapture(str(video)); SW, SH = cap.get(3), cap.get(4); cap.release()
    if not SW or not SH:
        return None
    s = out_h / SH; ow = int(round(SW * s / 2)) * 2
    frames = heads_mod.sample_frames(video, (ow, out_h), n)
    plate = heads_mod.plate_of(frames)
    if plate is None:
        return None
    thr = heads_mod.thresholds(plate)
    corners = table.corners * s
    table_top = float(corners[:, 1].min())
    net_x = float(np.mean(table.to_px([[NET_X, 0.0], [NET_X, W]])[:, 0]) * s)
    standing = heads_mod.standing_zone(table, s)
    tops = []
    for fr in frames:
        h = heads_mod.find_heads(fr, plate, table_top, net_x, thr=thr, standing=standing)
        if h["far"] and h["far"][2]:
            tops.append((h["far"][0], h["far"][1] - heads_mod.HEAD_H / 2))
    map_w_full = MAP_W + 2 * MAP_X0 + 8
    return dict(plate=plate, thr=thr, map_th=map_compact_height(tops, ow - map_w_full - 8), head_tops=tops, standing=standing)


PLATE_S, PLATE_N = 16.0, 25   # seconds around a rally, and frames out of them, that make the hall a clip's heads are found against
TAG_H, TAG_TAIL = 24, 13


@functools.lru_cache(maxsize=64)
def tag_sprite(name, rgb, tail="down"):
    """A name balloon, lettered caps on the player's colour. Its tail points down at the head below it, or sideways ("left" / "right")
    at a head beside it. The sprite is centred on the balloon's box whichever way the tail goes."""
    txt = str(name).upper()[:9]; f = _letter_font(14)
    w = int(f.getlength(txt)) + 20; h = TAG_H; m = TAG_TAIL + 5                      # m = margin all round, room for the tail and the shadow
    im = Image.new("RGBA", (w + 2 * m, h + 2 * m), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    x0, y0 = m, m
    d.rounded_rectangle((x0 + 4, y0 + 4, x0 + w + 4, y0 + h + 4), radius=8, fill=INK + (255,))
    if tail == "down":
        outer = [(x0 + w / 2 - 6, y0 + h - 1), (x0 + w / 2 + 8, y0 + h - 1), (x0 + w / 2 + 1, y0 + h + TAG_TAIL - 1)]
        inner = [(x0 + w / 2 - 4, y0 + h - 4), (x0 + w / 2 + 6, y0 + h - 4), (x0 + w / 2 + 1, y0 + h + TAG_TAIL - 5)]
    else:
        sx = -1 if tail == "left" else 1; ex = x0 if tail == "left" else x0 + w
        outer = [(ex - sx, y0 + h / 2 - 7), (ex - sx, y0 + h / 2 + 7), (ex + sx * (TAG_TAIL - 1), y0 + h / 2 + 3)]
        inner = [(ex - sx * 4, y0 + h / 2 - 5), (ex - sx * 4, y0 + h / 2 + 5), (ex + sx * (TAG_TAIL - 5), y0 + h / 2 + 2)]
    d.polygon(outer, fill=rgb + (255,), outline=INK + (255,))
    d.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=8, fill=rgb + (255,), outline=INK + (255,), width=3)
    d.polygon(inner, fill=rgb + (255,))
    d.text((x0 + w / 2, y0 + h / 2 - 1), txt, font=f, fill=INK, anchor="mm")
    return _bgra(im)


def tag_width(name):
    return int(_letter_font(14).getlength(str(name).upper()[:9])) + 20


def tag_spot(head, away, w, furniture, ow, keep=None, hgap=14, vgap=8, avoid=(), margin=8):
    """Where a player's name balloon goes for a head at `head` = (x, y): (slot, cx, cy, tail). First choice is BESIDE the head on the side
    away from the table (`away` = -1 left, +1 right) with the tail pointing at it; then above the head with the tail down; then the other
    side; then beside the neck, for a head with a panel right above it. A slot must be inside the frame and clear of the panels (point
    caption, scoreboard, landing map) by hgap / vgap and must not touch a box in `avoid` (the players' heads). `keep` = last frame's
    slot, kept while it is still clear, so a balloon does not
    hop. When nothing is clear, the slot that overlaps the panels least: a balloon is never moved on top of its player's head."""
    hx, hy = head
    d = w / 2 + 52                                                      # head centre to balloon centre, sideways: the tail tip stops ~40 px
    #                                                                     from the centre, clear of a big head near the camera (radius ~25)

    def side(k, dy):
        return (hx + k * d, hy + dy, "left" if k > 0 else "right")

    slots = [side(away, 4), (hx, hy - 60, "down"), side(-away, 4), side(away, 28), side(-away, 28)]       # side slots at ear level

    def clash(cx, cy):
        if cx - w / 2 - TAG_TAIL < margin or cx + w / 2 + TAG_TAIL > ow - margin:
            return math.inf
        r = (cx - w / 2 - hgap, cy - TAG_H / 2 - vgap, cx + w / 2 + hgap, cy + TAG_H / 2 + vgap)
        tail = (cx - w / 2 - TAG_TAIL, cy - TAG_H / 2, cx + w / 2 + TAG_TAIL, cy + TAG_H / 2 + TAG_TAIL)
        return sum(_overlap(r, fb) for fb in furniture) + 4 * sum(_overlap(tail, a) for a in avoid)

    order = ([keep] if keep is not None and 0 <= keep < len(slots) else []) + list(range(len(slots)))
    for i in order:
        if clash(*slots[i][:2]) == 0:
            return (i,) + slots[i]
    i = min(range(len(slots)), key=lambda i: (clash(*slots[i][:2]), i))
    return (i,) + slots[i]


TIP_W, TIP_H, TIP_STEPS = 456, 86, 40
TIP_LEAD_S, TIP_IN_S, TIP_AFTER_SERVE_S, TIP_MIN_S, TIP_CLEAR_S = 5.5, 0.30, 1.2, 4.2, 0.30   # 2026-09-26: longer to read
# a clip with a tip opens TIP_LEAD_S before the serve (if the dead time allows); the note appears TIP_IN_S into the clip, stays until
# TIP_AFTER_SERVE_S after the point starts, is never shown for less than TIP_MIN_S, and is gone TIP_CLEAR_S before the scoring moment


@functools.lru_cache(maxsize=256)
def tip_sprite(level, who, rgb, head, evidence, left=TIP_STEPS):
    """A note from the scout, along the bottom edge before the serve. A tab (SCOUT TIP, or EARLY READ when the evidence is thin), who it
    is for in that player's colour, the suggestion in lettered caps, the counts behind it, and an ink fuse under the text that burns
    down (`left` of TIP_STEPS remain) so the viewer can see the note is about to go."""
    w, h = TIP_W, TIP_H
    im, d = _panel(w, h)
    tab = "SCOUT TIP" if level == "tip" else "EARLY READ"
    tw = int(_font(21).getlength(tab)) + 22
    for yy in range(7, 27, 6):                                          # Ben-Day dots thickening toward the right end of the header row
        for xx in range(w - 170 + (3 if (yy // 6) % 2 else 0), w - 6, 6):
            rr = 0.4 + 2.0 * (xx - (w - 170)) / 164
            d.ellipse((xx - rr, yy - rr, xx + rr, yy + rr), fill=(255, 190, 70, 255))
    d.rectangle((0, 0, tw, 30), fill=(RED_RGB if level == "tip" else YELLOW_RGB) + (255,), outline=INK + (255,), width=4)
    if level == "tip":
        d.text((tw / 2, 15), tab, font=_font(21), fill=PAPER, anchor="mm", stroke_width=2, stroke_fill=INK)
    else:
        d.text((tw / 2, 15), tab, font=_font(21), fill=INK, anchor="mm")
    d.text((tw + 12, 16), "FOR", font=_letter_font(12), fill=(120, 20, 20), anchor="lm")
    d.text((tw + 12 + _letter_font(12).getlength("FOR "), 16), str(who).upper()[:12], font=_letter_font(17), fill=rgb, anchor="lm", stroke_width=2, stroke_fill=INK)
    for txt, y, size, floor, fill in ((head.upper(), 45, 22, 13, INK), (evidence.upper(), 64, 13, 9, (120, 20, 20))):
        while size > floor and _letter_font(size).getlength(txt) > w - 30:
            size -= 1
        d.text((15, y), txt, font=_letter_font(size), fill=fill, anchor="lm")
    x0, x1, fy = 14, w - 14, h - 9                                      # the fuse: burnt part as dots, what is left as an ink line, a spark between
    xs = x0 + (x1 - x0) * left / TIP_STEPS
    for xx in range(int(xs) + 6, x1, 6):
        d.ellipse((xx - 1, fy - 1, xx + 1, fy + 1), fill=(16, 16, 16, 110))
    if left > 0:
        d.line((x0, fy, xs, fy), fill=INK + (255,), width=3)
        star = [(xs + math.cos(a_) * r_, fy + math.sin(a_) * r_) for a_, r_ in zip(np.linspace(0, 2 * np.pi, 10, endpoint=False) + 0.35 * (left % 2), [8.5, 3.6] * 5)]
        d.polygon(star, fill=YELLOW_RGB + (255,), outline=INK + (255,)); d.ellipse((xs - 2.2, fy - 2.2, xs + 2.2, fy + 2.2), fill=RED_RGB + (255,))
    return _bgra(_slant(im, 0.10).rotate(-1.0, resample=Image.BICUBIC, expand=True))


CRIT_W, CRIT_LEAD_S = 440, 3.0           # the critique note's width; how far before the serve a clip opens to show it when there is no tip


BLUE_RGB, SKY_RGB = (35, 95, 205), (120, 200, 255)


def critique_sprite(rows):
    """The coach's criticism of ONE player along the bottom edge, in the comic's colours: a blue tab with yellow block lettering and
    halftone dots, his name on a chip in his colour, the fault in lettered caps, the counts behind it in blue ink and the fix on a red
    tab. rows = ((name, rgb, head, evidence, fix),); only the first row is drawn (one player at a time keeps it under the table)."""
    name, rgb, head, evidence, fix = rows[0]
    w, h = CRIT_W, 76
    im, d = _panel(w, h, fill=(255, 244, 196))
    for yy in range(5, 25, 6):                                          # halftone dots in the header row, thickening to the right
        for xx in range(w - 190 + (3 if (yy // 6) % 2 else 0), w - 6, 6):
            rr = 0.4 + 2.1 * (xx - (w - 190)) / 184
            d.ellipse((xx - rr, yy - rr, xx + rr, yy + rr), fill=SKY_RGB + (255,))
    d.rectangle((0, h - 6, w, h), fill=rgb + (255,))                    # a stripe in the player's colour along the foot
    tab = "COACH'S CRITIQUE"
    tw = int(_font(17).getlength(tab)) + 20
    d.rectangle((0, 0, tw, 26), fill=BLUE_RGB + (255,), outline=INK + (255,), width=3)
    d.text((tw / 2, 13), tab, font=_font(17), fill=YELLOW_RGB, anchor="mm", stroke_width=2, stroke_fill=INK)
    chip = str(name).upper()[:10]; cw = int(_letter_font(14).getlength(chip)) + 16
    d.rounded_rectangle((tw + 8, 3, tw + 8 + cw, 23), radius=7, fill=rgb + (255,), outline=INK + (255,), width=2)
    d.text((tw + 8 + cw / 2, 13), chip, font=_letter_font(14), fill=INK, anchor="mm")
    size = 19
    while size > 11 and _letter_font(size).getlength(head.upper()) > w - 26:
        size -= 1
    d.text((13, 38), head.upper(), font=_letter_font(size), fill=INK, anchor="lm")
    size = 11
    while size > 8 and _letter_font(size).getlength(evidence.upper()) > w - 26:
        size -= 1
    d.text((13, 53), evidence.upper(), font=_letter_font(size), fill=(20, 60, 150), anchor="lm")
    fw = int(_font(11).getlength("FIX")) + 10
    d.rectangle((11, 59, 11 + fw, 71), fill=RED_RGB + (255,), outline=INK + (255,), width=1)
    d.text((11 + fw / 2, 65), "FIX", font=_font(11), fill=PAPER, anchor="mm")
    size = 11
    while size > 8 and _letter_font(size).getlength(fix.upper()) > w - 30 - fw:
        size -= 1
    d.text((17 + fw, 65), fix.upper(), font=_letter_font(size), fill=(185, 25, 25), anchor="lm")
    return _bgra(_slant(im, 0.08).rotate(0.8, resample=Image.BICUBIC, expand=True))


def critique_rows(notes, names, prefer=None, k=0):
    """((name, rgb, head, evidence, fix),) for ONE player: `prefer` if he has a criticism, else whoever has one (near player first). A
    player's criticisms are a list; the k-th is shown (cycling), so over a game each one comes round."""
    order = ([prefer] if prefer in names else []) + [n for n in names if n != prefer]
    for n in order:
        c = (notes or {}).get(n)
        if isinstance(c, list):
            c = c[k % len(c)] if c else None
        if c is not None:
            return ((n, NEAR_RGB if n == names[0] else FAR_RGB, c.head, c.evidence, c.fix),)
    return ()


NOTE_BOTTOM = 18           # px of the 540 under the notes and the gauge (6 until 2026-09-26 evening): on the bottom edge, below the table (2026-09-26); the
                           # report's player leaves a strip under the picture for the browser's controls, so nothing hides them there
INTENSITY = {0: dict(scale=0.82, punch=0.45, rays=14, hold=1.8), 1: dict(scale=1.0, punch=1.0, rays=40, hold=2.1),
             2: dict(scale=1.14, punch=1.35, rays=56, hold=2.5), 3: dict(scale=1.28, punch=1.7, rays=72, hold=2.9)}
REPLAY_BEFORE_S, REPLAY_AFTER_S, REPLAY_SLOW = 2.8, 0.7, 3     # the replay: this long before and after the deciding moment (2026-09-26:
                                                                # longer); REPLAY_SLOW kept for callers, the speed now ramps
REPLAY_ZOOM, REPLAY_FADE_S = 1.12, 0.25                         # a gentle push-in toward the deciding spot; cross-fade into the replay
HIRES = True                                                    # write the recording's own resolution (1080p): every overlay is laid out
                                                                # at 540 lines as before, the footage under it stays full resolution


def _upcompose(rendered, base, full):
    """The frame at the footage's own resolution: `full` (the footage) wherever nothing was drawn, and the 540-line rendering scaled up
    where the overlays are. base = the 540-line footage before anything was drawn on it; drawn pixels are where the two differ."""
    H, W = full.shape[:2]
    m = (cv2.absdiff(rendered, base).max(axis=2) > 8).astype(np.uint8) * 255
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    up = cv2.resize(rendered, (W, H), interpolation=cv2.INTER_LANCZOS4)
    al = cv2.GaussianBlur(cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR), (5, 5), 0).astype(np.uint16)[..., None]
    return ((up.astype(np.uint16) * al + full.astype(np.uint16) * (255 - al)) // 255).astype(np.uint8)


def _speed_ramp(t, t_dec):
    """Output frames per source frame in a replay: 2 (half speed) away from the deciding moment, easing to 4 (quarter speed) around it."""
    return 2 + int(round(2 * math.exp(-((t - t_dec) / 0.45) ** 2)))


def tip_spot(size, boxes, frame, scales=(1.0, 0.86, 0.74), bottom=None):
    """(cx, cy, scale) for the note along the bottom edge: right of the speed gauge, where it covers the players' boxes least, then
    nearest the middle; it is only drawn smaller when full size cannot clear a player. A frame too narrow to fit it beside the
    gauge puts it just above the gauge instead."""
    ow, oh = frame; best = None
    for sc in scales:
        w, h = size[0] * sc, size[1] * sc
        lo, hi = GAUGE_W + 34 + w / 2, ow - 12 - w / 2
        if hi < lo:
            continue
        cy = oh - (NOTE_BOTTOM if bottom is None else bottom) - h / 2 - 2
        for cx in np.arange(lo, hi + 1, 12):
            cover = sum(_overlap((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), b) for b in boxes) / (w * h)
            key = (round(cover, 2) > 0.02, round(cover, 2) if cover > 0.02 else 0.0, -sc, abs(cx - (ow / 2 + 60)))
            if best is None or key < best[0]:
                best = (key, float(cx), float(cy), sc)
    if best is None:
        sc = min(scales[-1], (ow - 24) / size[0])
        return ow / 2, oh - NOTE_BOTTOM - GAUGE_H - 14 - size[1] * sc / 2, sc
    return best[1:]


def _ease_in_cubic(t):
    t = min(max(t, 0.0), 1.0)
    return t ** 3


def _corner(img, spr, x, y):
    """Blend a sprite with its top-left corner at (x, y)."""
    _blend(img, spr, x + spr.shape[1] / 2, y + spr.shape[0] / 2)


def _ease_out_back(t):
    t = min(max(t, 0.0), 1.0); c1 = 1.70158; c3 = c1 + 1
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def _ease_out_expo(t):
    t = min(max(t, 0.0), 1.0)
    return 1.0 if t >= 1 else 1 - 2 ** (-10 * t)


def posture_sprite(label, value, rgb):
    """A small lettered readout for a posture measure at a hit ("KNEES" over "128°"), on the player's colour like the name balloons."""
    f1, f2 = _letter_font(11), _letter_font(19)
    w = int(max(f1.getlength(label), f2.getlength(value))) + 18; h = 42; m = 6
    im = Image.new("RGBA", (w + 2 * m, h + 2 * m), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((m + 3, m + 3, m + w + 3, m + h + 3), radius=7, fill=INK + (255,))
    d.rounded_rectangle((m, m, m + w, m + h), radius=7, fill=rgb + (255,), outline=INK + (255,), width=3)
    d.text((m + w / 2, m + 12), label, font=f1, fill=INK, anchor="mm")
    d.text((m + w / 2, m + 29), value, font=f2, fill=INK, anchor="mm")
    return _bgra(im)


def draw_skeleton(img, a, s, col, min_c=0.25):
    """A player's 2D skeleton (tt_scout.pose joints) inked like the rest of the clip: dark keyline under the player's colour."""
    from .pose import DRAW_BONES, FACE, J
    for p, q in DRAW_BONES:                                          # nothing is drawn on a face
        A, B = a[J[p]], a[J[q]]
        if A[2] < min_c or B[2] < min_c:
            continue
        p0, p1 = (int(A[0] * s), int(A[1] * s)), (int(B[0] * s), int(B[1] * s))
        cv2.line(img, p0, p1, INK_BGR, 5, cv2.LINE_AA)
        cv2.line(img, p0, p1, col, 2, cv2.LINE_AA)
    for i in range(len(a)):
        if a[i, 2] >= min_c and i not in {J[n] for n in FACE}:
            c = (int(a[i, 0] * s), int(a[i, 1] * s))
            cv2.circle(img, c, 4, INK_BGR, -1, cv2.LINE_AA); cv2.circle(img, c, 2, YELLOW_BGR, -1, cv2.LINE_AA)


def draw_knee_arc(img, a, knee, s, col):
    """The measured knee's angle drawn as an arc between thigh and shin."""
    from .pose import J
    h, k, an = a[J[knee + "hip"], :2] * s, a[J[knee + "kne"], :2] * s, a[J[knee + "ank"], :2] * s
    a1 = math.degrees(math.atan2(h[1] - k[1], h[0] - k[0])); a2 = math.degrees(math.atan2(an[1] - k[1], an[0] - k[0]))
    lo, hi = sorted((a1, a2))
    if hi - lo > 180:
        lo, hi = hi, lo + 360
    c = (int(k[0]), int(k[1]))
    cv2.ellipse(img, c, (17, 17), 0, lo, hi, INK_BGR, 5, cv2.LINE_AA)
    cv2.ellipse(img, c, (17, 17), 0, lo, hi, col, 2, cv2.LINE_AA)


SETTLE_S = {"double_bounce": 0.35, "not_returned": 0.8, "long": 0.8}
TRAY_W, TRAY_H, TRAY_CY = 440, 48, 140   # each player's analysis, hung under the scoreboard (its centre y at 540 lines), never taken off


@functools.lru_cache(maxsize=1024)
def _tray(names, vals):
    w, h, sl = TRAY_W, TRAY_H, 12
    W2, H2 = w + 20, h + 16
    im = Image.new("RGBA", (W2, H2), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    x0, y0 = 6, 4
    outer = [(x0 + sl, y0), (x0 + w, y0), (x0 + w - sl, y0 + h), (x0, y0 + h)]
    d.polygon([(x + 6, y + 7) for x, y in outer], fill=INK + (255,))
    d.polygon(outer, fill=PAPER_BOX + (255,))
    mid = x0 + w / 2
    d.polygon([(x0 + sl, y0), (mid + 3, y0), (mid + 2, y0 + 6), (x0 + sl - 1, y0 + 6)], fill=NEAR_RGB + (255,))
    d.polygon([(mid + 3, y0), (x0 + w, y0), (x0 + w - 1, y0 + 6), (mid + 2, y0 + 6)], fill=FAR_RGB + (255,))
    d.line([(mid + sl / 2, y0), (mid - sl / 2, y0 + h)], fill=INK + (255,), width=3)
    d.line(outer + [outer[0]], fill=INK + (255,), width=4)
    half = w / 2 - sl - 18
    for i, (serve, winners, misses, fastest, knee) in enumerate(vals):
        l1 = f"SERVE {serve[0]}/{serve[1]}  \u00b7  WINNERS {winners}  \u00b7  MISSES {misses}"
        fast_txt = str(fastest) if fastest else "\u2013"
        knee_txt = f"{knee}\u00b0" if knee else "\u2013"
        l2 = f"FASTEST {fast_txt} KM/H  \u00b7  KNEES {knee_txt}"
        size = 13
        while size > 9 and max(_font(size).getlength(l1), _font(size).getlength(l2)) > half:
            size -= 1
        f = _font(size)
        if i == 0:
            d.text((x0 + sl + 10, y0 + 18), l1, font=f, fill=INK, anchor="lm"); d.text((x0 + sl + 4, y0 + 35), l2, font=f, fill=INK, anchor="lm")
        else:
            d.text((x0 + w - sl - 4, y0 + 18), l1, font=f, fill=INK, anchor="rm"); d.text((x0 + w - sl - 10, y0 + 35), l2, font=f, fill=INK, anchor="rm")
    return _bgra(im)


def tray_sprite(names, now):
    """Both players' live analysis as one lettered strip: serve points won, winning shots and shots that missed the table (up to the last
    point decided), the fastest shot so far and the knee bend at the last hit. now = match_stats.Live.at(t)."""
    vals = tuple((tuple(now[n]["serve"]), now[n]["winners"], now[n]["misses"], now[n]["fastest"], now[n]["knee"]) if n in now
                 else ((0, 0), 0, 0, None, None) for n in names)
    return _tray(tuple(names), vals)


def blur_people(fr, spec):
    """Make people who are not the two players unrecognisable in a full-resolution frame, in place, before anything is drawn on it: a
    heavy blur inside spec["zone"] (the back of the hall, whatever is found there) and inside feathered ellipses round everyone else
    Vision found (spec["hide"]), stopping hard at the players' own outlines and the ball (spec["keep"]: skeletons, or boxes)."""
    if not spec or not (spec.get("hide") or spec.get("zone")):
        return fr
    H, W = fr.shape[:2]; q = 4
    m = np.zeros((H // q, W // q), np.float32)
    for poly in spec.get("zone") or []:
        cv2.fillPoly(m, [(np.asarray(poly, float) / q).astype(np.int32)], 1.0)
    for x0, y0, x1, y1 in spec.get("hide") or []:
        cx, cy, ax, ay = (x0 + x1) / 2 / q, (y0 + y1) / 2 / q, max(2.0, (x1 - x0) / 2 / q), max(2.0, (y1 - y0) / 2 / q)
        cv2.ellipse(m, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, 1.0, -1)
    if not m.any():
        return fr
    keep = np.zeros_like(m)                                            # the players: drawn here, then cut out of the blur exactly
    for k_ in spec.get("keep") or []:
        if len(k_) == 4 and np.ndim(k_) == 1:                          # a box (the ball, the net)
            x0, y0, x1, y1 = k_
            cv2.rectangle(keep, (int(x0 / q), int(y0 / q)), (int(x1 / q), int(y1 / q)), 1.0, -1)
            continue
        from .pose import BONES, J                                     # a player's skeleton: sharp along the body itself
        sk = np.asarray(k_, float); ok = sk[:, 2] >= 0.2
        top = next((sk[J[n], :2] for n in ("neck", "nose") if ok[J[n]]), None)
        hip = next((sk[J[n], :2] for n in ("root", "lhip", "rhip") if ok[J[n]]), None)
        torso = float(np.hypot(*(top - hip))) if top is not None and hip is not None else 120.0
        pt = lambda n: tuple(int(v / q) for v in sk[J[n], :2])
        for a_, b_ in BONES:
            if ok[J[a_]] and ok[J[b_]]:
                pair = {a_, b_}
                wf = 0.14 if pair & {"lelb", "relb"} else 0.24 if pair & {"lank", "rank"} else 0.30 if pair & {"lkne", "rkne"} else 0.40
                cv2.line(keep, pt(a_), pt(b_), 1.0, max(3, int(wf * torso / q)))
        core = [n for n in ("lsho", "rsho", "rhip", "lhip") if ok[J[n]]]
        if len(core) == 4:                                             # the trunk, filled
            cv2.fillPoly(keep, [np.array([pt(n) for n in core], np.int32)], 1.0)
        for n_, r_ in (("nose", 0.42), ("neck", 0.30), ("lwri", 0.15), ("rwri", 0.15), ("lank", 0.20), ("rank", 0.20)):
            if n_ in J and ok[J[n_]]:                                  # head and hair, the racket hands, the feet
                cv2.circle(keep, pt(n_), max(2, int(r_ * torso / q)), 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), 3.0)
    m[keep > 0] = 0.0                                                  # soft round the strangers, a hard stop at the players' outline
    small = cv2.resize(fr, (W // q, H // q), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 6.0)
    blur = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    mm = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    fr[:] = (fr * (1.0 - mm) + blur * mm).astype(np.uint8)
    return fr
    H, W = fr.shape[:2]; q = 4
    m = np.zeros((H // q, W // q), np.float32)
    for x0, y0, x1, y1 in spec["hide"]:
        cx, cy, ax, ay = (x0 + x1) / 2 / q, (y0 + y1) / 2 / q, max(2.0, (x1 - x0) / 2 / q), max(2.0, (y1 - y0) / 2 / q)
        cv2.ellipse(m, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, 1.0, -1)
    keep = np.zeros_like(m)                                            # the players: drawn here, then cut out of the blur exactly
    for k_ in spec.get("keep") or []:
        if len(k_) == 4 and np.ndim(k_) == 1:                          # a box
            x0, y0, x1, y1 = k_
            cv2.rectangle(keep, (int(x0 / q), int(y0 / q)), (int(x1 / q), int(y1 / q)), 1.0, -1)
            continue
        from .pose import BONES, J                                     # a player's skeleton: sharp along the body itself
        sk = np.asarray(k_, float); ok = sk[:, 2] >= 0.2
        top = next((sk[J[n], :2] for n in ("neck", "nose") if ok[J[n]]), None)
        hip = next((sk[J[n], :2] for n in ("root", "lhip", "rhip") if ok[J[n]]), None)
        torso = float(np.hypot(*(top - hip))) if top is not None and hip is not None else 120.0
        width = max(3, int(0.34 * torso / q))
        for a_, b_ in BONES:
            if ok[J[a_]] and ok[J[b_]]:
                cv2.line(keep, tuple(int(v / q) for v in sk[J[a_], :2]), tuple(int(v / q) for v in sk[J[b_], :2]), 1.0, width)
        for n_, r_ in (("nose", 0.55), ("neck", 0.45), ("lwri", 0.35), ("rwri", 0.35), ("lank", 0.25), ("rank", 0.25)):
            if n_ in J and ok[J[n_]]:                                  # head and hair, the racket hands, the feet
                cv2.circle(keep, tuple(int(v / q) for v in sk[J[n_], :2]), max(2, int(r_ * torso / q)), 1.0, -1)
    if not m.any():
        return fr
    m = cv2.GaussianBlur(m, (0, 0), 3.0)
    m[keep > 0] = 0.0                                                  # soft round the strangers, a hard stop at the players' outline
    small = cv2.resize(fr, (W // q, H // q), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 6.0)
    blur = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    mm = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    fr[:] = (fr * (1.0 - mm) + blur * mm).astype(np.uint8)
    return fr


def render_point_clip(video, fps, table, track, events, obs_by_frame, point, dst, out_h=540, lead_s=1.0, tail_s=0.8, comic=True, scoreboard=True,
                      hold_s=2.1, tip=None, max_lead_s=None, context=None, show_table=False, pose=None, analysis=None, critique=None,
                      replay=True, critique_after=None, speeds=None, moment=None, confirmed=None, hide=None, shots=None):
    """The rally plays live with the tracking drawn on. When the point is won (and `comic`), the deciding frame FREEZES into an inked
    comic panel and the scoring moment plays over it for hold_s seconds; burst and caption are placed where no player, ball or
    furniture is (see place()).
    tip = a tips.Tip to show before the serve; max_lead_s = how far before the point the clip may open (the dead time since the
    previous point). Without enough of it to read the note, the clip is rendered without the note rather than flashing it.
    context = overlay_context() of the recording: balloons then follow heads found in the picture and the landing map takes the size the
    camera leaves above the far player's head; without it balloons use the tracker's player blobs and the table height.
    show_table draws the calibrated table outline and net line over the footage; it is for checking a calibration, and off by
    default because the lines sit on the table a viewer can already see."""
    if not shutil.which("ffmpeg"):
        return False
    cap = cv2.VideoCapture(str(video))
    SW, SH = int(cap.get(3)), int(cap.get(4))
    s = out_h / SH; ow = int(round(SW * s / 2)) * 2
    step = max(1, int(round(fps / 60.0)))                          # 120 fps source -> every 2nd frame, played at 60
    out_fps = fps / step
    winner = point.get("winner_name")
    last_t = max([e["t"] for e in point.get("events", [])] or [point["end_t"]])
    # celebrate only a point the camera saw end (match_stats.confirm): the miss or the winner seen dropping to the floor, a return that
    # died in the net, a second bounce. Without it, no burst and no caption: the point is only counted (2026-09-26)
    if confirmed is None:
        confirmed = (True, None, "")
    celebrate = bool(comic and winner and confirmed[0])
    ev_all = [e for e in events if point["start_t"] - lead_s <= e["t"] <= last_t + 0.05 and e["kind"] in ("bounce", "net")]     # never the longer tip lead: no strays
    # the scoring moment starts once the point has visibly been decided: the second bounce, or the miss / the ball flying out after the
    # last event (freezing 0.15 s after it cut the clip before the point was given: our 2026-09-25 recording)
    t_hit = confirmed[1] if (confirmed[0] and confirmed[1] is not None) else last_t + SETTLE_S.get(point.get("ending"), 0.15)
    lead, tip_win = lead_s, None
    if tip is not None:
        lead = max(lead_s, min(TIP_LEAD_S, lead_s if max_lead_s is None else max_lead_s, point["start_t"]))
        t_in = point["start_t"] - lead + TIP_IN_S
        t_out = min(max(point["start_t"] + TIP_AFTER_SERVE_S, t_in + TIP_MIN_S), t_hit - TIP_CLEAR_S)
        if t_out - t_in >= TIP_MIN_S:
            tip_win = (t_in, t_out)
        else:
            lead = lead_s
    pnames = (point.get("near_player") or "near", point.get("far_player") or "far")
    crit_rows = critique_rows(critique, pnames, prefer=pnames[point["id"] % 2], k=point["id"] // 2)   # players take turns, their notes cycle
    if crit_rows and tip_win is None:                               # no tip before this serve: the criticism gets the lull to be read in
        lead = max(lead, min(CRIT_LEAD_S, lead_s if max_lead_s is None else max_lead_s, point["start_t"]))
    crit_win = None
    if crit_rows:
        c_in = tip_win[1] + 0.25 if tip_win else point["start_t"] - lead + TIP_IN_S
        c_out = t_hit - TIP_CLEAR_S
        if c_out - c_in >= 1.2:
            crit_win = (c_in, c_out)
    f0 = max(0, int((point["start_t"] - lead) * fps))
    f1 = min(len(track) - 1, int((t_hit if celebrate else last_t + tail_s) * fps))
    quad = (table.corners * s).astype(np.int32).reshape(-1, 1, 2)
    table_top = float(quad.reshape(-1, 2)[:, 1].min())
    n1, n2 = (table.to_px([[NET_X, 0.0], [NET_X, W]]) * s).astype(int)
    c = table.corners
    ppm = 0.5 * (np.hypot(*(c[3] - c[0])) + np.hypot(*(c[2] - c[1]))) / L          # pixels per metre along the table, full resolution
    names = (point.get("near_player") or "near", point.get("far_player") or "far")
    sc = point.get("_score") or dict(before={}, after={}, games={})
    server_idx = 0 if point.get("serve_side") == "near" else 1
    win_idx = names.index(winner) if winner in names else None
    board_before = scoreboard_sprite(names, sc["before"], sc["games"], server_idx)
    board_after = scoreboard_sprite(names, sc["after"], sc.get("games_after", sc["games"]), server_idx, hot=win_idx) if celebrate else board_before
    board_x, board_y = ow // 2 + 14, 62                               # between the point caption (left) and the landing map (right)
    rng = np.random.default_rng(point["id"])
    if tip_win:
        tip_args = (tip.level, tip.player, NEAR_RGB if tip.player == names[0] else FAR_RGB, tip.head, tip.evidence)
        th_, tw_ = tip_sprite(*tip_args).shape[:2]
        seen = [b for fk in range(int(tip_win[0] * fps), int(tip_win[1] * fps) + 1, max(1, int(fps / 4))) for b in player_boxes(obs_by_frame, fk, s)]
        tip_cx, tip_cy, tip_sc = tip_spot((tw_, th_), seen, (ow, out_h))
    if crit_win:
        crit_spr = critique_sprite(crit_rows)
        ch2, cw2 = crit_spr.shape[:2]
        seen2 = [b for fk in range(int(crit_win[0] * fps), int(crit_win[1] * fps) + 1, max(1, int(fps / 4))) for b in player_boxes(obs_by_frame, fk, s)]
        crit_cx, crit_cy, crit_sc = tip_spot((cw2, ch2), seen2, (ow, out_h))
    OW, OH = (SW // 2 * 2, SH // 2 * 2) if HIRES else (ow, out_h)
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}", "-r", str(out_fps),
                             "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21" if HIRES else "24", "-pix_fmt", "yuv420p",
                             "-movflags", "+faststart", str(dst)], stdin=subprocess.PIPE)
    last_out = [None]

    ctx = context or {}
    plate, thr, map_th = ctx.get("plate"), ctx.get("thr"), ctx.get("map_th", 0)
    standing = ctx.get("standing")
    if plate is not None:                                          # a plate from the seconds around this rally: whoever stood still
        span = max(int(PLATE_S * fps), f1 - f0)                    # through it counts as hall, so a crowd behind the table drops out
        lo = max(0, (f0 + f1) // 2 - span // 2)
        near_plate = heads_mod.background_plate(video, (ow, out_h), PLATE_N, lo, lo + span)
        if near_plate is not None:
            plate, thr = near_plate, heads_mod.thresholds(near_plate)
    net_x = float((n1[0] + n2[0]) / 2)
    hud0, map0 = hud_sprite(point["id"], point.get("server") or point.get("serve_side"), 0), map_sprite(names[0], names[1], map_th)
    bh0, bw0 = board_before.shape[:2]
    panels = [(12, 12, 12 + hud0.shape[1], 12 + hud0.shape[0]), (board_x - bw0 / 2, board_y - bh0 / 2, board_x + bw0 / 2, board_y + bh0 / 2),
              (ow - map0.shape[1] - 8, 10, ow - 8, 10 + map0.shape[0])]                     # point caption, scoreboard, landing map as drawn
    if analysis is not None and scoreboard:                        # the analysis tray under the scoreboard counts as furniture too
        panels.append((board_x - TRAY_W / 2, TRAY_CY - TRAY_H / 2, board_x + TRAY_W / 2, TRAY_CY + TRAY_H / 2 + 8))
    head_y = table_top - 92                                        # without a plate: a head is 65 to 105 px above the table's far edge here
    tags = {}                                                      # per end: x, y (head centre), seen, slot, box
    glide = 1 - math.exp(-(step / fps) / (0.10 if plate is not None else 0.30))    # measured heads are followed closely, blobs loosely

    def follow(f, found=None, first=False):
        """Update each end's balloon anchor. With a plate: the head found in this frame (found = find_heads() of it). Without one: the
        largest player blob near frame f, at table height. Nothing for 2.5 s: the balloon goes."""
        for end in ("near", "far"):
            st = tags.get(end); x = y = None
            if plate is not None:
                h = (found or {}).get(end)
                if h is not None:
                    x, y = h[0], h[1]
            else:
                back = int(1.0 * fps) if first else 3
                blobs = [o for k in range(f - back, f + 1) for o in obs_by_frame.get(k, []) if o["end"] == end]
                if blobs:
                    x, y = max(blobs, key=lambda b: b["area"])["cx"] * s, head_y
            if x is not None:
                if st is None:
                    tags[end] = dict(x=x, y=head_y if y is None else y, seen=f)
                else:
                    st["x"] += glide * (x - st["x"])
                    if y is not None:
                        st["y"] += glide * (y - st["y"])
                    st["seen"] = f
            elif st and f - st["seen"] > 2.5 * fps:
                tags.pop(end)

    def head_boxes():
        return [(st["x"] - 20, st["y"] - 22, st["x"] + 20, st["y"] + 20) for st in tags.values()]

    def furniture(img, shots, needle, shot_v, pop, landed, M=None):
        """Everything lettered: name balloons, the point caption, the landing map, the speed gauge with its needle and the last shot's number.
        M = the camera move applied to the footage under it, so the balloons stay on their players."""
        _corner(img, hud_sprite(point["id"], point.get("server") or point.get("serve_side"), shots), 12, 12)
        m = map0; mx0, my0 = ow - m.shape[1] - 8, 10
        _corner(img, m, mx0, my0)
        tx0, ty0, tw_m, th_m = map_frame(map_th)
        landed = [b_ for b_ in landed if -0.08 <= b_["x_m"] <= L + 0.08 and -0.08 <= b_["y_m"] <= W + 0.08]   # a "landing" off the table is a
        for k, b_ in enumerate(landed):                                                              # detection error, not a dot on the rim
            px = int(mx0 + tx0 + np.clip(b_["x_m"] / L, 0, 1) * tw_m); py = int(my0 + ty0 + th_m - np.clip(b_["y_m"] / W, 0, 1) * th_m)
            col = NEAR_COL if b_["side"] == "far" else FAR_COL          # a ball landing on the far half was hit by the near player
            newest = k == len(landed) - 1
            cv2.circle(img, (px, py), 8 if newest else 6, INK_BGR, -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), 6 if newest else 4, col, -1, cv2.LINE_AA)
            if newest:
                cv2.circle(img, (px, py), 11, YELLOW_BGR, 2, cv2.LINE_AA)
        g = gauge_sprite(); gx, gy = 12, out_h - NOTE_BOTTOM - g.shape[0]              # shadow included, clear of the controls band
        _corner(img, g, gx, gy)
        bx_, by_, bw_, bh_ = GAUGE_BAR
        if needle is not None:                                     # the needle is live but eased; the numeral is the last shot's speed, held
            mxp = gx + bx_ + int(np.clip(needle / SPEED_STOPS[-1][0], 0, 1) * (bw_ - 1)); myp = gy + by_
            tri = np.array([(mxp - 6, myp - 9), (mxp + 6, myp - 9), (mxp, myp + 1)], np.int32)
            cv2.fillConvexPoly(img, tri, INK_BGR, cv2.LINE_AA); cv2.polylines(img, [tri], True, YELLOW_BGR, 1, cv2.LINE_AA)
            cv2.line(img, (mxp, myp), (mxp, myp + bh_), INK_BGR, 2, cv2.LINE_AA)
        if shot_v is not None:
            _blend(img, reading_sprite(shot_v), gx + GAUGE_NUM[0], gy + GAUGE_NUM[1], scale=1.0 + 0.12 * pop)
        def on_screen(x, y):                                       # footage coordinates -> where the camera move put them
            return (x, y) if M is None else (M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2])

        heads_now = [on_screen(st["x"], st["y"]) for st in tags.values()]
        heads_now = [(hx - 20, hy - 22, hx + 20, hy + 20) for hx, hy in heads_now]
        for end, st in tags.items():                               # balloons last, placed on screen after the camera move, so a push-in on
            name = names[0] if end == "near" else names[1]; tw = tag_width(name)          # the frozen panel cannot slide a head under one
            hx, hy = on_screen(st["x"], st["y"])
            slot, x, y, tail = tag_spot((hx, hy), -1 if end == "near" else 1, tw, panels if scoreboard else panels[::2], ow, st.get("slot"),
                                        avoid=heads_now, margin=8 if M is None else 22)     # inside the freeze's ink border
            st.update(slot=slot)
            if M is None:
                st["box"] = (x - tw / 2, y - TAG_H / 2, x + tw / 2, y + TAG_H / 2)
            _blend(img, tag_sprite(name, NEAR_RGB if end == "near" else FAR_RGB, tail), int(x), int(y))

    def write(img, full=None, base=None, reps=1, fade=None):
        """Write a 540-line rendering at the output size: composited over the full-resolution footage when there is some (full, base),
        else scaled up (the inked freeze is a drawing anyway). fade = (weight of the new frame): a cross-fade from the last frame out."""
        if HIRES:
            out = _upcompose(img, base, full[:OH, :OW]) if full is not None else cv2.resize(img, (OW, OH), interpolation=cv2.INTER_LANCZOS4)
        else:
            out = img
        if fade is not None and last_out[0] is not None and fade < 1:
            out = cv2.addWeighted(out, fade, last_out[0], 1 - fade, 0)
        last_out[0] = out
        try:
            for _ in range(reps):
                proc.stdin.write(out.tobytes())
            return True
        except BrokenPipeError:
            return False

    from . import pose as pose_mod
    posture = []                                                   # the posture at each racket hit in the clip, and where its knee is
    if pose:                                                       # (with the shots: the hit's end and its gated knee are the shot's, pose.at_hits)
        for h in pose_mod.at_hits([e for e in events if f0 / fps - 0.2 <= e["t"] <= f1 / fps + 0.2], pose, fps, shots=shots, points=[point]):
            if h.get("knee") is None:
                continue
            fk = int(round(h["t"] * fps))
            sk = next(((pose.get(fk + d) or {}).get(h["side"]) for d in sorted(range(-6, 7), key=abs) if (pose.get(fk + d) or {}).get(h["side"]) is not None), None)
            leg = h.get("leg") or (pose_mod.near_leg(sk)[0] if sk is not None else None)
            if leg is not None and all(sk[pose_mod.J[leg + j], 2] >= pose_mod.MIN_C for j in ("hip", "kne", "ank")):
                h["knee_joint"] = leg; h["skel"] = sk
                posture.append(h)
    if speeds is not None:                                         # the 3D-fitted shots of this point only (flight.py)
        from .flight import Speeds
        speeds = Speeds([sh for sh in speeds.shots if sh.get("point") == point["id"]], points=[point])
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    trail_n = int(0.30 * fps)
    last_speed, live, state = None, None, None
    needle, ease = None, 1 - math.exp(-(step / fps) / NEEDLE_S)    # the needle eases toward the live speed instead of twitching
    held = Held()                                                  # the number: the last shot's speed, net to bounce, readable
    if plate is None:
        follow(f0, first=True)
    for f in range(f0, f1 + 1):
        ok, fr = cap.read()
        if not ok:
            break
        if (f - f0) % step and f != f1:
            continue
        if hide:
            blur_people(fr, hide.get(f))                             # people who are not the two players, before anything is drawn
        t = f / fps
        img = cv2.resize(fr, (ow, out_h), interpolation=cv2.INTER_AREA)
        base = img.copy()                                          # the footage before anything is drawn on it (for the 1080p composite)
        found = heads_mod.find_heads(img, plate, table_top, net_x, thr=thr, standing=standing) if plate is not None else None   # before drawing
        if show_table:                                             # the calibration, drawn only when asked for
            cv2.polylines(img, [quad], True, GREEN, 1, cv2.LINE_AA); cv2.line(img, tuple(n1), tuple(n2), GREEN, 1, cv2.LINE_AA)
        # --- ball trail, coloured by speed; only the current track, never joined across a tracker jump
        seg = track[max(0, f - trail_n):f + 1]
        seg = seg[~np.isnan(seg[:, 2])]
        if len(seg):
            seg = seg[seg[:, 4] == seg[-1, 4]]                         # an older fragment is usually a shoe or a racket
        v_now = None
        if len(seg) > 1:
            gaps = np.maximum(1.0, np.diff(seg[:, 0]))
            dist = np.hypot(*np.diff(seg[:, 2:4], axis=0).T)
            v = dist / gaps * fps / ppm                                # m/s along the image plane
            v = np.array([np.median(v[max(0, k - 1):k + 2]) for k in range(len(v))])      # a 3-sample median calms detection jitter
            for k in range(1, len(seg)):
                if dist[k - 1] > 110.0 * gaps[k - 1]:
                    continue
                a = k / len(seg)
                vk = speeds.live(float(seg[k, 1])) if speeds is not None else None
                col = tuple(int(ch * (0.35 + 0.65 * a)) for ch in speed_colour(v[k - 1] if vk is None else vk))
                p0, p1 = (seg[k - 1, 2:4] * s).astype(int), (seg[k, 2:4] * s).astype(int)
                cv2.line(img, tuple(p0), tuple(p1), INK_BGR, 3 + int(3 * a), cv2.LINE_AA)       # ink keyline under the colour
                cv2.line(img, tuple(p0), tuple(p1), col, 1 + int(3 * a), cv2.LINE_AA)
            if seg[-1, 0] == f:
                v_now = float(v[-1])
        if speeds is not None:                                     # the fitted 3D speed inside a measured flight; otherwise hold
            v_now = speeds.live(t)
            if v_now is None and speeds.reading(t) == DASH:
                needle = None                                      # the shot in play was not measured: no needle, not the last one's
        if v_now is not None:
            needle = v_now if needle is None else needle + ease * (v_now - needle)
        if not np.isnan(track[f, 2]):
            centre = (int(track[f, 2] * s), int(track[f, 3] * s))
            cv2.circle(img, centre, 8, INK_BGR, 4, cv2.LINE_AA); cv2.circle(img, centre, 8, speed_colour(v_now or 0.0), 2, cv2.LINE_AA)
        shots = sum(1 for e in point.get("events", []) if e["kind"] == "net" and e["t"] <= t)     # inferred crossings count too
        for e in ev_all:
            age = t - e["t"]
            if age < 0 or "x_px" not in e:
                continue
            if e["kind"] == "net" and age < 0.25:
                cv2.line(img, (int(e["x_px"] * s), int(e["y_px"] * s) - 14), (int(e["x_px"] * s), int(e["y_px"] * s) + 14), MAGENTA, 3, cv2.LINE_AA)
            elif e["kind"] == "bounce" and age < 0.6:
                cc = (int(e["x_px"] * s), int(e["y_px"] * s))
                cv2.circle(img, cc, int(5 + 28 * age), GREEN, 2, cv2.LINE_AA); cv2.circle(img, cc, 3, GREEN, -1, cv2.LINE_AA)
        if pose:                                                   # both players' skeletons, then the knee bend at each hit for 0.9 s
            for end in ("near", "far"):
                sk = (pose.get(f) or {}).get(end)
                if sk is not None:
                    draw_skeleton(img, sk, s, NEAR_COL if end == "near" else FAR_COL)
            for h in posture:
                age = t - h["t"]
                if 0 <= age < 0.9 and h.get("knee") is not None:
                    col = NEAR_COL if h["side"] == "near" else FAR_COL
                    draw_knee_arc(img, h["skel"], h["knee_joint"], s, col)
                    kx, ky = h["skel"][pose_mod.J[h["knee_joint"] + "kne"], :2] * s
                    spr = posture_sprite("KNEES", f"{h['knee']:.0f}\u00b0", NEAR_RGB if h["side"] == "near" else FAR_RGB)
                    off = -(spr.shape[1] / 2 + 26) if h["side"] == "near" else spr.shape[1] / 2 + 26
                    cx = float(np.clip(kx + off, spr.shape[1] / 2 + 4, ow - spr.shape[1] / 2 - 4))
                    _blend(img, spr, cx, float(np.clip(ky, 60, out_h - NOTE_BOTTOM - 30)), alpha=min(1.0, (0.9 - age) / 0.25))
        landed = [e for e in ev_all if e["kind"] == "bounce" and e["t"] <= t]
        crossed = [e for e in ev_all if e["kind"] == "net" and e["t"] <= t]
        if speeds is not None:                                     # the speed off the racket of the shot in play, from its fitted flight
            last_speed = speeds.reading(t)
        else:
            last_speed = last_shot_speed(landed, crossed, last_speed)
        follow(f, found)
        shown = held.update(t, last_speed)
        pop = max(0.0, 1 - (t - held.changed) / 0.2) if shown is not None else 0.0      # the number punches up when it changes
        if celebrate and f == f1:
            live, state = img, (shots, last_speed, needle, last_speed or shown, landed, f)   # the deciding frame: tracking on, furniture not yet
            break
        furniture(img, shots, needle, shown, pop, landed)
        if scoreboard:
            _blend(img, board_before, board_x, board_y)
            if analysis is not None:
                _blend(img, tray_sprite(names, analysis.at(t)), board_x, TRAY_CY)
        if tip_win and tip_win[0] <= t <= tip_win[1]:                 # the scout's note: up from the bottom edge, fuse burning, down again
            span = tip_win[1] - tip_win[0]; u = t - tip_win[0]
            off = (1 - _ease_out_expo(u / 0.38)) if u < 0.38 else _ease_in_cubic((u - (span - 0.30)) / 0.30)
            burn = min(1.0, max(0.0, (u - 0.38) / max(0.1, span - 0.38 - 0.30)))
            _blend(img, tip_sprite(*tip_args, left=int(math.ceil(TIP_STEPS * (1 - burn)))), tip_cx, tip_cy + off * (th_ + SAFE_BOTTOM + 24), scale=tip_sc)
        if crit_win and crit_win[0] <= t <= crit_win[1]:              # the coach's criticism takes over from the tip and stays up
            span = crit_win[1] - crit_win[0]; u = t - crit_win[0]
            off = (1 - _ease_out_expo(u / 0.38)) if u < 0.38 else (_ease_in_cubic((u - (span - 0.25)) / 0.25) if u > span - 0.25 else 0.0)
            _blend(img, crit_spr, crit_cx, crit_cy + off * (ch2 + SAFE_BOTTOM + 24), scale=crit_sc)
        if not winner and t > t_hit:                               # no winner at all: say so (it is not counted)
            note = caption_sprite("WINNER: UNSURE", "NOT COUNTED ON THE SCOREBOARD")
            _blend(img, note, ow // 2 + 60, out_h - NOTE_BOTTOM - 42, scale=0.8)
        if not write(img, fr, base):
            break
    cap.release()

    if live is not None:                                           # ---------------- the scoring moment, over the frozen, inked panel
        shots, last_speed, needle, shot_v, landed, fh = state
        panel = comic_panel(live, seed=point["id"])
        mo = moment or {}
        inten = INTENSITY[mo.get("intensity", 1)]
        burst = burst_sprite(mo.get("word") or action_word(point, _num(last_speed)), seed=point["id"], palette=mo.get("palette", "classic"))
        how = {"long": "LONG OR WIDE", "not_returned": "NOT RETURNED", "double_bounce": "DOUBLE BOUNCE"}.get(point.get("ending"), "")
        line2 = mo.get("line2") or f"{how}  ·  {point.get('n_crossings', 0)} SHOT{'' if point.get('n_crossings', 0) == 1 else 'S'} OVER THE NET".strip(" ·")
        cap_spr = caption_sprite(f"POINT TO {winner.upper()[:14]}!", line2)
        hold_s = inten["hold"]
        spot = None
        if ev_all and "x_px" in ev_all[-1]:
            spot = (ev_all[-1]["x_px"] * s, ev_all[-1]["y_px"] * s)
        elif not np.isnan(track[fh, 2]):
            spot = (track[fh, 2] * s, track[fh, 3] * s)
        tb = quad.reshape(-1, 2)
        forbidden = [b + (14.0,) for b in player_boxes(obs_by_frame, fh, s)]
        for st in tags.values():                                   # the head, the upper body under it, and the name balloon: heavy weights,
            hx, hy = st["x"], st["y"]                              # because these boxes are small and the burst's spikes reach past the box
            forbidden.append((hx - 40, hy - 45, hx + 40, hy + 35, 40.0))                                  # that place() scores
            forbidden.append((hx - 60, hy + 10, hx + 60, hy + 200, 14.0))
            if "box" in st:
                b = st["box"]; forbidden.append((b[0] - 24, b[1] - 20, b[2] + 24, b[3] + 20, 60.0))
        forbidden += [panels[0] + (9.0,), panels[1] + (3.5,), panels[2] + (9.0,),
                      (0, out_h - NOTE_BOTTOM - GAUGE_H - 30, GAUGE_W + 40, out_h, 30.0),
                      (0, out_h - NOTE_BOTTOM - 4, ow, out_h, 20.0),                                # the browser's controls cover this band
                      (float(tb[:, 0].min()), float(tb[:, 1].min()), float(tb[:, 0].max()), float(tb[:, 1].max()), 0.8)]
        if spot:
            forbidden.append((spot[0] - 45, spot[1] - 45, spot[0] + 45, spot[1] + 45, 10.0))
        bh, bw = burst.shape[:2]
        bcx, bcy, bsc = place((bw, bh), forbidden, (ow * (0.42 if win_idx == 0 else 0.58), out_h * 0.30), (ow, out_h),
                              scales=tuple(v * inten["scale"] for v in (0.74, 0.66, 0.58, 0.50, 0.42)), inset=0.74)
        forbidden.append((bcx - bw * bsc * 0.40, bcy - bh * bsc * 0.43, bcx + bw * bsc * 0.40, bcy + bh * bsc * 0.43, 12.0))
        ch_, cw_ = cap_spr.shape[:2]
        ccx, ccy, csc = place((cw_, ch_), forbidden, (ow * 0.5, out_h - NOTE_BOTTOM - 50), (ow, out_h), scales=(1.0, 0.88, 0.76), inset=1.04)
        rays = [(rng.uniform(0, 2 * np.pi), rng.uniform(0.16, 0.26), rng.uniform(0.004, 0.016)) for _ in range(inten["rays"])]
        confetti = [(rng.uniform(0, ow), rng.uniform(-out_h * 0.6, 0), rng.uniform(40, 110), rng.uniform(0, 2 * np.pi), rng.uniform(4, 9),
                     (NEAR_COL, FAR_COL, YELLOW_BGR, RED_BGR)[rng.integers(0, 4)]) for _ in range(90)] if mo.get("intensity") == 3 else []
        focus = spot or (ow / 2, out_h / 2)
        for j in range(int(hold_s * out_fps)):
            tau = j / out_fps
            mix = min(1.0, tau / 0.10)                                 # the footage turns into the inked panel
            img = cv2.addWeighted(panel, mix, live, 1 - mix, 0) if mix < 1 else panel.copy()
            k = inten["punch"] * ((1 - tau / 0.25) ** 2 if tau < 0.25 else 0.0)   # camera punch (by how big the moment is), then a push-in
            z = 1 + 0.06 * k + 0.035 * _ease_out_expo(tau / hold_s)
            M = cv2.getRotationMatrix2D((float(focus[0]), float(focus[1])), rng.uniform(-0.8, 0.8) * k, z)
            M[:, 2] += rng.uniform(-5, 5, 2) * k
            img = cv2.warpAffine(img, M, (ow, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            if tau < 0.5:                                              # impact lines out of the burst
                lay = img.copy(); R = math.hypot(ow, out_h)
                for ang, r0, wdt in rays:
                    p_in = (bcx + math.cos(ang) * R * r0, bcy + math.sin(ang) * R * r0)
                    pa = (bcx + math.cos(ang - wdt) * R, bcy + math.sin(ang - wdt) * R); pb = (bcx + math.cos(ang + wdt) * R, bcy + math.sin(ang + wdt) * R)
                    cv2.fillConvexPoly(lay, np.array([p_in, pa, pb], np.int32), PAPER_BGR, cv2.LINE_AA)
                    cv2.polylines(lay, [np.array([p_in, pa, pb], np.int32)], True, INK_BGR, 1, cv2.LINE_AA)
                a_l = 0.8 * (1 - tau / 0.5)
                cv2.addWeighted(lay, a_l, img, 1 - a_l, 0, img)
            for cx_, cy_, vy_, ph_, sz_, col_ in confetti:              # a game won or a comeback: confetti in the players' colours
                yy = cy_ + vy_ * tau * 3.2; xx = cx_ + 14 * math.sin(ph_ + tau * 5)
                if -10 < yy < out_h:
                    ang = ph_ + tau * 7; dx, dy = math.cos(ang) * sz_, math.sin(ang) * sz_ * 0.45
                    pts_ = np.array([(xx - dx, yy - dy), (xx + dy, yy - dx * 0.4), (xx + dx, yy + dy), (xx - dy, yy + dx * 0.4)], np.int32)
                    cv2.fillConvexPoly(img, pts_, col_, cv2.LINE_AA); cv2.polylines(img, [pts_], True, INK_BGR, 1, cv2.LINE_AA)
            furniture(img, shots, needle, shot_v, 0.0, landed, M)             # panels over the impact lines
            pop = _ease_out_back(tau / 0.22)
            _blend(img, burst, bcx, bcy, scale=bsc * (0.15 + 0.85 * pop), angle=-5 + 1.6 * math.sin(tau * 6.0), alpha=min(1.0, tau / 0.05))
            if tau >= 0.22:                                            # the narration box drops into its free corner
                slide = _ease_out_expo((tau - 0.22) / 0.32)
                _blend(img, cap_spr, ccx, ccy + int((1 - slide) * 60), scale=csc, alpha=slide)
            cv2.rectangle(img, (0, 0), (ow - 1, out_h - 1), INK_BGR, 14); cv2.rectangle(img, (10, 10), (ow - 11, out_h - 11), YELLOW_BGR, 2)
            if scoreboard:
                punch = 1.0 + 0.28 * max(0.0, 1 - abs(tau - 0.12) / 0.12)   # the number punches up as it changes
                _blend(img, board_after if tau >= 0.06 else board_before, board_x, board_y, scale=punch)
                if analysis is not None:                               # the analysis takes this point in as the score does
                    _blend(img, tray_sprite(names, analysis.at(last_t + (5.0 if tau >= 0.06 else 0.0))), board_x, TRAY_CY)
            if tau < 0.09:                                             # flash
                cv2.addWeighted(np.full_like(img, YELLOW_BGR), 0.8 * (1 - tau / 0.09), img, 1 - 0.8 * (1 - tau / 0.09), 0, img)
            if not write(img):
                break
        if replay:                                                  # ---------------- the point again, in slow motion
            n_fade = max(1, int(REPLAY_FADE_S * out_fps)); k = 0
            for img, full, base, reps in replay_frames(video, fps, track, events, pose, point, s, ow, out_h, step, names,
                                                       board_after if scoreboard else None, (board_x, board_y), ppm, critique=critique_after, hide=hide):
                for r_ in range(reps):
                    if not write(img, full, base, fade=min(1.0, (k + 1) / n_fade) if k < n_fade else None):
                        break
                    k += 1
    proc.stdin.close(); proc.wait()
    return proc.returncode == 0


def replay_frames(video, fps, track, events, pose, point, s, ow, out_h, step, names, board, board_xy, ppm, critique=None, hide=None):
    """The last moments of a point again, REPLAY_BEFORE_S before the deciding moment to REPLAY_AFTER_S after it: yields (540-line
    rendering, full-resolution footage, 540-line footage before drawing, how many output frames to hold it for). Half speed, easing to
    quarter speed around the deciding moment; the picture pushes in slowly (to REPLAY_ZOOM) on the spot where the point was decided and
    a soft ring pulses there as it happens. The ball's trail, bounces, crossings and both skeletons are drawn; the score after the point,
    an INSTANT REPLAY caption and the coach's criticism of the player who lost the point sit over it. Calm on purpose: nothing flashes."""
    last_t = max([e["t"] for e in point.get("events", [])] or [point["end_t"]])
    r0 = max(0, int((last_t - REPLAY_BEFORE_S) * fps)); r1 = min(len(track) - 1, int((last_t + REPLAY_AFTER_S) * fps))
    if r1 <= r0:
        return
    cap = cv2.VideoCapture(str(video)); SW, SH = int(cap.get(3)), int(cap.get(4)); cap.set(cv2.CAP_PROP_POS_FRAMES, r0)
    badge = caption_sprite("INSTANT REPLAY", "SLOW MOTION")
    w_ = point.get("winner_name")                                    # the coach talks over the replay, about the player who lost it
    loser = next((n for n in names if n != w_), None) if w_ in names else None
    rows = critique_rows(critique, names, prefer=loser, k=point["id"])
    crit = critique_sprite(rows) if rows else None
    if crit is not None:
        cx_c, cy_c, sc_c = tip_spot((crit.shape[1], crit.shape[0]), [], (ow, out_h))
    evs = [e for e in events if e["kind"] in ("bounce", "net") and "x_px" in e and r0 / fps - 0.6 <= e["t"] <= r1 / fps]
    pev = [e for e in point.get("events", []) if "x_px" in e]
    spot = (pev[-1]["x_px"], pev[-1]["y_px"]) if pev else (SW / 2, SH / 2)
    spot = (float(np.clip(spot[0], SW * 0.2, SW * 0.8)), float(np.clip(spot[1], SH * 0.25, SH * 0.75)))   # never push into a corner
    trail_n = int(0.40 * fps); n_out = 0; total = (r1 - r0) // step + 1
    for f in range(r0, r1 + 1):
        ok, fr = cap.read()
        if not ok:
            break
        if (f - r0) % step:
            continue
        if hide:
            blur_people(fr, hide.get(f))
        t = f / fps
        u = min(1.0, max(0.0, (t - r0 / fps) / max(1e-6, last_t - r0 / fps)))
        z = 1 + (REPLAY_ZOOM - 1) * (u * u * (3 - 2 * u))              # smoothstep: the push-in starts and ends gently
        M = cv2.getRotationMatrix2D(spot, 0, z)
        full = cv2.warpAffine(fr, M, (SW, SH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        img = cv2.resize(full, (ow, out_h), interpolation=cv2.INTER_AREA); base = img.copy()

        def tp(x, y):                                                  # footage pixel -> where the push-in put it, at 540 lines
            return (int((M[0, 0] * x + M[0, 1] * y + M[0, 2]) * s), int((M[1, 0] * x + M[1, 1] * y + M[1, 2]) * s))
        if pose:
            for end in ("near", "far"):
                sk = (pose.get(f) or {}).get(end)
                if sk is not None:
                    sk2 = sk.copy(); sk2[:, :2] = sk[:, :2] @ M[:, :2].T + M[:, 2]
                    draw_skeleton(img, sk2, s, NEAR_COL if end == "near" else FAR_COL)
        seg = track[max(0, f - trail_n):f + 1]; seg = seg[~np.isnan(seg[:, 2])]
        if len(seg):
            seg = seg[seg[:, 4] == seg[-1, 4]]
        if len(seg) > 1:
            gaps = np.maximum(1.0, np.diff(seg[:, 0])); dist = np.hypot(*np.diff(seg[:, 2:4], axis=0).T)
            v = dist / gaps * fps / ppm
            for k in range(1, len(seg)):
                if dist[k - 1] > 110.0 * gaps[k - 1]:
                    continue
                a_ = k / len(seg)
                col = tuple(int(ch * (0.35 + 0.65 * a_)) for ch in speed_colour(v[k - 1]))
                p0, p1 = tp(*seg[k - 1, 2:4]), tp(*seg[k, 2:4])
                cv2.line(img, p0, p1, INK_BGR, 3 + int(3 * a_), cv2.LINE_AA); cv2.line(img, p0, p1, col, 1 + int(3 * a_), cv2.LINE_AA)
        if f < len(track) and not np.isnan(track[f, 2]):
            c_ = tp(track[f, 2], track[f, 3])
            cv2.circle(img, c_, 9, INK_BGR, 4, cv2.LINE_AA); cv2.circle(img, c_, 9, YELLOW_BGR, 2, cv2.LINE_AA)
        for e in evs:
            age = t - e["t"]
            if e["kind"] == "bounce" and 0 <= age < 0.6:
                cc = tp(e["x_px"], e["y_px"])
                cv2.circle(img, cc, int(5 + 28 * age), GREEN, 2, cv2.LINE_AA); cv2.circle(img, cc, 3, GREEN, -1, cv2.LINE_AA)
            elif e["kind"] == "net" and 0 <= age < 0.25:
                x_, y_ = tp(e["x_px"], e["y_px"])
                cv2.line(img, (x_, y_ - 14), (x_, y_ + 14), MAGENTA, 3, cv2.LINE_AA)
        if 0 <= t - last_t < 0.6:                                      # the deciding moment: one soft ring, widening and fading
            q = (t - last_t) / 0.6
            lay = img.copy(); cv2.circle(lay, tp(*spot), int(14 + 60 * q), YELLOW_BGR, 3, cv2.LINE_AA)
            cv2.addWeighted(lay, 0.7 * (1 - q), img, 1 - 0.7 * (1 - q), 0, img)
        cv2.rectangle(img, (0, 0), (ow - 1, out_h - 1), BLUE_BGR, 8)               # a comic-blue frame: this is not live
        cv2.rectangle(img, (6, 6), (ow - 7, out_h - 7), YELLOW_BGR, 2)
        if board is not None:
            _blend(img, board, board_xy[0], board_xy[1])
        slide = _ease_out_expo(min(1.0, n_out / 14))
        _blend(img, badge, 10 + badge.shape[1] * 0.31, 12 + badge.shape[0] * 0.31 - (1 - slide) * 80, scale=0.6)
        if crit is not None:
            cs = _ease_out_expo(min(1.0, max(0.0, (n_out - 6) / 16)))
            _blend(img, crit, cx_c, cy_c + (1 - cs) * (crit.shape[0] + SAFE_BOTTOM + 24), scale=sc_c)
        if total - n_out <= 8:                                        # ease out to black before play resumes
            k = (total - n_out) / 8
            ink_ = np.full_like(img, INK_BGR); ink_f = np.full_like(full, INK_BGR)
            img = cv2.addWeighted(img, k, ink_, 1 - k, 0)
            full = cv2.addWeighted(full, k, ink_f, 1 - k, 0); base = cv2.addWeighted(base, k, ink_, 1 - k, 0)
        n_out += 1
        yield img, full, base, _speed_ramp(t, last_t)
    cap.release()


MATCH_FPS, MATCH_CRF = 30.0, 26          # the whole recording is long: half the frames of a clip and a smaller file, for watching on a phone
MATCH_DOTS = 30                          # landing dots kept on the map; older ones fade off so the map stays readable over a whole match
MATCH_LEAD, MATCH_AFTER = 1.2, 1.6       # seconds the point caption is up before the serve and after the last shot
GONE_S = 2.0                             # seconds without finding a player before their balloon leaves too


def render_match_clip(video, fps, table, track, events, obs_by_frame, points, dst, out_h=540, scoreboard=True, comic=True,
                      context=None, t0=None, t1=None, show_table=False, plate_every_s=8.0, audio=True, pose=None, analysis=None,
                      critiques=None, replay=True, critiques_after=None, speeds=None, moments=None, hide=None):
    from .match_stats import confirm as confirm_point
    """The whole recording, played through with the tracking drawn on: ball trail, bounce rings, speed gauge, landing map, name balloons,
    the running scoreboard, and for each point a caption and, as it is won, a lettered burst over the live picture. No freeze and no
    comic panel: nothing stops the play, and the recording's own sound plays under it (audio=False for a silent file).
    render_point_clip() is the one-point version, with the scoring moment.
    context = overlay_context() of the recording; the hall the heads are found against is rebuilt every plate_every_s seconds from the
    16 s around the current moment, so whoever is standing still behind the table counts as hall rather than as a player."""
    if not shutil.which("ffmpeg"):
        return False
    if speeds is not None:                                             # the gauge reads each shot played, point by point (flight.Speeds)
        from .flight import Speeds
        speeds = Speeds(speeds.shots, points=points)
    cap = cv2.VideoCapture(str(video))
    SW, SH, total = int(cap.get(3)), int(cap.get(4)), int(cap.get(7))
    s = out_h / SH; ow = int(round(SW * s / 2)) * 2
    step = max(1, int(round(fps / MATCH_FPS))); out_fps = fps / step
    f0 = 0 if t0 is None else max(0, int(t0 * fps))
    f1 = min(total - 1, len(track) - 1 if len(track) else total - 1, int(t1 * fps) if t1 is not None else total - 1)
    if f1 <= f0:
        return False
    quad = (table.corners * s).astype(np.int32).reshape(-1, 1, 2)
    table_top = float(quad.reshape(-1, 2)[:, 1].min())
    n1, n2 = (table.to_px([[NET_X, 0.0], [NET_X, W]]) * s).astype(int)
    net_x = float((n1[0] + n2[0]) / 2)
    c = table.corners
    ppm = 0.5 * (np.hypot(*(c[3] - c[0])) + np.hypot(*(c[2] - c[1]))) / L
    first = points[0] if points else {}
    names = (first.get("near_player") or "near", first.get("far_player") or "far")
    scores = running_scores(points)

    live_points = []                                                   # one entry per point: when it is on screen, its boards, its burst
    for p in points:
        sc = scores[p["id"]]
        idx = 0 if p.get("serve_side") == "near" else 1
        win = p.get("winner_name")
        wi = names.index(win) if win in names else None
        last = max([e["t"] for e in p.get("events", [])] or [p["end_t"]])
        ok_, t_c, _ = confirm_point(p, events, track, table, fps)          # celebrate only what the camera saw end
        t_hit = t_c if (ok_ and t_c is not None) else max(last + 0.15, p["end_t"])   # unseen: given when the rules give it, not before
        live_points.append(dict(p=p, t_in=p["start_t"] - MATCH_LEAD, t_hit=t_hit, t_out=max(last + MATCH_AFTER, t_hit + 1.4), confirmed=ok_,
                                before=scoreboard_sprite(names, sc["before"], sc["games"], idx),
                                after=scoreboard_sprite(names, sc["after"], sc.get("games_after", sc["games"]), idx, hot=wi), win=win))
    board_x, board_y = ow // 2 + 14, 62
    map_th = (context or {}).get("map_th", 0)
    standing = (context or {}).get("standing") or heads_mod.standing_zone(table, s)
    map0 = map_sprite(names[0], names[1], map_th)
    hud0 = hud_sprite(1, names[0], 0)
    panels = [(12, 12, 12 + hud0.shape[1], 12 + hud0.shape[0]),
              (board_x - 170, board_y - 39, board_x + 170, board_y + 39),
              (ow - map0.shape[1] - 8, 10, ow - 8, 10 + map0.shape[0])]
    if analysis is not None and scoreboard:
        panels.append((board_x - TRAY_W / 2, TRAY_CY - TRAY_H / 2, board_x + TRAY_W / 2, TRAY_CY + TRAY_H / 2 + 8))
    ev_draw = [e for e in events if e["kind"] in ("bounce", "net") and "x_px" in e]
    bounces = [e for e in ev_draw if e["kind"] == "bounce"]
    sound = ["-ss", f"{f0 / fps:.3f}", "-i", str(video), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "128k",
             "-shortest"] if audio and not replay else []               # the recording's own sound, from where this render starts
    out_path = pathlib.Path(dst).with_suffix(".video.mp4") if replay else pathlib.Path(dst)
    OW, OH = (SW // 2 * 2, SH // 2 * 2) if HIRES else (ow, out_h)
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}", "-r", str(out_fps),
                             "-i", "-"] + sound + ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(MATCH_CRF - 3 if HIRES else MATCH_CRF),
                             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)], stdin=subprocess.PIPE)

    def out_frame(img, full=None, base=None):
        if not HIRES:
            return img
        return _upcompose(img, base, full[:OH, :OW]) if full is not None else cv2.resize(img, (OW, OH), interpolation=cv2.INTER_LANCZOS4)
    inserts = []                                                       # (seconds into this render, seconds of replay put in there)
    # the criticism on screen at any moment: before a point, what the points so far say; from after the last point, the whole match
    crit_segments = []
    prev_end = -1e9
    for lp in live_points:
        rows = critique_rows((critiques or {}).get(lp["p"]["id"]), names, prefer=names[lp["p"]["id"] % 2], k=lp["p"]["id"] // 2)
        crit_segments.append((prev_end, lp["t_hit"] - 0.3, rows))
        prev_end = lp["t_hit"] + 1.35
    crit_spot = {}
    tags = {}
    glide = 1 - math.exp(-(step / fps) / 0.10)

    def follow(f, found):
        """Each end's balloon follows the head found in this frame. Nothing found for GONE_S: that player has left the picture and the
        balloon goes with them, rather than hanging over an empty floor."""
        for end in ("near", "far"):
            st = tags.get(end); h = (found or {}).get(end)
            if h is None:
                if st and f - st["seen"] > GONE_S * fps:
                    tags.pop(end)
                continue
            x, y = h[0], h[1]
            if st is None:
                tags[end] = dict(x=x, y=table_top - 92 if y is None else y, seen=f)
            else:
                st["x"] += glide * (x - st["x"])
                if y is not None:
                    st["y"] += glide * (y - st["y"])
                st["seen"] = f

    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    trail_n = int(0.30 * fps)
    plate = thr = None; plate_until = -1
    last_speed, needle = None, None
    ease = 1 - math.exp(-(step / fps) / NEEDLE_S)
    held = Held()
    burst_of = {}                                                      # point id -> (sprite, caption, x, y, scale) worked out as it is won
    for f in range(f0, f1 + 1):
        ok, fr = cap.read()
        if not ok:
            break
        if (f - f0) % step and f != f1:
            continue
        if hide:
            blur_people(fr, hide.get(f))                             # people who are not the two players, before anything is drawn
        t = f / fps
        img = cv2.resize(fr, (ow, out_h), interpolation=cv2.INTER_AREA)
        base = img.copy()
        if t >= plate_until:                                           # the hall, from the seconds around now
            span = int(PLATE_S * fps); lo = max(0, f - span // 2)
            near_plate = heads_mod.background_plate(video, (ow, out_h), PLATE_N, lo, lo + span)
            if near_plate is not None:
                plate, thr = near_plate, heads_mod.thresholds(near_plate)
            plate_until = t + plate_every_s
        found = heads_mod.find_heads(img, plate, table_top, net_x, thr=thr, standing=standing) if plate is not None else None
        if show_table:
            cv2.polylines(img, [quad], True, GREEN, 1, cv2.LINE_AA); cv2.line(img, tuple(n1), tuple(n2), GREEN, 1, cv2.LINE_AA)
        seg = track[max(0, f - trail_n):f + 1]
        seg = seg[~np.isnan(seg[:, 2])]
        if len(seg):
            seg = seg[seg[:, 4] == seg[-1, 4]]
        v_now = None
        if len(seg) > 1:
            gaps = np.maximum(1.0, np.diff(seg[:, 0]))
            dist = np.hypot(*np.diff(seg[:, 2:4], axis=0).T)
            v = dist / gaps * fps / ppm
            v = np.array([np.median(v[max(0, k - 1):k + 2]) for k in range(len(v))])
            for k in range(1, len(seg)):
                if dist[k - 1] > 110.0 * gaps[k - 1]:
                    continue
                a = k / len(seg)
                vk = speeds.live(float(seg[k, 1])) if speeds is not None else None
                col = tuple(int(ch * (0.35 + 0.65 * a)) for ch in speed_colour(v[k - 1] if vk is None else vk))
                p0, p1 = (seg[k - 1, 2:4] * s).astype(int), (seg[k, 2:4] * s).astype(int)
                cv2.line(img, tuple(p0), tuple(p1), INK_BGR, 3 + int(3 * a), cv2.LINE_AA)
                cv2.line(img, tuple(p0), tuple(p1), col, 1 + int(3 * a), cv2.LINE_AA)
            if seg[-1, 0] == f:
                v_now = float(v[-1])
        if speeds is not None:
            v_now = speeds.live(t)
            if v_now is None and speeds.reading(t) == DASH:
                needle = None                                          # the shot in play was not measured: no needle, not the last one's
        if v_now is not None:
            needle = v_now if needle is None else needle + ease * (v_now - needle)
        if not np.isnan(track[f, 2]):
            centre = (int(track[f, 2] * s), int(track[f, 3] * s))
            cv2.circle(img, centre, 8, INK_BGR, 4, cv2.LINE_AA); cv2.circle(img, centre, 8, speed_colour(v_now or 0.0), 2, cv2.LINE_AA)
        for e in ev_draw:
            age = t - e["t"]
            if e["kind"] == "net" and 0 <= age < 0.25:
                cv2.line(img, (int(e["x_px"] * s), int(e["y_px"] * s) - 14), (int(e["x_px"] * s), int(e["y_px"] * s) + 14), MAGENTA, 3, cv2.LINE_AA)
            elif e["kind"] == "bounce" and 0 <= age < 0.6:
                cc = (int(e["x_px"] * s), int(e["y_px"] * s))
                cv2.circle(img, cc, int(5 + 28 * age), GREEN, 2, cv2.LINE_AA); cv2.circle(img, cc, 3, GREEN, -1, cv2.LINE_AA)
        if pose:                                                       # both players' skeletons (nothing drawn on a face)
            for end in ("near", "far"):
                sk = (pose.get(f) or {}).get(end)
                if sk is not None:
                    draw_skeleton(img, sk, s, NEAR_COL if end == "near" else FAR_COL)
        crossed = [e for e in ev_draw if e["kind"] == "net" and e["t"] <= t]
        landed = [e for e in bounces if e["t"] <= t]
        if speeds is not None:
            last_speed = speeds.reading(t)
        else:
            last_speed = last_shot_speed(landed, crossed, last_speed)
        follow(f, found)
        shown = held.update(t, last_speed)
        pop = max(0.0, 1 - (t - held.changed) / 0.2) if shown is not None else 0.0
        now = next((lp for lp in live_points if lp["t_in"] <= t <= lp["t_out"]), None)
        done = [lp for lp in live_points if t > lp["t_hit"]]
        # --- panels: caption of the point on now, landing map, speed gauge, balloons
        if now is not None:
            shots = sum(1 for e in now["p"].get("events", []) if e["kind"] == "net" and e["t"] <= t)
            _corner(img, hud_sprite(now["p"]["id"], now["p"].get("server") or now["p"].get("serve_side"), shots), 12, 12)
        mx0, my0 = ow - map0.shape[1] - 8, 10
        _corner(img, map0, mx0, my0)
        tx0, ty0, tw_m, th_m = map_frame(map_th)
        on_map = [b for b in landed if -0.08 <= b["x_m"] <= L + 0.08 and -0.08 <= b["y_m"] <= W + 0.08][-MATCH_DOTS:]
        for k, b_ in enumerate(on_map):
            px = int(mx0 + tx0 + np.clip(b_["x_m"] / L, 0, 1) * tw_m); py = int(my0 + ty0 + th_m - np.clip(b_["y_m"] / W, 0, 1) * th_m)
            col = NEAR_COL if b_["side"] == "far" else FAR_COL
            newest = k == len(on_map) - 1
            cv2.circle(img, (px, py), 8 if newest else 6, INK_BGR, -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), 6 if newest else 4, col, -1, cv2.LINE_AA)
            if newest:
                cv2.circle(img, (px, py), 11, YELLOW_BGR, 2, cv2.LINE_AA)
        g = gauge_sprite(); gx, gy = 12, out_h - NOTE_BOTTOM - g.shape[0]
        _corner(img, g, gx, gy)
        bx_, by_, bw_, bh_ = GAUGE_BAR
        if needle is not None:
            mxp = gx + bx_ + int(np.clip(needle / SPEED_STOPS[-1][0], 0, 1) * (bw_ - 1)); myp = gy + by_
            tri = np.array([(mxp - 6, myp - 9), (mxp + 6, myp - 9), (mxp, myp + 1)], np.int32)
            cv2.fillConvexPoly(img, tri, INK_BGR, cv2.LINE_AA); cv2.polylines(img, [tri], True, YELLOW_BGR, 1, cv2.LINE_AA)
            cv2.line(img, (mxp, myp), (mxp, myp + bh_), INK_BGR, 2, cv2.LINE_AA)
        if shown is not None:
            _blend(img, reading_sprite(shown), gx + GAUGE_NUM[0], gy + GAUGE_NUM[1], scale=1.0 + 0.12 * pop)
        heads_now = [(st["x"] - 20, st["y"] - 22, st["x"] + 20, st["y"] + 20) for st in tags.values()]
        if now is not None:                                            # names only while a point is on: between them both players wander
            for end, st in tags.items():                               # off and whoever is left moving by the table is not always a player
                name = names[0] if end == "near" else names[1]; tw = tag_width(name)
                slot, x, y, tail = tag_spot((st["x"], st["y"]), -1 if end == "near" else 1, tw, panels if scoreboard else panels[::2], ow,
                                            st.get("slot"), avoid=heads_now)
                st.update(slot=slot)
                _blend(img, tag_sprite(name, NEAR_RGB if end == "near" else FAR_RGB, tail), int(x), int(y))
        if scoreboard:
            board = None
            if now is not None:
                board = now["after"] if t > now["t_hit"] else now["before"]
            elif done:
                board = done[-1]["after"]
            if board is not None:
                punch = 1.0
                if now is not None and now["t_hit"] < t < now["t_hit"] + 0.24:
                    punch = 1.0 + 0.28 * (1 - abs(t - now["t_hit"] - 0.12) / 0.12)
                _blend(img, board, board_x, board_y, scale=punch)
            if analysis is not None:                                   # each player's analysis stays up the whole match
                _blend(img, tray_sprite(names, analysis.at(t)), board_x, TRAY_CY)
        seg_ = next((c for c in crit_segments if c[0] <= t <= c[1]), None) if critiques else None
        if seg_ and seg_[2]:                                           # the coach's criticism, along the bottom, all match long
            key = seg_[:2]
            if key not in crit_spot:
                spr = critique_sprite(seg_[2]); ch2, cw2 = spr.shape[:2]
                seen = [b for fk in range(f, f + int(2 * fps), max(1, int(fps / 4))) for b in player_boxes(obs_by_frame, fk, s)]
                crit_spot[key] = (spr, ch2) + tuple(tip_spot((cw2, ch2), seen, (ow, out_h)))
            spr, ch2, ccx_, ccy_, csc_ = crit_spot[key]
            u = t - max(seg_[0], f0 / fps); span = seg_[1] - max(seg_[0], f0 / fps)
            off = (1 - _ease_out_expo(u / 0.38)) if u < 0.38 else (_ease_in_cubic((u - (span - 0.25)) / 0.25) if u > span - 0.25 else 0.0)
            _blend(img, spr, ccx_, ccy_ + off * (ch2 + SAFE_BOTTOM + 24), scale=csc_)
        # --- the point is won: a lettered burst and a narration box over the live picture, where nobody is standing
        if comic and now is not None and now["win"] and now.get("confirmed") and now["t_hit"] <= t <= now["t_hit"] + 1.25:
            pid = now["p"]["id"]
            if pid not in burst_of:
                mo = (moments or {}).get(pid) or {}
                spr = burst_sprite(mo.get("word") or action_word(now["p"], _num(last_speed)), seed=pid, palette=mo.get("palette", "classic"))
                cap_spr = caption_sprite(f"POINT TO {now['win'].upper()[:14]}!", mo.get("line2") or f"{now['p'].get('n_crossings', 0)} SHOTS OVER THE NET")
                forbidden = [b + (14.0,) for b in player_boxes(obs_by_frame, f, s)]
                for st in tags.values():
                    forbidden.append((st["x"] - 40, st["y"] - 45, st["x"] + 40, st["y"] + 35, 40.0))
                    forbidden.append((st["x"] - 60, st["y"] + 10, st["x"] + 60, st["y"] + 200, 14.0))
                forbidden += [panels[0] + (9.0,), panels[1] + (3.5,), panels[2] + (9.0,),
                              (0, out_h - NOTE_BOTTOM - GAUGE_H - 30, GAUGE_W + 40, out_h, 30.0),
                              (0, out_h - NOTE_BOTTOM - 4, ow, out_h, 20.0)]
                bh, bw = spr.shape[:2]
                bcx, bcy, bsc = place((bw, bh), forbidden, (ow * 0.5, out_h * 0.30), (ow, out_h),
                                      scales=tuple(v * INTENSITY[mo.get("intensity", 1)]["scale"] for v in (0.62, 0.54, 0.46, 0.38)))
                forbidden.append((bcx - bw * bsc * 0.40, bcy - bh * bsc * 0.43, bcx + bw * bsc * 0.40, bcy + bh * bsc * 0.43, 12.0))
                ch_, cw_ = cap_spr.shape[:2]
                ccx, ccy, csc = place((cw_, ch_), forbidden, (ow * 0.5, out_h - NOTE_BOTTOM - 50), (ow, out_h), scales=(0.9, 0.8, 0.7))
                burst_of[pid] = (spr, cap_spr, bcx, bcy, bsc, ccx, ccy, csc)
            spr, cap_spr, bcx, bcy, bsc, ccx, ccy, csc = burst_of[pid]
            tau = t - now["t_hit"]
            fade = min(1.0, (now["t_hit"] + 1.25 - t) / 0.3)
            _blend(img, spr, bcx, bcy, scale=bsc * (0.15 + 0.85 * _ease_out_back(min(1.0, tau / 0.22))),
                   angle=-5 + 1.6 * math.sin(tau * 6.0), alpha=min(1.0, tau / 0.05) * fade)
            if tau >= 0.22:
                slide = _ease_out_expo((tau - 0.22) / 0.32)
                _blend(img, cap_spr, ccx, ccy + int((1 - slide) * 60), scale=csc, alpha=slide * fade)
        last = out_frame(img, fr, base)
        try:
            proc.stdin.write(last.tobytes())
        except BrokenPipeError:
            break
        if replay:                                                     # the burst is over: the point again, in slow motion
            for lp in live_points:
                if lp["t_hit"] + 1.3 < f0 / fps:                       # won before this render starts: no replay
                    lp["replayed"] = True
                if lp["win"] and not lp.get("replayed") and t >= lp["t_hit"] + 1.3:
                    lp["replayed"] = True; n = 0; n_fade = max(1, int(REPLAY_FADE_S * out_fps))
                    for rimg, rfull, rbase, reps in replay_frames(video, fps, track, events, pose, lp["p"], s, ow, out_h, step, names,
                                                                  lp["after"] if scoreboard else None, (board_x, board_y), ppm,
                                                                  critique=(critiques_after or {}).get(lp["p"]["id"]), hide=hide):
                        o = out_frame(rimg, rfull, rbase)
                        for _ in range(reps):
                            w_ = o if n >= n_fade else cv2.addWeighted(o, (n + 1) / n_fade, last, 1 - (n + 1) / n_fade, 0)
                            proc.stdin.write(w_.tobytes()); n += 1
                    if n:
                        inserts.append((t - f0 / fps + 1 / out_fps, n / out_fps))
    cap.release()
    proc.stdin.close(); proc.wait()
    if proc.returncode != 0:
        return False
    if replay:
        _mux_with_gaps(out_path, video, f0 / fps, inserts, pathlib.Path(dst), audio)
    return True


def _mux_with_gaps(video_only, source, t0, inserts, dst, audio=True):
    """Lay the recording's own sound under a video that has replays put in: the sound up to each replay, silence as long as the replay,
    then the sound carries on where it stopped. Without an audio track (or audio=False) the video is just moved into place."""
    has_audio = False
    if audio:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(source)],
                           capture_output=True, text=True)
        has_audio = bool(r.stdout.strip())
    if not has_audio:
        pathlib.Path(video_only).replace(dst); return
    fmt = "aformat=sample_rates=48000:channel_layouts=stereo"
    filt, parts, k, prev = [], [], 0, 0.0
    for ti, gap in inserts:
        filt.append(f"[1:a]atrim=start={prev:.4f}:end={ti:.4f},asetpts=PTS-STARTPTS,{fmt}[a{k}]"); parts.append(f"[a{k}]"); k += 1
        filt.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={gap:.4f},{fmt}[a{k}]"); parts.append(f"[a{k}]"); k += 1
        prev = ti
    filt.append(f"[1:a]atrim=start={prev:.4f},asetpts=PTS-STARTPTS,{fmt}[a{k}]"); parts.append(f"[a{k}]"); k += 1
    filt.append("".join(parts) + f"concat=n={k}:v=0:a=1[aout]")
    script = pathlib.Path(dst).with_suffix(".audio.txt"); script.write_text(";\n".join(filt))
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video_only), "-ss", f"{t0:.3f}", "-i", str(source), "-/filter_complex", str(script),
                        "-map", "0:v:0", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(dst)],
                       capture_output=True, text=True)
    if r.returncode == 0:
        pathlib.Path(video_only).unlink(missing_ok=True); script.unlink(missing_ok=True)
    else:                                                              # keep the picture even if the sound could not be laid in
        print("sound not added:", r.stderr[-400:]); pathlib.Path(video_only).replace(dst)
