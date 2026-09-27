"""A small 3D renderer for table tennis plates and their video: a perspective camera with near-plane clipping, drawing in call order
(painter's), group opacity, soft shadows cast straight down, and supersampled anti-aliasing with OpenCV; labels are set afterwards at
full resolution with PIL so the type stays crisp. Used by the after-match analysis (analysis3d.py).

World frame = the table's (camera.py): X along the table (0 at one end line, L at the other), Y across it, Z up from the playing
surface; the floor is 76 cm down.

    sc = Scene(eye=(-4, -1, 3), target=(0, 0.76, -0.5), theme="light")
    sc.floor(); sc.table()
    with sc.group(0.4):
        sc.line(path, sc.T["p1"], 1.5)
    sc.sphere(p, 0.03, sc.T["p1"])
    sc.label(p, "0.70 m", "behind the end line")
    img = sc.finish()                    # BGR uint8, size (w, h)
"""
import math
from contextlib import contextmanager
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from .config import TABLE_LENGTH as L, TABLE_WIDTH as W, NET_X

FLOOR = -0.76
NET_H, NET_OUT = 0.1525, 0.1525
SHIFT = 4                                                            # OpenCV sub-pixel bits: coordinates are drawn at 1/16 px


def oklch(l, c, h):
    """OKLCH (lightness 0 to 1, chroma, hue degrees) to an sRGB tuple 0 to 255, so the plates use the report's own colour tokens."""
    a, b = c * math.cos(math.radians(h)), c * math.sin(math.radians(h))
    l_, m_, s_ = l + 0.3963377774 * a + 0.2158037573 * b, l - 0.1055613458 * a - 0.0638541728 * b, l - 0.0894841775 * a - 1.2914855480 * b
    l3, m3, s3 = l_ ** 3, m_ ** 3, s_ ** 3
    rgb = (4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3, -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3,
           -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3)
    g = lambda x: 12.92 * x if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055
    return tuple(int(round(255 * g(min(1.0, max(0.0, x))))) for x in rgb)


def mix(a, b, t):
    """Colour a moved t of the way to b."""
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


# the report's tokens (report_html.py :root), light and dark; "comic" matches the clips' overlays (annotate.py)
THEMES = {
    "light": dict(paper=oklch(.955, .014, 85), floor=oklch(.93, .017, 85), floor2=oklch(.905, .02, 85), grid=oklch(.87, .02, 85),
                  ink=oklch(.21, .025, 165), ink2=oklch(.36, .02, 165), ink3=oklch(.48, .016, 165), table=oklch(.36, .075, 225),
                  table_edge=oklch(.25, .05, 225), line=oklch(.95, .015, 85), frame=oklch(.33, .02, 165), net=oklch(.21, .025, 165),
                  p1=oklch(.64, .155, 55), p2=oklch(.55, .12, 240), rust=oklch(.50, .13, 48), good=oklch(.43, .09, 155),
                  shadow=oklch(.35, .03, 85), pro=oklch(.21, .025, 165), halo=oklch(.955, .014, 85)),
    "dark": dict(paper=oklch(.145, .004, 250), floor=oklch(.172, .005, 250), floor2=oklch(.19, .006, 250), grid=oklch(.235, .006, 250),
                 ink=oklch(.96, .004, 90), ink2=oklch(.80, .006, 250), ink3=oklch(.63, .008, 250), table=oklch(.44, .085, 232),
                 table_edge=oklch(.31, .06, 232), line=oklch(.92, .01, 90), frame=oklch(.34, .008, 250), net=oklch(.96, .004, 90),
                 p1=oklch(.75, .15, 58), p2=oklch(.72, .12, 238), rust=oklch(.64, .17, 35), good=oklch(.78, .14, 152),
                 shadow=oklch(.05, .004, 250), pro=oklch(.96, .004, 90), halo=oklch(.145, .004, 250)),
    "comic": dict(paper=(255, 246, 190), floor=(255, 236, 168), floor2=(250, 226, 150), grid=(236, 208, 132), ink=(34, 32, 96),
                  ink2=(58, 56, 128), ink3=(92, 90, 150), table=(35, 95, 205), table_edge=(24, 60, 150), line=(255, 246, 190),
                  frame=(34, 32, 96), net=(34, 32, 96), p1=(255, 138, 0), p2=(40, 150, 235), rust=(228, 38, 38), good=(30, 150, 80),
                  shadow=(120, 90, 40), pro=(34, 32, 96), halo=(255, 246, 190)),
}

FONTS = dict(sans="/System/Library/Fonts/Avenir Next.ttc", cond="/System/Library/Fonts/Avenir Next Condensed.ttc",
             serif="/System/Library/Fonts/Supplemental/Iowan Old Style.ttc")
# "serif" is the display face of the report: Avenir Next Condensed since the black edition (the name is kept for the callers)
FACES = dict(regular=("sans", 7), medium=("sans", 5), demi=("sans", 2), bold=("sans", 0), serif=("cond", 2), serif_bold=("cond", 0),
             serif_italic=("cond", 3))
_font_cache = {}


def font(face, size):
    key = (face, int(size))
    if key not in _font_cache:
        fam, idx = FACES[face]
        try:
            _font_cache[key] = ImageFont.truetype(FONTS[fam], int(size), index=idx)
        except Exception:
            try:
                _font_cache[key] = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", int(size))
            except Exception:
                _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]


def bgr(c):
    return (int(c[2]), int(c[1]), int(c[0]))


def _clip_poly(X, near):
    """Sutherland-Hodgman against the plane z = near, in camera coordinates."""
    out = []
    for i in range(len(X)):
        a, b = X[i], X[(i + 1) % len(X)]
        ina, inb = a[2] >= near, b[2] >= near
        if ina:
            out.append(a)
        if ina != inb:
            out.append(a + (near - a[2]) / (b[2] - a[2]) * (b - a))
    return np.array(out)


def _clip_line(X, near):
    """A polyline cut into the runs in front of the plane z = near."""
    runs, cur = [], []
    for i in range(len(X)):
        a = X[i]
        if a[2] >= near:
            if not cur and i > 0 and X[i - 1][2] < near:
                b = X[i - 1]; cur.append(b + (near - b[2]) / (a[2] - b[2]) * (a - b))
            cur.append(a)
        elif cur:
            b = X[i - 1]; cur.append(b + (near - b[2]) / (a[2] - b[2]) * (a - b)); runs.append(np.array(cur)); cur = []
    if len(cur) > 1:
        runs.append(np.array(cur))
    return [r for r in runs if len(r) > 1]


def _dashes(uv, on, off):
    """Cut a 2D polyline into dashes (lengths in pixels)."""
    seg = np.hypot(*np.diff(uv, axis=0).T); s = np.r_[0, np.cumsum(seg)]
    out, t = [], 0.0
    while t < s[-1]:
        t1 = min(t + on, s[-1])
        ts = np.r_[t, s[(s > t) & (s < t1)], t1]
        out.append(np.stack([np.interp(ts, s, uv[:, 0]), np.interp(ts, s, uv[:, 1])], axis=1))
        t = t1 + off
    return out


def ellipse_pts(c, cov, k=1.5, n=72, z=0.0):
    """The k-sigma ellipse of a 2D covariance around c, as 3D points at height z."""
    w, V = np.linalg.eigh(np.asarray(cov, float))
    w = np.sqrt(np.maximum(w, 1e-6)) * k
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = np.asarray(c, float) + (np.stack([np.cos(a) * w[0], np.sin(a) * w[1]], axis=1) @ V.T)
    return np.c_[xy, np.full(n, z)]


class Scene:
    def __init__(self, eye, target, fov=40.0, size=(1600, 900), theme="light", ss=3):
        self.T = THEMES[theme] if isinstance(theme, str) else theme
        self.w, self.h, self.ss = int(size[0]), int(size[1]), int(ss)
        eye, target = np.asarray(eye, float), np.asarray(target, float)
        f = target - eye; f /= np.linalg.norm(f)
        up = np.array([0, 0, 1.0]) if abs(f[2]) < 0.97 else np.array([1.0, 0, 0])
        r = np.cross(f, up); r /= np.linalg.norm(r); u = np.cross(r, f)
        self.R, self.eye = np.stack([r, -u, f]), eye
        self.fp = (self.w * self.ss / 2) / math.tan(math.radians(fov) / 2)
        self.c = np.array([self.w * self.ss / 2, self.h * self.ss / 2])
        self.img = np.empty((self.h * self.ss, self.w * self.ss, 3), np.uint8); self.img[:] = bgr(self.T["paper"])
        self.tgt, self.sh, self.near = self.img, None, 0.05
        self.labels, self.overlays = [], []

    # ---- projection
    def cam(self, P):
        return (np.atleast_2d(np.asarray(P, float)) - self.eye) @ self.R.T

    def _uv(self, Xc):
        return self.fp * Xc[:, :2] / Xc[:, 2:3] + self.c

    def project(self, P):
        """World points to final-image pixels (and their depth); for placing labels."""
        Xc = self.cam(P)
        return self._uv(np.maximum(Xc, [[-1e9, -1e9, 1e-3]])) / self.ss, Xc[:, 2]

    def px(self, p, d=1.0):
        """Screen pixels (final image) per metre at world point p: for sizing things drawn at a distance."""
        return self.fp / self.ss / max(float(self.cam([p])[0, 2]), 1e-3) * d

    def _pts(self, uv):
        return np.round(uv * (1 << SHIFT)).astype(np.int32).reshape(-1, 1, 2)

    # ---- primitives (world coordinates; widths in final-image pixels)
    def poly(self, P, col, edge=None, width=1.0):
        Xc = _clip_poly(self.cam(P), self.near)
        if len(Xc) < 3:
            return
        pts = self._pts(self._uv(Xc))
        cv2.fillPoly(self.tgt, [pts], bgr(col), cv2.LINE_AA, SHIFT)
        if edge is not None:
            cv2.polylines(self.tgt, [pts], True, bgr(edge), max(1, int(round(width * self.ss))), cv2.LINE_AA, SHIFT)

    def line(self, P, col, width=1.0, dash=None, closed=False):
        P = np.asarray(P, float)
        if closed:
            P = np.r_[P, P[:1]]
        for run in _clip_line(self.cam(P), self.near):
            uv = self._uv(run)
            for piece in (_dashes(uv, dash[0] * self.ss, dash[1] * self.ss) if dash else [uv]):
                if len(piece) > 1:
                    cv2.polylines(self.tgt, [self._pts(piece)], False, bgr(col), max(1, int(round(width * self.ss))), cv2.LINE_AA, SHIFT)

    def tube(self, P, col, width=4.0, dark=0.35, light=0.45):
        """A trajectory drawn as a shaded tube: a darker rim, the colour, a light core."""
        self.line(P, mix(col, (0, 0, 0), dark), width)
        self.line(P, col, width * 0.72)
        self.line(P, mix(col, (255, 255, 255), light), max(1.0, width * 0.22))

    def sphere(self, p, r, col, hollow=False, edge=None, width=1.2):
        Xc = self.cam([p])[0]
        if Xc[2] < self.near:
            return
        (x, y), rr = self._uv(Xc[None])[0], self.fp * r / Xc[2]
        k = 1 << SHIFT
        c = (int(round(x * k)), int(round(y * k))); R = max(2 * k, int(round(rr * k)))
        if hollow:
            cv2.circle(self.tgt, c, R, bgr(col), max(1, int(round(width * self.ss))), cv2.LINE_AA, SHIFT)
            return
        cv2.circle(self.tgt, c, R, bgr(mix(col, (0, 0, 0), 0.28)), -1, cv2.LINE_AA, SHIFT)                 # the shaded side
        o = int(R * 0.14)
        cv2.circle(self.tgt, (c[0] - o, c[1] - o), int(R * 0.84), bgr(col), -1, cv2.LINE_AA, SHIFT)        # the lit side, light from top left
        cv2.circle(self.tgt, (c[0] - int(R * 0.38), c[1] - int(R * 0.38)), max(k, int(R * 0.26)), bgr(mix(col, (255, 255, 255), 0.55)), -1, cv2.LINE_AA, SHIFT)
        cv2.circle(self.tgt, c, R, bgr(edge or mix(col, (0, 0, 0), 0.55)), max(1, int(round(width * self.ss * 0.6))), cv2.LINE_AA, SHIFT)

    def disc(self, c, r, z, col, n=36, edge=None, width=1.0, sx=1.0, sy=1.0, angle=0.0):
        """A flat ellipse lying on the plane z (a footprint, a landing spot)."""
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        ca, sa = math.cos(angle), math.sin(angle)
        ex, ey = np.cos(a) * r * sx, np.sin(a) * r * sy
        P = np.stack([c[0] + ex * ca - ey * sa, c[1] + ex * sa + ey * ca, np.full(n, z)], axis=1)
        self.poly(P, col, edge, width)

    def arrow_flat(self, a, b, z, col, width=0.05, head=0.16):
        """An arrow lying on a horizontal plane, from a to b (metres)."""
        a, b = np.asarray(a[:2], float), np.asarray(b[:2], float)
        d = b - a; n = np.linalg.norm(d)
        if n < 1e-3:
            return
        d /= n; q = np.array([-d[1], d[0]]); hb = b - d * min(head, n * 0.6)
        body = [a + q * width / 2, hb + q * width / 2, hb - q * width / 2, a - q * width / 2]
        tip = [hb + q * width * 1.9, b, hb - q * width * 1.9]
        self.poly([[p[0], p[1], z] for p in body], col); self.poly([[p[0], p[1], z] for p in tip], col)

    # ---- group opacity and shadows
    @contextmanager
    def group(self, alpha):
        """Everything drawn inside is composited at this opacity as one layer (overlaps don't pile up)."""
        base = self.tgt; self.tgt = base.copy()
        try:
            yield
        finally:
            cv2.addWeighted(self.tgt, alpha, base, 1 - alpha, 0, dst=base); self.tgt = base

    @contextmanager
    def shadows(self, strength=0.3, blur=6.0):
        """Shadows drawn inside (shadow_*) are blurred (blur in final pixels) and darken what is already drawn."""
        self.sh = np.zeros(self.img.shape[:2], np.uint8)
        try:
            yield
        finally:
            m = cv2.GaussianBlur(self.sh, (0, 0), blur * self.ss).astype(np.float32) * (strength / 255.0)
            col = np.array(bgr(self.T["shadow"]), np.float32)
            t = self.tgt.astype(np.float32)
            t += (col - t) * m[..., None]
            self.tgt[:] = np.clip(t, 0, 255).astype(np.uint8); self.sh = None

    def shadow_poly(self, P):
        Xc = _clip_poly(self.cam(P), self.near)
        if len(Xc) >= 3:
            cv2.fillPoly(self.sh, [self._pts(self._uv(Xc))], 255, cv2.LINE_AA, SHIFT)

    def shadow_line(self, P, width):
        for run in _clip_line(self.cam(P), self.near):
            cv2.polylines(self.sh, [self._pts(self._uv(run))], False, 255, max(1, int(round(width * self.ss))), cv2.LINE_AA, SHIFT)

    def shadow_disc(self, c, r, z, n=28):
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        self.shadow_poly(np.stack([c[0] + np.cos(a) * r, c[1] + np.sin(a) * r, np.full(n, z)], axis=1))

    @staticmethod
    def under(p):
        """Where a point's shadow falls with the light straight above: on the table if it is over it, else on the floor."""
        x, y = p[0], p[1]
        return 0.0 if (0 <= x <= L and 0 <= y <= W and p[2] >= 0) else FLOOR

    def shadow_path(self, P, width, table_only=False):
        """A flight's shadow, on the table where it is over the table and on the floor elsewhere (or only the table's part)."""
        P = np.asarray(P, float)
        z = np.array([self.under(p) for p in P])
        start = 0
        for i in range(1, len(P) + 1):
            if i == len(P) or z[i] != z[start]:
                seg = P[max(0, start - (1 if start else 0)):i].copy()
                seg[:, 2] = z[start]
                if len(seg) > 1 and not (table_only and z[start] != 0.0):
                    self.shadow_line(seg, width)
                start = i

    # ---- the set
    def floor(self, x0=-4.5, x1=L + 3.5, y0=-3.5, y1=W + 3.5, bands=True, grid=0.5):
        T = self.T
        self.poly([[x0, y0, FLOOR], [x1, y0, FLOOR], [x1, y1, FLOOR], [x0, y1, FLOOR]], T["floor"])
        if bands:                                                    # half-metre bands behind each end line
            for k in range(4):
                if k % 2 == 0:
                    for xa, xb in ((-(k + 1) * 0.5, -k * 0.5), (L + k * 0.5, L + (k + 1) * 0.5)):
                        self.poly([[xa, -0.9, FLOOR], [xb, -0.9, FLOOR], [xb, W + 0.9, FLOOR], [xa, W + 0.9, FLOOR]], T["floor2"])
        for x in np.arange(math.ceil(x0 / grid) * grid, x1 + 1e-6, grid):
            self.line([[x, y0, FLOOR], [x, y1, FLOOR]], T["grid"], 1.0)
        for y in np.arange(math.ceil(y0 / grid) * grid, y1 + 1e-6, grid):
            self.line([[x0, y, FLOOR], [x1, y, FLOOR]], T["grid"], 1.0)

    def table(self, th=0.03):
        T = self.T
        with self.shadows(0.16, 20):                                 # the table's shadow on the floor
            self.shadow_poly([[0.05, 0.05, FLOOR], [L - 0.05, 0.05, FLOOR], [L - 0.05, W - 0.05, FLOOR], [0.05, W - 0.05, FLOOR]])
        leg = 0.028
        for x, y in ((0.35, 0.18), (0.35, W - 0.18), (L - 0.35, 0.18), (L - 0.35, W - 0.18)):
            self._box(x - leg, x + leg, y - leg, y + leg, FLOOR, -th, T["frame"])
        for x in (0.35, L - 0.35):                                   # the under-frame across each end
            self._box(x - 0.02, x + 0.02, 0.18, W - 0.18, -0.12, -th, T["frame"])
        self._box(0.35, L - 0.35, W / 2 - 0.02, W / 2 + 0.02, -0.1, -th, T["frame"])
        self._box(0, L, 0, W, -th, 0, T["table"], top=T["table"], side=T["table_edge"])
        lw, cw = 0.02, 0.004
        for q in ([0, 0, L, lw], [0, W - lw, L, W], [0, 0, lw, W], [L - lw, 0, L, W], [0, W / 2 - cw, L, W / 2 + cw]):
            self.poly([[q[0], q[1], 0.0005], [q[2], q[1], 0.0005], [q[2], q[3], 0.0005], [q[0], q[3], 0.0005]], T["line"])

    def net(self):
        T = self.T
        y0, y1 = -NET_OUT, W + NET_OUT
        with self.group(0.22):
            self.poly([[NET_X, y0, 0], [NET_X, y1, 0], [NET_X, y1, NET_H], [NET_X, y0, NET_H]], T["net"])
        with self.group(0.35):
            for y in np.arange(y0, y1, 0.05):
                self.line([[NET_X, y, 0], [NET_X, y, NET_H]], T["net"], 0.6)
        self.poly([[NET_X, y0, NET_H - 0.015], [NET_X, y1, NET_H - 0.015], [NET_X, y1, NET_H], [NET_X, y0, NET_H]], T["line"],
                  edge=mix(T["line"], T["ink"], 0.4), width=0.6)
        for y in (y0, y1):
            self._box(NET_X - 0.012, NET_X + 0.012, y - 0.012, y + 0.012, 0, NET_H + 0.005, T["frame"])

    def _box(self, x0, x1, y0, y1, z0, z1, col, top=None, side=None):
        """An axis-aligned box: the faces turned to the camera, lit from above."""
        side = side or mix(col, (0, 0, 0), 0.25)
        faces = [([[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]], (0, -1, 0)), ([[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]], (1, 0, 0)),
                 ([[x1, y1, z0], [x0, y1, z0], [x0, y1, z1], [x1, y1, z1]], (0, 1, 0)), ([[x0, y1, z0], [x0, y0, z0], [x0, y0, z1], [x0, y1, z1]], (-1, 0, 0)),
                 ([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], (0, 0, 1))]
        c = np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
        vis = []
        for P, n in faces:
            fc = np.mean(P, axis=0)
            if np.dot(self.eye - fc, n) > 0:
                vis.append((np.linalg.norm(self.eye - fc), P, n))
        for _, P, n in sorted(vis, key=lambda v: -v[0]):
            self.poly(P, (top or col) if n[2] > 0 else (side if n[1] == 0 else mix(side, (0, 0, 0), 0.12)))

    # ---- bodies (body3d.py joints): a clean mannequin, no face
    CAPS = [("left_hip", "left_knee", 0.072), ("right_hip", "right_knee", 0.072), ("left_knee", "left_ankle", 0.054), ("right_knee", "right_ankle", 0.054),
            ("left_shoulder", "left_elbow", 0.046), ("right_shoulder", "right_elbow", 0.046), ("left_elbow", "left_wrist", 0.038),
            ("right_elbow", "right_wrist", 0.038), ("center_head", "center_shoulder", 0.045), ("left_shoulder", "right_shoulder", 0.058),
            ("left_hip", "right_hip", 0.085), ("center_shoulder", "spine", 0.125), ("spine", "root", 0.115)]

    def _cap_items(self, a, b, r):
        Xc = self.cam([a, b])
        if min(Xc[:, 2]) < self.near:
            return None
        uv = self._uv(Xc); z = float(Xc[:, 2].mean())
        return (z, "cap", (uv, max(1.0, 2 * self.fp * r / z)))

    def body(self, J, col, ghost=False, racket=None, facing=None):
        """A mannequin from 17 joints (body3d.NAMES order): capsule limbs shaded as tubes, a torso of stacked capsules, a head ball,
        feet towards facing (a floor direction), and a racket (dict(hand, towards)) in one hand. ghost: flat, for laying over."""
        from .body3d import I
        items = []
        for a, b, r in self.CAPS:
            it = self._cap_items(J[I[a]], J[I[b]], r)
            if it:
                items.append(it + (col,))
        fwd = np.asarray(facing if facing is not None else (1.0, 0.0), float); fwd = fwd / max(np.linalg.norm(fwd), 1e-9)
        for s in ("left", "right"):                                    # feet: from under the ankle, toes forward
            an = J[I[f"{s}_ankle"]]
            heel = np.array([an[0] - 0.04 * fwd[0], an[1] - 0.04 * fwd[1], an[2] - 0.05]); toe = heel + np.r_[0.19 * fwd, 0.0]
            it = self._cap_items(heel, toe, 0.036)
            if it:
                items.append(it + (col,))
        for s in ("left", "right"):
            Xc = self.cam([J[I[f"{s}_wrist"]]])[0]
            if Xc[2] > self.near:
                items.append((float(Xc[2]), "ball", (J[I[f"{s}_wrist"]], 0.042)) + (col,))
        Xc = self.cam([J[I["center_head"]]])[0]
        if Xc[2] > self.near:
            items.append((float(Xc[2]), "ball", (J[I["center_head"]] + [0, 0, 0.02], 0.105)) + (col,))
        if racket:                                                     # the blade a hand's length beyond the wrist, the handle to it
            w, e = J[I[f"{racket['hand']}_wrist"]], J[I[f"{racket['hand']}_elbow"]]
            d = np.asarray(racket.get("towards"), float) - w if racket.get("towards") is not None else w - e
            d = d / max(np.linalg.norm(d), 1e-9)
            blade = w + 0.17 * d
            it = self._cap_items(w, w + 0.1 * d, 0.014)
            if it:
                items.append(it + (self.T["frame"] if not ghost else col,))
            Xc = self.cam([blade])[0]
            if Xc[2] > self.near:
                items.append((float(Xc[2]) - 0.01, "blade", (blade, 0.078)) + (col,))
        k = 1 << SHIFT
        for z, kind, data, c in sorted(items, key=lambda t: -t[0]):
            if kind == "cap":
                uv, w = data
                p0, p1 = (int(round(uv[0, 0] * k)), int(round(uv[0, 1] * k))), (int(round(uv[1, 0] * k)), int(round(uv[1, 1] * k)))
                if ghost:
                    cv2.line(self.tgt, p0, p1, bgr(c), max(1, int(w)), cv2.LINE_AA, SHIFT)
                    continue
                o = int(round(0.13 * w * k))
                cv2.line(self.tgt, p0, p1, bgr(mix(c, (0, 0, 0), 0.42)), max(1, int(w)), cv2.LINE_AA, SHIFT)
                cv2.line(self.tgt, (p0[0] - o, p0[1] - o), (p1[0] - o, p1[1] - o), bgr(c), max(1, int(w * 0.72)), cv2.LINE_AA, SHIFT)
                o2 = int(round(0.24 * w * k))
                cv2.line(self.tgt, (p0[0] - o2, p0[1] - o2), (p1[0] - o2, p1[1] - o2), bgr(mix(c, (255, 255, 255), 0.35)), max(1, int(w * 0.2)), cv2.LINE_AA, SHIFT)
            elif kind == "ball":
                if ghost:
                    Xc = self.cam([data[0]])[0]; uv = self._uv(Xc[None])[0]
                    cv2.circle(self.tgt, (int(uv[0] * k), int(uv[1] * k)), int(self.fp * data[1] / Xc[2] * k), bgr(c), -1, cv2.LINE_AA, SHIFT)
                else:
                    self.sphere(data[0], data[1], c, edge=mix(c, (0, 0, 0), 0.42), width=0.8)
            elif kind == "blade":
                Xc = self.cam([data[0]])[0]; uv = self._uv(Xc[None])[0]; R = int(self.fp * data[1] / Xc[2] * k)
                red = self.T["rust"] if not ghost else c
                cv2.circle(self.tgt, (int(uv[0] * k), int(uv[1] * k)), R, bgr(mix(red, (0, 0, 0), 0.35) if not ghost else red), -1, cv2.LINE_AA, SHIFT)
                if not ghost:
                    cv2.circle(self.tgt, (int(uv[0] * k) - R // 8, int(uv[1] * k) - R // 8), int(R * 0.82), bgr(red), -1, cv2.LINE_AA, SHIFT)

    # ---- annotation (placed now, drawn by finish())
    def label(self, p, text, sub=None, dx=0, dy=-50, col=None, align="left", size=26, sub_size=17, leader=True, dot=True, face="serif"):
        """A number or short phrase tied to a world point by a leader line (final-image pixels offset dx, dy)."""
        uv, _ = self.project([p]); x, y = float(uv[0, 0]), float(uv[0, 1])
        self.labels.append(dict(x=x, y=y, tx=x + dx, ty=y + dy, text=text, sub=sub, col=col, align=align, size=size, sub_size=sub_size,
                                leader=leader, dot=dot, face=face))

    def text(self, xy, text, face="regular", size=18, col=None, anchor="la", halo=False, spacing=0):
        self.overlays.append(("text", xy, text, face, size, col, anchor, halo, spacing))

    def rect(self, box, col, alpha=1.0):
        self.overlays.append(("rect", box, col, alpha))

    def finish(self):
        """Leaders at supersampled resolution, then down to size, then type."""
        T, k = self.T, 1 << SHIFT
        for lb in self.labels:                                        # keep every label inside the picture
            f1 = font(lb["face"], lb["size"]); f2 = font("medium", lb["sub_size"])
            wt = max([f1.getlength(lb["text"])] + [f2.getlength(x) for x in (lb["sub"] or "").split("\n") if x])
            x0 = lb["tx"] - {"left": 0, "right": wt, "center": wt / 2}[lb["align"]]
            lb["tx"] += max(0, 24 - x0) - max(0, x0 + wt - (self.w - 24))
            lb["ty"] = max(lb["ty"], lb["size"] + 16)
        for lb in self.labels:
            if lb["leader"]:
                col = bgr(lb["col"] or T["ink2"])
                a = (int(lb["x"] * self.ss * k), int(lb["y"] * self.ss * k)); b = (int(lb["tx"] * self.ss * k), int((lb["ty"] + 4) * self.ss * k))
                cv2.line(self.img, a, b, col, max(1, int(self.ss * 1.1)), cv2.LINE_AA, SHIFT)
            if lb["dot"]:
                cv2.circle(self.img, (int(lb["x"] * self.ss * k), int(lb["y"] * self.ss * k)), int(3.2 * self.ss * k), bgr(lb["col"] or T["ink2"]), -1, cv2.LINE_AA, SHIFT)
        img = cv2.resize(self.img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(pil, "RGBA")
        for o in self.overlays:
            if o[0] == "rect":
                _, box, col, alpha = o
                d.rectangle(box, fill=tuple(col) + (int(255 * alpha),))
        for lb in self.labels:
            col = tuple(lb["col"] or T["ink"])
            f1 = font(lb["face"], lb["size"]); f2 = font("medium", lb["sub_size"])
            anchor = {"left": "ls", "right": "rs", "center": "ms"}[lb["align"]]

            d.text((lb["tx"], lb["ty"]), lb["text"], font=f1, fill=col, anchor=anchor, stroke_width=3, stroke_fill=tuple(T["halo"]))
            if lb["sub"]:
                for i, line in enumerate(lb["sub"].split("\n")):
                    d.text((lb["tx"], lb["ty"] + 6 + (i + 1) * (lb["sub_size"] + 5)), line, font=f2, fill=tuple(T["ink2"]),
                           anchor=anchor.replace("s", "a") if False else anchor, stroke_width=3, stroke_fill=tuple(T["halo"]))
        for o in self.overlays:
            if o[0] == "text":
                _, xy, text, face, size, col, anchor, halo, spacing = o
                f = font(face, size)
                if spacing:                                          # tracked capitals (the report's kickers)
                    x, y = xy
                    wt = sum(f.getlength(ch) for ch in text) + spacing * max(0, len(text) - 1)
                    x -= {"r": wt, "m": wt / 2}.get(anchor[0], 0)       # set by hand, one letter at a time: the anchor too
                    for ch in text:
                        d.text((x, y), ch, font=f, fill=tuple(col or T["ink"]), anchor="ls")
                        x += f.getlength(ch) + spacing
                else:
                    d.text(xy, text, font=f, fill=tuple(col or T["ink"]), anchor=anchor,
                           **(dict(stroke_width=3, stroke_fill=tuple(T["halo"])) if halo else {}))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
