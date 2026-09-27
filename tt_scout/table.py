"""Table geometry: 4 clicked corners -> homography -> metres on the table surface.

The homography is only valid for points ON the table plane. That is exactly the bounce instant, which is
why bounces (and only bounces) get mapped to table coordinates.
"""
import json, pathlib
import numpy as np, cv2
from .config import TABLE_LENGTH as L, TABLE_WIDTH as W, BALL_RADIUS

# click order: corner 1 and 2 are the two corners of one END line (that end = "near", x = 0);
# corner 1 is the one you continue around the table from: 1 -> 2 -> 3 -> 4.
TABLE_PTS = np.float32([[0, 0], [0, W], [L, W], [L, 0]])


class Table:
    def __init__(self, corners_px, view="side", meta=None):
        self.corners = np.float32(corners_px).reshape(4, 2)
        self.view = view
        self.meta = meta or {}
        self.H, _ = cv2.findHomography(self.corners, TABLE_PTS)
        self.H_inv = np.linalg.inv(self.H)

    @classmethod
    def load(cls, path):
        d = json.loads(pathlib.Path(path).read_text())
        return cls(d["corners_px"], d.get("view", "side"), d)

    @staticmethod
    def save(path, corners_px, view, **meta):
        d = {"corners_px": [[float(x), float(y)] for x, y in corners_px], "view": view, **meta}
        pathlib.Path(path).write_text(json.dumps(d, indent=2))

    def to_table(self, pts_px):
        p = np.float32(pts_px).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.H).reshape(-1, 2)

    def to_px(self, pts_m):
        p = np.float32(pts_m).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.H_inv).reshape(-1, 2)

    def ppm_at(self, xy_m, eps=0.01):
        """Pixels per metre at a table point, along the least-compressed direction."""
        x, y = xy_m
        p0, px, py = self.to_px([[x, y], [x + eps, y], [x, y + eps]])
        J = np.stack([(px - p0) / eps, (py - p0) / eps], axis=1)
        return float(np.linalg.svd(J, compute_uv=False)[0])

    def ball_radius_px(self, xy_m):
        return self.ppm_at(xy_m) * BALL_RADIUS

    def inside(self, pt_px, mx=0.0, my=0.0):
        x, y = self.to_table([pt_px])[0]
        return (-mx <= x <= L + mx) and (-my <= y <= W + my)

    def side(self, pt_px):
        return "near" if self.to_table([pt_px])[0][0] < L / 2 else "far"

    def quad_int(self):
        return self.corners.astype(np.int32).reshape(-1, 1, 2)
