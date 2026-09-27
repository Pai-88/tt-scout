"""The whole camera from the table: focal length, where the phone stood and which way it pointed.
for measuring ball speed in 3D (flight.py) instead of from the table's plane only.

The four table corners give a homography from the table plane to the picture. With square pixels and the principal point in the
middle of the picture, a plane homography H = s K [r1 r2 t] fixes the one unknown left in K, the focal length: the first two columns
of K^-1 H must be orthogonal and of equal length (Zhang's constraints for one plane). The pose [R | t] follows. World frame = the
table's: X along its length (0 at the "near" end line), Y across it (0 on the camera's side), Z up from the playing surface.

    cam = Camera.from_table(table, width, height)
    cam.project(P)          # world points (N x 3) -> pixels (N x 2)
    cam.ray(uv)             # pixels -> camera centre and unit directions in the world
    cam.on_plane_z(uv, z)   # pixels -> the world points at height z seen there
"""
import numpy as np
from .config import TABLE_LENGTH as L_T, TABLE_WIDTH as W_T


class Camera:
    def __init__(self, K, R, t):
        self.K, self.R, self.t = K, R, t
        self.C = -R.T @ t                                               # the camera centre in the world
        self.f = float(K[0, 0])

    @classmethod
    def from_table(cls, table, width, height, f=None):
        """Calibrate from the table's homography. f given: use it (pixels) instead of solving for it."""
        Hi = table.H_inv / table.H_inv[2, 2]                             # table plane (X, Y, 1) -> picture (u, v, 1)
        cx, cy = width / 2.0, height / 2.0
        h1, h2, h3 = Hi[:, 0], Hi[:, 1], Hi[:, 2]
        if f is None:
            a1, b1, c1 = h1[0] - cx * h1[2], h1[1] - cy * h1[2], h1[2]
            a2, b2, c2 = h2[0] - cx * h2[2], h2[1] - cy * h2[2], h2[2]
            ests = []
            if abs(c1 * c2) > 1e-12:
                f2 = -(a1 * a2 + b1 * b2) / (c1 * c2)                    # orthogonal columns
                if f2 > 0:
                    ests.append(("orthogonal", float(np.sqrt(f2))))
            if abs(c2 * c2 - c1 * c1) > 1e-12:
                f2 = (a1 * a1 + b1 * b1 - a2 * a2 - b2 * b2) / (c2 * c2 - c1 * c1)   # equal length
                if f2 > 0:
                    ests.append(("equal length", float(np.sqrt(f2))))
            plausible = [v for _, v in ests if 0.4 * width < v < 3.0 * width]
            f = float(np.median(plausible)) if plausible else 0.8 * width
            cls.last_estimates = ests
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        Ki = np.linalg.inv(K)
        m1, m2, m3 = Ki @ h1, Ki @ h2, Ki @ h3
        lam = 2.0 / (np.linalg.norm(m1) + np.linalg.norm(m2))
        r1, r2, t = lam * m1, lam * m2, lam * m3
        if t[2] < 0:                                                    # the table must be in front of the camera
            r1, r2, t = -r1, -r2, -t
        r3 = np.cross(r1, r2)
        U, _, Vt = np.linalg.svd(np.stack([r1, r2, r3], axis=1))        # nearest true rotation
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = U @ np.diag([1, 1, -1]) @ Vt
        cam = cls(K, R, t)
        if cam.C[2] < 0:                                                # Z must point up: the phone is above the table top
            R = R @ np.diag([1, 1, -1]); cam = cls(K, R, t)
        return cam

    @classmethod
    def from_table_pnp(cls, table, width, height, f_range=(500, 2600), n=211):
        """Calibrate by pose solving: for each focal length in f_range, the pose that puts the four table corners back where they are
        seen (OpenCV's planar solver, IPPE); keep the focal length whose corners come back closest. More robust than the closed form
        when the table is seen nearly side-on. cls.last_scan = [(f, rms px)]."""
        import cv2
        obj = np.array([[0, 0, 0], [0, W_T, 0], [L_T, W_T, 0], [L_T, 0, 0]], np.float64)
        img = np.asarray(table.corners, np.float64)
        best, scan = None, []
        for f in np.geomspace(f_range[0], f_range[1], n):
            K = np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
            if not ok:
                continue
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
            e = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
            scan.append((float(f), e))
            if best is None or e < best[0]:
                best = (e, K, rvec, tvec)
        cls.last_scan = scan
        e, K, rvec, tvec = best
        R, _ = cv2.Rodrigues(rvec)
        cam = cls(K, R, tvec.ravel())
        if cam.C[2] < 0:
            R = R @ np.diag([1, 1, -1]); cam = cls(K, R, tvec.ravel())
        cam.corner_rms = e
        return cam

    def project(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        Xc = P @ self.R.T + self.t
        return np.stack([self.K[0, 0] * Xc[:, 0] / Xc[:, 2] + self.K[0, 2], self.K[1, 1] * Xc[:, 1] / Xc[:, 2] + self.K[1, 2]], axis=1)

    def ray(self, uv):
        uv = np.atleast_2d(np.asarray(uv, float))
        d_cam = np.stack([(uv[:, 0] - self.K[0, 2]) / self.K[0, 0], (uv[:, 1] - self.K[1, 2]) / self.K[1, 1], np.ones(len(uv))], axis=1)
        d = d_cam @ self.R                                              # camera -> world directions (R^T applied to rows)
        return self.C, d / np.linalg.norm(d, axis=1, keepdims=True)

    def on_plane_z(self, uv, z):
        C, d = self.ray(uv)
        s = (z - C[2]) / d[:, 2]
        return C + d * s[:, None]

    def summary(self, width):
        hfov = 2 * np.degrees(np.arctan(width / 2 / self.f))
        return dict(f_px=round(self.f, 1), hfov_deg=round(float(hfov), 1), camera_m=[round(float(v), 2) for v in self.C])
