"""
2D mask -> camera-space 3D geometry ("lifting").

Everything here is in the CAMERA frame (origin at the camera, +z forward, +x
right, +y down - the image convention), in *relative* depth units. The phone
turns this into world space and applies metric scale. See docs/LIFTING.md.
"""
from __future__ import annotations

import base64
import io

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Intrinsics + unprojection
# ---------------------------------------------------------------------------


def intrinsics_from_fov(
    width: int, height: int, fov_deg: float = 60.0
) -> tuple[float, float, float, float]:
    """Pinhole intrinsics from a horizontal field of view, centred principal point."""
    f = 0.5 * width / np.tan(np.radians(fov_deg) / 2.0)
    return f, f, width / 2.0, height / 2.0


def unproject_mask(
    mask: np.ndarray,
    depth: np.ndarray,
    intr: tuple[float, float, float, float],
    stride: int = 4,
) -> np.ndarray:
    """Return an Nx3 array of camera-space points for the masked pixels."""
    fx, fy, cx, cy = intr
    ys, xs = np.nonzero(mask[::stride, ::stride])
    ys = ys * stride
    xs = xs * stride
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float64)
    z = depth[ys, xs].astype(np.float64)
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return np.stack([x, y, z], axis=1)


# ---------------------------------------------------------------------------
# OBB (oriented bounding box) via PCA + robust extents
# ---------------------------------------------------------------------------


def make_obb(
    points: np.ndarray, low: float = 0.02, high: float = 0.98
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (centre, size, quat_xyzw) or None if there are too few points."""
    if len(points) < 4:
        return None
    centre = np.median(points, axis=0)
    p = points - centre
    axes = _principal_axes(p)
    local = p @ axes
    lo = np.quantile(local, low, axis=0)
    hi = np.quantile(local, high, axis=0)
    size = np.maximum(hi - lo, 1e-6)
    centre = centre + axes @ ((lo + hi) / 2.0)
    return centre, size, quat_from_matrix(axes)


def _principal_axes(p: np.ndarray) -> np.ndarray:
    """3x3 matrix whose columns are the principal axes (descending eigenvalue)."""
    cov = np.cov(p.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axes = eigvecs[:, np.argsort(eigvals)[::-1]]
    if np.linalg.det(axes) < 0:  # keep it a right-handed frame
        axes[:, -1] *= -1
    return axes


def quat_from_matrix(m: np.ndarray) -> np.ndarray:
    """Rotation matrix (columns = axes) -> quaternion [x, y, z, w]."""
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / (np.linalg.norm(q) or 1.0)


# ---------------------------------------------------------------------------
# 3D contour (the smooth glowing outline)
# ---------------------------------------------------------------------------


def contour2d(mask: np.ndarray, max_points: int = 48) -> list[list[float]]:
    """Resampled mask boundary in image pixels (closed polyline)."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    boundary = max(contours, key=cv2.contourArea).reshape(-1, 2)
    return _resample(boundary, max_points).tolist()


def contour_from_mask(
    mask: np.ndarray,
    depth: np.ndarray,
    intr: tuple[float, float, float, float],
    max_points: int = 48,
) -> tuple[list[list[float]], list[list[float]]]:
    """
    Return (contour2d, contourCam):
      contour2d  - Nx2 boundary points in image pixels (closed polyline)
      contourCam - Nx3 same points unprojected to camera space
    """
    pts = contour2d(mask, max_points)
    if not pts:
        return [], []

    fx, fy, cx, cy = intr
    h, w = mask.shape
    # Fall back to the object's median depth where the boundary depth is bad.
    med = float(np.median(depth[mask])) if mask.any() else 0.0
    contour_cam: list[list[float]] = []
    for u, v in pts:
        ui = int(np.clip(u, 0, w - 1))
        vi = int(np.clip(v, 0, h - 1))
        z = float(depth[vi, ui])
        if not np.isfinite(z) or z <= 0:
            z = med
        contour_cam.append([(u - cx) * z / fx, (v - cy) * z / fy, z])
    return pts, contour_cam


def _resample(contour_xy: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed contour to n evenly-spaced points along its length."""
    if len(contour_xy) < 2:
        return contour_xy.astype(np.float64)
    pts = np.vstack([contour_xy, contour_xy[0]]).astype(np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = seg.sum()
    if total <= 0:
        return pts[:n]
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    out = []
    for s in np.linspace(0.0, total, n, endpoint=False):
        i = min(max(np.searchsorted(cum, s, side="right") - 1, 0), len(seg) - 1)
        t = (s - cum[i]) / seg[i] if seg[i] > 0 else 0.0
        out.append(pts[i] + t * (pts[i + 1] - pts[i]))
    return np.array(out)


# ---------------------------------------------------------------------------
# Mask encoding for the wire (base64 PNG)
# ---------------------------------------------------------------------------


def encode_mask_png(mask: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray((mask.astype(np.uint8) * 255)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
