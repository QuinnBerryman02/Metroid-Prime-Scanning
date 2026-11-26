"""
Dense fusion (MS3.1 + MS3.2).

  * `fit_metric_scale` - recover the metric scale of Depth Anything V2 from
    ARCore's metric planes (docs/RECONSTRUCTION.md section 6).
  * `VoxelMap` - the running-mean voxel accumulator (section 7).

Background and the papers behind these: docs/FUSION.md.

Everything here is in metres once the scale is known. The pose arrives as a
row-major 4x4 `worldFromCamera` in the **OpenCV camera convention**
(+x right, +y down, +z forward) - which is the convention `app/lifting.py`
unprojects in, so the two agree. The phone builds that matrix explicitly; see
the note in docs/LIFTING.md about WebXR/camera-access not exposing intrinsics.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass

import numpy as np

# The offset `b` is only identifiable if the visible plane samples span some
# real depth. Below this, a least-squares line through a nearly-flat cloud of
# samples produces a junk pair (tiny `a`, huge `b`) that is *technically* a
# positive scale - so we assume b=0 instead. The scale `a` itself is fine with
# a single depth level (a fronto-parallel wall pins Z/rel directly).
AFFINE_MIN_SPREAD = 1.0

# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------


def pose_to_matrix(pose: list[float]) -> np.ndarray:
    """Row-major 4x4 -> (4,4) float array."""
    return np.asarray(pose, dtype=np.float64).reshape(4, 4)


def transform_points(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to an Nx3 array of points."""
    return pts @ m[:3, :3].T + m[:3, 3]


# ---------------------------------------------------------------------------
# MS3.1 - metric scale from ARCore planes
# ---------------------------------------------------------------------------


@dataclass
class ScaleFit:
    """Z_metres ~= a * rel_depth + b."""

    a: float
    b: float
    samples: int
    rms: float
    method: str  # "affine" | "scale" | "cached"

    def depth(self, rel: np.ndarray) -> np.ndarray:
        return self.a * rel + self.b


def _plane_samples(polygon: np.ndarray, n: int = 24) -> np.ndarray:
    """Points on a planar polygon: vertices, edge midpoints, and an interior fan."""
    centroid = polygon.mean(axis=0, keepdims=True)
    parts = [polygon]
    for i in range(len(polygon)):
        a = polygon[i]
        b = polygon[(i + 1) % len(polygon)]
        parts.append(((a + b) / 2.0)[None, :])
    rng = np.random.default_rng(0)
    tri = rng.integers(0, len(polygon), n)
    w = rng.random((n, 1))
    parts.append(centroid + w * (polygon[tri] - centroid) * 0.75)
    return np.vstack(parts)


def _scale_only(d: np.ndarray, z: np.ndarray) -> ScaleFit | None:
    """Assume the model has no offset: Z = a*rel."""
    a = float(np.median(z / d))
    if not np.isfinite(a) or a <= 0:
        return None
    return ScaleFit(a, 0.0, int(d.size), float(np.std(z - a * d)), "scale")


def fit_metric_scale(
    planes: list[dict],
    world_from_cam: np.ndarray,
    intr: tuple[float, float, float, float],
    rel_depth: np.ndarray,
    diag: dict | None = None,
) -> ScaleFit | None:
    """Fit the relative depth to ARCore's metric planes.

    For a sample point on a plane we know its true metric depth `Z` (it is the
    camera-space z of the point) and the model's `rel_depth` at the pixel it
    projects to. Least-squares `Z = a*rel + b`, with outlier rejection.

    Returns None when there is not enough (or not sane) evidence. `diag`, if
    given, is filled with plane/sample/spread numbers so the caller can explain
    *why* it failed - that string ends up on the viewer's screen.
    """
    if diag is not None:
        diag.clear()
        diag.update(planes=len(planes), samples=0, spread=0.0)
    if not planes:
        return None

    h, w = rel_depth.shape
    fx, fy, cx, cy = intr
    cam_from_world = np.linalg.inv(world_from_cam)

    ds: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    for plane in planes:
        poly = np.asarray(plane.get("polygon") or [], dtype=np.float64)
        if poly.ndim != 2 or len(poly) < 3:
            continue
        cam = transform_points(cam_from_world, _plane_samples(poly))
        z = cam[:, 2]
        ok = z > 0.15  # in front of the camera
        if not ok.any():
            continue
        cam, z = cam[ok], z[ok]
        u = fx * cam[:, 0] / z + cx
        v = fy * cam[:, 1] / z + cy
        inb = (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        if not inb.any():
            continue
        ui = u[inb].astype(np.int32)
        vi = v[inb].astype(np.int32)
        d = rel_depth[vi, ui].astype(np.float64)
        keep = np.isfinite(d) & (d > 1e-6)
        if keep.any():
            ds.append(d[keep])
            zs.append(z[inb][keep])

    if not ds:
        return None
    d = np.concatenate(ds)
    z = np.concatenate(zs)
    spread = float(np.ptp(z))
    if diag is not None:
        diag.update(samples=int(d.size), spread=spread)
    if d.size < 6:
        return None
    if float(np.ptp(d)) < 1e-3:
        return None  # rel_depth is flat: nothing to fit against
    if spread < AFFINE_MIN_SPREAD:
        # b is unidentifiable from this view; assume no offset.
        return _scale_only(d, z)

    a, b = (float(x) for x in np.polyfit(d, z, 1))
    for _ in range(3):  # robust refit: drop > 3*MAD residuals
        resid = z - (a * d + b)
        mad = float(np.median(np.abs(resid - np.median(resid)))) + 1e-6
        keep = np.abs(resid) < 3.0 * mad
        if keep.sum() < 6:
            break
        a, b = (float(x) for x in np.polyfit(d[keep], z[keep], 1))
    if not np.isfinite(a) or a <= 0:
        return None
    if abs(b) > 0.6 * float(np.median(z)):
        # The offset swamps the scale term: a degenerate fit despite the spread
        # test. Refuse it and fall back to b=0.
        return _scale_only(d, z)
    resid = z - (a * d + b)
    return ScaleFit(a, b, int(d.size), float(np.sqrt(np.mean(resid**2))), "affine")


class ScaleTracker:
    """Session-level metric scale, kept stable across frames.

    Per-frame fits alone are too jumpy: one degenerate fit (all the visible
    planes at the same depth) gets used for that frame, and every surface fused
    with it lands somewhere different from the frames around it - which shows up
    as points smeared outside the room and a ballooning voxel count. So we keep
    a short history of accepted fits and fuse with the running **median**.
    """

    def __init__(self, history: int = 20) -> None:
        self.history = history
        self.a: list[float] = []
        self.b: list[float] = []

    def update(self, fit: ScaleFit | None) -> ScaleFit | None:
        """Fold in a fresh fit (if any) and return the scale to fuse with."""
        if fit is not None:
            self.a.append(fit.a)
            self.b.append(fit.b)
            del self.a[: -self.history]
            del self.b[: -self.history]
        if not self.a:
            return None
        a = float(np.median(self.a))
        b = float(np.median(self.b))
        if fit is not None:
            return ScaleFit(a, b, fit.samples, fit.rms, fit.method)
        return ScaleFit(a, b, 0, 0.0, "tracked")


# ---------------------------------------------------------------------------
# MS3.2 - voxel accumulation
# ---------------------------------------------------------------------------


def voxel_keys(points: np.ndarray, voxel: float) -> np.ndarray:
    """Quantise Nx3 world points to a single int64 key per voxel."""
    q = np.floor(points / voxel).astype(np.int64) + (1 << 20)
    return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]


class VoxelMap:
    """Running-mean voxel accumulator (docs/FUSION.md section 2).

    Not a TSDF: no field, no truncation, no zero crossing - it is a decimated
    averaged point cloud. Its resolution is coupled to its accuracy, which is
    why TSDF is the eventual target; see docs/FUSION.md section 3.
    """

    def __init__(self, voxel: float = 0.05, capacity: int = 3_000_000) -> None:
        self.voxel = voxel
        self.capacity = capacity
        self.keys = np.empty(0, dtype=np.int64)
        self.spos = np.empty((0, 3), dtype=np.float64)
        self.scol = np.empty((0, 3), dtype=np.float64)
        self.wsum = np.empty(0, dtype=np.float64)

    @property
    def size(self) -> int:
        return int(self.keys.size)

    def integrate(self, points: np.ndarray, colors: np.ndarray) -> int:
        """Fold one keyframe's world points in. Returns voxels touched."""
        if points.shape[0] == 0:
            return 0
        k = voxel_keys(points, self.voxel)
        uniq, inv = np.unique(k, return_inverse=True)
        inv = inv.reshape(-1)
        n = int(uniq.size)

        cnt = np.bincount(inv, minlength=n).astype(np.float64)
        sp = np.empty((n, 3), dtype=np.float64)
        sc = np.empty((n, 3), dtype=np.float64)
        for j in range(3):
            sp[:, j] = np.bincount(inv, weights=points[:, j], minlength=n)
            sc[:, j] = np.bincount(inv, weights=colors[:, j], minlength=n)

        gk = self.keys
        if gk.size == 0:
            self.keys, self.spos, self.scol, self.wsum = uniq, sp, sc, cnt
            return n

        pos = np.clip(np.searchsorted(gk, uniq), 0, gk.size)
        exists = np.zeros(n, dtype=bool)
        ok = pos < gk.size
        exists[ok] = gk[pos[ok]] == uniq[ok]

        idx = pos[exists]
        self.spos[idx] += sp[exists]
        self.scol[idx] += sc[exists]
        self.wsum[idx] += cnt[exists]

        if not exists.all():
            newk = uniq[~exists]
            ins = np.searchsorted(gk, newk)
            self.keys = np.insert(gk, ins, newk)
            self.spos = np.insert(self.spos, ins, sp[~exists], axis=0)
            self.scol = np.insert(self.scol, ins, sc[~exists], axis=0)
            self.wsum = np.insert(self.wsum, ins, cnt[~exists])

        if self.keys.size > self.capacity:
            keep = np.argsort(-self.wsum)[: self.capacity]
            keep.sort()
            self.keys, self.spos = self.keys[keep], self.spos[keep]
            self.scol, self.wsum = self.scol[keep], self.wsum[keep]
        return n

    def snapshot(
        self, voxel: float = 0.0, max_points: int = 30_000
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """(positions float32 Nx3, colors uint8 Nx3, voxel used) for the viewer.

        Re-voxelising at a coarser size is the cheap way to keep the wire
        payload bounded - it is a decimation, not a separate map.
        """
        if self.keys.size == 0:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), self.voxel
        mu = self.spos / self.wsum[:, None]
        col = self.scol / self.wsum[:, None]  # 0..255

        used = voxel
        if voxel > self.voxel:
            k = voxel_keys(mu, voxel)
            uniq, inv = np.unique(k, return_inverse=True)
            inv = inv.reshape(-1)
            n = int(uniq.size)
            om = np.empty((n, 3), dtype=np.float64)
            oc = np.empty((n, 3), dtype=np.float64)
            for j in range(3):
                om[:, j] = np.bincount(inv, weights=mu[:, j], minlength=n) / np.bincount(
                    inv, minlength=n
                )
                oc[:, j] = np.bincount(inv, weights=col[:, j], minlength=n) / np.bincount(
                    inv, minlength=n
                )
            mu, col = om, oc
        else:
            used = self.voxel

        if mu.shape[0] > max_points:
            idx = np.linspace(0, mu.shape[0] - 1, max_points).astype(np.int64)
            mu, col = mu[idx], col[idx]
        return (
            mu.astype(np.float32),
            np.clip(col, 0, 255).astype(np.uint8),
            float(used),
        )


def unproject_depth(
    rel_depth: np.ndarray,
    world_from_cam: np.ndarray,
    intr: tuple[float, float, float, float],
    scale: ScaleFit,
    stride: int = 2,
    min_depth: float = 0.2,
    max_depth: float = 12.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unproject a relative depth map to world points.

    Returns (world_points Nx3, ys N, xs N) so the caller can sample colours.
    """
    h, w = rel_depth.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    d = rel_depth[ys, xs].astype(np.float64)
    z = scale.a * d + scale.b
    ok = np.isfinite(z) & (z > min_depth) & (z < max_depth)
    if not ok.any():
        return np.empty((0, 3)), np.empty(0, np.int64), np.empty(0, np.int64)

    fx, fy, cx, cy = intr
    zz = z[ok]
    cam = np.stack(
        [(xs[ok] - cx) / fx * zz, (ys[ok] - cy) / fy * zz, zz], axis=1
    )
    return transform_points(world_from_cam, cam), ys[ok].astype(np.int64), xs[
        ok
    ].astype(np.int64)


def encode_points(positions: np.ndarray, colors: np.ndarray) -> tuple[str, str]:
    """Base64 the point cloud for the wire (little-endian float32 / uint8)."""
    return (
        base64.b64encode(np.ascontiguousarray(positions, dtype="<f4").tobytes()).decode(),
        base64.b64encode(np.ascontiguousarray(colors, dtype=np.uint8).tobytes()).decode(),
    )
