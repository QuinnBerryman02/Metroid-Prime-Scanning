"""Truncated signed distance field (TSDF) fusion and marching-cubes meshing.

Replaces the running-mean voxel *points* as the surface representation:

* Each keyframe adds, for every grid node in view, a signed distance
  ``sdf = z_depth - z_node`` (positive in front of the surface, negative
  behind it), clamped to ``[-1, 1] * trunc`` and averaged with a per-node
  weight. The surface is the zero crossing, so it stays sharp where the
  mean smeared it.
* Colour is accumulated per node with the same weights.
* The mesh is the zero level set, extracted with marching cubes.

Units are metres, in the same world frame as the rest of the service.
"""

from __future__ import annotations

import numpy as np

try:  # scikit-image is only needed for meshing, not for integration
    from skimage.measure import marching_cubes
except ImportError:  # pragma: no cover - reported at extract time
    marching_cubes = None


class TSDFVolume:
    """Dense TSDF over a fixed axis-aligned box (world metres)."""

    def __init__(
        self,
        box_min: tuple[float, float, float],
        box_max: tuple[float, float, float],
        voxel: float = 0.04,
        trunc: float = 0.12,
        max_weight: float = 40.0,
    ) -> None:
        self.origin = np.asarray(box_min, dtype=np.float32)
        self.voxel = float(voxel)
        self.trunc = float(trunc)
        self.max_weight = float(max_weight)
        span = np.asarray(box_max, dtype=np.float64) - self.origin
        self.dims = tuple(int(d) for d in np.ceil(span / self.voxel).astype(int) + 1)
        self.tsdf = np.ones(self.dims, dtype=np.float32)  # 1 = free / unobserved
        self.weight = np.zeros(self.dims, dtype=np.float32)
        self.color = np.zeros(self.dims + (3,), dtype=np.float32)
        self._nodes_cache: np.ndarray | None = None
        self.frames = 0

    @property
    def observed_nodes(self) -> int:
        return int(np.count_nonzero(self.weight))

    def _nodes(self) -> np.ndarray:
        """World coordinates of every grid node, (N, 3) float32, cached."""
        if self._nodes_cache is None:
            axes = [
                self.origin[i] + np.arange(self.dims[i], dtype=np.float32) * self.voxel
                for i in range(3)
            ]
            X, Y, Z = np.meshgrid(*axes, indexing="ij")
            self._nodes_cache = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        return self._nodes_cache

    def integrate(
        self,
        depth: np.ndarray,
        rgb: np.ndarray,
        world_from_cam: np.ndarray,
        intr: tuple[float, float, float, float],
        obs_weight: float = 1.0,
    ) -> int:
        """Fold one keyframe into the volume. Returns nodes updated.

        ``depth`` is (H, W) metres, ``rgb`` is (H, W, 3) uint8 at the same size.
        """
        if depth.shape[:2] != rgb.shape[:2]:
            raise ValueError("depth and rgb must share a resolution")
        h, w = depth.shape[:2]
        fx, fy, cx, cy = intr
        rot = world_from_cam[:3, :3].astype(np.float32)
        trans = world_from_cam[:3, 3].astype(np.float32)

        # world -> camera for every node: c = R^T (p - t)  (row-vector form)
        pts = self._nodes()
        cam = (pts - trans) @ rot
        cz = cam[:, 2]
        idx = np.nonzero(cz > 0.05)[0]
        if idx.size == 0:
            return 0
        cz = cz[idx]
        u = np.round(fx * cam[idx, 0] / cz + cx).astype(np.int64)
        v = np.round(fy * cam[idx, 1] / cz + cy).astype(np.int64)
        inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        idx, cz, u, v = idx[inb], cz[inb], u[inb], v[inb]

        d = depth[v, u]
        sdf = d - cz
        keep = np.isfinite(d) & (d > 0.05) & (sdf > -self.trunc)
        idx, sdf, u, v = idx[keep], sdf[keep], u[keep], v[keep]
        if idx.size == 0:
            return 0

        obs = np.clip(sdf / self.trunc, -1.0, 1.0).astype(np.float32)
        tsdf_f = self.tsdf.reshape(-1)
        w_f = self.weight.reshape(-1)
        col_f = self.color.reshape(-1, 3)

        w_old = w_f[idx]
        w_new = w_old + obs_weight
        tsdf_f[idx] = (tsdf_f[idx] * w_old + obs * obs_weight) / w_new
        col_f[idx] = (col_f[idx] * w_old[:, None] + rgb[v, u].astype(np.float32) * obs_weight) / w_new[:, None]
        w_f[idx] = np.minimum(w_new, self.max_weight)
        self.frames += 1
        return int(idx.size)

    def extract(self, min_weight: float = 1.0):
        """Mesh the zero level set.

        Returns ``(vertices float32 Nx3 world, faces uint32 Mx3,
        colors uint8 Nx3)``. Empty arrays if there is no surface.
        """
        if marching_cubes is None:
            raise RuntimeError("scikit-image is required for TSDF meshing")
        empty = (
            np.zeros((0, 3), np.float32),
            np.zeros((0, 3), np.uint32),
            np.zeros((0, 3), np.uint8),
        )
        # Unobserved nodes count as free space, so the surface only closes
        # where the depth maps actually observed a boundary.
        vol = np.where(self.weight >= min_weight, self.tsdf, 1.0).astype(np.float32)
        if not (vol < 0).any():
            return empty
        verts, faces, _normals, _ = marching_cubes(
            vol, level=0.0, allow_degenerate=False
        )
        world = self.origin + verts.astype(np.float32) * self.voxel
        ii = np.clip(np.round(verts).astype(np.int64), 0, np.array(self.dims) - 1)
        cols = self.color[ii[:, 0], ii[:, 1], ii[:, 2]]
        return (
            world.astype(np.float32),
            faces.astype(np.uint32),
            np.clip(cols, 0, 255).astype(np.uint8),
        )
