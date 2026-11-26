"""Benchmark the voxel-map fusion loop (docs/RECONSTRUCTION.md section 7).

    python scripts/bench_voxel.py [frames] [points_per_frame]

Simulates a camera looking at the six faces of a room plus a sloped attic
ceiling, with per-frame depth noise, and times VoxelMap.integrate().
"""
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

VS = 0.02  # 2 cm voxels
ROOM = np.array([5.0, 2.5, 4.0])  # x, y, z extent (m)
NOISE = 0.03  # 3 cm per-frame depth noise


def keys_from_points(points: np.ndarray, vs: float) -> np.ndarray:
    """Quantise world points to a single int64 voxel key."""
    q = np.floor(points / vs).astype(np.int64) + (1 << 20)
    return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]


class VoxelMap:
    """Voxel accumulator: running sum of position, colour and weight per voxel."""

    def __init__(self, vs: float = VS, capacity: int = 1_000_000) -> None:
        self.vs = vs
        self.capacity = capacity
        self.keys = np.empty(0, dtype=np.int64)
        self.spos = np.empty((0, 3), dtype=np.float32)
        self.scol = np.empty((0, 3), dtype=np.float32)
        self.wsum = np.empty(0, dtype=np.float32)

    def integrate(self, points: np.ndarray, colors: np.ndarray) -> int:
        vs = self.vs
        k = keys_from_points(points, vs)
        uniq, inv = np.unique(k, return_inverse=True)
        inv = inv.reshape(-1)
        n = len(uniq)

        cnt = np.bincount(inv, minlength=n).astype(np.float32)
        sp = np.empty((n, 3), dtype=np.float32)
        sc = np.empty((n, 3), dtype=np.float32)
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


SURFACES = ["floor", "roof", "wall-x0", "wall-x1", "wall-z0", "wall-z1"]


def sample_room(rng: np.random.Generator, n: int) -> np.ndarray:
    """Points on the room's surfaces (a sloped attic ceiling on one axis)."""
    x, y, z = ROOM
    which = rng.integers(0, len(SURFACES), n)
    p = np.empty((n, 3), dtype=np.float64)
    for i, name in enumerate(SURFACES):
        m = which == i
        c = m.sum()
        if not c:
            continue
        u, v = rng.random(c), rng.random(c)
        if name == "floor":
            p[m] = np.stack([u * x, np.zeros(c), v * z], axis=1)
        elif name == "roof":
            # A real diagonal attic roof: slopes down along +x (NOT axis-aligned).
            p[m] = np.stack([u * x, y - 1.5 * u, v * z], axis=1)
        elif name == "wall-x0":
            p[m] = np.stack([np.zeros(c), v * y, u * z], axis=1)
        elif name == "wall-x1":
            p[m] = np.stack([np.full(c, x), v * y, u * z], axis=1)
        elif name == "wall-z0":
            p[m] = np.stack([u * x, v * y, np.zeros(c)], axis=1)
        else:
            p[m] = np.stack([u * x, v * y, np.full(c, z)], axis=1)
    return p


def main() -> None:
    frames = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    per_frame = int(sys.argv[2]) if len(sys.argv) > 2 else 150_000
    vs = float(sys.argv[3]) if len(sys.argv) > 3 else VS
    rng = np.random.default_rng(7)
    vmap = VoxelMap(vs=vs, capacity=8_000_000)

    # Static surfaces + per-frame independent noise (the fusion should average it).
    truth = sample_room(rng, 400_000)

    # Baseline: how noisy is one frame vs the fused mean?
    times: list[float] = []
    norms: list[int] = []
    for _ in range(frames):
        sel = rng.choice(truth.shape[0], size=per_frame, replace=False)
        pts = truth[sel] + rng.normal(0.0, NOISE, (per_frame, 3))
        cols = rng.random((per_frame, 3))
        t0 = time.perf_counter()
        vmap.integrate(pts, cols)
        times.append((time.perf_counter() - t0) * 1000)
        norms.append(pts.shape[0])

    ms = np.array(times)
    print(f"frames            {frames}")
    print(f"points / frame    {per_frame:,}")
    print(f"voxel size        {vs * 100:.1f} cm")
    print(f"integrate  mean   {ms.mean():6.1f} ms   (min {ms.min():.1f}  max {ms.max():.1f})")
    print(f"per second @3 Hz  {ms.mean() * 3:6.1f} ms/s  ({ms.mean() * 3 / 10:.1f}% of one core)")
    print(f"voxels            {vmap.keys.size:,}")
    print(f"mean weight       {vmap.wsum.mean():5.2f}  (observations per voxel = averaging factor)")
    print(f"voxel arrays      {(vmap.spos.nbytes + vmap.scol.nbytes + vmap.wsum.nbytes + vmap.keys.nbytes) / 1e6:.1f} MB")

    # Depth-error check: residual of the fused means vs the true surfaces.
    mu = vmap.spos / vmap.wsum[:, None]
    sample = rng.choice(mu.shape[0], size=min(20_000, mu.shape[0]), replace=False)
    tree = cKDTree(truth)
    dist, _ = tree.query(mu[sample])
    print(f"median |fused-truth|  {np.median(dist) * 100:5.2f} cm  (single-frame noise is {NOISE * 100:.0f} cm)")


if __name__ == "__main__":
    main()
