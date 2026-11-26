"""Synthetic checks for app/fusion.py (MS3.1 scale, MS3.2 voxels).

    python scripts/test_fusion.py

Builds a camera with two fronto-parallel metric planes and a relative depth map
that is exactly proportional to true depth, then checks the fit recovers the
known scale.
"""
import sys

import numpy as np

sys.path.insert(0, ".")

from app.fusion import (  # noqa: E402
    VoxelMap,
    encode_points,
    fit_metric_scale,
    unproject_depth,
)

H, W = 480, 640
F = 0.5 * W / np.tan(np.radians(60) / 2.0)
INTR = (F, F, W / 2.0, H / 2.0)
I4 = np.eye(4)
A_TRUE = 0.4  # rel_depth = Z / A_TRUE


def quad_at(z, u0, u1, v0, v1):
    fx, fy, cx, cy = INTR

    def p(u, v):
        return [(u - cx) / fx * z, (v - cy) / fy * z, z]

    return [p(u0, v0), p(u1, v0), p(u1, v1), p(u0, v1)]


# True metric depth: right half near (1.5 m), left half far (4 m).
Z_true = np.full((H, W), 1.5)
Z_true[:, : W // 2] = 4.0
rel = Z_true / A_TRUE

planes = [
    {"polygon": quad_at(1.5, 0.55 * W, 0.95 * W, 0.15 * H, 0.85 * H)},
    {"polygon": quad_at(4.0, 0.05 * W, 0.45 * W, 0.15 * H, 0.85 * H)},
]

fit = fit_metric_scale(planes, I4, INTR, rel)
print(f"scale fit: {fit}")
assert fit is not None, "fit returned None"
assert abs(fit.a - A_TRUE) / A_TRUE < 0.05, f"a={fit.a} true={A_TRUE}" 
print(f"  recovered a={fit.a:.4f} (true {A_TRUE}) b={fit.b:.4f} "
      f"rms={fit.rms:.4f} n={fit.samples} method={fit.method}")

# --- unproject + fuse -------------------------------------------------------
pts, ys, xs = unproject_depth(rel, I4, INTR, fit, stride=2)
print(f"unprojected {pts.shape[0]} pts, z {pts[:, 2].min():.2f}..{pts[:, 2].max():.2f} m")
assert 1.4 < pts[:, 2].min() and pts[:, 2].max() < 4.1

vm = VoxelMap(voxel=0.05)
colors = np.full((pts.shape[0], 3), 128.0)
for i in range(5):
    vm.integrate(pts, colors)
print(f"voxels {vm.size:,} after 5 frames, mean weight {vm.wsum.mean():.2f}")

pos, col, used = vm.snapshot(0.12, 30000)
p64, c64 = encode_points(pos, col)
print(f"snapshot {pos.shape[0]:,} pts at {used:.2f} m, {col.dtype}, "
      f"{len(p64) + len(c64):,} base64 chars")
assert vm.size > 0 and pos.shape[0] > 0

# Two planes 2.5 m apart must remain two planes after fusion.
zs = pos[:, 2]
near = int((zs < 2.5).sum())
far = int((zs >= 2.5).sum())
print(f"merged snapshot: {near:,} near pts, {far:,} far pts")
assert near > 0 and far > 0

# --- scale fallback: one plane only -> b is unobservable --------------------
single = fit_metric_scale(
    [{"polygon": quad_at(2.0, 0.3 * W, 0.7 * W, 0.2 * H, 0.8 * H)}],
    I4,
    INTR,
    np.full((H, W), 2.0 / A_TRUE),
)
print(f"single-plane fit: {single}")
assert single is not None and single.method == "scale" and abs(single.b) < 1e-9

# --- no planes -> None (caller keeps the cached fit) ------------------------
print("no-plane fit:", fit_metric_scale([], I4, INTR, rel))
assert fit_metric_scale([], I4, INTR, rel) is None

print("\nALL OK")
