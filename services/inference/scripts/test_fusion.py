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
    ScaleFit,
    ScaleTracker,
    VoxelMap,
    encode_points,
    fit_metric_scale,
    unproject_depth,
)

H, W = 480, 640
F = 0.5 * W / np.tan(np.radians(60) / 2.0)
INTR = (F, F, W / 2.0, H / 2.0)
I4 = np.eye(4)
# Synthetic affine-invariant disparity (what Depth Anything gives us):
#   disp = (1/Z) / A_TRUE + Q_TRUE   =>   1/Z = a*disp + b  with a=1/A... see below
DISP_K = 2.5   # disp = DISP_K / Z + Q_TRUE
Q_TRUE = 0.3   # unknown per-image shift the fit must absorb
A_TRUE = 1.0 / DISP_K  # expected a in 1/Z = a*disp + b
B_TRUE = -Q_TRUE / DISP_K


def quad_at(z, u0, u1, v0, v1):
    fx, fy, cx, cy = INTR

    def p(u, v):
        return [(u - cx) / fx * z, (v - cy) / fy * z, z]

    return [p(u0, v0), p(u1, v0), p(u1, v1), p(u0, v1)]


# True metric depth: right half near (1.5 m), left half far (4 m).
Z_true = np.full((H, W), 1.5)
Z_true[:, : W // 2] = 4.0
rel = DISP_K / Z_true + Q_TRUE

planes = [
    {"polygon": quad_at(1.5, 0.55 * W, 0.95 * W, 0.15 * H, 0.85 * H)},
    {"polygon": quad_at(4.0, 0.05 * W, 0.45 * W, 0.15 * H, 0.85 * H)},
]

fit = fit_metric_scale(planes, I4, INTR, rel)
print(f"scale fit: {fit}")
assert fit is not None, "fit returned None"
assert abs(fit.a - A_TRUE) / A_TRUE < 0.05, f"a={fit.a} true={A_TRUE}"
assert abs(fit.b - B_TRUE) < 0.05, f"b={fit.b} true={B_TRUE}"  # shift absorbed
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
# One fronto-parallel plane at a single depth: b is unidentifiable, but the
# scale `a` is still pinned (Z/rel). Real depth maps always vary a little, so
# give the fixture a small gradient rather than a perfectly constant map.
rel1 = DISP_K / (2.0 + 0.02 * np.tile(np.linspace(-1, 1, W), (H, 1)))  # no shift: scale-only is exact
single = fit_metric_scale(
    [{"polygon": quad_at(2.0, 0.3 * W, 0.7 * W, 0.2 * H, 0.8 * H)}],
    I4,
    INTR,
    rel1,
)
print(f"single-plane fit: {single}")
assert single is not None and single.method == "scale" and abs(single.b) < 1e-9

# --- two planes close in depth -> b unidentifiable, so b must stay 0 --------
Zc = np.full((H, W), 1.5)
Zc[:, : W // 2] = 1.8
close = fit_metric_scale(
    [
        {"polygon": quad_at(1.5, 0.55 * W, 0.95 * W, 0.15 * H, 0.85 * H)},
        {"polygon": quad_at(1.8, 0.05 * W, 0.45 * W, 0.15 * H, 0.85 * H)},
    ],
    I4,
    INTR,
    DISP_K / Zc,  # no shift: scale-only is exact
)
print(f"close-depth fit: {close}")
assert close is not None, "should still yield a scale"
assert close.method == "scale" and abs(close.b) < 1e-9, close
assert abs(close.a - A_TRUE) < 1e-6

# --- diagnostics reach the caller (this string lands on the viewer) ---------
diag = {}
full = fit_metric_scale(planes, I4, INTR, rel, diag)
print(f"diag: {diag}")
assert diag["planes"] == 2 and diag["samples"] > 0 and diag["spread"] > 1.0
assert full is not None and full.method == "affine"

# --- no planes -> None (caller keeps the last tracked scale) ----------------
print("no-plane fit:", fit_metric_scale([], I4, INTR, rel))
assert fit_metric_scale([], I4, INTR, rel) is None

# --- ScaleTracker: median across frames, survives frames with no fit --------
tr = ScaleTracker()
tr.update(ScaleFit(0.40, 0.02, 64, 0.01, "affine"))
tr.update(ScaleFit(0.44, 0.00, 64, 0.01, "affine"))
s = tr.update(None)  # a frame with no usable planes
print(f"tracked scale: {s}")
assert s is not None and s.method == "tracked"
assert abs(s.a - 0.42) < 1e-9 and abs(s.b - 0.01) < 1e-9, s
# an empty tracker returns None rather than inventing a scale
assert ScaleTracker().update(None) is None

print("\nALL OK")
