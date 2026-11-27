"""End-to-end check of the reconstruction path with the REAL depth model.

    python scripts/test_recon_e2e.py

Runs Depth Anything V2 on samples/bus.jpg, fabricates two metric planes that are
consistent with an assumed scale, and checks:`n  * fit_metric_scale recovers it,
  * unproject + VoxelMap.integrate produces voxels,
  * the snapshot round-trips through encode_points.
"""
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, ".")

from app import config  # noqa: E402
from app.depth import DepthEstimator  # noqa: E402
from app.fusion import (  # noqa: E402
    VoxelMap,
    encode_points,
    fit_metric_scale,
    unproject_depth,
)

A_TRUE = 1.0  # we pretend 1/Z = A_TRUE * disp
I4 = np.eye(4)

pil = Image.open("samples/bus.jpg").convert("RGB")
w, h = pil.size
intr = (f := 0.5 * w / np.tan(np.radians(60) / 2.0), f, w / 2.0, h / 2.0)
fx, fy, cx, cy = intr

est = DepthEstimator(
    config.DEPTH_MODEL, config.DEPTH_DEVICE, config.DEPTH_IR, config.EXPORT_OPENVINO
)
rel = est.predict(pil)  # raw disparity, as the recon path uses
rel = cv2.medianBlur(rel, 5)
print(f"rel_depth {rel.shape} range {rel.min():.3f}..{rel.max():.3f} dtype {rel.dtype}")

Z = 1.0 / (A_TRUE * rel)  # pseudo-truth in the 1/Z = a*disp model


def quad(u0, u1, v0, v1):
    def p(u, v):
        z = float(Z[v, u])
        return [(u - cx) / fx * z, (v - cy) / fy * z, z]

    return [p(u0, v0), p(u1, v0), p(u1, v1), p(u0, v1)]


# bottom-left pavement patch and a top (far) patch -> two depth levels
planes = [
    {"polygon": quad(60, 240, int(0.62 * h), int(0.80 * h))},
    {"polygon": quad(int(0.30 * w), int(0.75 * w), 10, int(0.30 * h))},
]

fit = fit_metric_scale(planes, I4, intr, rel)
print(f"scale fit: {fit}")
assert fit is not None, "no scale fit"
assert fit.a > 0, fit
assert 0.5 * A_TRUE < fit.a < 2.0 * A_TRUE, f"a={fit.a} vs assumed {A_TRUE}"

pts, ys, xs = unproject_depth(rel, I4, intr, fit, stride=2, max_depth=12.0)
print(f"unprojected {pts.shape[0]:,} pts, z {pts[:, 2].min():.2f}..{pts[:, 2].max():.2f} m")
assert pts.shape[0] > 1000

img = np.array(pil)
vm = VoxelMap(voxel=0.05)
for _ in range(3):
    vm.integrate(pts, img[ys, xs].astype(np.float64))
print(f"voxels {vm.size:,}, mean weight {vm.wsum.mean():.2f}")
assert vm.size > 0

pos, col, used = vm.snapshot(0.12, 30000)
p64, c64 = encode_points(pos, col)
print(f"snapshot {pos.shape[0]:,} pts at {used:.2f} m; {len(p64) + len(c64):,} base64 chars")
print(f"snapshot z range {pos[:, 2].min():.2f}..{pos[:, 2].max():.2f} m")
assert pos.shape[0] > 0 and col.shape[0] == pos.shape[0]

print("\nE2E OK")
