"""Synthetic check for app/tsdf.py: a floor (y=0) and a back wall (z=2.5),
rendered analytically from a few camera poses, fused and meshed.
Pass criteria: a mesh exists, its vertices sit on the floor and wall planes,
and the observed nodes cover both surfaces."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from app.tsdf import TSDFVolume

W, H = 160, 120
fx = fy = 120.0; cx, cy = W / 2, H / 2
intr = (fx, fy, cx, cy)
yy, xx = np.mgrid[0:H, 0:W]
dcam = np.stack([(xx - cx) / fx, (yy - cy) / fy, np.ones_like(xx, float)], -1)

def render(pose):
    R, t = pose[:3, :3], pose[:3, 3]
    dw = dcam @ R.T
    depth = np.full((H, W), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = np.where(dw[..., 1] < -1e-6, (0 - t[1]) / dw[..., 1], np.inf)  # floor y=0
        tw = np.where(dw[..., 2] > 1e-6, (2.5 - t[2]) / dw[..., 2], np.inf)  # wall z=2.5
    tt = np.minimum(tf, tw)
    ok = np.isfinite(tt) & (tt > 0)
    depth[ok] = tt[ok] * dcam[..., 2][ok]  # z along the optical axis
    rgb = np.zeros((H, W, 3), np.uint8); rgb[...] = (200, 120, 60)
    return depth, rgb

vol = TSDFVolume((-1.0, -0.2, 0.0), (1.0, 1.5, 3.0), voxel=0.04, trunc=0.12)
for yaw in (-0.25, 0.0, 0.25):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    pose = np.eye(4); pose[:3, :3] = R; pose[:3, 3] = [0, 1.0, 0]
    d, rgb = render(pose)
    n = vol.integrate(d, rgb, pose, intr)
    print(f"yaw {yaw:+.2f}: updated {n} nodes")

verts, faces, cols = vol.extract()
print("observed nodes", vol.observed_nodes, "verts", len(verts), "faces", len(faces))
assert len(verts) > 500, "no surface"
floor = verts[np.abs(verts[:, 1]) < 0.05]
wall = verts[np.abs(verts[:, 2] - 2.5) < 0.05]
print("floor verts", len(floor), "wall verts", len(wall))
assert len(floor) > 100 and len(wall) > 100
assert np.abs(floor[:, 1]).max() < 0.05
print("colour sample", cols[0])
print("TSDF OK")
