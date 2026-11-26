"""Synthetic checks for app/recon_methods.py.

Builds a known scene - a wall at z=2 m plus a scatter of outliers - and checks
that every method reconstructs something with the right bounds. No models, no
network, no camera: this is about the accumulators being correct.

    python scripts/test_recon_methods.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.recon_methods import (  # noqa: E402
    build,
    coerce_params,
    default_params,
    list_methods,
)

OK = "\033[32mPASS\033[0m"
BAD = "\033[31mFAIL\033[0m"
failures = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global failures
    if not cond:
        failures += 1
    print(f"  {OK if cond else BAD}  {name}{('  ' + detail) if detail else ''}")


rng = np.random.default_rng(7)
WALL_Z = 2.0
CAM = np.array([0.0, 0.0, -0.5])


def wall_frame(n: int, jitter: float) -> tuple[np.ndarray, np.ndarray]:
    """Points on a 2x2 m wall at z=WALL_Z, coloured by position."""
    x = rng.uniform(-1, 1, n)
    y = rng.uniform(-1, 1, n)
    z = np.full(n, WALL_Z) + rng.normal(0, jitter, n)
    pts = np.stack([x, y, z], axis=1)
    col = np.stack(
        [128 + 100 * x, 128 - 60 * y, np.full(n, 90.0)], axis=1
    )
    return pts, np.clip(col, 0, 255)


def scene(with_outliers: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    frames = []
    for _ in range(5):
        pts, col = wall_frame(700, 0.01)
        if with_outliers:
            k = 60
            out = rng.uniform(-6, 6, (k, 3))
            pts = np.concatenate([pts, out])
            col = np.concatenate([col, np.full((k, 3), 255.0)])
        frames.append((pts, col))
    return frames


def run(method_id: str, params: dict, frames, max_points=40_000):
    m = build(method_id, params)
    t0 = time.perf_counter()
    for pts, col in frames:
        m.add(pts, col, CAM, [])
    pos, colours, feat = m.cloud(max_points)
    return m, pos, colours, feat, (time.perf_counter() - t0) * 1000


print("registry")
methods = list_methods()
check("methods exposed", len(methods) >= 4, ", ".join(m["id"] for m in methods))
for m in methods:
    check(f"{m['id']} has params", len(m["params"]) >= 5)
check("coerce clamps", coerce_params("voxel", {"voxel": 99.0})["voxel"] == 0.25)
check("coerce falls back", coerce_params("voxel", {"voxel": "junk"})["voxel"] == 0.05)
# Compare against the declared default rather than a literal: the default has
# changed once already (auto -> fixed) and this should not break when it does.
check(
    "coerce rejects bad enum",
    coerce_params("voxel", {"scaleMode": "nope"})["scaleMode"]
    == default_params("voxel")["scaleMode"],
)
check(
    "coerce keeps a valid enum",
    coerce_params("voxel", {"scaleMode": "hit"})["scaleMode"] == "hit",
)
check(
    "params are json-stable",
    json.dumps(coerce_params("voxel", {}), sort_keys=True)
    == json.dumps(default_params("voxel"), sort_keys=True),
    "the map-rebuild key depends on this",
)

clean = scene(False)

print("\nvoxel (clean wall)")
m, pos, col, feat, ms = run("voxel", coerce_params("voxel", {}), clean)
check("returned points", pos.shape[0] > 50, f"{pos.shape[0]} pts")
check("z centred on wall", abs(float(np.median(pos[:, 2])) - WALL_Z) < 0.03,
      f"median z={np.median(pos[:, 2]):.3f}")
check("stays a thin shell", float(np.ptp(pos[:, 2])) < 0.25,
      f"z-extent={np.ptp(pos[:, 2]):.3f} m")
check("colours are uint8", col.dtype == np.uint8)
check("voxels counted", m.size > 0, f"{m.size} voxels, {ms:.0f} ms")

print("\ntsdf (clean wall)")
m, pos, col, feat, ms = run("tsdf", coerce_params("tsdf", {}), clean)
check("returned surface points", pos.shape[0] > 50, f"{pos.shape[0]} pts")
check("z centred on wall", abs(float(np.median(pos[:, 2])) - WALL_Z) < 0.05,
      f"median z={np.median(pos[:, 2]):.3f}")
check("thinner than a voxel shell", float(np.ptp(pos[:, 2])) <= 0.20,
      f"z-extent={np.ptp(pos[:, 2]):.3f} m")
check("x/y inside the wall", float(np.ptp(pos[:, 0])) < 2.3,
      f"x-extent={np.ptp(pos[:, 0]):.2f}")

print("\npoints + outlier removal")
noisy = scene(True)
m, pos, col, feat, ms = run("points", coerce_params("points", {}), noisy)
check("keeps everything by default", pos.shape[0] >= 3500, f"{pos.shape[0]} pts")
m, pos, col, feat, ms = run(
    "points", coerce_params("points", {"sorK": 12, "sorStd": 1.0}), noisy
)
far = np.sum(np.linalg.norm(pos - CAM[None, :], axis=1) > 4.0)
check("outliers removed", far < 20, f"{far} stray points left")
check("wall survived", pos.shape[0] > 500, f"{pos.shape[0]} pts, {ms:.0f} ms")

print("\ndbscan")
frames = []
for pts, col in noisy:
    keep = np.linalg.norm(pts - CAM[None, :], axis=1) < 5.0
    frames.append((pts[keep], col[keep]))
m, pos, col, feat, ms = run("dbscan", coerce_params("dbscan", {}), frames)
check("cluster kept", pos.shape[0] > 500, f"{pos.shape[0]} pts")
check("z centred on wall", abs(float(np.median(pos[:, 2])) - WALL_Z) < 0.05,
      f"median z={np.median(pos[:, 2]):.3f}")

print("\nempty input")
for mid in ("voxel", "points", "tsdf", "dbscan"):
    mm = build(mid, coerce_params(mid, {}))
    mm.add(np.empty((0, 3)), np.empty((0, 3)), CAM, [])
    p, c, f = mm.cloud()
    check(f"{mid} survives empty", p.shape[0] == 0)

print()
if failures:
    print(f"{BAD}: {failures} check(s) failed")
    sys.exit(1)
print(f"{OK}: all checks passed")
