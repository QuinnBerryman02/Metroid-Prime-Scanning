"""Pluggable dense-reconstruction methods (MS3.2+).

Every method consumes the same per-keyframe input - metric world points with
their colours, the camera position, and ARCore's planes - and maintains its own
map. They differ in what they *keep*:

  voxel   running-mean voxel accumulator (the baseline; docs/FUSION.md §2)
  points  raw points, optionally decimated and outlier-filtered on read
  tsdf    truncated signed distance field -> zero-crossing surface voxels
  dbscan  voxel map, then DBSCAN cluster filtering when read out

Params are declared per method so the viewer can build its own controls from
`GET /recon/methods` - this registry is the single source of truth, so adding a
knob here is all it takes for it to show up in the UI.

The scale/intrinsics knobs are deliberately part of every method: global metric
scale is *unobservable* from monocular geometry (docs/RECONSTRUCTION.md), so it
has to be supplied from a metric reference, and being able to override it by
hand is the fastest way to see whether a given frame is scaled or warped.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np

from .fusion import VoxelMap, voxel_keys
from .tsdf import TSDFVolume

# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Param:
    """One tunable knob. `kind` drives which control the viewer renders."""

    key: str
    label: str
    kind: str  # "float" | "int" | "bool" | "enum"
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple[str, ...] | None = None
    help: str = ""

    def json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "kind": self.kind,
            "default": self.default,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "choices": list(self.choices) if self.choices else None,
            "help": self.help,
        }


@dataclass(frozen=True)
class Method:
    id: str
    label: str
    description: str
    params: tuple[Param, ...]

    def json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "params": [p.json() for p in self.params],
        }


def common_params() -> tuple[Param, ...]:
    """The knobs that decide whether the cloud is the right *shape and size*."""
    return (
        Param(
            "depthFilter",
            "Depth filter",
            "enum",
            "median",
            choices=("none", "median", "bilateral"),
            help="median kills speckle; bilateral keeps edges but smooths flat",
        ),
        Param("minDepth", "Min depth (m)", "float", 0.2, 0.05, 5.0, 0.05),
        Param("maxDepth", "Max depth (m)", "float", 12.0, 1.0, 50.0, 0.5),
    )


REGISTRY: tuple[Method, ...] = (
    Method(
        "voxel",
        "Voxel mean (baseline)",
        "Running-mean voxel accumulator. Resolution is coupled to accuracy, so "
        "it smears; this is the reference the others are judged against.",
        common_params()
        + (Param("voxel", "Voxel size (m)", "float", 0.05, 0.01, 0.25, 0.005),),
    ),
    Method(
        "points",
        "Points + outlier removal",
        "Keeps every point (no decimation), then optionally voxel-downsamples "
        "and drops statistical outliers on read. Sharpest, but noisiest and "
        "unbounded - watch the point count.",
        common_params()
        + (
            Param("outVoxel", "Read-out voxel (0=off)", "float", 0.0, 0.0, 0.2, 0.005),
            Param("sorK", "Outlier k (0=off)", "int", 0, 0, 40, 1,
                  help="mean distance to these k neighbours; slow on big clouds"),
            Param("sorStd", "Outlier sigma", "float", 1.5, 0.2, 5.0, 0.1),
            Param("pointCap", "Point cap", "int", 400_000, 10_000, 3_000_000, 10_000),
        ),
    ),
    Method(
        "tsdf",
        "TSDF -> surface voxels",
        "Truncated signed distance field, updated per keyframe, read out at the "
        "zero crossing. Weighs observations, so it resists stray depth; the "
        "truncation and voxel size have to agree or the surface thickens.",
        common_params()
        + (
            Param("voxel", "Voxel size (m)", "float", 0.04, 0.01, 0.2, 0.005),
            Param("trunc", "Truncation (m)", "float", 0.12, 0.02, 0.6, 0.01,
                  help="~2-3x the voxel size; smaller = tighter but thinner"),
            Param("minWeight", "Min observations", "float", 1.0, 1.0, 10.0, 1.0,
                  help="a node needs this many depth observations before it counts"),
            Param("maxWeight", "Max weight (fusion memory)", "float", 40.0, 2.0, 500.0, 1.0,
                  help="caps the running average: low = adapts to new frames, high = stable"),
            Param("boxHalf", "Box half-size x/z (m)", "float", 4.0, 1.0, 10.0, 0.5,
                  help="the TSDF covers +/- this in x and z around the session origin"),
            Param("boxHalfY", "Box half-height y (m)", "float", 2.0, 0.5, 4.0, 0.25,
                  help="covers the floor and ceiling; the floor is ~1.1 m below the origin"),
        ),
    ),
    Method(
        "dbscan",
        "Voxel + DBSCAN clusters",
        "The voxel map, then DBSCAN on read-out: keeps dense clusters and drops "
        "the strays that belong to no surface. Good for seeing what the object "
        "pool in MS3.3 will actually get.",
        common_params()
        + (
            Param("voxel", "Voxel size (m)", "float", 0.05, 0.01, 0.25, 0.005),
            Param("eps", "Cluster radius (m)", "float", 0.12, 0.02, 1.0, 0.01),
            Param("minSamples", "Min samples", "int", 6, 2, 50, 1),
            Param("minCluster", "Min cluster pts", "int", 30, 5, 2000, 5),
        ),
    ),
)

_BY_ID = {m.id: m for m in REGISTRY}


def list_methods() -> list[dict[str, Any]]:
    return [m.json() for m in REGISTRY]


def resolve(method_id: str) -> str:
    """Fall back to the baseline rather than letting an unknown id through."""
    return method_id if method_id in _BY_ID else "voxel"


def default_params(method_id: str) -> dict[str, Any]:
    m = _BY_ID.get(method_id)
    return {p.key: p.default for p in m.params} if m else {}


def coerce_params(method_id: str, raw: dict[str, Any] | None) -> dict[str, Any]:
    """Merge caller params over the defaults, clamped to each Param's range.

    Never trust the client: an out-of-range voxel size can allocate gigabytes.
    """
    m = _BY_ID.get(method_id)
    if m is None:
        return {}
    out = default_params(method_id)
    for p in m.params:
        if not raw or p.key not in raw:
            continue
        v = raw[p.key]
        try:
            if p.kind == "int":
                v = int(round(float(v)))
            elif p.kind == "float":
                v = float(v)
            elif p.kind == "bool":
                v = bool(v)
            elif p.kind == "enum":
                if v not in (p.choices or ()):
                    continue
            else:
                continue
        except (TypeError, ValueError):
            continue
        if p.kind in ("int", "float"):
            if p.min is not None:
                v = max(p.min, v)
            if p.max is not None:
                v = min(p.max, v)
        out[p.key] = v
    return out


def build(method_id: str, params: dict[str, Any]) -> "BaseRecon":
    cls = _IMPL.get(method_id, VoxelRecon)
    return cls(params)


# --------------------------------------------------------------------------
# Accumulators
# --------------------------------------------------------------------------


class BaseRecon:
    """Interface every method implements. `cloud()` is the read-out."""

    # Only methods that build a surface set this; the viewer draws it if present.
    has_mesh = False

    def __init__(self, params: dict[str, Any]) -> None:
        self.p = params

    def integrate_depth(self, depth, rgb, world_from_cam, intr) -> int:
        """Full-resolution depth hook. Default: methods ignore it."""
        return 0

    def mesh(self):
        return None

    def add(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        cam_pos: np.ndarray,
        planes: list[np.ndarray],
    ) -> int:
        """Fold one keyframe in. Returns how many points were consumed."""
        return 0

    @property
    def size(self) -> int:
        return 0

    def cloud(self, max_points: int = 30_000) -> tuple[np.ndarray, np.ndarray, float]:
        """(positions float32 Nx3, colours uint8 Nx3, feature size in m)."""
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), 0.0

    def note(self) -> str:
        """Short extra description shown next to the method name."""
        return ""


class HashAccum:
    """Sorted-key accumulator holding a running mean of a small value vector.

    Same contract as `VoxelMap` but generic in what it averages, so TSDF can
    carry a signed distance alongside the colour. Keys must be comparable
    int64 (see `fusion.voxel_keys`), and are kept sorted for log-time merging.
    """

    def __init__(self, nvals: int, capacity: int = 3_000_000) -> None:
        self.capacity = capacity
        self.keys = np.empty(0, dtype=np.int64)
        self.sum = np.empty((0, nvals), dtype=np.float64)
        self.w = np.empty(0, dtype=np.float64)

    @property
    def size(self) -> int:
        return int(self.keys.size)

    def add(self, keys: np.ndarray, vals: np.ndarray) -> int:
        """Merge one frame's (key, value) observations into the running mean."""
        if keys.size == 0:
            return 0
        uniq, inv = np.unique(keys, return_inverse=True)
        inv = inv.reshape(-1)
        n = int(uniq.size)

        # Per-voxel sums and observation counts for this frame.
        cnt = np.bincount(inv, minlength=n).astype(np.float64)
        agg = np.empty((n, vals.shape[1]), dtype=np.float64)
        for j in range(vals.shape[1]):
            agg[:, j] = np.bincount(inv, weights=vals[:, j], minlength=n)

        gk = self.keys
        if gk.size == 0:
            self.keys, self.sum, self.w = uniq, agg, cnt
            return n

        pos = np.clip(np.searchsorted(gk, uniq), 0, gk.size)
        exists = np.zeros(n, dtype=bool)
        ok = pos < gk.size
        exists[ok] = gk[pos[ok]] == uniq[ok]

        idx = pos[exists]
        self.sum[idx] += agg[exists]
        self.w[idx] += cnt[exists]

        if not exists.all():
            newk = uniq[~exists]
            ins = np.searchsorted(gk, newk)
            self.keys = np.insert(gk, ins, newk)
            self.sum = np.insert(self.sum, ins, agg[~exists], axis=0)
            self.w = np.insert(self.w, ins, cnt[~exists])

        if self.keys.size > self.capacity:
            keep = np.argsort(-self.w)[: self.capacity]
            keep.sort()
            self.keys, self.sum, self.w = self.keys[keep], self.sum[keep], self.w[keep]
        return n

    def centres(self, voxel: float) -> np.ndarray:
        """Decode a voxel key back to the centre of its cell."""
        m = (1 << 21) - 1
        q = np.stack(
            [
                (self.keys >> 42) & m,
                (self.keys >> 21) & m,
                self.keys & m,
            ],
            axis=1,
        ) - (1 << 20)
        return (q + 0.5) * voxel


# --------------------------------------------------------------------------
# Methods
# --------------------------------------------------------------------------


class VoxelRecon(BaseRecon):
    """Running-mean voxel map - the baseline (docs/FUSION.md §2)."""

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self.voxel = float(params.get("voxel", 0.05))
        self.map = VoxelMap(voxel=self.voxel)

    def add(self, points, colors, cam_pos, planes) -> int:
        self.map.integrate(points, colors)
        return int(points.shape[0])

    @property
    def size(self) -> int:
        return self.map.size

    def cloud(self, max_points: int = 30_000):
        pos, col, used = self.map.snapshot(voxel=0.0, max_points=max_points)
        return pos, col, float(used)


class PointsRecon(BaseRecon):
    """Keep raw points; decimate and outlier-filter only on read-out."""

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self.cap = int(params.get("pointCap", 400_000))
        self.pos = np.empty((0, 3), dtype=np.float64)
        self.col = np.empty((0, 3), dtype=np.float64)
        self.dropped = 0

    def add(self, points, colors, cam_pos, planes) -> int:
        if points.shape[0] == 0:
            return 0
        self.pos = np.concatenate([self.pos, points])
        self.col = np.concatenate([self.col, colors])
        if self.pos.shape[0] > self.cap:
            # Keep the newest: a replay is a walk, so recent views are the ones
            # the user is looking at.
            cut = self.pos.shape[0] - self.cap
            self.dropped += cut
            self.pos = self.pos[cut:]
            self.col = self.col[cut:]
        return int(points.shape[0])

    @property
    def size(self) -> int:
        return int(self.pos.shape[0])

    def note(self) -> str:
        return f"{self.dropped} dropped" if self.dropped else ""

    def cloud(self, max_points: int = 30_000):
        pos, col = self.pos, self.col
        if pos.shape[0] == 0:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), 0.0

        k = int(self.p.get("sorK", 0))
        if k > 1 and pos.shape[0] > k:
            # Statistical outlier removal: drop points whose mean distance to
            # their k nearest neighbours is well above the cloud's average.
            from sklearn.neighbors import NearestNeighbors

            nn = NearestNeighbors(n_neighbors=min(k, pos.shape[0])).fit(pos)
            dist, _ = nn.kneighbors(pos)
            mean_d = dist[:, 1:].mean(axis=1)
            thr = mean_d.mean() + float(self.p.get("sorStd", 1.5)) * mean_d.std()
            keep = mean_d <= thr
            pos, col = pos[keep], col[keep]

        out_voxel = float(self.p.get("outVoxel", 0.0))
        if out_voxel > 0 and pos.shape[0]:
            kk = voxel_keys(pos, out_voxel)
            _, idx = np.unique(kk, return_index=True)
            pos, col = pos[idx], col[idx]

        feat = out_voxel if out_voxel > 0 else 0.0
        if pos.shape[0] > max_points:
            idx = np.linspace(0, pos.shape[0] - 1, max_points).astype(np.int64)
            pos, col = pos[idx], col[idx]
        return pos.astype(np.float32), np.clip(col, 0, 255).astype(np.uint8), feat


class TsdfRecon(BaseRecon):
    """Dense TSDF (app/tsdf.py): the only method that produces a surface.

    Each keyframe integrates the full-resolution metric depth into a truncated
    signed distance field; the surface is the zero crossing, extracted with
    marching cubes. The point cloud is the mesh vertices, so the viewer's cloud
    and surface always agree.
    """

    has_mesh = True

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        from . import config

        self.voxel = float(params.get("voxel", 0.04))
        self.trunc = max(float(params.get("trunc", 0.12)), self.voxel * 1.5)
        self.min_weight = float(params.get("minWeight", 1.0))
        self.min_depth = float(params.get("minDepth", 0.2))
        self.max_depth = float(params.get("maxDepth", 12.0))
        half = float(params.get("boxHalf", 4.0))
        half_y = float(params.get("boxHalfY", 2.0))
        # Memory guard: a 10 m box at 4 cm would be ~50M nodes (~1 GB). Grow the
        # voxel rather than the allocation; the note says when that happened.
        cap = 8_000_000
        voxel = self.voxel
        while (2 * half / voxel + 1) * (2 * half_y / voxel + 1) * (2 * half / voxel + 1) > cap:
            voxel *= 1.1
        self.box_voxel = voxel
        self.vol = TSDFVolume(
            (-half, -half_y, -half),
            (half, half_y, half),
            voxel=voxel,
            trunc=max(self.trunc, voxel * 1.5),
            max_weight=float(params.get("maxWeight", 40.0)),
        )
        self._mesh_cache: tuple[int, tuple] | None = None

    def integrate_depth(self, depth, rgb, world_from_cam, intr) -> int:
        self._mesh_cache = None
        # Same depth range the point methods use, so the methods are comparable.
        d = np.where((depth >= self.min_depth) & (depth <= self.max_depth), depth, np.nan)
        return self.vol.integrate(d, rgb, world_from_cam, intr)

    @property
    def size(self) -> int:
        return self.vol.observed_nodes

    def note(self) -> str:
        grown = "" if self.box_voxel == self.voxel else f" (grown to {self.box_voxel:.3f} m to fit)"
        return f"TSDF {self.box_voxel:.3f} m{grown}, trunc {self.trunc:.2f} m"

    def mesh(self):
        """(vertices, faces, colors), cached until the next keyframe."""
        if self._mesh_cache is None or self._mesh_cache[0] != self.vol.frames:
            self._mesh_cache = (self.vol.frames, self.vol.extract(self.min_weight))
        return self._mesh_cache[1]

    def cloud(self, max_points: int = 30_000):
        v, _f, c = self.mesh()
        if v.shape[0] == 0:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), self.voxel
        if v.shape[0] > max_points:
            idx = np.linspace(0, v.shape[0] - 1, max_points).astype(np.int64)
            v, c = v[idx], c[idx]
        return v, c, self.voxel


class DbscanRecon(BaseRecon):
    """Voxel map read out through DBSCAN, keeping only dense clusters."""

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self.voxel = float(params.get("voxel", 0.05))
        self.map = VoxelMap(voxel=self.voxel)
        self.kept = 0

    def add(self, points, colors, cam_pos, planes) -> int:
        self.map.integrate(points, colors)
        return int(points.shape[0])

    @property
    def size(self) -> int:
        return self.map.size

    def note(self) -> str:
        return f"{self.kept} pts kept" if self.kept else ""

    def cloud(self, max_points: int = 30_000):
        pos, col, used = self.map.snapshot(voxel=0.0, max_points=200_000)
        if pos.shape[0] < 2:
            return pos, col, float(used)

        from sklearn.cluster import DBSCAN

        labels = DBSCAN(
            eps=float(self.p.get("eps", 0.12)),
            min_samples=int(self.p.get("minSamples", 6)),
            n_jobs=-1,
        ).fit_predict(pos.astype(np.float64))

        min_cluster = int(self.p.get("minCluster", 30))
        if min_cluster > 1:
            ids, counts = np.unique(labels[labels >= 0], return_counts=True)
            good = set(ids[counts >= min_cluster].tolist())
            keep = np.isin(labels, list(good)) if good else np.zeros_like(labels, bool)
        else:
            keep = labels >= 0
        pos, col = pos[keep], col[keep]
        self.kept = int(pos.shape[0])

        if pos.shape[0] > max_points:
            idx = np.linspace(0, pos.shape[0] - 1, max_points).astype(np.int64)
            pos, col = pos[idx], col[idx]
        return pos, col, float(used)


_IMPL = {
    "voxel": VoxelRecon,
    "points": PointsRecon,
    "tsdf": TsdfRecon,
    "dbscan": DbscanRecon,
}
