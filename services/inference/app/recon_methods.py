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
            "scaleMode",
            "Metric scale from",
            "enum",
            "fixed",
            choices=("fixed", "auto", "planes", "hit", "none"),
            help="fixed = use a/b below (the default: on the test device both "
            "ARCore references measure ~5x too large); auto = plane fit "
            "cross-checked against the hit-test; planes/hit = that reference "
            "alone; none = raw model units (a=1)",
        ),
        Param(
            "scaleA",
            "Scale a  (Z = a·d + b)",
            "float",
            0.7,
            0.01,
            # Fitted by the y-origin test: the fused room must have its floor on
            # ARCore's local-floor plane (y=0). Not universal - Depth Anything
            # V2's scale drifts per image - but it is right for this room, and
            # it is what makes the baseline reconstruction correct rather than
            # 5x too big. Dial it in the viewer.
            30.0,
            0.01,
            help="used when scaleMode=fixed; 0.7 puts the floor on y=0 here",
        ),
        Param(
            "scaleB",
            "Offset b",
            "float",
            0.0,
            -5.0,
            5.0,
            0.01,
            help="0 = pure scale; nonzero shears near/far (usually a bad model)",
        ),
        Param(
            "fovDeg",
            "FOV override",
            "float",
            0.0,
            0.0,
            130.0,
            0.5,
            help="0 = use the FOV the phone reported; set it to test the "
            "intrinsics hypothesis, e.g. 50 or 65",
        ),
        Param("stride", "Pixel stride", "int", 2, 1, 6, 1,
              help="sample every Nth pixel - higher is faster and sparser"),
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
            Param("minWeight", "Min observations", "float", 1.0, 1.0, 10.0, 1.0),
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

    def __init__(self, params: dict[str, Any]) -> None:
        self.p = params

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
    """Truncated signed distance field, read out at the zero crossing.

    For every observed surface point p seen from camera c we update the voxels
    in a (2k+1)^3 block around it with

        sdf = clamp( (p - v) . r_hat / trunc, -1, +1 )     r_hat = unit(c -> p)

    i.e. the signed distance to the surface *along the viewing ray*. Positive
    means the voxel sits in front of the surface (empty space), which matches
    the usual TSDF sign convention. Averaging over keyframes is what makes this
    robust: a stray depth reading is outvoted rather than baked in.

    Reading out uses the sign change between neighbouring voxels to find the
    surface, which is the zero crossing without needing a mesh (§FUSION.md).
    """

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self.voxel = float(params.get("voxel", 0.04))
        self.trunc = max(float(params.get("trunc", 0.12)), self.voxel * 0.5)
        self.min_weight = float(params.get("minWeight", 1.0))
        self.acc = HashAccum(nvals=4)  # rgb + sdf
        self.updates = 0

    def add(self, points, colors, cam_pos, planes) -> int:
        n = int(points.shape[0])
        if n == 0:
            return 0
        v = self.voxel
        k = max(1, int(round(self.trunc / v)))
        offsets = np.array(
            list(itertools.product(range(-k, k + 1), repeat=3)), dtype=np.float64
        )

        ray = points - cam_pos[None, :]
        ray /= np.linalg.norm(ray, axis=1, keepdims=True) + 1e-9
        base = np.floor(points / v)

        # Chunk so the (points x offsets) block stays bounded.
        step = max(1, int(200_000 / max(offsets.shape[0], 1)))
        for s in range(0, n, step):
            e = min(n, s + step)
            centres = ((base[s:e, None, :] + offsets[None, :, :]) + 0.5) * v
            d = points[s:e, None, :] - centres
            sdf = np.sum(d * ray[s:e, None, :], axis=2) / self.trunc
            np.clip(sdf, -1.0, 1.0, out=sdf)

            keys = voxel_keys(centres.reshape(-1, 3), v)
            sdf_flat = sdf.reshape(-1)
            col = np.repeat(colors[s:e], offsets.shape[0], axis=0)
            vals = np.concatenate([col, sdf_flat[:, None]], axis=1)
            self.acc.add(keys, vals)
        self.updates += 1
        return n

    @property
    def size(self) -> int:
        return self.acc.size

    def note(self) -> str:
        return f"trunc {self.trunc:.2f} m"

    def cloud(self, max_points: int = 30_000):
        acc = self.acc
        if acc.keys.size == 0:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), 0.0

        mu = acc.sum / np.maximum(acc.w, 1e-9)[:, None]
        sdf = mu[:, 3]
        centres = acc.centres(self.voxel)
        pos_sign = sdf > 0

        # Surface = a voxel whose sign disagrees with a neighbour along any axis.
        surface = np.zeros(acc.keys.size, dtype=bool)
        for axis, shift in enumerate((1 << 42, 1 << 21, 1)):
            nk = acc.keys + shift
            at = np.clip(np.searchsorted(acc.keys, nk), 0, acc.keys.size - 1)
            found = acc.keys[at] == nk
            surface |= found & (pos_sign[at] != pos_sign)

        keep = surface & (acc.w >= self.min_weight) & (np.abs(sdf) <= 1.0)
        if not keep.any():
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), self.voxel

        p, c = centres[keep], np.clip(mu[keep, :3], 0, 255)
        if p.shape[0] > max_points:
            idx = np.linspace(0, p.shape[0] - 1, max_points).astype(np.int64)
            p, c = p[idx], c[idx]
        return p.astype(np.float32), c.astype(np.uint8), self.voxel


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
