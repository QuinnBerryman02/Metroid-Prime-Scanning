"""Scannable-object candidates: pair YOLO masks across keyframes.

One detection is one view, which can't place an object in the room. Two views of
the same label whose centroids agree, and whose cameras are far enough apart
(the baseline), become an object. Later views that agree attach to it.

Geometry comes from `unproject_depth`, the same code that builds the room cloud,
so object centroids and the room share one frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import base64

import cv2
import numpy as np

from .fusion import ScaleFit, unproject_depth

MIN_SCORE = 0.3  # YOLO confidence floor
MIN_POINTS = 50  # masked depth points needed for a usable centroid
MAX_RANGE = 6.0  # m; detections whose median depth is farther are depth errors
MATCH_RADIUS = 0.5  # m, closest approach of two camera rays (or ray to object) for a match
MERGE_RADIUS = 1.0  # m, same-label objects closer than this are one object
MIN_BASELINE_DEG = 20.0  # angle at the object between the two camera positions
MIN_BASELINE_M = 0.3  # the cameras must be this far apart in metres (angles alone pass for near points)
DEPTH_AGREE = 1.0  # m; a meeting point must sit this close to each view's measured centroid
PENDING_MAX = 200  # unmatched observations kept per tracker
MAX_OBS = 60  # observations kept per object (oldest dropped)
CANDIDATE_VIEWS = 3  # two views = tentative; three or more = candidate
# Carving (object extent): room depth points that are visible in the object's
# views and inside its masks, not in the background.
CARVE_VIEWS = 24  # most recent views used
CARVE_VOXEL = 0.03  # m, dedupe grid for candidate points
CARVE_RADIUS = 1.5  # m around the object's centre
CARVE_MIN_VOTES = 2  # views that must see a point inside its mask
CARVE_RATIO = 0.7  # inside / (inside + outside) needed to keep a point
CARVE_MIN_POINTS = 30  # below this the carve is too thin to trust
CARVE_MAX_CANDIDATES = 40000  # cap on candidate points per carve (speed)

_METRIC = ScaleFit(1.0, 0.0, 0, 0.0, "metric")


@dataclass
class Obs:
    frame_id: str
    label: str
    score: float
    centroid: np.ndarray  # world, metres: median of the masked depth points
    cam: np.ndarray  # camera centre, world
    ray: np.ndarray  # unit vector, camera -> centroid
    # Evidence for the viewer (debug): the frame, the mask outline, the pose.
    jpeg: bytes = b""
    size: tuple[int, int] = (0, 0)  # (width, height) of the frame
    contour: list = field(default_factory=list)  # mask outline, normalised [x, y]
    wfc: np.ndarray | None = None  # worldFromCamera for this frame
    intr: tuple[float, float, float, float] | None = None
    # Half-resolution depth (metres, float16) and mask, for carving.
    depth: np.ndarray | None = None
    mask: np.ndarray | None = None


def _half_intr(intr):
    fx, fy, cx, cy = intr
    return fx / 2, fy / 2, cx / 2, cy / 2


def _unproject_half(o: "Obs") -> np.ndarray:
    """World points of a view's half-resolution depth (valid pixels only)."""
    depth = o.depth.astype(np.float64)
    h2, w2 = depth.shape
    fx, fy, cx, cy = _half_intr(o.intr)
    vs, us = np.mgrid[0:h2, 0:w2]
    z = depth
    ok = np.isfinite(z) & (z > 0.2) & (z < MAX_RANGE)
    z = z[ok]
    x = (us[ok] - cx) * z / fx
    y = (vs[ok] - cy) * z / fy
    cam = np.stack([x, y, z], axis=1)
    R_, t = o.wfc[:3, :3], o.wfc[:3, 3]
    return cam @ R_.T + t


def project(x, wfc, intr, w, h):
    """World point -> [u, v] normalised to the frame, plus depth. None if behind."""
    inv = np.linalg.inv(wfc)
    c = inv[:3, :3] @ x + inv[:3, 3]
    if c[2] <= 0.05:
        return None
    fx, fy, cx, cy = intr
    return [round(float(fx * c[0] / c[2] + cx) / w, 4),
            round(float(fy * c[1] / c[2] + cy) / h, 4),
            round(float(c[2]), 3)]


def outline(mask: np.ndarray, w: int, h: int) -> list[list[float]]:
    """Largest external contour of the mask, simplified, normalised to the frame."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2)
    return [[round(float(x) / w, 4), round(float(y) / h, 4)] for x, y in c]


def closest_approach(p0, d0, p1, d1):
    """Closest points between two rays (origin, unit dir). Returns (gap, t0, t1)."""
    w = p0 - p1
    b = np.dot(d0, d1)
    d, e = np.dot(d0, w), np.dot(d1, w)
    denom = 1.0 - b * b
    if denom < 1e-6:  # parallel: no triangulation
        return float("inf"), 0.0, 0.0
    t0 = (b * e - d) / denom
    t1 = (e - b * d) / denom
    gap = float(np.linalg.norm((p0 + t0 * d0) - (p1 + t1 * d1)))
    return gap, float(t0), float(t1)


def point_ray_distance(x, origin, d):
    v = x - origin
    t = max(np.dot(v, d), 0.0)
    return float(np.linalg.norm(v - t * d))


@dataclass
class Obj:
    id: int
    label: str
    obs: list[Obs] = field(default_factory=list)
    # Triangulated meeting points of the view pairs (robust to surface-centroid
    # drift between views). The object centre is their median.
    points: list[np.ndarray] = field(default_factory=list)
    centre: np.ndarray = field(default_factory=lambda: np.zeros(3))
    carved: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    _carve_n: int = -1  # len(obs) the carve was computed for

    def update_carve(self) -> None:
        """Recompute the carve when the object has gained views."""
        if self._carve_n == len(self.obs):
            return
        self.carved = self.carve()
        self._carve_n = len(self.obs)

    def anchor(self) -> np.ndarray:
        """Where the object is: the carved median if the carve is solid, else the centre."""
        if len(self.carved) >= CARVE_MIN_POINTS:
            return np.median(self.carved, axis=0)
        return self.centre

    def carve(self) -> np.ndarray:
        """Room depth points that every seeing view places inside its mask.

        For each candidate point, each view where it is visible (its depth matches
        the measured surface) votes inside if it projects into the mask, outside
        otherwise. Points in front of a surface outside the mask are free space
        (outside). Points behind a surface are no evidence. Views that disagree
        with a point's depth add nothing, so a warped view can't carve in alone.
        """
        views = [o for o in self.obs[-CARVE_VIEWS:] if o.depth is not None and o.wfc is not None]
        if len(views) < 2:
            return np.zeros((0, 3))
        cands = []
        for o in views:
            P = _unproject_half(o)
            if len(P):
                cands.append(P[np.linalg.norm(P - self.centre, axis=1) < CARVE_RADIUS])
        if not cands:
            return np.zeros((0, 3))
        P = np.concatenate(cands)
        if len(P) == 0:
            return P
        q = np.round(P / CARVE_VOXEL).astype(np.int64) + 100000  # keep keys positive
        key = (q[:, 0] * 2_000_003 + q[:, 1]) * 2_000_003 + q[:, 2]
        _, idx = np.unique(key, return_index=True)
        P = P[idx]
        # Speed cap: the carve is a per-keyframe cost, so thin the candidates.
        if len(P) > CARVE_MAX_CANDIDATES:
            P = P[:: len(P) // CARVE_MAX_CANDIDATES + 1]

        inside = np.zeros(len(P), np.int32)
        outside = np.zeros(len(P), np.int32)
        for o in views:
            fx, fy, cx, cy = _half_intr(o.intr)
            depth, mask = o.depth.astype(np.float64), o.mask
            h2, w2 = depth.shape
            inv = np.linalg.inv(o.wfc)
            Xc = P @ inv[:3, :3].T + inv[:3, 3]
            z = Xc[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = fx * Xc[:, 0] / z + cx
                v = fy * Xc[:, 1] / z + cy
            ui = np.round(np.nan_to_num(u, nan=-1)).astype(np.int64)
            vi = np.round(np.nan_to_num(v, nan=-1)).astype(np.int64)
            in_img = (z > 0.2) & (ui >= 0) & (ui < w2) & (vi >= 0) & (vi < h2)
            uc, vc = np.clip(ui, 0, w2 - 1), np.clip(vi, 0, h2 - 1)
            d = depth[vc, uc]
            valid = in_img & np.isfinite(d) & (d > 0.2)
            tol = 0.05 + 0.04 * z
            vis = valid & (np.abs(z - d) < tol)
            front = valid & (z < d - tol)
            m = mask[vc, uc]
            inside += (vis & m).astype(np.int32)
            outside += ((vis & ~m) | (front & ~m)).astype(np.int32)

        total = inside + outside
        keep = (inside >= CARVE_MIN_VOTES) & (inside >= CARVE_RATIO * total)
        return P[keep]

    @property
    def views(self) -> int:
        return len({o.frame_id for o in self.obs})

    def add(self, o: Obs, point: np.ndarray | None = None) -> None:
        self.obs.append(o)
        del self.obs[:-MAX_OBS]
        if point is not None:
            self.points.append(point)
            del self.points[:-MAX_OBS]
        self.recentre()

    def absorb(self, other: "Obj") -> None:
        self.obs = (self.obs + other.obs)[-MAX_OBS:]
        self.points = (self.points + other.points)[-MAX_OBS:]
        self.recentre()

    def recentre(self) -> None:
        pts = self.points or [o.centroid for o in self.obs]
        self.centre = np.median(np.stack(pts), axis=0)

    def views_payload(self, limit: int = 12) -> list[dict[str, Any]]:
        """Per-view evidence: frame JPEG, mask outline, and where the centre projects."""
        out: list[dict[str, Any]] = []
        for o in self.obs[-limit:]:
            if not o.jpeg or o.wfc is None or o.intr is None:
                continue
            w, h = o.size
            out.append({
                "frameId": o.frame_id,
                "score": round(o.score, 3),
                "width": w,
                "height": h,
                "jpeg": base64.b64encode(o.jpeg).decode(),
                "contour": o.contour,
                "centroid": project(o.centroid, o.wfc, o.intr, w, h),
                "centre": project(self.anchor(), o.wfc, o.intr, w, h),
            })
        return out

    def summary(self) -> dict[str, Any]:
        if len(self.carved) >= CARVE_MIN_POINTS:
            lo = np.percentile(self.carved, 3, axis=0)
            hi = np.percentile(self.carved, 97, axis=0)
            centre = (lo + hi) / 2
            half = np.clip((hi - lo) / 2 + 0.03, 0.1, 1.5)
        else:
            pts = np.stack([o.centroid for o in self.obs])
            centre = self.centre
            half = np.clip(np.percentile(np.abs(pts - centre), 75, axis=0) + 0.1, 0.15, 1.5)
        return {
            "id": self.id,
            "label": self.label,
            "score": round(max(o.score for o in self.obs), 3),
            "views": self.views,
            "carved": int(len(self.carved)),
            "anchor": [round(float(v), 3) for v in self.anchor()],
            "centre": [round(float(v), 3) for v in centre],
            "halfExtent": [round(float(v), 3) for v in half],
            "status": "candidate" if self.views >= CANDIDATE_VIEWS else "tentative",
        }


class ObjectTracker:
    def __init__(self) -> None:
        self.pending: list[Obs] = []
        self.objects: list[Obj] = []
        self._next = 1

    def observe(self, o: Obs) -> Obj | None:
        """Attach to an object, or pair with a pending view. Returns the object touched."""
        # A view agrees with an object when its ray passes near the object's centre.
        for obj in self.objects:
            if obj.label == o.label and point_ray_distance(obj.centre, o.cam, o.ray) < MATCH_RADIUS:
                obj.add(o)
                return obj

        # Two views agree when their rays nearly meet in front of both cameras,
        # and the cameras are far enough apart that the meeting point is well
        # constrained (baseline angle at that point).
        for p in self.pending:
            if p.label != o.label or p.frame_id == o.frame_id:
                continue
            gap, tp, to = closest_approach(p.cam, p.ray, o.cam, o.ray)
            if gap >= MATCH_RADIUS or tp <= 0 or to <= 0:
                continue
            x = ((p.cam + tp * p.ray) + (o.cam + to * o.ray)) / 2
            a, b = p.cam - x, o.cam - x
            cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
            angle = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
            if angle < MIN_BASELINE_DEG:
                continue
            if np.linalg.norm(p.cam - o.cam) < MIN_BASELINE_M:
                continue
            # Consistency with the depth model: the meeting point must be where
            # each view measured the object. Rays that meet next to a camera fail.
            if np.linalg.norm(x - p.centroid) > DEPTH_AGREE or np.linalg.norm(x - o.centroid) > DEPTH_AGREE:
                continue
            self.pending.remove(p)
            # One object per place: if an object of this label is already nearby,
            # this pair is more evidence for it, not a new object.
            near = [
                obj for obj in self.objects
                if obj.label == o.label and np.linalg.norm(obj.centre - x) < MERGE_RADIUS
            ]
            if near:
                target = near[0]
                target.add(p)
                target.add(o, x)
                for other in near[1:]:
                    target.absorb(other)
                    self.objects.remove(other)
                return target
            obj = Obj(self._next, o.label, [p, o], [x])
            self._next += 1
            obj.recentre()
            self.objects.append(obj)
            return obj

        self.pending.append(o)
        del self.pending[:-PENDING_MAX]
        return None


def observations(
    frame_id: str,
    items: list[Any],
    rel: np.ndarray,
    world_from_cam: np.ndarray,
    intr: tuple[float, float, float, float],
    img: np.ndarray | None = None,
) -> list[Obs]:
    """One Obs per confident detection: the median world point of its masked pixels."""
    h, w = rel.shape[:2]
    pts, ys, xs = unproject_depth(rel, world_from_cam, intr, _METRIC, stride=1)
    cam = world_from_cam[:3, 3].astype(np.float64)
    out: list[Obs] = []
    jpeg = b""
    if img is not None:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpeg = buf.tobytes() if ok else b""
    # Detections the depth model puts beyond MAX_RANGE are depth errors.
    for it in items:
        if it.score < MIN_SCORE:
            continue
        mask = np.asarray(it.mask) > 0.5
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        inside = mask[ys, xs]
        if int(inside.sum()) < MIN_POINTS:
            continue
        centroid = np.median(pts[inside], axis=0).astype(np.float64)
        if np.linalg.norm(centroid - cam) > MAX_RANGE:
            continue
        ray = centroid - cam
        ray = ray / (np.linalg.norm(ray) + 1e-9)
        out.append(Obs(
            frame_id, it.label, float(it.score), centroid, cam, ray,
            jpeg=jpeg, size=(w, h), contour=outline(mask, w, h),
            wfc=np.asarray(world_from_cam, dtype=np.float64), intr=tuple(intr),
            depth=rel[::2, ::2].astype(np.float16),
            mask=mask[::2, ::2].copy(),
        ))
    return out
