"""
Metroid-scan inference service.

The client picks a `method` (see `GET /methods`) - SAM1, SAM2, FastSAM, YOLO or
DEPTH. `/scan` dispatches on it:

  * "prompt" methods (SAM1/SAM2) use the reticle as a point prompt and return
    one mask.
  * "detect" methods (FastSAM/YOLO11-seg) return 1..N labelled masks.
  * "depth" returns a colourised depth map.

For mask methods the aimed object is also lifted to camera-space geometry
(contour + OBB) for the phone to anchor - see `docs/LIFTING.md`. The server
stays pose-agnostic; the phone owns pose and the AR anchor API.

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image

from . import config
from . import recon_methods as recon
from .depth import DepthEstimator
from .detector import Detector
from .lifting import (
    contour2d,
    contour_from_mask,
    intrinsics_from_fov,
    make_obb,
    unproject_mask,
)
from .methods import METHODS, METHOD_SPECS, Segmenter, describe, device_for
from .fusion import (
    AFFINE_MIN_SPREAD,
    ScaleFit,
    ScaleTracker,
    encode_points,
    fit_metric_scale,
    pose_to_matrix,
    unproject_depth,
)
from .schemas import (
    Aimed,
    DepthResult,
    Detection,
    Obb,
    ReconCloud,
    ReconFrameRequest,
    ReconFrameResponse,
    ReconStatusResponse,
    ScanRequest,
    ScanResponse,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("inference")

state: dict[str, Any] = {}
_locks: dict[str, threading.Lock] = {}


def _try_load(name: str, factory: Any) -> Any:
    """Load an optional model; never let it stop the service."""
    try:
        return factory()
    except Exception as e:  # noqa: BLE001
        log.warning("'%s' unavailable (%s)", name, e)
        return None


def get_segmenter(method_id: str) -> Segmenter | None:
    """Lazily build and cache the segmenter for a method."""
    spec = METHODS.get(method_id)
    if spec is None or spec["kind"] == "depth":
        return None
    cache: dict[str, Segmenter | None] = state.setdefault("segmenters", {})
    if method_id in cache:
        return cache[method_id]
    # Per-method lock so the preload thread and a request never load twice.
    lock = _locks.setdefault(method_id, threading.Lock())
    with lock:
        if method_id not in cache:
            segmenter = _try_load(method_id, lambda: Segmenter(spec, device_for(spec)))
            if segmenter is not None:
                try:
                    segmenter.warmup()
                except Exception as e:  # noqa: BLE001
                    log.warning("warmup %s failed: %s", method_id, e)
            cache[method_id] = segmenter
        return cache[method_id]


def _preload_methods() -> None:
    """Warm every method in the background so the wheel feels instant.

    Detect methods (YOLO/FastSAM, iGPU) load first so the useful ones are ready
    almost immediately; SAM1/SAM2 (CPU) trickle in behind them.
    """
    ordered = sorted(METHOD_SPECS, key=lambda s: 0 if s["kind"] == "detect" else 1)
    for spec in ordered:
        if spec["kind"] != "depth":
            get_segmenter(spec["id"])


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state["detector"] = Detector(
        config.DETECT_MODEL,
        config.DEVICE,
        config.IMGSZ,
        config.CONF,
        config.EXPORT_OPENVINO,
    )
    state["depth"] = _try_load(
        "depth",
        lambda: DepthEstimator(
            config.DEPTH_MODEL,
            config.DEPTH_DEVICE,
            config.DEPTH_IR,
            config.EXPORT_OPENVINO,
        ),
    )
    state["segmenters"] = {}
    threading.Thread(target=_preload_methods, daemon=True).start()
    log.info("ready (device=%s)", config.DEVICE)
    yield
    state.clear()


app = FastAPI(title="metroid-scan inference", version="0.3.0", lifespan=lifespan)


def pick_aimed(
    detections: list[Detection], aim_x: float, aim_y: float, width: int, height: int
) -> int | None:
    """Index of the detection under the reticle, else the nearest by centre."""
    if not detections:
        return None
    ax, ay = aim_x * width, aim_y * height
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d.box
        if x1 <= ax <= x2 and y1 <= ay <= y2:
            return i
    best, best_dist = None, float("inf")
    for i, d in enumerate(detections):
        cx, cy = (d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2
        dist = (cx - ax) ** 2 + (cy - ay) ** 2
        if dist < best_dist:
            best, best_dist = i, dist
    return best


def _contains(box: list[float], x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _detector_label(image: np.ndarray, aim_x: float, aim_y: float, width: int, height: int) -> str:
    detector: Detector | None = state.get("detector")
    if detector is None:
        return "object"
    detections = detector.detect(image)
    idx = pick_aimed(detections, aim_x, aim_y, width, height)
    return detections[idx].label if idx is not None else "object"


def _colorize_depth(depth: np.ndarray) -> np.ndarray:
    lo, hi = float(depth.min()), float(depth.max())
    norm = (depth - lo) / (hi - lo + 1e-6)
    colour = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)


def _downscale(rgb: np.ndarray, max_edge: int = 512) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = min(1.0, max_edge / max(h, w))
    if scale >= 1.0:
        return rgb
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def _encode_png(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "device": config.DEVICE,
        "model": config.DETECT_MODEL,
        "loaded": "detector" in state,
        "loadedSegmenters": [k for k, v in state.get("segmenters", {}).items() if v],
        "depth": state.get("depth") is not None,
    }


@app.get("/methods")
def methods() -> dict[str, Any]:
    return {
        "device": config.DEVICE,
        "methods": [describe(spec) for spec in METHOD_SPECS],
    }


# ---------------------------------------------------------------------------
# MS3.1 + MS3.2 - dense reconstruction (docs/RECONSTRUCTION.md sections 6-7)
# ---------------------------------------------------------------------------

# One frame at a time through the GPU + fusion path. Frames that arrive while
# we are busy are dropped rather than queued - stale depth is worse than none.
_recon_lock = threading.Lock()


def _recon_state() -> dict[str, Any]:
    return state.setdefault("recon", {"keyframes": 0, "sessions": {}, "maps": {}})


def _scale_why(planes: list[dict], diag: dict[str, Any]) -> str:
    """One line explaining the scale fit, short enough to put on screen.

    Reads as a funnel: how many polygons we got, how many sample points landed
    in the image, how many of those we could pair, and how much depth they span.
    `inFrustum=0` means the pose/plane frames disagree; `planeZ<0` means the
    planes are behind the camera.
    """
    if not planes:
        return "no ARCore planes in this frame"
    nan = float("nan")
    return (
        f"{diag.get('polyCount', 0)}/{len(planes)} polys, "
        f"{diag.get('inFrustum', 0)} pts in frame, "
        f"{diag.get('samples', 0)} samples, spread "
        f"{diag.get('spread', 0.0):.2f}/{AFFINE_MIN_SPREAD:.1f} m, "
        f"planeZ {diag.get('planeZ', nan):.2f} m, rel {diag.get('relMed', nan):.2f}"
    )


def _record_keyframe(req: ReconFrameRequest, raw: bytes) -> None:
    """Persist a keyframe (JPEG + request JSON) so it can be replayed offline.

    On-device iteration is far too slow to explore a parameter space, so every
    walk-around is kept as a dataset: `scripts/replay_recon.py` can then re-fuse
    the same frames under different scales, voxel sizes and methods and compare
    them without touching the phone.
    """
    if not config.RECON_RECORD or not req.record:
        return
    try:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in req.sessionId)
        folder = config.RECON_DIR / (safe[:48] or "unknown")
        folder.mkdir(parents=True, exist_ok=True)
        idx = len(list(folder.glob("*.json"))) + 1
        (folder / f"{idx:04d}.jpg").write_bytes(raw)
        (folder / f"{idx:04d}.json").write_text(
            json.dumps(req.model_dump(exclude={"image"}), separators=(",", ":")),
            encoding="utf-8",
        )
    except Exception as e:  # noqa: BLE001 - recording must never break the loop
        log.warning("recon record failed: %s", e)


def _recon_entry(
    rs: dict[str, Any],
    session_id: str,
    method_id: str = "voxel",
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Per-session reconstruction method + scale tracker. Small LRU.

    Rebuilds on *any* parameter change, not just a method change. Two reasons,
    and both of them were silent bugs:

    * Methods read their params in `__init__` (voxel size, truncation, eps), so
      assigning `method.p` afterwards changed nothing that was already stored.
    * Nearly every param moves points: the scale decides where they land, the
      voxel size decides how they merge, the stride which of them exist. Reusing
      the map after a change fuses two different geometries together, which is
      what "fixed gives useless values" actually was - the fixed-scale frames
      were landing in a map built on the hit-test scale.
    """
    params = params or recon.default_params(method_id)
    key = json.dumps(params, sort_keys=True)
    maps: dict[str, Any] = rs["maps"]
    entry = maps.get(session_id)

    why: str | None = None
    if entry is not None and entry["method_id"] != method_id:
        why = f"method {entry['method_id']} -> {method_id}"
    elif entry is not None and entry["params_key"] != key:
        why = "parameters changed"
    if why is not None:
        log.info("recon %s map reset (%s)", session_id, why)
        entry = None

    if entry is None:
        while len(maps) >= 4:
            maps.pop(next(iter(maps)))
        entry = {
            "method_id": method_id,
            "params_key": key,
            "params": params,
            "method": recon.build(method_id, params),
            "tracker": ScaleTracker(),
            "fused": 0,
            "skipped": 0,
        }
        maps[session_id] = entry
    return entry


def _skip(
    rs: dict[str, Any],
    entry: dict[str, Any],
    why: str,
    hit_scale: float | None = None,
) -> ReconFrameResponse:
    entry["skipped"] += 1
    log.info("recon skipped (%s)", why)
    return ReconFrameResponse(
        keyframes=rs["keyframes"],
        fused=entry["fused"],
        skipped=entry["skipped"],
        voxels=entry["method"].size,
        reason=why,
        scaleHit=hit_scale,
        method=entry["method_id"],
        note=entry["method"].note(),
    )


def _fuse_keyframe(
    req: ReconFrameRequest, rs: dict[str, Any], entry: dict[str, Any]
) -> ReconFrameResponse:
    """Depth -> metric scale -> unproject -> fuse. MS3.1 + MS3.2.

    The *method* decides what is kept; everything up to the point cloud (depth,
    scale, intrinsics, unprojection) is shared, so the methods are comparable.
    """
    depth = state.get("depth")
    if depth is None:
        return _skip(rs, entry, "depth model unavailable")
    if not req.image or len(req.pose) != 16:
        return _skip(rs, entry, "missing image/pose")

    # Resolved (and clamped) once per session in _recon_entry, because these
    # values both drive the shared stages and decide whether the existing map
    # is still valid for them.
    params: dict[str, Any] = entry["params"]
    method: recon.BaseRecon = entry["method"]

    t0 = time.perf_counter()
    try:
        raw = base64.b64decode(req.image, validate=True)
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        return _skip(rs, entry, f"bad image: {e}")
    _record_keyframe(req, raw)
    img = np.array(pil)
    height, width = img.shape[:2]
    rs["last"] = {
        "frameId": req.frameId,
        "sessionId": req.sessionId,
        "width": width,
        "height": height,
        "fovDeg": req.fovDeg,
        "sharpness": req.sharpness,
        "jpegBytes": len(raw),
        "at": time.time() * 1000,
    }

    timing: dict[str, float] = {}

    # 1. relative depth (inverse of the model's disparity), metric scale unknown
    t = time.perf_counter()
    rel = depth.predict_depth(pil)
    if params["depthFilter"] == "median":
        rel = cv2.medianBlur(rel, 5)
    elif params["depthFilter"] == "bilateral":
        # Edge-preserving: smooths flat surfaces without smearing across the
        # boundary between a near object and the wall behind it.
        rel = cv2.bilateralFilter(
            rel.astype(np.float32), 5, 0.08 * abs(rel).max(), 6.0
        )
    timing["depth"] = round((time.perf_counter() - t) * 1000, 2)

    world_from_cam = pose_to_matrix(req.pose)
    # fovDeg=0 means "whatever the phone reported"; a real value overrides it,
    # which is how the intrinsics hypothesis gets tested against real frames.
    fov = params["fovDeg"] or req.fovDeg or config.FOV_DEG
    intr = intrinsics_from_fov(width, height, fov)

    # 2. MS3.1 - metric scale, from ARCore's planes. The tracker smooths the
    # per-frame estimate: fusing with a scale that jumps frame to frame smears
    # every surface into the surrounding space.
    t = time.perf_counter()
    diag: dict[str, Any] = {}
    fresh = fit_metric_scale(req.planes, world_from_cam, intr, rel, diag)
    why = _scale_why(req.planes, diag)

    # Independent cross-check: ARCore's hit-test gives a metric depth at the
    # reticle, which projects to roughly the image centre. The scan path has
    # always used this and its outlines landed correctly, so if the plane fit
    # and this disagree, the plane fit is the suspect.
    hit_scale: float | None = None
    if req.hitDistance and req.hitDistance > 0:
        rel_centre = float(rel[height // 2, width // 2])
        if rel_centre > 1e-6:
            hit_scale = req.hitDistance / rel_centre

    # A plane fit is only a fit if its residual is small. On real data the bad
    # ones show rms ~1-2 m over a ~1.6 m spread, i.e. noise, and they were being
    # averaged straight into the tracker.
    if fresh is not None and fresh.rms > config.RECON_SCALE_RMS_MAX:
        log.warning(
            "recon plane fit rejected: rms=%.3f m > %.2f (a=%.3f, %s)",
            fresh.rms,
            config.RECON_SCALE_RMS_MAX,
            fresh.a,
            fresh.method,
        )
        fresh = None

    # An independent metric measurement beats a bad fit. If the hit-test and the
    # planes disagree, trust the hit-test - it is the source the scan path has
    # always used, and it measures one point rather than fitting a line to noisy
    # correspondences.
    source = "planes"
    if hit_scale is not None and hit_scale > 0:
        if fresh is None or not 0.7 < hit_scale / fresh.a < 1.4:
            if fresh is not None:
                log.warning(
                    "recon scale disagreement: planes a=%.3f vs hit-test a=%.3f "
                    "(%.2fx) -> using hit-test",
                    fresh.a,
                    hit_scale,
                    hit_scale / fresh.a,
                )
            fresh = ScaleFit(hit_scale, 0.0, 0, 0.0, "hit")
            source = "hit-test"

    # The method may pin the scale source. "fixed"/"none" bypass the tracker:
    # they are exact by construction, so smoothing them would be wrong.
    scale_mode = params["scaleMode"]
    exact = False
    if scale_mode == "hit":
        if hit_scale is None:
            return _skip(rs, entry, f"scaleMode=hit, no hit-test ({why})", hit_scale)
        # scaleB applies here too. It used to be hardcoded to 0, which removed
        # the only lever for correcting a scale that is right about the near
        # surface and wrong further out - the exact thing `hit` gets wrong.
        fresh = ScaleFit(hit_scale, params["scaleB"], 0, 0.0, "hit")
        source = "hit-test"
    elif scale_mode == "fixed":
        fresh = ScaleFit(params["scaleA"], params["scaleB"], 0, 0.0, "fixed")
        source, exact = "fixed", True
    elif scale_mode == "none":
        fresh = ScaleFit(1.0, 0.0, 0, 0.0, "none")
        source, exact = "none", True
    # "auto" and "planes" fall through with whatever the fit cross-check left.

    fit = fresh if exact else entry["tracker"].update(fresh)
    timing["scale"] = round((time.perf_counter() - t) * 1000, 2)
    if fit is None:
        return _skip(rs, entry, why, hit_scale)
    # Fusing on the tracked scale is fine occasionally, but if this happens on
    # every frame the map is being built on a scale nothing is validating.
    scale_note = None if fresh is not None else f"scale held a={fit.a:.2f} - {why}"

    # 3. MS3.2 - unproject and fuse into the world voxel map
    t = time.perf_counter()
    points, ys, xs = unproject_depth(
        rel,
        world_from_cam,
        intr,
        fit,
        stride=params["stride"],
        min_depth=params["minDepth"],
        max_depth=params["maxDepth"],
    )
    if points.shape[0]:
        method.add(
            points,
            img[ys, xs].astype(np.float64),
            world_from_cam[:3, 3].astype(np.float64),
            [],
        )
    timing["fuse"] = round((time.perf_counter() - t) * 1000, 2)
    timing["total"] = round((time.perf_counter() - t0) * 1000, 2)

    entry["fused"] += 1
    log.info(
        "recon %s fused %d pts -> %d voxels | scale %s a=%.3f b=%.3f rms=%.3f "
        "src=%s (%d planes, %d tracked) %.0f ms",
        req.frameId,
        points.shape[0],
        method.size,
        fit.method,
        fit.a,
        fit.b,
        fit.rms,
        source,
        len(req.planes),
        len(entry["tracker"].a),
        timing["total"],
    )

    cloud = None
    if (
        req.forceSnapshot
        or entry["fused"] % config.RECON_CLOUD_EVERY == 0
        or entry["fused"] == 1
    ):
        pos, col, used = method.cloud(config.RECON_CLOUD_MAX)
        p_b64, c_b64 = encode_points(pos, col)
        cloud = ReconCloud(
            count=int(pos.shape[0]),
            positions=p_b64,
            colors=c_b64,
            voxel=used,
            method=entry["method_id"],
            note=method.note(),
            scaleA=fit.a,
            scaleB=fit.b,
            scaleMethod=fit.method,
            scaleRms=fit.rms,
            scaleSamples=fit.samples,
        )

    return ReconFrameResponse(
        keyframes=rs["keyframes"],
        fused=entry["fused"],
        skipped=entry["skipped"],
        voxels=method.size,
        cloud=cloud,
        timing=timing,
        method=entry["method_id"],
        note=method.note(),
        reason=scale_note,
        scaleA=fit.a,
        scaleB=fit.b,
        scaleMethod=fit.method,
        scaleHit=hit_scale,
    )


@app.post("/recon/frame", response_model=ReconFrameResponse)
def recon_frame(req: ReconFrameRequest) -> ReconFrameResponse:
    """Accept a streamed keyframe: depth -> metric scale -> fuse."""
    rs = _recon_state()
    rs["keyframes"] += 1
    sessions: dict[str, int] = rs["sessions"]
    sessions[req.sessionId] = sessions.get(req.sessionId, 0) + 1
    method_id = recon.resolve(req.method)
    entry = _recon_entry(
        rs, req.sessionId, method_id, recon.coerce_params(method_id, req.params)
    )

    if not _recon_lock.acquire(blocking=False):
        return _skip(rs, entry, "busy (previous frame still fusing)")
    try:
        return _fuse_keyframe(req, rs, entry)
    finally:
        _recon_lock.release()


@app.get("/recon/methods")
def recon_methods_list() -> dict[str, Any]:
    """The reconstruction methods and their tunable params.

    The viewer builds its controls straight from this, so it stays in step with
    app/recon_methods.py without any duplicated knowledge.
    """
    return {"default": "voxel", "methods": recon.list_methods()}


@app.get("/recon/status", response_model=ReconStatusResponse)
def recon_status() -> ReconStatusResponse:
    rs = _recon_state()
    last = rs.get("last") or {}
    entry = rs["maps"].get(last.get("sessionId")) if last else None
    tracker: ScaleTracker | None = entry.get("tracker") if entry else None
    fit = tracker.update(None) if tracker else None
    return ReconStatusResponse(
        keyframes=rs["keyframes"],
        fused=sum(e["fused"] for e in rs["maps"].values()),
        skipped=sum(e["skipped"] for e in rs["maps"].values()),
        sessions=rs["sessions"],
        lastFrameId=last.get("frameId"),
        lastJpegBytes=last.get("jpegBytes", 0),
        lastMs=last.get("at"),
        voxels=entry["method"].size if entry else 0,
        scaleA=fit.a if fit else None,
        scaleB=fit.b if fit else None,
        scaleMethod=fit.method if fit else None,
    )


@app.post("/scan", response_model=ScanResponse)
async def scan(req: ScanRequest) -> ScanResponse:
    spec = METHODS.get(req.method)
    if spec is None:
        raise HTTPException(status_code=400, detail=f"unknown method {req.method!r}")

    try:
        raw = base64.b64decode(req.image, validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"bad base64 image: {e}") from e
    if not raw:
        raise HTTPException(status_code=400, detail="empty image")
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"bad image: {e}") from e
    img = np.array(pil)
    height, width = img.shape[:2]
    frame_id = req.frameId or uuid.uuid4().hex

    if spec["kind"] == "depth":
        return _depth_response(frame_id, pil, width, height)
    return _segment_response(req, spec, pil, img, width, height, frame_id)


def _depth_response(frame_id: str, pil: Image.Image, width: int, height: int) -> ScanResponse:
    depth = state.get("depth")
    if depth is None:
        raise HTTPException(status_code=503, detail="depth model not loaded")
    t0 = time.perf_counter()
    depth_map = depth.predict(pil)
    timing = {"depth": round((time.perf_counter() - t0) * 1000, 2)}
    log.info("depth %s in %.0f ms", frame_id, timing["depth"])
    return ScanResponse(
        frameId=frame_id,
        width=width,
        height=height,
        method="depth",
        kind="depth",
        detections=[],
        depth=DepthResult(
            png=_encode_png(_downscale(_colorize_depth(depth_map))),
            width=width,
            height=height,
            min=float(depth_map.min()),
            max=float(depth_map.max()),
        ),
        timing=timing,
    )


def _segment_response(
    req: ScanRequest,
    spec: dict,
    pil: Image.Image,
    img: np.ndarray,
    width: int,
    height: int,
    frame_id: str,
) -> ScanResponse:
    segmenter = get_segmenter(spec["id"])
    if segmenter is None:
        raise HTTPException(status_code=503, detail=f"method {spec['id']!r} unavailable")
    timing: dict[str, float] = {}

    prompt = (req.aimX * width, req.aimY * height) if spec["kind"] == "prompt" else None
    t0 = time.perf_counter()
    items = segmenter.run(img, prompt)
    timing["segment"] = round((time.perf_counter() - t0) * 1000, 2)

    if spec["kind"] == "prompt":
        label = _detector_label(img, req.aimX, req.aimY, width, height)
        for item in items:
            item.label = label

    depth_est: DepthEstimator | None = state.get("depth")
    depth = None
    if depth_est is not None:
        t0 = time.perf_counter()
        # True depth, not the model's raw disparity (see DepthEstimator), and
        # median-blur it to knock down speckle before unprojecting.
        depth = depth_est.predict_depth(pil)
        depth = cv2.medianBlur(depth, 5)
        timing["depth"] = round((time.perf_counter() - t0) * 1000, 2)

    detections = [
        Detection(
            classId=0,
            label=item.label,
            score=item.score,
            box=item.box,
            contour2d=contour2d(item.mask),
        )
        for item in items
    ]

    aimed = _lift_aimed(req, items, depth, width, height, timing)
    log.info(
        "%s %s: %d masks in %s ms",
        spec["id"],
        frame_id,
        len(detections),
        timing.get("segment", 0),
    )
    return ScanResponse(
        frameId=frame_id,
        width=width,
        height=height,
        method=spec["id"],
        kind=spec["kind"],
        detections=detections,
        aimed=aimed,
        timing=timing,
    )


def _lift_aimed(
    req: ScanRequest,
    items: list,
    depth: np.ndarray | None,
    width: int,
    height: int,
    timing: dict[str, float],
) -> Aimed | None:
    if not items or depth is None:
        return None
    aim_x, aim_y = req.aimX * width, req.aimY * height
    idx = next((i for i, it in enumerate(items) if _contains(it.box, aim_x, aim_y)), None)
    if idx is None:
        idx = max(
            range(len(items)),
            key=lambda i: (items[i].box[2] - items[i].box[0])
            * (items[i].box[3] - items[i].box[1]),
        )

    item = items[idx]
    intr = intrinsics_from_fov(width, height, req.fovDeg or config.FOV_DEG)
    points = unproject_mask(item.mask, depth, intr)
    if len(points) == 0:
        return None

    aimed = Aimed(detectionIndex=idx, classId=0, name=item.label)
    obb = make_obb(points)
    if obb is not None:
        center, size, quat = obb
        aimed.obbCam = Obb(
            center=[float(v) for v in center],
            size=[float(v) for v in size],
            quat=[float(v) for v in quat],
        )
    c2d, c3d = contour_from_mask(item.mask, depth, intr)
    aimed.contour2d = c2d
    aimed.contourCam = c3d
    aimed.pointCount = int(len(points))
    aimed.scaleHint = "hit-test"
    aimed.depthAtCentre = float(depth[int(np.clip(aim_y, 0, height - 1)), int(np.clip(aim_x, 0, width - 1))])
    return aimed
