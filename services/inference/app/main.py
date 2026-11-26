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
from .schemas import Aimed, DepthResult, Detection, Obb, ScanRequest, ScanResponse

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
