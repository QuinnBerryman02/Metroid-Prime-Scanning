"""
Metroid-scan inference service.

`/scan` takes a keyframe and returns, depending on `mode`:
  * "scan"  - every YOLO detection (boxes + labels) plus Stage-B geometry for
              the aimed object: its mask, a camera-space contour and an oriented
              bounding box, ready for the phone to anchor (`docs/LIFTING.md`).
  * "depth" - a colourised depth map for the frame.

The server stays pose-agnostic: it works in camera space, the phone owns pose
and the AR anchor API.

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import base64
import binascii
import io
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image

from .config import (
    CONF,
    DEPTH_DEVICE,
    DEPTH_ENABLED,
    DEPTH_IR,
    DEPTH_MODEL,
    DEVICE,
    EXPORT_OPENVINO,
    FOV_DEG,
    IMGSZ,
    MODEL_PATH,
    SAM_DEVICE,
    SAM_MODEL,
    SEGMENT_ENABLED,
    SEGMENTER,
    YOLO_SEG_MODEL,
)
from .depth import DepthEstimator
from .detector import Detector
from .lifting import (
    contour_from_mask,
    encode_mask_png,
    intrinsics_from_fov,
    make_obb,
    unproject_mask,
)
from .schemas import Aimed, DepthResult, Detection, Mask, Obb, ScanRequest, ScanResponse
from .segment import Segmenter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("inference")

state: dict[str, Any] = {}


def _try_load(name: str, factory: Any) -> Any:
    """Load an optional model; never let it stop the service."""
    try:
        return factory()
    except Exception as e:  # noqa: BLE001
        log.warning("'%s' unavailable (%s)", name, e)
        return None


def _get_segmenter(backend: str) -> Segmenter | None:
    """Lazily build and cache a segmenter for the chosen backend."""
    if not SEGMENT_ENABLED:
        return None
    cache: dict[str, Segmenter | None] = state.setdefault("segmenters", {})
    if backend in cache:
        return cache[backend]
    device = DEVICE if backend == "yolo11" else SAM_DEVICE
    segmenter = _try_load(
        f"segmenter:{backend}",
        lambda: Segmenter(
            backend=backend,
            sam_model=SAM_MODEL,
            yolo_model=YOLO_SEG_MODEL,
            device=device,
            export=EXPORT_OPENVINO and device != "cpu",
            imgsz=IMGSZ,
            conf=CONF,
        ),
    )
    if segmenter is not None:
        try:
            segmenter.warmup()
        except Exception as e:  # noqa: BLE001
            log.warning("segmenter warmup failed: %s", e)
    cache[backend] = segmenter
    return segmenter


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state["detector"] = Detector(MODEL_PATH, DEVICE, IMGSZ, CONF, EXPORT_OPENVINO)
    state["segmenters"] = {}
    state["depth"] = (
        _try_load(
            "depth",
            lambda: DepthEstimator(DEPTH_MODEL, DEPTH_DEVICE, DEPTH_IR, EXPORT_OPENVINO),
        )
        if DEPTH_ENABLED
        else None
    )
    if SEGMENT_ENABLED:  # preload the default backend so the first scan is warm
        _get_segmenter(SEGMENTER)
    log.info("ready (device=%s, segmenter=%s)", DEVICE, SEGMENTER)
    yield
    state.clear()


app = FastAPI(title="metroid-scan inference", version="0.2.0", lifespan=lifespan)


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


def _colorize_depth(depth: np.ndarray) -> np.ndarray:
    lo, hi = float(depth.min()), float(depth.max())
    norm = (depth - lo) / (hi - lo + 1e-6)
    gray = (norm * 255).astype(np.uint8)
    colour = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
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
    segmenters = state.get("segmenters", {})
    return {
        "ok": True,
        "device": DEVICE,
        "model": MODEL_PATH,
        "loaded": "detector" in state,
        "classes": len(getattr(state.get("detector"), "names", {}) or {}),
        "segmenter": SEGMENTER,
        "loadedSegmenters": [k for k, v in segmenters.items() if v],
        "depth": state.get("depth") is not None,
    }


@app.post("/scan", response_model=ScanResponse)
async def scan(req: ScanRequest) -> ScanResponse:
    detector: Detector | None = state.get("detector")
    if detector is None:
        raise HTTPException(status_code=503, detail="detector not loaded")

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
    timing: dict[str, float] = {}

    if req.mode == "depth":
        depth = state.get("depth")
        if depth is None:
            raise HTTPException(status_code=503, detail="depth model not loaded")
        t0 = time.perf_counter()
        depth_map = depth.predict(pil)
        timing["depth"] = round((time.perf_counter() - t0) * 1000, 2)
        log.info("depth %s: %.0f ms", frame_id, timing["depth"])
        return ScanResponse(
            frameId=frame_id,
            width=width,
            height=height,
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

    t0 = time.perf_counter()
    detections = detector.detect(img)
    timing["detect"] = round((time.perf_counter() - t0) * 1000, 2)

    aimed = None
    idx = pick_aimed(detections, req.aimX, req.aimY, width, height)
    if idx is not None:
        d = detections[idx]
        aimed = Aimed(detectionIndex=idx, classId=d.classId, name=d.label)
        _lift_aimed(aimed, d, pil, img, req, (width, height), timing)

    log.info(
        "scan %s: %d detections (aimed=%s) in %s ms %s",
        frame_id,
        len(detections),
        aimed.name if aimed else "-",
        timing.get("detect", 0),
        f"(+{aimed.pointCount} pts)" if aimed and aimed.pointCount else "",
    )
    return ScanResponse(
        frameId=frame_id,
        width=width,
        height=height,
        detections=detections,
        aimed=aimed,
        timing=timing,
    )


def _lift_aimed(
    aimed: Aimed,
    detection: Detection,
    pil: Image.Image,
    img: np.ndarray,
    req: ScanRequest,
    size: tuple[int, int],
    timing: dict[str, float],
) -> None:
    """Stage B: segment the aimed box, estimate depth, unproject to camera space."""
    segmenter = _get_segmenter(req.segmenter or SEGMENTER)
    depth_est: DepthEstimator | None = state.get("depth")
    if segmenter is None or depth_est is None:
        return
    width, height = size
    try:
        t0 = time.perf_counter()
        mask = segmenter.mask_from_box(img, detection.box)
        timing["segment"] = round((time.perf_counter() - t0) * 1000, 2)
        if mask is None:
            return

        t0 = time.perf_counter()
        depth = depth_est.predict(pil)
        timing["depth"] = round((time.perf_counter() - t0) * 1000, 2)

        intr = intrinsics_from_fov(width, height, req.fovDeg or FOV_DEG)
        points = unproject_mask(mask, depth, intr)
        if len(points) == 0:
            return

        obb = make_obb(points)
        if obb is not None:
            center, box_size, quat = obb
            aimed.obbCam = Obb(
                center=[float(v) for v in center],
                size=[float(v) for v in box_size],
                quat=[float(v) for v in quat],
            )

        contour2d, contour_cam = contour_from_mask(mask, depth, intr)
        aimed.contour2d = contour2d
        aimed.contourCam = contour_cam
        aimed.pointCount = int(len(points))
        aimed.scaleHint = "hit-test"

        ai = int(np.clip(req.aimY * height, 0, height - 1))
        aj = int(np.clip(req.aimX * width, 0, width - 1))
        aimed.depthAtCentre = float(depth[ai, aj])
        detection.mask = Mask(encoding="png", data=encode_mask_png(mask))
    except Exception as e:  # noqa: BLE001
        log.warning("lifting failed: %s", e)
