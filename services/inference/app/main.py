"""
Metroid-scan inference service.

`/scan` takes a keyframe and returns:
  * every YOLO detection (boxes + labels), and
  * Stage-B geometry for the aimed object: its mask, a camera-space contour and
    an oriented bounding box, ready for the phone to anchor - `docs/LIFTING.md`.

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

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image

from .config import (
    CONF,
    DEPTH_DEVICE,
    DEPTH_ENABLED,
    DEPTH_MODEL,
    DEVICE,
    EXPORT_OPENVINO,
    FOV_DEG,
    IMGSZ,
    MODEL_PATH,
    SAM_DEVICE,
    SAM_MODEL,
    SEGMENT_ENABLED,
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
from .schemas import Aimed, Detection, Mask, Obb, ScanRequest, ScanResponse
from .segment import Segmenter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("inference")

state: dict[str, Any] = {}


def _try_load(name: str, factory: Any) -> Any:
    """Load an optional Stage-B model; never let it stop the service."""
    try:
        return factory()
    except Exception as e:  # noqa: BLE001
        log.warning("Stage B '%s' unavailable (%s) - serving boxes only", name, e)
        return None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state["detector"] = Detector(MODEL_PATH, DEVICE, IMGSZ, CONF, EXPORT_OPENVINO)
    # Stage B models are optional: detection still works if they fail to load.
    state["segmenter"] = _try_load(
        "segmenter",
        lambda: Segmenter(SAM_MODEL, SAM_DEVICE) if SEGMENT_ENABLED else None,
    )
    state["depth"] = _try_load(
        "depth",
        lambda: DepthEstimator(DEPTH_MODEL, DEPTH_DEVICE) if DEPTH_ENABLED else None,
    )
    log.info("ready (device=%s)", DEVICE)
    yield
    state.clear()


app = FastAPI(title="metroid-scan inference", version="0.1.0", lifespan=lifespan)


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


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "device": DEVICE,
        "model": MODEL_PATH,
        "loaded": "detector" in state,
        "classes": len(getattr(state.get("detector"), "names", {}) or {}),
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
    frame_id = req.frameId
    aim_x, aim_y = req.aimX, req.aimY

    t0 = time.perf_counter()
    detections = detector.detect(img)
    detect_ms = (time.perf_counter() - t0) * 1000

    aimed = None
    idx = pick_aimed(detections, aim_x, aim_y, width, height)
    timing: dict[str, float] = {"detect": round(detect_ms, 2)}
    if idx is not None:
        d = detections[idx]
        aimed = Aimed(detectionIndex=idx, classId=d.classId, name=d.label)
        _lift_aimed(aimed, d, pil, img, req, (width, height), timing)

    log.info(
        "scan %s: %d detections (aimed=%s) in %.0f ms %s",
        frame_id or "(no id)",
        len(detections),
        aimed.name if aimed else "-",
        detect_ms,
        f"(+{aimed.pointCount} pts)" if aimed and aimed.pointCount else "",
    )
    return ScanResponse(
        frameId=frame_id or uuid.uuid4().hex,
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
    segmenter: Segmenter | None = state.get("segmenter")
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
