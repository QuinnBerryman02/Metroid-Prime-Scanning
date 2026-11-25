"""
Metroid-scan inference service.

Stage A: `/scan` returns YOLO detections (boxes + labels) for a keyframe.
The response schema already carries the Stage-B/C fields (mask / contourCam /
obbCam) so the phone can lift the aimed object into the world later - see
`docs/LIFTING.md`.

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

from .config import CONF, DEVICE, EXPORT_OPENVINO, IMGSZ, MODEL_PATH
from .detector import Detector
from .schemas import Aimed, Detection, ScanRequest, ScanResponse

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("inference")

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state["detector"] = Detector(MODEL_PATH, DEVICE, IMGSZ, CONF, EXPORT_OPENVINO)
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
        img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"bad image: {e}") from e
    height, width = img.shape[:2]
    frame_id = req.frameId
    aim_x, aim_y = req.aimX, req.aimY

    t0 = time.perf_counter()
    detections = detector.detect(img)
    detect_ms = (time.perf_counter() - t0) * 1000

    aimed = None
    idx = pick_aimed(detections, aim_x, aim_y, width, height)
    if idx is not None:
        d = detections[idx]
        # Stage A: no geometry yet. Stage B will fill contourCam/obbCam here.
        aimed = Aimed(detectionIndex=idx, classId=d.classId, name=d.label)

    log.info(
        "scan %s: %d detections (aimed=%s) in %.0f ms",
        frame_id or "(no id)",
        len(detections),
        aimed.name if aimed else "-",
        detect_ms,
    )
    return ScanResponse(
        frameId=frame_id or uuid.uuid4().hex,
        width=width,
        height=height,
        detections=detections,
        aimed=aimed,
        timing={"detect": round(detect_ms, 2)},
    )
