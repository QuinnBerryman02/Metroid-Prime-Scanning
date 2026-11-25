"""
Metroid-scan inference service (skeleton).

Endpoints (see docs/MODELS.md):
  GET  /health          - readiness + which OpenVINO device is in use
  POST /scan            - detect + segment a keyframe  -> Detection[]
  POST /depth           - monocular depth for a frame
  POST /classify        - CLIP zero-shot classification of an image crop
  POST /describe        - VLM caption / explanation for an object crop

Everything is OpenVINO-based and device-selectable via OV_DEVICE
(GPU | CPU | AUTO | MULTI:GPU,CPU).
"""
from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

OV_DEVICE = os.environ.get("OV_DEVICE", "AUTO")
MODELS_DIR = os.environ.get("MODELS_DIR", "./models")

app = FastAPI(title="metroid-scan inference", version="0.0.1")


class Health(BaseModel):
    ok: bool
    device: str
    models_loaded: list[str]


@app.get("/health", response_model=Health)
def health() -> Health:
    return Health(ok=True, device=OV_DEVICE, models_loaded=[])


@app.post("/scan")
async def scan(image: UploadFile = File(...)) -> dict[str, Any]:
    """Detect + segment objects in a keyframe.

    TODO: YOLO11n (OpenVINO) for boxes, FastSAM-s / MobileSAM for masks.
    """
    _ = await image.read()
    # TODO: real inference
    return {"detections": []}


@app.post("/depth")
async def depth(image: UploadFile = File(...)) -> dict[str, Any]:
    """Monocular depth map (Depth Anything V2 small -> fp16 OpenVINO).

    TODO: return a normalised float depth map (e.g. base64 npy/png).
    """
    _ = await image.read()
    return {"depth": None}


@app.post("/classify")
async def classify(image: UploadFile = File(...)) -> dict[str, Any]:
    """CLIP zero-shot classification of an object crop."""
    _ = await image.read()
    return {"label": "unknown", "confidence": 0.0}


@app.post("/describe")
async def describe(image: UploadFile = File(...)) -> dict[str, Any]:
    """Florence-2 / BLIP caption for the Metroid-style scan info card."""
    _ = await image.read()
    return {"description": ""}
