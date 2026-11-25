"""
Wire schemas for the inference service.

Field names are camelCase to match the TypeScript `@metroid-scan/shared`
protocol, so the gateway can pass them straight through.

Stage A only fills `detections` (boxes + labels). The `Aimed` object is the
placeholder for Stage B (3D contour / OBB in camera space) - see
`docs/LIFTING.md`.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Mask(BaseModel):
    """Segmentation mask in image space (Stage B)."""

    encoding: Literal["rle", "png"] = "rle"
    data: str


class Detection(BaseModel):
    classId: int
    label: str
    score: float
    # [x1, y1, x2, y2] in pixels of the submitted image.
    box: list[float]
    mask: Mask | None = None


class Obb(BaseModel):
    """Oriented bounding box in camera space (Stage B)."""

    center: list[float] = Field(..., min_length=3, max_length=3)
    size: list[float] = Field(..., min_length=3, max_length=3)
    quat: list[float] = Field(..., min_length=4, max_length=4)


class Aimed(BaseModel):
    """
    The object under the reticle - the one the user is scanning.

    Stage A: only `detectionIndex`/`classId`.
    Stage B: adds the camera-space geometry the phone lifts into the world.
    """

    detectionIndex: int
    classId: int
    name: str | None = None
    # All Stage-B fields below are in CAMERA space (camera at origin, +z forward).
    contourCam: list[list[float]] | None = None
    obbCam: Obb | None = None
    depthAtCentre: float | None = None
    scaleHint: str | None = None


class ScanResponse(BaseModel):
    frameId: str
    width: int
    height: int
    detections: list[Detection]
    aimed: Aimed | None = None
    timing: dict[str, float] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    ok: bool
    device: str
    model: str
    loaded: bool


class ScanRequest(BaseModel):
    """A keyframe to scan. `image` is base64 (no data-URI prefix)."""

    image: str
    frameId: str = ""
    aimX: float = 0.5
    aimY: float = 0.5
